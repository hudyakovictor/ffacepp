"""
DataManager - Менеджер данных с парсингом дат и валидацией качества
Версия: 2.0
Дата: 2025-06-15
Исправлены все критические ошибки согласно правкам
"""

import os
import re
import cv2
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import pickle
from collections import defaultdict

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('logs/datamanager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Импорт конфигурации
try:
    from core_config import (
        PUTIN_BIRTH_DATE, START_ANALYSIS_DATE, END_ANALYSIS_DATE,
        DATA_DIR, RESULTS_DIR, CACHE_DIR, ERROR_CODES, CRITICAL_THRESHOLDS
    )
    logger.info("Конфигурация успешно импортирована")
except ImportError as e:
    logger.error(f"Ошибка импорта конфигурации: {e}")
    # Значения по умолчанию
    PUTIN_BIRTH_DATE = date(1952, 10, 7)
    START_ANALYSIS_DATE = datetime(1999, 1, 1)
    END_ANALYSIS_DATE = datetime(2025, 12, 31)
    DATA_DIR = Path("data")
    RESULTS_DIR = Path("results")
    CACHE_DIR = Path("cache")
    ERROR_CODES = {
        "E001": "NO_FACE_DETECTED",
        "E002": "LOW_QUALITY_IMAGE",
        "E006": "INVALID_DATE_FORMAT"
    }

# ==================== КОНСТАНТЫ КАЧЕСТВА ИЗОБРАЖЕНИЙ ====================

# ИСПРАВЛЕНО: Пороги качества изображений согласно правкам
IMAGE_QUALITY_THRESHOLDS = {
    "min_face_size": 100,              # Минимальный размер лица в пикселях
    "blur_detection_threshold": 100,    # Порог детекции размытия (Laplacian variance)
    "max_noise_level": 30,             # Максимальный уровень шума (std)
    "min_resolution_width": 200,       # Минимальная ширина изображения
    "min_resolution_height": 200,      # Минимальная высота изображения
    "hist_extreme_ratio_threshold": 0.1, # Порог для экстремальных значений гистограммы
    "default_quality_threshold": 0.6   # Порог качества по умолчанию
}

# Веса для расчета общего качества
QUALITY_WEIGHTS = {
    "resolution": 0.25,
    "blur": 0.25,
    "noise": 0.25,
    "lighting": 0.25,
    "face_visibility": 0.0  # Пока не реализовано
}

# Исторические события для корреляции
DOCUMENTED_HISTORICAL_EVENTS = {
    "2000-03-26": {"type": "political", "description": "Президентские выборы", "importance": "high"},
    "2004-03-14": {"type": "political", "description": "Президентские выборы", "importance": "high"},
    "2008-05-07": {"type": "political", "description": "Передача полномочий", "importance": "medium"},
    "2012-05-07": {"type": "political", "description": "Президентские выборы", "importance": "high"},
    "2018-05-07": {"type": "political", "description": "Президентские выборы", "importance": "high"},
    "2020-03-01": {"type": "health", "description": "Пандемия COVID-19", "importance": "very_high"},
    "2024-03-17": {"type": "political", "description": "Президентские выборы", "importance": "high"}
}

# ==================== ОСНОВНОЙ КЛАСС ====================

class DataManager:
    """
    Менеджер данных с полной функциональностью
    ИСПРАВЛЕНО: Все критические ошибки согласно правкам
    """
    
    def __init__(self):
        """Инициализация менеджера данных"""
        logger.info("Инициализация DataManager")
        
        # Хронологический индекс
        self.chronological_index = {}
        
        # Кэш качества изображений
        self.image_quality_cache = {}
        
        # Исторические события
        self.historical_events = self._load_historical_events()
        
        # Кэш результатов
        self.results_cache = {}
        
        # Статистика обработки
        self.processing_stats = {
            "total_processed": 0,
            "successful_parses": 0,
            "failed_parses": 0,
            "quality_passed": 0,
            "quality_failed": 0
        }
        
        logger.info("DataManager инициализирован")

    def _load_historical_events(self) -> Dict[datetime, Dict[str, str]]:
        """Загрузка исторических событий"""
        try:
            events = {}
            for date_str, event_data in DOCUMENTED_HISTORICAL_EVENTS.items():
                try:
                    event_date = datetime.strptime(date_str, "%Y-%m-%d")
                    events[event_date] = event_data
                except ValueError as e:
                    logger.warning(f"Неверный формат даты события: {date_str}, ошибка: {e}")
            
            logger.info(f"Загружено {len(events)} исторических событий")
            return events
            
        except Exception as e:
            logger.error(f"Ошибка загрузки исторических событий: {e}")
            return {}

    def parse_date_from_filename(self, filename: str) -> Tuple[Optional[datetime], int]:
        """
        ИСПРАВЛЕНО: Парсинг даты из имени файла с обработкой ошибок
        Согласно правкам: обработка INVALID_DATE_FORMAT и различных форматов
        """
        if not filename:
            logger.warning("Пустое имя файла")
            return None, 0
        
        try:
            logger.debug(f"Парсинг даты из файла: {filename}")
            
            # ИСПРАВЛЕНО: Поддержка различных форматов дат
            date_patterns = [
                # Формат с полным годом: ddmmyyyy.jpg (проверяем первым)
                (r'(\d{2})(\d{2})(\d{4})(?:-(\d+))?\.jpg$', "%d%m%Y"),
                # Основной формат: ddmmyy.jpg или ddmmyy-N.jpg
                (r'(\d{2})(\d{2})(\d{2})(?:-(\d+))?\.jpg$', "%d%m%y"),
                # Альтернативный формат: dd-mm-yy.jpg
                (r'(\d{2})-(\d{2})-(\d{2})(?:-(\d+))?\.jpg$', "%d-%m-%y"),
                # ISO формат: yyyy-mm-dd.jpg
                (r'(\d{4})-(\d{2})-(\d{2})(?:-(\d+))?\.jpg$', "%Y-%m-%d")
            ]
            
            for pattern, date_format in date_patterns:
                match = re.search(pattern, filename, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    sequence = int(groups[-1]) if groups[-1] else 1
                    
                    # Обработка разных форматов
                    if date_format == "%d%m%y":
                        day, month, year = groups[:3]
                        year = int(year)
                        # ИСПРАВЛЕНО: Правильная обработка 2-значного года
                        if year <= 25:  # 00-25 -> 2000-2025
                            year += 2000
                        else:  # 26-99 -> 1926-1999
                            year += 1900
                        
                        parsed_date = datetime(year, int(month), int(day))
                        
                    elif date_format == "%d-%m-%y":
                        day, month, year = groups[:3]
                        year = int(year)
                        if year <= 25:
                            year += 2000
                        else:
                            year += 1900
                        
                        parsed_date = datetime(year, int(month), int(day))
                        
                    elif date_format == "%d%m%Y":
                        # Для формата ddmmyyyy.jpg, группы: (день, месяц, год)
                        day, month, year = groups[0], groups[1], groups[2]
                        parsed_date = datetime(int(year), int(month), int(day))
                        
                    elif date_format == "%Y-%m-%d":
                        year, month, day = groups[:3]
                        parsed_date = datetime(int(year), int(month), int(day))
                    
                    # Валидация даты
                    if START_ANALYSIS_DATE <= parsed_date <= END_ANALYSIS_DATE:
                        logger.debug(f"Успешно распарсена дата: {parsed_date}, последовательность: {sequence}")
                        self.processing_stats["successful_parses"] += 1
                        return parsed_date, sequence
                    else:
                        logger.warning(f"Дата {parsed_date} вне диапазона анализа")
                        return None, 0
            
            # Если ни один паттерн не подошел
            logger.warning(f"Не удалось распарсить дату из файла: {filename}")
            self.processing_stats["failed_parses"] += 1
            return None, 0
            
        except ValueError as e:
            logger.error(f"Ошибка парсинга даты из {filename}. Raw groups: {match.groups() if match else 'No match'}. Error: {e}")
            self.processing_stats["failed_parses"] += 1
            return None, 0
        except Exception as e:
            logger.error(f"Неожиданная ошибка парсинга даты из {filename}: {e}")
            self.processing_stats["failed_parses"] += 1
            return None, 0

    def calculate_putin_age_on_date(self, photo_date: datetime) -> float:
        """Расчет возраста Путина на заданную дату"""
        try:
            if not isinstance(photo_date, datetime):
                logger.error(f"Неверный тип даты: {type(photo_date)}")
                return 0.0
            
            birth_datetime = datetime.combine(PUTIN_BIRTH_DATE, datetime.min.time())
            age_delta = photo_date - birth_datetime
            age_years = age_delta.days / 365.25
            
            logger.debug(f"Возраст на {photo_date.strftime('%Y-%m-%d')}: {age_years:.2f} лет")
            return age_years
            
        except Exception as e:
            logger.error(f"Ошибка расчета возраста: {e}")
            return 0.0

    def create_master_chronological_index(self, image_paths: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        ИСПРАВЛЕНО: Создание мастер-индекса с сортировкой по датам
        Согласно правкам: правильная структура индекса
        """
        if not image_paths:
            logger.warning("Пустой список путей к изображениям")
            return {}
        
        try:
            logger.info(f"Создание хронологического индекса для {len(image_paths)} изображений")
            
            chronological_data = {}
            processed_count = 0
            
            for image_path in image_paths:
                try:
                    filename = os.path.basename(image_path)
                    date, sequence = self.parse_date_from_filename(filename)
                    
                    if date:
                        age_on_date = self.calculate_putin_age_on_date(date)
                        date_key = date.strftime("%Y-%m-%d")
                        
                        # ИСПРАВЛЕНО: Структура данных согласно правкам
                        if date_key not in chronological_data:
                            chronological_data[date_key] = {
                                "date": date,
                                "age_on_date": age_on_date,
                                "image_paths": []
                            }
                        
                        # Добавление информации об изображении
                        image_info = {
                            "path": image_path,
                            "sequence": sequence,
                            "filename": filename
                        }
                        
                        chronological_data[date_key]["image_paths"].append(image_info)
                        processed_count += 1
                        
                except Exception as e:
                    logger.warning(f"Ошибка обработки файла {image_path}: {e}")
                    continue
            
            # ИСПРАВЛЕНО: Сортировка по датам
            sorted_dates = sorted(chronological_data.keys())
            sorted_data = {date_key: chronological_data[date_key] for date_key in sorted_dates}
            
            # Обновление внутреннего индекса
            self.chronological_index = sorted_data
            
            logger.info(f"Хронологический индекс создан: {len(sorted_data)} дат, {processed_count} изображений")
            return sorted_data
            
        except Exception as e:
            logger.error(f"Ошибка создания хронологического индекса: {e}")
            return {}

    def validate_image_quality_for_analysis(self, image_path: str) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Валидация качества изображения для анализа
        Согласно правкам: blur_score, noise_level, min_face_size, E002
        """
        if not image_path or not os.path.exists(image_path):
            logger.error(f"Файл не существует: {image_path}")
            return {
                "quality_score": 0.0,
                "issues": ["File not found"],
                "error_code": ERROR_CODES["E002"]
            }
        
        # Проверка кэша
        if image_path in self.image_quality_cache:
            logger.debug(f"Качество из кэша для {image_path}")
            return self.image_quality_cache[image_path]
        
        try:
            logger.debug(f"Валидация качества изображения: {image_path}")
            
            # Загрузка изображения
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Не удалось загрузить изображение: {image_path}")
                quality_result = {
                    "quality_score": 0.0,
                    "issues": ["File not readable"],
                    "error_code": ERROR_CODES["E002"]
                }
                self.image_quality_cache[image_path] = quality_result
                return quality_result
            
            # Преобразование в grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            h, w = gray.shape
            
            issues = []
            quality_scores = {}
            
            # ИСПРАВЛЕНО: 1. Проверка разрешения
            min_width = IMAGE_QUALITY_THRESHOLDS["min_resolution_width"]
            min_height = IMAGE_QUALITY_THRESHOLDS["min_resolution_height"]
            
            if w >= min_width and h >= min_height:
                quality_scores["resolution"] = 1.0
            else:
                issues.append("Low resolution")
                quality_scores["resolution"] = 0.5
                logger.warning(f"Низкое разрешение: {w}x{h}")
            
            # ИСПРАВЛЕНО: 2. Детекция размытия (blur_score)
            blur_threshold = IMAGE_QUALITY_THRESHOLDS["blur_detection_threshold"]
            blur_score_val = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if blur_score_val >= blur_threshold:
                quality_scores["blur"] = 1.0
            else:
                issues.append("Image too blurry")
                quality_scores["blur"] = max(0.0, blur_score_val / 150.0)  # Нормализация
                logger.warning(f"Размытое изображение: blur_score={blur_score_val}")
            
            # ИСПРАВЛЕНО: 3. Детекция шума (noise_level)
            max_noise = IMAGE_QUALITY_THRESHOLDS["max_noise_level"]
            noise_level = np.std(gray)
            
            if noise_level <= max_noise:
                quality_scores["noise"] = 1.0
            else:
                issues.append("High noise level")
                quality_scores["noise"] = max(0.0, 1.0 - (noise_level - 10) / 40.0)
                logger.warning(f"Высокий уровень шума: {noise_level}")
            
            # ИСПРАВЛЕНО: 4. Анализ освещения
            hist_threshold = IMAGE_QUALITY_THRESHOLDS["hist_extreme_ratio_threshold"]
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            total_pixels = np.sum(hist)
            
            if total_pixels > 0:
                dark_ratio = hist[0] / total_pixels
                bright_ratio = hist[255] / total_pixels
                
                if dark_ratio <= hist_threshold and bright_ratio <= hist_threshold:
                    quality_scores["lighting"] = 1.0
                else:
                    issues.append("Poor lighting (too dark or too bright)")
                    quality_scores["lighting"] = 0.5
                    logger.warning(f"Плохое освещение: dark_ratio={dark_ratio:.3f}, bright_ratio={bright_ratio:.3f}")
            else:
                quality_scores["lighting"] = 0.0
                issues.append("Invalid histogram")
            
            # ИСПРАВЛЕНО: 5. Видимость лица (пока заглушка)
            quality_scores["face_visibility"] = 1.0  # Будет реализовано с Face3DAnalyzer
            
            # ИСПРАВЛЕНО: Расчет общего качества с весами
            total_weight = sum(QUALITY_WEIGHTS.values())
            if total_weight == 0:
                total_weight = 1.0
            
            combined_quality_score = sum(
                quality_scores.get(component, 0.0) * weight
                for component, weight in QUALITY_WEIGHTS.items()
            ) / total_weight
            
            # Результат валидации
            quality_result = {
                "quality_score": float(np.clip(combined_quality_score, 0.0, 1.0)),
                "issues": issues,
                "resolution": (w, h),
                "blur_score": float(blur_score_val),
                "noise_level": float(noise_level),
                "lighting_hist_extreme_ratio": (float(hist[0] / total_pixels), float(hist[255] / total_pixels)) if total_pixels > 0 else (0.0, 0.0),
                "component_scores": quality_scores
            }
            
            # Добавление кода ошибки если качество низкое
            if quality_result["quality_score"] < CRITICAL_THRESHOLDS.get("min_quality_score", 0.6):
                quality_result["error_code"] = ERROR_CODES["E002"]
                self.processing_stats["quality_failed"] += 1
            else:
                self.processing_stats["quality_passed"] += 1
            
            # Кэширование результата
            self.image_quality_cache[image_path] = quality_result
            
            logger.debug(f"Качество изображения: {quality_result['quality_score']:.3f}")
            return quality_result
            
        except Exception as e:
            logger.error(f"Ошибка валидации качества {image_path}: {e}")
            quality_result = {
                "quality_score": 0.0,
                "issues": [f"Processing error: {str(e)}"],
                "error_code": ERROR_CODES["E002"]
            }
            self.image_quality_cache[image_path] = quality_result
            return quality_result

    def correlate_with_historical_events(self, dates_list: List[datetime], 
                                       correlation_window_days: int = 30) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Корреляция дат с историческими событиями
        Согласно правкам: временные окна и медицинские события
        """
        if not dates_list:
            logger.warning("Пустой список дат для корреляции")
            return {}
        
        try:
            logger.info(f"Корреляция {len(dates_list)} дат с историческими событиями")
            
            correlations = {}
            total_correlations = 0
            
            for date in dates_list:
                date_str = date.strftime("%Y-%m-%d")
                nearby_events = []
                
                # Поиск событий в окне корреляции
                for event_date, event_info in self.historical_events.items():
                    days_diff = abs((date - event_date).days)
                    
                    if days_diff <= correlation_window_days:
                        # Определение типа события
                        is_medical_event = False
                        event_description = event_info.get("description", "").lower()
                        medical_keywords = ["здоровье", "болезнь", "пандемия", "covid", "медицинский"]
                        
                        if any(keyword in event_description for keyword in medical_keywords):
                            is_medical_event = True
                        
                        nearby_events.append({
                            "event_date": event_date,
                            "days_diff": (date - event_date).days,  # Положительное = после события
                            "absolute_diff": days_diff,
                            "event_info": event_info,
                            "is_medical_event": is_medical_event,
                            "correlation_strength": self._calculate_correlation_strength(days_diff, correlation_window_days)
                        })
                
                if nearby_events:
                    # Сортировка по близости
                    nearby_events.sort(key=lambda x: x["absolute_diff"])
                    correlations[date_str] = nearby_events
                    total_correlations += len(nearby_events)
            
            correlation_summary = {
                "correlations": correlations,
                "total_correlations": total_correlations,
                "correlation_rate": total_correlations / len(dates_list) if dates_list else 0.0,
                "correlation_window_days": correlation_window_days
            }
            
            logger.info(f"Найдено {total_correlations} корреляций для {len(correlations)} дат")
            return correlation_summary
            
        except Exception as e:
            logger.error(f"Ошибка корреляции с историческими событиями: {e}")
            return {}

    def _calculate_correlation_strength(self, days_difference: int, max_window: int) -> str:
        """Расчет силы корреляции на основе временной близости"""
        try:
            ratio = days_difference / max_window
            
            if ratio <= 0.1:
                return "very_strong"
            elif ratio <= 0.3:
                return "strong"
            elif ratio <= 0.6:
                return "moderate"
            else:
                return "weak"
                
        except Exception as e:
            logger.error(f"Ошибка расчета силы корреляции: {e}")
            return "unknown"

    def detect_temporal_gaps_and_patterns(self, min_gap_days: int = 30) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Обнаружение временных пропусков и паттернов
        Согласно правкам: анализ gaps и patterns
        """
        if not self.chronological_index:
            logger.warning("Хронологический индекс пуст")
            return {}
        
        try:
            logger.info("Обнаружение временных пропусков и паттернов")
            
            # Получение отсортированных дат
            dates = [datetime.strptime(date_str, "%Y-%m-%d") for date_str in self.chronological_index.keys()]
            dates.sort()
            
            if len(dates) < 2:
                logger.warning("Недостаточно дат для анализа пропусков")
                return {"gaps": [], "patterns": {}}
            
            gaps = []
            intervals = []
            
            # Анализ пропусков
            for i in range(1, len(dates)):
                gap_days = (dates[i] - dates[i-1]).days
                intervals.append(gap_days)
                
                if gap_days >= min_gap_days:
                    gaps.append({
                        "start_date": dates[i-1],
                        "end_date": dates[i],
                        "gap_days": gap_days,
                        "severity": self._classify_gap_severity(gap_days)
                    })
            
            # Анализ паттернов
            patterns = self._analyze_temporal_patterns(intervals)
            
            temporal_analysis = {
                "gaps": gaps,
                "patterns": patterns,
                "total_gaps": len(gaps),
                "max_gap_days": max(intervals) if intervals else 0,
                "avg_interval_days": np.mean(intervals) if intervals else 0,
                "total_date_range_days": (dates[-1] - dates[0]).days if len(dates) > 1 else 0
            }
            
            logger.info(f"Найдено {len(gaps)} пропусков, проанализировано {len(intervals)} интервалов")
            return temporal_analysis
            
        except Exception as e:
            logger.error(f"Ошибка анализа временных пропусков: {e}")
            return {}

    def _classify_gap_severity(self, gap_days: int) -> str:
        """Классификация серьезности пропуска"""
        if gap_days >= 365:
            return "critical"
        elif gap_days >= 180:
            return "high"
        elif gap_days >= 90:
            return "medium"
        else:
            return "low"

    def _analyze_temporal_patterns(self, intervals: List[int]) -> Dict[str, Any]:
        """Анализ временных паттернов"""
        try:
            if not intervals:
                return {}
            
            patterns = {}
            
            # Статистический анализ интервалов
            intervals_array = np.array(intervals)
            patterns["statistics"] = {
                "mean": float(np.mean(intervals_array)),
                "std": float(np.std(intervals_array)),
                "median": float(np.median(intervals_array)),
                "min": int(np.min(intervals_array)),
                "max": int(np.max(intervals_array))
            }
            
            # Поиск периодических паттернов
            common_periods = [7, 14, 30, 60, 90, 180, 365]  # дни, 2 недели, месяц, и т.д.
            periodic_patterns = []
            
            for period in common_periods:
                matches = sum(1 for interval in intervals if abs(interval % period) < 3 or abs(interval % period) > period - 3)
                if matches >= len(intervals) * 0.3:  # 30% совпадений
                    periodic_patterns.append({
                        "period_days": period,
                        "matches": matches,
                        "match_rate": matches / len(intervals)
                    })
            
            patterns["periodic_patterns"] = periodic_patterns
            
            # Анализ трендов
            if len(intervals) >= 3:
                from scipy import stats
                x = np.arange(len(intervals))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, intervals)
                
                patterns["trend_analysis"] = {
                    "slope": float(slope),
                    "r_squared": float(r_value ** 2),
                    "p_value": float(p_value),
                    "trend_direction": "increasing" if slope > 0.5 else "decreasing" if slope < -0.5 else "stable"
                }
            
            return patterns
            
        except Exception as e:
            logger.error(f"Ошибка анализа временных паттернов: {e}")
            return {}

    def auto_parse_historical_events(self, date_range: Tuple[datetime, datetime]) -> Dict[str, List[Dict[str, Any]]]:
        """
        ИСПРАВЛЕНО: Автоматический парсинг исторических событий
        Согласно правкам: расширенный анализ событий
        """
        try:
            logger.info(f"Автоматический парсинг событий в диапазоне: {date_range}")
            
            # В реальной реализации здесь был бы API к внешним источникам
            # Пока используем расширенные mock данные
            mock_events = [
                {
                    "date": datetime(2000, 3, 26),
                    "type": "political",
                    "description": "Президентские выборы",
                    "importance": "high",
                    "stress_level": "high"
                },
                {
                    "date": datetime(2004, 3, 14),
                    "type": "political", 
                    "description": "Президентские выборы",
                    "importance": "high",
                    "stress_level": "high"
                },
                {
                    "date": datetime(2008, 5, 7),
                    "type": "political",
                    "description": "Передача полномочий",
                    "importance": "medium",
                    "stress_level": "medium"
                },
                {
                    "date": datetime(2012, 5, 7),
                    "type": "political",
                    "description": "Президентские выборы",
                    "importance": "high",
                    "stress_level": "high"
                },
                {
                    "date": datetime(2018, 5, 7),
                    "type": "political",
                    "description": "Президентские выборы",
                    "importance": "high",
                    "stress_level": "high"
                },
                {
                    "date": datetime(2020, 3, 1),
                    "type": "health",
                    "description": "Пандемия COVID-19",
                    "importance": "very_high",
                    "stress_level": "very_high"
                },
                {
                    "date": datetime(2024, 3, 17),
                    "type": "political",
                    "description": "Президентские выборы",
                    "importance": "high",
                    "stress_level": "high"
                }
            ]
            
            # Фильтрация по диапазону дат
            filtered_events = [
                event for event in mock_events
                if date_range[0] <= event["date"] <= date_range[1]
            ]
            
            # Группировка по датам
            events_by_date = defaultdict(list)
            for event in filtered_events:
                date_str = event["date"].strftime("%Y-%m-%d")
                events_by_date[date_str].append(event)
            
            logger.info(f"Найдено {len(filtered_events)} событий в заданном диапазоне")
            return dict(events_by_date)
            
        except Exception as e:
            logger.error(f"Ошибка автоматического парсинга событий: {e}")
            return {}

    def create_data_quality_report(self, processed_images: List[str]) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Создание отчета о качестве данных
        Согласно правкам: детальная статистика качества
        """
        if not processed_images:
            logger.warning("Нет обработанных изображений для отчета о качестве")
            return {}
        
        try:
            logger.info(f"Создание отчета о качестве для {len(processed_images)} изображений")
            
            quality_stats = {
                "total_images": len(processed_images),
                "processed_images": 0,
                "high_quality": 0,
                "medium_quality": 0,
                "low_quality": 0,
                "failed_processing": 0,
                "quality_distribution": {},
                "common_issues": defaultdict(int),
                "average_scores": {},
                "date_range_coverage": {},
                "processing_errors": []
            }
            
            quality_scores = []
            component_scores = defaultdict(list)
            
            for image_path in processed_images:
                try:
                    quality_result = self.validate_image_quality_for_analysis(image_path)
                    quality_score = quality_result.get("quality_score", 0.0)
                    
                    quality_stats["processed_images"] += 1
                    quality_scores.append(quality_score)
                    
                    # Классификация качества
                    if quality_score >= 0.8:
                        quality_stats["high_quality"] += 1
                    elif quality_score >= 0.6:
                        quality_stats["medium_quality"] += 1
                    else:
                        quality_stats["low_quality"] += 1
                    
                    # Сбор статистики по компонентам
                    for component, score in quality_result.get("component_scores", {}).items():
                        component_scores[component].append(score)
                    
                    # Подсчет проблем
                    for issue in quality_result.get("issues", []):
                        quality_stats["common_issues"][issue] += 1
                        
                except Exception as e:
                    quality_stats["failed_processing"] += 1
                    quality_stats["processing_errors"].append({
                        "image_path": image_path,
                        "error": str(e)
                    })
            
            # Расчет средних оценок
            if quality_scores:
                quality_stats["average_scores"]["overall"] = float(np.mean(quality_scores))
                quality_stats["average_scores"]["std"] = float(np.std(quality_scores))
                
                for component, scores in component_scores.items():
                    quality_stats["average_scores"][component] = float(np.mean(scores))
            
            # Распределение качества
            if quality_scores:
                hist, bins = np.histogram(quality_scores, bins=10, range=(0, 1))
                quality_stats["quality_distribution"] = {
                    f"{bins[i]:.1f}-{bins[i+1]:.1f}": int(hist[i])
                    for i in range(len(hist))
                }
            
            # Покрытие по датам
            date_coverage = {}
            for image_path in processed_images:
                filename = os.path.basename(image_path)
                date, _ = self.parse_date_from_filename(filename)
                if date:
                    year = date.year
                    if year not in date_coverage:
                        date_coverage[year] = 0
                    date_coverage[year] += 1
            
            quality_stats["date_range_coverage"] = date_coverage
            
            # Общая статистика обработки
            quality_stats["processing_statistics"] = self.processing_stats.copy()
            
            logger.info(f"Отчет о качестве создан: {quality_stats['high_quality']} высокого качества, "
                       f"{quality_stats['medium_quality']} среднего, {quality_stats['low_quality']} низкого")
            
            return quality_stats
            
        except Exception as e:
            logger.error(f"Ошибка создания отчета о качестве: {e}")
            return {}

    def save_cache(self, cache_file: str = "datamanager_cache.pkl") -> None:
        """Сохранение кэша"""
        try:
            cache_path = CACHE_DIR / cache_file
            CACHE_DIR.mkdir(exist_ok=True)
            
            cache_data = {
                "chronological_index": self.chronological_index,
                "image_quality_cache": self.image_quality_cache,
                "results_cache": self.results_cache,
                "processing_stats": self.processing_stats,
                "historical_events": self.historical_events
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"Кэш сохранен: {cache_path}")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения кэша: {e}")

    def load_cache(self, cache_file: str = "datamanager_cache.pkl") -> None:
        """Загрузка кэша"""
        try:
            cache_path = CACHE_DIR / cache_file
            
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                
                self.chronological_index = cache_data.get("chronological_index", {})
                self.image_quality_cache = cache_data.get("image_quality_cache", {})
                self.results_cache = cache_data.get("results_cache", {})
                self.processing_stats = cache_data.get("processing_stats", {
                    "total_processed": 0, "successful_parses": 0, "failed_parses": 0,
                    "quality_passed": 0, "quality_failed": 0
                })
                self.historical_events = cache_data.get("historical_events", {})
                
                logger.info(f"Кэш загружен: {cache_path}")
            else:
                logger.info("Файл кэша не найден, используется пустой кэш")
                
        except Exception as e:
            logger.error(f"Ошибка загрузки кэша: {e}")

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Получение статистики обработки"""
        return {
            "processing_stats": self.processing_stats.copy(),
            "chronological_index_size": len(self.chronological_index),
            "quality_cache_size": len(self.image_quality_cache),
            "results_cache_size": len(self.results_cache),
            "historical_events_count": len(self.historical_events)
        }

    def self_test(self) -> None:
        """Самотестирование модуля"""
        logger.info("=== Самотестирование DataManager ===")
        
        # Информация о конфигурации
        logger.info(f"Дата рождения: {PUTIN_BIRTH_DATE}")
        logger.info(f"Диапазон анализа: {START_ANALYSIS_DATE} - {END_ANALYSIS_DATE}")
        logger.info(f"Исторических событий: {len(self.historical_events)}")
        
        # Тестовые данные
        test_filenames = [
            "26101999.jpg",
            "01012000-1.jpg", 
            "15-06-01.jpg",
            "invalid_format.jpg",
            "31122024.jpg"
        ]
        
        try:
            # Тест парсинга дат
            logger.info("Тест парсинга дат:")
            for filename in test_filenames:
                date, sequence = self.parse_date_from_filename(filename)
                logger.info(f"  {filename} -> {date}, seq={sequence}")
            
            # Тест расчета возраста
            test_date = datetime(2020, 1, 1)
            age = self.calculate_putin_age_on_date(test_date)
            logger.info(f"Возраст на {test_date}: {age:.2f} лет")
            
            # Тест создания тестового изображения для валидации качества
            test_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
            test_image_path = "test_image.jpg"
            cv2.imwrite(test_image_path, test_image)
            
            # Тест валидации качества
            quality_result = self.validate_image_quality_for_analysis(test_image_path)
            logger.info(f"Тест качества: score={quality_result['quality_score']:.3f}")
            
            # Очистка тестового файла
            if os.path.exists(test_image_path):
                os.remove(test_image_path)
            
            # Тест корреляции с событиями
            test_dates = [datetime(2020, 3, 15), datetime(2024, 3, 20)]
            correlations = self.correlate_with_historical_events(test_dates)
            logger.info(f"Тест корреляций: {correlations.get('total_correlations', 0)} найдено")
            
            # Статистика
            stats = self.get_processing_statistics()
            logger.info(f"Статистика: {stats}")
            
        except Exception as e:
            logger.error(f"Ошибка самотестирования: {e}")
        
        logger.info("=== Самотестирование завершено ===")

# ==================== ТОЧКА ВХОДА ====================

if __name__ == "__main__":
    manager = DataManager()
    manager.self_test()