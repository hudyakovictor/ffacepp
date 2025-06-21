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
import datetime as dt
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

# 01_01_10.jpg  01_01_10-2.jpeg
PATTERN = re.compile(
    r"^(?P<d>\d{2})_(?P<m>\d{2})_(?P<y>\d{2})(?:-(?P<idx>\d{1,2}))?\.jpe?g$", re.I
)

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

    def _safe_load_image(self, img_input: Union[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        Безопасно загружает изображение, если передан путь, или возвращает его, если это уже массив.
        Возвращает None, если загрузка не удалась.
        """
        if isinstance(img_input, str):
            try:
                img = cv2.imread(img_input)
                if img is None:
                    logger.error(f"Не удалось загрузить изображение по пути: {img_input}. Проверьте путь и формат файла.")
                return img
            except Exception as e:
                logger.error(f"Ошибка при чтении изображения {img_input}: {e}")
                return None
        elif isinstance(img_input, np.ndarray):
            return img_input
        else:
            logger.error(f"Неподдерживаемый тип ввода изображения: {type(img_input)}")
            return None

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

    def parse_date_from_filename(self, fname: str) -> Tuple[dt.date, int]:
        """
        Возвращает дату и порядковый индекс снимка,
        либо бросает ValueError, если имя не соответствует формату.
        """
        m = PATTERN.match(fname)
        if not m:
            raise ValueError(f"Bad filename format: {fname!r}")
        d, mth, y = int(m['d']), int(m['m']), int(m['y'])
        # годы 00-25 → 2000-2025, 26-99 → 1926-1999
        year = 2000 + y if y <= 25 else 1900 + y
        seq  = int(m['idx'] or 0)
        return dt.date(year, mth, d), seq

    def calculate_putin_age_on_date(self, photo_date: dt.date) -> float:
        """Расчет возраста Путина на заданную дату"""
        try:
            if not isinstance(photo_date, dt.date):
                logger.error(f"Неверный тип даты: {type(photo_date)}")
                return 0.0
            
            age_delta = photo_date - PUTIN_BIRTH_DATE
            age_years = age_delta.days / 365.25
            
            logger.debug(f"Возраст на {photo_date.strftime('%Y-%m-%d')}: {age_years:.2f} лет")
            return age_years
            
        except Exception as e:
            logger.error(f"Ошибка расчета возраста: {e}")
            return 0.0

    def create_master_chronological_index(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Создание мастер-индекса с сортировкой по датам и расчетом возраста.
        Согласно правкам: включение возраста, использование 365.25 для високосных лет.
        """
        if not image_paths:
            logger.warning("Пустой список путей к изображениям")
            return []
        
        try:
            logger.info(f"Создание хронологического индекса для {len(image_paths)} изображений")
            
            by_date = defaultdict(list)
            for p in image_paths:
                try:
                    file_name = os.path.basename(p)
                    photo_date, order = self.parse_date_from_filename(file_name)
                    if photo_date:
                        by_date[(photo_date, order)].append(p)
                except ValueError as e:
                    logger.warning(f"Пропуск файла {p} из-за ошибки парсинга: {e}")
                    self.processing_stats["failed_parses"] += 1

            sorted_items = sorted(by_date.items())
            chronological_index_list = []
            
            for (d, _), imgs in sorted_items:
                age = self.calculate_putin_age_on_date(d)
                chronological_index_list.append(dict(
                    date=d,
                    images=imgs,
                    age_on_date=age
                ))
            
            self.chronological_index = {item["date"].isoformat(): item for item in chronological_index_list}
            logger.info(f"Хронологический индекс создан с {len(chronological_index_list)} записями")
            self.processing_stats["total_processed"] += len(image_paths)
            return chronological_index_list
            
        except Exception as e:
            logger.error(f"Ошибка создания хронологического индекса: {e}")
            return []

    def validate_image_quality_for_analysis(self, img_input: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        Валидация качества изображения на основе пороговых правил.
        """
        try:
            img = self._safe_load_image(img_input)
            if img is None or not hasattr(img, 'size') or img.size == 0:
                logger.warning(f"Изображение для валидации качества пусто или None после безопасной загрузки.")
                self.processing_stats["quality_failed"] += 1
                return {"quality_score": 0.0, "issues": ["empty_or_failed_load"], "error_code": ERROR_CODES["E002"]}

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            
            blur = cv2.Laplacian(gray, cv2.CV_64F).var()
            noise = np.std(gray)
            hist = cv2.calcHist([gray], [0], None, [256], [0,256])
            
            total_pixels = hist.sum()
            dark_frac = hist[:13].sum() / total_pixels if total_pixels > 0 else 0.0
            bright_frac = hist[-13:].sum() / total_pixels if total_pixels > 0 else 0.0

            issues = []
            if blur < 100: issues.append("blur"); blur_flag = True
            else: blur_flag = False

            if noise > 30: issues.append("noise"); noise_flag = True
            else: noise_flag = False

            if dark_frac > 0.1 or bright_frac > 0.1: issues.append("lighting"); lighting_flag = True
            else: lighting_flag = False

            quality = 1 - 0.25*bool(blur_flag) - 0.25*bool(noise_flag) - 0.25*bool(lighting_flag)
            
            quality_result = {
                "quality_score": float(np.clip(quality, 0.0, 1.0)),
                "issues": issues,
                "blur_detection_result": float(blur),
                "noise_level": float(noise),
                "lighting_quality": 0.0 if lighting_flag else 1.0 # 0 if issue, 1 if no issue
            }
            
            if quality_result["quality_score"] < CRITICAL_THRESHOLDS.get("min_quality_score", 0.6):
                quality_result["error_code"] = ERROR_CODES["E002"]
                self.processing_stats["quality_failed"] += 1
            else:
                self.processing_stats["quality_passed"] += 1
            
            logger.debug(f"Качество изображения: {quality_result['quality_score']:.3f}, Issues: {issues}")
            return quality_result
            
        except Exception as e:
            logger.error(f"Ошибка валидации качества изображения: {e}")
            quality_result = {
                "quality_score": 0.0,
                "issues": [f"Processing error: {str(e)}"],
                "blur_detection_result": 0.0,
                "noise_level": 0.0,
                "lighting_quality": 0.0,
                "error_code": ERROR_CODES["E002"]
            }
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
            "01_01_10.jpg",
            "15_06_01-2.jpeg",
            "31_12_99.jpg",
        ]
        
        try:
            # Тест парсинга дат
            logger.info("Тест парсинга дат:")
            for filename in test_filenames:
                date, sequence = self.parse_date_from_filename(filename)
                logger.info(f"  {filename} -> {date}, seq={sequence}")
            
            # Тест расчета возраста
            test_date = datetime(2020, 1, 1)
            age = self.calculate_putin_age_on_date(test_date.date())
            logger.info(f"Возраст на {test_date}: {age:.2f} лет")
            
            # Тест создания тестового изображения для валидации качества
            test_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
            test_image_path = "test_image.jpg"
            cv2.imwrite(test_image_path, test_image)
            
            # Тест валидации качества
            quality_result = self.validate_image_quality_for_analysis(test_image)
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