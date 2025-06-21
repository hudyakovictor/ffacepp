# data_manager.py
import os
import re
import json
import logging
import hashlib
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
import numpy as np
import cv2
from PIL import Image
import psutil

from core_config import get_config

# Настройка логирования
logger = logging.getLogger(__name__)

# === КОНСТАНТЫ И РЕГУЛЯРНЫЕ ВЫРАЖЕНИЯ ===

# Паттерн для парсинга имен файлов DD_MM_YY[-SEQ].jpg
FILENAME_PATTERN = re.compile(r'^(\d{2})_(\d{2})_(\d{2})(?:-(\d+))?\.jpe?g$', re.IGNORECASE)

# Дата рождения Владимира Путина для расчета возраста
PUTIN_BIRTH_DATE = datetime.date(1952, 10, 7)

# Поддерживаемые форматы изображений
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

# Пороги качества изображений
QUALITY_THRESHOLDS = {
    'min_resolution': (400, 400),
    'max_resolution': (2000, 2000),
    'min_brightness': 50,
    'max_brightness': 200,
    'min_contrast': 0.3,
    'blur_threshold': 100.0,
    'noise_threshold': 0.1,
    'min_file_size': 10240,  # 10 KB
    'max_file_size': 50 * 1024 * 1024  # 50 MB
}

# === СТРУКТУРЫ ДАННЫХ ===

@dataclass
class ImageMetadata:
    """Метаданные изображения"""
    filepath: str
    filename: str
    date: datetime.date
    age_on_date: float
    sequence: Optional[int]
    file_size: int
    file_hash: str
    resolution: Tuple[int, int]
    quality_score: float
    quality_flags: List[str]
    processing_status: str = "pending"
    error_message: Optional[str] = None

@dataclass
class QualityAssessment:
    """Результат оценки качества изображения"""
    overall_score: float
    brightness_score: float
    contrast_score: float
    sharpness_score: float
    noise_score: float
    resolution_score: float
    flags: List[str]
    is_valid: bool

@dataclass
class ChronologicalIndex:
    """Хронологический индекс изображений"""
    images_by_date: OrderedDict[datetime.date, List[ImageMetadata]]
    images_by_year: Dict[int, List[ImageMetadata]]
    total_images: int
    date_range: Tuple[datetime.date, datetime.date]
    age_range: Tuple[float, float]
    quality_stats: Dict[str, float]
    gaps_analysis: List[Dict[str, Any]]

@dataclass
class HistoricalEvent:
    """Историческое событие для корреляции"""
    date: datetime.date
    title: str
    category: str
    importance: int  # 1-5
    description: str

# === ОСНОВНОЙ КЛАСС МЕНЕДЖЕРА ДАННЫХ ===

class DataManager:
    """Менеджер данных для обработки и индексации изображений"""
    
    def __init__(self):
        self.config = get_config()
        self.cache_dir = Path("./cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.chronological_index: Optional[ChronologicalIndex] = None
        self.historical_events: List[HistoricalEvent] = []
        self.processing_stats = {
            'total_processed': 0,
            'valid_images': 0,
            'invalid_images': 0,
            'parsing_errors': 0,
            'quality_failures': 0
        }
        
        # Загрузка исторических событий
        self._load_historical_events()
        
        logger.info("DataManager инициализирован")

    def parse_date_from_filename(self, filename: str) -> Optional[Tuple[datetime.date, Optional[int]]]:
        """
        Парсинг даты из имени файла в формате DD_MM_YY[-SEQ].jpg
        
        Args:
            filename: Имя файла
            
        Returns:
            Кортеж (дата, последовательность) или None при ошибке
        """
        try:
            match = FILENAME_PATTERN.match(filename)
            if not match:
                logger.warning(f"Имя файла не соответствует паттерну: {filename}")
                return None
            
            day, month, year, sequence = match.groups()
            
            # Преобразование двузначного года в четырехзначный
            year_int = int(year)
            if year_int >= 50:  # 50-99 -> 1950-1999
                full_year = 1900 + year_int
            else:  # 00-49 -> 2000-2049
                full_year = 2000 + year_int
            
            # Создание объекта даты
            date_obj = datetime.date(full_year, int(month), int(day))
            
            # Валидация даты (должна быть после рождения Путина)
            if date_obj < PUTIN_BIRTH_DATE:
                logger.warning(f"Дата {date_obj} раньше даты рождения Путина: {filename}")
                return None
            
            # Валидация даты (не должна быть в будущем)
            if date_obj > datetime.date.today():
                logger.warning(f"Дата {date_obj} в будущем: {filename}")
                return None
            
            seq = int(sequence) if sequence else None
            
            logger.debug(f"Распознана дата {date_obj}, последовательность {seq} из файла {filename}")
            return date_obj, seq
            
        except ValueError as e:
            logger.error(f"Ошибка парсинга даты из файла {filename}: {e}")
            return None
        except Exception as e:
            logger.error(f"Неожиданная ошибка при парсинге {filename}: {e}")
            return None

    def calculate_putin_age_on_date(self, date: datetime.date) -> float:
        """
        Расчет возраста Путина на указанную дату
        
        Args:
            date: Дата для расчета возраста
            
        Returns:
            Возраст в годах (с точностью до дня)
        """
        try:
            delta = date - PUTIN_BIRTH_DATE
            age_years = delta.days / 365.25  # Учет високосных лет
            return round(age_years, 2)
        except Exception as e:
            logger.error(f"Ошибка расчета возраста для даты {date}: {e}")
            return 0.0

    def validate_image_quality_for_analysis(self, filepath: str) -> QualityAssessment:
        """
        Комплексная оценка качества изображения для анализа
        
        Args:
            filepath: Путь к файлу изображения
            
        Returns:
            Объект оценки качества
        """
        flags = []
        scores = {}
        
        try:
            # Проверка существования файла
            if not os.path.exists(filepath):
                return QualityAssessment(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                                       ["file_not_found"], False)
            
            # Проверка размера файла
            file_size = os.path.getsize(filepath)
            if file_size < QUALITY_THRESHOLDS['min_file_size']:
                flags.append("file_too_small")
            elif file_size > QUALITY_THRESHOLDS['max_file_size']:
                flags.append("file_too_large")
            
            # Загрузка изображения
            try:
                image = cv2.imread(filepath)
                if image is None:
                    return QualityAssessment(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                                           ["cannot_read_image"], False)
                
                height, width = image.shape[:2]
                
                # Проверка разрешения
                min_w, min_h = QUALITY_THRESHOLDS['min_resolution']
                max_w, max_h = QUALITY_THRESHOLDS['max_resolution']
                
                if width < min_w or height < min_h:
                    flags.append("resolution_too_low")
                    scores['resolution_score'] = 0.3
                elif width > max_w or height > max_h:
                    flags.append("resolution_too_high")
                    scores['resolution_score'] = 0.8
                else:
                    scores['resolution_score'] = 1.0
                
                # Оценка яркости
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                mean_brightness = np.mean(gray)
                
                if mean_brightness < QUALITY_THRESHOLDS['min_brightness']:
                    flags.append("too_dark")
                    scores['brightness_score'] = mean_brightness / QUALITY_THRESHOLDS['min_brightness']
                elif mean_brightness > QUALITY_THRESHOLDS['max_brightness']:
                    flags.append("too_bright")
                    scores['brightness_score'] = 1.0 - (mean_brightness - QUALITY_THRESHOLDS['max_brightness']) / 55
                else:
                    scores['brightness_score'] = 1.0
                
                # Оценка контрастности
                contrast = np.std(gray) / 255.0
                if contrast < QUALITY_THRESHOLDS['min_contrast']:
                    flags.append("low_contrast")
                    scores['contrast_score'] = contrast / QUALITY_THRESHOLDS['min_contrast']
                else:
                    scores['contrast_score'] = min(1.0, contrast / 0.5)
                
                # Оценка резкости (Variance of Laplacian)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                if laplacian_var < QUALITY_THRESHOLDS['blur_threshold']:
                    flags.append("blurry")
                    scores['sharpness_score'] = laplacian_var / QUALITY_THRESHOLDS['blur_threshold']
                else:
                    scores['sharpness_score'] = min(1.0, laplacian_var / 300.0)
                
                # Оценка шума
                noise_level = np.std(cv2.GaussianBlur(gray, (5, 5), 0) - gray)
                if noise_level > QUALITY_THRESHOLDS['noise_threshold'] * 255:
                    flags.append("noisy")
                    scores['noise_score'] = max(0.0, 1.0 - noise_level / (QUALITY_THRESHOLDS['noise_threshold'] * 255))
                else:
                    scores['noise_score'] = 1.0
                
            except Exception as e:
                logger.error(f"Ошибка анализа изображения {filepath}: {e}")
                return QualityAssessment(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                                       ["analysis_error"], False)
            
            # Расчет общего балла
            overall_score = np.mean(list(scores.values()))
            is_valid = overall_score >= 0.5 and len([f for f in flags if f in 
                                                   ["file_not_found", "cannot_read_image", "analysis_error"]]) == 0
            
            return QualityAssessment(
                overall_score=overall_score,
                brightness_score=scores.get('brightness_score', 0.0),
                contrast_score=scores.get('contrast_score', 0.0),
                sharpness_score=scores.get('sharpness_score', 0.0),
                noise_score=scores.get('noise_score', 0.0),
                resolution_score=scores.get('resolution_score', 0.0),
                flags=flags,
                is_valid=is_valid
            )
            
        except Exception as e:
            logger.error(f"Критическая ошибка оценки качества {filepath}: {e}")
            return QualityAssessment(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                                   ["critical_error"], False)

    def create_master_chronological_index(self, image_paths: List[str]) -> ChronologicalIndex:
        """
        Создание мастер-индекса изображений по хронологии
        
        Args:
            image_paths: Список путей к изображениям
            
        Returns:
            Хронологический индекс
        """
        logger.info(f"Создание хронологического индекса для {len(image_paths)} файлов")
        
        images_by_date = OrderedDict()
        images_by_year = defaultdict(list)
        valid_images = []
        
        self.processing_stats = {
            'total_processed': 0,
            'valid_images': 0,
            'invalid_images': 0,
            'parsing_errors': 0,
            'quality_failures': 0
        }
        
        for filepath in image_paths:
            self.processing_stats['total_processed'] += 1
            
            try:
                filename = os.path.basename(filepath)
                
                # Парсинг даты из имени файла
                date_result = self.parse_date_from_filename(filename)
                if date_result is None:
                    self.processing_stats['parsing_errors'] += 1
                    self.processing_stats['invalid_images'] += 1
                    continue
                
                date_obj, sequence = date_result
                
                # Расчет возраста
                age = self.calculate_putin_age_on_date(date_obj)
                
                # Вычисление хеша файла
                try:
                    with open(filepath, 'rb') as f:
                        file_hash = hashlib.sha256(f.read()).hexdigest()
                except Exception as e:
                    logger.error(f"Ошибка вычисления хеша для {filepath}: {e}")
                    file_hash = "unknown"
                
                # Получение размера файла и разрешения
                file_size = os.path.getsize(filepath)
                
                try:
                    with Image.open(filepath) as img:
                        resolution = img.size
                except Exception as e:
                    logger.warning(f"Не удалось получить разрешение {filepath}: {e}")
                    resolution = (0, 0)
                
                # Оценка качества
                quality_assessment = self.validate_image_quality_for_analysis(filepath)
                
                if not quality_assessment.is_valid:
                    self.processing_stats['quality_failures'] += 1
                    self.processing_stats['invalid_images'] += 1
                    logger.warning(f"Изображение не прошло проверку качества: {filepath}")
                
                # Создание метаданных
                metadata = ImageMetadata(
                    filepath=filepath,
                    filename=filename,
                    date=date_obj,
                    age_on_date=age,
                    sequence=sequence,
                    file_size=file_size,
                    file_hash=file_hash,
                    resolution=resolution,
                    quality_score=quality_assessment.overall_score,
                    quality_flags=quality_assessment.flags,
                    processing_status="valid" if quality_assessment.is_valid else "quality_failed"
                )
                
                # Добавление в индексы
                if date_obj not in images_by_date:
                    images_by_date[date_obj] = []
                images_by_date[date_obj].append(metadata)
                
                images_by_year[date_obj.year].append(metadata)
                
                if quality_assessment.is_valid:
                    valid_images.append(metadata)
                    self.processing_stats['valid_images'] += 1
                
            except Exception as e:
                logger.error(f"Ошибка обработки файла {filepath}: {e}")
                self.processing_stats['invalid_images'] += 1
                continue
        
        # Сортировка по датам
        images_by_date = OrderedDict(sorted(images_by_date.items()))
        
        # Сортировка внутри каждой даты по sequence
        for date_key in images_by_date:
            images_by_date[date_key].sort(key=lambda x: x.sequence or 0)
        
        # Расчет статистики
        if valid_images:
            dates = [img.date for img in valid_images]
            ages = [img.age_on_date for img in valid_images]
            quality_scores = [img.quality_score for img in valid_images]
            
            date_range = (min(dates), max(dates))
            age_range = (min(ages), max(ages))
            
            quality_stats = {
                'mean_quality': np.mean(quality_scores),
                'min_quality': np.min(quality_scores),
                'max_quality': np.max(quality_scores),
                'std_quality': np.std(quality_scores)
            }
        else:
            date_range = (datetime.date.today(), datetime.date.today())
            age_range = (0.0, 0.0)
            quality_stats = {'mean_quality': 0.0, 'min_quality': 0.0, 'max_quality': 0.0, 'std_quality': 0.0}
        
        # Анализ пропусков
        gaps_analysis = self._analyze_temporal_gaps(list(images_by_date.keys()))
        
        # Создание индекса
        chronological_index = ChronologicalIndex(
            images_by_date=images_by_date,
            images_by_year=dict(images_by_year),
            total_images=len(valid_images),
            date_range=date_range,
            age_range=age_range,
            quality_stats=quality_stats,
            gaps_analysis=gaps_analysis
        )
        
        self.chronological_index = chronological_index
        
        logger.info(f"Создан хронологический индекс: {len(valid_images)} валидных изображений "
                   f"из {len(image_paths)} обработанных")
        logger.info(f"Диапазон дат: {date_range[0]} - {date_range[1]}")
        logger.info(f"Диапазон возрастов: {age_range[0]:.1f} - {age_range[1]:.1f} лет")
        
        return chronological_index

    def _analyze_temporal_gaps(self, dates: List[datetime.date]) -> List[Dict[str, Any]]:
        """
        Анализ временных пропусков в данных
        
        Args:
            dates: Отсортированный список дат
            
        Returns:
            Список пропусков с метаданными
        """
        if len(dates) < 2:
            return []
        
        gaps = []
        gap_threshold = datetime.timedelta(days=180)  # 6 месяцев
        
        for i in range(1, len(dates)):
            gap = dates[i] - dates[i-1]
            if gap > gap_threshold:
                gaps.append({
                    'start_date': dates[i-1],
                    'end_date': dates[i],
                    'duration_days': gap.days,
                    'duration_months': gap.days / 30.44,
                    'severity': 'major' if gap.days > 365 else 'minor'
                })
        
        return gaps

    def _load_historical_events(self):
        """Загрузка исторических событий для корреляции"""
        # Примеры ключевых событий (в реальной системе загружались бы из базы данных)
        mock_events = [
            HistoricalEvent(datetime.date(1999, 12, 31), "Назначение и.о. президента", "political", 5, 
                          "Борис Ельцин объявил об отставке"),
            HistoricalEvent(datetime.date(2000, 3, 26), "Избрание президентом", "political", 5,
                          "Первое избрание президентом РФ"),
            HistoricalEvent(datetime.date(2004, 3, 14), "Переизбрание", "political", 4,
                          "Второе избрание президентом РФ"),
            HistoricalEvent(datetime.date(2008, 5, 8), "Назначение премьер-министром", "political", 4,
                          "Переход на пост премьер-министра"),
            HistoricalEvent(datetime.date(2012, 5, 7), "Возвращение на пост президента", "political", 5,
                          "Третье избрание президентом РФ"),
            HistoricalEvent(datetime.date(2018, 5, 7), "Четвертый срок", "political", 4,
                          "Четвертое избрание президентом РФ"),
            HistoricalEvent(datetime.date(2024, 5, 7), "Пятый срок", "political", 4,
                          "Пятое избрание президентом РФ")
        ]
        
        self.historical_events = mock_events
        logger.info(f"Загружено {len(self.historical_events)} исторических событий")

    def correlate_with_historical_events(self, date: datetime.date, 
                                       tolerance_days: int = 30) -> List[HistoricalEvent]:
        """
        Поиск исторических событий рядом с указанной датой
        
        Args:
            date: Дата для поиска
            tolerance_days: Допустимое отклонение в днях
            
        Returns:
            Список ближайших событий
        """
        nearby_events = []
        tolerance = datetime.timedelta(days=tolerance_days)
        
        for event in self.historical_events:
            if abs((event.date - date).days) <= tolerance_days:
                nearby_events.append(event)
        
        # Сортировка по близости к дате
        nearby_events.sort(key=lambda e: abs((e.date - date).days))
        
        return nearby_events

    def get_images_by_date_range(self, start_date: datetime.date, 
                                end_date: datetime.date) -> List[ImageMetadata]:
        """
        Получение изображений в указанном диапазоне дат
        
        Args:
            start_date: Начальная дата
            end_date: Конечная дата
            
        Returns:
            Список метаданных изображений
        """
        if self.chronological_index is None:
            logger.warning("Хронологический индекс не создан")
            return []
        
        result = []
        for date, images in self.chronological_index.images_by_date.items():
            if start_date <= date <= end_date:
                result.extend(images)
        
        return result

    def get_images_by_year(self, year: int) -> List[ImageMetadata]:
        """
        Получение изображений за указанный год
        
        Args:
            year: Год
            
        Returns:
            Список метаданных изображений
        """
        if self.chronological_index is None:
            logger.warning("Хронологический индекс не создан")
            return []
        
        return self.chronological_index.images_by_year.get(year, [])

    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики обработки
        
        Returns:
            Словарь со статистикой
        """
        stats = self.processing_stats.copy()
        
        if stats['total_processed'] > 0:
            stats['success_rate'] = stats['valid_images'] / stats['total_processed']
            stats['quality_failure_rate'] = stats['quality_failures'] / stats['total_processed']
            stats['parsing_error_rate'] = stats['parsing_errors'] / stats['total_processed']
        else:
            stats['success_rate'] = 0.0
            stats['quality_failure_rate'] = 0.0
            stats['parsing_error_rate'] = 0.0
        
        # Добавление информации о памяти
        process = psutil.Process()
        memory_info = process.memory_info()
        stats['memory_usage_mb'] = memory_info.rss / 1024 / 1024
        
        return stats

    def save_index_to_cache(self, cache_filename: str = "chronological_index.json"):
        """
        Сохранение индекса в кэш
        
        Args:
            cache_filename: Имя файла кэша
        """
        if self.chronological_index is None:
            logger.warning("Нет индекса для сохранения")
            return
        
        cache_path = self.cache_dir / cache_filename
        
        try:
            # Подготовка данных для сериализации
            cache_data = {
                'images_by_date': {
                    date.isoformat(): [
                        {
                            'filepath': img.filepath,
                            'filename': img.filename,
                            'date': img.date.isoformat(),
                            'age_on_date': img.age_on_date,
                            'sequence': img.sequence,
                            'file_size': img.file_size,
                            'file_hash': img.file_hash,
                            'resolution': img.resolution,
                            'quality_score': img.quality_score,
                            'quality_flags': img.quality_flags,
                            'processing_status': img.processing_status,
                            'error_message': img.error_message
                        }
                        for img in images
                    ]
                    for date, images in self.chronological_index.images_by_date.items()
                },
                'total_images': self.chronological_index.total_images,
                'date_range': [
                    self.chronological_index.date_range[0].isoformat(),
                    self.chronological_index.date_range[1].isoformat()
                ],
                'age_range': self.chronological_index.age_range,
                'quality_stats': self.chronological_index.quality_stats,
                'gaps_analysis': [
                    {
                        'start_date': gap['start_date'].isoformat(),
                        'end_date': gap['end_date'].isoformat(),
                        'duration_days': gap['duration_days'],
                        'duration_months': gap['duration_months'],
                        'severity': gap['severity']
                    }
                    for gap in self.chronological_index.gaps_analysis
                ],
                'processing_stats': self.processing_stats,
                'cache_timestamp': datetime.datetime.now().isoformat()
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Индекс сохранен в кэш: {cache_path}")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения индекса в кэш: {e}")

    def load_index_from_cache(self, cache_filename: str = "chronological_index.json") -> bool:
        """
        Загрузка индекса из кэша
        
        Args:
            cache_filename: Имя файла кэша
            
        Returns:
            True если загрузка успешна
        """
        cache_path = self.cache_dir / cache_filename
        
        if not cache_path.exists():
            logger.info("Файл кэша не найден")
            return False
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Восстановление структуры данных
            images_by_date = OrderedDict()
            
            for date_str, images_data in cache_data['images_by_date'].items():
                date_obj = datetime.date.fromisoformat(date_str)
                images_list = []
                
                for img_data in images_data:
                    metadata = ImageMetadata(
                        filepath=img_data['filepath'],
                        filename=img_data['filename'],
                        date=datetime.date.fromisoformat(img_data['date']),
                        age_on_date=img_data['age_on_date'],
                        sequence=img_data['sequence'],
                        file_size=img_data['file_size'],
                        file_hash=img_data['file_hash'],
                        resolution=tuple(img_data['resolution']),
                        quality_score=img_data['quality_score'],
                        quality_flags=img_data['quality_flags'],
                        processing_status=img_data['processing_status'],
                        error_message=img_data.get('error_message')
                    )
                    images_list.append(metadata)
                
                images_by_date[date_obj] = images_list
            
            # Восстановление gaps_analysis
            gaps_analysis = []
            for gap_data in cache_data['gaps_analysis']:
                gap = {
                    'start_date': datetime.date.fromisoformat(gap_data['start_date']),
                    'end_date': datetime.date.fromisoformat(gap_data['end_date']),
                    'duration_days': gap_data['duration_days'],
                    'duration_months': gap_data['duration_months'],
                    'severity': gap_data['severity']
                }
                gaps_analysis.append(gap)
            
            # Восстановление images_by_year
            images_by_year = defaultdict(list)
            for images_list in images_by_date.values():
                for img in images_list:
                    images_by_year[img.date.year].append(img)
            
            # Создание индекса
            self.chronological_index = ChronologicalIndex(
                images_by_date=images_by_date,
                images_by_year=dict(images_by_year),
                total_images=cache_data['total_images'],
                date_range=(
                    datetime.date.fromisoformat(cache_data['date_range'][0]),
                    datetime.date.fromisoformat(cache_data['date_range'][1])
                ),
                age_range=tuple(cache_data['age_range']),
                quality_stats=cache_data['quality_stats'],
                gaps_analysis=gaps_analysis
            )
            
            self.processing_stats = cache_data.get('processing_stats', {})
            
            logger.info(f"Индекс загружен из кэша: {cache_path}")
            logger.info(f"Загружено {self.chronological_index.total_images} изображений")
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки индекса из кэша: {e}")
            return False

    def generate_quality_report(self) -> Dict[str, Any]:
        """
        Генерация отчета о качестве изображений
        
        Returns:
            Словарь с отчетом о качестве
        """
        if self.chronological_index is None:
            return {"error": "Индекс не создан"}
        
        all_images = []
        for images_list in self.chronological_index.images_by_date.values():
            all_images.extend(images_list)
        
        if not all_images:
            return {"error": "Нет изображений для анализа"}
        
        # Статистика по качеству
        quality_scores = [img.quality_score for img in all_images]
        
        # Группировка по флагам качества
        flag_counts = defaultdict(int)
        for img in all_images:
            for flag in img.quality_flags:
                flag_counts[flag] += 1
        
        # Статистика по годам
        quality_by_year = defaultdict(list)
        for img in all_images:
            quality_by_year[img.date.year].append(img.quality_score)
        
        year_stats = {}
        for year, scores in quality_by_year.items():
            year_stats[year] = {
                'count': len(scores),
                'mean_quality': np.mean(scores),
                'min_quality': np.min(scores),
                'max_quality': np.max(scores),
                'std_quality': np.std(scores)
            }
        
        report = {
            'total_images': len(all_images),
            'overall_quality_stats': {
                'mean': np.mean(quality_scores),
                'median': np.median(quality_scores),
                'std': np.std(quality_scores),
                'min': np.min(quality_scores),
                'max': np.max(quality_scores)
            },
            'quality_distribution': {
                'excellent': len([s for s in quality_scores if s >= 0.9]),
                'good': len([s for s in quality_scores if 0.7 <= s < 0.9]),
                'fair': len([s for s in quality_scores if 0.5 <= s < 0.7]),
                'poor': len([s for s in quality_scores if s < 0.5])
            },
            'common_issues': dict(flag_counts),
            'quality_by_year': year_stats,
            'processing_stats': self.get_processing_statistics()
        }
        
        return report

# === ФУНКЦИИ САМОТЕСТИРОВАНИЯ ===

def self_test():
    """Самотестирование модуля data_manager"""
    try:
        logger.info("Запуск самотестирования data_manager...")
        
        # Создание экземпляра менеджера
        dm = DataManager()
        
        # Тест парсинга даты
        test_cases = [
            ("01_01_00.jpg", datetime.date(2000, 1, 1), None),
            ("15_06_99.jpg", datetime.date(1999, 6, 15), None),
            ("31_12_23-2.jpg", datetime.date(2023, 12, 31), 2),
            ("invalid.jpg", None, None)
        ]
        
        for filename, expected_date, expected_seq in test_cases:
            result = dm.parse_date_from_filename(filename)
            if expected_date is None:
                assert result is None, f"Ожидался None для {filename}"
            else:
                assert result is not None, f"Ожидался результат для {filename}"
                date, seq = result
                assert date == expected_date, f"Неверная дата для {filename}"
                assert seq == expected_seq, f"Неверная последовательность для {filename}"
        
        # Тест расчета возраста
        test_date = datetime.date(2000, 1, 1)
        age = dm.calculate_putin_age_on_date(test_date)
        expected_age = (test_date - PUTIN_BIRTH_DATE).days / 365.25
        assert abs(age - expected_age) < 0.01, "Неверный расчет возраста"
        
        # Тест корреляции с событиями
        events = dm.correlate_with_historical_events(datetime.date(2000, 1, 1), 60)
        assert len(events) > 0, "Должны быть найдены события"
        
        logger.info("Самотестирование data_manager завершено успешно")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка самотестирования: {e}")
        return False

# === ИНИЦИАЛИЗАЦИЯ ===

if __name__ == "__main__":
    # Запуск самотестирования при прямом вызове модуля
    success = self_test()
    if success:
        print("✅ Модуль data_manager работает корректно")
        
        # Демонстрация основной функциональности
        dm = DataManager()
        print(f"📊 Загружено исторических событий: {len(dm.historical_events)}")
        print(f"🔧 Поддерживаемые форматы: {SUPPORTED_FORMATS}")
        print(f"📏 Пороги качества настроены")
        print(f"💾 Кэш-директория: {dm.cache_dir}")
    else:
        print("❌ Обнаружены ошибки в модуле data_manager")
        exit(1)
