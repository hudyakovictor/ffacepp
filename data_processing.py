# data_processing.py
import os
import json
import logging
import asyncio
import hashlib
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from collections import OrderedDict, defaultdict
import numpy as np
import cv2
from PIL import Image
import torch
import psutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pickle
import msgpack
from functools import lru_cache
import time

from core_config import get_config

# Настройка логирования
logger = logging.getLogger(__name__)

# === КОНСТАНТЫ И КОНФИГУРАЦИЯ ===

# Размеры изображений
TARGET_SIZE = (800, 800)
INSIGHTFACE_SIZE = (112, 112)

# Пороги качества для предварительной фильтрации
QUALITY_THRESHOLDS = {
    'min_variance_of_laplacian': 100.0,  # Порог размытия
    'min_shannon_entropy': 5.0,          # Минимальная энтропия
    'max_shannon_entropy': 8.5,          # Максимальная энтропия
    'min_brightness': 50,                # Минимальная яркость
    'max_brightness': 200,               # Максимальная яркость
    'min_contrast': 0.3,                 # Минимальный контраст
    'noise_threshold': 0.15              # Порог шума
}

# Параметры батч-обработки
BATCH_SIZE = 8
MAX_CONCURRENT_FILES = 16
CACHE_SIZE_LIMIT_MB = 1024

# === СТРУКТУРЫ ДАННЫХ ===

@dataclass
class AnalysisResult:
    """Результат анализа одного изображения"""
    image_id: str
    filepath: str
    filename: str
    date: datetime.date
    age_on_date: float
    processing_status: str
    quality_score: float
    quality_flags: List[str]
    
    # Результаты анализа
    landmarks_3d: Optional[np.ndarray] = None
    pose_angles: Optional[Dict[str, float]] = None
    identity_metrics: Optional[Dict[str, float]] = None
    embedding_vector: Optional[np.ndarray] = None
    texture_metrics: Optional[Dict[str, Any]] = None
    temporal_score: Optional[float] = None
    authenticity_score: Optional[float] = None
    
    # Метаданные обработки
    processing_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # Кэш-информация
    cache_hit: bool = False
    cache_key: Optional[str] = None

@dataclass
class BatchProcessingStats:
    """Статистика батч-обработки"""
    total_files: int = 0
    processed_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    
    total_processing_time_ms: float = 0.0
    average_processing_time_ms: float = 0.0
    peak_memory_usage_mb: float = 0.0
    
    quality_distribution: Dict[str, int] = field(default_factory=dict)
    error_counts: Dict[str, int] = field(default_factory=dict)
    
    start_time: Optional[datetime.datetime] = None
    end_time: Optional[datetime.datetime] = None

# === ОСНОВНОЙ КЛАСС ОБРАБОТКИ ДАННЫХ ===

class DataProcessing:
    """Основной класс для обработки и предварительного анализа изображений"""
    
    def __init__(self):
        self.config = get_config()
        self.cache_dir = Path("./cache/data_processing")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Кэш результатов
        self.results_cache: Dict[str, AnalysisResult] = {}
        self.cache_size_mb = 0.0
        
        # Статистика
        self.batch_stats = BatchProcessingStats()
        
        # Пулы для параллельной обработки
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.process_pool = ProcessPoolExecutor(max_workers=2)
        
        # Семафор для ограничения одновременно загруженных файлов
        self.file_semaphore = asyncio.Semaphore(MAX_CONCURRENT_FILES)
        
        logger.info("DataProcessing инициализирован")

    def preprocessing_pipeline(self, image_path: str, target_size: Tuple[int, int] = TARGET_SIZE) -> Optional[Dict[str, Any]]:
        """
        Основной пайплайн предварительной обработки изображения
        
        Args:
            image_path: Путь к изображению
            target_size: Целевой размер (по умолчанию 800x800)
            
        Returns:
            Словарь с обработанными данными или None при ошибке
        """
        try:
            start_time = time.time()
            
            # Проверка существования файла
            if not os.path.exists(image_path):
                logger.error(f"Файл не найден: {image_path}")
                return None
            
            # Загрузка изображения
            original_image = cv2.imread(image_path)
            if original_image is None:
                logger.error(f"Не удалось загрузить изображение: {image_path}")
                return None
            
            # Создание копии для обработки (избегаем мутации оригинала)
            image = original_image.copy()
            
            # Получение исходных размеров
            original_height, original_width = image.shape[:2]
            
            # Ресайз только если размер отличается от целевого
            if (original_width, original_height) != target_size:
                image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
                logger.debug(f"Изображение изменено с {original_width}x{original_height} до {target_size}")
            
            # Создание версии для InsightFace (112x112)
            insightface_image = cv2.resize(image, INSIGHTFACE_SIZE, interpolation=cv2.INTER_AREA)
            
            # Конвертация в RGB для дальнейшей обработки
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            insightface_rgb = cv2.cvtColor(insightface_image, cv2.COLOR_BGR2RGB)
            
            # Нормализация для нейронных сетей
            image_normalized = image_rgb.astype(np.float32) / 255.0
            insightface_normalized = insightface_rgb.astype(np.float32) / 255.0
            
            # Создание теневых массивов для анализа
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # CLAHE для улучшения контраста
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_image = clahe.apply(gray_image)
            contrast_factor = np.std(enhanced_image) / np.std(gray_image) if np.std(gray_image) > 0 else 1.0
            
            # Подготовка результата
            result = {
                'original_image': original_image,
                'processed_image': image,
                'image_rgb': image_rgb,
                'image_normalized': image_normalized,
                'insightface_image': insightface_image,
                'insightface_rgb': insightface_rgb,
                'insightface_normalized': insightface_normalized,
                'gray_image': gray_image,
                'enhanced_image': enhanced_image,
                'contrast_factor': contrast_factor,
                'original_size': (original_width, original_height),
                'target_size': target_size,
                'processing_time_ms': (time.time() - start_time) * 1000
            }
            
            logger.debug(f"Предобработка завершена за {result['processing_time_ms']:.1f}мс: {image_path}")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка предобработки {image_path}: {e}")
            return None

    def assess_image_quality(self, image: np.ndarray, enhanced_image: Optional[np.ndarray] = None) -> Tuple[float, List[str]]:
        """
        Оценка качества изображения для анализа
        
        Args:
            image: Исходное изображение (цветное)
            enhanced_image: Улучшенное изображение (опционально)
            
        Returns:
            Кортеж (общий балл качества, список флагов проблем)
        """
        try:
            flags = []
            scores = {}
            
            # Конвертация в градации серого если нужно
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # 1. Оценка резкости (Variance of Laplacian)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < QUALITY_THRESHOLDS['min_variance_of_laplacian']:
                flags.append("blurry")
                scores['sharpness'] = laplacian_var / QUALITY_THRESHOLDS['min_variance_of_laplacian']
            else:
                scores['sharpness'] = min(1.0, laplacian_var / 300.0)
            
            # 2. Оценка энтропии (информационное содержание)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_norm = hist / hist.sum()
            hist_norm = hist_norm[hist_norm > 0]  # Убираем нули для логарифма
            entropy = -np.sum(hist_norm * np.log2(hist_norm))
            
            if entropy < QUALITY_THRESHOLDS['min_shannon_entropy']:
                flags.append("low_entropy")
                scores['entropy'] = entropy / QUALITY_THRESHOLDS['min_shannon_entropy']
            elif entropy > QUALITY_THRESHOLDS['max_shannon_entropy']:
                flags.append("high_entropy")
                scores['entropy'] = 1.0 - (entropy - QUALITY_THRESHOLDS['max_shannon_entropy']) / 2.0
            else:
                scores['entropy'] = 1.0
            
            # 3. Оценка яркости
            mean_brightness = np.mean(gray)
            if mean_brightness < QUALITY_THRESHOLDS['min_brightness']:
                flags.append("too_dark")
                scores['brightness'] = mean_brightness / QUALITY_THRESHOLDS['min_brightness']
            elif mean_brightness > QUALITY_THRESHOLDS['max_brightness']:
                flags.append("too_bright")
                scores['brightness'] = 1.0 - (mean_brightness - QUALITY_THRESHOLDS['max_brightness']) / 55
            else:
                scores['brightness'] = 1.0
            
            # 4. Оценка контрастности
            contrast = np.std(gray) / 255.0
            if contrast < QUALITY_THRESHOLDS['min_contrast']:
                flags.append("low_contrast")
                scores['contrast'] = contrast / QUALITY_THRESHOLDS['min_contrast']
            else:
                scores['contrast'] = min(1.0, contrast / 0.5)
            
            # 5. Оценка шума
            if enhanced_image is not None:
                noise_level = np.std(cv2.GaussianBlur(gray, (5, 5), 0) - gray) / 255.0
                if noise_level > QUALITY_THRESHOLDS['noise_threshold']:
                    flags.append("noisy")
                    scores['noise'] = max(0.0, 1.0 - noise_level / QUALITY_THRESHOLDS['noise_threshold'])
                else:
                    scores['noise'] = 1.0
            else:
                scores['noise'] = 1.0
            
            # Расчет общего балла
            overall_score = np.mean(list(scores.values()))
            
            logger.debug(f"Оценка качества: {overall_score:.3f}, флаги: {flags}")
            return overall_score, flags
            
        except Exception as e:
            logger.error(f"Ошибка оценки качества: {e}")
            return 0.0, ["quality_assessment_error"]

    def generate_cache_key(self, filepath: str) -> str:
        """
        Генерация ключа кэша на основе содержимого файла
        
        Args:
            filepath: Путь к файлу
            
        Returns:
            SHA-256 хэш содержимого файла
        """
        try:
            with open(filepath, 'rb') as f:
                file_content = f.read()
            return hashlib.sha256(file_content).hexdigest()
        except Exception as e:
            logger.error(f"Ошибка генерации ключа кэша для {filepath}: {e}")
            return hashlib.sha256(filepath.encode()).hexdigest()

    def save_to_cache(self, cache_key: str, result: AnalysisResult):
        """
        Сохранение результата в кэш
        
        Args:
            cache_key: Ключ кэша
            result: Результат для сохранения
        """
        try:
            cache_file = self.cache_dir / f"{cache_key}.msgpack"
            
            # Подготовка данных для сериализации
            cache_data = asdict(result)
            
            # Конвертация numpy массивов в списки
            for key, value in cache_data.items():
                if isinstance(value, np.ndarray):
                    cache_data[key] = value.tolist()
                elif key == 'date' and hasattr(value, 'isoformat'):
                    cache_data[key] = value.isoformat()
            
            # Сохранение в msgpack
            with open(cache_file, 'wb') as f:
                msgpack.pack(cache_data, f)
            
            # Обновление размера кэша
            file_size_mb = cache_file.stat().st_size / (1024 * 1024)
            self.cache_size_mb += file_size_mb
            
            # Добавление в память
            self.results_cache[cache_key] = result
            
            logger.debug(f"Результат сохранен в кэш: {cache_key}")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения в кэш {cache_key}: {e}")

    def load_from_cache(self, cache_key: str) -> Optional[AnalysisResult]:
        """
        Загрузка результата из кэша
        
        Args:
            cache_key: Ключ кэша
            
        Returns:
            Результат из кэша или None
        """
        try:
            # Проверка в памяти
            if cache_key in self.results_cache:
                logger.debug(f"Кэш-попадание в памяти: {cache_key}")
                return self.results_cache[cache_key]
            
            # Проверка на диске
            cache_file = self.cache_dir / f"{cache_key}.msgpack"
            if not cache_file.exists():
                return None
            
            # Загрузка из файла
            with open(cache_file, 'rb') as f:
                cache_data = msgpack.unpack(f)
            
            # Восстановление типов данных
            if 'date' in cache_data and isinstance(cache_data['date'], str):
                cache_data['date'] = datetime.date.fromisoformat(cache_data['date'])
            
            # Восстановление numpy массивов
            for key, value in cache_data.items():
                if isinstance(value, list) and key.endswith(('_3d', '_vector', '_metrics')):
                    cache_data[key] = np.array(value) if value else None
            
            # Создание объекта результата
            result = AnalysisResult(**cache_data)
            result.cache_hit = True
            
            # Добавление в память
            self.results_cache[cache_key] = result
            
            logger.debug(f"Результат загружен из кэша: {cache_key}")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка загрузки из кэша {cache_key}: {e}")
            return None

    def cleanup_cache(self):
        """Очистка кэша при превышении лимита размера"""
        try:
            if self.cache_size_mb <= CACHE_SIZE_LIMIT_MB:
                return
            
            logger.info(f"Очистка кэша: текущий размер {self.cache_size_mb:.1f}МБ")
            
            # Получение списка файлов кэша с временем модификации
            cache_files = []
            for cache_file in self.cache_dir.glob("*.msgpack"):
                mtime = cache_file.stat().st_mtime
                size_mb = cache_file.stat().st_size / (1024 * 1024)
                cache_files.append((cache_file, mtime, size_mb))
            
            # Сортировка по времени (старые первыми)
            cache_files.sort(key=lambda x: x[1])
            
            # Удаление старых файлов
            removed_size = 0.0
            target_size = CACHE_SIZE_LIMIT_MB * 0.8  # Очищаем до 80% от лимита
            
            for cache_file, mtime, size_mb in cache_files:
                if self.cache_size_mb - removed_size <= target_size:
                    break
                
                try:
                    cache_file.unlink()
                    removed_size += size_mb
                    
                    # Удаление из памяти
                    cache_key = cache_file.stem
                    if cache_key in self.results_cache:
                        del self.results_cache[cache_key]
                        
                except Exception as e:
                    logger.warning(f"Не удалось удалить файл кэша {cache_file}: {e}")
            
            self.cache_size_mb -= removed_size
            logger.info(f"Очистка завершена: удалено {removed_size:.1f}МБ")
            
        except Exception as e:
            logger.error(f"Ошибка очистки кэша: {e}")

    async def process_single_file_async(self, filepath: str, data_manager=None) -> Optional[AnalysisResult]:
        """
        Асинхронная обработка одного файла
        
        Args:
            filepath: Путь к файлу
            data_manager: Экземпляр DataManager для получения метаданных
            
        Returns:
            Результат анализа или None при ошибке
        """
        async with self.file_semaphore:
            try:
                start_time = time.time()
                
                # Генерация ключа кэша
                cache_key = self.generate_cache_key(filepath)
                
                # Проверка кэша
                cached_result = self.load_from_cache(cache_key)
                if cached_result is not None:
                    cached_result.cache_hit = True
                    return cached_result
                
                # Получение метаданных файла
                filename = os.path.basename(filepath)
                
                # Парсинг даты и возраста (если доступен data_manager)
                if data_manager:
                    date_result = data_manager.parse_date_from_filename(filename)
                    if date_result:
                        date_obj, sequence = date_result
                        age = data_manager.calculate_putin_age_on_date(date_obj)
                    else:
                        logger.warning(f"Не удалось распарсить дату из {filename}")
                        date_obj = datetime.date.today()
                        age = 0.0
                else:
                    date_obj = datetime.date.today()
                    age = 0.0
                
                # Предварительная обработка изображения
                processed_data = self.preprocessing_pipeline(filepath)
                if processed_data is None:
                    return self._create_failed_result(filepath, filename, date_obj, age, 
                                                    "preprocessing_failed", cache_key)
                
                # Оценка качества
                quality_score, quality_flags = self.assess_image_quality(
                    processed_data['image_rgb'], 
                    processed_data['enhanced_image']
                )
                
                # Определение статуса обработки
                if quality_score < 0.5:
                    processing_status = "quality_failed"
                    logger.warning(f"Изображение не прошло проверку качества: {filepath} (балл: {quality_score:.3f})")
                else:
                    processing_status = "ready_for_analysis"
                
                # Создание результата
                result = AnalysisResult(
                    image_id=cache_key,
                    filepath=filepath,
                    filename=filename,
                    date=date_obj,
                    age_on_date=age,
                    processing_status=processing_status,
                    quality_score=quality_score,
                    quality_flags=quality_flags,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    memory_usage_mb=psutil.Process().memory_info().rss / (1024 * 1024),
                    cache_key=cache_key,
                    cache_hit=False
                )
                
                # Сохранение в кэш
                self.save_to_cache(cache_key, result)
                
                logger.debug(f"Файл обработан: {filename} (статус: {processing_status})")
                return result
                
            except Exception as e:
                logger.error(f"Ошибка обработки файла {filepath}: {e}")
                return self._create_failed_result(filepath, os.path.basename(filepath), 
                                                datetime.date.today(), 0.0, "processing_error", 
                                                cache_key, str(e))

    def _create_failed_result(self, filepath: str, filename: str, date: datetime.date, 
                            age: float, status: str, cache_key: str, 
                            error_message: str = None) -> AnalysisResult:
        """Создание результата для неудачной обработки"""
        return AnalysisResult(
            image_id=cache_key,
            filepath=filepath,
            filename=filename,
            date=date,
            age_on_date=age,
            processing_status=status,
            quality_score=0.0,
            quality_flags=["processing_failed"],
            processing_time_ms=0.0,
            memory_usage_mb=0.0,
            error_message=error_message,
            cache_key=cache_key,
            cache_hit=False
        )

    async def process_batch_async(self, file_paths: List[str], data_manager=None) -> List[AnalysisResult]:
        """
        Асинхронная батч-обработка файлов
        
        Args:
            file_paths: Список путей к файлам
            data_manager: Экземпляр DataManager
            
        Returns:
            Список результатов анализа
        """
        try:
            # Инициализация статистики
            self.batch_stats = BatchProcessingStats(
                total_files=len(file_paths),
                start_time=datetime.datetime.now()
            )
            
            logger.info(f"Начинаем батч-обработку {len(file_paths)} файлов")
            
            # Создание задач для асинхронной обработки
            tasks = []
            for filepath in file_paths:
                task = asyncio.create_task(self.process_single_file_async(filepath, data_manager))
                tasks.append(task)
            
            # Выполнение всех задач
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Обработка результатов и статистики
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Ошибка обработки {file_paths[i]}: {result}")
                    self.batch_stats.failed_files += 1
                    self.batch_stats.error_counts[str(type(result).__name__)] = \
                        self.batch_stats.error_counts.get(str(type(result).__name__), 0) + 1
                elif result is None:
                    self.batch_stats.failed_files += 1
                    self.batch_stats.error_counts["null_result"] = \
                        self.batch_stats.error_counts.get("null_result", 0) + 1
                else:
                    valid_results.append(result)
                    self.batch_stats.processed_files += 1
                    
                    if result.processing_status in ["ready_for_analysis", "quality_failed"]:
                        self.batch_stats.successful_files += 1
                    else:
                        self.batch_stats.failed_files += 1
                    
                    # Статистика качества
                    if result.quality_score >= 0.8:
                        quality_category = "excellent"
                    elif result.quality_score >= 0.6:
                        quality_category = "good"
                    elif result.quality_score >= 0.4:
                        quality_category = "fair"
                    else:
                        quality_category = "poor"
                    
                    self.batch_stats.quality_distribution[quality_category] = \
                        self.batch_stats.quality_distribution.get(quality_category, 0) + 1
                    
                    # Обновление времени и памяти
                    self.batch_stats.total_processing_time_ms += result.processing_time_ms
                    self.batch_stats.peak_memory_usage_mb = max(
                        self.batch_stats.peak_memory_usage_mb, 
                        result.memory_usage_mb
                    )
            
            # Финализация статистики
            self.batch_stats.end_time = datetime.datetime.now()
            if self.batch_stats.processed_files > 0:
                self.batch_stats.average_processing_time_ms = \
                    self.batch_stats.total_processing_time_ms / self.batch_stats.processed_files
            
            # Очистка кэша при необходимости
            self.cleanup_cache()
            
            logger.info(f"Батч-обработка завершена: {self.batch_stats.successful_files}/{len(file_paths)} успешно")
            return valid_results
            
        except Exception as e:
            logger.error(f"Критическая ошибка батч-обработки: {e}")
            return []

    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики обработки
        
        Returns:
            Словарь со статистикой
        """
        stats = asdict(self.batch_stats)
        
        # Добавление информации о кэше
        stats['cache_info'] = {
            'cache_size_mb': self.cache_size_mb,
            'cache_entries': len(self.results_cache),
            'cache_hit_rate': self._calculate_cache_hit_rate()
        }
        
        # Добавление системной информации
        process = psutil.Process()
        memory_info = process.memory_info()
        stats['system_info'] = {
            'memory_usage_mb': memory_info.rss / (1024 * 1024),
            'cpu_percent': process.cpu_percent(),
            'thread_count': process.num_threads()
        }
        
        return stats

    def _calculate_cache_hit_rate(self) -> float:
        """Расчет процента попаданий в кэш"""
        if not self.results_cache:
            return 0.0
        
        cache_hits = sum(1 for result in self.results_cache.values() if result.cache_hit)
        return cache_hits / len(self.results_cache) * 100

    def export_results_to_json(self, results: List[AnalysisResult], output_path: str):
        """
        Экспорт результатов в JSON
        
        Args:
            results: Список результатов
            output_path: Путь для сохранения
        """
        try:
            export_data = {
                'metadata': {
                    'export_time': datetime.datetime.now().isoformat(),
                    'total_results': len(results),
                    'processing_stats': self.get_processing_statistics()
                },
                'results': []
            }
            
            for result in results:
                result_dict = asdict(result)
                
                # Конвертация специальных типов
                if 'date' in result_dict and hasattr(result_dict['date'], 'isoformat'):
                    result_dict['date'] = result_dict['date'].isoformat()
                
                # Конвертация numpy массивов
                for key, value in result_dict.items():
                    if isinstance(value, np.ndarray):
                        result_dict[key] = value.tolist()
                
                export_data['results'].append(result_dict)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Результаты экспортированы в {output_path}")
            
        except Exception as e:
            logger.error(f"Ошибка экспорта результатов: {e}")

    def clear_cache(self):
        """Полная очистка кэша"""
        try:
            # Очистка памяти
            self.results_cache.clear()
            
            # Удаление файлов кэша
            for cache_file in self.cache_dir.glob("*.msgpack"):
                cache_file.unlink()
            
            self.cache_size_mb = 0.0
            logger.info("Кэш полностью очищен")
            
        except Exception as e:
            logger.error(f"Ошибка очистки кэша: {e}")

    def __del__(self):
        """Деструктор для закрытия пулов"""
        try:
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=False)
            if hasattr(self, 'process_pool'):
                self.process_pool.shutdown(wait=False)
        except:
            pass

# === КЛАСС АГРЕГАТОРА РЕЗУЛЬТАТОВ ===

class ResultsAggregator:
    """Агрегатор для сбора и анализа результатов обработки"""
    
    def __init__(self):
        self.results: List[AnalysisResult] = []
        self.results_by_status: Dict[str, List[AnalysisResult]] = defaultdict(list)
        self.results_by_date: Dict[datetime.date, List[AnalysisResult]] = defaultdict(list)
        
    def add_results(self, results: List[AnalysisResult]):
        """Добавление результатов в агрегатор"""
        for result in results:
            self.results.append(result)
            self.results_by_status[result.processing_status].append(result)
            self.results_by_date[result.date].append(result)
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Получение сводной статистики"""
        if not self.results:
            return {"error": "Нет результатов для анализа"}
        
        total_results = len(self.results)
        quality_scores = [r.quality_score for r in self.results]
        processing_times = [r.processing_time_ms for r in self.results if r.processing_time_ms > 0]
        
        return {
            'total_files': total_results,
            'status_distribution': {status: len(results) for status, results in self.results_by_status.items()},
            'quality_statistics': {
                'mean': np.mean(quality_scores),
                'median': np.median(quality_scores),
                'std': np.std(quality_scores),
                'min': np.min(quality_scores),
                'max': np.max(quality_scores)
            },
            'performance_statistics': {
                'mean_processing_time_ms': np.mean(processing_times) if processing_times else 0,
                'total_processing_time_ms': np.sum(processing_times) if processing_times else 0
            },
            'date_range': {
                'earliest': min(r.date for r in self.results).isoformat(),
                'latest': max(r.date for r in self.results).isoformat(),
                'unique_dates': len(self.results_by_date)
            }
        }
    
    def filter_by_quality(self, min_quality: float = 0.5) -> List[AnalysisResult]:
        """Фильтрация результатов по качеству"""
        return [r for r in self.results if r.quality_score >= min_quality]
    
    def filter_by_status(self, status: str) -> List[AnalysisResult]:
        """Фильтрация результатов по статусу"""
        return self.results_by_status.get(status, [])

# === ФУНКЦИИ САМОТЕСТИРОВАНИЯ ===

def self_test():
    """Самотестирование модуля data_processing"""
    try:
        logger.info("Запуск самотестирования data_processing...")
        
        # Создание экземпляра процессора
        processor = DataProcessing()
        
        # Тест создания тестового изображения
        test_image = np.random.randint(0, 255, (800, 800, 3), dtype=np.uint8)
        test_path = processor.cache_dir / "test_image.jpg"
        cv2.imwrite(str(test_path), test_image)
        
        # Тест предобработки
        processed_data = processor.preprocessing_pipeline(str(test_path))
        assert processed_data is not None, "Предобработка не удалась"
        assert processed_data['target_size'] == (800, 800), "Неверный размер изображения"
        
        # Тест оценки качества
        quality_score, flags = processor.assess_image_quality(processed_data['image_rgb'])
        assert 0.0 <= quality_score <= 1.0, "Неверный диапазон балла качества"
        
        # Тест генерации ключа кэша
        cache_key = processor.generate_cache_key(str(test_path))
        assert len(cache_key) == 64, "Неверная длина ключа кэша"
        
        # Тест агрегатора
        aggregator = ResultsAggregator()
        test_result = AnalysisResult(
            image_id="test",
            filepath=str(test_path),
            filename="test_image.jpg",
            date=datetime.date.today(),
            age_on_date=70.0,
            processing_status="ready_for_analysis",
            quality_score=0.8,
            quality_flags=[]
        )
        
        aggregator.add_results([test_result])
        stats = aggregator.get_summary_statistics()
        assert stats['total_files'] == 1, "Неверная статистика агрегатора"
        
        # Очистка тестовых файлов
        test_path.unlink()
        processor.clear_cache()
        
        logger.info("Самотестирование data_processing завершено успешно")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка самотестирования: {e}")
        return False

# === ИНИЦИАЛИЗАЦИЯ ===

if __name__ == "__main__":
    # Запуск самотестирования при прямом вызове модуля
    success = self_test()
    if success:
        print("✅ Модуль data_processing работает корректно")
        
        # Демонстрация основной функциональности
        processor = DataProcessing()
        print(f"📊 Размер кэша: {processor.cache_size_mb:.1f}МБ")
        print(f"🔧 Лимит одновременных файлов: {MAX_CONCURRENT_FILES}")
        print(f"📏 Целевой размер изображений: {TARGET_SIZE}")
        print(f"💾 Кэш-директория: {processor.cache_dir}")
    else:
        print("❌ Обнаружены ошибки в модуле data_processing")
        exit(1)
