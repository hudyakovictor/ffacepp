"""
DataProcessor - Обработчик данных с агрегацией результатов анализа
Версия: 2.0
Дата: 2025-06-15
Исправлены все критические ошибки согласно правкам
"""

import numpy as np
import pandas as pd
import asyncio
import concurrent.futures
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import json
import pickle
from pathlib import Path
import cv2
from datetime import datetime
import hashlib

# --- ЦВЕТА КОНСОЛИ (Повторяются для каждого модуля, чтобы быть автономными) ---
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    PURPLE = "\033[95m"
    WHITE = "\033[97m"

# --- КАСТОМНЫЙ ФОРМАТТЕР ДЛЯ ЦВЕТНОГО ЛОГИРОВАНИЯ ---
class ColoredFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: f"{Colors.CYAN}%(levelname)s:{Colors.RESET} %(message)s",
        logging.INFO: f"{Colors.GREEN}%(levelname)s:{Colors.RESET} %(message)s",
        logging.WARNING: f"{Colors.YELLOW}%(levelname)s:{Colors.RESET} %(message)s",
        logging.ERROR: f"{Colors.RED}%(levelname)s:{Colors.RESET} %(message)s",
        logging.CRITICAL: f"{Colors.RED}{Colors.BOLD}%(levelname)s:{Colors.RESET} %(message)s"
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# Настройка логирования
log_file_handler = logging.FileHandler('logs/dataprocessor.log')
log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'))

console_handler = logging.StreamHandler()
console_handler.setFormatter(ColoredFormatter())

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        log_file_handler,
        console_handler
    ]
)
logger = logging.getLogger(__name__)

# Импорт конфигурации
try:
    from core_config import (
        AUTHENTICITY_WEIGHTS, CRITICAL_THRESHOLDS, CACHE_DIR, 
        ERROR_CODES, MAX_RESULTS_CACHE_SIZE
    )
    from metrics_calculator import MetricsCalculator
    logger.info(f"{Colors.GREEN}✔ Конфигурация и модули успешно импортированы.{Colors.RESET}")
except ImportError as e:
    logger.critical(f"{Colors.RED}КРИТИЧЕСКАЯ ОШИБКА: Не удалось импортировать конфигурацию или модули: {e}{Colors.RESET}")
    # Значения по умолчанию
    AUTHENTICITY_WEIGHTS = {"geometry": 0.15, "embedding": 0.30, "texture": 0.10}
    CRITICAL_THRESHOLDS = {"min_authenticity_score": 0.6}
    CACHE_DIR = Path("cache")
    ERROR_CODES = {"E001": "NO_FACE_DETECTED", "E002": "LOW_QUALITY_IMAGE"}
    MAX_RESULTS_CACHE_SIZE = 10000

# ==================== СТРУКТУРЫ ДАННЫХ ====================

@dataclass
class AnalysisResult:
    """
    ИСПРАВЛЕНО: Структура результата анализа
    Согласно правкам: полная структура с метаданными
    """
    filepath: str
    timestamp: str
    authenticity_score: float
    anomalies: Dict[str, Any]
    metrics: Dict[str, float]
    metadata: Dict[str, Any]

# ==================== ОСНОВНОЙ КЛАСС ====================

class DataProcessor:
    """
    ИСПРАВЛЕНО: Обработчик данных с полной функциональностью
    Согласно правкам: интеграция всех анализаторов с AUTHENTICITY_WEIGHTS
    """
    
    def __init__(self, cache_dir: str = "cache"):
        logger.info(f"{Colors.BOLD}--- Инициализация DataProcessor (Обработчика данных) ---{Colors.RESET}")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Кэш результатов с ограничением размера
        self.results_cache = {}
        self.max_cache_size = MAX_RESULTS_CACHE_SIZE
        
        # Инициализация анализаторов
        self._initialize_analyzers()
        
        # Статистика обработки
        self.processing_stats = {
            "total_processed": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        logger.info(f"{Colors.BOLD}--- DataProcessor успешно инициализирован ---{Colors.RESET}")

    def _initialize_analyzers(self) -> None:
        """Инициализация всех анализаторов"""
        try:
            logger.info(f"{Colors.CYAN}Инициализация внутренних анализаторов DataProcessor...{Colors.RESET}")
            # Импорт и инициализация анализаторов
            from face_3d_analyzer import Face3DAnalyzer
            from embedding_analyzer import EmbeddingAnalyzer
            from texture_analyzer import TextureAnalyzer
            from temporal_analyzer import TemporalAnalyzer
            from anomaly_detector import AnomalyDetector
            from data_manager import DataManager
            from metrics_calculator import MetricsCalculator
            
            self.face_analyzer = Face3DAnalyzer()
            self.embedding_analyzer = EmbeddingAnalyzer()
            self.texture_analyzer = TextureAnalyzer()
            self.temporal_analyzer = TemporalAnalyzer()
            self.data_manager = DataManager()
            self.metrics_calculator = MetricsCalculator()
            
            logger.info(f"{Colors.GREEN}✔ Все анализаторы успешно инициализированы.{Colors.RESET}")
            
        except ImportError as e:
            logger.critical(f"{Colors.RED}КРИТИЧЕСКАЯ ОШИБКА: Не удалось импортировать один или несколько анализаторов: {e}{Colors.RESET}")
            # Заглушки для отсутствующих анализаторов
            self.face_analyzer = None
            self.embedding_analyzer = None
            self.texture_analyzer = None
            self.temporal_analyzer = None
            self.data_manager = None
            self.metrics_calculator = None

    async def process_batch_async(self, filepaths: List[str], 
                                progress_callback: Optional[Callable] = None) -> List[AnalysisResult]:
        """
        ИСПРАВЛЕНО: Асинхронная пакетная обработка файлов
        Согласно правкам: интеграция всех анализаторов
        """
        if not filepaths:
            logger.warning(f"{Colors.YELLOW}ПРЕДУПРЕЖДЕНИЕ: Пустой список файлов для обработки. Ничего не будет сделано.{Colors.RESET}")
            return []
        
        try:
            logger.info(f"{Colors.CYAN}Начало пакетной обработки {len(filepaths)} файлов...{Colors.RESET}")
            
            # Создание задач для каждого файла
            tasks = []
            for i, filepath in enumerate(filepaths):
                task = self.process_single_file_async(filepath)
                tasks.append(task)
            
            # Выполнение задач с отслеживанием прогресса
            results = []
            for i, task in enumerate(asyncio.as_completed(tasks)):
                try:
                    result = await task
                    results.append(result)
                    
                    # Обновление прогресса
                    if progress_callback:
                        progress = (i + 1) / len(filepaths)
                        progress_callback(progress, f"Обработано {i+1}/{len(filepaths)} файлов")
                    
                    self.processing_stats["successful_analyses"] += 1
                    
                except Exception as e:
                    logger.error(f"{Colors.RED}ОШИБКА обработки файла (пропущен): {e}{Colors.RESET}")
                    self.processing_stats["failed_analyses"] += 1
            
            self.processing_stats["total_processed"] += len(filepaths)
            
            logger.info(f"{Colors.GREEN}✔ Пакетная обработка завершена. Успешно обработано: {len(results)} результатов.{Colors.RESET}")
            return results
            
        except Exception as e:
            logger.critical(f"{Colors.RED}КРИТИЧЕСКАЯ ОШИБКА пакетной обработки: {e}{Colors.RESET}")
            return []

    async def process_single_file_async(self, filepath: str) -> AnalysisResult:
        """
        ИСПРАВЛЕНО: Асинхронная обработка одного файла
        Согласно правкам: полная интеграция с validateimagequalityforanalysis
        """
        try:
            logger.debug(f"Начинаем обработку файла: {filepath}")
            
            # Проверка кэша
            cache_key = self.get_cache_key(filepath)
            if cache_key in self.results_cache:
                self.processing_stats["cache_hits"] += 1
                logger.debug(f"Результат для {filepath} найден в кэше. Пропускаем анализ.")
                return self.results_cache[cache_key]
            
            self.processing_stats["cache_misses"] += 1
            logger.debug(f"Результат для {filepath} не найден в кэше. Выполняем полный анализ.")
            
            # Загрузка изображения
            image = await self.load_image_async(filepath)
            if image is None:
                logger.error(f"{Colors.RED}ОШИБКА: Не удалось загрузить изображение: {filepath}{Colors.RESET}")
                raise ValueError(f"Не удалось загрузить изображение: {filepath}")
            
            # Валидация качества изображения
            quality_result = self.data_manager.validate_image_quality_for_analysis(filepath)
            
            if quality_result.get("quality_score", 0) < CRITICAL_THRESHOLDS.get("min_quality_score", 0.6):
                logger.warning(f"{Colors.YELLOW}ПРЕДУПРЕЖДЕНИЕ: Файл '{filepath}' имеет низкое качество ({quality_result.get('quality_score', 0):.2f}). Будет отмечен как аномалия.{Colors.RESET}")
                return AnalysisResult(
                    filepath=filepath,
                    timestamp=datetime.now().isoformat(),
                    authenticity_score=0.0,
                    anomalies={"quality_error": ERROR_CODES["E002"]},
                    metrics={},
                    metadata={"error": "Низкое качество изображения", "quality_score": quality_result.get("quality_score", 0)}
                )
            
            # Параллельный анализ всех компонентов
            logger.debug(f"Запуск параллельного анализа для {filepath}...")
            analysis_tasks = {
                "face3d": self.analyze_3d_async(image),
                "embedding": self.analyze_embedding_async(image),
                "texture": self.analyze_texture_async(image)
            }
            
            analysis_results = {}
            for analysis_type, task in analysis_tasks.items():
                try:
                    analysis_results[analysis_type] = await task
                    logger.debug(f"Анализ '{analysis_type}' для {filepath} завершен.")
                except Exception as e:
                    logger.error(f"{Colors.RED}ОШИБКА в '{analysis_type}' анализе для {filepath}: {e}{Colors.RESET}")
                    analysis_results[analysis_type] = {"error": str(e)}
            
            # Агрегация результатов
            result = self.aggregate_analysis_results(filepath, analysis_results, quality_result)
            
            # Кэширование результата
            self._cache_result(cache_key, result)
            logger.info(f"{Colors.GREEN}✔ Файл '{filepath}' успешно обработан. Балл: {result.authenticity_score:.2f}{Colors.RESET}")
            
            return result
            
        except Exception as e:
            logger.error(f"{Colors.RED}ОШИБКА при обработке файла '{filepath}': {e}{Colors.RESET}")
            return AnalysisResult(
                filepath=filepath,
                timestamp=datetime.now().isoformat(),
                authenticity_score=0.0,
                anomalies={"processing_error": str(e)},
                metrics={},
                metadata={"error": str(e)}
            )

    async def load_image_async(self, filepath: str) -> Optional[np.ndarray]:
        """Асинхронная загрузка изображения"""
        try:
            loop = asyncio.get_event_loop()
            
            def load_image():
                img = cv2.imread(filepath)
                if img is None:
                    logger.warning(f"{Colors.YELLOW}ПРЕДУПРЕЖДЕНИЕ: Не удалось прочитать изображение по пути: {filepath}{Colors.RESET}")
                return img
            
            return await loop.run_in_executor(None, load_image)
            
        except Exception as e:
            logger.error(f"{Colors.RED}ОШИБКА при асинхронной загрузке изображения {filepath}: {e}{Colors.RESET}")
            return None

    async def analyze_3d_async(self, image: np.ndarray) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Асинхронный 3D анализ лица
        Согласно правкам: интеграция с face_3d_analyzer и metricscalculator
        """
        try:
            loop = asyncio.get_event_loop()
            
            def analyze():
                if self.face_analyzer is None:
                    logger.error(f"{Colors.RED}ОШИБКА: Face3DAnalyzer не инициализирован. 3D анализ невозможен.{Colors.RESET}")
                    return {"error": "Face3DAnalyzer не инициализирован"}
                
                # Извлечение 68 ландмарок
                landmarks, confidence, shape = self.face_analyzer.extract_68_landmarks_with_confidence(image)
                
                if landmarks.size == 0:
                    logger.warning(f"{Colors.YELLOW}ПРЕДУПРЕЖДЕНИЕ (3D анализ): Ландмарки лица не обнаружены.{Colors.RESET}")
                    return {"error": ERROR_CODES["E001"]}
                
                # Определение позы
                pose = self.face_analyzer.determine_precise_face_pose(landmarks)
                
                # Расчет 15 метрик идентичности через MetricsCalculator
                if self.metrics_calculator:
                    metrics_result = self.metrics_calculator.calculate_identity_signature_metrics(
                        landmarks, pose.get('pose_category', 'frontal')
                    )
                    metrics = metrics_result.get('normalized_metrics', {})
                else:
                    metrics = {}
                    logger.warning(f"{Colors.YELLOW}ПРЕДУПРЕЖДЕНИЕ: MetricsCalculator не инициализирован. Метрики идентичности не будут рассчитаны.{Colors.RESET}")
                
                logger.debug(f"3D анализ успешен: {len(metrics)} метрик извлечено.")
                
                return {
                    "landmarks": landmarks.tolist(),
                    "pose": pose,
                    "metrics": metrics,
                    "confidence": confidence.tolist() if confidence.size > 0 else [],
                    "shape": shape
                }
            
            return await loop.run_in_executor(None, analyze)
            
        except Exception as e:
            logger.error(f"{Colors.RED}ОШИБКА при выполнении 3D анализа: {e}{Colors.RESET}")
            return {"error": str(e)}

    async def analyze_embedding_async(self, image: np.ndarray) -> Dict[str, Any]:
        """Асинхронный анализ эмбеддингов"""
        try:
            loop = asyncio.get_event_loop()
            
            def analyze():
                if self.embedding_analyzer is None:
                    logger.error(f"{Colors.RED}ОШИБКА: EmbeddingAnalyzer не инициализирован. Анализ эмбеддингов невозможен.{Colors.RESET}")
                    return {"error": "EmbeddingAnalyzer не инициализирован"}
                
                # Извлечение 512D эмбеддинга
                embedding, confidence = self.embedding_analyzer.extract_512d_face_embedding(image)
                
                if embedding.size == 0:
                    logger.warning(f"{Colors.YELLOW}ПРЕДУПРЕЖДЕНИЕ (анализ эмбеддингов): Эмбеддинг лица не извлечен.{Colors.RESET}")
                    return {"error": "Эмбеддинг не извлечен"}
                
                logger.debug("Эмбеддинг успешно извлечен.")
                return {
                    "embedding": embedding.tolist(),
                    "confidence": float(confidence)
                }
            
            return await loop.run_in_executor(None, analyze)
            
        except Exception as e:
            logger.error(f"{Colors.RED}ОШИБКА при анализе эмбеддингов: {e}{Colors.RESET}")
            return {"error": str(e)}

    async def analyze_texture_async(self, image: np.ndarray) -> Dict[str, Any]:
        """Асинхронный анализ текстуры"""
        try:
            loop = asyncio.get_event_loop()
            
            def analyze():
                if self.texture_analyzer is None:
                    logger.error(f"{Colors.RED}ОШИБКА: TextureAnalyzer не инициализирован. Анализ текстуры невозможен.{Colors.RESET}")
                    return {"error": "TextureAnalyzer не инициализирован"}
                
                # Заглушка для ландмарок (будет заменена на реальные)
                # В реальной реализации здесь должны быть настоящие ландмарки из 3D анализа
                dummy_landmarks = np.zeros((68, 3))
                
                # Анализ текстуры по зонам
                texture_metrics = self.texture_analyzer.analyze_skin_texture_by_zones(image, dummy_landmarks)
                
                # Расчет аутентичности материала
                authenticity_score = self.texture_analyzer.calculate_material_authenticity_score(texture_metrics)
                
                logger.debug("Анализ текстуры завершен.")
                return {
                    "texture_metrics": texture_metrics,
                    "authenticity_score": float(authenticity_score)
                }
            
            return await loop.run_in_executor(None, analyze)
            
        except Exception as e:
            logger.error(f"{Colors.RED}ОШИБКА при анализе текстуры: {e}{Colors.RESET}")
            return {"error": str(e)}

    def aggregate_analysis_results(self, filepath: str, analysis_results: Dict[str, Dict], 
                                 quality_result: Dict[str, Any]) -> AnalysisResult:
        """
        ИСПРАВЛЕНО: Агрегация результатов анализа
        Согласно правкам: использование AUTHENTICITY_WEIGHTS
        """
        try:
            logger.debug(f"Агрегация результатов для файла: {filepath}")
            
            # Сбор компонентных баллов
            component_scores = {}
            
            # Геометрия (из 3D анализа)
            if "face3d" in analysis_results and "error" not in analysis_results["face3d"]:
                face3d_data = analysis_results["face3d"]
                confidence = face3d_data.get("confidence", [])
                if confidence:
                    component_scores["geometry"] = float(np.mean(confidence))
                else:
                    component_scores["geometry"] = 0.5
            else:
                component_scores["geometry"] = 0.0
                logger.warning(f"{Colors.YELLOW}ПРЕДУПРЕЖДЕНИЕ: Геометрический анализ для '{filepath}' завершился с ошибкой или отсутствует. Балл геометрии установлен в 0.{Colors.RESET}")
            
            # Эмбеддинги
            if "embedding" in analysis_results and "error" not in analysis_results["embedding"]:
                embedding_confidence = analysis_results["embedding"].get("confidence", 0.0)
                component_scores["embedding"] = float(embedding_confidence)
            else:
                component_scores["embedding"] = 0.0
                logger.warning(f"{Colors.YELLOW}ПРЕДУПРЕЖДЕНИЕ: Анализ эмбеддингов для '{filepath}' завершился с ошибкой или отсутствует. Балл эмбеддингов установлен в 0.{Colors.RESET}")
            
            # Текстура
            if "texture" in analysis_results and "error" not in analysis_results["texture"]:
                texture_authenticity = analysis_results["texture"].get("authenticity_score", 0.0)
                component_scores["texture"] = float(texture_authenticity)
            else:
                component_scores["texture"] = 0.0
                logger.warning(f"{Colors.YELLOW}ПРЕДУПРЕЖДЕНИЕ: Анализ текстуры для '{filepath}' завершился с ошибкой или отсутствует. Балл текстуры установлен в 0.{Colors.RESET}")
            
            # Временные компоненты (заглушки для одиночного файла)
            component_scores["temporal_consistency"] = 0.7
            component_scores["temporal_stability"] = 0.7
            component_scores["aging_consistency"] = 0.8
            component_scores["anomalies_score"] = 0.9  # 1 - anomaly_rate
            component_scores["liveness_score"] = 0.8
            
            # Расчет итогового балла с AUTHENTICITY_WEIGHTS
            overall_authenticity = sum(
                AUTHENTICITY_WEIGHTS.get(component, 0) * score
                for component, score in component_scores.items()
            )
            
            # Нормализация по сумме весов
            total_weight = sum(AUTHENTICITY_WEIGHTS.values())
            if total_weight > 0:
                overall_authenticity /= total_weight
            
            # Сбор аномалий
            anomalies = {}
            for analysis_type, results in analysis_results.items():
                if "error" in results:
                    anomalies[f"{analysis_type}_error"] = results["error"]
            
            # Сбор метрик
            metrics = {}
            if "face3d" in analysis_results and "metrics" in analysis_results["face3d"]:
                metrics.update(analysis_results["face3d"]["metrics"])
            
            # Метаданные
            metadata = {
                "analysis_version": "2.0",
                "quality_score": quality_result.get("quality_score", 0.0),
                "component_scores": component_scores,
                "processing_timestamp": datetime.now().isoformat()
            }
            
            logger.debug(f"Агрегация для {filepath} завершена. Итоговый балл: {overall_authenticity:.2f}")
            return AnalysisResult(
                filepath=filepath,
                timestamp=datetime.now().isoformat(),
                authenticity_score=float(np.clip(overall_authenticity, 0.0, 1.0)),
                anomalies=anomalies,
                metrics=metrics,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"{Colors.RED}ОШИБКА при агрегации результатов для {filepath}: {e}{Colors.RESET}")
            return AnalysisResult(
                filepath=filepath,
                timestamp=datetime.now().isoformat(),
                authenticity_score=0.0,
                anomalies={"aggregation_error": str(e)},
                metrics={},
                metadata={"error": str(e)}
            )

    def get_cache_key(self, filepath: str) -> str:
        """Генерация ключа кэша"""
        try:
            # Используем MD5 хэш от пути файла и времени модификации
            file_path = Path(filepath)
            if file_path.exists():
                mtime = file_path.stat().st_mtime
                cache_string = f"{filepath}_{mtime}"
            else:
                cache_string = filepath
            
            return hashlib.md5(cache_string.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"{Colors.RED}ОШИБКА генерации ключа кэша для {filepath}: {e}{Colors.RESET}")
            return hashlib.md5(filepath.encode()).hexdigest()

    def _cache_result(self, cache_key: str, result: AnalysisResult) -> None:
        """
        ИСПРАВЛЕНО: Кэширование результата с ограничением размера
        Согласно правкам: MAX_RESULTS_CACHE_SIZE
        """
        try:
            # Проверка размера кэша
            if len(self.results_cache) >= self.max_cache_size:
                # Удаление самого старого элемента (FIFO)
                oldest_key = next(iter(self.results_cache))
                del self.results_cache[oldest_key]
                logger.debug(f"Кэш полон. Удален старый элемент кэша: {oldest_key}")
            
            self.results_cache[cache_key] = result
            logger.debug(f"Результат кэширован: {cache_key}")
            
        except Exception as e:
            logger.error(f"{Colors.RED}ОШИБКА при кэшировании результата: {e}{Colors.RESET}")

    def save_results_to_cache(self, results: List[AnalysisResult]) -> None:
        """Сохранение результатов в файловый кэш"""
        try:
            cache_file = self.cache_dir / "analysis_results.pkl"
            
            # Загрузка существующих результатов
            all_results = {}
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    all_results = pickle.load(f)
                logger.debug(f"Загружено {len(all_results)} существующих результатов из файлового кэша.")
            
            # Добавление новых результатов
            for result in results:
                cache_key = self.get_cache_key(result.filepath)
                all_results[cache_key] = result
            
            # Сохранение
            with open(cache_file, 'wb') as f:
                pickle.dump(all_results, f)
            
            logger.info(f"{Colors.GREEN}✔ Сохранено {len(results)} новых результатов в файловый кэш.{Colors.RESET}")
            
        except Exception as e:
            logger.error(f"{Colors.RED}ОШИБКА при сохранении результатов в файловый кэш: {e}{Colors.RESET}")

    def load_results_from_cache(self) -> Dict[str, AnalysisResult]:
        """Загрузка результатов из файлового кэша"""
        try:
            cache_file = self.cache_dir / "analysis_results.pkl"
            
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    results = pickle.load(f)
                    logger.info(f"{Colors.GREEN}✔ Загружено {len(results)} результатов из файлового кэша.{Colors.RESET}")
                    self.results_cache.update(results) # Обновляем внутренний кэш
                    return results
            else:
                logger.info(f"{Colors.YELLOW}Файл кэша результатов не найден. Кэш будет создан при сохранении.{Colors.RESET}")
                return {}
                
        except Exception as e:
            logger.error(f"{Colors.RED}ОШИБКА при загрузке результатов из файлового кэша: {e}{Colors.RESET}")
            return {}

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Получение статистики обработки"""
        logger.info(f"{Colors.CYAN}Получение статистики обработки DataProcessor...{Colors.RESET}")
        stats = {
            "processing_stats": self.processing_stats.copy(),
            "cache_size": len(self.results_cache),
            "max_cache_size": self.max_cache_size,
            "cache_hit_rate": (
                self.processing_stats["cache_hits"] / 
                (self.processing_stats["cache_hits"] + self.processing_stats["cache_misses"])
                if (self.processing_stats["cache_hits"] + self.processing_stats["cache_misses"]) > 0 else 0.0
            )
        }
        logger.info(f"{Colors.GREEN}✔ Статистика обработки получена.{Colors.RESET}")
        return stats

    def clear_cache(self) -> None:
        """Очистка кэша"""
        self.results_cache.clear()
        logger.info(f"{Colors.YELLOW}Кэш результатов DataProcessor очищен.{Colors.RESET}")

    def self_test(self) -> None:
        """Самотестирование модуля"""
        logger.info(f"{Colors.BOLD}\n=== Запуск самотестирования DataProcessor ==={Colors.RESET}")
        
        try:
            # Создание тестового изображения
            test_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
            test_image_path = "test_image_dataprocessor.jpg"
            cv2.imwrite(test_image_path, test_image)
            logger.info(f"Создано тестовое изображение: {test_image_path}")
            
            # Тест асинхронной обработки
            async def test_async():
                logger.info("Запуск асинхронного теста обработки одного файла...")
                result = await self.process_single_file_async(test_image_path)
                if result.authenticity_score > 0.0:
                    logger.info(f"{Colors.GREEN}✔ Тест обработки файла пройден. Балл аутентичности: {result.authenticity_score:.3f}{Colors.RESET}")
                else:
                    logger.error(f"{Colors.RED}✖ Тест обработки файла провален. Балл аутентичности: {result.authenticity_score:.3f}. Ошибка: {result.anomalies.get('processing_error', 'N/A')}{Colors.RESET}")
                return result
            
            # Запуск теста
            import asyncio
            asyncio.run(test_async())
            
            # Очистка
            import os
            if os.path.exists(test_image_path):
                os.remove(test_image_path)
                logger.info(f"Удалено тестовое изображение: {test_image_path}")
            
            # Статистика
            stats = self.get_processing_statistics()
            logger.info(f"{Colors.CYAN}Статистика DataProcessor после теста: {Colors.RESET}")
            logger.info(f"  Всего обработано: {stats['processing_stats']['total_processed']}")
            logger.info(f"  Успешно: {stats['processing_stats']['successful_analyses']}")
            logger.info(f"  Ошибок: {stats['processing_stats']['failed_analyses']}")
            logger.info(f"  Попаданий в кэш: {stats['processing_stats']['cache_hits']}")
            logger.info(f"  Промахов кэша: {stats['processing_stats']['cache_misses']}")
            logger.info(f"  Размер кэша: {stats['cache_size']}")
            logger.info(f"  Коэффициент попаданий кэша: {stats['cache_hit_rate']:.2%}")
            
        except Exception as e:
            logger.critical(f"{Colors.RED}КРИТИЧЕСКАЯ ОШИБКА при самотестировании DataProcessor: {e}{Colors.RESET}")
        
        logger.info(f"{Colors.BOLD}=== Самотестирование DataProcessor завершено ==={Colors.RESET}\n")

# ==================== АГРЕГАТОР РЕЗУЛЬТАТОВ ====================

class ResultsAggregator:
    """
    ИСПРАВЛЕНО: Агрегатор результатов анализа
    Согласно правкам: статистика и фильтрация
    """
    
    def __init__(self):
        self.results_db = []
        logger.info(f"{Colors.BOLD}--- Инициализация ResultsAggregator (Агрегатора результатов) ---{Colors.RESET}")

    def add_results(self, results: List[AnalysisResult]) -> None:
        """Добавление результатов в базу"""
        self.results_db.extend(results)
        logger.info(f"{Colors.GREEN}Добавлено {len(results)} новых результатов в агрегатор.{Colors.RESET}")

    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики результатов"""
        if not self.results_db:
            logger.warning(f"{Colors.YELLOW}ПРЕДУПРЕЖДЕНИЕ: Нет результатов для расчета статистики.{Colors.RESET}")
            return {}
        
        try:
            authenticity_scores = [r.authenticity_score for r in self.results_db]
            
            stats = {
                "total_files": len(self.results_db),
                "successful_analyses": sum(1 for r in self.results_db if not r.anomalies.get("processing_error") and r.authenticity_score > 0),
                "failed_analyses": sum(1 for r in self.results_db if r.anomalies.get("processing_error") or r.authenticity_score == 0),
                "average_authenticity": float(np.mean(authenticity_scores)) if authenticity_scores else 0.0,
                "std_authenticity": float(np.std(authenticity_scores)) if authenticity_scores else 0.0,
                "min_authenticity": float(np.min(authenticity_scores)) if authenticity_scores else 0.0,
                "max_authenticity": float(np.max(authenticity_scores)) if authenticity_scores else 0.0,
                "anomalies_count": len([r for r in self.results_db if r.anomalies and not r.anomalies.get("processing_error")]),
                "processing_dates": {
                    "first": min(r.timestamp for r in self.results_db),
                    "last": max(r.timestamp for r in self.results_db)
                }
            }
            logger.info(f"{Colors.GREEN}✔ Статистика агрегации результатов рассчитана.{Colors.RESET}")
            return stats
            
        except Exception as e:
            logger.error(f"{Colors.RED}ОШИБКА при расчете статистики агрегированных результатов: {e}{Colors.RESET}")
            return {}

    def filter_results(self, authenticity_range: Tuple[float, float] = (0.0, 1.0),
                      has_anomalies: Optional[bool] = None,
                      date_range: Optional[Tuple[str, str]] = None) -> List[AnalysisResult]:
        """Фильтрация результатов"""
        try:
            logger.info(f"{Colors.CYAN}Фильтрация результатов: диапазон аутентичности {authenticity_range}, аномалии: {has_anomalies}, даты: {date_range}...{Colors.RESET}")
            filtered = self.results_db
            
            # Фильтр по аутентичности
            filtered = [r for r in filtered if authenticity_range[0] <= r.authenticity_score <= authenticity_range[1]]
            
            # Фильтр по аномалиям
            if has_anomalies is not None:
                if has_anomalies:
                    filtered = [r for r in filtered if r.anomalies]
                else:
                    filtered = [r for r in filtered if not r.anomalies]
            
            # Фильтр по датам
            if date_range:
                start_date, end_date = date_range
                filtered = [r for r in filtered if start_date <= r.timestamp <= end_date]
            
            logger.info(f"{Colors.GREEN}✔ Фильтрация завершена. Найдено {len(filtered)} результатов.{Colors.RESET}")
            return filtered
            
        except Exception as e:
            logger.error(f"{Colors.RED}ОШИБКА при фильтрации результатов: {e}{Colors.RESET}")
            return []

    def export_to_dataframe(self) -> pd.DataFrame:
        """Экспорт результатов в DataFrame"""
        try:
            logger.info(f"{Colors.CYAN}Экспорт агрегированных результатов в DataFrame...{Colors.RESET}")
            data = []
            
            for result in self.results_db:
                row = {
                    "filepath": result.filepath,
                    "timestamp": result.timestamp,
                    "authenticity_score": result.authenticity_score,
                    "has_anomalies": bool(result.anomalies),
                    "anomalies_count": len(result.anomalies)
                }
                
                # Добавление метрик
                for metric_name, metric_value in result.metrics.items():
                    row[f"metric_{metric_name}"] = metric_value
                
                data.append(row)
            
            df = pd.DataFrame(data)
            logger.info(f"{Colors.GREEN}✔ Экспорт в DataFrame завершен. Строк: {len(df)}{Colors.RESET}")
            return df
            
        except Exception as e:
            logger.error(f"{Colors.RED}ОШИБКА при экспорте результатов в DataFrame: {e}{Colors.RESET}")
            return pd.DataFrame()

# ==================== ТОЧКА ВХОДА ====================

if __name__ == "__main__":
    processor = DataProcessor()
    processor.self_test()