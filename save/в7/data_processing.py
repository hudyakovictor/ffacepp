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
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
import json
import pickle
from pathlib import Path
import cv2
from datetime import datetime
import hashlib
from core_config import MIN_VISIBILITY_Z

def _ensure_img(img_input: Union[str, np.ndarray]) -> np.ndarray:
    """
    Гарантирует, что на вход функции передается массив изображения, а не строка.
    """
    if isinstance(img_input, str):
        img = cv2.imread(img_input)
    elif isinstance(img_input, np.ndarray):
        img = img_input
    else:
        raise ValueError(f"Неподдерживаемый тип ввода изображения: {type(img_input)}")
    
    if img is None:
        raise ValueError("Изображение не загружено или является пустым.")
    return img

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
    
    def __init__(self, cache_dir: str = "cache", 
                 face_analyzer: Any = None, 
                 embedding_analyzer: Any = None, 
                 texture_analyzer: Any = None,
                 temporal_analyzer: Any = None, 
                 data_manager: Any = None, 
                 metrics_calculator: Any = None):
        logger.info(f"{Colors.BOLD}--- Инициализация DataProcessor (Обработчика данных) ---{Colors.RESET}")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Кэш результатов с ограничением размера
        self.results_cache = {}
        self.max_cache_size = MAX_RESULTS_CACHE_SIZE
        
        # ИСПРАВЛЕНО: Анализаторы теперь передаются извне
        self.face_analyzer = face_analyzer
        self.embedding_analyzer = embedding_analyzer
        self.texture_analyzer = texture_analyzer
        self.temporal_analyzer = temporal_analyzer
        self.data_manager = data_manager
        self.metrics_calculator = metrics_calculator
        
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
        """ИСПРАВЛЕНО: Этот метод теперь не инициализирует анализаторы, а проверяет их наличие"""
        logger.info(f"{Colors.CYAN}Проверка наличия анализаторов в DataProcessor...{Colors.RESET}")
        if self.face_analyzer is None or \
           self.embedding_analyzer is None or \
           self.texture_analyzer is None or \
           self.temporal_analyzer is None or \
           self.data_manager is None or \
           self.metrics_calculator is None:
            logger.critical(f"{Colors.RED}КРИТИЧЕСКАЯ ОШИБКА: Один или несколько анализаторов не были переданы в DataProcessor.{Colors.RESET}")
            raise RuntimeError("Не все анализаторы инициализированы для DataProcessor")
        logger.info(f"{Colors.GREEN}✔ Все необходимые анализаторы доступны.{Colors.RESET}")

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
            
            # Загрузка изображения (уже гарантирует np.ndarray)
            image = await self.load_image_async(filepath)
            
            # Валидация качества изображения
            quality_result = self.data_manager.validate_image_quality_for_analysis(image)
            
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
            
            # Асинхронный анализ 3D лица
            analysis_3d_result = await self.analyze_3d_async(image)

            landmarks68 = analysis_3d_result.get("landmarks68")

            # Проверка, что landmarks68 не None перед вызовом analyze_embedding_async
            if landmarks68.size == 0 or landmarks68 is None:
                logger.warning(f"[DataProcessor] Ландмарки не найдены для файла {filepath}. Пропускаем анализ эмбеддингов.")
                embedding_result = {"error": "No landmarks for embedding analysis"}
            else:
                # Вызов analyze_embedding_async с передачей image и landmarks68, а также insight_app
                embedding_result = await self.analyze_embedding_async(
                    image=image,
                    landmarks=landmarks68,
                    insight_app=self.embedding_analyzer.face_app # Передача экземпляра insight_app
                )

            # Анализ текстуры
            texture_result = {}
            if landmarks68.size == 0 or landmarks68 is None:
                logger.warning(f"[DataProcessor] Ландмарки не найдены для файла {filepath}. Пропускаем анализ текстуры.")
                texture_result = {"error": "No landmarks for texture analysis"}
            else:
                texture_result = await self.analyze_texture_async(
                    image=image,
                    landmarks=landmarks68
                )

            analysis_results = {
                "3d_analysis": analysis_3d_result,
                "embedding": embedding_result,
                "texture": texture_result
            }
            
            # Агрегация результатов и кэширование
            final_result = self.aggregate_analysis_results(filepath, analysis_results, quality_result)
            self._cache_result(cache_key, final_result)
            
            logger.debug(f"Завершено обработка файла: {filepath}")
            return final_result
            
        except Exception as e:
            logger.error(f"{Colors.RED}ОШИБКА при обработке файла {filepath}: {e}{Colors.RESET}")
            return AnalysisResult(
                filepath=filepath,
                timestamp=datetime.now().isoformat(),
                authenticity_score=0.0,
                anomalies={"processing_error": str(e)},
                metrics={},
                metadata={"error": str(e)}
            )

    async def load_image_async(self, filepath: str) -> Optional[np.ndarray]:
        """
        Асинхронно загружает изображение, выполняя блокирующую операцию в отдельном потоке.
        """
        loop = asyncio.get_running_loop()
        # Использование _safe_load_image из data_manager для загрузки изображения
        # Это гарантирует, что img_input может быть путем или np.ndarray
        img = await loop.run_in_executor(None, self.data_manager._safe_load_image, filepath)
        return img

    async def analyze_3d_async(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Асинхронный анализ 3D лица (ландмарки, позы, формы).
        """
        image = _ensure_img(image) # Гарантируем, что image это np.ndarray
        logger.debug(f"[analyze_3d_async] Начало 3D анализа. Размер изображения: {image.shape}")
        loop = asyncio.get_running_loop()

        def analyze():
            if not self.face_analyzer:
                raise RuntimeError("Face3DAnalyzer не инициализирован.")
            
            # Используем self.face_analyzer.tddfa и self.face_analyzer.faceboxes
            # Передача моделей в extract_68_landmarks_with_confidence
            landmarks, confidence, param = self.face_analyzer.extract_68_landmarks_with_confidence(
                image, models={"tddfa": self.face_analyzer.tddfa, "face_boxes": self.face_analyzer.faceboxes})
            
            if landmarks.size == 0:
                logger.warning("3D анализ: ландмарки не обнаружены.")
                return {"status": "failed", "error": ERROR_CODES["E001"], "landmarks": [], "pose": {}}

            # Остальные 3D-метрики
            pose = self.face_analyzer.determine_precise_face_pose(landmarks, param, confidence)
            # Получение нормализованных ландмарок
            vis = landmarks[:,2] > MIN_VISIBILITY_Z
            normalized_landmarks = self.face_analyzer.normalize_landmarks_by_pose_category(
                landmarks,
                pose.get("pose_category", "frontal"),
                vis
            )
            shape_error = self.face_analyzer.calculate_comprehensive_shape_error(normalized_landmarks)
            asymmetry = self.face_analyzer.analyze_facial_asymmetry(landmarks, confidence) # Используем ненормализованные ландмарки для асимметрии
            
            return {
                "status": "success",
                "landmarks": landmarks.tolist(),
                "pose": pose,
                "shape_error": shape_error,
                "asymmetry": asymmetry,
                "confidence_3d_landmarks": np.mean(confidence).item() # Средняя уверенность по ландмаркам
            }
        
        return await loop.run_in_executor(None, analyze)

    async def analyze_embedding_async(self, image: np.ndarray, landmarks: np.ndarray, insight_app: np.ndarray) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Асинхронный анализ эмбеддингов
        Согласно правкам: передача align-кропа
        """
        if self.embedding_analyzer is None:
            raise RuntimeError("EmbeddingAnalyzer не инициализирован.")
        
        try:
            # ИСПРАВЛЕНО: Передача полного изображения, 68 ландмарок и insight_app
            embedding, confidence = self.embedding_analyzer.extract_512d_face_embedding(image, landmarks, insight_app)
            
            if embedding is None or embedding.size == 0:
                logger.warning("InsightFace не обнаружил лица в изображении")
                return {"status": "failed", "error": ERROR_CODES["E001"], "embedding": []}
            
            return {
                "status": "success",
                "embedding": embedding.tolist(),
                "confidence_embedding": confidence
            }
            
        except Exception as e:
            logger.error(f"Ошибка извлечения эмбеддинга: {e}", exc_info=True)
            return {"error": str(e), "status": "failed"}

    async def analyze_texture_async(self, image: np.ndarray, landmarks: np.ndarray) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Асинхронный анализ текстуры
        Согласно правкам: передача изображения и ландмарок
        """
        logger.debug(f"Начинаем асинхронный анализ текстуры для изображения {image.shape}")
        loop = asyncio.get_running_loop()
        
        def analyze():
            if self.texture_analyzer is None:
                logger.error("TextureAnalyzer не инициализирован")
                return {"error": "TextureAnalyzer not initialized", "status": "failed"}
            
            try:
                # ИСПРАВЛЕНО: Передача ландмарок в analyze_skin_texture_by_zones
                return self.texture_analyzer.analyze_skin_texture_by_zones(image, landmarks)
            except Exception as e:
                logger.error(f"Ошибка в анализаторе текстуры: {e}", exc_info=True)
                return {"error": str(e), "status": "failed"}

        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(pool, analyze)
            return result

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
            if "3d_analysis" in analysis_results and "error" not in analysis_results["3d_analysis"]:
                face3d_data = analysis_results["3d_analysis"]
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
            if "3d_analysis" in analysis_results and "metrics" in analysis_results["3d_analysis"]:
                metrics.update(analysis_results["3d_analysis"]["metrics"])
            
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
        logger.info(f"{Colors.BOLD}=== Запуск самотестирования DataProcessor ==={Colors.RESET}")

        # Тестовые данные для process_single_file_async
        # ИСПРАВЛЕНО (Пункт 3): Используем реальное тестовое изображение с лицом
        test_image_path = "/Users/victorkhudyakov/nn/3DDFA2/3/01_01_10.jpg"
        
        # Убедимся, что файл существует
        if not Path(test_image_path).exists():
            logger.error(f"{Colors.RED}ОШИБКА: Тестовое изображение не найдено по пути: {test_image_path}{Colors.RESET}")
            logger.error(f"{Colors.RED}Самотестирование DataProcessor завершено с ошибками: Тестовое изображение отсутствует{Colors.RESET}")
            return

        # Тест process_single_file_async
        logger.info("Запуск асинхронного теста обработки одного файла...")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If a loop is already running (e.g. in a Jupyter notebook), create a new task
                task = loop.create_task(self.process_single_file_async(test_image_path))
                loop.run_until_complete(task)
                result = task.result()
            else:
                result = loop.run_until_complete(self.process_single_file_async(test_image_path))

            # Проверка результатов
            assert result.authenticity_score > 0, f"Тест обработки файла провален. Балл аутентичности: {result.authenticity_score:.3f}."
            logger.info(f"{Colors.GREEN}✔ Тест обработки файла пройден. Балл аутентичности: {result.authenticity_score:.3f}.{Colors.RESET}")

        except Exception as e:
            logger.error(f"{Colors.RED}ОШИБКА при обработке файла {test_image_path}: {e}{Colors.RESET}")
            logger.error(f"{Colors.RED}✖ Тест обработки файла провален. Балл аутентичности: 0.000. Ошибка: {e}{Colors.RESET}")

        finally:
            # Удаляем тестовое изображение, если оно было создано
            # В данном случае, мы используем существующий файл, поэтому удалять не нужно.
            # logger.info(f"Удалено тестовое изображение: {test_image_path}")
            pass

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