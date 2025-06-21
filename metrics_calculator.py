# metrics_calculator.py
import os
import json
import logging
import hashlib
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
import numpy as np
import cv2
from scipy.spatial.distance import euclidean, cosine
from scipy.stats import zscore
import pickle
import time
import psutil
from functools import lru_cache
import threading
from collections import OrderedDict, defaultdict

from core_config import get_config

# Настройка логирования
logger = logging.getLogger(__name__)

# === КОНСТАНТЫ И КОНФИГУРАЦИЯ ===

# Индексы ключевых точек для расчета метрик идентичности
SKULL_LANDMARKS = {
    'forehead_width': [19, 24],  # Ширина лба
    'temple_width': [0, 16],     # Ширина висков
    'cheekbone_width': [1, 15],  # Ширина скул
    'jaw_width': [5, 11],        # Ширина челюсти
    'chin_width': [6, 10]        # Ширина подбородка
}

FACIAL_PROPORTIONS = {
    'eye_distance': [36, 45],           # Межзрачковое расстояние
    'nose_width': [31, 35],             # Ширина носа
    'mouth_width': [48, 54],            # Ширина рта
    'face_height': [27, 8],             # Высота лица
    'nose_height': [27, 33],            # Высота носа
    'upper_face_height': [19, 33],      # Высота верхней части лица
    'lower_face_height': [33, 8]        # Высота нижней части лица
}

# Группировка 15 метрик по категориям
SKULL_METRICS = [
    'forehead_width_ratio',
    'temple_width_ratio', 
    'cheekbone_width_ratio',
    'jaw_width_ratio',
    'chin_width_ratio',
    'skull_length_ratio',
    'cranial_vault_ratio'
]

PROPORTION_METRICS = [
    'eye_distance_ratio',
    'nose_width_ratio',
    'mouth_width_ratio',
    'face_height_ratio',
    'nose_height_ratio',
    'upper_face_ratio',
    'lower_face_ratio',
    'facial_index'
]

# Пороги для валидации метрик
METRIC_RANGES = {
    'forehead_width_ratio': (0.8, 1.2),
    'temple_width_ratio': (0.85, 1.15),
    'cheekbone_width_ratio': (0.9, 1.1),
    'jaw_width_ratio': (0.85, 1.15),
    'chin_width_ratio': (0.8, 1.2),
    'skull_length_ratio': (0.9, 1.1),
    'cranial_vault_ratio': (0.85, 1.15),
    'eye_distance_ratio': (0.9, 1.1),
    'nose_width_ratio': (0.8, 1.2),
    'mouth_width_ratio': (0.85, 1.15),
    'face_height_ratio': (0.9, 1.1),
    'nose_height_ratio': (0.85, 1.15),
    'upper_face_ratio': (0.9, 1.1),
    'lower_face_ratio': (0.9, 1.1),
    'facial_index': (0.8, 1.2)
}

# === СТРУКТУРЫ ДАННЫХ ===

@dataclass
class IdentityMetrics:
    """Структура для хранения 15 метрик идентичности"""
    image_id: str
    filepath: str
    
    # Костные метрики (7 штук)
    forehead_width_ratio: float = 0.0
    temple_width_ratio: float = 0.0
    cheekbone_width_ratio: float = 0.0
    jaw_width_ratio: float = 0.0
    chin_width_ratio: float = 0.0
    skull_length_ratio: float = 0.0
    cranial_vault_ratio: float = 0.0
    
    # Пропорциональные метрики (8 штук)
    eye_distance_ratio: float = 0.0
    nose_width_ratio: float = 0.0
    mouth_width_ratio: float = 0.0
    face_height_ratio: float = 0.0
    nose_height_ratio: float = 0.0
    upper_face_ratio: float = 0.0
    lower_face_ratio: float = 0.0
    facial_index: float = 0.0
    
    # Метаданные
    pose_category: str = "frontal"
    confidence_score: float = 0.0
    normalization_factor: float = 1.0
    baseline_reference: Optional[str] = None
    
    # Флаги качества
    quality_flags: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Статистика обработки
    processing_time_ms: float = 0.0
    calculation_method: str = "standard"
    
    def to_array(self) -> np.ndarray:
        """Конвертация метрик в numpy массив"""
        return np.array([
            self.forehead_width_ratio, self.temple_width_ratio, self.cheekbone_width_ratio,
            self.jaw_width_ratio, self.chin_width_ratio, self.skull_length_ratio,
            self.cranial_vault_ratio, self.eye_distance_ratio, self.nose_width_ratio,
            self.mouth_width_ratio, self.face_height_ratio, self.nose_height_ratio,
            self.upper_face_ratio, self.lower_face_ratio, self.facial_index
        ])
    
    def get_skull_metrics(self) -> Dict[str, float]:
        """Получение только костных метрик"""
        return {
            'forehead_width_ratio': self.forehead_width_ratio,
            'temple_width_ratio': self.temple_width_ratio,
            'cheekbone_width_ratio': self.cheekbone_width_ratio,
            'jaw_width_ratio': self.jaw_width_ratio,
            'chin_width_ratio': self.chin_width_ratio,
            'skull_length_ratio': self.skull_length_ratio,
            'cranial_vault_ratio': self.cranial_vault_ratio
        }
    
    def get_proportion_metrics(self) -> Dict[str, float]:
        """Получение только пропорциональных метрик"""
        return {
            'eye_distance_ratio': self.eye_distance_ratio,
            'nose_width_ratio': self.nose_width_ratio,
            'mouth_width_ratio': self.mouth_width_ratio,
            'face_height_ratio': self.face_height_ratio,
            'nose_height_ratio': self.nose_height_ratio,
            'upper_face_ratio': self.upper_face_ratio,
            'lower_face_ratio': self.lower_face_ratio,
            'facial_index': self.facial_index
        }

@dataclass
class BaselineMetrics:
    """Базовые метрики для нормализации"""
    reference_period: str
    mean_values: Dict[str, float]
    std_values: Dict[str, float]
    sample_count: int
    creation_date: datetime.datetime
    confidence_level: float

@dataclass
class MetricsComparison:
    """Результат сравнения метрик"""
    similarity_score: float
    distance_euclidean: float
    distance_cosine: float
    outlier_metrics: List[str]
    consistency_score: float
    temporal_trend: Optional[str] = None

# === ОСНОВНОЙ КЛАСС КАЛЬКУЛЯТОРА МЕТРИК ===

class MetricsCalculator:
    """Калькулятор для вычисления 15 метрик идентичности лица"""
    
    def __init__(self):
        self.config = get_config()
        self.cache_dir = Path("./cache/metrics")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Кэш результатов
        self.metrics_cache: Dict[str, IdentityMetrics] = {}
        self.baseline_cache: Dict[str, BaselineMetrics] = {}
        
        # Статистика
        self.processing_stats = {
            'total_calculated': 0,
            'successful_calculations': 0,
            'failed_calculations': 0,
            'cache_hits': 0,
            'baseline_normalizations': 0
        }
        
        # Блокировка для потокобезопасности
        self.calculation_lock = threading.Lock()
        
        # Загрузка baseline метрик
        self._load_baseline_metrics()
        
        logger.info("MetricsCalculator инициализирован")

    def _load_baseline_metrics(self):
        """Загрузка базовых метрик для нормализации"""
        try:
            baseline_path = self.cache_dir / "baseline_metrics.pkl"
            if baseline_path.exists():
                with open(baseline_path, 'rb') as f:
                    self.baseline_cache = pickle.load(f)
                logger.info(f"Загружено {len(self.baseline_cache)} baseline метрик")
            else:
                logger.info("Baseline метрики не найдены, будут созданы при первом использовании")
                
        except Exception as e:
            logger.error(f"Ошибка загрузки baseline метрик: {e}")
            self.baseline_cache = {}

    def calculate_identity_signature_metrics(self, landmarks_package) -> Optional[IdentityMetrics]:
        """
        Основная функция расчета 15 метрик идентичности
        
        Args:
            landmarks_package: Пакет с нормализованными ландмарками
            
        Returns:
            Объект с метриками идентичности или None при ошибке
        """
        try:
            start_time = time.time()
            
            # Проверка входных данных
            if landmarks_package is None or landmarks_package.normalized_landmarks is None:
                logger.error("Отсутствуют нормализованные ландмарки")
                self.processing_stats['failed_calculations'] += 1
                return None
            
            # Проверка кэша
            cache_key = landmarks_package.image_id
            if cache_key in self.metrics_cache:
                self.processing_stats['cache_hits'] += 1
                cached_result = self.metrics_cache[cache_key]
                cached_result.processing_time_ms = (time.time() - start_time) * 1000
                return cached_result
            
            landmarks = landmarks_package.normalized_landmarks
            pose_category = landmarks_package.pose_category
            
            # Проверка количества ландмарков
            if len(landmarks) < 68:
                logger.error(f"Недостаточно ландмарков: {len(landmarks)}")
                self.processing_stats['failed_calculations'] += 1
                return None
            
            # Создание объекта метрик
            metrics = IdentityMetrics(
                image_id=landmarks_package.image_id,
                filepath=landmarks_package.filepath,
                pose_category=pose_category
            )
            
            # Расчет костных метрик (7 штук)
            self._calculate_skull_metrics(landmarks, metrics)
            
            # Расчет пропорциональных метрик (8 штук)
            self._calculate_proportion_metrics(landmarks, metrics)
            
            # Нормализация по baseline
            self._normalize_by_baseline(metrics, pose_category)
            
            # Валидация результатов
            self._validate_metrics(metrics)
            
            # Расчет общего балла достоверности
            metrics.confidence_score = self._calculate_confidence_score(metrics)
            
            # Метаданные обработки
            metrics.processing_time_ms = (time.time() - start_time) * 1000
            metrics.calculation_method = "normalized_landmarks"
            
            # Сохранение в кэш
            self.metrics_cache[cache_key] = metrics
            
            self.processing_stats['successful_calculations'] += 1
            self.processing_stats['total_calculated'] += 1
            
            logger.debug(f"Метрики рассчитаны за {metrics.processing_time_ms:.1f}мс")
            return metrics
            
        except Exception as e:
            logger.error(f"Ошибка расчета метрик: {e}")
            self.processing_stats['failed_calculations'] += 1
            self.processing_stats['total_calculated'] += 1
            return None

    def _calculate_skull_metrics(self, landmarks: np.ndarray, metrics: IdentityMetrics):
        """Расчет 7 костных метрик"""
        try:
            # 1. Ширина лба (forehead_width_ratio)
            forehead_points = [landmarks[i] for i in SKULL_LANDMARKS['forehead_width']]
            forehead_width = self._calculate_distance_2d(forehead_points[0], forehead_points[1])
            
            # 2. Ширина висков (temple_width_ratio)
            temple_points = [landmarks[i] for i in SKULL_LANDMARKS['temple_width']]
            temple_width = self._calculate_distance_2d(temple_points[0], temple_points[1])
            
            # 3. Ширина скул (cheekbone_width_ratio)
            cheekbone_points = [landmarks[i] for i in SKULL_LANDMARKS['cheekbone_width']]
            cheekbone_width = self._calculate_distance_2d(cheekbone_points[0], cheekbone_points[1])
            
            # 4. Ширина челюсти (jaw_width_ratio)
            jaw_points = [landmarks[i] for i in SKULL_LANDMARKS['jaw_width']]
            jaw_width = self._calculate_distance_2d(jaw_points[0], jaw_points[1])
            
            # 5. Ширина подбородка (chin_width_ratio)
            chin_points = [landmarks[i] for i in SKULL_LANDMARKS['chin_width']]
            chin_width = self._calculate_distance_2d(chin_points[0], chin_points[1])
            
            # 6. Длина черепа (skull_length_ratio)
            skull_length = self._calculate_distance_2d(landmarks[27], landmarks[8])  # Переносица - подбородок
            
            # 7. Свод черепа (cranial_vault_ratio)
            cranial_vault = self._calculate_cranial_vault_ratio(landmarks)
            
            # Нормализация относительно межзрачкового расстояния
            iod = self._calculate_distance_2d(landmarks[36], landmarks[45])
            if iod > 0:
                metrics.forehead_width_ratio = forehead_width / iod
                metrics.temple_width_ratio = temple_width / iod
                metrics.cheekbone_width_ratio = cheekbone_width / iod
                metrics.jaw_width_ratio = jaw_width / iod
                metrics.chin_width_ratio = chin_width / iod
                metrics.skull_length_ratio = skull_length / iod
                metrics.cranial_vault_ratio = cranial_vault
                metrics.normalization_factor = iod
            else:
                logger.warning("Нулевое межзрачковое расстояние")
                
        except Exception as e:
            logger.error(f"Ошибка расчета костных метрик: {e}")
            metrics.warnings.append(f"Ошибка костных метрик: {str(e)}")

    def _calculate_proportion_metrics(self, landmarks: np.ndarray, metrics: IdentityMetrics):
        """Расчет 8 пропорциональных метрик"""
        try:
            # Базовые расстояния
            iod = self._calculate_distance_2d(landmarks[36], landmarks[45])  # Межзрачковое расстояние
            
            if iod <= 0:
                logger.warning("Нулевое межзрачковое расстояние для пропорций")
                return
            
            # 1. Межзрачковое расстояние (eye_distance_ratio) - нормализовано к 1.0
            metrics.eye_distance_ratio = 1.0
            
            # 2. Ширина носа (nose_width_ratio)
            nose_points = [landmarks[i] for i in FACIAL_PROPORTIONS['nose_width']]
            nose_width = self._calculate_distance_2d(nose_points[0], nose_points[1])
            metrics.nose_width_ratio = nose_width / iod
            
            # 3. Ширина рта (mouth_width_ratio)
            mouth_points = [landmarks[i] for i in FACIAL_PROPORTIONS['mouth_width']]
            mouth_width = self._calculate_distance_2d(mouth_points[0], mouth_points[1])
            metrics.mouth_width_ratio = mouth_width / iod
            
            # 4. Высота лица (face_height_ratio)
            face_points = [landmarks[i] for i in FACIAL_PROPORTIONS['face_height']]
            face_height = self._calculate_distance_2d(face_points[0], face_points[1])
            metrics.face_height_ratio = face_height / iod
            
            # 5. Высота носа (nose_height_ratio)
            nose_height_points = [landmarks[i] for i in FACIAL_PROPORTIONS['nose_height']]
            nose_height = self._calculate_distance_2d(nose_height_points[0], nose_height_points[1])
            metrics.nose_height_ratio = nose_height / iod
            
            # 6. Высота верхней части лица (upper_face_ratio)
            upper_face_points = [landmarks[i] for i in FACIAL_PROPORTIONS['upper_face_height']]
            upper_face_height = self._calculate_distance_2d(upper_face_points[0], upper_face_points[1])
            metrics.upper_face_ratio = upper_face_height / iod
            
            # 7. Высота нижней части лица (lower_face_ratio)
            lower_face_points = [landmarks[i] for i in FACIAL_PROPORTIONS['lower_face_height']]
            lower_face_height = self._calculate_distance_2d(lower_face_points[0], lower_face_points[1])
            metrics.lower_face_ratio = lower_face_height / iod
            
            # 8. Лицевой индекс (facial_index)
            metrics.facial_index = face_height / cheekbone_width if hasattr(metrics, 'cheekbone_width_ratio') and metrics.cheekbone_width_ratio > 0 else face_height / iod
            
        except Exception as e:
            logger.error(f"Ошибка расчета пропорциональных метрик: {e}")
            metrics.warnings.append(f"Ошибка пропорциональных метрик: {str(e)}")

    def _calculate_distance_2d(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Расчет 2D расстояния между точками"""
        try:
            return float(np.linalg.norm(point1[:2] - point2[:2]))
        except Exception as e:
            logger.error(f"Ошибка расчета расстояния: {e}")
            return 0.0

    def _calculate_cranial_vault_ratio(self, landmarks: np.ndarray) -> float:
        """Расчет соотношения свода черепа"""
        try:
            # Используем точки бровей для оценки высоты свода
            left_brow = np.mean(landmarks[22:27, :2], axis=0)
            right_brow = np.mean(landmarks[17:22, :2], axis=0)
            brow_center = (left_brow + right_brow) / 2
            
            # Точка подбородка
            chin = landmarks[8, :2]
            
            # Высота от бровей до подбородка
            vault_height = np.linalg.norm(brow_center - chin)
            
            # Ширина лица на уровне скул
            cheek_width = np.linalg.norm(landmarks[1, :2] - landmarks[15, :2])
            
            if cheek_width > 0:
                return float(vault_height / cheek_width)
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"Ошибка расчета свода черепа: {e}")
            return 1.0

    def _normalize_by_baseline(self, metrics: IdentityMetrics, pose_category: str):
        """Нормализация метрик по baseline"""
        try:
            baseline_key = f"baseline_{pose_category}"
            
            if baseline_key in self.baseline_cache:
                baseline = self.baseline_cache[baseline_key]
                
                # Нормализация каждой метрики
                for metric_name in SKULL_METRICS + PROPORTION_METRICS:
                    current_value = getattr(metrics, metric_name, 0.0)
                    baseline_mean = baseline.mean_values.get(metric_name, current_value)
                    baseline_std = baseline.std_values.get(metric_name, 1.0)
                    
                    if baseline_std > 0:
                        # Z-score нормализация
                        normalized_value = (current_value - baseline_mean) / baseline_std
                        # Конвертация обратно в ratio (добавляем 1.0 для центрирования)
                        setattr(metrics, metric_name, 1.0 + normalized_value * 0.1)
                    
                metrics.baseline_reference = baseline_key
                self.processing_stats['baseline_normalizations'] += 1
                
        except Exception as e:
            logger.error(f"Ошибка нормализации по baseline: {e}")
            metrics.warnings.append(f"Ошибка нормализации: {str(e)}")

    def _validate_metrics(self, metrics: IdentityMetrics):
        """Валидация рассчитанных метрик"""
        try:
            for metric_name in SKULL_METRICS + PROPORTION_METRICS:
                value = getattr(metrics, metric_name, 0.0)
                min_val, max_val = METRIC_RANGES.get(metric_name, (0.0, 2.0))
                
                if not (min_val <= value <= max_val):
                    metrics.quality_flags.append(f"out_of_range_{metric_name}")
                    metrics.warnings.append(f"Метрика {metric_name} вне диапазона: {value:.3f}")
                
                if np.isnan(value) or np.isinf(value):
                    metrics.quality_flags.append(f"invalid_{metric_name}")
                    metrics.warnings.append(f"Недопустимое значение {metric_name}: {value}")
                    setattr(metrics, metric_name, 1.0)  # Устанавливаем нейтральное значение
                    
        except Exception as e:
            logger.error(f"Ошибка валидации метрик: {e}")
            metrics.warnings.append(f"Ошибка валидации: {str(e)}")

    def _calculate_confidence_score(self, metrics: IdentityMetrics) -> float:
        """Расчет общего балла достоверности метрик"""
        try:
            # Количество валидных метрик
            total_metrics = len(SKULL_METRICS + PROPORTION_METRICS)
            invalid_metrics = len([flag for flag in metrics.quality_flags if 'invalid_' in flag])
            out_of_range_metrics = len([flag for flag in metrics.quality_flags if 'out_of_range_' in flag])
            
            # Базовый балл
            base_score = (total_metrics - invalid_metrics) / total_metrics
            
            # Штраф за выход из диапазона
            range_penalty = out_of_range_metrics / total_metrics * 0.5
            
            # Итоговый балл
            confidence = max(0.0, base_score - range_penalty)
            
            return float(confidence)
            
        except Exception as e:
            logger.error(f"Ошибка расчета достоверности: {e}")
            return 0.5

    def compare_metrics(self, metrics1: IdentityMetrics, metrics2: IdentityMetrics) -> MetricsComparison:
        """
        Сравнение двух наборов метрик
        
        Args:
            metrics1: Первый набор метрик
            metrics2: Второй набор метрик
            
        Returns:
            Результат сравнения
        """
        try:
            # Конвертация в массивы
            array1 = metrics1.to_array()
            array2 = metrics2.to_array()
            
            # Расчет расстояний
            euclidean_dist = float(euclidean(array1, array2))
            cosine_dist = float(cosine(array1, array2))
            
            # Поиск выбросов
            diff = np.abs(array1 - array2)
            outlier_threshold = np.mean(diff) + 2 * np.std(diff)
            outlier_indices = np.where(diff > outlier_threshold)[0]
            
            metric_names = SKULL_METRICS + PROPORTION_METRICS
            outlier_metrics = [metric_names[i] for i in outlier_indices if i < len(metric_names)]
            
            # Расчет схожести (обратная величина расстояния)
            similarity = 1.0 / (1.0 + euclidean_dist)
            
            # Расчет консистентности
            consistency = 1.0 - min(1.0, len(outlier_metrics) / len(metric_names))
            
            return MetricsComparison(
                similarity_score=similarity,
                distance_euclidean=euclidean_dist,
                distance_cosine=cosine_dist,
                outlier_metrics=outlier_metrics,
                consistency_score=consistency
            )
            
        except Exception as e:
            logger.error(f"Ошибка сравнения метрик: {e}")
            return MetricsComparison(
                similarity_score=0.0,
                distance_euclidean=float('inf'),
                distance_cosine=1.0,
                outlier_metrics=[],
                consistency_score=0.0
            )

    def analyze_metrics_consistency(self, metrics_list: List[IdentityMetrics]) -> Dict[str, Any]:
        """
        Анализ консистентности метрик во времени
        
        Args:
            metrics_list: Список метрик для анализа
            
        Returns:
            Словарь с результатами анализа
        """
        try:
            if len(metrics_list) < 2:
                return {'error': 'Недостаточно данных для анализа'}
            
            # Конвертация в матрицу
            metrics_matrix = np.array([m.to_array() for m in metrics_list])
            
            # Статистика по каждой метрике
            means = np.mean(metrics_matrix, axis=0)
            stds = np.std(metrics_matrix, axis=0)
            cvs = stds / means  # Коэффициент вариации
            
            metric_names = SKULL_METRICS + PROPORTION_METRICS
            
            # Анализ стабильности
            stable_metrics = []
            unstable_metrics = []
            
            for i, (name, cv) in enumerate(zip(metric_names, cvs)):
                if cv < 0.1:  # Менее 10% вариации
                    stable_metrics.append(name)
                elif cv > 0.3:  # Более 30% вариации
                    unstable_metrics.append(name)
            
            # Общая оценка консистентности
            overall_consistency = 1.0 - np.mean(cvs)
            
            # Поиск трендов
            trends = self._analyze_temporal_trends(metrics_matrix, metric_names)
            
            return {
                'total_samples': len(metrics_list),
                'overall_consistency': float(overall_consistency),
                'stable_metrics': stable_metrics,
                'unstable_metrics': unstable_metrics,
                'mean_values': {name: float(mean) for name, mean in zip(metric_names, means)},
                'std_values': {name: float(std) for name, std in zip(metric_names, stds)},
                'cv_values': {name: float(cv) for name, cv in zip(metric_names, cvs)},
                'trends': trends
            }
            
        except Exception as e:
            logger.error(f"Ошибка анализа консистентности: {e}")
            return {'error': str(e)}

    def _analyze_temporal_trends(self, metrics_matrix: np.ndarray, metric_names: List[str]) -> Dict[str, str]:
        """Анализ временных трендов в метриках"""
        try:
            trends = {}
            
            for i, name in enumerate(metric_names):
                values = metrics_matrix[:, i]
                
                # Простой линейный тренд
                x = np.arange(len(values))
                correlation = np.corrcoef(x, values)[0, 1]
                
                if correlation > 0.3:
                    trends[name] = 'increasing'
                elif correlation < -0.3:
                    trends[name] = 'decreasing'
                else:
                    trends[name] = 'stable'
            
            return trends
            
        except Exception as e:
            logger.error(f"Ошибка анализа трендов: {e}")
            return {}

    def create_baseline_from_metrics(self, metrics_list: List[IdentityMetrics], 
                                   baseline_name: str, pose_category: str = "frontal"):
        """
        Создание baseline метрик из списка образцов
        
        Args:
            metrics_list: Список метрик для создания baseline
            baseline_name: Имя baseline
            pose_category: Категория позы
        """
        try:
            if len(metrics_list) < 5:
                logger.warning("Недостаточно образцов для создания надежного baseline")
            
            # Конвертация в матрицу
            metrics_matrix = np.array([m.to_array() for m in metrics_list])
            
            # Расчет статистики
            means = np.mean(metrics_matrix, axis=0)
            stds = np.std(metrics_matrix, axis=0)
            
            metric_names = SKULL_METRICS + PROPORTION_METRICS
            
            # Создание baseline объекта
            baseline = BaselineMetrics(
                reference_period=baseline_name,
                mean_values={name: float(mean) for name, mean in zip(metric_names, means)},
                std_values={name: float(std) for name, std in zip(metric_names, stds)},
                sample_count=len(metrics_list),
                creation_date=datetime.datetime.now(),
                confidence_level=min(1.0, len(metrics_list) / 20.0)  # Максимальная уверенность при 20+ образцах
            )
            
            # Сохранение в кэш
            baseline_key = f"baseline_{pose_category}"
            self.baseline_cache[baseline_key] = baseline
            
            # Сохранение на диск
            self._save_baseline_metrics()
            
            logger.info(f"Создан baseline '{baseline_name}' для позы '{pose_category}' из {len(metrics_list)} образцов")
            
        except Exception as e:
            logger.error(f"Ошибка создания baseline: {e}")

    def _save_baseline_metrics(self):
        """Сохранение baseline метрик на диск"""
        try:
            baseline_path = self.cache_dir / "baseline_metrics.pkl"
            with open(baseline_path, 'wb') as f:
                pickle.dump(self.baseline_cache, f)
            logger.debug("Baseline метрики сохранены")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения baseline метрик: {e}")

    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики обработки
        
        Returns:
            Словарь со статистикой
        """
        stats = self.processing_stats.copy()
        
        if stats['total_calculated'] > 0:
            stats['success_rate'] = stats['successful_calculations'] / stats['total_calculated']
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_calculated']
        else:
            stats['success_rate'] = 0.0
            stats['cache_hit_rate'] = 0.0
        
        # Информация о кэше
        stats['cache_info'] = {
            'metrics_cached': len(self.metrics_cache),
            'baselines_loaded': len(self.baseline_cache)
        }
        
        # Информация о памяти
        process = psutil.Process()
        memory_info = process.memory_info()
        stats['memory_usage_mb'] = memory_info.rss / (1024 * 1024)
        
        return stats

    def clear_cache(self):
        """Очистка кэша метрик"""
        try:
            self.metrics_cache.clear()
            logger.info("Кэш метрик очищен")
            
        except Exception as e:
            logger.error(f"Ошибка очистки кэша: {e}")

    def export_metrics_to_csv(self, metrics_list: List[IdentityMetrics], output_path: str):
        """
        Экспорт метрик в CSV файл
        
        Args:
            metrics_list: Список метрик для экспорта
            output_path: Путь для сохранения CSV
        """
        try:
            import pandas as pd
            
            # Подготовка данных
            data = []
            for metrics in metrics_list:
                row = {
                    'image_id': metrics.image_id,
                    'filepath': metrics.filepath,
                    'pose_category': metrics.pose_category,
                    'confidence_score': metrics.confidence_score,
                    'processing_time_ms': metrics.processing_time_ms
                }
                
                # Добавление всех метрик
                for metric_name in SKULL_METRICS + PROPORTION_METRICS:
                    row[metric_name] = getattr(metrics, metric_name, 0.0)
                
                data.append(row)
            
            # Создание DataFrame и сохранение
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
            
            logger.info(f"Метрики экспортированы в {output_path}")
            
        except ImportError:
            logger.error("Pandas не установлен, экспорт в CSV недоступен")
        except Exception as e:
            logger.error(f"Ошибка экспорта в CSV: {e}")

# === ФУНКЦИИ САМОТЕСТИРОВАНИЯ ===

def self_test():
    """Самотестирование модуля metrics_calculator"""
    try:
        logger.info("Запуск самотестирования metrics_calculator...")
        
        # Создание экземпляра калькулятора
        calculator = MetricsCalculator()
        
        # Создание тестовых ландмарков
        test_landmarks = np.random.rand(68, 3) * 100
        
        # Создание тестового пакета ландмарков
        class MockLandmarksPackage:
            def __init__(self):
                self.image_id = "test_image"
                self.filepath = "test.jpg"
                self.normalized_landmarks = test_landmarks
                self.pose_category = "frontal"
        
        test_package = MockLandmarksPackage()
        
        # Тест расчета метрик
        metrics = calculator.calculate_identity_signature_metrics(test_package)
        assert metrics is not None, "Метрики не рассчитаны"
        assert len(metrics.to_array()) == 15, "Неверное количество метрик"
        
        # Тест сравнения метрик
        metrics2 = calculator.calculate_identity_signature_metrics(test_package)
        comparison = calculator.compare_metrics(metrics, metrics2)
        assert comparison.similarity_score > 0.9, "Идентичные метрики должны быть похожи"
        
        # Тест анализа консистентности
        metrics_list = [metrics, metrics2]
        consistency = calculator.analyze_metrics_consistency(metrics_list)
        assert 'overall_consistency' in consistency, "Отсутствует анализ консистентности"
        
        # Тест статистики
        stats = calculator.get_processing_statistics()
        assert 'success_rate' in stats, "Отсутствует статистика"
        
        logger.info("Самотестирование metrics_calculator завершено успешно")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка самотестирования: {e}")
        return False

# === ИНИЦИАЛИЗАЦИЯ ===

if __name__ == "__main__":
    # Запуск самотестирования при прямом вызове модуля
    success = self_test()
    if success:
        print("✅ Модуль metrics_calculator работает корректно")
        
        # Демонстрация основной функциональности
        calculator = MetricsCalculator()
        print(f"📊 Метрик в кэше: {len(calculator.metrics_cache)}")
        print(f"🔧 Baseline загружено: {len(calculator.baseline_cache)}")
        print(f"📏 Костных метрик: {len(SKULL_METRICS)}")
        print(f"📐 Пропорциональных метрик: {len(PROPORTION_METRICS)}")
        print(f"💾 Кэш-директория: {calculator.cache_dir}")
    else:
        print("❌ Обнаружены ошибки в модуле metrics_calculator")
        exit(1)