# texture_analyzer.py
"""
TextureAnalyzer - Анализатор текстуры кожи с Haralick, LBP, Gabor метриками
Версия: 2.0
Дата: 2025-06-15
Исправлены все критические ошибки согласно правкам
"""

import numpy as np
import cv2
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import json
import math
from datetime import datetime
import os
import hashlib
import time
import psutil
from functools import lru_cache
import threading
from collections import OrderedDict, defaultdict
import msgpack
from dataclasses import dataclass, field, asdict

# Научные библиотеки для анализа текстуры
from scipy import stats
from scipy.spatial.distance import euclidean, cosine
from scipy.ndimage import convolve
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.measure import shannon_entropy, label, regionprops
from skimage.morphology import disk, binary_opening, binary_closing, remove_small_objects
from skimage.filters import threshold_local, gaussian, gabor_kernel
from skimage import filters
import mahotas

from core_config import get_config

# Настройка логирования
logger = logging.getLogger(__name__)

# === КОНСТАНТЫ И КОНФИГУРАЦИЯ ===

# Параметры LBP (Local Binary Pattern)
LBP_RADIUS = 3
LBP_POINTS = 24
LBP_METHOD = 'uniform'

# Параметры Габора
GABOR_FREQUENCIES = [0.1, 0.3, 0.5, 0.7]
GABOR_ORIENTATIONS = 11  # количество ориентаций
GABOR_SIGMA_X = 2.0
GABOR_SIGMA_Y = 2.0

# Параметры GLCM (Gray-Level Co-occurrence Matrix)
GLCM_DISTANCES = [1, 2, 3]
GLCM_ANGLES = [0, 45, 90, 135]
GLCM_LEVELS = 256

# Пороги энтропии для разных уровней масок
ENTROPY_THRESHOLDS = {
    'level_1': {'min_entropy': 4.0, 'max_entropy': 7.5},
    'level_2': {'min_entropy': 4.5, 'max_entropy': 7.8},
    'level_3': {'min_entropy': 5.0, 'max_entropy': 8.0},
    'level_4': {'min_entropy': 5.5, 'max_entropy': 8.2},
    'level_5': {'min_entropy': 6.0, 'max_entropy': 8.5}
}

# Определение зон лица для анализа
FACE_ZONES = {
    'forehead': {'landmarks': list(range(17, 27)), 'expansion': 1.2},
    'left_cheek': {'landmarks': [1, 2, 3, 31, 39, 40, 41], 'expansion': 1.1},
    'right_cheek': {'landmarks': [13, 14, 15, 35, 42, 43, 44], 'expansion': 1.1},
    'nose': {'landmarks': list(range(27, 36)), 'expansion': 1.0},
    'chin': {'landmarks': [6, 7, 8, 9, 10, 57, 58, 59], 'expansion': 1.1}
}

# ИСПРАВЛЕНО: Пороги анализа текстуры согласно правкам
TEXTURE_ANALYSIS_THRESHOLDS = {
    "entropy_natural_min": 6.2,
    "entropy_natural_max": 7.8,
    "lbp_uniformity_min": 0.6,
    "gabor_energy_min": 100.0,
    "haralick_contrast_max": 0.8,
    "haralick_homogeneity_min": 0.8,
    "fourier_spectral_centroid_min": 50,
    "fourier_dominant_frequency_min": 50,
    "lbp_variance_min": 0.005,
    "lbp_entropy_mean": 4.0,
    "lbp_entropy_std": 0.5,
    "gabor_energy_mean": 150.0,
    "gabor_energy_std": 50.0,
    "pore_min_area": 3,
    "pore_max_area": 200,
    "pore_block_size": 35,
    "pore_offset": 0.05,
    "pore_selem_open_disk_size": 1,
    "pore_selem_close_disk_size": 2,
    "canny_threshold1": 100,
    "canny_threshold2": 200,
    "search_radius": 10
}

# === ДОБАВЛЕНО: Порог для градиента (используется в анализе швов и пор)
gradient_magnitude_threshold = 25

# === СТРУКТУРЫ ДАННЫХ ===

@dataclass
class ZoneTextureMetrics:
    """Метрики текстуры для одной зоны лица"""
    zone_name: str
    
    # LBP метрики
    lbp_histogram: np.ndarray
    lbp_uniformity: float
    lbp_entropy: float
    
    # Габор метрики
    gabor_responses: np.ndarray  # [orientations x frequencies]
    gabor_energy: float
    gabor_mean_amplitude: float
    
    # GLCM метрики
    glcm_contrast: float
    glcm_dissimilarity: float
    glcm_homogeneity: float
    glcm_energy: float
    glcm_correlation: float
    
    # Спектральные характеристики
    fourier_spectrum: np.ndarray
    spectral_energy: float
    dominant_frequency: float
    
    # Статистические метрики
    shannon_entropy: float
    mean_intensity: float
    std_intensity: float
    skewness: float
    kurtosis: float
    
    # Флаги детекции артефактов
    seam_artifacts_detected: bool = False
    texture_transitions_detected: bool = False
    
    # Метаданные
    zone_area_pixels: int = 0
    processing_time_ms: float = 0.0

@dataclass
class TexturePackage:
    """Пакет результатов текстурного анализа"""
    image_id: str
    filepath: str
    
    # Метрики по зонам
    zone_metrics: Dict[str, ZoneTextureMetrics]
    
    # Общие метрики
    overall_entropy: float
    material_authenticity_score: float
    mask_level: int  # 1-5 уровень технологии маски
    
    # Детекция артефактов
    seam_artifacts_detected: bool = False
    texture_transitions_detected: bool = False
    edge_artifacts_count: int = 0
    
    # Спектральный анализ
    global_fourier_signature: np.ndarray = None
    frequency_anomalies: List[str] = field(default_factory=list)
    
    # Метаданные обработки
    processing_time_ms: float = 0.0
    extraction_method: str = "multi_zone_analysis"
    device_used: str = "cpu"
    
    # Флаги качества
    quality_flags: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

@dataclass
class MaskDetectionResult:
    """Результат детекции маски"""
    is_mask_detected: bool
    confidence_score: float
    mask_technology_level: int
    evidence_factors: List[str]
    suspicious_zones: List[str]

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def calculate_mean_std(data: np.ndarray) -> Tuple[float, float]:
    """Расчет среднего и стандартного отклонения"""
    if data is None or not hasattr(data, 'size') or data.size == 0:
        return 0.0, 0.0
    return float(np.mean(data)), float(np.std(data))

def create_zone_mask(image_shape: Tuple[int, int], zone_coords: np.ndarray) -> np.ndarray:
    """Создание маски для зоны лица"""
    try:
        mask = np.zeros(image_shape, dtype=np.uint8)
        if zone_coords is not None and hasattr(zone_coords, 'size') and zone_coords.size > 0:
            cv2.fillPoly(mask, [zone_coords.astype(np.int32)], 255)
        return mask
    except Exception as e:
        logger.error(f"Ошибка создания маски зоны: {e}")
        return np.zeros(image_shape, dtype=np.uint8)

def sigmoid_score(value: float, mean: float, std: float, reverse: bool = False) -> float:
    """Расчет sigmoid score для нормализации метрик"""
    try:
        if std == 0:
            return 1.0 if value == mean else 0.0
        z_score = (value - mean) / std
        sigmoid = 1 / (1 + np.exp(-z_score))
        return 1 - sigmoid if reverse else sigmoid
    except Exception as e:
        logger.error(f"Ошибка расчета sigmoid score: {e}")
        return 0.5

# === ОСНОВНОЙ КЛАСС АНАЛИЗАТОРА ТЕКСТУР ===

class TextureAnalyzer:
    """
    Анализатор текстуры кожи с полной функциональностью
    ИСПРАВЛЕНО: Все критические ошибки согласно правкам
    """

    def __init__(self):
        """Инициализация анализатора текстуры"""
        logger.info("Инициализация TextureAnalyzer")
        
        self.config = get_config()
        self.cache_dir = Path("./cache/texture_analyzer")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # ИСПРАВЛЕНО: Параметры LBP согласно правкам
        self.lbp_params = {
            "radius": 3,
            "n_points": 24,  # 8 * radius для uniform patterns
            "method": "uniform"
        }

        # ИСПРАВЛЕНО: Параметры Gabor согласно правкам
        self.gabor_params = {
            "orientations": np.linspace(0, 180, 11, endpoint=False),
            "frequencies": [0.1, 0.2, 0.3, 0.4]
        }

        # Параметры энтропии
        self.entropy_params = {
            "disk_size": 9
        }

        # ИСПРАВЛЕНО: Пороги для натуральной кожи
        self.entropy_natural_threshold_mean = (
            TEXTURE_ANALYSIS_THRESHOLDS["entropy_natural_min"] +
            TEXTURE_ANALYSIS_THRESHOLDS["entropy_natural_max"]
        ) / 2
        self.entropy_natural_threshold_std = (
            TEXTURE_ANALYSIS_THRESHOLDS["entropy_natural_max"] -
            TEXTURE_ANALYSIS_THRESHOLDS["entropy_natural_min"]
        ) / 4

        # Базовые линии текстуры
        self.texture_baselines = self.load_texture_baselines()
        
        # Кэш результатов
        self.texture_cache: Dict[str, TexturePackage] = {}
        self.gabor_kernels_cache: Dict[Tuple[float, float], np.ndarray] = {}
        
        # Статистика
        self.processing_stats = {
            'total_processed': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'cache_hits': 0,
            'mask_detections': 0
        }
        
        # Блокировка для потокобезопасности
        self.analysis_lock = threading.Lock()
        
        # Флаг калибровки
        self.calibrated = False
        
        # Предварительная генерация Габор-ядер
        self._precompute_gabor_kernels()
        
        logger.info("TextureAnalyzer инициализирован")

    def _precompute_gabor_kernels(self):
        """Предварительное вычисление Габор-ядер"""
        try:
            for frequency in self.gabor_params["frequencies"]:
                for i in range(len(self.gabor_params["orientations"])):
                    theta = np.radians(self.gabor_params["orientations"][i])
                    kernel = gabor_kernel(frequency, theta=theta, 
                                        sigma_x=GABOR_SIGMA_X, sigma_y=GABOR_SIGMA_Y)
                    self.gabor_kernels_cache[(frequency, theta)] = kernel
            
            logger.debug(f"Предвычислено {len(self.gabor_kernels_cache)} Габор-ядер")
            
        except Exception as e:
            logger.error(f"Ошибка предвычисления Габор-ядер: {e}")

    def load_texture_baselines(self) -> Dict[str, Dict[str, float]]:
        """
        ИСПРАВЛЕНО: Загрузка базовых линий текстуры из JSON
        Согласно правкам: правильная структура baseline данных
        """
        baseline_file = self.config.get_config_path() / "texture_baselines.json"
        try:
            if baseline_file.exists():
                with open(baseline_file, 'r', encoding='utf-8') as f:
                    baselines = json.load(f)
                logger.info(f"Базовые линии загружены из {baseline_file}")
                return baselines
            else:
                logger.warning(f"Файл базовых линий не найден: {baseline_file}. Используются значения по умолчанию")
        except Exception as e:
            logger.error(f"Ошибка загрузки базовых линий: {e}")

        # ИСПРАВЛЕНО: Значения по умолчанию для всех зон
        return {
            "forehead": {
                "haralick_mean_contrast": 0.15,
                "haralick_std_contrast": 0.05,
                "shannon_entropy_mean": 7.5,
                "shannon_entropy_std": 0.5,
                "gabor_energy_mean": 100.0,
                "gabor_energy_std": 20.0,
                "lbp_uniformity_mean": 0.9,
                "lbp_uniformity_std": 0.05
            },
            "cheek": {
                "haralick_mean_contrast": 0.18,
                "haralick_std_contrast": 0.06,
                "shannon_entropy_mean": 7.8,
                "shannon_entropy_std": 0.6,
                "gabor_energy_mean": 110.0,
                "gabor_energy_std": 25.0,
                "lbp_uniformity_mean": 0.88,
                "lbp_uniformity_std": 0.06
            },
            "default": {
                "haralick_mean_contrast": 0.17,
                "haralick_std_contrast": 0.05,
                "shannon_entropy_mean": 7.6,
                "shannon_entropy_std": 0.55,
                "gabor_energy_mean": 105.0,
                "gabor_energy_std": 22.0,
                "lbp_uniformity_mean": 0.89,
                "lbp_uniformity_std": 0.055
            }
        }

    def define_skin_zones(self, landmarks: np.ndarray, image_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """
        ИСПРАВЛЕНО: Определение зон кожи по ландмаркам
        Согласно правкам: правильное определение 5 зон лица
        """
        if landmarks is None or not hasattr(landmarks, 'size') or landmarks.size == 0 or landmarks.shape[0] < 68:
            logger.warning("Недостаточно ландмарок для определения зон кожи")
            return {}

        try:
            logger.info("Определение зон кожи лица")
            zones = {}

            # Вычисление высоты лица для адаптивного расширения лба (пункт 21)
            min_x, min_y = np.min(landmarks[:, 0]), np.min(landmarks[:, 1])
            max_x, max_y = np.max(landmarks[:, 0]), np.max(landmarks[:, 1])
            face_bbox_height = max_y - min_y

            # Определяем относительное расширение (например, 15% от высоты лица)
            forehead_expansion_px = int(face_bbox_height * 0.15)  # 15% от высоты BBox лица

            # Лоб (forehead) - точки 17-26
            forehead_points = landmarks[17:27]  # Брови

            # Расширение области лба вверх (ИСПРАВЛЕНО: пункт 21)
            forehead_top = forehead_points.copy()
            forehead_top[:, 1] -= forehead_expansion_px  # Смещение вверх
            forehead_zone = np.vstack([forehead_top, forehead_points[::-1]])
            zones["forehead"] = forehead_zone[:, :2].astype(np.int32)

            # Левая щека (left_cheek) - точки 1-5, 31, 41
            left_cheek_points = np.vstack([
                landmarks[1:6],  # Левая сторона лица
                landmarks[[31]],  # Нос
                landmarks[[41]]  # Левый глаз
            ])
            zones["left_cheek"] = left_cheek_points[:, :2].astype(np.int32)

            # Правая щека (right_cheek) - точки 11-15, 35, 46
            right_cheek_points = np.vstack([
                landmarks[11:16],  # Правая сторона лица
                landmarks[[35]],  # Нос
                landmarks[[46]]  # Правый глаз
            ])
            zones["right_cheek"] = right_cheek_points[:, :2].astype(np.int32)

            # Нос (nose) - точки 27-35
            nose_points = landmarks[27:36]
            zones["nose"] = nose_points[:, :2].astype(np.int32)

            # Подбородок (chin) - точки 6-10
            chin_points = landmarks[6:11]

            # Расширение области подбородка вниз
            chin_bottom = chin_points.copy()
            chin_bottom[:, 1] += int(face_bbox_height * 0.05)  # Смещение вниз, например, 5% от высоты лица
            chin_zone = np.vstack([chin_points, chin_bottom[::-1]])
            zones["chin"] = chin_zone[:, :2].astype(np.int32)

            logger.info(f"Определено {len(zones)} зон кожи")
            return zones

        except Exception as e:
            logger.error(f"Ошибка определения зон кожи: {e}")
            return {}

    def analyze_skin_texture_by_zones(self, image: np.ndarray, landmarks: np.ndarray) -> Optional[TexturePackage]:
        """
        ИСПРАВЛЕНО: Анализ текстуры кожи по зонам
        Согласно правкам: robust analysis for 5 zones
        """
        try:
            start_time = time.time()
            
            # Генерация ID изображения
            image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
            image_id = hashlib.sha256(image_bytes).hexdigest()
            
            # Проверка кэша
            if image_id in self.texture_cache:
                self.processing_stats['cache_hits'] += 1
                cached_result = self.texture_cache[image_id]
                cached_result.processing_time_ms = (time.time() - start_time) * 1000
                return cached_result
            
            logger.info(f"Анализ текстуры кожи по зонам. Изображение: {image.shape}")

            # ИСПРАВЛЕНО: Конвертация типа изображения для OpenCV
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)

            # Преобразование в оттенки серого
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            zone_metrics = {}

            # Определение зон кожи
            skin_zones = self.define_skin_zones(landmarks, image.shape[:2])

            for zone_name, zone_coords in skin_zones.items():
                if zone_coords is None or not hasattr(zone_coords, 'size') or zone_coords.size == 0:
                    continue

                logger.info(f"Анализ зоны: {zone_name}")

                # Создание маски зоны
                mask = create_zone_mask(image.shape[:2], zone_coords)

                # Создание 3D маски для цветного изображения
                mask_3d = np.stack([mask, mask, mask], axis=-1)

                # Извлечение региона зоны
                zone_region_colored = image * (mask_3d / 255.0)

                # Получение координат маски
                y_coords, x_coords = np.where(mask > 0)

                if y_coords is None or not hasattr(y_coords, 'size') or y_coords.size == 0 or x_coords is None or not hasattr(x_coords, 'size') or x_coords.size == 0:
                    continue

                # Bounding box зоны
                min_y, max_y = np.min(y_coords), np.max(y_coords)
                min_x, max_x = np.min(x_coords), np.max(x_coords)

                # Обрезка региона
                cropped_zone_region = zone_region_colored[min_y:max_y+1, min_x:max_x+1]
                cropped_mask = mask[min_y:max_y+1, min_x:max_x+1]

                if cropped_zone_region is None or not hasattr(cropped_zone_region, 'size') or cropped_zone_region.size == 0:
                    continue

                # ИСПРАВЛЕНО: Преобразование к типу np.uint8 перед cv2.cvtColor
                if cropped_zone_region.dtype != np.uint8:
                    cropped_zone_region = cropped_zone_region.astype(np.uint8)

                # Валидные пиксели
                valid_pixels_mask = cropped_mask > 0

                if cropped_zone_region.ndim == 3:
                    valid_pixels_colored = cropped_zone_region[valid_pixels_mask]
                else:
                    valid_pixels_colored = cropped_zone_region[valid_pixels_mask]

                # Преобразование в grayscale
                zone_region_gray = cv2.cvtColor(cropped_zone_region, cv2.COLOR_BGR2GRAY) if cropped_zone_region.ndim == 3 else cropped_zone_region
                valid_pixels_gray = zone_region_gray[valid_pixels_mask]

                if valid_pixels_colored is None or not hasattr(valid_pixels_colored, 'size') or valid_pixels_colored.size == 0 or valid_pixels_gray is None or not hasattr(valid_pixels_gray, 'size') or valid_pixels_gray.size == 0:
                    logger.warning(f"Зона {zone_name} не содержит валидных пикселей")
                    continue

                # Анализ текстуры зоны
                zone_analysis = self.analyze_zone_texture(zone_region_gray, valid_pixels_gray)
                zone_metrics[zone_name] = zone_analysis

            # Общий анализ изображения
            overall_entropy = self.calculate_shannon_entropy(gray_image.flatten())
            
            # Детекция артефактов
            seam_detected = self._detect_seam_artifacts(gray_image, landmarks)
            transitions_detected = self._detect_texture_transitions(gray_image, landmarks)
            edge_count = self._count_edge_artifacts(gray_image)
            
            # Глобальный спектральный анализ
            global_fourier = self._calculate_global_fourier_signature(gray_image)
            frequency_anomalies = self._detect_frequency_anomalies(global_fourier)
            
            # Расчет балла подлинности материала
            material_score = self.calculate_material_authenticity_score(zone_metrics)
            
            # Классификация уровня технологии маски
            mask_level = self._classify_mask_technology_level(zone_metrics, overall_entropy)
            
            # Создание пакета результатов
            package = TexturePackage(
                image_id=image_id,
                filepath="",  # Будет заполнено вызывающей функцией
                zone_metrics=zone_metrics,
                overall_entropy=overall_entropy,
                material_authenticity_score=material_score,
                mask_level=mask_level,
                seam_artifacts_detected=seam_detected,
                texture_transitions_detected=transitions_detected,
                edge_artifacts_count=edge_count,
                global_fourier_signature=global_fourier,
                frequency_anomalies=frequency_anomalies,
                processing_time_ms=(time.time() - start_time) * 1000,
                device_used="cpu"
            )
            
            # Валидация результата
            self._validate_texture_package(package)
            
            # Сохранение в кэш
            self.texture_cache[image_id] = package
            
            self.processing_stats['successful_analyses'] += 1
            self.processing_stats['total_processed'] += 1
            
            if mask_level > 3:
                self.processing_stats['mask_detections'] += 1

            logger.info(f"Анализ текстуры завершен для {len(zone_metrics)} зон")
            logger.debug(f"Текстурный анализ завершен за {package.processing_time_ms:.1f}мс")
            return package

        except Exception as e:
            logger.error(f"Ошибка анализа текстуры по зонам: {e}")
            self.processing_stats['failed_analyses'] += 1
            self.processing_stats['total_processed'] += 1
            return None

    def analyze_zone_texture(self, zone_region_2d: np.ndarray, valid_pixels_1d: np.ndarray) -> ZoneTextureMetrics:
        """
        ИСПРАВЛЕНО: Анализ текстуры конкретной зоны
        Согласно правкам: все метрики Haralick, LBP, Gabor, Fourier
        """
        try:
            start_time = time.time()
            logger.info("Анализ текстуры зоны")

            # ИСПРАВЛЕНО: Shannon entropy
            shannon_entropy = self.calculate_shannon_entropy(valid_pixels_1d)

            # ИСПРАВЛЕНО: LBP features
            lbp_features = self.calculate_lbp_features(zone_region_2d)

            # ИСПРАВЛЕНО: Gabor responses
            gabor_responses = self.calculate_gabor_responses(zone_region_2d)

            # ИСПРАВЛЕНО: Fourier spectrum
            fourier_spectrum = self.calculate_fourier_spectrum(zone_region_2d)

            # ИСПРАВЛЕНО: Haralick features
            haralick_features = self.calculate_haralick_features(zone_region_2d)

            # Статистические метрики
            stats_features = self._calculate_statistical_features(valid_pixels_1d)

            # Детекция артефактов в зоне
            seam_detected = self._detect_zone_seam_artifacts(zone_region_2d)
            transitions_detected = self._detect_zone_texture_transitions(zone_region_2d)

            # Создание объекта метрик
            metrics = ZoneTextureMetrics(
                zone_name="",  # Будет заполнено вызывающей функцией
                lbp_histogram=lbp_features.get('lbp_histogram', np.array([])),
                lbp_uniformity=lbp_features.get('lbp_uniformity', 0.0),
                lbp_entropy=lbp_features.get('histogram_entropy', 0.0),
                gabor_responses=gabor_responses.get('gabor_responses_matrix', np.array([])),
                gabor_energy=gabor_responses.get('gabor_total_energy', 0.0),
                gabor_mean_amplitude=gabor_responses.get('gabor_mean_energy', 0.0),
                glcm_contrast=haralick_features.get('contrast', 0.0),
                glcm_dissimilarity=haralick_features.get('dissimilarity', 0.0),
                glcm_homogeneity=haralick_features.get('homogeneity', 0.0),
                glcm_energy=haralick_features.get('energy', 0.0),
                glcm_correlation=haralick_features.get('correlation', 0.0),
                fourier_spectrum=fourier_spectrum.get('magnitude_spectrum', np.array([])),
                spectral_energy=fourier_spectrum.get('spectral_energy', 0.0),
                dominant_frequency=fourier_spectrum.get('dominant_frequency', 0.0),
                shannon_entropy=shannon_entropy,
                mean_intensity=stats_features.get('mean', 0.0),
                std_intensity=stats_features.get('std', 0.0),
                skewness=stats_features.get('skewness', 0.0),
                kurtosis=stats_features.get('kurtosis', 0.0),
                seam_artifacts_detected=seam_detected,
                texture_transitions_detected=transitions_detected,
                zone_area_pixels=len(valid_pixels_1d),
                processing_time_ms=(time.time() - start_time) * 1000
            )

            return metrics

        except Exception as e:
            logger.error(f"Ошибка анализа текстуры зоны: {e}")
            # Возвращаем пустые метрики
            return ZoneTextureMetrics(
                zone_name="",
                lbp_histogram=np.zeros(LBP_POINTS + 2),
                lbp_uniformity=0.0,
                lbp_entropy=0.0,
                gabor_responses=np.zeros((len(GABOR_FREQUENCIES), GABOR_ORIENTATIONS)),
                gabor_energy=0.0,
                gabor_mean_amplitude=0.0,
                glcm_contrast=0.0,
                glcm_dissimilarity=0.0,
                glcm_homogeneity=0.0,
                glcm_energy=0.0,
                glcm_correlation=0.0,
                fourier_spectrum=np.zeros((8, 8)),
                spectral_energy=0.0,
                dominant_frequency=0.0,
                shannon_entropy=0.0,
                mean_intensity=0.0,
                std_intensity=0.0,
                skewness=0.0,
                kurtosis=0.0,
                processing_time_ms=0.0
            )

    def calculate_lbp_features(self, image_region: np.ndarray) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Расчет Local Binary Patterns (LBP) features
        Согласно правкам: правильные параметры radius=3, n_points=24, uniform
        """
        try:
            logger.info("Расчет LBP features")

            # ИСПРАВЛЕНО: Параметры LBP согласно правкам
            radius = self.lbp_params["radius"]
            n_points = self.lbp_params["n_points"]
            method = self.lbp_params["method"]

            # Проверка корректности параметров для uniform method
            if method == "uniform" and n_points != 8 * radius:
                logger.warning(f"Для uniform method рекомендуется n_points = 8 * radius. Текущие: n_points={n_points}, radius={radius}")

            # Расчет LBP
            lbp_image = local_binary_pattern(image_region, n_points, radius, method=method)

            # ИСПРАВЛЕНО: Гистограмма LBP
            n_bins = int(lbp_image.max()) + 1
            hist, _ = np.histogram(lbp_image.ravel(), bins=n_bins, range=(0, n_bins))

            # Нормализация гистограммы
            hist = hist.astype(float)
            hist = hist / (hist.sum() + 1e-8)

            # ИСПРАВЛЕНО: Метрики LBP
            lbp_features = {
                "lbp_histogram": hist.tolist(),
                "histogram_variance": float(np.var(hist)),
                "histogram_entropy": float(-np.sum(hist * np.log2(hist + 1e-8))),
                "lbp_uniformity": float(np.sum(hist**2))  # Uniformity measure
            }

            logger.info("LBP features рассчитаны успешно")
            return lbp_features

        except Exception as e:
            logger.error(f"Ошибка расчета LBP features: {e}")
            return {
                "lbp_histogram": [],
                "histogram_variance": 0.0,
                "histogram_entropy": 0.0,
                "lbp_uniformity": 0.0
            }

    def calculate_shannon_entropy(self, pixel_values: np.ndarray) -> float:
        """
        ИСПРАВЛЕНО: Расчет Shannon entropy
        Согласно правкам: правильная обработка гистограммы
        """
        if pixel_values is None or not hasattr(pixel_values, 'size') or pixel_values.size == 0:
            return 0.0

        try:
            logger.info("Расчет Shannon entropy")

            # Преобразование в 1D массив
            if pixel_values.ndim > 1:
                pixel_values = pixel_values.flatten()

            # ИСПРАВЛЕНО: Нормализация к диапазону 0-255
            if pixel_values.max() > 255 or pixel_values.min() < 0:
                pixel_values = 255 * (pixel_values - pixel_values.min()) / (pixel_values.max() - pixel_values.min() + 1e-8)

            # Преобразование к uint8
            pixel_values = pixel_values.astype(np.uint8)

            # ИСПРАВЛЕНО: Использование cv2.calcHist для корректной гистограммы
            if not pixel_values.flags.c_contiguous:
                pixel_values = np.ascontiguousarray(pixel_values)

            hist = cv2.calcHist([pixel_values], [0], None, [256], [0, 256])
            hist = hist.ravel()
            hist = hist / (hist.sum() + 1e-8)

            # Shannon entropy
            entropy = -np.sum(hist * np.log2(hist + 1e-8))

            logger.info(f"Shannon entropy рассчитан: {entropy:.3f}")
            return float(entropy)

        except Exception as e:
            logger.error(f"Ошибка расчета Shannon entropy: {e}")
            return 0.0

    def calculate_gabor_responses(self, image_region: np.ndarray) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Расчет Gabor responses (11 ориентаций, 4 частоты)
        """
        try:
            logger.info("Расчет Gabor responses")

            gabor_results = {}
            responses_matrix = []

            if image_region.dtype != np.float32 and image_region.dtype != np.float64:
                image_region = image_region.astype(np.float32) / 255.0

            orientations = self.gabor_params["orientations"]
            frequencies = self.gabor_params["frequencies"]

            for theta in orientations:
                freq_responses = []
                for freq in frequencies:
                    # Получение предвычисленного ядра
                    theta_rad = np.radians(theta)
                    if (freq, theta_rad) in self.gabor_kernels_cache:
                        kernel = self.gabor_kernels_cache[(freq, theta_rad)]
                    else:
                        current_sigma = 0.56 / freq if freq > 0 else 1.0
                        kernel = gabor_kernel(
                            frequency=freq,
                            theta=theta_rad,
                            sigma_x=current_sigma,
                            sigma_y=current_sigma
                        )

                    filtered_image = convolve(image_region, np.real(kernel), mode='nearest')
                    energy = float(np.sum(filtered_image**2))
                    freq_responses.append(energy)

                    gabor_results[f"gabor_theta{theta:.1f}_freq{freq}"] = {
                        "mean": float(np.mean(filtered_image)),
                        "std": float(np.std(filtered_image)),
                        "energy": energy
                    }

                responses_matrix.append(freq_responses)

            all_energies = [response["energy"] for response in gabor_results.values()]
            gabor_results["gabor_total_energy"] = float(np.sum(all_energies))
            gabor_results["gabor_mean_energy"] = float(np.mean(all_energies))
            gabor_results["gabor_energy_std"] = float(np.std(all_energies))
            gabor_results["gabor_responses_matrix"] = np.array(responses_matrix)

            logger.info(f"Gabor responses рассчитаны для {len(orientations)}x{len(frequencies)} комбинаций")
            return gabor_results

        except Exception as e:
            logger.error(f"Ошибка расчета Gabor responses: {e}")
            return {
                "gabor_total_energy": 0.0,
                "gabor_mean_energy": 0.0,
                "gabor_energy_std": 0.0,
                "gabor_responses_matrix": np.array([])
            }

    def calculate_fourier_spectrum(self, image_region: np.ndarray) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Расчет Fourier spectrum + FFT пики 0.15, 0.3, 0.6
        """
        if image_region is None or not hasattr(image_region, 'size') or image_region.size == 0 or image_region.ndim == 0:
            logger.warning("calculate_fourier_spectrum: Пустой или невалидный регион изображения")
            return {
                "spectral_centroid": 0.0,
                "spectral_rolloff": 0.0,
                "dominant_frequency": 0,
                "fft_peaks": {0.15: 0.0, 0.3: 0.0, 0.6: 0.0},
                "magnitude_spectrum": np.array([]),
                "spectral_energy": 0.0
            }

        try:
            logger.info("Расчет Fourier spectrum")

            fourier = np.fft.fft2(image_region)
            fourier_shifted = np.fft.fftshift(fourier)
            magnitude_spectrum = np.abs(fourier_shifted)

            if np.sum(magnitude_spectrum) == 0:
                logger.warning("calculate_fourier_spectrum: Нулевой спектр магнитуд")
                return {
                    "spectral_centroid": 0.0,
                    "spectral_rolloff": 0.0,
                    "dominant_frequency": 0,
                    "fft_peaks": {0.15: 0.0, 0.3: 0.0, 0.6: 0.0},
                    "magnitude_spectrum": magnitude_spectrum,
                    "spectral_energy": 0.0
                }

            # FFT пики на частотах 0.15, 0.3, 0.6
            h, w = magnitude_spectrum.shape
            ky = np.fft.fftfreq(h)
            kx = np.fft.fftfreq(w)
            kgrid = np.sqrt(np.add.outer(ky**2, kx**2))

            peaks = {}
            for target in [0.15, 0.3, 0.6]:
                mask = (kgrid > target-0.01) & (kgrid < target+0.01)
                peaks[target] = float(magnitude_spectrum[mask].mean()) if np.any(mask) else 0.0

            spectral_metrics = {
                "spectral_centroid": self.calculate_spectral_centroid(magnitude_spectrum),
                "spectral_rolloff": self.calculate_spectral_rolloff(magnitude_spectrum),
                "dominant_frequency": self.find_dominant_frequency(magnitude_spectrum),
                "fft_peaks": peaks,
                "magnitude_spectrum": magnitude_spectrum,
                "spectral_energy": float(np.sum(magnitude_spectrum**2))
            }

            logger.info("Fourier spectrum рассчитан успешно")
            return spectral_metrics

        except Exception as e:
            logger.error(f"Ошибка расчета Fourier spectrum: {e}")
            return {
                "spectral_centroid": 0.0,
                "spectral_rolloff": 0.0,
                "dominant_frequency": 0,
                "fft_peaks": {0.15: 0.0, 0.3: 0.0, 0.6: 0.0},
                "magnitude_spectrum": np.array([]),
                "spectral_energy": 0.0
            }

    def calculate_spectral_centroid(self, magnitude: np.ndarray) -> float:
        """
        ИСПРАВЛЕНО: Расчет spectral centroid
        Согласно правкам: правильная формула центроида
        """
        try:
            rows, cols = magnitude.shape
            freq_y, freq_x = np.indices(magnitude.shape)
            total_magnitude = np.sum(magnitude)

            if total_magnitude == 0:
                return 0.0

            # Центроид по X и Y
            centroid_x = np.sum(freq_x * magnitude) / total_magnitude
            centroid_y = np.sum(freq_y * magnitude) / total_magnitude

            # Евклидово расстояние от центра
            return float(np.sqrt(centroid_x**2 + centroid_y**2))

        except Exception as e:
            logger.error(f"Ошибка расчета spectral centroid: {e}")
            return 0.0

    def calculate_spectral_rolloff(self, magnitude: np.ndarray, rolloff_threshold: float = 0.85) -> float:
        """
        ИСПРАВЛЕНО: Расчет spectral rolloff
        Согласно правкам: правильный алгоритм rolloff
        """
        try:
            # Преобразование в 1D и сортировка
            flattened_magnitude = np.sort(magnitude.ravel())[::-1]  # По убыванию
            cumulative_magnitude = np.cumsum(flattened_magnitude)
            total_energy = cumulative_magnitude[-1]

            if total_energy == 0:
                return 0.0

            # Поиск точки rolloff
            rolloff_energy = rolloff_threshold * total_energy
            rolloff_idx = np.where(cumulative_magnitude >= rolloff_energy)[0]

            if len(rolloff_idx) == 0:
                return float(len(flattened_magnitude))

            return float(rolloff_idx[0])

        except Exception as e:
            logger.error(f"Ошибка расчета spectral rolloff: {e}")
            return 0.0

    def find_dominant_frequency(self, magnitude: np.ndarray) -> int:
        """
        ИСПРАВЛЕНО: Поиск доминантной частоты
        Согласно правкам: исключение DC компоненты
        """
        try:
            rows, cols = magnitude.shape
            center_row, center_col = rows // 2, cols // 2

            # Копия спектра для обработки
            magnitude_copy = magnitude.copy()

            # ИСПРАВЛЕНО: Исключение DC компоненты (центр)
            magnitude_copy[center_row-1:center_row+2, center_col-1:center_col+2] = 0

            # Поиск максимума
            dominant_idx = np.unravel_index(np.argmax(magnitude_copy), magnitude_copy.shape)

            # Расстояние от центра
            return int(np.sqrt((dominant_idx[0] - center_row)**2 + (dominant_idx[1] - center_col)**2))

        except Exception as e:
            logger.error(f"Ошибка поиска доминантной частоты: {e}")
            return 0

    def calculate_haralick_features(self, image_region: np.ndarray) -> Dict[str, float]:
        """
        ИСПРАВЛЕНО: Расчет Haralick features (GLCM)
        Согласно правкам: правильные параметры GLCM
        """
        try:
            logger.info("Расчет Haralick features")

            # ИСПРАВЛЕНО: Нормализация к 8-битному изображению
            image_8bit = cv2.normalize(image_region, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # ИСПРАВЛЕНО: Параметры GLCM
            distances = [1]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            levels = 256

            # Расчет GLCM
            glcm = graycomatrix(
                image_8bit,
                distances=distances,
                angles=angles,
                levels=levels,
                symmetric=True
            )

            # ИСПРАВЛЕНО: Извлечение Haralick features
            haralick_features = {
                "contrast": float(np.mean(graycoprops(glcm, 'contrast'))),
                "dissimilarity": float(np.mean(graycoprops(glcm, 'dissimilarity'))),
                "homogeneity": float(np.mean(graycoprops(glcm, 'homogeneity'))),
                "energy": float(np.mean(graycoprops(glcm, 'energy'))),
                "correlation": float(np.mean(graycoprops(glcm, 'correlation')))
            }

            logger.info("Haralick features рассчитаны успешно")
            return haralick_features

        except Exception as e:
            logger.error(f"Ошибка расчета Haralick features: {e}")
            return {
                "contrast": 0.0,
                "dissimilarity": 0.0,
                "homogeneity": 0.0,
                "energy": 0.0,
                "correlation": 0.0
            }

    def _calculate_statistical_features(self, pixel_values: np.ndarray) -> Dict[str, float]:
        """Расчет статистических характеристик"""
        try:
            # Основные статистики
            mean_val = float(np.mean(pixel_values))
            std_val = float(np.std(pixel_values))
            
            # Асимметрия и эксцесс
            skewness = float(stats.skew(pixel_values.flatten()))
            kurtosis = float(stats.kurtosis(pixel_values.flatten()))
            
            return {
                'mean': mean_val,
                'std': std_val,
                'skewness': skewness,
                'kurtosis': kurtosis
            }
            
        except Exception as e:
            logger.error(f"Ошибка расчета статистических характеристик: {e}")
            return {
                'mean': 0.0,
                'std': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0
            }

    def _detect_seam_artifacts(self, image: np.ndarray, landmarks: np.ndarray) -> bool:
        """Детекция артефактов швов маски"""
        try:
            # Детекция краев
            edges = cv2.Canny(image, 50, 150)
            
            # Поиск линий Хафа
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is None:
                return False
            
            # Анализ линий на предмет подозрительных швов
            suspicious_lines = 0
            for line in lines:
                rho, theta = line[0]
                # Проверяем линии, которые могут быть швами маски
                if abs(theta - np.pi/2) < 0.2 or abs(theta) < 0.2:  # Вертикальные или горизонтальные
                    suspicious_lines += 1
            
            # Если найдено много подозрительных линий
            return suspicious_lines > 5
            
        except Exception as e:
            logger.error(f"Ошибка детекции швов: {e}")
            return False

    def _detect_texture_transitions(self, image: np.ndarray, landmarks: np.ndarray) -> bool:
        """Детекция резких переходов текстуры"""
        try:
            # Расчет градиентов
            grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            
            # Магнитуда градиента
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Поиск областей с резкими переходами
            threshold = np.percentile(gradient_magnitude, 95)
            high_gradient_areas = gradient_magnitude > threshold
            
            # Подсчет связных компонент
            from skimage.measure import label
            labeled_areas = label(high_gradient_areas)
            num_areas = len(np.unique(labeled_areas)) - 1  # -1 для фона
            
            # Если много областей с резкими переходами
            return num_areas > 10
            
        except Exception as e:
            logger.error(f"Ошибка детекции переходов текстуры: {e}")
            return False

    def _count_edge_artifacts(self, image: np.ndarray) -> int:
        """Подсчет артефактов краев"""
        try:
            # Детекция краев
            edges = cv2.Canny(image, 30, 100)
            
            # Морфологические операции для очистки
            kernel = np.ones((3, 3), np.uint8)
            edges_cleaned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # Подсчет контуров
            contours, _ = cv2.findContours(edges_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Фильтрация мелких контуров
            significant_contours = [c for c in contours if cv2.contourArea(c) > 50]
            
            return len(significant_contours)
            
        except Exception as e:
            logger.error(f"Ошибка подсчета артефактов краев: {e}")
            return 0

    def _detect_zone_seam_artifacts(self, zone_image: np.ndarray) -> bool:
        """Детекция швов в отдельной зоне"""
        try:
            if zone_image.size < 100:  # Слишком маленькая зона
                return False
            
            # Детекция краев
            edges = cv2.Canny(zone_image.astype(np.uint8), 30, 100)
            
            # Подсчет пикселей краев
            edge_ratio = np.sum(edges > 0) / edges.size
            
            # Если слишком много краев, возможно есть швы
            return edge_ratio > 0.1
            
        except Exception as e:
            logger.error(f"Ошибка детекции швов в зоне: {e}")
            return False

    def _detect_zone_texture_transitions(self, zone_image: np.ndarray) -> bool:
        """Детекция переходов текстуры в зоне"""
        try:
            if zone_image.size < 100:
                return False
            
            # Расчет локальной дисперсии
            kernel = np.ones((5, 5)) / 25
            local_mean = cv2.filter2D(zone_image.astype(np.float32), -1, kernel)
            local_variance = cv2.filter2D((zone_image.astype(np.float32) - local_mean)**2, -1, kernel)
            
            # Поиск областей с высокой вариацией
            high_var_threshold = np.percentile(local_variance, 90)
            high_var_areas = local_variance > high_var_threshold
            
            # Если много областей с высокой вариацией
            return np.sum(high_var_areas) / high_var_areas.size > 0.2
            
        except Exception as e:
            logger.error(f"Ошибка детекции переходов в зоне: {e}")
            return False

    def _calculate_global_fourier_signature(self, image: np.ndarray) -> np.ndarray:
        """Расчет глобальной Фурье-подписи"""
        try:
            # 2D FFT
            fft_image = np.fft.fft2(image)
            fft_shifted = np.fft.fftshift(fft_image)
            
            # Спектр мощности
            power_spectrum = np.abs(fft_shifted) ** 2
            
            # Радиальное усреднение для получения 1D подписи
            center = np.array(power_spectrum.shape) // 2
            y, x = np.ogrid[:power_spectrum.shape[0], :power_spectrum.shape[1]]
            r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
            
            # Усреднение по радиусам
            r_int = r.astype(int)
            max_radius = min(center)
            radial_profile = np.zeros(max_radius)
            
            for radius in range(max_radius):
                mask = (r_int == radius)
                if np.any(mask):
                    radial_profile[radius] = np.mean(power_spectrum[mask])
            
            return radial_profile
            
        except Exception as e:
            logger.error(f"Ошибка расчета Фурье-подписи: {e}")
            return np.zeros(100)

    def _detect_frequency_anomalies(self, fourier_signature: np.ndarray) -> List[str]:
        """Детекция частотных аномалий"""
        try:
            anomalies = []
            
            # Поиск пиков
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(fourier_signature, height=np.mean(fourier_signature) * 2)
            
            if len(peaks) > 10:
                anomalies.append("excessive_frequency_peaks")
            
            # Поиск аномально высоких частот
            high_freq_energy = np.sum(fourier_signature[len(fourier_signature)//2:])
            total_energy = np.sum(fourier_signature)
            
            if high_freq_energy / total_energy > 0.3:
                anomalies.append("high_frequency_dominance")
            
            # Поиск периодических паттернов
            autocorr = np.correlate(fourier_signature, fourier_signature, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            if len(autocorr) > 10:
                secondary_peaks = find_peaks(autocorr[10:], height=np.max(autocorr) * 0.5)
                if len(secondary_peaks[0]) > 0:
                    anomalies.append("periodic_artifacts")
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Ошибка детекции частотных аномалий: {e}")
            return []

    def calculate_material_authenticity_score(self, texture_metrics: Dict[str, Any]) -> float:
        """
        ИСПРАВЛЕНО: Расчет оценки аутентичности материала
        Согласно правкам: учет всех зон с правильными весами
        """
        # Проверка на пустые или невалидные значения
        if not texture_metrics or any(v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) for v in texture_metrics.values()):
            logger.error("[TextureAnalyzer] Пустые или невалидные метрики текстуры для расчёта аутентичности. Возвращаю 0.0.")
            return 0.0

        try:
            logger.info("Расчет оценки аутентичности материала")

            scores = []

            # ИСПРАВЛЕНО: Веса зон согласно важности
            zone_weights = {
                "forehead": 0.2,
                "left_cheek": 0.15,
                "right_cheek": 0.15,
                "nose": 0.2,
                "chin": 0.1,
                "cheek": 0.3  # Объединенные щеки
            }

            for zone_name, metrics in texture_metrics.items():
                if isinstance(metrics, ZoneTextureMetrics):
                    zone_score = self.calculate_zone_authenticity(metrics, zone_name)
                    weight = zone_weights.get(zone_name, 0.1)
                    scores.append(zone_score * weight)

            if not scores:
                return 0.0

            overall_score = np.sum(scores)

            # Нормализация к диапазону [0, 1]
            overall_score = min(1.0, max(0.0, overall_score))

            logger.info(f"Оценка аутентичности материала: {overall_score:.3f}")
            return overall_score

        except Exception as e:
            logger.error(f"Ошибка расчета аутентичности материала: {e}")
            return 0.0

    def calculate_zone_authenticity(self, zone_metrics: ZoneTextureMetrics, zone_name: str = "default") -> float:
        """
        ИСПРАВЛЕНО: Расчет аутентичности конкретной зоны
        Согласно правкам: учет всех метрик с правильными весами
        """
        try:
            score = 0.0
            baseline = self.texture_baselines.get(zone_name, self.texture_baselines["default"])

            # ИСПРАВЛЕНО: Shannon entropy (вес 0.2)
            entropy_score = sigmoid_score(
                zone_metrics.shannon_entropy,
                baseline.get("shannon_entropy_mean", 7.6),
                baseline.get("shannon_entropy_std", 0.55)
            )
            score += entropy_score * 0.2

            # ИСПРАВЛЕНО: Haralick contrast (вес 0.15)
            contrast_score = sigmoid_score(
                zone_metrics.glcm_contrast,
                baseline.get("haralick_mean_contrast", 0.17),
                baseline.get("haralick_std_contrast", 0.05),
                reverse=True  # Меньший контраст = более натурально
            )
            score += contrast_score * 0.15

            # ИСПРАВЛЕНО: LBP uniformity (вес 0.15)
            lbp_score = sigmoid_score(
                zone_metrics.lbp_uniformity,
                baseline.get("lbp_uniformity_mean", 0.89),
                baseline.get("lbp_uniformity_std", 0.055)
            )
            score += lbp_score * 0.15

            # ИСПРАВЛЕНО: Gabor energy (вес 0.2)
            gabor_score = sigmoid_score(
                zone_metrics.gabor_energy,
                baseline.get("gabor_energy_mean", 105.0),
                baseline.get("gabor_energy_std", 22.0)
            )
            score += gabor_score * 0.2

            # ИСПРАВЛЕНО: Spectral features (вес 0.15)
            spectral_score = sigmoid_score(
                zone_metrics.spectral_energy,
                50.0,  # Базовое значение
                20.0
            )
            score += spectral_score * 0.15

            # ИСПРАВЛЕНО: Homogeneity (вес 0.15)
            homogeneity_score = sigmoid_score(
                zone_metrics.glcm_homogeneity,
                0.8,  # Высокая гомогенность = натуральная кожа
                0.1
            )
            score += homogeneity_score * 0.15

            return float(max(0.0, min(1.0, score)))

        except Exception as e:
            logger.error(f"Ошибка расчета аутентичности зоны: {e}")
            return 0.0

    def classify_mask_technology_level(self, texture_data: Dict[str, ZoneTextureMetrics], 
                                    photo_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Классификация уровня технологии маски
        Согласно правкам: 5 уровней по годам с правильными параметрами
        """
        try:
            logger.info("Классификация уровня технологии маски")

            if not texture_data:
                return {"level": "Unknown", "confidence": 0.0, "reason": "Нет данных текстуры"}

            # Определение года фото
            if photo_date:
                photo_year = photo_date.year
            else:
                photo_year = 2020  # Значение по умолчанию

            # Агрегированные метрики
            avg_entropy = np.mean([
                zone.shannon_entropy for zone in texture_data.values()
                if isinstance(zone, ZoneTextureMetrics)
            ])

            avg_contrast = np.mean([
                zone.glcm_contrast for zone in texture_data.values()
                if isinstance(zone, ZoneTextureMetrics)
            ])

            avg_gabor_energy = np.mean([
                zone.gabor_energy for zone in texture_data.values()
                if isinstance(zone, ZoneTextureMetrics)
            ])

            # ИСПРАВЛЕНО: Классификация по уровням согласно MASK_DETECTION_LEVELS
            mask_levels = [
                {
                    "level": "Level1_Primitive",
                    "years": (1999, 2005),
                    "entropy_threshold": 4.2,
                    "contrast_threshold": 0.6,
                    "gabor_threshold": 50.0,
                    "description": "Примитивные маски 1999-2005"
                },
                {
                    "level": "Level2_Basic",
                    "years": (2006, 2010),
                    "entropy_threshold": 5.2,
                    "contrast_threshold": 0.4,
                    "gabor_threshold": 75.0,
                    "description": "Базовые маски 2006-2010"
                },
                {
                    "level": "Level3_Commercial",
                    "years": (2011, 2015),
                    "entropy_threshold": 6.0,
                    "contrast_threshold": 0.3,
                    "gabor_threshold": 100.0,
                    "description": "Коммерческие маски 2011-2015"
                },
                {
                    "level": "Level4_Professional",
                    "years": (2016, 2020),
                    "entropy_threshold": 6.5,
                    "contrast_threshold": 0.2,
                    "gabor_threshold": 125.0,
                    "description": "Профессиональные маски 2016-2020"
                },
                {
                    "level": "Level5_Advanced",
                    "years": (2021, 2025),
                    "entropy_threshold": 7.0,
                    "contrast_threshold": 0.15,
                    "gabor_threshold": 150.0,
                    "description": "Продвинутые маски 2021-2025"
                }
            ]

            # Поиск подходящего уровня
            detected_level = None
            confidence = 0.0

            for level_info in mask_levels:
                year_start, year_end = level_info["years"]
                
                # Проверка временного диапазона
                if year_start <= photo_year <= year_end:
                    # Проверка метрик
                    entropy_match = avg_entropy <= level_info["entropy_threshold"]
                    contrast_match = avg_contrast >= level_info["contrast_threshold"]
                    gabor_match = avg_gabor_energy <= level_info["gabor_threshold"]
                    
                    if entropy_match and contrast_match and gabor_match:
                        detected_level = level_info
                        
                        # Расчет confidence на основе близости к порогам
                        entropy_conf = 1.0 - abs(avg_entropy - level_info["entropy_threshold"]) / level_info["entropy_threshold"]
                        contrast_conf = 1.0 - abs(avg_contrast - level_info["contrast_threshold"]) / level_info["contrast_threshold"]
                        gabor_conf = 1.0 - abs(avg_gabor_energy - level_info["gabor_threshold"]) / level_info["gabor_threshold"]
                        
                        confidence = (entropy_conf + contrast_conf + gabor_conf) / 3
                        break

            if detected_level:
                result = {
                    "level": detected_level["level"],
                    "confidence": float(np.clip(confidence, 0.0, 1.0)),
                    "reason": f"Энтропия: {avg_entropy:.2f}, Контраст: {avg_contrast:.2f}, Габор: {avg_gabor_energy:.1f}",
                    "description": detected_level["description"],
                    "year_range": detected_level["years"],
                    "metrics": {
                        "entropy": avg_entropy,
                        "contrast": avg_contrast,
                        "gabor_energy": avg_gabor_energy
                    }
                }
            else:
                result = {
                    "level": "Natural_Skin",
                    "confidence": 0.8,
                    "reason": f"Метрики соответствуют натуральной коже (энтропия: {avg_entropy:.2f})",
                    "description": "Натуральная кожа",
                    "year_range": None,
                    "metrics": {
                        "entropy": avg_entropy,
                        "contrast": avg_contrast,
                        "gabor_energy": avg_gabor_energy
                    }
                }

            logger.info(f"Классификация маски: {result['level']}, confidence: {result['confidence']:.3f}")
            return result

        except Exception as e:
            logger.error(f"Ошибка классификации уровня маски: {e}")
            return {
                "level": "Error",
                "confidence": 0.0,
                "reason": f"Ошибка: {str(e)}",
                "description": "Ошибка анализа",
                "year_range": None,
                "metrics": {}
            }

    def adaptive_texture_analysis(self, image: np.ndarray, landmarks: np.ndarray,
                                lighting_conditions: Dict[str, float]) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Адаптивный анализ текстуры
        Согласно правкам: учет условий освещения и адаптация параметров
        """
        if image is None or landmarks is None or landmarks.size == 0:
            logger.warning("Невалидные данные для адаптивного анализа")
            return {}

        try:
            logger.info("Адаптивный анализ текстуры")

            # Предобработка изображения
            processed_image = image.copy()
            analysis_parameters = {}

            # ИСПРАВЛЕНО: Анализ условий освещения
            brightness = lighting_conditions.get("brightness", 0.5)  # 0-1
            contrast = lighting_conditions.get("contrast", 0.5)  # 0-1
            uniformity = lighting_conditions.get("uniformity", 0.5)  # 0-1 (1.0 - неравномерность)

            # Адаптация параметров анализа
            if brightness < 0.3 or contrast < 0.3 or uniformity < 0.4:
                logger.info("Плохие условия освещения. Применение улучшений")
                
                # CLAHE для улучшения контраста
                gray_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY) if len(processed_image.shape) == 3 else processed_image
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced_gray = clahe.apply(gray_image.astype(np.uint8))
                processed_image = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)  # Обратно в BGR
                
                analysis_parameters["preprocessing"] = "CLAHE_applied"
                
                # Адаптация параметров LBP и Gabor
                self.lbp_params["radius"] = 2
                self.gabor_params["frequencies"] = [0.2, 0.3, 0.4]
                
            elif brightness > 0.8 or contrast > 0.8:
                logger.info("Яркие условия освещения. Нормализация яркости")
                
                # Нормализация яркости
                processed_image = cv2.convertScaleAbs(processed_image, alpha=255.0/processed_image.max(), beta=0)
                analysis_parameters["preprocessing"] = "brightness_normalized"
                
            else:
                logger.info("Нормальные условия освещения. Предобработка не требуется")
                analysis_parameters["preprocessing"] = "none"

            # Основной анализ текстуры
            texture_package = self.analyze_skin_texture_by_zones(processed_image, landmarks)
            
            if texture_package:
                texture_package.extraction_method = "adaptive_texture_analysis"
                texture_package.quality_flags.append(f"lighting_preprocessing_{analysis_parameters['preprocessing']}")

            logger.info("Адаптивный анализ текстуры завершен")
            return texture_package

        except Exception as e:
            logger.error(f"Ошибка адаптивного анализа текстуры: {e}")
            return None

    def calibrate_texture_analysis_thresholds(self, historical_texture_data: List[TexturePackage]) -> None:
        """
        ИСПРАВЛЕНО: Калибровка порогов анализа текстуры
        Согласно правкам: автокалибровка на исторических данных
        """
        if not historical_texture_data:
            logger.warning("Нет исторических данных для калибровки")
            return

        try:
            logger.info(f"Калибровка порогов на {len(historical_texture_data)} образцах")

            # Сбор всех метрик по зонам
            zone_metrics_aggregated = defaultdict(list)
            
            for package in historical_texture_data:
                if isinstance(package, TexturePackage) and package.zone_metrics:
                    for zone_name, zone_metrics in package.zone_metrics.items():
                        if isinstance(zone_metrics, ZoneTextureMetrics):
                            zone_metrics_aggregated[zone_name].append({
                                'shannon_entropy': zone_metrics.shannon_entropy,
                                'glcm_contrast': zone_metrics.glcm_contrast,
                                'gabor_energy': zone_metrics.gabor_energy,
                                'lbp_uniformity': zone_metrics.lbp_uniformity,
                                'glcm_homogeneity': zone_metrics.glcm_homogeneity
                            })

            # ИСПРАВЛЕНО: Обновление базовых линий для каждой зоны
            new_baselines = self.texture_baselines.copy()

            for zone_name, metrics_list in zone_metrics_aggregated.items():
                if len(metrics_list) >= 5:  # Минимум 5 образцов для статистики
                    # Агрегация метрик
                    entropies = [m['shannon_entropy'] for m in metrics_list]
                    contrasts = [m['glcm_contrast'] for m in metrics_list]
                    gabor_energies = [m['gabor_energy'] for m in metrics_list]
                    lbp_uniformities = [m['lbp_uniformity'] for m in metrics_list]
                    homogeneities = [m['glcm_homogeneity'] for m in metrics_list]

                    # Обновление baseline для зоны
                    if zone_name not in new_baselines:
                        new_baselines[zone_name] = {}

                    new_baselines[zone_name].update({
                        "shannon_entropy_mean": float(np.mean(entropies)),
                        "shannon_entropy_std": float(np.std(entropies)),
                        "haralick_mean_contrast": float(np.mean(contrasts)),
                        "haralick_std_contrast": float(np.std(contrasts)),
                        "gabor_energy_mean": float(np.mean(gabor_energies)),
                        "gabor_energy_std": float(np.std(gabor_energies)),
                        "lbp_uniformity_mean": float(np.mean(lbp_uniformities)),
                        "lbp_uniformity_std": float(np.std(lbp_uniformities)),
                        "homogeneity_mean": float(np.mean(homogeneities)),
                        "homogeneity_std": float(np.std(homogeneities))
                    })

                    logger.info(f"Калибровка зоны {zone_name}: {len(metrics_list)} образцов")

            # Обновление общих baseline
            all_entropies = []
            all_contrasts = []
            all_gabor_energies = []
            all_lbp_uniformities = []

            for metrics_list in zone_metrics_aggregated.values():
                all_entropies.extend([m['shannon_entropy'] for m in metrics_list])
                all_contrasts.extend([m['glcm_contrast'] for m in metrics_list])
                all_gabor_energies.extend([m['gabor_energy'] for m in metrics_list])
                all_lbp_uniformities.extend([m['lbp_uniformity'] for m in metrics_list])

            if all_entropies:
                new_baselines["default"].update({
                    "shannon_entropy_mean": float(np.mean(all_entropies)),
                    "shannon_entropy_std": float(np.std(all_entropies)),
                    "haralick_mean_contrast": float(np.mean(all_contrasts)),
                    "haralick_std_contrast": float(np.std(all_contrasts)),
                    "gabor_energy_mean": float(np.mean(all_gabor_energies)),
                    "gabor_energy_std": float(np.std(all_gabor_energies)),
                    "lbp_uniformity_mean": float(np.mean(all_lbp_uniformities)),
                    "lbp_uniformity_std": float(np.std(all_lbp_uniformities))
                })

            self.texture_baselines = new_baselines
            self.calibrated = True

            # Сохранение обновленных baseline
            self._save_texture_baselines()

            logger.info("Калибровка порогов завершена успешно")

        except Exception as e:
            logger.error(f"Ошибка калибровки порогов: {e}")

    def _save_texture_baselines(self):
        """Сохранение baseline текстуры в файл"""
        try:
            baseline_file = self.cache_dir / "texture_baselines_calibrated.json"
            with open(baseline_file, 'w', encoding='utf-8') as f:
                json.dump(self.texture_baselines, f, indent=2, ensure_ascii=False)
            logger.info(f"Калиброванные baseline сохранены: {baseline_file}")
        except Exception as e:
            logger.error(f"Ошибка сохранения baseline: {e}")

    def detect_texture_transition_artifacts(self, image: np.ndarray, zone_mask: np.ndarray) -> Dict[str, Any]:
        """
        Детектирование швов/краёв маски по Canny+Hough (по ТЗ)
        Возвращает: есть ли артефакт, длина края, координаты
        """
        try:
            logger.info("Детектирование швов/краёв маски (Canny+Hough)")

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            masked = cv2.bitwise_and(gray, gray, mask=zone_mask)
            
            edges = cv2.Canny(masked, 
                            TEXTURE_ANALYSIS_THRESHOLDS["canny_threshold1"], 
                            TEXTURE_ANALYSIS_THRESHOLDS["canny_threshold2"])
            
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=10)

            edge_length = 0
            coords = []
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    edge_length += length
                    coords.append(((x1, y1), (x2, y2)))

            h = zone_mask.shape[0]
            artifact = edge_length > 0.1 * h  # >10% высоты лица

            return {
                "artifact": artifact, 
                "edge_length": float(edge_length), 
                "coords": coords,
                "edge_density": float(edge_length / (h * zone_mask.shape[1])) if h > 0 else 0.0
            }

        except Exception as e:
            logger.error(f"Ошибка detect_texture_transition_artifacts: {e}")
            return {"artifact": False, "edge_length": 0.0, "coords": [], "edge_density": 0.0}

    def pore_circularity_score(self, region_img: np.ndarray) -> float:
        """
        Оценка округлости пор (по ТЗ: area, perimeter, circularity)
        """
        try:
            logger.info("Расчет pore_circularity_score")

            gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY) if region_img.ndim == 3 else region_img
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            
            thresh = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 
                TEXTURE_ANALYSIS_THRESHOLDS["pore_block_size"], 
                int(TEXTURE_ANALYSIS_THRESHOLDS["pore_offset"] * 255)
            )
            
            # Морфологические операции
            open_kernel = disk(TEXTURE_ANALYSIS_THRESHOLDS["pore_selem_open_disk_size"])
            close_kernel = disk(TEXTURE_ANALYSIS_THRESHOLDS["pore_selem_close_disk_size"])
            
            opened = binary_opening(thresh > 0, open_kernel)
            closed = binary_closing(opened, close_kernel)
            
            labeled = label(closed)
            props = regionprops(labeled)

            circularities = []
            for prop in props:
                if (TEXTURE_ANALYSIS_THRESHOLDS["pore_min_area"] < prop.area < 
                    TEXTURE_ANALYSIS_THRESHOLDS["pore_max_area"]):
                    perim = prop.perimeter if prop.perimeter > 0 else 1
                    circ = 4 * math.pi * prop.area / (perim ** 2)
                    circularities.append(circ)

            if not circularities:
                return 0.0

            return float(np.mean(circularities))

        except Exception as e:
            logger.error(f"Ошибка pore_circularity_score: {e}")
            return 0.0

    def _analyze_pore_distribution(self, region_img: np.ndarray) -> Dict[str, float]:
        """
        Анализ распределения пор: число, средний размер, плотность, округлость
        """
        try:
            logger.info("Анализ распределения пор")

            gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY) if region_img.ndim == 3 else region_img
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            
            thresh = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                TEXTURE_ANALYSIS_THRESHOLDS["pore_block_size"],
                int(TEXTURE_ANALYSIS_THRESHOLDS["pore_offset"] * 255)
            )
            
            # Морфологические операции
            open_kernel = disk(TEXTURE_ANALYSIS_THRESHOLDS["pore_selem_open_disk_size"])
            opened = binary_opening(thresh > 0, open_kernel)
            
            labeled = label(opened)
            props = regionprops(labeled)

            areas = []
            circularities = []
            
            for prop in props:
                if (TEXTURE_ANALYSIS_THRESHOLDS["pore_min_area"] < prop.area < 
                    TEXTURE_ANALYSIS_THRESHOLDS["pore_max_area"]):
                    areas.append(prop.area)
                    if prop.perimeter > 0:
                        circ = 4 * math.pi * prop.area / (prop.perimeter ** 2)
                        circularities.append(circ)

            total_pixels = region_img.shape[0] * region_img.shape[1]
            
            return {
                "pore_count": len(areas),
                "pore_mean_area": float(np.mean(areas)) if areas else 0.0,
                "pore_density": float(len(areas)) / total_pixels if total_pixels > 0 else 0.0,
                "pore_circularity_score": float(np.mean(circularities)) if circularities else 0.0,
                "pore_total_area": float(np.sum(areas)) if areas else 0.0,
                "pore_area_std": float(np.std(areas)) if areas else 0.0
            }

        except Exception as e:
            logger.error(f"Ошибка _analyze_pore_distribution: {e}")
            return {
                "pore_count": 0, 
                "pore_mean_area": 0.0, 
                "pore_density": 0.0, 
                "pore_circularity_score": 0.0,
                "pore_total_area": 0.0,
                "pore_area_std": 0.0
            }

    def _analyze_micro_wrinkles(self, region_img: np.ndarray) -> Dict[str, float]:
        """
        Анализ микроморщин: число, длина, средний угол
        """
        try:
            logger.info("Анализ микроморщин")

            gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY) if region_img.ndim == 3 else region_img
            
            # Градиентный анализ
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
            
            mask = mag > gradient_magnitude_threshold
            
            lines = cv2.HoughLinesP(
                mask.astype(np.uint8) * 255, 1, np.pi/180, 
                threshold=10, minLineLength=5, maxLineGap=2
            )

            count = 0
            total_length = 0
            angles = []
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    total_length += length
                    angle = np.arctan2(y2-y1, x2-x1)
                    angles.append(angle)
                    count += 1

            return {
                "wrinkle_count": count,
                "wrinkle_total_length": float(total_length),
                "wrinkle_mean_angle": float(np.mean(angles)) if angles else 0.0,
                "wrinkle_density": float(total_length) / (region_img.shape[0] * region_img.shape[1]) if region_img.size > 0 else 0.0,
                "wrinkle_angle_std": float(np.std(angles)) if angles else 0.0
            }

        except Exception as e:
            logger.error(f"Ошибка _analyze_micro_wrinkles: {e}")
            return {
                "wrinkle_count": 0, 
                "wrinkle_total_length": 0.0, 
                "wrinkle_mean_angle": 0.0,
                "wrinkle_density": 0.0,
                "wrinkle_angle_std": 0.0
            }

    def calculate_spectral_material_signature(self, texture_regions: List[np.ndarray]) -> Dict[str, float]:
        """
        Агрегация спектральных признаков по зонам (по ТЗ)
        """
        try:
            logger.info("Агрегация спектральных признаков по зонам")

            centroids = []
            rolloffs = []
            dom_freqs = []
            spectral_energies = []
            
            for region in texture_regions:
                if region is not None and region.size > 0:
                    spectrum = self.calculate_fourier_spectrum(region)
                    centroids.append(spectrum.get("spectral_centroid", 0.0))
                    rolloffs.append(spectrum.get("spectral_rolloff", 0.0))
                    dom_freqs.append(spectrum.get("dominant_frequency", 0.0))
                    
                    # Дополнительная спектральная энергия
                    fourier = np.fft.fft2(region)
                    spectral_energy = np.sum(np.abs(fourier) ** 2)
                    spectral_energies.append(spectral_energy)

            return {
                "mean_spectral_centroid": float(np.mean(centroids)) if centroids else 0.0,
                "mean_spectral_rolloff": float(np.mean(rolloffs)) if rolloffs else 0.0,
                "mean_dominant_frequency": float(np.mean(dom_freqs)) if dom_freqs else 0.0,
                "mean_spectral_energy": float(np.mean(spectral_energies)) if spectral_energies else 0.0,
                "spectral_centroid_std": float(np.std(centroids)) if centroids else 0.0,
                "spectral_rolloff_std": float(np.std(rolloffs)) if rolloffs else 0.0,
                "regions_analyzed": len(texture_regions)
            }

        except Exception as e:
            logger.error(f"Ошибка calculate_spectral_material_signature: {e}")
            return {
                "mean_spectral_centroid": 0.0, 
                "mean_spectral_rolloff": 0.0, 
                "mean_dominant_frequency": 0.0,
                "mean_spectral_energy": 0.0,
                "spectral_centroid_std": 0.0,
                "spectral_rolloff_std": 0.0,
                "regions_analyzed": 0
            }

    def detect_mask_by_texture_analysis(self, texture_package: TexturePackage) -> MaskDetectionResult:
        """
        Детекция маски на основе текстурного анализа
        
        Args:
            texture_package: Пакет результатов текстурного анализа
            
        Returns:
            Результат детекции маски
        """
        try:
            evidence_factors = []
            suspicious_zones = []
            confidence_score = 0.0
            
            # Анализ общих показателей
            if texture_package.overall_entropy < 5.5:
                evidence_factors.append("low_overall_entropy")
                confidence_score += 0.3
            
            if texture_package.seam_artifacts_detected:
                evidence_factors.append("seam_artifacts_detected")
                confidence_score += 0.4
            
            if texture_package.texture_transitions_detected:
                evidence_factors.append("texture_transitions_detected")
                confidence_score += 0.3
            
            # Анализ зон
            for zone_name, metrics in texture_package.zone_metrics.items():
                zone_suspicious = False
                
                # Низкая энтропия в зоне
                if metrics.shannon_entropy < 5.0:
                    zone_suspicious = True
                    confidence_score += 0.1
                
                # Аномальные Габор-отклики
                if metrics.gabor_energy < 50 or metrics.gabor_energy > 200:
                    zone_suspicious = True
                    confidence_score += 0.1
                
                # Артефакты в зоне
                if metrics.seam_artifacts_detected or metrics.texture_transitions_detected:
                    zone_suspicious = True
                    confidence_score += 0.15
                
                if zone_suspicious:
                    suspicious_zones.append(zone_name)
            
            # Анализ уровня технологии
            if texture_package.mask_level >= 4:
                evidence_factors.append("high_technology_level")
                confidence_score += 0.2
            
            # Нормализация confidence_score
            confidence_score = min(1.0, confidence_score)
            
            # Определение результата
            is_mask_detected = confidence_score > 0.5 or len(suspicious_zones) >= 3
            
            return MaskDetectionResult(
                is_mask_detected=is_mask_detected,
                confidence_score=confidence_score,
                mask_technology_level=texture_package.mask_level,
                evidence_factors=evidence_factors,
                suspicious_zones=suspicious_zones
            )
            
        except Exception as e:
            logger.error(f"Ошибка детекции маски: {e}")
            return MaskDetectionResult(
                is_mask_detected=False,
                confidence_score=0.0,
                mask_technology_level=1,
                evidence_factors=["analysis_error"],
                suspicious_zones=[]
            )

    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики обработки
        
        Returns:
            Словарь со статистикой
        """
        stats = self.processing_stats.copy()
        
        if stats['total_processed'] > 0:
            stats['success_rate'] = stats['successful_analyses'] / stats['total_processed']
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_processed']
            stats['mask_detection_rate'] = stats['mask_detections'] / stats['total_processed']
        else:
            stats['success_rate'] = 0.0
            stats['cache_hit_rate'] = 0.0
            stats['mask_detection_rate'] = 0.0
        
        # Информация о кэше
        stats['cache_info'] = {
            'texture_cache_size': len(self.texture_cache),
            'gabor_kernels_cached': len(self.gabor_kernels_cache)
        }
        
        # Информация о калибровке
        stats['calibration_info'] = {
            'is_calibrated': self.calibrated,
            'baseline_zones': list(self.texture_baselines.keys()),
            'lbp_params': self.lbp_params,
            'gabor_params': {
                'orientations_count': len(self.gabor_params['orientations']),
                'frequencies_count': len(self.gabor_params['frequencies'])
            }
        }
        
        # Информация о памяти
        process = psutil.Process()
        memory_info = process.memory_info()
        stats['memory_usage_mb'] = memory_info.rss / (1024 * 1024)
        
        return stats

    def clear_cache(self):
        """Очистка кэша результатов"""
        try:
            self.texture_cache.clear()
            logger.info("Кэш TextureAnalyzer очищен")
            
        except Exception as e:
            logger.error(f"Ошибка очистки кэша: {e}")

    def save_cache(self, cache_filename: str = "texture_cache.pkl"):
        """Сохранение кэша на диск"""
        try:
            cache_path = self.cache_dir / cache_filename
            
            cache_data = {
                'texture_cache': self.texture_cache,
                'processing_stats': self.processing_stats,
                'calibrated': self.calibrated,
                'texture_baselines': self.texture_baselines
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"Кэш сохранен: {cache_path}")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения кэша: {e}")

    def load_cache(self, cache_filename: str = "texture_cache.pkl") -> bool:
        """Загрузка кэша с диска"""
        try:
            cache_path = self.cache_dir / cache_filename
            
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                
                self.texture_cache = cache_data.get('texture_cache', {})
                self.processing_stats.update(cache_data.get('processing_stats', {}))
                self.calibrated = cache_data.get('calibrated', False)
                
                # Обновляем baseline если есть калиброванные данные
                if 'texture_baselines' in cache_data:
                    self.texture_baselines.update(cache_data['texture_baselines'])
                
                logger.info(f"Кэш загружен: {cache_path}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Ошибка загрузки кэша: {e}")
            return False

    # === ФУНКЦИИ САМОТЕСТИРОВАНИЯ ===

    def self_test():
        """Самотестирование модуля texture_analyzer"""
        try:
            logger.info("Запуск самотестирования texture_analyzer...")
            
            # Создание экземпляра анализатора
            analyzer = TextureAnalyzer()
            
            # Создание тестового изображения и ландмарков
            test_image = np.random.randint(0, 255, (800, 800, 3), dtype=np.uint8)
            test_landmarks = np.random.rand(68, 3) * 100
            
            # Тест создания маски зоны
            zone_config = FACE_ZONES['forehead']
            test_coords = np.random.rand(10, 2) * 100
            mask = create_zone_mask(test_image.shape[:2], test_coords)
            assert mask is not None, "Маска зоны не создана"
            
            # Тест расчета энтропии
            gray_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
            entropy = analyzer.calculate_shannon_entropy(gray_image.flatten())
            assert 0.0 <= entropy <= 10.0, "Неверный диапазон энтропии"
            
            # Тест LBP характеристик
            test_zone = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
            lbp_features = analyzer.calculate_lbp_features(test_zone)
            assert 'lbp_uniformity' in lbp_features, "Отсутствуют LBP характеристики"
            
            # Тест Габор откликов
            gabor_features = analyzer.calculate_gabor_responses(test_zone.astype(np.float32))
            assert 'gabor_total_energy' in gabor_features, "Отсутствуют Габор характеристики"
            
            # Тест статистики
            stats = analyzer.get_processing_statistics()
            assert 'success_rate' in stats, "Отсутствует статистика"
            
            logger.info("Самотестирование texture_analyzer завершено успешно")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка самотестирования: {e}")
            return False

    # === ИНИЦИАЛИЗАЦИЯ ===

    if __name__ == "__main__":
        # Запуск самотестирования при прямом вызове модуля
        success = self_test()
        if success:
            print("✅ Модуль texture_analyzer работает корректно")
            
            # Демонстрация основной функциональности
            analyzer = TextureAnalyzer()
            print(f"📊 Результатов в кэше: {len(analyzer.texture_cache)}")
            print(f"🔧 Габор-ядер в кэше: {len(analyzer.gabor_kernels_cache)}")
            print(f"📏 LBP параметры: radius={analyzer.lbp_params['radius']}, points={analyzer.lbp_params['n_points']}")
            print(f"🎯 Габор частоты: {analyzer.gabor_params['frequencies']}")
            print(f"💾 Кэш-директория: {analyzer.cache_dir}")
            print(f"🎛️ Калиброван: {analyzer.calibrated}")
        else:
            print("❌ Обнаружены ошибки в модуле texture_analyzer")
            exit(1)