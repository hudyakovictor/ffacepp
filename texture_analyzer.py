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

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('logs/textureanalyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Импорт конфигурации
try:
    from core_config import (
        MASK_DETECTION_LEVELS, BREAKTHROUGH_YEARS, CRITICAL_THRESHOLDS,
        CONFIG_DIR, CACHE_DIR, ERROR_CODES
    )
    logger.info("Конфигурация успешно импортирована")
except ImportError as e:
    logger.error(f"Ошибка импорта конфигурации: {e}")
    # Значения по умолчанию
    MASK_DETECTION_LEVELS = {}
    BREAKTHROUGH_YEARS = [2008, 2014, 2019, 2022]
    CRITICAL_THRESHOLDS = {"texture_authenticity_threshold": 0.7}
    CONFIG_DIR = Path("configs")
    CACHE_DIR = Path("cache")
    ERROR_CODES = {"E002": "LOW_QUALITY_IMAGE"}

# ==================== КОНСТАНТЫ АНАЛИЗА ТЕКСТУРЫ ====================

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

# Зоны лица для анализа
FACE_ZONES = {
    "forehead": "Лоб",
    "left_cheek": "Левая щека", 
    "right_cheek": "Правая щека",
    "nose": "Нос",
    "chin": "Подбородок"
}

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def calculate_mean_std(data: np.ndarray) -> Tuple[float, float]:
    """Расчет среднего и стандартного отклонения"""
    if data.size == 0:
        return 0.0, 0.0
    return float(np.mean(data)), float(np.std(data))

def create_zone_mask(image_shape: Tuple[int, int], zone_coords: np.ndarray) -> np.ndarray:
    """Создание маски для зоны лица"""
    try:
        mask = np.zeros(image_shape, dtype=np.uint8)
        if zone_coords.size > 0:
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

# ==================== ОСНОВНОЙ КЛАСС ====================

class TextureAnalyzer:
    """
    Анализатор текстуры кожи с полной функциональностью
    ИСПРАВЛЕНО: Все критические ошибки согласно правкам
    """
    
    def __init__(self):
        """Инициализация анализатора текстуры"""
        logger.info("Инициализация TextureAnalyzer")
        
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
        self.analysis_cache = {}
        
        # Флаг калибровки
        self.calibrated = False
        
        logger.info("TextureAnalyzer инициализирован")

    def load_texture_baselines(self) -> Dict[str, Dict[str, float]]:
        """
        ИСПРАВЛЕНО: Загрузка базовых линий текстуры из JSON
        Согласно правкам: правильная структура baseline данных
        """
        baseline_file = CONFIG_DIR / "texture_baselines.json"
        
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
        if landmarks.size == 0 or landmarks.shape[0] < 68:
            logger.warning("Недостаточно ландмарок для определения зон кожи")
            return {}
        
        try:
            logger.info("Определение зон кожи лица")
            
            zones = {}
            
            # Вычисление высоты лица для адаптивного расширения лба (пункт 21)
            # Используем точки 8 (подбородок) и 27 (переносица) для определения высоты лица
            # face_height = np.linalg.norm(landmarks[8] - landmarks[27])
            
            # Альтернатива: использовать BBox всего лица для более надежной высоты
            min_x, min_y = np.min(landmarks[:, 0]), np.min(landmarks[:, 1])
            max_x, max_y = np.max(landmarks[:, 0]), np.max(landmarks[:, 1])
            face_bbox_height = max_y - min_y
            
            # Определяем относительное расширение (например, 15% от высоты лица)
            forehead_expansion_px = int(face_bbox_height * 0.15) # 15% от высоты BBox лица
            
            # Лоб (forehead) - точки 17-26
            forehead_points = landmarks[17:27]  # Брови
            # Расширение области лба вверх (ИСПРАВЛЕНО: пункт 21)
            forehead_top = forehead_points.copy()
            forehead_top[:, 1] -= forehead_expansion_px  # Смещение вверх
            forehead_zone = np.vstack([forehead_top, forehead_points[::-1]])
            zones["forehead"] = forehead_zone[:, :2].astype(np.int32)
            
            # Левая щека (left_cheek) - точки 1-5, 31, 41
            left_cheek_points = np.vstack([
                landmarks[1:6],    # Левая сторона лица
                landmarks[[31]],   # Нос
                landmarks[[41]]    # Левый глаз
            ])
            zones["left_cheek"] = left_cheek_points[:, :2].astype(np.int32)
            
            # Правая щека (right_cheek) - точки 11-15, 35, 46
            right_cheek_points = np.vstack([
                landmarks[11:16],  # Правая сторона лица
                landmarks[[35]],   # Нос
                landmarks[[46]]    # Правый глаз
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

    def analyze_skin_texture_by_zones(self, image: np.ndarray, landmarks: np.ndarray) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Анализ текстуры кожи по зонам
        Согласно правкам: robust analysis for 5 zones
        """
        try:
            logger.info(f"Анализ текстуры кожи по зонам. Изображение: {image.shape}")
            
            # ИСПРАВЛЕНО: Конвертация типа изображения для OpenCV
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)

            # Преобразование в оттенки серого
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            texture_metrics = {}
            
            # Определение зон кожи
            skin_zones = self.define_skin_zones(landmarks, image.shape[:2])
            
            for zone_name, zone_coords in skin_zones.items():
                if zone_coords.size == 0:
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
                if y_coords.size == 0 or x_coords.size == 0:
                    continue
                
                # Bounding box зоны
                min_y, max_y = np.min(y_coords), np.max(y_coords)
                min_x, max_x = np.min(x_coords), np.max(x_coords)
                
                # Обрезка региона
                cropped_zone_region = zone_region_colored[min_y:max_y+1, min_x:max_x+1]
                cropped_mask = mask[min_y:max_y+1, min_x:max_x+1]
                
                if cropped_zone_region.size == 0:
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
                
                if valid_pixels_colored.size == 0 or valid_pixels_gray.size == 0:
                    logger.warning(f"Зона {zone_name} не содержит валидных пикселей")
                    continue
                
                # Анализ текстуры зоны
                zone_analysis = self.analyze_zone_texture(zone_region_gray, valid_pixels_gray)
                texture_metrics[zone_name] = zone_analysis
            
            logger.info(f"Анализ текстуры завершен для {len(texture_metrics)} зон")
            return texture_metrics
            
        except Exception as e:
            logger.error(f"Ошибка анализа текстуры по зонам: {e}")
            return {}

    def analyze_zone_texture(self, zone_region_2d: np.ndarray, valid_pixels_1d: np.ndarray) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Анализ текстуры конкретной зоны
        Согласно правкам: все метрики Haralick, LBP, Gabor, Fourier
        """
        try:
            logger.info("Анализ текстуры зоны")
            
            zone_metrics = {}
            
            # ИСПРАВЛЕНО: Shannon entropy
            zone_metrics["shannon_entropy"] = self.calculate_shannon_entropy(valid_pixels_1d)
            
            # ИСПРАВЛЕНО: LBP features
            lbp_features = self.calculate_lbp_features(zone_region_2d)
            zone_metrics.update(lbp_features)
            
            # ИСПРАВЛЕНО: Gabor responses
            gabor_responses = self.calculate_gabor_responses(zone_region_2d)
            zone_metrics.update(gabor_responses)
            
            # ИСПРАВЛЕНО: Fourier spectrum
            fourier_spectrum = self.calculate_fourier_spectrum(zone_region_2d)
            zone_metrics.update(fourier_spectrum)
            
            # ИСПРАВЛЕНО: Haralick features
            haralick_features = self.calculate_haralick_features(zone_region_2d)
            zone_metrics.update(haralick_features)
            
            return zone_metrics
            
        except Exception as e:
            logger.error(f"Ошибка анализа текстуры зоны: {e}")
            return {}

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
        if pixel_values.size == 0:
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
            if image_region.dtype != np.float32 and image_region.dtype != np.float64:
                image_region = image_region.astype(np.float32) / 255.0
            orientations = self.gabor_params["orientations"]
            frequencies = self.gabor_params["frequencies"]
            for theta in orientations:
                for freq in frequencies:
                    current_sigma = 0.56 / freq if freq > 0 else 1.0
                    kernel = np.real(gabor_kernel(
                        frequency=freq, 
                        theta=np.radians(theta), 
                        sigma_x=current_sigma,
                        sigma_y=current_sigma
                    ))
                    filtered_image = convolve(image_region, kernel, mode='nearest')
                    gabor_results[f"gabor_theta{theta:.1f}_freq{freq}"] = {
                        "mean": float(np.mean(filtered_image)),
                        "std": float(np.std(filtered_image)),
                        "energy": float(np.sum(filtered_image**2))
                    }
            all_energies = [response["energy"] for response in gabor_results.values()]
            gabor_results["gabor_total_energy"] = float(np.sum(all_energies))
            gabor_results["gabor_mean_energy"] = float(np.mean(all_energies))
            gabor_results["gabor_energy_std"] = float(np.std(all_energies))
            logger.info(f"Gabor responses рассчитаны для {len(orientations)}x{len(frequencies)} комбинаций")
            return gabor_results
        except Exception as e:
            logger.error(f"Ошибка расчета Gabor responses: {e}")
            return {}

    def calculate_fourier_spectrum(self, image_region: np.ndarray) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Расчет Fourier spectrum + FFT пики 0.15, 0.3, 0.6
        """
        if image_region.size == 0 or image_region.ndim == 0:
            logger.warning("calculate_fourier_spectrum: Пустой или невалидный регион изображения")
            return {
                "spectral_centroid": 0.0,
                "spectral_rolloff": 0.0,
                "dominant_frequency": 0,
                "fft_peaks": {0.15: 0.0, 0.3: 0.0, 0.6: 0.0}
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
                    "fft_peaks": {0.15: 0.0, 0.3: 0.0, 0.6: 0.0}
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
                "fft_peaks": peaks
            }
            logger.info("Fourier spectrum рассчитан успешно")
            return spectral_metrics
        except Exception as e:
            logger.error(f"Ошибка расчета Fourier spectrum: {e}")
            return {
                "spectral_centroid": 0.0,
                "spectral_rolloff": 0.0,
                "dominant_frequency": 0,
                "fft_peaks": {0.15: 0.0, 0.3: 0.0, 0.6: 0.0}
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
            
            # ИСПРАВLЕНО: Извлечение Haralick features
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

    def calculate_material_authenticity_score(self, texture_metrics: Dict[str, Any]) -> float:
        """
        ИСПРАВЛЕНО: Расчет оценки аутентичности материала
        Согласно правкам: учет всех зон с правильными весами
        """
        if not texture_metrics:
            logger.warning("Пустые метрики текстуры для расчета аутентичности")
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

    def calculate_zone_authenticity(self, zone_metrics: Dict[str, Any], zone_name: str = "default") -> float:
        """
        ИСПРАВЛЕНО: Расчет аутентичности конкретной зоны
        Согласно правкам: учет всех метрик с правильными весами
        """
        try:
            score = 0.0
            baseline = self.texture_baselines.get(zone_name, self.texture_baselines["default"])
            
            # ИСПРАВЛЕНО: Shannon entropy (вес 0.2)
            if "shannon_entropy" in zone_metrics:
                entropy_score = sigmoid_score(
                    zone_metrics["shannon_entropy"],
                    baseline.get("shannon_entropy_mean", 7.6),
                    baseline.get("shannon_entropy_std", 0.55)
                )
                score += entropy_score * 0.2
            
            # ИСПРАВЛЕНО: Haralick contrast (вес 0.15)
            if "contrast" in zone_metrics:
                contrast_score = sigmoid_score(
                    zone_metrics["contrast"],
                    baseline.get("haralick_mean_contrast", 0.17),
                    baseline.get("haralick_std_contrast", 0.05),
                    reverse=True  # Меньший контраст = более натурально
                )
                score += contrast_score * 0.15
            
            # ИСПРАВЛЕНО: LBP uniformity (вес 0.15)
            if "lbp_uniformity" in zone_metrics:
                lbp_score = sigmoid_score(
                    zone_metrics["lbp_uniformity"],
                    baseline.get("lbp_uniformity_mean", 0.89),
                    baseline.get("lbp_uniformity_std", 0.055)
                )
                score += lbp_score * 0.15
            
            # ИСПРАВЛЕНО: Gabor energy (вес 0.2)
            if "gabor_mean_energy" in zone_metrics:
                gabor_score = sigmoid_score(
                    zone_metrics["gabor_mean_energy"],
                    baseline.get("gabor_energy_mean", 105.0),
                    baseline.get("gabor_energy_std", 22.0)
                )
                score += gabor_score * 0.2
            
            # ИСПРАВЛЕНО: Spectral features (вес 0.15)
            if "spectral_centroid" in zone_metrics:
                spectral_score = sigmoid_score(
                    zone_metrics["spectral_centroid"],
                    50.0,  # Базовое значение
                    20.0
                )
                score += spectral_score * 0.15
            
            # ИСПРАВЛЕНО: Homogeneity (вес 0.15)
            if "homogeneity" in zone_metrics:
                homogeneity_score = sigmoid_score(
                    zone_metrics["homogeneity"],
                    0.8,  # Высокая гомогенность = натуральная кожа
                    0.1
                )
                score += homogeneity_score * 0.15
            
            return score
            
        except Exception as e:
            logger.error(f"Ошибка расчета аутентичности зоны: {e}")
            return 0.0

    def classify_mask_technology_level(self, texture_data: Dict[str, Any], photo_date: Optional[datetime] = None) -> Dict[str, Any]:
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
                zone.get("shannon_entropy", 0) 
                for zone in texture_data.values() 
                if isinstance(zone, dict)
            ])
            
            avg_contrast = np.mean([
                zone.get("contrast", 0) 
                for zone in texture_data.values() 
                if isinstance(zone, dict)
            ])
            
            # ИСПРАВЛЕНО: Классификация по уровням согласно MASK_DETECTION_LEVELS
            mask_levels = [
                {
                    "level": "Level1_Primitive",
                    "years": (1999, 2005),
                    "entropy_threshold": 4.2,
                    "contrast_threshold": 0.6,
                    "description": "Примитивные маски 1999-2005"
                },
                {
                    "level": "Level2_Basic", 
                    "years": (2006, 2010),
                    "entropy_threshold": 5.2,
                    "contrast_threshold": 0.4,
                    "description": "Базовые маски 2006-2010"
                },
                {
                    "level": "Level3_Commercial",
                    "years": (2011, 2015),
                    "entropy_threshold": 6.0,
                    "contrast_threshold": 0.3,
                    "description": "Коммерческие маски 2011-2015"
                },
                {
                    "level": "Level4_Professional",
                    "years": (2016, 2020),
                    "entropy_threshold": 6.5,
                    "contrast_threshold": 0.2,
                    "description": "Профессиональные маски 2016-2020"
                },
                {
                    "level": "Level5_Advanced",
                    "years": (2021, 2025),
                    "entropy_threshold": 7.0,
                    "contrast_threshold": 0.15,
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
                    
                    if entropy_match and contrast_match:
                        detected_level = level_info
                        # Расчет confidence на основе близости к порогам
                        entropy_conf = 1.0 - abs(avg_entropy - level_info["entropy_threshold"]) / level_info["entropy_threshold"]
                        contrast_conf = 1.0 - abs(avg_contrast - level_info["contrast_threshold"]) / level_info["contrast_threshold"]
                        confidence = (entropy_conf + contrast_conf) / 2
                        break
            
            if detected_level:
                result = {
                    "level": detected_level["level"],
                    "confidence": float(np.clip(confidence, 0.0, 1.0)),
                    "reason": f"Энтропия: {avg_entropy:.2f}, Контраст: {avg_contrast:.2f}",
                    "description": detected_level["description"],
                    "year_range": detected_level["years"]
                }
            else:
                result = {
                    "level": "Natural_Skin",
                    "confidence": 0.8,
                    "reason": f"Метрики соответствуют натуральной коже (энтропия: {avg_entropy:.2f})",
                    "description": "Натуральная кожа",
                    "year_range": None
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
                "year_range": None
            }

    def adaptive_texture_analysis(self, image: np.ndarray, landmarks: np.ndarray, 
                                lighting_conditions: Dict[str, float]) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Адаптивный анализ текстуры
        Согласно правкам: учет условий освещения и адаптация параметров
        """
        if image is None or landmarks.size == 0:
            logger.warning("Невалидные данные для адаптивного анализа")
            return {}
        
        try:
            logger.info("Адаптивный анализ текстуры")
            
            # Предобработка изображения
            processed_image = image.copy()
            analysis_parameters = {}
            
            # ИСПРАВЛЕНО: Анализ условий освещения
            brightness = lighting_conditions.get("brightness", 0.5)  # 0-1
            contrast = lighting_conditions.get("contrast", 0.5)      # 0-1  
            uniformity = lighting_conditions.get("uniformity", 0.5)  # 0-1 (1.0 - неравномерность)
            
            # Адаптация параметров анализа
            if brightness < 0.3 or contrast < 0.3 or uniformity < 0.4:
                logger.info("Плохие условия освещения. Применение улучшений")
                
                # CLAHE для улучшения контраста
                gray_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY) if len(processed_image.shape) == 3 else processed_image
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                processed_image = clahe.apply(gray_image.astype(np.uint8))
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)  # Обратно в BGR
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
            texture_results = self.analyze_skin_texture_by_zones(processed_image, landmarks)
            texture_results["adaptive_parameters"] = analysis_parameters
            
            logger.info("Адаптивный анализ текстуры завершен")
            return texture_results
            
        except Exception as e:
            logger.error(f"Ошибка адаптивного анализа текстуры: {e}")
            return {}

    def calibrate_texture_analysis_thresholds(self, historical_texture_data: List[Dict[str, Any]]) -> None:
        """
        ИСПРАВЛЕНО: Калибровка порогов анализа текстуры
        Согласно правкам: автокалибровка на исторических данных
        """
        if not historical_texture_data:
            logger.warning("Нет исторических данных для калибровки")
            return
        
        try:
            logger.info(f"Калибровка порогов на {len(historical_texture_data)} образцах")
            
            # Сбор всех метрик
            all_haralick_contrasts = []
            all_shannon_entropies = []
            all_gabor_energies = []
            all_lbp_variances = []
            
            for data in historical_texture_data:
                for zone_name, zone_metrics in data.items():
                    if isinstance(zone_metrics, dict):
                        if "contrast" in zone_metrics:
                            all_haralick_contrasts.append(zone_metrics["contrast"])
                        if "shannon_entropy" in zone_metrics:
                            all_shannon_entropies.append(zone_metrics["shannon_entropy"])
                        if "gabor_mean_energy" in zone_metrics:
                            all_gabor_energies.append(zone_metrics["gabor_mean_energy"])
                        if "histogram_variance" in zone_metrics:
                            all_lbp_variances.append(zone_metrics["histogram_variance"])
            
            # ИСПРАВЛЕНО: Обновление базовых линий
            new_baselines = self.texture_baselines.copy()
            
            if all_haralick_contrasts:
                mean_val, std_val = calculate_mean_std(np.array(all_haralick_contrasts))
                new_baselines["default"]["haralick_mean_contrast"] = mean_val
                new_baselines["default"]["haralick_std_contrast"] = std_val
            
            if all_shannon_entropies:
                mean_val, std_val = calculate_mean_std(np.array(all_shannon_entropies))
                new_baselines["default"]["shannon_entropy_mean"] = mean_val
                new_baselines["default"]["shannon_entropy_std"] = std_val
            
            if all_gabor_energies:
                mean_val, std_val = calculate_mean_std(np.array(all_gabor_energies))
                new_baselines["default"]["gabor_energy_mean"] = mean_val
                new_baselines["default"]["gabor_energy_std"] = std_val
            
            if all_lbp_variances:
                mean_val, std_val = calculate_mean_std(np.array(all_lbp_variances))
                new_baselines["default"]["lbp_uniformity_mean"] = mean_val
                new_baselines["default"]["lbp_uniformity_std"] = std_val
            
            self.texture_baselines = new_baselines
            self.calibrated = True
            
            logger.info("Калибровка порогов завершена успешно")
            
        except Exception as e:
            logger.error(f"Ошибка калибровки порогов: {e}")

    def save_analysis_cache(self, cache_file: str = "texture_cache.pkl") -> None:
        """Сохранение кэша анализа"""
        try:
            cache_path = CACHE_DIR / cache_file
            CACHE_DIR.mkdir(exist_ok=True)
            
            import pickle
            with open(cache_path, 'wb') as f:
                pickle.dump(self.analysis_cache, f)
            
            logger.info(f"Кэш анализа сохранен: {cache_path}")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения кэша: {e}")

    def load_analysis_cache(self, cache_file: str = "texture_cache.pkl") -> None:
        """Загрузка кэша анализа"""
        try:
            cache_path = CACHE_DIR / cache_file
            
            if cache_path.exists():
                import pickle
                with open(cache_path, 'rb') as f:
                    self.analysis_cache = pickle.load(f)
                
                logger.info(f"Кэш анализа загружен: {cache_path}")
            else:
                logger.info("Файл кэша не найден, используется пустой кэш")
                
        except Exception as e:
            logger.error(f"Ошибка загрузки кэша: {e}")

    def self_test(self) -> None:
        """Самотестирование модуля"""
        logger.info("=== Самотестирование TextureAnalyzer ===")
        
        # Информация о параметрах
        logger.info(f"LBP параметры: {self.lbp_params}")
        logger.info(f"Gabor параметры: {self.gabor_params}")
        logger.info(f"Калиброван: {self.calibrated}")
        
        # Тестовое изображение
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        test_landmarks = np.random.rand(68, 3) * 200
        
        try:
            # Тест анализа текстуры
            texture_metrics = self.analyze_skin_texture_by_zones(test_image, test_landmarks)
            logger.info(f"Тест анализа текстуры: {len(texture_metrics)} зон проанализировано")
            
            if texture_metrics:
                # Тест аутентичности материала
                authenticity_score = self.calculate_material_authenticity_score(texture_metrics)
                logger.info(f"Тест аутентичности: {authenticity_score:.3f}")
                
                # Тест классификации маски
                mask_classification = self.classify_mask_technology_level(texture_metrics)
                logger.info(f"Тест классификации маски: {mask_classification['level']}")
                
                # Тест адаптивного анализа
                lighting_conditions = {"brightness": 0.5, "contrast": 0.7, "uniformity": 0.6}
                adaptive_results = self.adaptive_texture_analysis(test_image, test_landmarks, lighting_conditions)
                logger.info(f"Тест адаптивного анализа: {len(adaptive_results)} результатов")
                
        except Exception as e:
            logger.error(f"Ошибка самотестирования: {e}")
        
        logger.info("=== Самотестирование завершено ===")

# ==================== ТОЧКА ВХОДА ====================

if __name__ == "__main__":
    analyzer = TextureAnalyzer()
    analyzer.self_test()