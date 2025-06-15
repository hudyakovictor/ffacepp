# texture_analyzer.py
# Анализ текстуры и детекция масок

import numpy as np
import cv2
from scipy import stats
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from skimage.morphology import disk
from skimage import filters
import mahotas
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
from core_config import MASK_DETECTION_THRESHOLDS, BREAKTHROUGH_YEARS, TEXTURE_ANALYSIS_THRESHOLDS
import math
from scipy.spatial.distance import euclidean
from scipy.stats import mode
import os
import json # Добавлен для обработки JSON в _load_texture_baselines
from pathlib import Path # Добавлен для работы с путями

# Для анализа текстуры и пор
from skimage.filters import threshold_local, gaussian
from skimage.measure import label, regionprops
from skimage.morphology import binary_opening, binary_closing, remove_small_objects

# Для Gabor фильтров
from scipy.ndimage import convolve
from skimage.filters import gabor_kernel

import skimage.feature
import skimage.filters
import skimage.morphology

# ==================== Вспомогательные функции ====================

def _calculate_mean_std(data: np.ndarray) -> Tuple[float, float]:
    """Вспомогательная функция для расчета среднего и стандартного отклонения."""
    if data.size == 0:
        return 0.0, 0.0
    return np.mean(data), np.std(data)

class TextureAnalyzer:
    def __init__(self):
        self.lbp_params = {
            'radius': 3,
            'n_points': 24,  # Для uniform: n_points = 8 * radius
            'method': 'default'  # или изменить n_points на 24
        }
        self.entropy_params = {
            'disk_size': 9
        }
        self.gabor_params = {
            'orientations': [0, 45, 90, 135],
            'frequencies': [0.1, 0.2, 0.3, 0.4]
        }
        # Эталонные параметры для энтропии натуральной кожи
        # Эти значения калибруются на основе ожидаемого диапазона (6.2-7.8) для натуральной кожи
        # Среднее значение (mean) выбрано для центра сигмоидного перехода,
        # а стандартное отклонение (std) для крутизны перехода.
        self.entropy_natural_threshold_mean = TEXTURE_ANALYSIS_THRESHOLDS['entropy_natural_min'] + (TEXTURE_ANALYSIS_THRESHOLDS['entropy_natural_max'] - TEXTURE_ANALYSIS_THRESHOLDS['entropy_natural_min']) / 2
        self.entropy_natural_threshold_std = (TEXTURE_ANALYSIS_THRESHOLDS['entropy_natural_max'] - TEXTURE_ANALYSIS_THRESHOLDS['entropy_natural_min']) / 4 # Примерное стандартное отклонение
        
        # Базовые значения для анализа пор (инициализация)
        self.pore_density_mean = TEXTURE_ANALYSIS_THRESHOLDS['pore_density_mean'] if 'pore_density_mean' in TEXTURE_ANALYSIS_THRESHOLDS else 0.10
        self.pore_density_std = TEXTURE_ANALYSIS_THRESHOLDS['pore_density_std'] if 'pore_density_std' in TEXTURE_ANALYSIS_THRESHOLDS else 0.03
        self.pore_size_std_mean = TEXTURE_ANALYSIS_THRESHOLDS['pore_size_std_mean'] if 'pore_size_std_mean' in TEXTURE_ANALYSIS_THRESHOLDS else 0.03
        self.pore_size_std_std = TEXTURE_ANALYSIS_THRESHOLDS['pore_size_std_std'] if 'pore_size_std_std' in TEXTURE_ANALYSIS_THRESHOLDS else 0.01
        self.pore_circularity_mean = TEXTURE_ANALYSIS_THRESHOLDS['pore_circularity_mean'] if 'pore_circularity_mean' in TEXTURE_ANALYSIS_THRESHOLDS else 0.85
        self.pore_circularity_std = TEXTURE_ANALYSIS_THRESHOLDS['pore_circularity_std'] if 'pore_circularity_std' in TEXTURE_ANALYSIS_THRESHOLDS else 0.05
        self.texture_baselines = self._load_texture_baselines() # Загрузка эталонных текстур
        self.entropy_mean = 0.0
        self.entropy_std = 1.0
        self.gabor_orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        self.gabor_frequencies = [0.1, 0.2, 0.3]
        self.pore_density_baselines = {'mean': 0.005, 'std': 0.002}
        self.pore_circularity_baselines = {'mean': 0.5, 'std': 0.1}
        self.calibrated_thresholds = {}
        self.calibrated = False

    def set_entropy_thresholds(self, mean: float, std: float):
        """Устанавливает параметры порогов энтропии для анализа текстуры."""
        self.entropy_natural_threshold_mean = mean
        self.entropy_natural_threshold_std = std
        logging.info(f"Параметры энтропии натуральной кожи обновлены: mean={mean}, std={std}")

    def set_gabor_params(self, orientations: List[float], frequencies: List[float]):
        """Устанавливает параметры для Gabor фильтров."""
        self.gabor_params['orientations'] = orientations
        self.gabor_params['frequencies'] = frequencies
        logging.info(f"Параметры Gabor фильтров обновлены: Ориентации={orientations}, Частоты={frequencies}")

    def calibrate_pore_baselines(self, 
                                 density_mean: float, density_std: float,
                                 size_std_mean: float, size_std_std: float,
                                 circularity_mean: float, circularity_std: float):
        """Калибрует базовые линии для анализа пор на основе исторических данных."""
        self.pore_density_mean = density_mean
        self.pore_density_std = density_std
        self.pore_size_std_mean = size_std_mean
        self.pore_size_std_std = size_std_std
        self.pore_circularity_mean = circularity_mean
        self.pore_circularity_std = circularity_std
        logging.info("Базовые линии для анализа пор откалиброваны.")

    def _load_texture_baselines(self) -> Dict:
        """Загружает эталонные текстурные характеристики из файла JSON."""
        baseline_file = Path(os.path.dirname(__file__)) / "configs" / "texture_baselines.json"
        if baseline_file.exists():
            with open(baseline_file, 'r', encoding='utf-8') as f:
                logging.info(f"Загрузка эталонных текстурных характеристик из {baseline_file}")
                return json.load(f)
        else:
            logging.warning(f"Файл эталонных данных не найден: {baseline_file}. Использование хардкодных значений.")
            # Возвращаем хардкодные значения в случае отсутствия файла для избежания ошибок
            return {
                'forehead': {
                    'haralick_mean_contrast': 0.15, 'haralick_std_contrast': 0.05,
                    'shannon_entropy_mean': 7.5, 'shannon_entropy_std': 0.5,
                    'gabor_energy_mean': 100.0, 'gabor_energy_std': 20.0,
                    'lbp_uniformity_mean': 0.9, 'lbp_uniformity_std': 0.05
                },
                'cheek': {
                    'haralick_mean_contrast': 0.18, 'haralick_std_contrast': 0.06,
                    'shannon_entropy_mean': 7.8, 'shannon_entropy_std': 0.6,
                    'gabor_energy_mean': 110.0, 'gabor_energy_std': 25.0,
                    'lbp_uniformity_mean': 0.88, 'lbp_uniformity_std': 0.06
                },
                'default': {
                    'haralick_mean_contrast': 0.17, 'haralick_std_contrast': 0.05,
                    'shannon_entropy_mean': 7.6, 'shannon_entropy_std': 0.55,
                    'gabor_energy_mean': 105.0, 'gabor_energy_std': 22.0,
                    'lbp_uniformity_mean': 0.89, 'lbp_uniformity_std': 0.055
                }
            }

    def analyze_skin_texture_by_zones(self, image: np.ndarray, landmarks: np.ndarray) -> Dict:
        """
        Анализирует текстуру кожи по предопределенным зонам лица.
        Обновлено для включения анализа микроморщин и распределения пор.
        """
        texture_metrics = {}
        if image is None or landmarks.size == 0:
            return texture_metrics

        skin_zones = self._define_skin_zones(landmarks, image.shape)

        for zone_name, zone_coords in skin_zones.items():
            if zone_coords.size == 0:
                continue

            # Создаем маску для зоны
            mask = self._create_zone_mask(image.shape[:2], zone_coords)
            
            # Создаем цветной регион из изображения с помощью маски
            # Преобразуем 1-канальную маску в 3-канальную для поэлементного умножения
            mask_3d = np.stack([mask, mask, mask], axis=-1)
            zone_region_colored = image * (mask_3d // 255) # Маска 255, поэтому делим, чтобы получить 0 или 1

            # Обрезаем регион до минимального ограничивающего прямоугольника, чтобы избежать обработки пустых пикселей
            # Find the bounding box of the mask
            y_coords, x_coords = np.where(mask > 0)
            if y_coords.size == 0 or x_coords.size == 0:
                continue # Если маска пустая

            min_y, max_y = np.min(y_coords), np.max(y_coords)
            min_x, max_x = np.min(x_coords), np.max(x_coords)

            cropped_zone_region = zone_region_colored[min_y:max_y+1, min_x:max_x+1]
            cropped_mask = mask[min_y:max_y+1, min_x:max_x+1]
            
            # Убедимся, что изображение не пустое после обрезки
            if cropped_zone_region.size == 0:
                continue

            # Получаем только "активные" пиксели внутри маски
            valid_pixels_mask = cropped_mask > 0
            if cropped_zone_region.ndim == 3:
                valid_pixels_colored = cropped_zone_region[valid_pixels_mask]
            else:
                valid_pixels_colored = cropped_zone_region[valid_pixels_mask]
            
            # Преобразование в оттенки серого для функций, требующих 2D-массивы
            zone_region_gray = cv2.cvtColor(cropped_zone_region, cv2.COLOR_BGR2GRAY) if cropped_zone_region.ndim == 3 else cropped_zone_region
            valid_pixels_gray = zone_region_gray[valid_pixels_mask]

            if valid_pixels_colored.size == 0 or valid_pixels_gray.size == 0:
                logging.warning(f"В зоне {zone_name} нет действительных пикселей после фильтрации. Пропускаем.")
                continue
            
            # Выполняем детальный анализ текстуры, передавая grayscale версии
            zone_analysis = self._analyze_zone_texture(zone_region_gray, valid_pixels_gray)
            texture_metrics[zone_name] = zone_analysis
        
        return texture_metrics

    def _analyze_zone_texture(self, zone_region_2d: np.ndarray, valid_pixels_1d: np.ndarray) -> Dict:
        """Выполняет детальный текстурный анализ для одной зоны. Принимает 2D-массивы."""
        zone_metrics = {}
        
        # Гистограммные признаки
        zone_metrics['shannon_entropy'] = self._calculate_shannon_entropy(valid_pixels_1d)

        # LBP признаки
        lbp_features = self._calculate_lbp_features(zone_region_2d)
        zone_metrics.update(lbp_features)

        # Gabor фильтры
        gabor_responses = self._calculate_gabor_responses(zone_region_2d)
        zone_metrics.update({'gabor_responses': gabor_responses})

        # Fourier Spectrum
        fourier_spectrum = self._calculate_fourier_spectrum(zone_region_2d)
        zone_metrics.update(fourier_spectrum)

        # Haralick Features (GLCM)
        haralick_features = self._calculate_haralick_features(zone_region_2d)
        zone_metrics.update({'haralick_features': haralick_features})

        return zone_metrics

    def _calculate_lbp_features(self, image_region: np.ndarray) -> Dict:
        """Вычисляет Local Binary Patterns (LBP) для региона изображения."""
        radius = 3
        n_points = 8 * radius  # Правильное соотношение для uniform метода: 24 точки

        # Проверка корректности параметров для uniform метода
        if self.lbp_params.get('method') == 'uniform' and n_points % 8 != 0:
            n_points = 8 * radius

        lbp_image = local_binary_pattern(image_region, n_points, radius, method='uniform')

        # Гистограмма LBP
        n_bins = int(lbp_image.max() + 1)
        hist, _ = np.histogram(lbp_image.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-8)  # Нормализация

        return {
            'lbp_histogram': hist.tolist(),
            'histogram_variance': np.var(hist),
            'histogram_entropy': -np.sum(hist * np.log2(hist + 1e-8))
        }

    def _calculate_shannon_entropy(self, pixel_values: np.ndarray) -> float:
        """Рассчитывает энтропию Шеннона для распределения значений пикселей."""
        if pixel_values.size == 0: 
            return 0.0
        
        # Убеждаемся, что pixel_values является одномерным массивом
        if pixel_values.ndim > 1:
            pixel_values = pixel_values.flatten()
        
        # Нормализация до 0-255 и приведение к uint8
        if pixel_values.max() > 255 or pixel_values.min() < 0:
            pixel_values = (255 * (pixel_values - pixel_values.min()) / 
                           (pixel_values.max() - pixel_values.min() + 1e-8))
        
        # Приведение к uint8 для cv2.calcHist
        pixel_values = pixel_values.astype(np.uint8)
        
        # Убеждаемся, что массив непрерывный в памяти
        if not pixel_values.flags['C_CONTIGUOUS']:
            pixel_values = np.ascontiguousarray(pixel_values)
        
        # Вычисление гистограммы
        hist = cv2.calcHist([pixel_values], [0], None, [256], [0, 256])
        hist = hist.ravel() / (hist.sum() + 1e-8)  # Нормализация гистограммы
        
        # Вычисление энтропии
        entropy = -np.sum(hist * np.log2(hist + 1e-8))
        return entropy

    def _calculate_gabor_responses(self, image_region: np.ndarray) -> Dict:
        """Вычисляет отклики Gabor фильтров для региона изображения."""
        gabor_results = {}
        # Масштабирование до 0-1, если изображение не float
        if image_region.dtype != np.float32 and image_region.dtype != np.float64:
            image_region = image_region.astype(np.float32) / 255.0

        for theta in self.gabor_params['orientations']:
            for freq in self.gabor_params['frequencies']:
                kernel = np.real(gabor_kernel(frequency=freq, theta=np.radians(theta), 
                                            sigma_x=1.0, sigma_y=1.0))
                filtered_image = convolve(image_region, kernel, mode='nearest')
                
                gabor_results[f'theta_{theta}_freq_{freq}'] = {
                    'mean': np.mean(filtered_image),
                    'std': np.std(filtered_image),
                    'energy': np.sum(filtered_image**2)
                }
        return gabor_results

    def _calculate_fourier_spectrum(self, image_region: np.ndarray) -> Dict:
        """Вычисляет Фурье-спектр и извлекает частотные характеристики."""
        if image_region.size == 0 or image_region.ndim == 0:
            logging.warning("_calculate_fourier_spectrum: Пустой или некорректный регион изображения. Возвращаем значения по умолчанию.")
            return {
                'spectral_centroid': 0.0,
                'spectral_rolloff': 0.0,
                'dominant_frequency': 0
            }

        fourier = np.fft.fft2(image_region)
        fourier_shifted = np.fft.fftshift(fourier)
        magnitude_spectrum = np.abs(fourier_shifted) # Изменено для обеспечения неотрицательных значений

        # Проверка на нулевой спектр
        if np.sum(magnitude_spectrum) == 0:
            logging.warning("_calculate_fourier_spectrum: Общая энергия спектра равна нулю. Возвращаем значения по умолчанию.")
            return {
                'spectral_centroid': 0.0,
                'spectral_rolloff': 0.0,
                'dominant_frequency': 0
            }

        spectral_metrics = {
            'spectral_centroid': self._calculate_spectral_centroid(magnitude_spectrum),
            'spectral_rolloff': self._calculate_spectral_rolloff(magnitude_spectrum),
            'dominant_frequency': self._find_dominant_frequency(magnitude_spectrum)
        }

        return spectral_metrics

    def _calculate_haralick_features(self, image_region: np.ndarray) -> Dict:
        """Вычисляет текстурные признаки Харалика (GLCM) для региона изображения."""
        # GLCM требует 8-битного изображения
        image_8bit = cv2.normalize(image_region, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

        # Параметры GLCM
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        levels = 256

        # ИСПРАВЛЕНО: graycomatrix не принимает аргумент 'normalize' или 'normed'.
        # Нормализация выполняется в graycoprops. Удаляем glcm_kwargs.
        # glcm_kwargs = {'normed': True}

        glcm = graycomatrix(image_8bit, distances=distances, angles=angles,
                        levels=levels, symmetric=True)

        haralick_features = {
            'contrast': np.mean(graycoprops(glcm, 'contrast')),
            'dissimilarity': np.mean(graycoprops(glcm, 'dissimilarity')),
            'homogeneity': np.mean(graycoprops(glcm, 'homogeneity')),
            'energy': np.mean(graycoprops(glcm, 'energy')),
            'correlation': np.mean(graycoprops(glcm, 'correlation'))
        }

        return haralick_features

    def calculate_material_authenticity_score(self, texture_metrics: Dict) -> float:
        """
        Рассчитывает общий балл аутентичности материала на основе текстурных метрик,
        включая новые метрики микрорельефа.
        """
        scores = []
        
        # Веса для разных зон (пример, можно настроить)
        zone_weights = {
            'forehead': 0.2, 'cheeks': 0.3, 'nose': 0.2,
            'chin': 0.1, 'eyes': 0.2
        }

        for zone_name, metrics in texture_metrics.items():
            zone_score = self._calculate_zone_authenticity(metrics, zone_name)
            scores.append(zone_score * zone_weights.get(zone_name, 0.1)) # Применяем веса

        if not scores:
            return 0.0

        overall_score = np.sum(scores)
        return min(1.0, max(0.0, overall_score)) # Ограничиваем от 0 до 1

    def _calculate_zone_authenticity(self, zone_metrics: Dict, zone_name: str = 'default') -> float:
        """
        Рассчитывает балл аутентичности для отдельной зоны,
        включая метрики микроморщин и пор.
        """
        score = 0.0
        
        # Базовые метрики
        if 'shannon_entropy' in zone_metrics:
            # Энтропия: чем выше, тем более "случайна" текстура, что хорошо для натуральной кожи
            # Пороговые значения для энтропии могут быть определены на основе калибровки
            entropy_score = self._calculate_sigmoid_score(zone_metrics['shannon_entropy'], self.entropy_mean, self.entropy_std) # Примерные значения
            score += entropy_score * 0.2
        
        if 'mean_color_std' in zone_metrics:
            # Стандартное отклонение цвета: небольшое разнообразие хорошо
            color_std_score = self._calculate_sigmoid_score(zone_metrics['mean_color_std'], 20.0, 5.0, reverse=True) # Чем выше std, тем хуже
            score += color_std_score * 0.15

        if 'lbp_uniformity' in zone_metrics:
            # Равномерность LBP: равномерность микропаттернов
            lbp_score = self._calculate_sigmoid_score(zone_metrics['lbp_uniformity'], 0.9, 0.1)
            score += lbp_score * 0.15
        
        # Новые метрики микрорельефа
        if 'micro_wrinkles' in zone_metrics and zone_metrics['micro_wrinkles']:
            wrinkles = zone_metrics['micro_wrinkles']
            # Чем меньше анизотропия, тем более однородна кожа (лучше для "гладких" зон, хуже для морщин)
            # mean_gabor_response: чем выше, тем больше текстурных деталей
            gabor_score = self._calculate_sigmoid_score(wrinkles.get('mean_gabor_response', 0.0), 50.0, 20.0) # Примерные значения
            score += gabor_score * 0.2
            
            anisotropy_score = self._calculate_sigmoid_score(wrinkles.get('anisotropy_score', 0.0), 0.5, 0.2, reverse=True) # Высокая анизотропия плохо
            score += anisotropy_score * 0.05 # Меньший вес

        if 'pore_distribution' in zone_metrics and zone_metrics['pore_distribution']:
            pores = zone_metrics['pore_distribution']
            # Плотность пор: естественная кожа имеет определенную плотность
            pore_density_score = self._calculate_sigmoid_score(pores.get('pore_density', 0.0), self.pore_density_baselines['mean'], self.pore_density_baselines['std']) # Примерные значения (0.005 пор на пиксель)
            score += pore_density_score * 0.15
            
            # Циркулярность пор: естественные поры обычно неидеальной формы
            circularity_score = self._calculate_sigmoid_score(pores.get('mean_pore_circularity', 0.0), self.pore_circularity_baselines['mean'], self.pore_circularity_baselines['std']) # Примерные значения
            score += circularity_score * 0.1

        return score # Нормализовать на основе суммы весов, чтобы получить значение от 0 до 1

    def _calculate_sigmoid_score(self, value: float, mean: float, std: float, reverse: bool = False) -> float:
        """
        Вычисляет балл от 0 до 1 на основе сигмоидной функции.
        `reverse=True` означает, что меньшие значения дают более высокий балл.
        """
        if std == 0:
            return 1.0 if value == mean else 0.0 # Избежать деления на ноль

        # Z-score
        z = (value - mean) / std
        
        # Сигмоидальная функция
        if reverse:
            return 1 / (1 + np.exp(z)) # Чем больше Z, тем меньше score
        else:
            return 1 / (1 + np.exp(-z)) # Чем больше Z, тем больше score

    def _analyze_micro_wrinkles(self, region_img: np.ndarray) -> Dict:
        """Анализирует микроморщины в регионе изображения.
        Использует фильтры Габора для выявления ориентированных структур.
        """
        if region_img.size == 0 or region_img.ndim not in [2, 3]:
            return {'mean_gabor_response': 0.0, 'anisotropy_score': 0.0}

        if region_img.ndim == 3:
            region_img = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
        
        # Применяем фильтры Габора для разных ориентаций
        gabor_responses = []
        angles = np.linspace(0, np.pi, 8, endpoint=False) # 8 ориентаций
        freq = 0.1 # Частота (можно настроить)

        for angle in angles:
            filt_real, filt_imag = skimage.filters.gabor(region_img, frequency=freq, theta=angle)
            gabor_responses.append(np.mean(np.sqrt(filt_real**2 + filt_imag**2))) # Амплитуда ответа

        if not gabor_responses:
            return {'mean_gabor_response': 0.0, 'anisotropy_score': 0.0}

        mean_gabor_response = np.mean(gabor_responses)
        # Анизотропия: насколько ответы по разным направлениям отличаются
        anisotropy_score = np.std(gabor_responses) / (mean_gabor_response + 1e-6)

        return {
            'mean_gabor_response': mean_gabor_response,
            'anisotropy_score': anisotropy_score
        }

    def _analyze_pore_distribution(self, region_img: np.ndarray) -> Dict:
        """Анализирует распределение пор в регионе изображения.
        Использует морфологические операции и детектор блобов.
        """
        if region_img.size == 0 or region_img.ndim not in [2, 3]:
            return {'pore_count': 0, 'mean_pore_area': 0.0, 'pore_density': 0.0, 'mean_pore_circularity': 0.0}

        if region_img.ndim == 3:
            region_img = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)

        # Нормализация для лучшей детекции
        normalized_img = cv2.normalize(region_img, None, 0, 255, cv2.NORM_MINMAX)
        
        # Адаптивная бинаризация для выделения пор (более темные области)
        block_size = 31 # Размер окна для адаптивного порога
        C_param = 5   # Вычитаемое из среднего
        binary_img = cv2.adaptiveThreshold(normalized_img, 255, 
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, block_size, C_param)
        
        # Морфологические операции для улучшения выделения пор
        kernel = np.ones((2,2), np.uint8)
        # Открытие (эрозия затем дилатация) для удаления мелкого шума
        processed_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=1)
        # Закрытие (дилатация затем эрозия) для заполнения мелких разрывов
        processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Детекция контуров (предполагаемых пор)
        contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pore_areas = []
        pore_circularities = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1 and area < 100: # Фильтр по размеру пор (1-100 пикселей, можно настроить)
                pore_areas.append(area)
                
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter**2)
                    pore_circularities.append(circularity)

        pore_count = len(pore_areas)
        mean_pore_area = np.mean(pore_areas) if pore_areas else 0.0
        
        # Плотность пор: количество пор на площадь региона
        region_area = region_img.shape[0] * region_img.shape[1]
        pore_density = pore_count / (region_area + 1e-6) if region_area > 0 else 0.0
        mean_pore_circularity = np.mean(pore_circularities) if pore_circularities else 0.0

        return {
            'pore_count': pore_count,
            'mean_pore_area': mean_pore_area,
            'pore_density': pore_density,
            'mean_pore_circularity': mean_pore_circularity
        }

    def _define_skin_zones(self, landmarks: np.ndarray, image_shape: Tuple) -> Dict:
        """Определяет ключевые зоны кожи лица на основе landmarks."""
        zones = {}
        h, w = image_shape[:2]

        # Лоб (приблизительно) - по landmarks 19-24 и выше
        if len(landmarks) >= 27:
            forehead_pts = np.array([
                (landmarks[19, 0], landmarks[19, 1] - (landmarks[19,1] - landmarks[27,1])/2), # Выше бровей
                (landmarks[24, 0], landmarks[24, 1] - (landmarks[24,1] - landmarks[27,1])/2), 
                (landmarks[27,0], 0), # Верхняя часть изображения
                (landmarks[19,0], 0), # Верхняя часть изображения
            ], np.int32)
            # Корректировка, чтобы лоб был прямоугольником или трапецией над бровями
            forehead_rect_coords = np.array([
                [landmarks[19, 0] - 10, landmarks[19, 1] - 50],  # Левый верхний угол
                [landmarks[24, 0] + 10, landmarks[24, 1] - 50],  # Правый верхний угол
                [landmarks[24, 0] + 10, landmarks[24, 1] + 10],  # Правый нижний угол (около бровей)
                [landmarks[19, 0] - 10, landmarks[19, 1] + 10]   # Левый нижний угол (около бровей)
            ], np.int32)
            # Обрезка по границам изображения
            forehead_rect_coords[:,0] = np.clip(forehead_rect_coords[:,0], 0, w-1)
            forehead_rect_coords[:,1] = np.clip(forehead_rect_coords[:,1], 0, h-1)
            zones['forehead'] = forehead_rect_coords

        # Левая щека (приблизительно) - между глазом, носом и линией челюсти
        if len(landmarks) >= 14:
            left_cheek_pts = np.array([
                landmarks[1], landmarks[2], landmarks[3], landmarks[4], landmarks[5], # Контур челюсти
                landmarks[48], # Уголок рта
                landmarks[31], # Левая сторона носа
                landmarks[36], # Левый глаз
                landmarks[17]  # Левая бровь
            ], np.int32)
            zones['left_cheek'] = left_cheek_pts

        # Правая щека (приблизительно)
        if len(landmarks) >= 14:
            right_cheek_pts = np.array([
                landmarks[15], landmarks[14], landmarks[13], landmarks[12], landmarks[11], # Контур челюсти
                landmarks[54], # Уголок рта
                landmarks[35], # Правая сторона носа
                landmarks[45], # Правый глаз
                landmarks[26]  # Правая бровь
            ], np.int32)
            zones['right_cheek'] = right_cheek_pts

        # Нос (переносица и кончик)
        if len(landmarks) >= 35:
            nose_pts = np.array([
                landmarks[27], landmarks[28], landmarks[29], landmarks[30], # Переносица и кончик
                landmarks[31], landmarks[35] # Крылья носа
            ], np.int32)
            zones['nose'] = nose_pts

        # Подбородок
        if len(landmarks) >= 9:
            chin_pts = np.array([
                landmarks[6], landmarks[7], landmarks[8], landmarks[9], landmarks[10], # Нижняя часть подбородка
                landmarks[58], landmarks[56] # Нижняя губа
            ], np.int32)
            zones['chin'] = chin_pts

        return zones

    def _create_zone_mask(self, image_shape: Tuple, zone_coords: np.ndarray) -> np.ndarray:
        """Создает бинарную маску для заданной зоны."""
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        if zone_coords.size > 0:
            cv2.fillPoly(mask, [zone_coords.astype(np.int32)], 255)
        return mask

    def _calculate_spectral_centroid(self, magnitude: np.ndarray) -> float:
        """Вычисляет спектральный центроид из спектра величины."""
        rows, cols = magnitude.shape
        if rows == 0 or cols == 0: return 0.0

        total_magnitude = np.sum(magnitude)
        if total_magnitude == 0: return 0.0

        # Создаем сетку частот
        freq_y, freq_x = np.indices(magnitude.shape)
        
        centroid_x = np.sum(freq_x * magnitude) / total_magnitude
        centroid_y = np.sum(freq_y * magnitude) / total_magnitude

        return np.sqrt(centroid_x**2 + centroid_y**2)

    def _calculate_spectral_rolloff(self, magnitude: np.ndarray, rolloff_threshold: float = 0.85) -> float:
        """Вычисляет спектральный роллофф (частоту, ниже которой находится заданный процент энергии)."""
        flattened_magnitude = np.sort(magnitude.ravel()) # Сортируем для кумулятивной суммы
        cumulative_magnitude = np.cumsum(flattened_magnitude)
        
        # Дополнительная проверка: если кумулятивная сумма пуста, возвращаем 0.0
        if cumulative_magnitude.size == 0:
            logging.warning(f"_calculate_spectral_rolloff: cumulative_magnitude пуст. Возвращаем 0.0.")
            return 0.0

        total_energy = cumulative_magnitude[-1]
        
        if total_energy == 0: 
            logging.warning(f"_calculate_spectral_rolloff: total_energy равен 0. Возвращаем 0.0.")
            return 0.0

        threshold_energy = total_energy * rolloff_threshold
        
        # Находим индекс, где кумулятивная энергия превышает порог
        rolloff_indices = np.where(cumulative_magnitude >= threshold_energy)[0]
        
        logging.debug(f"_calculate_spectral_rolloff: cumulative_magnitude.size={cumulative_magnitude.size}, threshold_energy={threshold_energy}, rolloff_indices.size={rolloff_indices.size}")

        if rolloff_indices.size == 0:
            # Если порог не достигнут, это может указывать на очень низкую энергию или равномерный спектр
            logging.warning(f"_calculate_spectral_rolloff: rolloff_indices пуст. Возвращаем 0.0. cumulative_magnitude={cumulative_magnitude}, threshold_energy={threshold_energy}")
            return 0.0
            
        rolloff_idx = rolloff_indices[0]
        
        # Возвращаем частоту, соответствующую этому индексу (прокси)
        # Здесь упрощено, в реальной реализации нужно маппировать индекс на реальную частоту
        return rolloff_idx / len(flattened_magnitude) # Нормализованное значение

    def _find_dominant_frequency(self, magnitude: np.ndarray) -> int:
        """Находит доминирующую частоту в спектре величины."""
        # Игнорируем центральную (низкочастотную) компоненту
        rows, cols = magnitude.shape
        center_row, center_col = rows // 2, cols // 2
        magnitude_copy = magnitude.copy()
        magnitude_copy[center_row-1:center_row+2, center_col-1:center_col+2] = 0 # Убираем центр

        dominant_idx = np.unravel_index(np.argmax(magnitude_copy), magnitude_copy.shape)
        
        # Возвращаем "частоту" как расстояние от центра
        return int(np.sqrt((dominant_idx[0] - center_row)**2 + (dominant_idx[1] - center_col)**2))

    def calculate_spectral_material_signature(self, texture_regions: List[np.ndarray]) -> Dict:
        """Вычисляет спектральную сигнатуру материала для верификации аутентичности."""
        # Этот метод анализирует Фурье-спектры регионов, выявляя характерные частоты
        # и распределение энергии, которые могут указывать на синтетические материалы.
        # В основном опирается на _calculate_fourier_spectrum
        
        spectral_signatures = {
            'spectral_centroids': [],
            'spectral_rolloffs': [],
            'dominant_frequencies': []
        }

        for region in texture_regions:
            if region.size > 0:
                fourier_metrics = self._calculate_fourier_spectrum(region)
                spectral_signatures['spectral_centroids'].append(fourier_metrics.get('spectral_centroid', 0.0))
                spectral_signatures['spectral_rolloffs'].append(fourier_metrics.get('spectral_rolloff', 0.0))
                spectral_signatures['dominant_frequencies'].append(fourier_metrics.get('dominant_frequency', 0))

        # Агрегация результатов (среднее, стандартное отклонение)
        return {
            'mean_centroid': np.mean(spectral_signatures['spectral_centroids']) if spectral_signatures['spectral_centroids'] else 0.0,
            'std_centroid': np.std(spectral_signatures['spectral_centroids']) if spectral_signatures['spectral_centroids'] else 0.0,
            'mean_rolloff': np.mean(spectral_signatures['spectral_rolloffs']) if spectral_signatures['spectral_rolloffs'] else 0.0,
            'std_rolloff': np.std(spectral_signatures['spectral_rolloffs']) if spectral_signatures['spectral_rolloffs'] else 0.0,
            'mean_dominant_freq': np.mean(spectral_signatures['dominant_frequencies']) if spectral_signatures['dominant_frequencies'] else 0.0,
            'std_dominant_freq': np.std(spectral_signatures['dominant_frequencies']) if spectral_signatures['dominant_frequencies'] else 0.0
        }

    def adaptive_texture_analysis(self, image: np.ndarray, landmarks: np.ndarray, lighting_conditions: Dict) -> Dict:
        """
        Адаптивный анализ текстуры с учетом условий освещения.
        Корректирует параметры анализа и/или применяет предварительную обработку
        в зависимости от яркости, контраста и равномерности освещения.

        Args:
            image (np.ndarray): Входное изображение.
            landmarks (np.ndarray): Ландмарки лица на изображении.
            lighting_conditions (Dict): Словарь с параметрами освещения (e.g., 'brightness', 'contrast', 'uniformity').

        Returns:
            Dict: Результаты анализа текстуры, адаптированные к условиям освещения.
        """
        if image is None or landmarks.size == 0:
            logging.warning("Изображение или ландмарки отсутствуют для адаптивного анализа текстуры.")
            return {}
        
        processed_image = image.copy()
        analysis_parameters = {}

        # Извлечение параметров освещения
        brightness = lighting_conditions.get('brightness', 0.5)  # 0-1
        contrast = lighting_conditions.get('contrast', 0.5)      # 0-1
        uniformity = lighting_conditions.get('uniformity', 0.5)  # 0-1 (1.0 - идеально равномерное)

        # Адаптация на основе условий освещения
        if brightness < 0.3 or contrast < 0.3 or uniformity < 0.4: # Низкое освещение или неравномерность
            logging.info("Адаптивный анализ: Обнаружены сложные условия освещения (низкая яркость/контраст/равномерность). Применение усиления.")
            # Применение CLAHE для улучшения контраста в темных областях
            gray_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY) if len(processed_image.shape) == 3 else processed_image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed_image = clahe.apply(gray_image.astype(np.uint8))
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR) # Возвращаем в BGR
            analysis_parameters['preprocessing'] = 'CLAHE_applied'
            # Можно также скорректировать параметры LBP/Gabor для большей робастности к шуму
            self.lbp_params['radius'] = 2 # Меньший радиус для локальных особенностей
            self.gabor_params['frequencies'] = [0.2, 0.3, 0.4] # Больше высоких частот

        elif brightness > 0.8 or contrast > 0.8: # Пересвет
            logging.info("Адаптивный анализ: Обнаружен пересвет. Применение нормализации.")
            # Нормализация яркости (можно использовать более сложные методы, например гамма-коррекцию)
            processed_image = cv2.convertScaleAbs(processed_image, alpha=(255.0/processed_image.max()), beta=0)
            analysis_parameters['preprocessing'] = 'brightness_normalized'
            # Можно сосредоточиться на признаках, менее чувствительных к яркости
            self.lbp_params['method'] = 'default' # Менее чувствителен к яркости

        else: # Оптимальное освещение
            logging.info("Адаптивный анализ: Оптимальные условия освещения. Стандартный анализ.")
            analysis_parameters['preprocessing'] = 'none'
            # Восстановление стандартных параметров, если они были изменены
            self.lbp_params = {'radius': 3, 'n_points': 24, 'method': 'uniform'}
            self.gabor_params = {'orientations': [0, 45, 90, 135], 'frequencies': [0.1, 0.2, 0.3, 0.4]}
        
        # Выполняем основной анализ текстуры с адаптированными параметрами
        texture_results = self.analyze_skin_texture_by_zones(processed_image, landmarks)
        texture_results['adaptive_parameters'] = analysis_parameters # Добавляем использованные параметры в результат

        return texture_results

    def calculate_texture_authenticity_score(self, texture_data: Dict) -> float:
        """Вычисляет общий балл аутентичности текстуры на основе всех доступных метрик."""
        scores = []

        # Аутентичность материала
        material_score = self.calculate_material_authenticity_score(texture_data)
        if material_score is not None: scores.append(material_score)

        # Однородность текстуры
        uniformity_score = self._analyze_texture_uniformity(texture_data)
        if uniformity_score is not None: scores.append(uniformity_score)

        # Микродетали
        micro_details_score = self._analyze_micro_details(texture_data)
        if micro_details_score is not None: scores.append(micro_details_score)
        
        return np.mean(scores) if scores else 0.0

    def calibrate_texture_analysis_thresholds(self, historical_texture_data: List[Dict]):
        """Калибрует пороги для анализа текстуры на основе исторических данных.
        
        Эти пороги используются для оценки отклонений от нормы в различных текстурных метриках.
        """
        if not historical_texture_data: return

        # Собираем данные по всем метрикам из исторических данных
        all_haralick_contrasts = []
        all_shannon_entropies = []
        all_gabor_energies = []
        all_lbp_variances = []

        for data in historical_texture_data:
            for zone_name, zone_metrics in data.items():
                if 'haralick_features' in zone_metrics: all_haralick_contrasts.append(zone_metrics['haralick_features'].get('contrast', 0))
                if 'shannon_entropy' in zone_metrics: all_shannon_entropies.append(zone_metrics['shannon_entropy'])
                if 'gabor_responses' in zone_metrics:
                    total_energy = sum([stats.get('energy', 0) for stats in zone_metrics['gabor_responses'].values()])
                    all_gabor_energies.append(total_energy)
                if 'histogram_variance' in zone_metrics: all_lbp_variances.append(zone_metrics['histogram_variance'])

        # Пересчитываем средние и стандартные отклонения для baselines
        # (Эти пороги будут использоваться в _calculate_zone_authenticity)
        new_baselines = self.texture_baselines.copy()

        if all_haralick_contrasts: 
            mean_val, std_val = _calculate_mean_std(np.array(all_haralick_contrasts))
            new_baselines['default']['haralick_mean_contrast'] = mean_val
            new_baselines['default']['haralick_std_contrast'] = std_val
            # TODO: Можно калибровать и для каждой зоны отдельно

        if all_shannon_entropies: 
            mean_val, std_val = _calculate_mean_std(np.array(all_shannon_entropies))
            new_baselines['default']['shannon_entropy_mean'] = mean_val
            new_baselines['default']['shannon_entropy_std'] = std_val
        
        if all_gabor_energies: 
            mean_val, std_val = _calculate_mean_std(np.array(all_gabor_energies))
            new_baselines['default']['gabor_energy_mean'] = mean_val
            new_baselines['default']['gabor_energy_std'] = std_val

        if all_lbp_variances: 
            mean_val, std_val = _calculate_mean_std(np.array(all_lbp_variances))
            new_baselines['default']['lbp_uniformity_mean'] = mean_val
            new_baselines['default']['lbp_uniformity_std'] = std_val

        self.texture_baselines = new_baselines
        logging.info("Пороги анализа текстуры успешно откалиброваны.")
