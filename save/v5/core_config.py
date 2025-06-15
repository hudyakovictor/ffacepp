# core_config.py

# Конфигурация и константы для системы анализа лицевой идентичности

import os
import platform
import sys
import onnxruntime
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional
import logging
import json
from pathlib import Path

# =====================
# Пути и директории
# =====================

ROOT_DIR = Path(__file__).parent.absolute()
DATA_DIR = ROOT_DIR / 'data'
MODELS_DIR = ROOT_DIR / 'models'
RESULTS_DIR = ROOT_DIR / 'results'
CONFIG_DIR = ROOT_DIR / 'configs'
CACHE_DIR = ROOT_DIR / 'cache'
LOGS_DIR = ROOT_DIR / 'logs'

# Создание необходимых директорий
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, CONFIG_DIR, CACHE_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# =====================
# Определение системной архитектуры и ОС
# =====================

IS_MACOS = platform.system() == "Darwin"
IS_ARM64 = platform.machine() == "arm64"

# =====================
# Конфигурация 3DDFA_V2
# =====================

# Пути к файлам 3DDFA_V2
TDDFA_CONFIG = CONFIG_DIR / 'mb1_120x120.yml'
BFM_MODEL = MODELS_DIR / 'bfm_noneck_v3.pkl'

# Словарь обязательных файлов 3DDFA_V2
REQUIRED_3DDFA_FILES = {
    'TDDFA_CONFIG': TDDFA_CONFIG,
    'BFM_MODEL': BFM_MODEL,
    'weights_mb1_120x120': ROOT_DIR / 'weights' / 'mb1_120x120.pth',
    'onnx_model': ROOT_DIR / 'weights' / 'mb1_120x120.onnx'
}

# Адаптация для Apple Silicon (macOS ARM64)
if IS_MACOS and IS_ARM64:
    logging.info("Обнаружен Apple Silicon (M1/M2/M3). Адаптация конфигурации.")
    
    # ИСПРАВЛЕНО: CoreML не поддерживает все операции 3DDFA_V2
    # Используем только CPU для совместимости
    ONNX_EXECUTION_PROVIDERS = ['CPUExecutionProvider']
    logging.info("macOS M1: Используется только CPUExecutionProvider для совместимости с 3DDFA_V2")
    
    # Оптимизация для Apple Silicon
    import os
    os.environ['VECLIB_MAXIMUM_THREADS'] = '4'
    # Убираем конфликтующие OpenMP настройки
    if 'OMP_NUM_THREADS' in os.environ:
        del os.environ['OMP_NUM_THREADS']
    if 'KMP_DUPLICATE_LIB_OK' in os.environ:
        del os.environ['KMP_DUPLICATE_LIB_OK']
    
    USE_ONNX = True
    logging.info(f"USE_ONNX установлен в {USE_ONNX} для 3DDFA_V2 на M1.")
    
else:
    ONNX_EXECUTION_PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in onnxruntime.get_available_providers() else ['CPUExecutionProvider']
    
    if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
        logging.info("Обнаружен CUDA Execution Provider для ONNX Runtime.")
    else:
        logging.warning("CUDA Execution Provider не найден, используется CPUExecutionProvider для ONNX Runtime.")

def validate_3ddfa_files() -> bool:
    """Проверяет наличие всех необходимых файлов для 3DDFA_V2."""
    missing_files = []
    for key, path in REQUIRED_3DDFA_FILES.items():
        if not path.exists():
            missing_files.append(f"{key}: {path}")
            logging.error(f"Отсутствует файл 3DDFA_V2: {key} по пути {path}")

    if missing_files:
        logging.error(f"Не найдены следующие обязательные файлы 3DDFA_V2: {', '.join(missing_files)}")
        return False
    
    logging.info("Все обязательные файлы 3DDFA_V2 найдены.")
    return True

# =====================
# Конфигурация InsightFace
# =====================

INSIGHT_MODEL = 'buffalo_l'
INSIGHT_FACE_DET_THRESHOLD = 0.1
INSIGHT_FACE_DET_SIZE = (640, 640)

# Автоматическое определение ctx_id на основе архитектуры
if IS_MACOS and IS_ARM64:
    INSIGHT_CTX_ID = -1  # Принудительное использование CPU для Apple Silicon
    logging.info("INSIGHT_CTX_ID установлен в -1 (CPU) для Apple Silicon.")
else:
    INSIGHT_CTX_ID = 0  # GPU ID по умолчанию, -1 для CPU
    logging.info(f"INSIGHT_CTX_ID установлен в {INSIGHT_CTX_ID}. Проверьте доступность GPU.")

# =====================
# Временные константы
# =====================

PUTIN_BIRTH_DATE = date(1952, 10, 7) # Демонстрационная дата рождения, может быть заменена на общую константу или получена из данных
START_ANALYSIS_DATE = datetime(1999, 1, 1)
END_ANALYSIS_DATE = datetime(2025, 12, 31)

# Возрастные периоды для анализа
AGE_PERIODS = {
    'young_adult': (25, 35),
    'middle_age': (35, 50),
    'mature': (50, 65),
    'senior': (65, 80)
}

# =====================
# Параметры кластеризации DBSCAN
# =====================

DBSCAN_EPSILON = 0.35
DBSCAN_MIN_SAMPLES = 3
DBSCAN_METRIC = 'cosine'
DISTANCE_METRIC_DEFAULT = 'cosine'

# Адаптивные параметры DBSCAN по качеству изображений
ADAPTIVE_DBSCAN_PARAMS = {
    'high_quality': {'eps': 0.30, 'min_samples': 2},
    'medium_quality': {'eps': 0.35, 'min_samples': 3},
    'low_quality': {'eps': 0.40, 'min_samples': 4}
}

# =====================
# Медицинские константы старения
# =====================

class AgingModel:
    # Потеря эластичности кожи (% в год после 40 лет)
    ELASTICITY_LOSS_PER_YEAR = 0.015

    # Опущение мягких тканей (мм в год после 40 лет)
    TISSUE_SAGGING_PER_YEAR = 1.5

    # Возраст стабилизации костной структуры
    BONE_STABILITY_THRESHOLD = 25

    # Максимальное изменение межзрачкового расстояния после 25 лет
    IPD_VARIATION_MAX = 0.02

AGING_MODEL_ENABLED = False # По умолчанию модель старения отключена

# Коэффициенты возрастных изменений по зонам лица
AGING_COEFFICIENTS = {
    'forehead': 0.8,  # Менее подвержен изменениям
    'eye_area': 1.2,  # Более подвержен изменениям
    'cheeks': 1.0,    # Средняя подверженность
    'nose': 0.3,      # Минимальные изменения
    'mouth': 0.9,     # Умеренные изменения
    'chin_jaw': 0.7   # Относительно стабильная зона
}

TEMPORAL_STABILITY_THRESHOLD = 0.05  # Порог для временной стабильности
CONSECUTIVE_APPEARANCES_THRESHOLD = 2 # Количество последовательных появлений для подтверждения

# =====================
# Параметры хронологического анализа
# =====================

def get_chronological_analysis_parameters() -> Dict:
    """
    Возвращает параметры для хронологического анализа:
    min_appearance_count (int): минимальное количество появлений для подтверждения личности
    max_gap_days (int): максимальный разрыв между появлениями одного лица в днях
    confidence_threshold (float): минимальная уверенность для идентификации
    aging_tolerance_per_year (float): допустимое изменение метрик в год
    """
    return {
        'min_appearance_count': 3,
        'max_gap_days': 180,
        'confidence_threshold': 0.85,
        'aging_tolerance_per_year': 0.02
    }

# Пороги для детекции аномальных изменений
ANOMALY_DETECTION_THRESHOLDS = {
    'rapid_change': 0.15,      # Быстрое изменение метрик
    'reverse_aging': -0.05,    # Обратное старение
    'inconsistent_pattern': 0.20,  # Несогласованные паттерны
    'Z_SCORE_ANOMALY_THRESHOLD': 2.5, # Порог Z-score для детекции аномалий
    'RAPID_CHANGE_STD_MULTIPLIER': 3.0 # Множитель для стандартного отклонения при детекции быстрых изменений
}

# =====================
# Пороги каскадной верификации
# =====================

CASCADE_VERIFICATION_THRESHOLDS = {
    'geometry': 0.7,
    'embedding': 0.6,
    'texture': 0.5,
    'temporal': 0.6
}

# =====================
# Веса аутентичности
# =====================
AUTHENTICITY_WEIGHTS = {
    'geometry': 0.15,
    'embedding': 0.30,
    'texture': 0.10,
    'temporal_consistency': 0.15,
    'temporal_stability': 0.10,
    'aging_consistency': 0.10,
    'anomalies_score': 0.05,
    'liveness_score': 0.05
}

# =====================
# Константы для медицинского валидатора
# =====================

MEDICAL_VALIDATOR_THRESHOLDS = {
    'AGING_CONSISTENCY_SCALE_FACTOR': 2.0,
    'BONE_METRIC_STABILITY_THRESHOLD': 0.02,
    'SURGERY_MIN_INTERVAL_DAYS': 180,
    'MEDICAL_EVENT_CORRELATION_DAYS': 60,
    'PHYSIOLOGICAL_CHANGE_LIMIT': 0.02,
    'SOFT_TISSUE_AGING_SLOPE_THRESHOLD': -0.005 # Примерный порог для нормального "опущения"
}

# =====================
# Параметры детекции аномалий
# =====================

ANOMALY_DETECTION_ADVANCED_THRESHOLDS = {
    'SURGERY_MIN_INTERVAL_DAYS_ANOMALY': 180, # Минимальный интервал в днях для рассмотрения хирургии
    'MASK_QUALITY_JUMP_THRESHOLD': 0.15,      # Порог значительного улучшения качества маски
    'CROSS_SOURCE_CRITICAL_DISTANCE_THRESHOLD': 0.5 # Порог косинусного расстояния для критических аномалий между источниками
}

TECHNOLOGY_BREAKTHROUGH_YEARS = [2010, 2015, 2020] # Предполагаемые годы технологических прорывов в масках

# =====================
# Цвета для визуализации
# =====================

COLORS_RGB = {
    'RED': (0, 0, 255),
    'GREEN': (0, 255, 0),
    'BLUE': (255, 0, 0)
}

# =====================
# Стандартные размеры для нормализации
# =====================

STANDARD_IOD = 64.0          # Межзрачковое расстояние (пиксели)
STANDARD_NOSE_EYE = 45.0     # Расстояние нос-глаз
STANDARD_FACE_HEIGHT = 120.0  # Высота лица
STANDARD_PROFILE_HEIGHT = 140.0  # Высота профиля
MIN_VISIBILITY_Z = 0.0       # Минимальная Z-координата для видимых точек

# Коэффициенты для обработки bounding box
BBOX_CENTER_Y_OFFSET_FACTOR = 0.14 # Смещение центра Y для ROI (относительно размера bbox)
BBOX_SIZE_FACTOR = 1.58 # Множитель размера ROI относительно bbox

# =====================
# Конфигурация рендеринга 3D-модели
# =====================

RENDER_CONFIG = {
    'intensity_ambient': 0.5,
    'color_ambient': (0.0, 0.0, 0.0),
    'intensity_directional': 1.0,
    'color_directional': (0.0, 0.0, 1.0),
    'intensity_specular': 0.3,
    'specular_exp': 90,
    'light_pos': (0, 10, 5),
    'view_pos': (0, 0, 5)
}

# =====================
# Константы для построения 3D-рамки позы
# =====================

CAMERA_BOX_REAR_SIZE = 90 # Размер задней части рамки камеры
CAMERA_BOX_FRONT_SIZE_FACTOR = 4/3 # Коэффициент для расчета передней части рамки
CAMERA_BOX_FRONT_DEPTH_FACTOR = 4/3 # Коэффициент для расчета глубины передней части рамки

# Допустимые диапазоны для валидации
VALIDATION_RANGES = {
    'iod': (40.0, 100.0),
    'face_height': (80.0, 200.0),
    'face_width': (60.0, 150.0),
    'nose_width': (15.0, 40.0)
}

# =====================
# Пороги детекции масок по технологическим уровням
# =====================

MASK_DETECTION_THRESHOLDS = {
    'Level1_Primitive': {
        'years': (1999, 2005),
        'shape_error': 0.6,
        'entropy': 4.2,
        'embedding_dist': 0.8,
        'thickness': '3-5mm',
        'material': 'latex_basic',
        'characteristics': ['visible_seams', 'poor_texture', 'rigid_movement']
    },
    'Level2_Basic': {
        'years': (2006, 2010),
        'shape_error': 0.4,
        'entropy': 5.2,
        'embedding_dist': 0.7,
        'thickness': '2-3mm',
        'material': 'silicone_basic',
        'characteristics': ['improved_texture', 'better_fit', 'limited_expressions']
    },
    'Level3_Commercial': {
        'years': (2011, 2015),
        'shape_error': 0.3,
        'entropy': 6.0,
        'embedding_dist': 0.5,
        'thickness': '1-2mm',
        'material': 'advanced_silicone',
        'characteristics': ['realistic_texture', 'good_mobility', 'micro_details']
    },
    'Level4_Professional': {
        'years': (2016, 2020),
        'shape_error': 0.2,
        'entropy': 6.5,
        'embedding_dist': 0.4,
        'thickness': '<1mm',
        'material': 'medical_grade',
        'characteristics': ['pore_simulation', 'dynamic_expressions', 'color_matching']
    },
    'Level5_Advanced': {
        'years': (2021, 2025),
        'shape_error': 0.15,
        'entropy': 7.0,
        'embedding_dist': 0.3,
        'thickness': '0.5mm',
        'material': 'bio_compatible',
        'characteristics': ['perfect_texture', 'natural_aging', 'micro_expressions']
    }
}

# Годы технологических прорывов
BREAKTHROUGH_YEARS = [2008, 2014, 2019, 2022]

# =====================
# Веса для формулы аутентичности
# =====================

ANALYSIS_WEIGHTS = {
    'cranial_structure': 0.25,
    'facial_proportions': 0.25,
    'soft_tissue_analysis': 0.20,
    'temporal_consistency': 0.15,
    'texture_authenticity': 0.15
}

# =====================
# Пороги для различных типов анализа
# =====================

EMBEDDING_DRIFT_THRESHOLD = 0.1
MILD_ASYMMETRY_THRESHOLD = 0.05
SEVERE_ASYMMETRY_THRESHOLD = 0.1
LANDMARK_MEAN_DISTANCE_THRESHOLD = 5.0
LANDMARK_MAX_DISTANCE_THRESHOLD = 20.0
EYE_REGION_ERROR_THRESHOLD = 0.6

# Пороги для текстурного анализа
TEXTURE_ANALYSIS_THRESHOLDS = {
    'entropy_natural_min': 6.2,
    'entropy_natural_max': 7.8,
    'lbp_uniformity_min': 0.6,
    'gabor_energy_min': 100.0,
    'haralick_contrast_max': 0.8,
    'haralick_homogeneity_min': 0.8,
    'fourier_spectral_centroid_min': 50,
    'fourier_dominant_frequency_min': 50,
    'lbp_variance_min': 0.005,
    'lbp_entropy_mean': 4.0,
    'lbp_entropy_std': 0.5,
    'gabor_energy_mean': 150.0,
    'gabor_energy_std': 50.0,
    'pore_min_area': 3,
    'pore_max_area': 200,
    'pore_block_size': 35,
    'pore_offset': 0.05,
    'pore_selem_open_disk_size': 1,
    'pore_selem_close_disk_size': 2,
    'canny_threshold1': 100,
    'canny_threshold2': 200,
    'search_radius': 10
}

# =====================
# Конфигурации ракурсов
# =====================

def get_view_configs() -> Dict:
    """Конфигурации для 4 основных ракурсов лица."""
    return {
        'Frontal': {
            'yaw_range': (0, 15),
            'pitch_range': (-10, 10),
            'roll_range': (-5, 5),
            'target_angles': (0, 0, 0),
            'reference_points': [36, 45],  # Внешние углы глаз
            'scale_metric': 'IOD',
            'standard_value': STANDARD_IOD,
            'quality_weight': 1.0,
            'landmarks_visibility': list(range(68))
        },
        'Frontal_Edge': {
            'yaw_range': (15, 35),
            'pitch_range': (-15, 15),
            'roll_range': (-10, 10),
            'target_angles': (0, 25, 0),
            'reference_points': [27, 30, 36, 45],
            'scale_metric': 'NOSE_EYE',
            'standard_value': STANDARD_NOSE_EYE,
            'quality_weight': 0.9,
            'landmarks_visibility': list(range(17)) + list(range(27, 68))
        },
        'Semi_Profile': {
            'yaw_range': (35, 65),
            'pitch_range': (-20, 20),
            'roll_range': (-15, 15),
            'target_angles': (0, 50, 0),
            'reference_points': [8, 27, 31, 33, 35],
            'scale_metric': 'FACE_HEIGHT',
            'standard_value': STANDARD_FACE_HEIGHT,
            'quality_weight': 0.8,
            'landmarks_visibility': list(range(9)) + list(range(27, 36)) + list(range(48, 68))
        },
        'Profile': {
            'yaw_range': (65, 90),
            'pitch_range': (-25, 25),
            'roll_range': (-20, 20),
            'target_angles': (0, 85, 0),
            'reference_points': [1, 3, 8, 13, 15, 27, 30],
            'scale_metric': 'PROFILE_HEIGHT',
            'standard_value': STANDARD_PROFILE_HEIGHT,
            'quality_weight': 0.7,
            'landmarks_visibility': list(range(9)) + [27, 28, 29, 30, 31, 32, 33, 34, 35]
        }
    }

# =====================
# Метрики идентификации личности
# =====================

def get_identity_signature_metrics() -> Dict:
    """
    Возвращает метрики и веса для расчета сигнатуры идентичности.
    """
    return {} # Теперь возвращаем пустой словарь, так как AUTHENTICITY_WEIGHTS теперь глобальная переменная

# =====================
# Пороги качества изображений
# =====================

IMAGE_QUALITY_THRESHOLDS = {
    'MIN_FACE_SIZE': 100, # Минимальный размер лица в пикселях для обработки
    'BLUR_DETECTION_THRESHOLD': 0.5, # Порог размытия (меньше = более размыто)
    'DEFAULT_QUALITY_THRESHOLD': 0.6 # Порог качества по умолчанию для Gradio GUI
}

# =====================
# Параметры Gradio Interface
# =====================

GRADIO_INTERFACE_SETTINGS = {
    'MAX_CONCURRENT_PROCESSES': 4, # Ограничение параллельных процессов
    'BATCH_SIZE': 50,              # Размер батча для обработки
    'MAX_GALLERY_ITEMS': 100,      # Максимум элементов в галерее
    'MAX_FILE_UPLOAD_COUNT': 1500  # Максимальное количество файлов для загрузки в GUI
}

GRADIO_DEFAULTS = {
    'DBSCAN_EPSILON_DEFAULT': 0.35,
    'DBSCAN_MIN_SAMPLES_DEFAULT': 3,
    'CONFIDENCE_FILTER_DEFAULT': 0.5,
    'TEMPORAL_RESOLUTION_DEFAULT': "Квартал",
    'TEMPORAL_METRICS_DEFAULT': ["skull_width_ratio", "nose_width_ratio"],
    'MASK_DETECTION_SENSITIVITY_DEFAULT': 0.7,
    'MASK_TECH_LEVELS_DEFAULT': ["Level3_Commercial", "Level4_Professional", "Level5_Advanced"],
    'REPORT_TYPE_DEFAULT': "Подробный",
    'REPORT_SECTIONS_DEFAULT': [
        "Исполнительное резюме",
        "Результаты кластеризации",
        "Временной анализ",
        "Заключение"
    ],
    'REPORT_CONFIDENCE_THRESHOLD_DEFAULT': 0.85
}

# =====================
# Пороги анализа эмбеддингов
# =====================
EMBEDDING_ANALYSIS_THRESHOLDS = {
    'MIN_COSINE_SIMILARITY': 0.6, # Минимальное косинусное сходство для подтверждения
    'MAX_EUCLIDEAN_DISTANCE': 1.0, # Максимальное евклидово расстояние
    'RECOGNITION_CONFIDENCE': 0.8 # Порог уверенности для распознавания лица
}

FACE_CONFIDENCE_THRESHOLD_EMBEDDING = 0.8 # Порог уверенности для эмбеддинга лица при детекции

# =====================
# Пороги для дрейфа эмбеддингов
# =====================

EMBEDDING_DRIFT_THRESHOLDS = {
    'DRIFT_DISTANCE_THRESHOLD': 0.1 # Порог косинусного расстояния для детекции дрейфа эмбеддингов
}

# =====================
# Конфигурация логирования
# =====================

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
        }
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'DEBUG',
            'formatter': 'detailed',
            'class': 'logging.FileHandler',
            'filename': str(LOGS_DIR / 'analysis.log'),
            'mode': 'a',
        }
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}

# =====================
# Функции валидации и калибровки
# =====================

def validate_configuration() -> bool:
    """Проверяет корректность всех конфигурационных параметров."""
    validation_errors = []
    
    # Проверка путей
    if not validate_3ddfa_files():
        validation_errors.append("Отсутствуют файлы 3DDFA_V2")
    
    # Проверка наличия ONNX Execution Providers
    if USE_ONNX and not ONNX_EXECUTION_PROVIDERS:
        validation_errors.append("ONNX включен, но доступные провайдеры выполнения не найдены.")
    
    # Проверка временных интервалов
    if START_ANALYSIS_DATE >= END_ANALYSIS_DATE:
        validation_errors.append("Некорректный временной интервал анализа")
    
    # Проверка пороговых значений
    if DBSCAN_EPSILON <= 0 or DBSCAN_EPSILON >= 1:
        validation_errors.append("Некорректное значение DBSCAN_EPSILON")
    
    # Проверка весов аутентичности
    if abs(sum(AUTHENTICITY_WEIGHTS.values()) - 1.0) > 0.001:
        validation_errors.append("Сумма весов аутентичности не равна 1.0")
    
    if validation_errors:
        logging.error(f"Ошибки конфигурации: {validation_errors}")
        return False
    
    return True

def auto_calibrate_thresholds(historical_data: List[Dict]) -> Dict:
    """Автоматическая калибровка порогов на основе исторических данных."""
    if not historical_data:
        logging.warning("Нет исторических данных для калибровки")
        return {}
    
    calibrated_thresholds = {}
    
    # Калибровка DBSCAN epsilon на основе распределения расстояний
    if len(historical_data) > 10:
        distances = []
        for i in range(len(historical_data)):
            for j in range(i+1, len(historical_data)):
                if 'embedding' in historical_data[i] and 'embedding' in historical_data[j]:
                    dist = np.linalg.norm(historical_data[i]['embedding'] - historical_data[j]['embedding'])
                    distances.append(dist)
        
        if distances:
            calibrated_thresholds['epsilon'] = np.percentile(distances, 25)
    
    # Калибровка порогов аномалий
    shape_errors = [item.get('shape_error', 0) for item in historical_data if 'shape_error' in item]
    if shape_errors:
        calibrated_thresholds['shape_error_threshold'] = np.percentile(shape_errors, 95)
    
    # Калибровка энтропийных порогов
    entropies = [item.get('entropy', 0) for item in historical_data if 'entropy' in item]
    if entropies:
        calibrated_thresholds['entropy_threshold'] = np.percentile(entropies, 5)
    
    logging.info(f"Калиброванные пороги: {calibrated_thresholds}")
    return calibrated_thresholds

def get_age_adjusted_thresholds(current_age: int) -> Dict:
    """
    Возвращает пороги, скорректированные по возрасту, для детекции аномалий.
    Пороги адаптируются, чтобы быть более или менее строгими в зависимости от возраста,
    отражая естественные возрастные изменения и их вариабельность.
    """
    adjusted_thresholds = ANOMALY_DETECTION_THRESHOLDS.copy()

    # Примеры адаптации порогов по возрасту:

    # 1. rapid_change (Быстрое изменение метрик)
    # Для молодых людей (до 30) резкие изменения менее естественны -> более строгий порог
    # Для пожилых (после 60) естественная динамика может быть быстрее -> более мягкий порог
    if current_age < 30:
        adjusted_thresholds['rapid_change'] *= 0.8  # Более строгий
    elif current_age > 60:
        adjusted_thresholds['rapid_change'] *= 1.2  # Менее строгий

    # 2. reverse_aging (Обратное старение)
    # Всегда должно быть очень чувствительно, но для пожилых это может быть более необычно
    if current_age > 50:
        adjusted_thresholds['reverse_aging'] *= 0.8 # Сделать порог для 'омоложения' более строгим для пожилых

    # 3. Z_SCORE_ANOMALY_THRESHOLD (Порог Z-score для детекции аномалий)
    # Для молодых (меньше вариаций) -> более строгий Z-score
    # Для пожилых (больше естественных вариаций) -> менее строгий Z-score
    if current_age < 30:
        adjusted_thresholds['Z_SCORE_ANOMALY_THRESHOLD'] *= 1.1 # Выше Z-score, значит аномалия должна быть более выраженной
    elif current_age > 60:
        adjusted_thresholds['Z_SCORE_ANOMALY_THRESHOLD'] *= 0.9 # Ниже Z-score, легче обнаружить аномалию

    # 4. RAPID_CHANGE_STD_MULTIPLIER (Множитель для стандартного отклонения при детекции быстрых изменений)
    # Молодые: меньше естественных быстрых изменений -> меньше множитель
    # Пожилые: больше естественных быстрых изменений -> больше множитель
    if current_age < 30:
        adjusted_thresholds['RAPID_CHANGE_STD_MULTIPLIER'] *= 0.9
    elif current_age > 60:
        adjusted_thresholds['RAPID_CHANGE_STD_MULTIPLIER'] *= 1.1

    # Можно добавить логику для других порогов, если это необходимо
    # Например, для 'inconsistent_pattern'
    if current_age > 50:
        adjusted_thresholds['inconsistent_pattern'] *= 1.1 # Снижаем чувствительность для пожилых, так как паттерны могут быть менее стабильными
    
    return adjusted_thresholds

def save_configuration_snapshot(output_path: str) -> bool:
    """Сохраняет снимок текущей конфигурации."""
    try:
        config_snapshot = {
            'timestamp': datetime.now().isoformat(),
            'dbscan_params': {
                'epsilon': DBSCAN_EPSILON,
                'min_samples': DBSCAN_MIN_SAMPLES,
                'metric': DBSCAN_METRIC
            },
            'thresholds': MASK_DETECTION_THRESHOLDS,
            'weights': AUTHENTICITY_WEIGHTS,
            'view_configs': get_view_configs(),
            'identity_metrics': get_identity_signature_metrics(),
            'validation_status': validate_configuration()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config_snapshot, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Снимок конфигурации сохранен: {output_path}")
        return True
        
    except Exception as e:
        logging.error(f"Ошибка сохранения конфигурации: {e}")
        return False

# =====================
# Инициализация при импорте
# =====================

# Проверка конфигурации при импорте модуля
if not validate_configuration():
    logging.warning("Обнаружены проблемы в конфигурации. Проверьте настройки.")

# Создание снимка конфигурации при запуске
config_snapshot_path = CONFIG_DIR / f"config_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
save_configuration_snapshot(str(config_snapshot_path))

logging.info("Конфигурация системы анализа лицевой идентичности загружена")

# =====================
# Константы для низкоуровневого рендера (ctypes)
# =====================

CTYPES_RENDER_LIGHTING_CONFIG = {
    'light': (0, 0, 5),
    'direction': (0.6, 0.6, 0.6),
    'ambient': (0.3, 0.3, 0.3)
}

# =====================
# Пороги детекции масок
# =====================
MASK_DETECTION_THRESHOLDS = {
    'MASK_CONFIDENCE_THRESHOLD': 0.7, # Порог уверенности для детекции маски
    'MASK_OVERLAP_THRESHOLD': 0.6, # Порог перекрытия маски с лицом
    'MASK_OR_DOUBLE_THRESHOLD': 0.4, # Порог для определения маски или двойника (для _interpret_authenticity_score)
    'REQUIRES_ANALYSIS_THRESHOLD': 0.6 # Порог для случая, требующего дополнительного анализа (для _interpret_authenticity_score)
}

# Дополнительные пороги для анализа эмбеддингов
EMBEDDING_ANALYSIS_THRESHOLDS.update({
    'DIMENSION_ANOMALY_THRESHOLD': 2.5,
    'TEXTURE_ANOMALY_DIMS': (45, 67),
    'GEOMETRIC_ANOMALY_DIMS': (120, 145),
    'LIGHTING_ANOMALY_DIMS': (200, 230),
    'APPEARANCE_SATURATION_FACTOR': 50
})

# Пороги для временной стабильности
TEMPORAL_STABILITY_THRESHOLD = {
    'COEFFICIENT_OF_VARIATION_THRESHOLD': 0.3
}
