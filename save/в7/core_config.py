"""
CoreConfig - Центральная конфигурация системы анализа 3D лиц
Версия: 2.0
Дата: 2025-06-15
"""

import os
import logging
import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, date
import numpy as np
from dataclasses import dataclass, asdict
import hashlib

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('logs/coreconfig.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== ВЕРСИЯ СИСТЕМЫ ====================
SYSTEM_VERSION = "2.0"

# ==================== ОСНОВНЫЕ ПУТИ ====================
try:
    ROOT_DIR = Path(__file__).parent.absolute()
    DATA_DIR = ROOT_DIR / "data"
    MODELS_DIR = ROOT_DIR / "models"
    RESULTS_DIR = ROOT_DIR / "results"
    CONFIG_DIR = ROOT_DIR / "configs"
    CACHE_DIR = ROOT_DIR / "cache"
    LOGS_DIR = ROOT_DIR / "logs"
    TEMPLATE_DIR = ROOT_DIR / "templates"
    WEIGHTS_DIR = ROOT_DIR / "weights"
    
    # Создание директорий
    for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, CONFIG_DIR, CACHE_DIR, LOGS_DIR, TEMPLATE_DIR, WEIGHTS_DIR]:
        directory.mkdir(exist_ok=True)
        logger.info(f"Директория создана/проверена: {directory}")
        
except Exception as e:
    logger.error(f"Ошибка создания директорий: {e}")
    raise

# ==================== ИСПРАВЛЕННЫЕ ВЕСА АУТЕНТИЧНОСТИ ====================
# Согласно правкам: 7 компонентов вместо 4
AUTHENTICITY_WEIGHTS = {
    "geometry": 0.15,           # Геометрические метрики
    "embedding": 0.30,          # Эмбеддинги лица
    "texture": 0.10,            # Текстурный анализ
    "temporal_consistency": 0.15, # Временная согласованность
    "temporal_stability": 0.10,  # Временная стабильность
    "aging_consistency": 0.10,   # Согласованность старения
    "anomalies_score": 0.05,     # Оценка аномалий
    "liveness_score": 0.05       # Оценка живости
}

# Проверка суммы весов
total_weight = sum(AUTHENTICITY_WEIGHTS.values())
if abs(total_weight - 1.0) > 1e-6:
    logger.error(f"Сумма весов аутентичности не равна 1.0: {total_weight}")
    raise ValueError(f"Неверная сумма весов: {total_weight}")
else:
    logger.info(f"Веса аутентичности валидны, сумма: {total_weight}")

# ==================== ПАРАМЕТРЫ DBSCAN ====================
DBSCAN_PARAMS = {
    "epsilon": 0.35,           # Радиус соседства
    "min_samples": 3,          # Минимальное количество точек в кластере
    "metric": "cosine"         # Метрика расстояния
}

# ==================== ПОРОГИ АНОМАЛИЙ ====================
ANOMALY_THRESHOLDS = {
    "rapid_change": 0.15,              # Быстрые изменения
    "reverse_aging": -0.05,            # Обратное старение
    "inconsistent_pattern": 0.20,      # Несогласованные паттерны
    "z_score_threshold": 2.5,          # Z-score для аномалий
    "rapid_change_std_multiplier": 3.0  # Множитель std для быстрых изменений
}

# ==================== ИСПРАВЛЕННЫЕ УРОВНИ МАСОК ====================
# Согласно правкам: 5 уровней по годам с правильными параметрами
MASK_DETECTION_LEVELS = {
    "Level1_Primitive": {
        "years": (1999, 2005),
        "shape_error": 0.6,
        "entropy": 4.2,
        "embedding_dist": 0.8,
        "thickness_mm": (3, 5),
        "description": "Примитивные маски 1999-2005"
    },
    "Level2_Basic": {
        "years": (2006, 2010),
        "shape_error": 0.4,
        "entropy": 5.2,
        "embedding_dist": 0.7,
        "thickness_mm": (2, 3),
        "description": "Базовые маски 2006-2010"
    },
    "Level3_Commercial": {
        "years": (2011, 2015),
        "shape_error": 0.3,
        "entropy": 6.0,
        "embedding_dist": 0.5,
        "thickness_mm": (1, 2),
        "description": "Коммерческие маски 2011-2015"
    },
    "Level4_Professional": {
        "years": (2016, 2020),
        "shape_error": 0.2,
        "entropy": 6.5,
        "embedding_dist": 0.4,
        "thickness_mm": (1, 1),
        "description": "Профессиональные маски 2016-2020"
    },
    "Level5_Advanced": {
        "years": (2021, 2025),
        "shape_error": 0.15,
        "entropy": 7.0,
        "embedding_dist": 0.3,
        "thickness_mm": (0.5, 0.5),
        "description": "Продвинутые маски 2021-2025"
    }
}

# ==================== BREAKTHROUGH YEARS ====================
BREAKTHROUGH_YEARS = [2008, 2014, 2019, 2022]
logger.info(f"Breakthrough years для технологий масок: {BREAKTHROUGH_YEARS}")

# ==================== ИСПРАВЛЕННАЯ AGING MODEL ====================
# Согласно правкам: обновленные параметры
AGING_MODEL = {
    "elasticity_loss_per_year": 0.015,    # Потеря эластичности в год (исправлено)
    "tissue_sagging_per_year": 1.51,      # Провисание тканей в год (исправлено)
    "bone_stability_threshold": 25,        # Возраст стабильности костей
    "ipd_stability": True,                 # Стабильность межзрачкового расстояния
    "skull_growth_stops_age": 18,          # Возраст остановки роста черепа
    "soft_tissue_aging_rate": 0.02,       # Скорость старения мягких тканей
    "baseline_year": 1999                  # Базовый год для расчетов
}

# ==================== EMBEDDING ANALYSIS THRESHOLDS ====================
# Согласно правкам: добавлены диапазоны аномалий по измерениям
EMBEDDING_ANALYSIS_THRESHOLDS = {
    "dimension_anomaly_threshold": 2.5,
    "texture_anomaly_dims": (45, 67),      # Диапазон для текстурных аномалий
    "geometric_anomaly_dims": (120, 145),   # Диапазон для геометрических аномалий
    "lighting_anomaly_dims": (200, 230),    # Диапазон для световых аномалий
    "min_confidence": 0.5,
    "max_embedding_drift": 0.3,
    "age_corrected_drift_enabled": True
}

# ==================== СТАНДАРТНЫЕ ИЗМЕРЕНИЯ ЛИЦА ====================
STANDARD_IOD = 64           # межзрачковое расстояние, px
STANDARD_NOSE_EYE = 45      # Frontal-Edge, px
STANDARD_FACE_HEIGHT = 120  # Semi-Profile, px
STANDARD_PROFILE_HEIGHT = 140  # Profile, px
MIN_VISIBILITY_Z = 0.01     # минимальная видимость

# ==================== VIEW CONFIGURATIONS ====================
# Согласно правкам: 4 категории поз с target_angles
VIEW_CONFIGS = {
    "Frontal": {
        "yaw_range": (0, 15),
        "target_angles": (0, 0, 0),  # pitch, yaw, roll
        "reference_points": [36, 45],   # внешние уголки глаз
        "scale_type": "IOD",
        "standard": STANDARD_IOD,
    },
    "Frontal_Edge": {
        "yaw_range": (15, 35),
        "target_angles": (0, 25, 0),
        "reference_points": [27, 30, "visible_eye", "visible_mouth"],
        "scale_type": "NOSE_EYE",
        "standard": STANDARD_NOSE_EYE,
    },
    "Semi_Profile": {
        "yaw_range": (35, 65),
        "target_angles": (0, 50, 0),
        "reference_points": [8, 27, 31, 33, 35, "visible_eye"],
        "scale_type": "FACE_HEIGHT",
        "standard": STANDARD_FACE_HEIGHT,
    },
    "Profile": {
        "yaw_range": (65, 90),
        "target_angles": (0, 85, 0),
        "reference_points": [1, 3, 8, 13, 15, 27, 30],
        "scale_type": "PROFILE_HEIGHT",
        "standard": STANDARD_PROFILE_HEIGHT,
    }
}

# ==================== 3DDFA КОНФИГУРАЦИЯ ====================
USE_ONNX = False  # Использовать ONNX модель
IS_MACOS = os.uname().sysname == "Darwin" if hasattr(os, 'uname') else False
IS_ARM64 = os.uname().machine == "arm64" if hasattr(os, 'uname') else False

ONNX_EXECUTION_PROVIDERS = ['CPUExecutionProvider']
if not (IS_MACOS and IS_ARM64):
    ONNX_EXECUTION_PROVIDERS.insert(0, 'CUDAExecutionProvider')

CAMERA_BOX_REAR_SIZE = 100 # Добавлена константа для FaceBoxes
CAMERA_BOX_FRONT_SIZE_FACTOR = 1.2

REQUIRED_3DDFA_FILES = {
    "config": CONFIG_DIR / "mb1_120x120.yml",
    "weights": WEIGHTS_DIR / "mb1_120x120.pth",
    "onnx_model": MODELS_DIR / "mb1_120x120.onnx",
    "bfm_model": CONFIG_DIR / "bfm_noneck_v3.pkl"
}

# ==================== КРИТИЧЕСКИЕ ПОРОГИ ====================
# Согласно правкам: обновленные пороги
CRITICAL_THRESHOLDS = {
    "min_authenticity_score": 0.6,
    "min_quality_score": 0.2,
    "max_shape_error": 0.3,
    "min_cluster_size": 3,
    "max_temporal_gap_days": 180,
    "min_confidence_threshold": 0.5,
    "anomaly_z_score_threshold": 2.5,
    "cross_source_critical_distance": 0.5,      # Добавлено согласно правкам
    "landmark_mean_distance_threshold": 3.0,     # Добавлено согласно правкам
    "severe_asymmetry_threshold": 0.1,           # Добавлено согласно правкам
    "eye_region_error_threshold": 0.6,
    "mild_asymmetry_threshold": 0.05,
    "FEATURE_CHANGE_THRESHOLD": 0.1 # Добавлен новый порог для анализа изменений признаков
}

# ==================== ПАРАМЕТРЫ ВИЗУАЛИЗАЦИИ ====================
VISUALIZATION_PARAMS = {
    "height": 600,
    "width": 800,
    "color_scheme": "viridis",
    "interactive": True,          # Согласно правкам
    "show_legend": True,
    "title": "3D Face Analysis",
    "wireframe_enabled": True,    # Для 3D визуализации
    "dense_points_enabled": True, # Для плотных точек
    "confidence_colors": {
        "high": "#00FF00",
        "medium": "#FFFF00", 
        "low": "#FF0000"
    }
}

# ==================== ЛИМИТЫ СИСТЕМЫ ====================
MAX_RESULTS_CACHE_SIZE = 10000        # Согласно правкам
MAX_FILE_UPLOAD_COUNT = 1500          # Согласно правкам (было неопределено)
MAX_BATCH_SIZE = 50

# ==================== КОДЫ ОШИБОК ====================
ERROR_CODES = {
    "E001": "Лицо не обнаружено",
    "E002": "Низкое качество изображения",
    "E003": "Несоответствие 3D-модели",
    "E004": "Аномалии в эмбеддингах",
    "E005": "Несоответствие текстуры кожи",
    "E006": "Несогласованность временных данных",
    "E007": "Неожиданные аномалии",
    "E008": "Не пройдена медицинская валидация",
    "E009": "Недостаточно данных для анализа",
    "E010": "Ошибка инициализации компонента",
    "E011": "Ошибка загрузки данных",
    "E012": "Ошибка обработки файла",
    "E013": "Ошибка рендеринга",
    "E014": "Ошибка экспорта отчета",
    "E015": "Неподдерживаемый формат",
    "E016": "Внутренняя ошибка системы",
    "E017": "Ошибка API",
    "E018": "Неверные входные параметры",
    "E019": "Ошибка самотестирования компонента",
    "E020": "Отсутствие необходимого файла/ресурса"
}

# ==================== ИСТОРИЧЕСКИЕ ДАННЫЕ ====================
PUTIN_BIRTH_DATE = date(1952, 10, 7)
START_ANALYSIS_DATE = datetime(1999, 1, 1)
END_ANALYSIS_DATE = datetime(2025, 12, 31)

# ==================== ЦВЕТОВЫЕ СХЕМЫ ====================
COLORS_RGB = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128)
}

# ==================== DATACLASS ДЛЯ КОНФИГУРАЦИИ ====================
@dataclass
class ConfigSnapshot:
    """Снимок конфигурации для версионирования"""
    timestamp: str
    version: str
    authenticity_weights: Dict[str, float]
    critical_thresholds: Dict[str, float]
    aging_model: Dict[str, Union[float, int, bool]]
    hash: str

# ==================== ФУНКЦИИ КОНФИГУРАЦИИ ====================

def get_view_configs() -> Dict[str, Dict]:
    """Получить конфигурации видов"""
    logger.info("Получение конфигураций видов")
    return VIEW_CONFIGS.copy()

def get_identity_signature_metrics() -> Dict[str, List[str]]:
    """Получить 15 метрик идентичности, разделенных по группам"""
    logger.info("Получение метрик идентичности")
    
    metrics = {
        "skull_geometry_signature": [
            "skull_width_ratio", "skull_depth_ratio",
            "forehead_height_ratio", "temple_width_ratio",
            "zygomatic_arch_width", "occipital_curve"],
        "facial_proportions_signature": [
            "eye_distance_ratio", "nose_width_ratio",
            "mouth_width_ratio", "chin_width_ratio",
            "jaw_angle_ratio", "forehead_angle"],
        "bone_structure_signature": [
            "nose_projection_ratio", "chin_projection_ratio",
            "jaw_line_angle"],
    }
    
    # Проверка: должно быть ровно 15 метрик
    total_metrics = sum(len(group) for group in metrics.values())
    if total_metrics != 15:
        logger.error(f"Неверное количество метрик: {total_metrics}, ожидается 15")
        raise ValueError(f"Должно быть 15 метрик, получено {total_metrics}")
    
    logger.info(f"Загружено {total_metrics} метрик идентичности")
    return metrics

def get_mask_detection_thresholds() -> Dict[str, Dict]:
    """Получить пороги обнаружения масок по уровням"""
    logger.info("Получение порогов обнаружения масок")
    return MASK_DETECTION_LEVELS.copy()

def get_aging_model_parameters() -> Dict[str, Union[float, int, bool]]:
    """Получить параметры модели старения"""
    logger.info("Получение параметров модели старения")
    return AGING_MODEL.copy()

def get_chronological_analysis_parameters() -> Dict[str, Union[int, float]]:
    """Получить параметры хронологического анализа"""
    logger.info("Получение параметров хронологического анализа")
    
    return {
        "min_appearance_count": 3,
        "max_gap_days": 180,
        "confidence_threshold": 0.85,
        "aging_tolerance_per_year": 0.02,
        "systematic_pattern_threshold": 0.15,
        "temporal_stability_threshold": 0.8
    }

def validate_configuration() -> bool:
    """Валидация целостности конфигурации"""
    logger.info("Начало валидации конфигурации")
    
    try:
        # Проверка весов аутентичности
        total_weight = sum(AUTHENTICITY_WEIGHTS.values())
        if abs(total_weight - 1.0) > 1e-6:
            logger.error(f"Сумма весов аутентичности: {total_weight}")
            return False
        
        # Проверка наличия всех 7 компонентов весов
        required_weights = {
            "geometry", "embedding", "texture", "temporal_consistency",
            "temporal_stability", "aging_consistency", "anomalies_score", "liveness_score"
        }
        if set(AUTHENTICITY_WEIGHTS.keys()) != required_weights:
            logger.error(f"Неверные ключи весов: {set(AUTHENTICITY_WEIGHTS.keys())}")
            return False
        
        # Проверка параметров DBSCAN
        if not (DBSCAN_PARAMS["epsilon"] == 0.35 and DBSCAN_PARAMS["min_samples"] == 3):
            logger.error(f"Неверные параметры DBSCAN: epsilon={DBSCAN_PARAMS['epsilon']}, min_samples={DBSCAN_PARAMS['min_samples']}")
            raise ValueError("Параметры DBSCAN должны быть eps=0.35, min_samples=3")
        
        # Проверка уровней масок
        if len(MASK_DETECTION_LEVELS) != 5:
            logger.error(f"Неверное количество уровней масок: {len(MASK_DETECTION_LEVELS)}")
            raise ValueError(f"Должно быть 5 уровней масок, получено {len(MASK_DETECTION_LEVELS)}")
        
        # Проверка метрик идентичности
        identity_metrics = get_identity_signature_metrics()
        if len(identity_metrics["skull_geometry_signature"]) != 6:
            logger.error(f"Неверное количество метрик в skull_geometry_signature: {len(identity_metrics['skull_geometry_signature'])}")
            raise ValueError("skull_geometry_signature должен содержать ровно 6 метрик")
        if len(identity_metrics["bone_structure_signature"]) != 3:
            logger.error(f"Неверное количество метрик в bone_structure_signature: {len(identity_metrics['bone_structure_signature'])}")
            raise ValueError("bone_structure_signature должен содержать ровно 3 метрики")
        
        # Проверка параметров хронологического анализа
        chrono_params = get_chronological_analysis_parameters()
        if abs(chrono_params["confidence_threshold"] - 0.85) > 1e-6:
            logger.error(f"Неверный confidence_threshold: {chrono_params['confidence_threshold']}")
            raise ValueError("confidence_threshold должен быть равен 0.85")
        
        # Проверка наличия константы MIN_VISIBILITY_Z
        if MIN_VISIBILITY_Z is None:
            logger.error("Отсутствует константа MIN_VISIBILITY_Z")
            raise ValueError("MIN_VISIBILITY_Z должен быть определен")
        
        # Проверка диапазонов эмбеддингов
        texture_range = EMBEDDING_ANALYSIS_THRESHOLDS["texture_anomaly_dims"]
        geometric_range = EMBEDDING_ANALYSIS_THRESHOLDS["geometric_anomaly_dims"]
        lighting_range = EMBEDDING_ANALYSIS_THRESHOLDS["lighting_anomaly_dims"]
        
        if not (texture_range == (45, 67) and geometric_range == (120, 145) and lighting_range == (200, 230)):
            logger.error("Неверные диапазоны аномалий эмбеддингов")
            raise ValueError("Диапазоны аномалий эмбеддингов должны быть (45, 67), (120, 145), (200, 230)")
        
        # Проверка наличия файлов моделей
        for file_key, file_path in REQUIRED_3DDFA_FILES.items():
            if not file_path.parent.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)
                logger.warning(f"Создана директория для {file_key}: {file_path.parent}")
        
        logger.info("Валидация конфигурации успешно завершена")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка валидации конфигурации: {e}")
        return False

def auto_calibrate_thresholds_historical_data(historical_data: Optional[List[Dict]] = None) -> Dict[str, float]:
    """
    Автокалибровка порогов на исторических данных
    Согласно правкам: реализация отсутствовала
    """
    global MASK_DETECTION_LEVELS # Declare intent to modify global variable
    logger.info("Начало автокалибровки порогов")
    
    if not historical_data:
        logger.warning("Исторические данные не предоставлены, используются значения по умолчанию")
        return CRITICAL_THRESHOLDS.copy()
    
    try:
        calibrated_thresholds = CRITICAL_THRESHOLDS.copy()
        
        # Извлечение метрик из исторических данных для общей калибровки
        authenticity_scores = [item.get("authenticity_score", 0.5) for item in historical_data]
        shape_errors_overall = [item.get("shape_error", 0.2) for item in historical_data]
        
        if authenticity_scores:
            # Калибровка на основе percentile
            calibrated_thresholds["min_authenticity_score"] = np.percentile(authenticity_scores, 25)
            logger.info(f"Калиброван min_authenticity_score: {calibrated_thresholds['min_authenticity_score']:.2f}")
        
        if shape_errors_overall:
            calibrated_thresholds["max_shape_error"] = np.percentile(shape_errors_overall, 75)
            logger.info(f"Калиброван max_shape_error: {calibrated_thresholds['max_shape_error']:.2f}")
        
        # Автокалибровка для MASK_DETECTION_LEVELS по эпохам
        epoch_shape_errors = {level_name: [] for level_name in MASK_DETECTION_LEVELS.keys()}
        epoch_entropies = {level_name: [] for level_name in MASK_DETECTION_LEVELS.keys()}

        for item in historical_data:
            item_date = item.get('date') # Предполагается, что 'date' является объектом datetime.date
            item_shape_error = item.get('shape_error')
            item_entropy = item.get('entropy')

            if item_date and isinstance(item_date, (date, datetime)) and \
               item_shape_error is not None and item_entropy is not None:
                item_year = item_date.year
                for level_name, level_config in MASK_DETECTION_LEVELS.items():
                    start_year, end_year = level_config['years']
                    if start_year <= item_year <= end_year:
                        epoch_shape_errors[level_name].append(item_shape_error)
                        epoch_entropies[level_name].append(item_entropy)
                        break
            elif item_date is None or not isinstance(item_date, (date, datetime)):
                logger.warning(f"Данные для калибровки по эпохам пропущены: отсутствует 'date' или неверный формат в {item}")
            else:
                logger.warning(f"Данные для калибровки по эпохам пропущены: отсутствует 'shape_error' или 'entropy' в {item}")

        for level_name, level_config in MASK_DETECTION_LEVELS.items():
            # Калибровка shape_error
            if epoch_shape_errors[level_name]:
                level_mean_shape_error = np.mean(epoch_shape_errors[level_name])
                spec_shape_error = level_config['shape_error']
                calibrated_value = max(level_mean_shape_error * 1.2, spec_shape_error)
                MASK_DETECTION_LEVELS[level_name]['shape_error'] = calibrated_value
                logger.info(f"Калиброван shape_error для {level_name}: {calibrated_value:.2f}")
            else:
                logger.warning(f"Недостаточно данных для калибровки shape_error для {level_name}. Используются значения по умолчанию.")

            # Калибровка entropy
            if epoch_entropies[level_name]:
                level_mean_entropy = np.mean(epoch_entropies[level_name])
                spec_entropy = level_config['entropy']
                calibrated_value = max(level_mean_entropy * 1.2, spec_entropy)
                MASK_DETECTION_LEVELS[level_name]['entropy'] = calibrated_value
                logger.info(f"Калиброван entropy для {level_name}: {calibrated_value:.2f}")
            else:
                logger.warning(f"Недостаточно данных для калибровки entropy для {level_name}. Используются значения по умолчанию.")

        # Адаптивная калибровка аномалий на основе Z-score (существующая логика)
        for threshold_name in ["anomaly_z_score_threshold"]:
            if threshold_name in calibrated_thresholds:
                calibrated_thresholds[threshold_name] = max(2.0, min(3.5, calibrated_thresholds[threshold_name]))
        
        logger.info("Автокалибровка порогов завершена успешно")
        return calibrated_thresholds
        
    except Exception as e:
        logger.error(f"Ошибка автокалибровки: {e}")
        return CRITICAL_THRESHOLDS.copy()

def get_age_adjusted_thresholds(current_age: int) -> Dict[str, float]:
    """Получить пороги, адаптированные под возраст"""
    logger.info(f"Получение порогов для возраста: {current_age}")
    
    base_thresholds = CRITICAL_THRESHOLDS.copy()
    
    try:
        # Адаптация порогов под возраст
        if current_age >= AGING_MODEL["bone_stability_threshold"]:
            # После 25 лет кости стабильны
            base_thresholds["max_shape_error"] *= 0.8  # Более строгие требования
            logger.info("Применены строгие пороги для стабильного возраста")
        else:
            # До 25 лет возможны изменения
            base_thresholds["max_shape_error"] *= 1.2  # Более мягкие требования
            logger.info("Применены мягкие пороги для растущего возраста")
        
        # Адаптация под старение
        aging_factor = max(0.8, 1.0 - (current_age - 40) * AGING_MODEL["elasticity_loss_per_year"])
        base_thresholds["min_authenticity_score"] *= aging_factor
        
        logger.info(f"Пороги адаптированы для возраста {current_age}")
        return base_thresholds
        
    except Exception as e:
        logger.error(f"Ошибка адаптации порогов: {e}")
        return base_thresholds

def save_configuration_snapshot() -> str:
    """Сохранить снимок конфигурации для версионирования"""
    logger.info("Сохранение снимка конфигурации")
    
    try:
        timestamp = datetime.now().isoformat()
        version = "2.0"
        
        # Создание хеша конфигурации
        config_data = {
            "authenticity_weights": AUTHENTICITY_WEIGHTS,
            "critical_thresholds": CRITICAL_THRESHOLDS,
            "aging_model": AGING_MODEL
        }
        
        config_str = json.dumps(config_data, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()
        
        snapshot = ConfigSnapshot(
            timestamp=timestamp,
            version=version,
            authenticity_weights=AUTHENTICITY_WEIGHTS,
            critical_thresholds=CRITICAL_THRESHOLDS,
            aging_model=AGING_MODEL,
            hash=config_hash
        )
        
        # Сохранение в файл
        snapshot_file = CONFIG_DIR / f"config_snapshot_{timestamp.replace(':', '-')}.json"
        with open(snapshot_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(snapshot), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Снимок конфигурации сохранен: {snapshot_file}")
        return str(snapshot_file)
        
    except Exception as e:
        logger.error(f"Ошибка сохранения снимка конфигурации: {e}")
        return ""

# ==================== ИНИЦИАЛИЗАЦИЯ ====================
def initialize_config():
    """Инициализация конфигурации при импорте модуля"""
    logger.info("Инициализация CoreConfig")
    
    try:
        # Валидация конфигурации
        if not validate_configuration():
            raise ValueError("Конфигурация не прошла валидацию")
        
        # Сохранение снимка конфигурации
        snapshot_path = save_configuration_snapshot()
        if snapshot_path:
            logger.info(f"Снимок конфигурации сохранен: {snapshot_path}")
        
        logger.info("CoreConfig успешно инициализирован")
        
    except Exception as e:
        logger.error(f"Ошибка инициализации CoreConfig: {e}")
        raise

# Автоматическая инициализация при импорте
if __name__ != "__main__":
    initialize_config()

# ==================== ТЕСТИРОВАНИЕ ====================
if __name__ == "__main__":
    print("=== Тестирование CoreConfig ===")
    
    # Тест валидации
    print(f"Валидация конфигурации: {validate_configuration()}")
    
    # Тест получения метрик
    metrics = get_identity_signature_metrics()
    total_metrics = sum(len(group) for group in metrics.values())
    print(f"Всего метрик идентичности: {total_metrics}")
    
    # Тест весов
    print(f"Сумма весов аутентичности: {sum(AUTHENTICITY_WEIGHTS.values())}")
    
    # Тест уровней масок
    print(f"Количество уровней масок: {len(MASK_DETECTION_LEVELS)}")
    
    # Тест автокалибровки
    test_data = [
        {"authenticity_score": 0.7, "shape_error": 0.25},
        {"authenticity_score": 0.8, "shape_error": 0.15},
        {"authenticity_score": 0.6, "shape_error": 0.35}
    ]
    calibrated = auto_calibrate_thresholds_historical_data(test_data)
    print(f"Автокалибровка выполнена: {len(calibrated)} порогов")
    
    print("=== Тестирование завершено ===")