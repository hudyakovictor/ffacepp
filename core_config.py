# core_config.py
import os
import json
import logging
import hashlib
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
import numpy as np

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Пути к конфигурационным файлам
CONFIG_DIR = Path("./configs")
CACHE_DIR = Path("./cache")
LOGS_DIR = Path("./logs")
MODELS_DIR = Path("./models")
RESULTS_DIR = Path("./results")

# Создание необходимых директорий
for directory in [CONFIG_DIR, CACHE_DIR, LOGS_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# === ОСНОВНЫЕ КОНСТАНТЫ СИСТЕМЫ ===

# Веса для расчета итогового показателя подлинности (0.3 + 0.3 + 0.2 + 0.2 = 1.0)
AUTHENTICITY_WEIGHTS = {
    'geometry': 0.3,
    'embedding': 0.3, 
    'texture': 0.2,
    'temporal': 0.2
}

# Пороги геометрических ошибок для разных эпох масок
GEOMETRY_ERROR_THRESHOLDS = {
    'level_1': {'shape_error': 0.8, 'eye_region_error': 0.6},  # 2000-2005
    'level_2': {'shape_error': 0.6, 'eye_region_error': 0.4},  # 2006-2010
    'level_3': {'shape_error': 0.4, 'eye_region_error': 0.3},  # 2011-2015
    'level_4': {'shape_error': 0.25, 'eye_region_error': 0.2}, # 2016-2020
    'level_5': {'shape_error': 0.15, 'eye_region_error': 0.1}  # 2021+
}

# Пороги энтропии для анализа текстуры
ENTROPY_THRESHOLDS = {
    'level_1': {'min_entropy': 4.0, 'max_entropy': 7.5},
    'level_2': {'min_entropy': 4.5, 'max_entropy': 7.8},
    'level_3': {'min_entropy': 5.0, 'max_entropy': 8.0},
    'level_4': {'min_entropy': 5.5, 'max_entropy': 8.2},
    'level_5': {'min_entropy': 6.0, 'max_entropy': 8.5}
}

# Уровни масок и их характеристики
MASK_LEVELS = {
    'level_1': {
        'years': '2000-2005',
        'technology': 'Простые силиконовые маски',
        'shape_error_threshold': 0.8,
        'entropy_threshold': 4.0,
        'embedding_distance_threshold': 0.6
    },
    'level_2': {
        'years': '2006-2010', 
        'technology': 'Улучшенные силиконовые маски',
        'shape_error_threshold': 0.6,
        'entropy_threshold': 4.5,
        'embedding_distance_threshold': 0.5
    },
    'level_3': {
        'years': '2011-2015',
        'technology': 'Профессиональные маски с текстурой',
        'shape_error_threshold': 0.4,
        'entropy_threshold': 5.0,
        'embedding_distance_threshold': 0.4
    },
    'level_4': {
        'years': '2016-2020',
        'technology': 'Высокотехнологичные маски',
        'shape_error_threshold': 0.25,
        'entropy_threshold': 5.5,
        'embedding_distance_threshold': 0.35
    },
    'level_5': {
        'years': '2021+',
        'technology': 'Современные маски с ИИ',
        'shape_error_threshold': 0.15,
        'entropy_threshold': 6.0,
        'embedding_distance_threshold': 0.3
    }
}

# Конфигурации ракурсов для нормализации
VIEW_CONFIGS = {
    'frontal': {
        'name': 'Frontal',
        'yaw': (-15, 15),
        'pitch': (-10, 10), 
        'roll': (-10, 10),
        'reference_points': [36, 45, 30, 48, 54],  # глаза, нос, рот
        'scale_type': 'IOD'  # межзрачковое расстояние
    },
    'frontal_edge': {
        'name': 'Frontal-Edge',
        'yaw': (-30, -15),
        'pitch': (-15, 15),
        'roll': (-15, 15),
        'reference_points': [36, 39, 30, 48, 54],
        'scale_type': 'nose_eye'
    },
    'semi_profile': {
        'name': 'Semi-Profile', 
        'yaw': (-45, -30),
        'pitch': (-20, 20),
        'roll': (-20, 20),
        'reference_points': [36, 30, 33, 48, 54],
        'scale_type': 'face_height'
    },
    'profile': {
        'name': 'Profile',
        'yaw': (-90, -45),
        'pitch': (-25, 25),
        'roll': (-25, 25),
        'reference_points': [36, 30, 33, 48],
        'scale_type': 'profile_height'
    }
}

# Референсные расстояния для нормализации (в пикселях для изображений 800x800)
REFERENCE_DISTANCES = {
    'IOD': 120.0,           # межзрачковое расстояние
    'nose_eye': 85.0,       # расстояние нос-глаз
    'face_height': 280.0,   # высота лица
    'profile_height': 260.0 # высота профиля
}

# Параметры DBSCAN для кластеризации эмбеддингов
DBSCAN_EPS = 0.35
DBSCAN_MIN_SAMPLES = 3
EMBEDDING_DISTANCE_THRESHOLD = 0.35

# Параметры модели старения
ELASTICITY_LOSS_PER_YEAR = 0.015  # 1.5% в год после 40 лет
TISSUE_SAGGING_PER_YEAR = 1.2     # 1.2 мм в год после 40 лет

# Пороги для анализа формы глаз
SHAPE_ERROR_EYE_REGION = 0.3

# Пути к моделям
MODEL_PATHS = {
    '3ddfa_v2': './models/phase1_wpdc_vdc.pth.tar',
    'insightface': './models/w600k_r50.onnx',
    'face_detection': './models/Resnet50_Final.pth'
}

# Параметры качества изображения
IMAGE_QUALITY_THRESHOLDS = {
    'min_resolution': (400, 400),
    'max_resolution': (2000, 2000),
    'min_brightness': 50,
    'max_brightness': 200,
    'min_contrast': 0.3,
    'blur_threshold': 100.0,
    'noise_threshold': 0.1
}

# Параметры текстурного анализа
TEXTURE_ANALYSIS_PARAMS = {
    'lbp_radius': 3,
    'lbp_points': 24,
    'gabor_frequencies': [0.1, 0.3, 0.5, 0.7],
    'gabor_orientations': 11,
    'glcm_distances': [1, 2, 3],
    'glcm_angles': [0, 45, 90, 135]
}

# Параметры временного анализа
TEMPORAL_ANALYSIS_PARAMS = {
    'anomaly_threshold': 2.5,  # Z-score
    'gap_threshold_days': 180,
    'min_samples_for_trend': 5,
    'aging_start_year': 40
}

# Лимиты производительности
PERFORMANCE_LIMITS = {
    'max_concurrent_files': 16,
    'max_cache_size_mb': 1024,
    'max_memory_usage_mb': 4096,
    'batch_size': 8,
    'max_points_3d_viz': 50000
}

# Коды ошибок
ERROR_CODES = {
    'CONFIG_INVALID': 1001,
    'MODEL_LOAD_FAILED': 1002,
    'IMAGE_PROCESSING_FAILED': 1003,
    'FACE_NOT_DETECTED': 1004,
    'LANDMARKS_EXTRACTION_FAILED': 1005,
    'EMBEDDING_EXTRACTION_FAILED': 1006,
    'TEXTURE_ANALYSIS_FAILED': 1007,
    'TEMPORAL_ANALYSIS_FAILED': 1008,
    'REPORT_GENERATION_FAILED': 1009,
    'CACHE_ERROR': 1010
}

# Цветовые схемы для визуализации
COLOR_SCHEMES = {
    'authenticity': {
        'authentic': '#2E8B57',      # зеленый
        'suspicious': '#FFD700',     # желтый
        'fake': '#DC143C'            # красный
    },
    'clusters': [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ],
    'timeline': {
        'normal': '#4CAF50',
        'anomaly': '#F44336',
        'gap': '#FF9800'
    }
}

# === КЛАССЫ КОНФИГУРАЦИИ ===

@dataclass
class ConfigSnapshot:
    """Снапшот конфигурации для обеспечения консистентности"""
    timestamp: str
    config_hash: str
    model_versions: Dict[str, str]
    authenticity_weights: Dict[str, float]
    geometry_thresholds: Dict[str, Dict[str, float]]
    entropy_thresholds: Dict[str, Dict[str, float]]
    mask_levels: Dict[str, Dict[str, Any]]
    view_configs: Dict[str, Dict[str, Any]]

class CoreConfig:
    """Основной класс конфигурации системы"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CoreConfig, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.config_snapshot = None
            self.autocalib_thresholds = {}
            self.runtime_params = {}
            self.degraded_mode = False
            self._load_configuration()
            self._initialized = True
    
    def _load_configuration(self):
        """Загрузка и валидация конфигурации"""
        try:
            # Создание снапшота текущей конфигурации
            config_data = {
                'authenticity_weights': AUTHENTICITY_WEIGHTS,
                'geometry_thresholds': GEOMETRY_ERROR_THRESHOLDS,
                'entropy_thresholds': ENTROPY_THRESHOLDS,
                'mask_levels': MASK_LEVELS,
                'view_configs': VIEW_CONFIGS,
                'reference_distances': REFERENCE_DISTANCES,
                'model_paths': MODEL_PATHS,
                'performance_limits': PERFORMANCE_LIMITS
            }
            
            # Вычисление хеша конфигурации
            config_str = json.dumps(config_data, sort_keys=True)
            config_hash = hashlib.sha256(config_str.encode()).hexdigest()
            
            # Получение версий моделей
            model_versions = self._get_model_versions()
            
            # Создание снапшота
            self.config_snapshot = ConfigSnapshot(
                timestamp=datetime.datetime.now().isoformat(),
                config_hash=config_hash,
                model_versions=model_versions,
                authenticity_weights=AUTHENTICITY_WEIGHTS.copy(),
                geometry_thresholds=GEOMETRY_ERROR_THRESHOLDS.copy(),
                entropy_thresholds=ENTROPY_THRESHOLDS.copy(),
                mask_levels=MASK_LEVELS.copy(),
                view_configs=VIEW_CONFIGS.copy()
            )
            
            # Сохранение снапшота
            self._save_snapshot()
            
            # Загрузка автокалиброванных порогов
            self._load_autocalib_thresholds()
            
            logger.info(f"Конфигурация загружена успешно. Hash: {config_hash[:8]}")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации: {e}")
            self.degraded_mode = True
            raise
    
    def _get_model_versions(self) -> Dict[str, str]:
        """Получение версий моделей"""
        versions = {}
        for model_name, model_path in MODEL_PATHS.items():
            if os.path.exists(model_path):
                # Вычисление хеша файла модели
                with open(model_path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                versions[model_name] = file_hash[:16]
            else:
                versions[model_name] = "NOT_FOUND"
                logger.warning(f"Модель {model_name} не найдена: {model_path}")
        return versions
    
    def _save_snapshot(self):
        """Сохранение снапшота конфигурации"""
        try:
            snapshot_path = CONFIG_DIR / "config_snapshot.json"
            with open(snapshot_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.config_snapshot), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Ошибка сохранения снапшота: {e}")
    
    def _load_autocalib_thresholds(self):
        """Загрузка автокалиброванных порогов"""
        try:
            autocalib_path = CONFIG_DIR / "autocalib_thresholds.json"
            if autocalib_path.exists():
                with open(autocalib_path, 'r', encoding='utf-8') as f:
                    self.autocalib_thresholds = json.load(f)
                logger.info("Автокалиброванные пороги загружены")
        except Exception as e:
            logger.warning(f"Не удалось загрузить автокалиброванные пороги: {e}")
            self.autocalib_thresholds = {}
    
    def validate_config_integrity(self) -> bool:
        """Валидация целостности конфигурации"""
        try:
            # Проверка весов аутентичности
            weights_sum = sum(AUTHENTICITY_WEIGHTS.values())
            if abs(weights_sum - 1.0) > 1e-6:
                raise ValueError(f"Сумма весов аутентичности должна быть 1.0, получено: {weights_sum}")
            
            # Проверка наличия всех уровней масок
            required_levels = ['level_1', 'level_2', 'level_3', 'level_4', 'level_5']
            for level in required_levels:
                if level not in MASK_LEVELS:
                    raise ValueError(f"Отсутствует уровень маски: {level}")
            
            # Проверка конфигураций ракурсов
            for view_name, view_config in VIEW_CONFIGS.items():
                required_keys = ['name', 'yaw', 'pitch', 'roll', 'reference_points', 'scale_type']
                for key in required_keys:
                    if key not in view_config:
                        raise ValueError(f"Отсутствует ключ {key} в конфигурации ракурса {view_name}")
            
            # Проверка путей к моделям
            missing_models = []
            for model_name, model_path in MODEL_PATHS.items():
                if not os.path.exists(model_path):
                    missing_models.append(f"{model_name}: {model_path}")
            
            if missing_models:
                logger.warning(f"Отсутствующие модели: {missing_models}")
                self.degraded_mode = True
            
            # Проверка положительности порогов
            for level_data in GEOMETRY_ERROR_THRESHOLDS.values():
                for threshold_name, threshold_value in level_data.items():
                    if threshold_value <= 0:
                        raise ValueError(f"Порог {threshold_name} должен быть положительным")
            
            for level_data in ENTROPY_THRESHOLDS.values():
                for threshold_name, threshold_value in level_data.items():
                    if threshold_value <= 0:
                        raise ValueError(f"Порог энтропии {threshold_name} должен быть положительным")
            
            logger.info("Валидация конфигурации прошла успешно")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка валидации конфигурации: {e}")
            self.degraded_mode = True
            return False
    
    def get_authenticity_weights(self) -> Dict[str, float]:
        """Получение весов для расчета аутентичности"""
        return AUTHENTICITY_WEIGHTS.copy()
    
    def get_geometry_thresholds(self, level: str = None) -> Dict[str, Any]:
        """Получение порогов геометрических ошибок"""
        if level:
            return GEOMETRY_ERROR_THRESHOLDS.get(level, {}).copy()
        return GEOMETRY_ERROR_THRESHOLDS.copy()
    
    def get_entropy_thresholds(self, level: str = None) -> Dict[str, Any]:
        """Получение порогов энтропии"""
        if level:
            return ENTROPY_THRESHOLDS.get(level, {}).copy()
        return ENTROPY_THRESHOLDS.copy()
    
    def get_mask_levels(self) -> Dict[str, Dict[str, Any]]:
        """Получение уровней масок"""
        return MASK_LEVELS.copy()
    
    def get_view_configs(self) -> Dict[str, Dict[str, Any]]:
        """Получение конфигураций ракурсов"""
        return VIEW_CONFIGS.copy()
    
    def get_reference_distances(self) -> Dict[str, float]:
        """Получение референсных расстояний"""
        return REFERENCE_DISTANCES.copy()
    
    def get_model_paths(self) -> Dict[str, str]:
        """Получение путей к моделям"""
        return MODEL_PATHS.copy()
    
    def get_model_path(self, model_name: str) -> str:
        """Получение пути к конкретной модели"""
        return MODEL_PATHS.get(model_name, "")
    
    def get_dbscan_params(self) -> Dict[str, Union[float, int]]:
        """Получение параметров DBSCAN"""
        return {
            'eps': DBSCAN_EPS,
            'min_samples': DBSCAN_MIN_SAMPLES,
            'distance_threshold': EMBEDDING_DISTANCE_THRESHOLD
        }
    
    def get_aging_params(self) -> Dict[str, float]:
        """Получение параметров модели старения"""
        return {
            'elasticity_loss_per_year': ELASTICITY_LOSS_PER_YEAR,
            'tissue_sagging_per_year': TISSUE_SAGGING_PER_YEAR
        }
    
    def get_image_quality_thresholds(self) -> Dict[str, Any]:
        """Получение порогов качества изображения"""
        return IMAGE_QUALITY_THRESHOLDS.copy()
    
    def get_texture_analysis_params(self) -> Dict[str, Any]:
        """Получение параметров анализа текстуры"""
        return TEXTURE_ANALYSIS_PARAMS.copy()
    
    def get_temporal_analysis_params(self) -> Dict[str, Any]:
        """Получение параметров временного анализа"""
        return TEMPORAL_ANALYSIS_PARAMS.copy()
    
    def get_performance_limits(self) -> Dict[str, int]:
        """Получение лимитов производительности"""
        return PERFORMANCE_LIMITS.copy()
    
    def get_color_schemes(self) -> Dict[str, Any]:
        """Получение цветовых схем"""
        return COLOR_SCHEMES.copy()
    
    def get_error_codes(self) -> Dict[str, int]:
        """Получение кодов ошибок"""
        return ERROR_CODES.copy()
    
    def get_config_snapshot(self) -> ConfigSnapshot:
        """Получение снапшота конфигурации"""
        return self.config_snapshot
    
    def get_snapshot_hash(self) -> str:
        """Получение хеша снапшота"""
        return self.config_snapshot.config_hash if self.config_snapshot else ""
    
    def is_degraded_mode(self) -> bool:
        """Проверка режима деградации"""
        return self.degraded_mode
    
    def save_autocalib_thresholds(self, thresholds: Dict[str, Any]):
        """Сохранение автокалиброванных порогов"""
        try:
            self.autocalib_thresholds.update(thresholds)
            autocalib_path = CONFIG_DIR / "autocalib_thresholds.json"
            with open(autocalib_path, 'w', encoding='utf-8') as f:
                json.dump(self.autocalib_thresholds, f, indent=2, ensure_ascii=False)
            logger.info("Автокалиброванные пороги сохранены")
        except Exception as e:
            logger.error(f"Ошибка сохранения автокалиброванных порогов: {e}")
    
    def get_autocalib_thresholds(self) -> Dict[str, Any]:
        """Получение автокалиброванных порогов"""
        return self.autocalib_thresholds.copy()
    
    def classify_mask_technology_level(self, year: int) -> str:
        """Классификация уровня технологии маски по году"""
        if year <= 2005:
            return 'level_1'
        elif year <= 2010:
            return 'level_2'
        elif year <= 2015:
            return 'level_3'
        elif year <= 2020:
            return 'level_4'
        else:
            return 'level_5'
    
    def get_mask_thresholds_for_year(self, year: int) -> Dict[str, float]:
        """Получение порогов маски для конкретного года"""
        level = self.classify_mask_technology_level(year)
        return MASK_LEVELS.get(level, {})
    
    def determine_pose_category(self, yaw: float, pitch: float, roll: float) -> str:
        """Определение категории позы по углам"""
        for view_name, view_config in VIEW_CONFIGS.items():
            yaw_range = view_config['yaw']
            pitch_range = view_config['pitch']
            roll_range = view_config['roll']
            
            if (yaw_range[0] <= yaw <= yaw_range[1] and
                pitch_range[0] <= pitch <= pitch_range[1] and
                roll_range[0] <= roll <= roll_range[1]):
                return view_name
        
        return 'frontal'  # по умолчанию
    
    def get_reference_points_for_pose(self, pose_category: str) -> List[int]:
        """Получение референсных точек для позы"""
        return VIEW_CONFIGS.get(pose_category, {}).get('reference_points', [])
    
    def get_scale_type_for_pose(self, pose_category: str) -> str:
        """Получение типа масштабирования для позы"""
        return VIEW_CONFIGS.get(pose_category, {}).get('scale_type', 'IOD')
    
    def reload_configuration(self):
        """Перезагрузка конфигурации"""
        logger.info("Перезагрузка конфигурации...")
        self._load_configuration()
    
    def export_configuration(self, filepath: str):
        """Экспорт конфигурации в файл"""
        try:
            config_data = {
                'snapshot': asdict(self.config_snapshot) if self.config_snapshot else None,
                'authenticity_weights': AUTHENTICITY_WEIGHTS,
                'geometry_thresholds': GEOMETRY_ERROR_THRESHOLDS,
                'entropy_thresholds': ENTROPY_THRESHOLDS,
                'mask_levels': MASK_LEVELS,
                'view_configs': VIEW_CONFIGS,
                'reference_distances': REFERENCE_DISTANCES,
                'model_paths': MODEL_PATHS,
                'autocalib_thresholds': self.autocalib_thresholds,
                'degraded_mode': self.degraded_mode
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Конфигурация экспортирована в {filepath}")
            
        except Exception as e:
            logger.error(f"Ошибка экспорта конфигурации: {e}")

# === ГЛОБАЛЬНЫЕ ФУНКЦИИ ===

def get_config() -> CoreConfig:
    """Получение экземпляра конфигурации (singleton)"""
    return CoreConfig()

def validate_config_integrity() -> bool:
    """Валидация целостности конфигурации"""
    config = get_config()
    return config.validate_config_integrity()

def get_authenticity_weights() -> Dict[str, float]:
    """Получение весов аутентичности"""
    return get_config().get_authenticity_weights()

def get_mask_levels() -> Dict[str, Dict[str, Any]]:
    """Получение уровней масок"""
    return get_config().get_mask_levels()

def get_view_configs() -> Dict[str, Dict[str, Any]]:
    """Получение конфигураций ракурсов"""
    return get_config().get_view_configs()

def classify_mask_technology_level(year: int) -> str:
    """Классификация уровня технологии маски"""
    return get_config().classify_mask_technology_level(year)

def determine_pose_category(yaw: float, pitch: float, roll: float) -> str:
    """Определение категории позы"""
    return get_config().determine_pose_category(yaw, pitch, roll)

# === САМОТЕСТИРОВАНИЕ ===

def self_test():
    """Самотестирование модуля конфигурации"""
    try:
        logger.info("Запуск самотестирования core_config...")
        
        # Создание экземпляра конфигурации
        config = get_config()
        
        # Валидация конфигурации
        assert config.validate_config_integrity(), "Валидация конфигурации не прошла"
        
        # Проверка весов аутентичности
        weights = config.get_authenticity_weights()
        assert abs(sum(weights.values()) - 1.0) < 1e-6, "Сумма весов не равна 1.0"
        
        # Проверка уровней масок
        mask_levels = config.get_mask_levels()
        assert len(mask_levels) == 5, "Неверное количество уровней масок"
        
        # Проверка конфигураций ракурсов
        view_configs = config.get_view_configs()
        assert len(view_configs) == 4, "Неверное количество конфигураций ракурсов"
        
        # Проверка классификации уровня маски
        assert classify_mask_technology_level(2003) == 'level_1'
        assert classify_mask_technology_level(2018) == 'level_4'
        assert classify_mask_technology_level(2023) == 'level_5'
        
        # Проверка определения категории позы
        assert determine_pose_category(0, 0, 0) == 'frontal'
        assert determine_pose_category(-20, 0, 0) == 'frontal_edge'
        
        # Проверка снапшота
        snapshot = config.get_config_snapshot()
        assert snapshot is not None, "Снапшот конфигурации не создан"
        assert len(snapshot.config_hash) == 64, "Неверная длина хеша конфигурации"
        
        logger.info("Самотестирование core_config завершено успешно")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка самотестирования: {e}")
        return False

# === ИНИЦИАЛИЗАЦИЯ ===

if __name__ == "__main__":
    # Запуск самотестирования при прямом вызове модуля
    success = self_test()
    if success:
        print("✅ Модуль core_config работает корректно")
        
        # Вывод основной информации о конфигурации
        config = get_config()
        print(f"📊 Хеш конфигурации: {config.get_snapshot_hash()[:8]}")
        print(f"🎭 Уровней масок: {len(config.get_mask_levels())}")
        print(f"📐 Конфигураций ракурсов: {len(config.get_view_configs())}")
        print(f"⚖️ Веса аутентичности: {config.get_authenticity_weights()}")
        print(f"🔧 Режим деградации: {'Да' if config.is_degraded_mode() else 'Нет'}")
    else:
        print("❌ Обнаружены ошибки в модуле core_config")
        exit(1)
