# face_3d_analyzer.py
import os
import sys
import json
import logging
import hashlib
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
import pickle
import time
import psutil
import msgpack
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import OrderedDict, defaultdict

from core_config import get_config

# Настройка логирования
logger = logging.getLogger(__name__)

# === КОНСТАНТЫ И КОНФИГУРАЦИЯ ===

# Размеры изображений для 3DDFA_V2
TARGET_SIZE = (120, 120)
ORIGINAL_SIZE = (800, 800)
DENSE_MESH_SIZE = 38365  # Количество точек плотной поверхности

# Параметры субпиксельной точности
SUBPIXEL_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
SUBPIXEL_WIN_SIZE = (11, 11)
SUBPIXEL_ZERO_ZONE = (-1, -1)

# Индексы ключевых точек лица (68-точечная модель)
FACIAL_LANDMARKS_68_IDXS = {
    "mouth": (48, 68),
    "right_eyebrow": (17, 22),
    "left_eyebrow": (22, 27),
    "right_eye": (36, 42),
    "left_eye": (42, 48),
    "nose": (27, 35),
    "jaw": (0, 17)
}

# Расширенные индексы для детального анализа
DETAILED_LANDMARKS_MAPPING = {
    "outer_jaw": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    "right_eyebrow": [17, 18, 19, 20, 21],
    "left_eyebrow": [22, 23, 24, 25, 26],
    "nose_bridge": [27, 28, 29, 30],
    "nose_tip": [31, 32, 33, 34, 35],
    "right_eye": [36, 37, 38, 39, 40, 41],
    "left_eye": [42, 43, 44, 45, 46, 47],
    "outer_mouth": [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
    "inner_mouth": [60, 61, 62, 63, 64, 65, 66, 67]
}

# Индексы для расчета метрик идентичности
EYE_CORNERS = [36, 39, 42, 45]  # Внешние углы глаз
NOSE_TIP = 30
MOUTH_CORNERS = [48, 54]
CHIN_POINT = 8
FOREHEAD_POINTS = [19, 24]  # Брови для оценки лба

# Пороги для валидации
CONFIDENCE_THRESHOLD = 0.8
POSE_ANGLE_LIMITS = {
    'yaw': (-90, 90),
    'pitch': (-60, 60),
    'roll': (-45, 45)
}

# Параметры для анализа качества
QUALITY_THRESHOLDS = {
    'min_face_size': 80,
    'max_face_size': 600,
    'min_landmarks_confidence': 0.7,
    'max_pose_deviation': 45.0,
    'min_eye_distance': 20,
    'max_eye_distance': 200
}

# Параметры для плотной поверхности
DENSE_SURFACE_PARAMS = {
    'enable_dense_extraction': True,
    'dense_points_limit': 38365,
    'surface_smoothing': True,
    'texture_mapping': True
}

# === СТРУКТУРЫ ДАННЫХ ===

@dataclass
class LandmarksPackage:
    """Пакет данных с результатами 3D-анализа лица"""
    image_id: str
    filepath: str
    landmarks_68: np.ndarray  # 68x3 координаты
    landmarks_confidence: float
    pose_angles: Dict[str, float]  # yaw, pitch, roll
    pose_category: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    
    # Нормализованные данные
    normalized_landmarks: Optional[np.ndarray] = None
    scale_factor: float = 1.0
    reference_distance: float = 0.0
    
    # Дополнительные метрики
    shape_error: float = 0.0
    eye_region_error: float = 0.0
    dense_vertices: Optional[np.ndarray] = None  # 38365x3 для полной поверхности
    dense_triangles: Optional[np.ndarray] = None  # Треугольники меша
    
    # Детальные анализы
    landmarks_visibility: Optional[np.ndarray] = None  # Видимость каждой точки
    landmarks_quality_scores: Optional[np.ndarray] = None  # Качество каждой точки
    pose_stability_score: float = 0.0
    facial_symmetry_score: float = 0.0
    
    # Метаданные обработки
    processing_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    extraction_method: str = "3DDFA_V2"
    device_used: str = "cpu"
    model_version: str = ""
    
    # Флаги качества
    quality_flags: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Кэш-информация
    cache_key: Optional[str] = None
    cache_timestamp: Optional[datetime.datetime] = None

@dataclass
class Face3DModelConfig:
    """Конфигурация 3D-модели лица"""
    model_path: str
    bfm_path: str
    device: str = "cpu"
    use_mps: bool = False
    batch_size: int = 1
    enable_dense_mesh: bool = True
    enable_texture_extraction: bool = True
    enable_pose_estimation: bool = True
    enable_expression_analysis: bool = True

@dataclass
class PoseEstimationResult:
    """Результат оценки позы лица"""
    yaw: float
    pitch: float
    roll: float
    rotation_matrix: np.ndarray
    translation_vector: np.ndarray
    pose_confidence: float
    pose_category: str
    stability_score: float

@dataclass
class FaceQualityAssessment:
    """Оценка качества лица для анализа"""
    overall_quality: float
    face_size_score: float
    pose_score: float
    lighting_score: float
    blur_score: float
    occlusion_score: float
    symmetry_score: float
    quality_flags: List[str]
    recommendations: List[str]

# === ОСНОВНОЙ КЛАСС 3D-АНАЛИЗАТОРА ===

class Face3DAnalyzer:
    """Анализатор для извлечения 3D-ландмарков и определения позы лица"""
    
    def __init__(self):
        self.config = get_config()
        self.cache_dir = Path("./cache/face_3d")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Конфигурация модели
        self.model_config = Face3DModelConfig(
            model_path=self.config.get_model_path("3ddfa_v2"),
            bfm_path="./models/BFM",
            device="mps" if torch.backends.mps.is_available() else "cpu",
            use_mps=torch.backends.mps.is_available()
        )
        
        # Модели и компоненты
        self.face_detector = None
        self.ddfa_model = None
        self.bfm_model = None
        self.triangles = None
        self.texture_extractor = None
        
        # Кэш результатов
        self.landmarks_cache: Dict[str, LandmarksPackage] = {}
        self.cache_size_mb = 0.0
        self.max_cache_size_mb = 1024  # 1GB лимит
        
        # Статистика
        self.processing_stats = {
            'total_processed': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'cache_hits': 0,
            'subpixel_corrections': 0,
            'dense_mesh_extractions': 0,
            'pose_estimations': 0
        }
        
        # Пулы для параллельной обработки
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.processing_lock = threading.Lock()
        
        # Инициализация компонентов
        self._initialize_components()
        
        logger.info(f"Face3DAnalyzer инициализирован (устройство: {self.model_config.device})")

    def _initialize_components(self):
        """Инициализация всех компонентов 3DDFA_V2"""
        try:
            # Инициализация детектора лиц
            self._initialize_face_detector()
            
            # Инициализация 3DDFA модели
            self._initialize_3ddfa_model()
            
            # Инициализация BFM модели
            self._initialize_bfm_model()
            
            # Инициализация экстрактора текстур
            self._initialize_texture_extractor()
            
            logger.info("Все компоненты 3DDFA_V2 успешно инициализированы")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации компонентов: {e}")
            raise

    def _initialize_face_detector(self):
        """Инициализация детектора лиц FaceBoxes"""
        try:
            # Попытка импорта FaceBoxes из 3DDFA_V2
            sys.path.append('./models/3DDFA_V2')
            try:
                from FaceBoxes import FaceBoxes
                self.face_detector = FaceBoxes()
                logger.debug("FaceBoxes детектор инициализирован")
            except ImportError:
                logger.warning("FaceBoxes недоступен, используется fallback")
                self._initialize_opencv_detector()
            
        except Exception as e:
            logger.error(f"Ошибка инициализации FaceBoxes: {e}")
            self._initialize_opencv_detector()

    def _initialize_opencv_detector(self):
        """Fallback инициализация OpenCV детектора"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                self.face_detector = cv2.CascadeClassifier(cascade_path)
                logger.warning("Используется OpenCV Haar Cascade как fallback детектор")
            else:
                # Альтернативный путь для некоторых систем
                alt_cascade_path = '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
                if os.path.exists(alt_cascade_path):
                    self.face_detector = cv2.CascadeClassifier(alt_cascade_path)
                else:
                    raise FileNotFoundError("Не найден файл каскада Haar")
            
        except Exception as e:
            logger.error(f"Критическая ошибка инициализации детектора: {e}")
            raise

    def _initialize_3ddfa_model(self):
        """Инициализация основной модели 3DDFA_V2"""
        try:
            # Импорт модели из 3DDFA_V2
            sys.path.append('./models/3DDFA_V2')
            from TDDFA import TDDFA
            
            # Конфигурация модели
            cfg = {
                'arch': 'mobilenet_v1',
                'checkpoint_fp': self.model_config.model_path,
                'bfm_fp': os.path.join(self.model_config.bfm_path, 'BFM_model_front.mat'),
                'size': 120,
                'device': self.model_config.device
            }
            
            # Проверка существования файлов
            if not os.path.exists(cfg['checkpoint_fp']):
                raise FileNotFoundError(f"Модель 3DDFA не найдена: {cfg['checkpoint_fp']}")
            
            if not os.path.exists(cfg['bfm_fp']):
                logger.warning(f"BFM файл не найден: {cfg['bfm_fp']}")
            
            self.ddfa_model = TDDFA(cfg)
            self.model_config.model_version = "3DDFA_V2_MobileNet"
            logger.debug("3DDFA модель инициализирована")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации 3DDFA модели: {e}")
            raise

    def _initialize_bfm_model(self):
        """Инициализация Basel Face Model"""
        try:
            # Загрузка BFM параметров
            bfm_path = os.path.join(self.model_config.bfm_path, 'BFM_model_front.mat')
            if os.path.exists(bfm_path):
                from scipy.io import loadmat
                bfm_data = loadmat(bfm_path)
                
                self.bfm_model = {
                    'shapeMU': bfm_data.get('shapeMU'),
                    'shapePC': bfm_data.get('shapePC'),
                    'shapeEV': bfm_data.get('shapeEV'),
                    'texMU': bfm_data.get('texMU'),
                    'texPC': bfm_data.get('texPC'),
                    'texEV': bfm_data.get('texEV'),
                    'expPC': bfm_data.get('expPC'),  # Коэффициенты выражений
                    'expEV': bfm_data.get('expEV')
                }
                
                # Загрузка треугольников для меша
                tri_path = os.path.join(self.model_config.bfm_path, 'tri.mat')
                if os.path.exists(tri_path):
                    tri_data = loadmat(tri_path)
                    self.triangles = tri_data['tri'] - 1  # Matlab to Python indexing
                
                logger.debug("BFM модель загружена")
            else:
                logger.warning(f"BFM файл не найден: {bfm_path}")
                self.bfm_model = None
                
        except Exception as e:
            logger.error(f"Ошибка загрузки BFM модели: {e}")
            self.bfm_model = None

    def _initialize_texture_extractor(self):
        """Инициализация экстрактора текстур"""
        try:
            if self.model_config.enable_texture_extraction:
                # Простой экстрактор текстур на основе UV-маппинга
                self.texture_extractor = {
                    'uv_coords': None,
                    'texture_size': (256, 256),
                    'interpolation': cv2.INTER_LINEAR
                }
                logger.debug("Экстрактор текстур инициализирован")
            
        except Exception as e:
            logger.warning(f"Ошибка инициализации экстрактора текстур: {e}")
            self.texture_extractor = None

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Детекция лиц на изображении
        
        Args:
            image: Входное изображение
            
        Returns:
            Список bounding boxes в формате (x, y, w, h)
        """
        try:
            if hasattr(self.face_detector, '__call__'):
                # FaceBoxes детектор
                bboxes = self.face_detector(image)
                if len(bboxes) > 0:
                    # Конвертация в формат (x, y, w, h)
                    boxes = []
                    for bbox in bboxes:
                        if len(bbox) >= 4:
                            x1, y1, x2, y2 = bbox[:4]
                            w, h = x2 - x1, y2 - y1
                            # Валидация размеров
                            if (w >= QUALITY_THRESHOLDS['min_face_size'] and 
                                w <= QUALITY_THRESHOLDS['max_face_size'] and
                                h >= QUALITY_THRESHOLDS['min_face_size'] and 
                                h <= QUALITY_THRESHOLDS['max_face_size']):
                                boxes.append((int(x1), int(y1), int(w), int(h)))
                    return boxes
            else:
                # OpenCV Haar Cascade
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
                faces = self.face_detector.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(QUALITY_THRESHOLDS['min_face_size'], QUALITY_THRESHOLDS['min_face_size']),
                    maxSize=(QUALITY_THRESHOLDS['max_face_size'], QUALITY_THRESHOLDS['max_face_size'])
                )
                return [tuple(face) for face in faces]
            
            return []
            
        except Exception as e:
            logger.error(f"Ошибка детекции лиц: {e}")
            return []

    def assess_face_quality(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                          landmarks: Optional[np.ndarray] = None) -> FaceQualityAssessment:
        """
        Комплексная оценка качества лица
        
        Args:
            image: Входное изображение
            bbox: Bounding box лица
            landmarks: Ландмарки лица (опционально)
            
        Returns:
            Оценка качества лица
        """
        try:
            x, y, w, h = bbox
            face_region = image[y:y+h, x:x+w]
            
            scores = {}
            flags = []
            recommendations = []
            
            # 1. Оценка размера лица
            face_area = w * h
            min_area = QUALITY_THRESHOLDS['min_face_size'] ** 2
            max_area = QUALITY_THRESHOLDS['max_face_size'] ** 2
            
            if face_area < min_area:
                scores['face_size_score'] = face_area / min_area
                flags.append("face_too_small")
                recommendations.append("Увеличить размер лица в кадре")
            elif face_area > max_area:
                scores['face_size_score'] = max_area / face_area
                flags.append("face_too_large")
                recommendations.append("Уменьшить размер лица в кадре")
            else:
                scores['face_size_score'] = 1.0
            
            # 2. Оценка освещения
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if len(face_region.shape) == 3 else face_region
            mean_brightness = np.mean(gray_face)
            brightness_std = np.std(gray_face)
            
            if mean_brightness < 50:
                scores['lighting_score'] = mean_brightness / 50
                flags.append("too_dark")
                recommendations.append("Улучшить освещение")
            elif mean_brightness > 200:
                scores['lighting_score'] = 1.0 - (mean_brightness - 200) / 55
                flags.append("too_bright")
                recommendations.append("Уменьшить яркость освещения")
            else:
                scores['lighting_score'] = 1.0
            
            # 3. Оценка размытия
            laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            if laplacian_var < 100:
                scores['blur_score'] = laplacian_var / 100
                flags.append("blurry")
                recommendations.append("Уменьшить размытие")
            else:
                scores['blur_score'] = min(1.0, laplacian_var / 300)
            
            # 4. Оценка позы (если есть ландмарки)
            if landmarks is not None:
                pose_angles = self._estimate_pose_from_landmarks(landmarks)
                max_angle = max(abs(pose_angles['yaw']), abs(pose_angles['pitch']), abs(pose_angles['roll']))
                
                if max_angle > QUALITY_THRESHOLDS['max_pose_deviation']:
                    scores['pose_score'] = 1.0 - (max_angle - QUALITY_THRESHOLDS['max_pose_deviation']) / 45
                    flags.append("extreme_pose")
                    recommendations.append("Выровнять позу лица")
                else:
                    scores['pose_score'] = 1.0
            else:
                scores['pose_score'] = 0.8  # Нейтральная оценка без ландмарков
            
            # 5. Оценка окклюзии (простая проверка)
            # Проверяем однородность яркости - большие перепады могут указывать на окклюзию
            brightness_gradient = np.gradient(gray_face.astype(float))
            gradient_magnitude = np.sqrt(brightness_gradient[0]**2 + brightness_gradient[1]**2)
            occlusion_indicator = np.mean(gradient_magnitude)
            
            if occlusion_indicator > 50:
                scores['occlusion_score'] = max(0.0, 1.0 - (occlusion_indicator - 50) / 100)
                flags.append("possible_occlusion")
                recommendations.append("Проверить наличие окклюзии")
            else:
                scores['occlusion_score'] = 1.0
            
            # 6. Оценка симметрии (если есть ландмарки)
            if landmarks is not None:
                symmetry_score = self._calculate_facial_symmetry(landmarks)
                scores['symmetry_score'] = symmetry_score
                if symmetry_score < 0.7:
                    flags.append("asymmetric_face")
                    recommendations.append("Проверить симметрию лица")
            else:
                scores['symmetry_score'] = 0.8
            
            # Общая оценка качества
            overall_quality = np.mean(list(scores.values()))
            
            return FaceQualityAssessment(
                overall_quality=overall_quality,
                face_size_score=scores.get('face_size_score', 0.0),
                pose_score=scores.get('pose_score', 0.0),
                lighting_score=scores.get('lighting_score', 0.0),
                blur_score=scores.get('blur_score', 0.0),
                occlusion_score=scores.get('occlusion_score', 0.0),
                symmetry_score=scores.get('symmetry_score', 0.0),
                quality_flags=flags,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Ошибка оценки качества лица: {e}")
            return FaceQualityAssessment(
                overall_quality=0.0,
                face_size_score=0.0,
                pose_score=0.0,
                lighting_score=0.0,
                blur_score=0.0,
                occlusion_score=0.0,
                symmetry_score=0.0,
                quality_flags=["assessment_error"],
                recommendations=["Повторить анализ"]
            )

    def _calculate_facial_symmetry(self, landmarks: np.ndarray) -> float:
        """Расчет симметрии лица по ландмаркам"""
        try:
            # Используем точки левой и правой стороны лица
            left_points = landmarks[0:9, :2]  # Левая сторона челюсти
            right_points = landmarks[8:17, :2]  # Правая сторона челюсти
            
            # Центр лица (нос)
            nose_center = landmarks[30, :2]
            
            # Расстояния от центра до левой и правой стороны
            left_distances = np.linalg.norm(left_points - nose_center, axis=1)
            right_distances = np.linalg.norm(right_points[::-1] - nose_center, axis=1)
            
            # Симметрия как обратная величина средней разности расстояний
            mean_diff = np.mean(np.abs(left_distances - right_distances))
            max_distance = np.max(np.concatenate([left_distances, right_distances]))
            
            if max_distance > 0:
                symmetry_score = 1.0 - (mean_diff / max_distance)
                return max(0.0, min(1.0, symmetry_score))
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Ошибка расчета симметрии: {e}")
            return 0.0

    def _estimate_pose_from_landmarks(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Быстрая оценка позы по ландмаркам"""
        try:
            # Простая оценка на основе геометрии ключевых точек
            left_eye_center = np.mean(landmarks[42:48, :2], axis=0)
            right_eye_center = np.mean(landmarks[36:42, :2], axis=0)
            nose_tip = landmarks[30, :2]
            
            # Yaw (поворот влево-вправо)
            eye_line = right_eye_center - left_eye_center
            yaw = np.degrees(np.arctan2(eye_line[1], eye_line[0]))
            
            # Pitch (наклон вверх-вниз) - примерная оценка
            eye_center = (left_eye_center + right_eye_center) / 2
            nose_to_eye = nose_tip - eye_center
            pitch = np.degrees(np.arctan2(nose_to_eye[1], np.linalg.norm(nose_to_eye)))
            
            # Roll (наклон головы)
            roll = yaw  # Упрощенная оценка
            
            return {
                'yaw': float(yaw),
                'pitch': float(pitch),
                'roll': float(roll)
            }
            
        except Exception as e:
            logger.error(f"Ошибка оценки позы: {e}")
            return {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}

    def extract_68_landmarks_with_confidence(self, image: np.ndarray, 
                                           bbox: Optional[Tuple[int, int, int, int]] = None) -> Optional[LandmarksPackage]:
        """
        Извлечение 68 3D-ландмарков с оценкой достоверности
        
        Args:
            image: Входное изображение
            bbox: Bounding box лица (опционально)
            
        Returns:
            Пакет с ландмарками или None при ошибке
        """
        try:
            start_time = time.time()
            
            # Генерация ID изображения
            image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
            image_id = hashlib.sha256(image_bytes).hexdigest()
            
            # Проверка кэша
            if image_id in self.landmarks_cache:
                self.processing_stats['cache_hits'] += 1
                cached_result = self.landmarks_cache[image_id]
                cached_result.processing_time_ms = (time.time() - start_time) * 1000
                return cached_result
            
            # Детекция лица если bbox не предоставлен
            if bbox is None:
                faces = self.detect_faces(image)
                if not faces:
                    logger.warning("Лицо не обнаружено на изображении")
                    self.processing_stats['failed_extractions'] += 1
                    return None
                bbox = faces[0]  # Берем первое лицо
            
            # Предварительная оценка качества
            quality_assessment = self.assess_face_quality(image, bbox)
            if quality_assessment.overall_quality < QUALITY_THRESHOLDS['min_landmarks_confidence']:
                logger.warning(f"Низкое качество лица для анализа: {quality_assessment.overall_quality:.3f}")
            
            # Подготовка изображения для 3DDFA
            x, y, w, h = bbox
            face_crop = image[y:y+h, x:x+w]
            face_resized = cv2.resize(face_crop, TARGET_SIZE)
            
            # Извлечение параметров 3D-модели
            if self.ddfa_model is not None:
                param = self.ddfa_model(face_resized)
                
                # Извлечение 68 ландмарков
                landmarks_68 = self._extract_landmarks_from_param(param, bbox)
                
                # Субпиксельная коррекция ключевых точек
                landmarks_68_corrected = self._apply_subpixel_correction(image, landmarks_68)
                if landmarks_68_corrected is not None:
                    landmarks_68 = landmarks_68_corrected
                    self.processing_stats['subpixel_corrections'] += 1
                
                # Определение позы
                pose_result = self._calculate_detailed_pose_from_param(param, landmarks_68)
                
                # Классификация категории позы
                pose_category = self._determine_pose_category(pose_result.yaw, pose_result.pitch, pose_result.roll)
                
                # Расчет достоверности
                confidence = self._calculate_landmarks_confidence(landmarks_68, image, bbox)
                
                # Расчет ошибок формы
                shape_error, eye_region_error = self._calculate_shape_errors(landmarks_68)
                
                # Извлечение плотной поверхности (если включено)
                dense_vertices = None
                dense_triangles = None
                if (self.model_config.enable_dense_mesh and 
                    self.bfm_model is not None and 
                    DENSE_SURFACE_PARAMS['enable_dense_extraction']):
                    dense_vertices, dense_triangles = self._extract_dense_surface(param)
                    if dense_vertices is not None:
                        self.processing_stats['dense_mesh_extractions'] += 1
                
                # Анализ видимости и качества точек
                landmarks_visibility = self._analyze_landmarks_visibility(landmarks_68, image, bbox)
                landmarks_quality_scores = self._calculate_landmarks_quality_scores(landmarks_68, image)
                
                # Расчет стабильности позы
                pose_stability_score = self._calculate_pose_stability(pose_result)
                
                # Расчет симметрии лица
                facial_symmetry_score = self._calculate_facial_symmetry(landmarks_68)
                
                # Создание пакета результатов
                package = LandmarksPackage(
                    image_id=image_id,
                    filepath="",  # Будет заполнено вызывающей функцией
                    landmarks_68=landmarks_68,
                    landmarks_confidence=confidence,
                    pose_angles={
                        'yaw': pose_result.yaw,
                        'pitch': pose_result.pitch,
                        'roll': pose_result.roll
                    },
                    pose_category=pose_category,
                    bbox=bbox,
                    shape_error=shape_error,
                    eye_region_error=eye_region_error,
                    dense_vertices=dense_vertices,
                    dense_triangles=dense_triangles,
                    landmarks_visibility=landmarks_visibility,
                    landmarks_quality_scores=landmarks_quality_scores,
                    pose_stability_score=pose_stability_score,
                    facial_symmetry_score=facial_symmetry_score,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    memory_usage_mb=psutil.Process().memory_info().rss / (1024 * 1024),
                    device_used=self.model_config.device,
                    model_version=self.model_config.model_version,
                    cache_key=image_id,
                    cache_timestamp=datetime.datetime.now()
                )
                
                # Валидация результата
                self._validate_landmarks_package(package)
                
                # Сохранение в кэш
                self._save_to_cache(image_id, package)
                self.processing_stats['successful_extractions'] += 1
                self.processing_stats['total_processed'] += 1
                self.processing_stats['pose_estimations'] += 1
                
                logger.debug(f"Ландмарки извлечены за {package.processing_time_ms:.1f}мс")
                return package
            
            else:
                logger.error("3DDFA модель не инициализирована")
                self.processing_stats['failed_extractions'] += 1
                return None
                
        except Exception as e:
            logger.error(f"Ошибка извлечения ландмарков: {e}")
            self.processing_stats['failed_extractions'] += 1
            self.processing_stats['total_processed'] += 1
            return None

    def _extract_landmarks_from_param(self, param: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Извлечение 68 ландмарков из параметров 3DDFA"""
        try:
            # Реконструкция 3D-вершин из параметров
            if self.ddfa_model is not None:
                vertices = self.ddfa_model.recon_vers(param, roi_box=bbox)
                
                # Извлечение 68 ключевых точек
                landmarks_68 = vertices[:68]  # Первые 68 точек - это ландмарки
                
                return landmarks_68
            else:
                raise ValueError("3DDFA модель не инициализирована")
                
        except Exception as e:
            logger.error(f"Ошибка извлечения ландмарков из параметров: {e}")
            raise

    def _apply_subpixel_correction(self, image: np.ndarray, landmarks: np.ndarray) -> Optional[np.ndarray]:
        """Применение субпиксельной коррекции к ключевым точкам"""
        try:
            # Конвертация в grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Применение cornerSubPix к ключевым точкам
            corrected_landmarks = landmarks.copy()
            
            # Выбираем только 2D координаты для коррекции
            points_2d = landmarks[:, :2].astype(np.float32)
            
            # Проверяем, что точки находятся в пределах изображения
            h, w = gray.shape
            valid_mask = ((points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) & 
                         (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h))
            
            if np.any(valid_mask):
                valid_points = points_2d[valid_mask]
                
                # Применяем субпиксельную коррекцию
                corners = cv2.cornerSubPix(
                    gray, valid_points, SUBPIXEL_WIN_SIZE, 
                    SUBPIXEL_ZERO_ZONE, SUBPIXEL_CRITERIA
                )
                
                # Обновляем координаты только для валидных точек
                corrected_landmarks[valid_mask, :2] = corners
                
                logger.debug(f"Применена субпиксельная коррекция к {np.sum(valid_mask)} точкам")
                return corrected_landmarks
            else:
                logger.warning("Нет валидных точек для субпиксельной коррекции")
                return landmarks
            
        except Exception as e:
            logger.warning(f"Ошибка субпиксельной коррекции: {e}")
            return landmarks

    def _calculate_detailed_pose_from_param(self, param: np.ndarray, landmarks: np.ndarray) -> PoseEstimationResult:
        """Детальный расчет позы из параметров 3DDFA"""
        try:
            # Извлечение матрицы поворота из параметров
            # param содержит: [pitch, yaw, roll, tx, ty, tz, scale, ...]
            pitch = float(param[0])
            yaw = float(param[1])
            roll = float(param[2])
            
            # Извлечение трансляции
            tx = float(param[3])
            ty = float(param[4])
            tz = float(param[5])
            
            # Конвертация в градусы
            pitch_deg = np.degrees(pitch)
            yaw_deg = np.degrees(yaw)
            roll_deg = np.degrees(roll)
            
            # Создание матрицы поворота
            rotation_matrix = R.from_euler('xyz', [pitch, yaw, roll]).as_matrix()
            
            # Вектор трансляции
            translation_vector = np.array([tx, ty, tz])
            
            # Расчет достоверности позы на основе стабильности ландмарков
            pose_confidence = self._calculate_pose_confidence(landmarks, rotation_matrix)
            
            # Расчет стабильности позы
            stability_score = self._calculate_pose_stability_from_angles(pitch_deg, yaw_deg, roll_deg)
            
            # Определение категории позы
            pose_category = self._determine_pose_category(yaw_deg, pitch_deg, roll_deg)
            
            return PoseEstimationResult(
                yaw=yaw_deg,
                pitch=pitch_deg,
                roll=roll_deg,
                rotation_matrix=rotation_matrix,
                translation_vector=translation_vector,
                pose_confidence=pose_confidence,
                pose_category=pose_category,
                stability_score=stability_score
            )
            
        except Exception as e:
            logger.error(f"Ошибка расчета позы: {e}")
            return PoseEstimationResult(
                yaw=0.0, pitch=0.0, roll=0.0,
                rotation_matrix=np.eye(3),
                translation_vector=np.zeros(3),
                pose_confidence=0.0,
                pose_category='frontal',
                stability_score=0.0
            )

    def _calculate_pose_confidence(self, landmarks: np.ndarray, rotation_matrix: np.ndarray) -> float:
        """Расчет достоверности позы"""
        try:
            # Проверка консистентности ландмарков с матрицей поворота
            # Используем симметричные точки для проверки
            left_eye_points = landmarks[42:48, :3]
            right_eye_points = landmarks[36:42, :3]
            
            # Центры глаз
            left_eye_center = np.mean(left_eye_points, axis=0)
            right_eye_center = np.mean(right_eye_points, axis=0)
            
            # Ожидаемое расстояние между глазами (нормализованное)
            eye_distance = np.linalg.norm(left_eye_center - right_eye_center)
            
            # Проверка соответствия поворота и расположения глаз
            eye_line = right_eye_center - left_eye_center
            eye_line_normalized = eye_line / np.linalg.norm(eye_line)
            
            # Ожидаемое направление линии глаз после поворота
            expected_direction = rotation_matrix @ np.array([1, 0, 0])
            
            # Косинусное сходство
            similarity = np.dot(eye_line_normalized, expected_direction[:len(eye_line_normalized)])
            confidence = max(0.0, min(1.0, similarity))
            
            return float(confidence)
            
        except Exception as e:
            logger.error(f"Ошибка расчета достоверности позы: {e}")
            return 0.5

    def _calculate_pose_stability_from_angles(self, pitch: float, yaw: float, roll: float) -> float:
        """Расчет стабильности позы на основе углов"""
        try:
            # Стабильность обратно пропорциональна отклонению от нейтральной позы
            max_deviation = max(abs(pitch), abs(yaw), abs(roll))
            
            # Нормализация относительно максимального допустимого угла
            max_allowed_angle = 45.0
            stability = 1.0 - min(max_deviation / max_allowed_angle, 1.0)
            
            return float(max(0.0, stability))
            
        except Exception as e:
            logger.error(f"Ошибка расчета стабильности позы: {e}")
            return 0.0

    def _determine_pose_category(self, yaw: float, pitch: float, roll: float) -> str:
        """Определение категории позы по углам"""
        try:
            # Получение конфигураций ракурсов из конфига
            view_configs = self.config.get_view_configs()
            
            for view_name, view_config in view_configs.items():
                yaw_range = view_config['yaw']
                pitch_range = view_config['pitch']
                roll_range = view_config['roll']
                
                if (yaw_range[0] <= yaw <= yaw_range[1] and
                    pitch_range[0] <= pitch <= pitch_range[1] and
                    roll_range[0] <= roll <= roll_range[1]):
                    return view_name
            
            return 'frontal'  # По умолчанию
            
        except Exception as e:
            logger.error(f"Ошибка определения категории позы: {e}")
            return 'frontal'

    def _calculate_landmarks_confidence(self, landmarks: np.ndarray, 
                                      image: np.ndarray, 
                                      bbox: Tuple[int, int, int, int]) -> float:
        """Расчет достоверности извлеченных ландмарков"""
        try:
            confidence_factors = []
            
            # 1. Проверка, что все точки в пределах изображения
            h, w = image.shape[:2]
            valid_points = np.all((landmarks[:, 0] >= 0) & (landmarks[:, 0] < w) & 
                                (landmarks[:, 1] >= 0) & (landmarks[:, 1] < h))
            confidence_factors.append(1.0 if valid_points else 0.3)
            
            # 2. Проверка симметрии лица
            left_eye_center = np.mean(landmarks[42:48, :2], axis=0)
            right_eye_center = np.mean(landmarks[36:42, :2], axis=0)
            eye_distance = np.linalg.norm(left_eye_center - right_eye_center)
            
            # Проверка разумности расстояния между глазами
            bbox_width = bbox[2]
            normalized_eye_distance = eye_distance / bbox_width
            
            # Ожидаемое расстояние между глазами (примерно 0.3 от ширины лица)
            expected_ratio = 0.3
            eye_ratio_confidence = 1.0 - abs(normalized_eye_distance - expected_ratio) / expected_ratio
            confidence_factors.append(max(0.0, min(1.0, eye_ratio_confidence)))
            
            # 3. Проверка анатомической корректности
            nose_tip = landmarks[30, :2]
            mouth_center = np.mean(landmarks[48:68, :2], axis=0)
            
            # Расстояние нос-рот должно быть разумным
            nose_mouth_distance = np.linalg.norm(nose_tip - mouth_center)
            normalized_nm_distance = nose_mouth_distance / bbox_width
            
            # Ожидаемое расстояние нос-рот
            expected_nm_ratio = 0.15
            nm_confidence = 1.0 - abs(normalized_nm_distance - expected_nm_ratio) / expected_nm_ratio
            confidence_factors.append(max(0.0, min(1.0, nm_confidence)))
            
            # 4. Проверка консистентности Z-координат
            if landmarks.shape[1] >= 3:
                z_coords = landmarks[:, 2]
                z_std = np.std(z_coords)
                z_range = np.max(z_coords) - np.min(z_coords)
                
                # Z-координаты должны быть в разумных пределах
                if z_range > 0:
                    z_confidence = 1.0 - min(z_std / z_range, 1.0)
                    confidence_factors.append(max(0.0, z_confidence))
            
            # 5. Проверка качества ключевых точек глаз
            eye_quality = self._assess_eye_landmarks_quality(landmarks)
            confidence_factors.append(eye_quality)
            
            # Общая достоверность
            overall_confidence = np.mean(confidence_factors)
            
            return float(np.clip(overall_confidence, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Ошибка расчета достоверности: {e}")
            return 0.5

    def _assess_eye_landmarks_quality(self, landmarks: np.ndarray) -> float:
        """Оценка качества ландмарков глаз"""
        try:
            # Проверка формы глаз
            left_eye = landmarks[42:48, :2]
            right_eye = landmarks[36:42, :2]
            
            # Расчет площади глаз (приблизительно)
            left_eye_area = self._calculate_polygon_area(left_eye)
            right_eye_area = self._calculate_polygon_area(right_eye)
            
            # Проверка симметрии площадей глаз
            if left_eye_area > 0 and right_eye_area > 0:
                area_ratio = min(left_eye_area, right_eye_area) / max(left_eye_area, right_eye_area)
                return float(area_ratio)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Ошибка оценки качества глаз: {e}")
            return 0.5

    def _calculate_polygon_area(self, points: np.ndarray) -> float:
        """Расчет площади полигона по координатам точек"""
        try:
            if len(points) < 3:
                return 0.0
            
            # Формула площади полигона (Shoelace formula)
            x = points[:, 0]
            y = points[:, 1]
            
            area = 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] 
                                for i in range(-1, len(x)-1)))
            return area
            
        except Exception as e:
            logger.error(f"Ошибка расчета площади полигона: {e}")
            return 0.0

    def _calculate_shape_errors(self, landmarks: np.ndarray) -> Tuple[float, float]:
        """Расчет ошибок формы лица"""
        try:
            # Общая ошибка формы (среднеквадратичное отклонение от среднего)
            center = np.mean(landmarks[:, :2], axis=0)
            distances = np.linalg.norm(landmarks[:, :2] - center, axis=1)
            shape_error = float(np.std(distances))
            
            # Ошибка области глаз
            left_eye = landmarks[42:48, :2]
            right_eye = landmarks[36:42, :2]
            
            left_eye_center = np.mean(left_eye, axis=0)
            right_eye_center = np.mean(right_eye, axis=0)
            
            left_eye_distances = np.linalg.norm(left_eye - left_eye_center, axis=1)
            right_eye_distances = np.linalg.norm(right_eye - right_eye_center, axis=1)
            
            eye_region_error = float(np.mean([np.std(left_eye_distances), np.std(right_eye_distances)]))
            
            return shape_error, eye_region_error
            
        except Exception as e:
            logger.error(f"Ошибка расчета ошибок формы: {e}")
            return 0.0, 0.0

    def _extract_dense_surface(self, param: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Извлечение плотной поверхности лица (38365 точек)"""
        try:
            if self.bfm_model is None:
                return None, None
            
            # Извлечение коэффициентов формы и выражения
            shape_coeff = param[12:52]  # 40 коэффициентов формы
            exp_coeff = param[52:62] if len(param) > 62 else np.zeros(10)  # 10 коэффициентов выражения
            
            # Реконструкция 3D-вершин
            shape_base = self.bfm_model['shapeMU'] + self.bfm_model['shapePC'] @ shape_coeff
            
            # Добавление выражения (если доступно)
            if 'expPC' in self.bfm_model and self.bfm_model['expPC'] is not None:
                exp_base = self.bfm_model['expPC'] @ exp_coeff
                vertices = (shape_base + exp_base).reshape(-1, 3)
            else:
                vertices = shape_base.reshape(-1, 3)
            
            # Ограничение количества точек
            if len(vertices) > DENSE_SURFACE_PARAMS['dense_points_limit']:
                vertices = vertices[:DENSE_SURFACE_PARAMS['dense_points_limit']]
            
            # Применение сглаживания если включено
            if DENSE_SURFACE_PARAMS['surface_smoothing']:
                vertices = self._apply_surface_smoothing(vertices)
            
            return vertices, self.triangles
            
        except Exception as e:
            logger.warning(f"Ошибка извлечения плотной поверхности: {e}")
            return None, None

    def _apply_surface_smoothing(self, vertices: np.ndarray) -> np.ndarray:
        """Применение сглаживания к поверхности"""
        try:
            # Простое сглаживание через усреднение соседних точек
            if self.triangles is not None and len(vertices) > 100:
                smoothed_vertices = vertices.copy()
                
                # Для каждой вершины усредняем с соседними
                for i in range(len(vertices)):
                    # Находим треугольники, содержащие данную вершину
                    connected_triangles = self.triangles[np.any(self.triangles == i, axis=1)]
                    
                    if len(connected_triangles) > 0:
                        # Получаем все соседние вершины
                        neighbors = np.unique(connected_triangles.flatten())
                        neighbors = neighbors[neighbors != i]  # Исключаем саму вершину
                        
                        if len(neighbors) > 0:
                            # Усредняем позицию с соседними вершинами
                            neighbor_positions = vertices[neighbors]
                            smoothed_position = np.mean(neighbor_positions, axis=0)
                            
                            # Применяем сглаживание с весом 0.3
                            smoothed_vertices[i] = 0.7 * vertices[i] + 0.3 * smoothed_position
                
                return smoothed_vertices
            else:
                return vertices
                
        except Exception as e:
            logger.warning(f"Ошибка сглаживания поверхности: {e}")
            return vertices

    def _analyze_landmarks_visibility(self, landmarks: np.ndarray, 
                                    image: np.ndarray, 
                                    bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Анализ видимости каждой ландмарки"""
        try:
            visibility = np.ones(len(landmarks), dtype=float)
            h, w = image.shape[:2]
            
            for i, point in enumerate(landmarks):
                x, y = point[:2]
                
                # Проверка, что точка в пределах изображения
                if x < 0 or x >= w or y < 0 or y >= h:
                    visibility[i] = 0.0
                    continue
                
                # Проверка локального контраста вокруг точки
                try:
                    x_int, y_int = int(x), int(y)
                    window_size = 5
                    
                    y1 = max(0, y_int - window_size)
                    y2 = min(h, y_int + window_size + 1)
                    x1 = max(0, x_int - window_size)
                    x2 = min(w, x_int + window_size + 1)
                    
                    if len(image.shape) == 3:
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = image
                    
                    window = gray[y1:y2, x1:x2]
                    
                    if window.size > 0:
                        contrast = np.std(window.astype(float))
                        # Нормализация контраста к диапазону [0, 1]
                        visibility[i] = min(1.0, contrast / 30.0)
                    
                except Exception:
                    visibility[i] = 0.5  # Нейтральная видимость при ошибке
            
            return visibility
            
        except Exception as e:
            logger.error(f"Ошибка анализа видимости ландмарков: {e}")
            return np.ones(len(landmarks), dtype=float) * 0.5

    def _calculate_landmarks_quality_scores(self, landmarks: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Расчет балла качества для каждой ландмарки"""
        try:
            quality_scores = np.zeros(len(landmarks))
            
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            h, w = gray.shape
            
            for i, point in enumerate(landmarks):
                x, y = point[:2]
                
                if 0 <= x < w and 0 <= y < h:
                    # Анализ локальных градиентов
                    x_int, y_int = int(x), int(y)
                    
                    # Размер окна для анализа
                    window_size = 3
                    y1 = max(0, y_int - window_size)
                    y2 = min(h, y_int + window_size + 1)
                    x1 = max(0, x_int - window_size)
                    x2 = min(w, x_int + window_size + 1)
                    
                    window = gray[y1:y2, x1:x2].astype(float)
                    
                    if window.size > 0:
                        # Расчет градиентов
                        grad_x = cv2.Sobel(window, cv2.CV_64F, 1, 0, ksize=3)
                        grad_y = cv2.Sobel(window, cv2.CV_64F, 0, 1, ksize=3)
                        
                        # Магнитуда градиента
                        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                        
                        # Качество как средняя магнитуда градиента
                        quality = np.mean(gradient_magnitude) / 255.0
                        quality_scores[i] = min(1.0, quality)
                    else:
                        quality_scores[i] = 0.0
                else:
                    quality_scores[i] = 0.0
            
            return quality_scores
            
        except Exception as e:
            logger.error(f"Ошибка расчета качества ландмарков: {e}")
            return np.ones(len(landmarks)) * 0.5

    def _calculate_pose_stability(self, pose_result: PoseEstimationResult) -> float:
        """Расчет стабильности позы"""
        try:
            # Стабильность на основе достоверности и углов
            angle_stability = pose_result.stability_score
            confidence_stability = pose_result.pose_confidence
            
            # Проверка экстремальных углов
            max_angle = max(abs(pose_result.yaw), abs(pose_result.pitch), abs(pose_result.roll))
            angle_penalty = max(0.0, (max_angle - 30.0) / 60.0)  # Штраф за углы > 30°
            
            stability = (angle_stability + confidence_stability) / 2.0 - angle_penalty
            
            return float(max(0.0, min(1.0, stability)))
            
        except Exception as e:
            logger.error(f"Ошибка расчета стабильности позы: {e}")
            return 0.0

    def _validate_landmarks_package(self, package: LandmarksPackage):
        """Валидация пакета ландмарков"""
        warnings = []
        quality_flags = []
        
        try:
            # Проверка достоверности
            if package.landmarks_confidence < CONFIDENCE_THRESHOLD:
                warnings.append(f"Низкая достоверность: {package.landmarks_confidence:.3f}")
                quality_flags.append("low_confidence")
            
            # Проверка углов позы
            for angle_name, angle_value in package.pose_angles.items():
                min_val, max_val = POSE_ANGLE_LIMITS[angle_name]
                if not (min_val <= angle_value <= max_val):
                    warnings.append(f"Экстремальный угол {angle_name}: {angle_value:.1f}°")
                    quality_flags.append(f"extreme_{angle_name}")
            
            # Проверка ошибок формы
            if package.shape_error > 50.0:  # Пиксели
                warnings.append(f"Высокая ошибка формы: {package.shape_error:.1f}")
                quality_flags.append("high_shape_error")
            
            if package.eye_region_error > 20.0:  # Пиксели
                warnings.append(f"Высокая ошибка области глаз: {package.eye_region_error:.1f}")
                quality_flags.append("high_eye_error")
            
            # Проверка видимости ландмарков
            if package.landmarks_visibility is not None:
                low_visibility_count = np.sum(package.landmarks_visibility < 0.5)
                if low_visibility_count > 10:  # Более 10 невидимых точек
                    warnings.append(f"Низкая видимость {low_visibility_count} ландмарков")
                    quality_flags.append("poor_visibility")
            
            # Проверка качества отдельных точек
            if package.landmarks_quality_scores is not None:
                low_quality_count = np.sum(package.landmarks_quality_scores < 0.3)
                if low_quality_count > 15:  # Более 15 точек низкого качества
                    warnings.append(f"Низкое качество {low_quality_count} ландмарков")
                    quality_flags.append("poor_landmark_quality")
            
            # Проверка симметрии лица
            if package.facial_symmetry_score < 0.6:
                warnings.append(f"Низкая симметрия лица: {package.facial_symmetry_score:.3f}")
                quality_flags.append("asymmetric_face")
            
            # Проверка стабильности позы
            if package.pose_stability_score < 0.5:
                warnings.append(f"Нестабильная поза: {package.pose_stability_score:.3f}")
                quality_flags.append("unstable_pose")
            
            # Проверка плотной поверхности
            if (package.dense_vertices is not None and 
                len(package.dense_vertices) < DENSE_SURFACE_PARAMS['dense_points_limit'] * 0.8):
                warnings.append("Неполная плотная поверхность")
                quality_flags.append("incomplete_dense_mesh")
            
            package.warnings = warnings
            package.quality_flags = quality_flags
            
            logger.debug(f"Валидация завершена: {len(warnings)} предупреждений, {len(quality_flags)} флагов")
            
        except Exception as e:
            logger.error(f"Ошибка валидации пакета ландмарков: {e}")
            package.warnings = [f"Ошибка валидации: {str(e)}"]
            package.quality_flags = ["validation_error"]

    def normalize_landmarks_by_pose_category(self, package: LandmarksPackage) -> LandmarksPackage:
        """
        Нормализация ландмарков по категории позы
        
        Args:
            package: Пакет с ландмарками
            
        Returns:
            Обновленный пакет с нормализованными ландмарками
        """
        try:
            # Получение конфигурации для данной позы
            view_configs = self.config.get_view_configs()
            pose_config = view_configs.get(package.pose_category, view_configs['frontal'])
            
            # Получение опорных точек
            reference_points = pose_config['reference_points']
            scale_type = pose_config['scale_type']
            
            # Расчет масштабного коэффициента
            scale_factor, reference_distance = self._calculate_scale_factor(
                package.landmarks_68, reference_points, scale_type
            )
            
            # Нормализация ландмарков
            normalized_landmarks = package.landmarks_68.copy()
            
            # Центрирование по носу (точка 30)
            nose_center = package.landmarks_68[30, :2]
            normalized_landmarks[:, :2] -= nose_center
            
            # Масштабирование
            if scale_factor > 0:
                normalized_landmarks[:, :2] /= scale_factor
            else:
                logger.warning("Нулевой масштабный коэффициент, нормализация пропущена")
            
            # Обновление пакета
            package.normalized_landmarks = normalized_landmarks
            package.scale_factor = scale_factor
            package.reference_distance = reference_distance
            
            logger.debug(f"Ландмарки нормализованы для позы {package.pose_category}, масштаб: {scale_factor:.3f}")
            return package
            
        except Exception as e:
            logger.error(f"Ошибка нормализации ландмарков: {e}")
            package.warnings.append(f"Ошибка нормализации: {str(e)}")
            return package

    def _calculate_scale_factor(self, landmarks: np.ndarray, 
                              reference_points: List[int], 
                              scale_type: str) -> Tuple[float, float]:
        """Расчет масштабного коэффициента"""
        try:
            reference_distances = self.config.get_reference_distances()
            
            if scale_type == 'IOD':
                # Межзрачковое расстояние
                left_eye_center = np.mean(landmarks[42:48, :2], axis=0)
                right_eye_center = np.mean(landmarks[36:42, :2], axis=0)
                measured_distance = np.linalg.norm(left_eye_center - right_eye_center)
                expected_distance = reference_distances['IOD']
                
            elif scale_type == 'nose_eye':
                # Расстояние нос-глаз
                nose_tip = landmarks[30, :2]
                eye_center = np.mean(landmarks[36:48, :2], axis=0)
                measured_distance = np.linalg.norm(nose_tip - eye_center)
                expected_distance = reference_distances['nose_eye']
                
            elif scale_type == 'face_height':
                # Высота лица
                top_point = landmarks[27, :2]  # Переносица
                bottom_point = landmarks[8, :2]  # Подбородок
                measured_distance = np.linalg.norm(top_point - bottom_point)
                expected_distance = reference_distances['face_height']
                
            elif scale_type == 'profile_height':
                # Высота профиля
                top_point = landmarks[27, :2]
                bottom_point = landmarks[8, :2]
                measured_distance = np.linalg.norm(top_point - bottom_point)
                expected_distance = reference_distances['profile_height']
                
            else:
                # По умолчанию используем IOD
                left_eye_center = np.mean(landmarks[42:48, :2], axis=0)
                right_eye_center = np.mean(landmarks[36:42, :2], axis=0)
                measured_distance = np.linalg.norm(left_eye_center - right_eye_center)
                expected_distance = reference_distances['IOD']
            
            scale_factor = measured_distance / expected_distance if expected_distance > 0 else 1.0
            
            return float(scale_factor), float(measured_distance)
            
        except Exception as e:
            logger.error(f"Ошибка расчета масштабного коэффициента: {e}")
            return 1.0, 0.0

    def determine_precise_face_pose(self, package: LandmarksPackage) -> Dict[str, float]:
        """
        Точное определение позы лица с дополнительными метриками
        
        Args:
            package: Пакет с ландмарками
            
        Returns:
            Расширенный словарь с углами позы
        """
        try:
            pose_metrics = package.pose_angles.copy()
            
            # Дополнительные метрики позы
            landmarks = package.landmarks_68
            
            # Наклон головы (по линии глаз)
            left_eye_center = np.mean(landmarks[42:48, :2], axis=0)
            right_eye_center = np.mean(landmarks[36:42, :2], axis=0)
            eye_line_angle = np.degrees(np.arctan2(
                left_eye_center[1] - right_eye_center[1],
                left_eye_center[0] - right_eye_center[0]
            ))
            pose_metrics['head_tilt'] = float(eye_line_angle)
            
            # Асимметрия лица
            nose_tip = landmarks[30, :2]
            left_face_points = landmarks[0:9, :2]  # Левая сторона лица
            right_face_points = landmarks[8:17, :2]  # Правая сторона лица
            
            left_distances = np.linalg.norm(left_face_points - nose_tip, axis=1)
            right_distances = np.linalg.norm(right_face_points - nose_tip, axis=1)
            
            asymmetry = float(np.mean(np.abs(left_distances - right_distances[::-1])))
            pose_metrics['face_asymmetry'] = asymmetry
            
            # Степень поворота (комбинированная метрика)
            rotation_magnitude = np.sqrt(
                pose_metrics['yaw']**2 + 
                pose_metrics['pitch']**2 + 
                pose_metrics['roll']**2
            )
            pose_metrics['rotation_magnitude'] = float(rotation_magnitude)
            
            # Оценка фронтальности
            frontality_score = 1.0 - min(rotation_magnitude / 45.0, 1.0)
            pose_metrics['frontality_score'] = float(frontality_score)
            
            # Оценка профильности
            profile_score = abs(pose_metrics['yaw']) / 90.0
            pose_metrics['profile_score'] = float(min(profile_score, 1.0))
            
            return pose_metrics
            
        except Exception as e:
            logger.error(f"Ошибка определения точной позы: {e}")
            return package.pose_angles

    def calculate_comprehensive_shape_error(self, package: LandmarksPackage) -> Dict[str, float]:
        """
        Комплексный расчет ошибок формы лица
        
        Args:
            package: Пакет с ландмарками
            
        Returns:
            Словарь с различными метриками ошибок
        """
        try:
            landmarks = package.landmarks_68
            shape_errors = {}
            
            # Общая ошибка формы
            shape_errors['overall_shape_error'] = package.shape_error
            shape_errors['eye_region_error'] = package.eye_region_error
            
            # Ошибки отдельных областей
            regions = {
                'jaw': landmarks[0:17],
                'right_eyebrow': landmarks[17:22],
                'left_eyebrow': landmarks[22:27],
                'nose': landmarks[27:36],
                'right_eye': landmarks[36:42],
                'left_eye': landmarks[42:48],
                'mouth': landmarks[48:68]
            }
            
            for region_name, region_points in regions.items():
                if len(region_points) > 1:
                    center = np.mean(region_points[:, :2], axis=0)
                    distances = np.linalg.norm(region_points[:, :2] - center, axis=1)
                    region_error = float(np.std(distances))
                    shape_errors[f'{region_name}_error'] = region_error
            
            # Симметрия лица
            left_points = landmarks[0:9, :2]  # Левая сторона
            right_points = landmarks[8:17, :2]  # Правая сторона (в обратном порядке)
            
            nose_center = landmarks[30, :2]
            left_distances = np.linalg.norm(left_points - nose_center, axis=1)
            right_distances = np.linalg.norm(right_points[::-1] - nose_center, axis=1)
            
            symmetry_error = float(np.mean(np.abs(left_distances - right_distances)))
            shape_errors['symmetry_error'] = symmetry_error
            
            # Пропорциональность
            eye_distance = np.linalg.norm(
                np.mean(landmarks[42:48, :2], axis=0) - np.mean(landmarks[36:42, :2], axis=0)
            )
            nose_mouth_distance = np.linalg.norm(
                landmarks[30, :2] - np.mean(landmarks[48:68, :2], axis=0)
            )
            
            proportion_ratio = nose_mouth_distance / eye_distance if eye_distance > 0 else 0
            expected_ratio = 0.5  # Примерное ожидаемое соотношение
            proportion_error = float(abs(proportion_ratio - expected_ratio))
            shape_errors['proportion_error'] = proportion_error
            
            # Компактность лица
            face_center = np.mean(landmarks[:, :2], axis=0)
            distances_from_center = np.linalg.norm(landmarks[:, :2] - face_center, axis=1)
            compactness = float(np.std(distances_from_center) / np.mean(distances_from_center))
            shape_errors['compactness_error'] = compactness
            
            # Анализ контура лица
            jaw_points = landmarks[0:17, :2]
            jaw_smoothness = self._calculate_contour_smoothness(jaw_points)
            shape_errors['jaw_smoothness_error'] = float(jaw_smoothness)
            
            return shape_errors
            
        except Exception as e:
            logger.error(f"Ошибка расчета комплексных ошибок формы: {e}")
            return {'overall_shape_error': package.shape_error, 'eye_region_error': package.eye_region_error}

    def _calculate_contour_smoothness(self, contour_points: np.ndarray) -> float:
        """Расчет гладкости контура"""
        try:
            if len(contour_points) < 3:
                return 0.0
            
            # Расчет кривизны в каждой точке
            curvatures = []
            for i in range(1, len(contour_points) - 1):
                p1 = contour_points[i-1]
                p2 = contour_points[i]
                p3 = contour_points[i+1]
                
                # Векторы
                v1 = p2 - p1
                v2 = p3 - p2
                
                # Угол между векторами
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                
                curvatures.append(angle)
            
            # Стандартное отклонение кривизны как мера негладкости
            return np.std(curvatures) if curvatures else 0.0
            
        except Exception as e:
            logger.error(f"Ошибка расчета гладкости контура: {e}")
            return 0.0

    def extract_dense_surface_points(self, package: LandmarksPackage) -> Optional[np.ndarray]:
        """
        Извлечение точек плотной поверхности лица
        
        Args:
            package: Пакет с ландмарками
            
        Returns:
            Массив точек плотной поверхности или None
        """
        try:
            if not self.model_config.enable_dense_mesh or self.bfm_model is None:
                return None
            
            if package.dense_vertices is not None:
                return package.dense_vertices
            
            # Если плотная поверхность не была извлечена ранее, пытаемся извлечь сейчас
            logger.warning("Плотная поверхность не была извлечена при основном анализе")
            return None
            
        except Exception as e:
            logger.error(f"Ошибка извлечения плотной поверхности: {e}")
            return None

    def analyze_landmark_stability(self, packages: List[LandmarksPackage]) -> Dict[str, float]:
        """
        Анализ стабильности ландмарков во времени
        
        Args:
            packages: Список пакетов ландмарков
            
        Returns:
            Словарь с метриками стабильности
        """
        try:
            if len(packages) < 2:
                return {'stability_score': 1.0, 'temporal_variance': 0.0}
            
            # Сортировка по времени
            sorted_packages = sorted(packages, key=lambda p: p.cache_timestamp or datetime.datetime.now())
            
            # Извлечение нормализованных ландмарков
            landmarks_sequence = []
            for pkg in sorted_packages:
                if pkg.normalized_landmarks is not None:
                    landmarks_sequence.append(pkg.normalized_landmarks[:, :2])
                else:
                    landmarks_sequence.append(pkg.landmarks_68[:, :2])
            
            if len(landmarks_sequence) < 2:
                return {'stability_score': 1.0, 'temporal_variance': 0.0}
            
            # Расчет временной вариации
            landmarks_array = np.array(landmarks_sequence)  # [time, points, coords]
            temporal_variance = np.mean(np.var(landmarks_array, axis=0))
            
            # Расчет стабильности (обратная величина вариации)
            stability_score = 1.0 / (1.0 + temporal_variance)
            
            # Анализ трендов
            trends = self._analyze_landmark_trends(landmarks_array)
            
            return {
                'stability_score': float(stability_score),
                'temporal_variance': float(temporal_variance),
                'trend_magnitude': float(trends['magnitude']),
                'trend_direction': trends['direction']
            }
            
        except Exception as e:
            logger.error(f"Ошибка анализа стабильности ландмарков: {e}")
            return {'stability_score': 0.5, 'temporal_variance': 0.0}

    def _analyze_landmark_trends(self, landmarks_array: np.ndarray) -> Dict[str, Any]:
        """Анализ трендов изменения ландмарков"""
        try:
            # Расчет средних смещений между соседними кадрами
            displacements = np.diff(landmarks_array, axis=0)
            mean_displacement = np.mean(np.linalg.norm(displacements, axis=2))
            
            # Направление общего тренда
            total_displacement = landmarks_array[-1] - landmarks_array[0]
            trend_vector = np.mean(total_displacement, axis=0)
            trend_magnitude = np.linalg.norm(trend_vector)
            
            # Классификация направления
            if trend_magnitude < 1.0:
                direction = 'stable'
            elif abs(trend_vector[1]) > abs(trend_vector[0]):
                direction = 'vertical' if trend_vector[1] > 0 else 'vertical_up'
            else:
                direction = 'horizontal_right' if trend_vector[0] > 0 else 'horizontal_left'
            
            return {
                'magnitude': trend_magnitude,
                'direction': direction,
                'mean_displacement': float(mean_displacement)
            }
            
        except Exception as e:
            logger.error(f"Ошибка анализа трендов: {e}")
            return {'magnitude': 0.0, 'direction': 'unknown', 'mean_displacement': 0.0}

    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики обработки
        
        Returns:
            Словарь со статистикой
        """
        stats = self.processing_stats.copy()
        
        if stats['total_processed'] > 0:
            stats['success_rate'] = stats['successful_extractions'] / stats['total_processed']
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_processed']
        else:
            stats['success_rate'] = 0.0
            stats['cache_hit_rate'] = 0.0
        
        # Информация о модели
        stats['model_info'] = {
            'device': self.model_config.device,
            'use_mps': self.model_config.use_mps,
            'dense_mesh_enabled': self.model_config.enable_dense_mesh,
            'model_loaded': self.ddfa_model is not None,
            'bfm_loaded': self.bfm_model is not None,
            'model_version': self.model_config.model_version
        }
        
        # Информация о кэше
        stats['cache_info'] = {
            'cache_size_mb': self.cache_size_mb,
            'cache_entries': len(self.landmarks_cache),
            'max_cache_size_mb': self.max_cache_size_mb
        }
        
        # Производительность
        process = psutil.Process()
        memory_info = process.memory_info()
        stats['performance'] = {
            'memory_usage_mb': memory_info.rss / (1024 * 1024),
            'cpu_percent': process.cpu_percent(),
            'thread_count': process.num_threads()
        }
        
        return stats

    def _save_to_cache(self, cache_key: str, package: LandmarksPackage):
        """Сохранение пакета в кэш"""
        try:
            self.landmarks_cache[cache_key] = package
            
            # Проверка лимита кэша
            if self.cache_size_mb > self.max_cache_size_mb:
                self._cleanup_cache()
                
        except Exception as e:
            logger.error(f"Ошибка сохранения в кэш: {e}")

    def _cleanup_cache(self):
        """Очистка кэша при превышении лимита"""
        try:
            # Удаляем старые записи
            sorted_items = sorted(
                self.landmarks_cache.items(),
                key=lambda x: x[1].cache_timestamp or datetime.datetime.min
            )
            
            # Удаляем половину записей
            items_to_remove = len(sorted_items) // 2
            for i in range(items_to_remove):
                del self.landmarks_cache[sorted_items[i][0]]
            
            self.cache_size_mb *= 0.5  # Примерная оценка
            logger.info(f"Кэш очищен: удалено {items_to_remove} записей")
            
        except Exception as e:
            logger.error(f"Ошибка очистки кэша: {e}")

    def clear_cache(self):
        """Полная очистка кэша результатов"""
        try:
            self.landmarks_cache.clear()
            self.cache_size_mb = 0.0
            
            # Очистка файлов кэша
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            
            logger.info("Кэш Face3DAnalyzer полностью очищен")
            
        except Exception as e:
            logger.error(f"Ошибка очистки кэша: {e}")

    def save_cache(self, cache_filename: str = "face3d_cache.pkl"):
        """Сохранение кэша на диск"""
        try:
            cache_path = self.cache_dir / cache_filename
            
            with open(cache_path, 'wb') as f:
                pickle.dump(self.landmarks_cache, f)
            
            logger.info(f"Кэш сохранен: {cache_path}")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения кэша: {e}")

    def load_cache(self, cache_filename: str = "face3d_cache.pkl") -> bool:
        """Загрузка кэша с диска"""
        try:
            cache_path = self.cache_dir / cache_filename
            
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    self.landmarks_cache = pickle.load(f)
                
                logger.info(f"Кэш загружен: {cache_path}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Ошибка загрузки кэша: {e}")
            return False

    def __del__(self):
        """Деструктор для освобождения ресурсов"""
        try:
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=False)
            
            # Очистка GPU памяти
            if hasattr(torch, 'mps') and torch.mps.is_available():
                torch.mps.empty_cache()
                
        except Exception:
            pass

# === ФУНКЦИИ САМОТЕСТИРОВАНИЯ ===

def self_test():
    """Самотестирование модуля face_3d_analyzer"""
    try:
        logger.info("Запуск самотестирования face_3d_analyzer...")
        
        # Создание экземпляра анализатора
        analyzer = Face3DAnalyzer()
        
        # Создание тестового изображения
        test_image = np.random.randint(0, 255, (800, 800, 3), dtype=np.uint8)
        
        # Тест детекции лиц
        faces = analyzer.detect_faces(test_image)
        logger.info(f"Обнаружено лиц: {len(faces)}")
        
        # Тест определения категории позы
        test_angles = {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
        pose_category = analyzer._determine_pose_category(0.0, 0.0, 0.0)
        assert pose_category == 'frontal', "Неверная категория позы"
        
        # Тест расчета масштабного коэффициента
        test_landmarks = np.random.rand(68, 3) * 100
        scale_factor, ref_distance = analyzer._calculate_scale_factor(
            test_landmarks, [36, 45], 'IOD'
        )
        assert scale_factor > 0, "Неверный масштабный коэффициент"
        
        # Тест оценки качества лица
        test_bbox = (100, 100, 200, 200)
        quality_assessment = analyzer.assess_face_quality(test_image, test_bbox)
        assert hasattr(quality_assessment, 'overall_quality'), "Отсутствует оценка качества"
        
        # Тест статистики
        stats = analyzer.get_processing_statistics()
        assert 'success_rate' in stats, "Отсутствует статистика"
        
        logger.info("Самотестирование face_3d_analyzer завершено успешно")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка самотестирования: {e}")
        return False

# === ИНИЦИАЛИЗАЦИЯ ===

if __name__ == "__main__":
    # Запуск самотестирования при прямом вызове модуля
    success = self_test()
    if success:
        print("✅ Модуль face_3d_analyzer работает корректно")
        
        # Демонстрация основной функциональности
        analyzer = Face3DAnalyzer()
        print(f"📊 Устройство: {analyzer.model_config.device}")
        print(f"🔧 MPS доступен: {analyzer.model_config.use_mps}")
        print(f"📏 Целевой размер: {TARGET_SIZE}")
        print(f"💾 Кэш-директория: {analyzer.cache_dir}")
        print(f"🎯 Плотная поверхность: {analyzer.model_config.enable_dense_mesh}")
        print(f"📈 Максимум точек: {DENSE_SURFACE_PARAMS['dense_points_limit']}")
    else:
        print("❌ Обнаружены ошибки в модуле face_3d_analyzer")
        exit(1)
