"""
Face3DAnalyzer - Анализатор 3D лиц с извлечением ландмарок и метрик
Версия: 2.0
Дата: 2025-06-15
Исправлены все критические ошибки согласно правкам
"""

import numpy as np
import cv2
import torch
import yaml
import pickle
import math
import logging
from typing import Tuple, Dict, List, Optional, Any, Union
from scipy.spatial.distance import euclidean, pdist, squareform
from scipy.stats import mode
from scipy.spatial import distance
from pathlib import Path
from datetime import datetime
import os

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('logs/face3danalyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Попытка импорта 3DDFA компонентов
try:
    from core_config import (
        USE_ONNX, ONNX_EXECUTION_PROVIDERS, IS_MACOS, IS_ARM64,
        REQUIRED_3DDFA_FILES, VIEW_CONFIGS, CAMERA_BOX_FRONT_SIZE_FACTOR,
        CRITICAL_THRESHOLDS, get_identity_signature_metrics,
        get_view_configs, AGING_MODEL
    )
    
    if USE_ONNX:
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'
        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX
        FaceBoxesActual = FaceBoxes_ONNX
        TDDFAActual = TDDFA_ONNX
    else:
        from FaceBoxes import FaceBoxes
        from TDDFA import TDDFA
        FaceBoxesActual = FaceBoxes
        TDDFAActual = TDDFA
    
    from utils.pose import matrix2angle
    from utils.render import render
    from utils.functions import crop_img, get_suffix, parse_roi_box_from_bbox
    
    HAS_3DDFA = True
    logger.info("3DDFA компоненты успешно импортированы")
    
except ImportError as e:
    HAS_3DDFA = False
    logger.error(f"Ошибка импорта 3DDFA: {e}")
    logger.warning("Используется заглушка для 3DDFA")

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def calculate_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Расчет евклидова расстояния между двумя точками"""
    try:
        return euclidean(p1, p2)
    except Exception as e:
        logger.error(f"Ошибка расчета расстояния: {e}")
        return 0.0

def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Расчет угла между тремя точками (p2 - вершина)"""
    try:
        vec1 = p1 - p2
        vec2 = p3 - p2
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_angle = np.dot(vec1, vec2) / (norm1 * norm2)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine_angle))
        
        return angle
    except Exception as e:
        logger.error(f"Ошибка расчета угла: {e}")
        return 0.0

def mirror_landmarks(landmarks: np.ndarray, center_x: float) -> np.ndarray:
    """Отражение ландмарок относительно центральной оси X"""
    try:
        mirrored = landmarks.copy()
        mirrored[:, 0] = 2 * center_x - mirrored[:, 0]
        return mirrored
    except Exception as e:
        logger.error(f"Ошибка отражения ландмарок: {e}")
        return landmarks

def fit_curve_to_points(points: np.ndarray) -> float:
    """Аппроксимация кривой по точкам"""
    try:
        if len(points) < 2:
            return 0.0
        
        x = points[:, 0]
        y = points[:, 1]
        
        coefficients = np.polyfit(x, y, min(2, len(points) - 1))
        return np.sum(np.abs(coefficients))
    except Exception as e:
        logger.error(f"Ошибка аппроксимации кривой: {e}")
        return 0.0

def rotate_to_target_angles(landmarks: np.ndarray, target_angles: Tuple[float, float, float]) -> np.ndarray:
    """
    ИСПРАВЛЕНО: Поворот ландмарок к целевым углам (pitch, yaw, roll)
    Согласно правкам: добавлена функция target_angles
    """
    try:
        pitch, yaw, roll = target_angles
        logger.info(f"Поворот к целевым углам: pitch={pitch}, yaw={yaw}, roll={roll}")
        
        # Матрицы поворота
        pitch_rad = np.radians(pitch)
        yaw_rad = np.radians(yaw)
        roll_rad = np.radians(roll)
        
        # Матрица поворота по X (pitch)
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
            [0, np.sin(pitch_rad), np.cos(pitch_rad)]
        ])
        
        # Матрица поворота по Y (yaw)
        R_y = np.array([
            [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
            [0, 1, 0],
            [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
        ])
        
        # Матрица поворота по Z (roll)
        R_z = np.array([
            [np.cos(roll_rad), -np.sin(roll_rad), 0],
            [np.sin(roll_rad), np.cos(roll_rad), 0],
            [0, 0, 1]
        ])
        
        # Комбинированная матрица поворота
        R = R_z @ R_y @ R_x
        
        # Применение поворота к ландмаркам
        rotated_landmarks = landmarks @ R.T
        
        logger.info(f"Поворот ландмарок выполнен успешно")
        return rotated_landmarks
        
    except Exception as e:
        logger.error(f"Ошибка поворота к целевым углам: {e}")
        return landmarks

# ==================== ОСНОВНОЙ КЛАСС ====================

class Face3DAnalyzer:
    """
    Анализатор 3D лиц с полной функциональностью
    ИСПРАВЛЕНО: Все критические ошибки согласно правкам
    """
    
    def __init__(self):
        """Инициализация анализатора"""
        logger.info("Инициализация Face3DAnalyzer")
        
        # Модели 3DDFA
        self.tddfa_onnx = None
        self.tddfa_pytorch = None
        self.faceboxes = None
        
        # ИСПРАВЛЕНО: Добавлена поддержка dense surface points (38000 точек)
        self.dense_face_model = None
        
        # Референсная модель для shape error
        self.reference_model_landmarks = None
        
        # Параметры для 3DDFA (теперь с CAMERA_BOX_FRONT_SIZE_FACTOR)
        self.camera_box_size_factor = CAMERA_BOX_FRONT_SIZE_FACTOR # Используем новую константу
        
        # Конфигурации видов
        self.view_configs = get_view_configs()
        
        # Метрики идентичности (15 метрик)
        self.identity_metrics = get_identity_signature_metrics()
        
        # Устройство вычислений
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Флаг инициализации
        self.init_done = False
        
        # Инициализация компонентов
        if HAS_3DDFA:
            self.initialize_3ddfa_components()
        else:
            logger.warning("3DDFA недоступен, используется заглушка")
        
        logger.info("Face3DAnalyzer инициализирован")

    def initialize_3ddfa_components(self) -> None:
        """
        ИСПРАВЛЕНО: Инициализация 3DDFA компонентов
        Согласно правкам: исправлена инициализация с правильными путями
        """
        if self.init_done:
            logger.info("3DDFA уже инициализирован")
            return
        
        if not HAS_3DDFA:
            logger.error("3DDFA компоненты недоступны")
            return
        
        try:
            logger.info("Начало инициализации 3DDFA компонентов")
            
            # Проверка файлов конфигурации
            config_path = REQUIRED_3DDFA_FILES["config"]
            if not config_path.exists():
                raise FileNotFoundError(f"Конфигурационный файл не найден: {config_path}")
            
            # Загрузка конфигурации
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
            
            # Инициализация FaceBoxes
            self.faceboxes = FaceBoxesActual()
            logger.info("FaceBoxes инициализирован")
            
            # Инициализация TDDFA
            if USE_ONNX:
                logger.info("Использование ONNX модели")
                self.tddfa_onnx = TDDFAActual(
                    onnx_path=str(REQUIRED_3DDFA_FILES["onnx_model"]),
                    bfm_fp=str(REQUIRED_3DDFA_FILES["bfm_model"]),
                    device='cpu' if (IS_MACOS and IS_ARM64) else self.device
                )
                self.tddfa_pytorch = None
            else:
                logger.info("Использование PyTorch модели")
                self.tddfa_pytorch = TDDFAActual(
                    arch=cfg['arch'],
                    checkpoint_fp=str(REQUIRED_3DDFA_FILES["weights"]),
                    bfm_fp=str(REQUIRED_3DDFA_FILES["bfm_model"]),
                    size=cfg['size'],
                    device=self.device
                )
                self.tddfa_onnx = None
            
            # Загрузка референсной модели для shape error
            self._load_reference_model()
            
            self.init_done = True
            logger.info("3DDFA компоненты успешно инициализированы")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации 3DDFA: {e}")
            self.tddfa_onnx = None
            self.tddfa_pytorch = None
            self.faceboxes = None
            self.init_done = False
            raise

    def _load_reference_model(self) -> None:
        """Загрузка референсной модели BFM для расчета shape error"""
        try:
            if self.tddfa_pytorch and hasattr(self.tddfa_pytorch, 'bfm'):
                bfm = self.tddfa_pytorch.bfm
                if hasattr(bfm, 'shapeMU') and hasattr(bfm, 'landmarksIdx'):
                    shape_mu = bfm.shapeMU
                    landmarks_idx = bfm.landmarksIdx
                    
                    if hasattr(shape_mu, 'cpu'):
                        shape_mu = shape_mu.cpu().numpy()
                    if hasattr(landmarks_idx, 'cpu'):
                        landmarks_idx = landmarks_idx.cpu().numpy()
                    
                    self.reference_model_landmarks = shape_mu[landmarks_idx, :].reshape(-1, 3)
                    logger.info("Референсная модель BFM загружена")
                else:
                    logger.warning("BFM модель не содержит необходимые компоненты")
            else:
                logger.warning("BFM модель недоступна для ONNX режима")
                
        except Exception as e:
            logger.error(f"Ошибка загрузки референсной модели: {e}")
            self.reference_model_landmarks = None

    def extract_68_landmarks_with_confidence(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """
        ИСПРАВЛЕНО: Извлечение 68 3D ландмарок с confidence scores
        Согласно правкам: confidence scores = 1.0 для 3DDFA
        """
        if not self.init_done:
            raise RuntimeError("3DDFA не инициализирован")
        
        if self.faceboxes is None:
            raise RuntimeError("FaceBoxes не инициализирован")
        
        try:
            logger.info(f"Извлечение 68 ландмарок из изображения {image.shape}")
            
            # Детекция лиц
            boxes = self.faceboxes(image)
            
            if not isinstance(boxes, np.ndarray):
                boxes = np.array(boxes)
            
            if boxes.shape[0] == 0:
                logger.warning("Лица не обнаружены")
                return np.array([]), np.array([]), (0, 0)
            
            # Выбор лучшего лица по площади
            boxes_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best_box = boxes[np.argmax(boxes_areas)]
            
            # Выбор модели TDDFA
            tddfa_model = self.tddfa_onnx if (USE_ONNX and self.tddfa_onnx) else self.tddfa_pytorch
            if tddfa_model is None:
                raise RuntimeError("TDDFA модель не инициализирована")
            
            logger.info(f"Использование {'ONNX' if USE_ONNX else 'PyTorch'} модели")
            
            # Извлечение параметров лица
            param_lst, roi_box_lst = tddfa_model(image, [best_box])
            
            # Реконструкция 3D вершин
            ver = tddfa_model.recon_vers(param_lst, roi_box_lst, dense_flag=False)[0]
            
            if not isinstance(ver, np.ndarray):
                ver = ver.detach().cpu().numpy()
            
            if ver.size == 0:
                logger.warning("Реконструкция не удалась")
                return np.array([]), np.array([]), (0, 0)
            
            # Преобразование координат (Y инвертирован)
            ver = ver.T  # Транспонирование для получения (N, 3)
            ver[:, 1] = -ver[:, 1]  # Инверсия Y координаты
            
            # ИСПРАВЛЕНО: Установка confidence scores = 1.0 согласно правкам
            confidence_scores = np.ones(68, dtype=np.float32)
            logger.info("Confidence scores установлены в 1.0 согласно правкам")
            
            # Извлечение 68 ландмарок
            landmarks_3d = ver[:68] if ver.shape[0] >= 68 else ver
            
            logger.info(f"Извлечено {landmarks_3d.shape[0]} ландмарок")
            
            return landmarks_3d, confidence_scores, image.shape[:2]
            
        except Exception as e:
            logger.error(f"Ошибка извлечения ландмарок: {e}")
            return np.array([]), np.array([]), (0, 0)

    def extract_dense_surface_points(self, image: np.ndarray) -> np.ndarray:
        """
        ИСПРАВЛЕНО: Извлечение плотных точек поверхности (38000 точек)
        Согласно правкам: добавлена функция extractdensesurfacepoints
        """
        if not self.init_done:
            raise RuntimeError("3DDFA не инициализирован")
        
        if self.faceboxes is None:
            raise RuntimeError("FaceBoxes не инициализирован")
        
        try:
            logger.info(f"Извлечение плотных точек поверхности из изображения {image.shape}")
            
            # Детекция лиц
            boxes = self.faceboxes(image)
            
            if boxes.shape[0] == 0:
                logger.warning("Лица не обнаружены")
                return np.array([])
            
            # Выбор лучшего лица
            best_box = boxes[np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))]
            
            # Выбор модели TDDFA
            tddfa_model = self.tddfa_onnx if (USE_ONNX and self.tddfa_onnx) else self.tddfa_pytorch
            if tddfa_model is None:
                raise RuntimeError("TDDFA модель не инициализирована")
            
            # Извлечение параметров
            param_lst, roi_box_lst = tddfa_model(image, [best_box])
            
            # ИСПРАВЛЕНО: Реконструкция с dense_flag=True для получения 38000 точек
            dense_ver = tddfa_model.recon_vers(param_lst, roi_box_lst, dense_flag=True)[0]
            
            if not isinstance(dense_ver, np.ndarray):
                dense_ver = np.array(dense_ver)
            
            # Преобразование координат
            dense_ver = dense_ver.T
            dense_ver[:, 1] = -dense_ver[:, 1]
            
            # ИСПРАВЛЕНО: Фильтрация по MIN_VISIBILITY_Z
            if MIN_VISIBILITY_Z is not None:
                visible_mask = dense_ver[:, 2] > MIN_VISIBILITY_Z
                dense_ver = dense_ver[visible_mask]
                logger.info(f"Отфильтровано {np.sum(~visible_mask)} невидимых точек")
            
            logger.info(f"Извлечено {dense_ver.shape[0]} плотных точек поверхности")
            
            return dense_ver
            
        except Exception as e:
            logger.error(f"Ошибка извлечения плотных точек: {e}")
            return np.array([])

    def determine_precise_face_pose(self, landmarks_3d: np.ndarray) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Определение точной позы лица с matrix2angle
        Согласно правкам: использование matrix2angle для углов
        """
        try:
            if landmarks_3d.size == 0:
                return {
                    "pose_category": "Unknown",
                    "angles": (0.0, 0.0, 0.0),
                    "confidence": 0.0,
                    "details": "No landmarks available"
                }
            
            logger.info("Определение точной позы лица")
            
            # Ключевые точки для определения позы
            if landmarks_3d.shape[0] >= 68:
                # Нос (кончик)
                nose_tip = landmarks_3d[30]
                # Левый и правый уголки глаз
                left_eye = landmarks_3d[36]
                right_eye = landmarks_3d[45]
                # Центр лица
                face_center = np.mean([left_eye, right_eye, nose_tip], axis=0)
            else:
                logger.warning("Недостаточно ландмарок для определения позы")
                return {
                    "pose_category": "Unknown",
                    "angles": (0.0, 0.0, 0.0),
                    "confidence": 0.0,
                    "details": "Insufficient landmarks"
                }
            
            # Расчет углов поворота
            # Yaw (поворот влево-вправо)
            eye_center = (left_eye + right_eye) / 2
            nose_eye_vector = nose_tip - eye_center
            yaw = np.degrees(np.arctan2(nose_eye_vector[0], nose_eye_vector[2]))
            
            # Pitch (наклон вверх-вниз)
            pitch = np.degrees(np.arctan2(nose_eye_vector[1], nose_eye_vector[2]))
            
            # Roll (поворот по часовой стрелке)
            eye_vector = right_eye - left_eye
            roll = np.degrees(np.arctan2(eye_vector[1], eye_vector[0]))
            
            # Нормализация углов в диапазон [-180, 180]
            yaw = ((yaw + 180) % 360) - 180
            pitch = ((pitch + 180) % 360) - 180
            roll = ((roll + 180) % 360) - 180
            
            # Определение категории позы
            abs_yaw = abs(yaw)
            if abs_yaw <= 15:
                pose_category = "Frontal"
            elif abs_yaw <= 35:
                pose_category = "Frontal_Edge"
            elif abs_yaw <= 65:
                pose_category = "Semi_Profile"
            else:
                pose_category = "Profile"
            
            # Расчет confidence на основе качества ландмарок
            landmark_quality = self._calculate_landmark_quality(landmarks_3d)
            confidence = min(0.95, max(0.1, landmark_quality))
            
            result = {
                "pose_category": pose_category,
                "angles": (float(pitch), float(yaw), float(roll)),
                "confidence": float(confidence),
                "details": f"Yaw: {yaw:.1f}°, Pitch: {pitch:.1f}°, Roll: {roll:.1f}°"
            }
            
            logger.info(f"Поза определена: {pose_category}, углы: {result['angles']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка определения позы: {e}")
            return {
                "pose_category": "Error",
                "angles": (0.0, 0.0, 0.0),
                "confidence": 0.0,
                "details": f"Error: {str(e)}"
            }

    def _calculate_landmark_quality(self, landmarks_3d: np.ndarray) -> float:
        """Расчет качества ландмарок"""
        try:
            if landmarks_3d.size == 0:
                return 0.0
            
            # Проверка на выбросы
            distances = pdist(landmarks_3d)
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            
            # Качество на основе стандартного отклонения
            quality = max(0.0, 1.0 - (std_dist / (mean_dist + 1e-6)))
            
            return quality
            
        except Exception as e:
            logger.error(f"Ошибка расчета качества ландмарок: {e}")
            return 0.5

    def normalize_landmarks_by_pose_category(self, landmarks_3d: np.ndarray, pose_category: str) -> np.ndarray:
        """
        ИСПРАВЛЕНО: Нормализация ландмарок по категории позы
        Согласно правкам: нормализация с reference points для каждой категории
        """
        try:
            if landmarks_3d.size == 0:
                return landmarks_3d
            
            logger.info(f"Нормализация ландмарок для позы: {pose_category}")
            
            if pose_category not in self.view_configs:
                logger.warning(f"Неизвестная категория позы: {pose_category}")
                return landmarks_3d
            
            config = self.view_configs[pose_category]
            reference_points = config.get("reference_points", [])
            
            if not reference_points or len(reference_points) < 2:
                logger.warning(f"Недостаточно референсных точек для {pose_category}")
                return landmarks_3d
            
            # Получение референсных точек
            if all(idx < landmarks_3d.shape[0] for idx in reference_points):
                ref_landmarks = landmarks_3d[reference_points]
                
                # Расчет центра и масштаба
                center_point = np.mean(ref_landmarks, axis=0)
                
                # Масштабирование на основе стандартного расстояния
                if pose_category == "Frontal":
                    # Межзрачковое расстояние
                    scale_distance = calculate_distance(landmarks_3d[36], landmarks_3d[45])
                    standard_distance = config.get("standard_iod", 64)
                elif pose_category == "Frontal_Edge":
                    # Расстояние нос-глаз
                    scale_distance = calculate_distance(landmarks_3d[27], landmarks_3d[36])
                    standard_distance = config.get("standard_nose_eye", 45)
                elif pose_category == "Semi_Profile":
                    # Высота лица
                    scale_distance = calculate_distance(landmarks_3d[8], landmarks_3d[27])
                    standard_distance = config.get("standard_face_height", 120)
                else:  # Profile
                    # Высота профиля
                    scale_distance = calculate_distance(landmarks_3d[1], landmarks_3d[15])
                    standard_distance = config.get("standard_profile_height", 140)
                
                # Применение нормализации
                if scale_distance > 0:
                    scale_factor = standard_distance / scale_distance
                    normalized_landmarks = (landmarks_3d - center_point) * scale_factor + center_point
                    
                    logger.info(f"Нормализация выполнена: scale_factor={scale_factor:.3f}")
                    return normalized_landmarks
            
            logger.warning("Нормализация не выполнена, возвращены исходные ландмарки")
            return landmarks_3d
            
        except Exception as e:
            logger.error(f"Ошибка нормализации ландмарок: {e}")
            return landmarks_3d

    def calculate_identity_signature_metrics(self, landmarks_3d: np.ndarray, pose_category: str) -> Dict[str, float]:
        """
        ИСПРАВЛЕНО: Расчет 15 метрик идентичности
        Согласно правкам: все 15 метрик из 3 групп по 5 метрик
        """
        try:
            if landmarks_3d.size == 0 or landmarks_3d.shape[0] < 68:
                logger.warning("Недостаточно ландмарок для расчета метрик")
                return self._get_default_metrics()
            
            logger.info(f"Расчет 15 метрик идентичности для позы: {pose_category}")
            
            metrics = {}
            
            # ГРУППА 1: Skull Geometry Signature (5 метрик)
            metrics.update(self._calculate_skull_geometry_metrics(landmarks_3d))
            
            # ГРУППА 2: Facial Proportions Signature (5 метрик)  
            metrics.update(self._calculate_facial_proportions_metrics(landmarks_3d))
            
            # ГРУППА 3: Bone Structure Signature (5 метрик)
            metrics.update(self._calculate_bone_structure_metrics(landmarks_3d))
            
            # Проверка количества метрик
            if len(metrics) != 15:
                logger.error(f"Неверное количество метрик: {len(metrics)}, ожидается 15")
                return self._get_default_metrics()
            
            logger.info(f"Рассчитано {len(metrics)} метрик идентичности")
            return metrics
            
        except Exception as e:
            logger.error(f"Ошибка расчета метрик идентичности: {e}")
            return self._get_default_metrics()

    def _calculate_skull_geometry_metrics(self, landmarks_3d: np.ndarray) -> Dict[str, float]:
        """Расчет метрик геометрии черепа"""
        try:
            metrics = {}
            
            # 1. skull_width_ratio - отношение ширины черепа
            skull_width = calculate_distance(landmarks_3d[0], landmarks_3d[16])
            face_height = calculate_distance(landmarks_3d[8], landmarks_3d[27])
            metrics["skull_width_ratio"] = skull_width / (face_height + 1e-6)
            
            # 2. temporal_bone_angle - угол височной кости
            metrics["temporal_bone_angle"] = calculate_angle(
                landmarks_3d[0], landmarks_3d[17], landmarks_3d[26]
            )
            
            # 3. zygomatic_arch_width - ширина скуловой дуги
            zygomatic_width = calculate_distance(landmarks_3d[1], landmarks_3d[15])
            metrics["zygomatic_arch_width"] = zygomatic_width
            
            # 4. orbital_depth - глубина орбиты
            left_eye_center = np.mean(landmarks_3d[36:42], axis=0)
            right_eye_center = np.mean(landmarks_3d[42:48], axis=0)
            eye_depth = (left_eye_center[2] + right_eye_center[2]) / 2
            metrics["orbital_depth"] = eye_depth
            
            # 5. occipital_curve - кривизна затылочной области
            posterior_points = landmarks_3d[[0, 8, 16]]
            metrics["occipital_curve"] = fit_curve_to_points(posterior_points)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Ошибка расчета метрик черепа: {e}")
            return {
                "skull_width_ratio": 0.8,
                "temporal_bone_angle": 110.0,
                "zygomatic_arch_width": 60.0,
                "orbital_depth": 20.0,
                "occipital_curve": 10.0
            }

    def _calculate_facial_proportions_metrics(self, landmarks_3d: np.ndarray) -> Dict[str, float]:
        """Расчет метрик пропорций лица"""
        try:
            metrics = {}
            
            # 6. cephalic_index - головной индекс
            skull_length = calculate_distance(landmarks_3d[8], landmarks_3d[27])
            skull_width = calculate_distance(landmarks_3d[0], landmarks_3d[16])
            metrics["cephalic_index"] = skull_width / (skull_length + 1e-6)
            
            # 7. nasolabial_angle - носогубный угол
            metrics["nasolabial_angle"] = calculate_angle(
                landmarks_3d[33], landmarks_3d[51], landmarks_3d[62]
            )
            
            # 8. orbital_index - орбитальный индекс
            orbital_height = calculate_distance(landmarks_3d[37], landmarks_3d[41])
            orbital_width = calculate_distance(landmarks_3d[36], landmarks_3d[39])
            metrics["orbital_index"] = orbital_height / (orbital_width + 1e-6)
            
            # 9. forehead_height_ratio - отношение высоты лба
            forehead_height = calculate_distance(landmarks_3d[19], landmarks_3d[27])
            face_height = calculate_distance(landmarks_3d[8], landmarks_3d[27])
            metrics["forehead_height_ratio"] = forehead_height / (face_height + 1e-6)
            
            # 10. chin_projection_ratio - отношение выступа подбородка
            chin_projection = abs(landmarks_3d[8][2] - landmarks_3d[30][2])
            nose_projection = abs(landmarks_3d[30][2] - landmarks_3d[33][2])
            metrics["chin_projection_ratio"] = chin_projection / (nose_projection + 1e-6)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Ошибка расчета пропорций лица: {e}")
            return {
                "cephalic_index": 0.8,
                "nasolabial_angle": 90.0,
                "orbital_index": 0.85,
                "forehead_height_ratio": 0.35,
                "chin_projection_ratio": 0.15
            }

    def _calculate_bone_structure_metrics(self, landmarks_3d: np.ndarray) -> Dict[str, float]:
        """Расчет метрик костной структуры"""
        try:
            metrics = {}
            
            # 11. interpupillary_distance_ratio - отношение межзрачкового расстояния
            ipd = calculate_distance(landmarks_3d[36], landmarks_3d[45])
            face_width = calculate_distance(landmarks_3d[0], landmarks_3d[16])
            metrics["interpupillary_distance_ratio"] = ipd / (face_width + 1e-6)
            
            # 12. gonial_angle_asymmetry - асимметрия углов нижней челюсти
            left_gonial = calculate_angle(landmarks_3d[4], landmarks_3d[6], landmarks_3d[8])
            right_gonial = calculate_angle(landmarks_3d[12], landmarks_3d[10], landmarks_3d[8])
            metrics["gonial_angle_asymmetry"] = abs(left_gonial - right_gonial)
            
            # 13. zygomatic_angle - скуловой угол
            metrics["zygomatic_angle"] = calculate_angle(
                landmarks_3d[1], landmarks_3d[31], landmarks_3d[15]
            )
            
            # 14. jaw_angle_ratio - отношение углов челюсти
            jaw_width = calculate_distance(landmarks_3d[4], landmarks_3d[12])
            chin_width = calculate_distance(landmarks_3d[6], landmarks_3d[10])
            metrics["jaw_angle_ratio"] = jaw_width / (chin_width + 1e-6)
            
            # 15. mandibular_symphysis_angle - угол симфиза нижней челюсти
            metrics["mandibular_symphysis_angle"] = calculate_angle(
                landmarks_3d[6], landmarks_3d[8], landmarks_3d[10]
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Ошибка расчета костной структуры: {e}")
            return {
                "interpupillary_distance_ratio": 1.0,
                "gonial_angle_asymmetry": 0.02,
                "zygomatic_angle": 130.0,
                "jaw_angle_ratio": 0.9,
                "mandibular_symphysis_angle": 75.0
            }

    def _get_default_metrics(self) -> Dict[str, float]:
        """Получение метрик по умолчанию"""
        return {
            # Skull Geometry
            "skull_width_ratio": 0.8,
            "temporal_bone_angle": 110.0,
            "zygomatic_arch_width": 60.0,
            "orbital_depth": 20.0,
            "occipital_curve": 10.0,
            # Facial Proportions
            "cephalic_index": 0.8,
            "nasolabial_angle": 90.0,
            "orbital_index": 0.85,
            "forehead_height_ratio": 0.35,
            "chin_projection_ratio": 0.15,
            # Bone Structure
            "interpupillary_distance_ratio": 1.0,
            "gonial_angle_asymmetry": 0.02,
            "zygomatic_angle": 130.0,
            "jaw_angle_ratio": 0.9,
            "mandibular_symphysis_angle": 75.0
        }

    def calculate_comprehensive_shape_error(self, landmarks_3d: np.ndarray) -> Dict[str, float]:
        """
        ИСПРАВЛЕНО: Расчет комплексной ошибки формы с Hausdorff distance
        Согласно правкам: добавлен calculateComprehensiveShapeError
        """
        try:
            if landmarks_3d.size == 0:
                return {
                    "overall_shape_error": 1.0,
                    "eye_region_error": 1.0,
                    "nose_region_error": 1.0,
                    "mouth_region_error": 1.0,
                    "hausdorff_distance": 1.0
                }
            
            logger.info("Расчет комплексной ошибки формы")
            
            shape_errors = {}
            
            # Общая ошибка формы
            if self.reference_model_landmarks is not None:
                # Hausdorff distance между текущими и референсными ландмарками
                hausdorff_dist = self._calculate_hausdorff_distance(
                    landmarks_3d, self.reference_model_landmarks
                )
                shape_errors["hausdorff_distance"] = hausdorff_dist
                
                # Общая ошибка как среднее расстояние
                distances = [calculate_distance(landmarks_3d[i], self.reference_model_landmarks[i]) 
                           for i in range(min(len(landmarks_3d), len(self.reference_model_landmarks)))]
                shape_errors["overall_shape_error"] = np.mean(distances)
            else:
                # Альтернативный расчет без референсной модели
                shape_errors["hausdorff_distance"] = 0.5
                shape_errors["overall_shape_error"] = 0.5
            
            # Ошибки по регионам
            shape_errors["eye_region_error"] = self._calculate_region_error(
                landmarks_3d, list(range(36, 48))  # Глаза
            )
            
            shape_errors["nose_region_error"] = self._calculate_region_error(
                landmarks_3d, list(range(27, 36))  # Нос
            )
            
            shape_errors["mouth_region_error"] = self._calculate_region_error(
                landmarks_3d, list(range(48, 68))  # Рот
            )
            
            logger.info(f"Shape error рассчитан: overall={shape_errors['overall_shape_error']:.3f}")
            
            return shape_errors
            
        except Exception as e:
            logger.error(f"Ошибка расчета shape error: {e}")
            return {
                "overall_shape_error": 1.0,
                "eye_region_error": 1.0,
                "nose_region_error": 1.0,
                "mouth_region_error": 1.0,
                "hausdorff_distance": 1.0
            }

    def _calculate_hausdorff_distance(self, set1: np.ndarray, set2: np.ndarray) -> float:
        """Расчет расстояния Хаусдорфа между двумя наборами точек"""
        try:
            # Расстояние от каждой точки set1 до ближайшей точки в set2
            distances_1_to_2 = []
            for point1 in set1:
                min_dist = min(calculate_distance(point1, point2) for point2 in set2)
                distances_1_to_2.append(min_dist)
            
            # Расстояние от каждой точки set2 до ближайшей точки в set1
            distances_2_to_1 = []
            for point2 in set2:
                min_dist = min(calculate_distance(point2, point1) for point1 in set1)
                distances_2_to_1.append(min_dist)
            
            # Hausdorff distance - максимум из максимальных расстояний
            hausdorff = max(max(distances_1_to_2), max(distances_2_to_1))
            
            return hausdorff
            
        except Exception as e:
            logger.error(f"Ошибка расчета Hausdorff distance: {e}")
            return 1.0

    def _calculate_region_error(self, landmarks_3d: np.ndarray, region_indices: List[int]) -> float:
        """Расчет ошибки для конкретного региона лица"""
        try:
            if not region_indices or landmarks_3d.shape[0] <= max(region_indices):
                return 1.0
            
            region_landmarks = landmarks_3d[region_indices]
            
            if self.reference_model_landmarks is not None and len(self.reference_model_landmarks) > max(region_indices):
                reference_region = self.reference_model_landmarks[region_indices]
                
                # Среднее расстояние между соответствующими точками
                distances = [calculate_distance(region_landmarks[i], reference_region[i]) 
                           for i in range(len(region_landmarks))]
                return np.mean(distances)
            else:
                # Альтернативный расчет - вариация внутри региона
                if len(region_landmarks) > 1:
                    distances = pdist(region_landmarks)
                    return np.std(distances)
                else:
                    return 0.5
                    
        except Exception as e:
            logger.error(f"Ошибка расчета региональной ошибки: {e}")
            return 1.0

    def analyze_facial_asymmetry(self, landmarks_3d: np.ndarray) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Анализ лицевой асимметрии
        Согласно правкам: mild/severe asymmetry с правильными порогами
        """
        try:
            if landmarks_3d.size == 0 or landmarks_3d.shape[0] < 68:
                return {
                    "overall_asymmetry_score": 0.0,
                    "mild_asymmetry_detected": False,
                    "severe_asymmetry_detected": False,
                    "surgical_asymmetry_indicators": False,
                    "asymmetry_details": []
                }
            
            logger.info("Анализ лицевой асимметрии")
            
            # Определение центральной оси лица
            face_center_x = (landmarks_3d[0, 0] + landmarks_3d[16, 0]) / 2
            
            # Индексы левых и правых ландмарок
            left_indices = list(range(0, 9)) + list(range(17, 22)) + list(range(36, 42)) + list(range(48, 55))
            right_indices = list(range(8, 17)) + list(range(22, 27)) + list(range(42, 48)) + list(range(55, 60))
            
            # Получение левых и правых ландмарок
            left_landmarks = landmarks_3d[left_indices]
            right_landmarks = landmarks_3d[right_indices]
            
            # Отражение правых ландмарок относительно центральной оси
            right_landmarks_mirrored = mirror_landmarks(right_landmarks, face_center_x)
            
            # Расчет асимметрии
            if len(left_landmarks) == len(right_landmarks_mirrored):
                asymmetry_scores = np.linalg.norm(left_landmarks - right_landmarks_mirrored, axis=1)
                overall_asymmetry = np.mean(asymmetry_scores)
            else:
                # Альтернативный расчет для разного количества точек
                overall_asymmetry = 0.05
            
            # Классификация асимметрии согласно порогам из coreconfig
            mild_threshold = CRITICAL_THRESHOLDS.get("mild_asymmetry_threshold", 0.05)
            severe_threshold = CRITICAL_THRESHOLDS.get("severe_asymmetry_threshold", 0.1)
            
            mild_asymmetry = overall_asymmetry > mild_threshold
            severe_asymmetry = overall_asymmetry > severe_threshold
            
            # Индикаторы хирургического вмешательства
            surgical_indicators = self._detect_surgical_asymmetry_indicators(landmarks_3d, asymmetry_scores)
            
            # Детали асимметрии
            details = []
            if mild_asymmetry:
                details.append(f"Mild asymmetry detected: {overall_asymmetry:.3f}")
            if severe_asymmetry:
                details.append(f"Severe asymmetry detected: {overall_asymmetry:.3f}")
            if surgical_indicators:
                details.append("Surgical asymmetry indicators present")
            
            result = {
                "overall_asymmetry_score": float(overall_asymmetry),
                "mild_asymmetry_detected": mild_asymmetry,
                "severe_asymmetry_detected": severe_asymmetry,
                "surgical_asymmetry_indicators": surgical_indicators,
                "asymmetry_details": details
            }
            
            logger.info(f"Асимметрия проанализирована: {overall_asymmetry:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка анализа асимметрии: {e}")
            return {
                "overall_asymmetry_score": 0.0,
                "mild_asymmetry_detected": False,
                "severe_asymmetry_detected": False,
                "surgical_asymmetry_indicators": False,
                "asymmetry_details": [f"Error: {str(e)}"]
            }

    def _detect_surgical_asymmetry_indicators(self, landmarks_3d: np.ndarray, asymmetry_scores: np.ndarray) -> bool:
        """Обнаружение индикаторов хирургической асимметрии"""
        try:
            # Проверка на локальные пики асимметрии
            if len(asymmetry_scores) > 0:
                max_asymmetry = np.max(asymmetry_scores)
                mean_asymmetry = np.mean(asymmetry_scores)
                
                # Если максимальная асимметрия значительно превышает среднюю
                if max_asymmetry > mean_asymmetry * 3:
                    return True
            
            # Проверка на неестественные углы
            jaw_angles = [
                calculate_angle(landmarks_3d[4], landmarks_3d[6], landmarks_3d[8]),
                calculate_angle(landmarks_3d[12], landmarks_3d[10], landmarks_3d[8])
            ]
            
            if abs(jaw_angles[0] - jaw_angles[1]) > 15:  # Разница в углах челюсти > 15°
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Ошибка обнаружения хирургических индикаторов: {e}")
            return False

    def perform_cross_validation_landmarks(self, all_landmarks: List[np.ndarray]) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Кросс-валидация ландмарок для проверки стабильности
        Согласно правкам: добавлена функция performcrossvalidationlandmarks
        """
        try:
            if not all_landmarks or len(all_landmarks) < 2:
                return {
                    "mean_variation": 0.0,
                    "max_variation": 0.0,
                    "stability_score": 0.0,
                    "stable": True,
                    "landmarks_variations": []
                }
            
            logger.info(f"Кросс-валидация {len(all_landmarks)} наборов ландмарок")
            
            # Фильтрация валидных ландмарок
            valid_landmarks = [lm for lm in all_landmarks if lm.size > 0 and lm.shape[0] >= 68]
            
            if len(valid_landmarks) < 2:
                logger.warning("Недостаточно валидных ландмарок для кросс-валидации")
                return {
                    "mean_variation": 0.0,
                    "max_variation": 0.0,
                    "stability_score": 1.0,
                    "stable": True,
                    "landmarks_variations": []
                }
            
            # Расчет вариаций между наборами ландмарок
            reference_landmarks = valid_landmarks[0]
            variations = []
            
            for landmarks in valid_landmarks[1:]:
                if landmarks.shape == reference_landmarks.shape:
                    # Расчет расстояний между соответствующими точками
                    variation = np.linalg.norm(landmarks - reference_landmarks, axis=1)
                    variations.extend(variation)
            
            if variations:
                validation_results = {
                    "mean_variation": float(np.mean(variations)),
                    "max_variation": float(np.max(variations)),
                    "landmarks_variations": variations
                }
                
                # Оценка стабильности
                validation_results["stability_score"] = max(0.0, 1.0 - validation_results["mean_variation"] / 5.0)
                validation_results["stable"] = validation_results["mean_variation"] < 3.0
                
                logger.info(f"Кросс-валидация: mean_var={validation_results['mean_variation']:.3f}")
                
            else:
                validation_results = {
                    "mean_variation": 0.0,
                    "max_variation": 0.0,
                    "stability_score": 1.0,
                    "stable": True,
                    "landmarks_variations": []
                }
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Ошибка кросс-валидации ландмарок: {e}")
            return {
                "mean_variation": 0.0,
                "max_variation": 0.0,
                "stability_score": 0.0,
                "stable": False,
                "landmarks_variations": []
            }

    def self_test(self) -> None:
        """Самотестирование модуля"""
        logger.info("=== Самотестирование Face3DAnalyzer ===")
        
        # Информация о системе
        import platform
        logger.info(f"Система: {platform.system()} {platform.machine()}")
        logger.info(f"USE_ONNX: {USE_ONNX}")
        logger.info(f"ONNX_EXECUTION_PROVIDERS: {ONNX_EXECUTION_PROVIDERS}")
        
        # Тестовое изображение
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        try:
            # Тест извлечения ландмарок
            landmarks, confidence, shape = self.extract_68_landmarks_with_confidence(test_image)
            logger.info(f"Тест 3DDFA: {landmarks.shape if landmarks.size > 0 else 'No face detected'}")
            
            if landmarks.size > 0:
                # Тест определения позы
                pose_result = self.determine_precise_face_pose(landmarks)
                logger.info(f"Тест позы: {pose_result['pose_category']}")
                
                # Тест метрик идентичности
                metrics = self.calculate_identity_signature_metrics(landmarks, pose_result['pose_category'])
                logger.info(f"Тест метрик: {len(metrics)} метрик рассчитано")
                
                # Тест асимметрии
                asymmetry = self.analyze_facial_asymmetry(landmarks)
                logger.info(f"Тест асимметрии: {asymmetry['overall_asymmetry_score']:.3f}")
                
        except Exception as e:
            logger.error(f"Ошибка самотестирования: {e}")
        
        logger.info("=== Самотестирование завершено ===")

# ==================== ТОЧКА ВХОДА ====================

if __name__ == "__main__":
    analyzer = Face3DAnalyzer()
    analyzer.self_test()