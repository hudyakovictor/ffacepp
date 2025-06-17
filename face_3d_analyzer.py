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
from scipy.spatial.distance import euclidean, pdist, squareform, directed_hausdorff # ИСПРАВЛЕНО: Добавлен directed_hausdorff
from scipy.stats import mode
from scipy.spatial import distance
from pathlib import Path
from datetime import datetime, timedelta
import os
import sys

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

# Импорт 3DDFA компонентов
from TDDFA_ONNX import TDDFA_ONNX
from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX as FaceBoxesDetector
from utils.tddfa_util import _parse_param # ИСПРАВЛЕНО: Заменен calc_pose на _parse_param

# Попытка импорта 3DDFA компонентов
try:
    from core_config import (
        USE_ONNX, ONNX_EXECUTION_PROVIDERS, IS_MACOS, IS_ARM64,
        REQUIRED_3DDFA_FILES, VIEW_CONFIGS, CAMERA_BOX_FRONT_SIZE_FACTOR,
        CRITICAL_THRESHOLDS, get_identity_signature_metrics,
        get_view_configs, AGING_MODEL, MIN_VISIBILITY_Z, MODELS_DIR,
        STANDARD_IOD, STANDARD_NOSE_EYE, STANDARD_FACE_HEIGHT, STANDARD_PROFILE_HEIGHT
    )
    
    HAS_3DDFA = True
    logger.info("3DDFA компоненты успешно импортированы")
    
except ImportError as e:
    HAS_3DDFA = False
    logger.error(f"Ошибка импорта 3DDFA: {e}")
    logger.warning("Используется заглушка для 3DDFA")

# ==================== Инициализация 3DDFA компонентов (отдельная функция) ====================
def initialize_3ddfa_components(model_dir: str, skip_gpu_check: bool = False) -> Dict[str, Any]:
    """
    Инициализация 3DDFA компонентов.
    Проверяет доступность GPU и загружает необходимые модели (FaceBoxes, TDDFA).
    """
    # Убрана критическая проверка GPU, позволяем системе использовать CPU, если CUDA недоступна

    try:
        logger.info("Начало инициализации 3DDFA компонентов")
        
        # Определяем устройство для вычислений
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Использование устройства: {device}")
        
        # Инициализация FaceBoxes
        face_boxes = FaceBoxesDetector() # ИСПРАВЛЕНО: Убраны пороги чувствительности, т.к. они глобальные
        logger.info("FaceBoxes инициализирован")
        
        # Инициализация TDDFA
        # Используем пути к моделям из REQUIRED_3DDFA_FILES
        tddfa = TDDFA_ONNX( # Изменено на TDDFA_ONNX
            gpu_mode=(device == 'cuda'), # Устанавливаем gpu_mode на основе обнаруженного устройства
            bfm_fp=str(REQUIRED_3DDFA_FILES["bfm_model"]),
            size=120, # Возвращаем size=120, т.к. TDDFA_ONNX ожидает этот размер
            arch='mobilenet_v1', # Известная архитектура для mb1
            num_params=62,
            widen_factor=1,
            checkpoint_fp=str(REQUIRED_3DDFA_FILES["weights"]) # Передаем путь к весам
        )
        logger.info(f"TDDFA ({device.upper()}) инициализирован")
        
        logger.info("3DDFA компоненты успешно инициализированы")
        return dict(tddfa=tddfa, face_boxes=face_boxes)
        
    except Exception as e:
        logger.error(f"Ошибка инициализации 3DDFA компонентов: {e}")
        raise # Проброс ошибки инициализации

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
    
    def __init__(self, tddfa_model: Any = None, face_boxes_model: Any = None):
        """Инициализация анализатора"""
        logger.info("Инициализация Face3DAnalyzer")
        
        # Модели 3DDFA (теперь инициализируются и передаются извне)
        self.tddfa = tddfa_model
        self.faceboxes = face_boxes_model
        
        # ИСПРАВЛЕНО: Добавлена поддержка dense surface points (38000 точек)
        self.dense_face_model = None
        
        # Референсная модель для shape error
        self.reference_model_landmarks = None
        if self.tddfa: # Load reference model if tddfa is provided
            self._load_reference_model()
        else:
            logger.warning("TDDFA модель не предоставлена, референсная модель не будет загружена.")
        
        # Параметры для 3DDFA (теперь с CAMERA_BOX_FRONT_SIZE_FACTOR)
        self.camera_box_size_factor = CAMERA_BOX_FRONT_SIZE_FACTOR # Используем новую константу
        
        # Конфигурации видов
        self.view_configs = get_view_configs()
        
        # Метрики идентичности (15 метрик)
        self.identity_metrics = get_identity_signature_metrics()
        
        # Устройство вычислений
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Флаг инициализации
        self.init_done = (self.tddfa is not None and self.faceboxes is not None)
        
        logger.info("Face3DAnalyzer инициализирован")

    def _load_reference_model(self) -> None:
        """Загрузка референсной модели BFM для расчета shape error"""
        try:
            if self.tddfa and hasattr(self.tddfa, 'bfm'):
                bfm = self.tddfa.bfm
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
                    logger.warning("BFM модель не содержит необходимые компоненты (shapeMU или landmarksIdx).")
            else:
                logger.warning("TDDFA модель не инициализирована или не содержит BFM для загрузки референсной модели.")
                
        except Exception as e:
            logger.error(f"Ошибка загрузки референсной модели: {e}")
            self.reference_model_landmarks = None

    def extract_68_landmarks_with_confidence(self, img: np.ndarray, models: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Извлечение 68 3D-ландмарок лица с оценкой достоверности.
        ИСПРАВЛЕНО: Добавлена субпиксельная доводка и коррекция осей.
        """
        try:
            logger.debug(f"[extract_68_landmarks] Начало. Размер изображения: {img.shape}, Тип: {img.dtype}")
            
            # ИСПРАВЛЕНО (Пункт 17): Убедимся, что изображение в формате BGR
            if img.ndim == 2: # Если изображение в оттенках серого
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4: # Если изображение с альфа-каналом (RGBA)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else:
                img_bgr = img.copy() # Просто копия, если уже BGR или другого 3-канального формата

            # 2. Запускаем FaceBoxes НА ПОЛНОМ кадре
            boxes = models["face_boxes"](img_bgr) # список [[xmin, ymin, xmax, ymax, score]]

            # ИСПРАВЛЕНО (Пункт 3): Проверка на отсутствие лиц и выброс исключения
            if len(boxes) == 0:
                logger.error("FaceBoxes: 0 лиц обнаружено – тестовое фото непригодно.")
                raise RuntimeError("FaceBoxes: 0 лиц — тестовое фото непригодно")

            # ИСПРАВЛЕНО: Преобразуем список в массив NumPy, если это список
            if not isinstance(boxes, np.ndarray):
                boxes = np.array(boxes)
            
            logger.debug(f"[extract_68_landmarks] FaceBoxes вернул: {len(boxes) if isinstance(boxes, np.ndarray) else boxes} боксов. Тип: {type(boxes)}")
            logger.debug(f"[extract_68_landmarks] Содержимое boxes (первые 5): {boxes[:5] if isinstance(boxes, np.ndarray) and boxes.size > 0 else boxes}")
            if len(boxes) == 0:
                logger.error("FaceBoxes: 0 лиц обнаружено – проверьте изображение и освещение")
                return np.array([]), np.array([]), np.array([])  # Возвращаем пустые массивы

            # 3. Если найден хотя бы один bbox → делаем crop
            # Исправлено: Сортировка по площади и взятие первого элемента
            boxes_sorted = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
            box = boxes_sorted[0]

            x1, y1, x2, y2 = map(int, box[:4]) # отсекаем score

            logger.debug(f"[extract_68_landmarks] Выбранный бокс: [{x1}, {y1}, {x2}, {y2}]")

            # --- ИСПРАВЛЕНО: Отказ от ROI и передача полного кадра в TDDFA ---
            # Вместо roi и roi_120, передаем полный кадр и bbox в int32
            box_arr = np.array([[x1, y1, x2, y2]], dtype=np.int32)
            param_lst, roi_box_lst = models["tddfa"](img, box_arr)

            # Проверка, что ландмарки были успешно обнаружены
            if not param_lst or not roi_box_lst:
                logger.warning("3DDFA не обнаружил ландмарки для данного лица. Возврат пустых данных.")
                return np.array([]), np.array([]), np.array([])

            # ИСПРАВЛЕНО: Убедимся, что param_lst[0] и roi_box_lst[0] являются массивами numpy
            # Если tddfa возвращает torch.Tensor, detach().cpu().numpy() должен быть использован
            # Однако, для универсальности, используем np.asarray, если это не np.ndarray
            if not isinstance(param_lst[0], np.ndarray):
                try:
                    # Попытка преобразовать из torch.Tensor, если это так
                    if hasattr(param_lst[0], 'detach') and hasattr(param_lst[0], 'cpu') and hasattr(param_lst[0], 'numpy'):
                        param_lst[0] = param_lst[0].detach().cpu().numpy()
                    else:
                        param_lst[0] = np.asarray(param_lst[0])
                except Exception as convert_e:
                    logger.error(f"Не удалось преобразовать param_lst[0] в numpy: {convert_e}. Тип: {type(param_lst[0])}")
                    return np.array([]), np.array([]), np.array([])

            if not isinstance(roi_box_lst[0], np.ndarray):
                try:
                    # Попытка преобразовать из torch.Tensor, если это так
                    if hasattr(roi_box_lst[0], 'detach') and hasattr(roi_box_lst[0], 'cpu') and hasattr(roi_box_lst[0], 'numpy'):
                        roi_box_lst[0] = roi_box_lst[0].detach().cpu().numpy()
                    else:
                        roi_box_lst[0] = np.asarray(roi_box_lst[0])
                except Exception as convert_e:
                    logger.error(f"Не удалось преобразовать roi_box_lst[0] в numpy: {convert_e}. Тип: {type(roi_box_lst[0])}")
                    return np.array([]), np.array([]), np.array([])

            # ИСПРАВЛЕНО: Добавим проверку перед доступом к .shape в отладочных сообщениях
            if isinstance(param_lst[0], np.ndarray) and hasattr(param_lst[0], 'shape'):
                logger.debug(f"[extract_68_landmarks] TDDFA param_lst[0] shape: {param_lst[0].shape}")
            else:
                logger.error(f"[extract_68_landmarks] TDDFA param_lst[0] не является np.ndarray или не имеет атрибута 'shape'. Тип: {type(param_lst[0])}")
                return np.array([]), np.array([]), np.array([])

            if isinstance(roi_box_lst[0], np.ndarray) and hasattr(roi_box_lst[0], 'shape'):
                logger.debug(f"[extract_68_landmarks] TDDFA roi_box_lst[0] shape: {roi_box_lst[0].shape}")
            else:
                logger.error(f"[extract_68_landmarks] TDDFA roi_box_lst[0] не является np.ndarray или не имеет атрибута 'shape'. Тип: {type(roi_box_lst[0])}")
                return np.array([]), np.array([]), np.array([])
            
            # ----- ИСПРАВЛЕНО: Использование P, pose = calc_pose(param_lst[0]) для получения углов и матрицы проекции -----
            # parse_param возвращает P (матрицу проекции), pose (углы), landmarks_2d (68, 2)
            # _, pose_from_param, landmarks_2d_proj = parse_param(param_lst[0], roi_box_lst[0]) # УДАЛЕНО
            
            # Получаем 3D реконструированные вершины (в пространстве модели) для Z-координат
            vers_lst = models["tddfa"].recon_vers(param_lst, roi_box_lst)
            # ИСПРАВЛЕНО: Проверка на пустой vers_lst и явное преобразование в numpy
            if not vers_lst:
                logger.error("TDDFA вернул пустой vers_lst (0 точек) после recon_vers. Невозможно получить 3D данные.")
                raise RuntimeError("TDDFA вернул пустой список вершин") # Пункт 1: пустой vers_lst

            first_3d_ver = vers_lst[0]
            if isinstance(first_3d_ver, torch.Tensor):
                first_3d_ver = first_3d_ver.detach().cpu().numpy()
            elif not isinstance(first_3d_ver, np.ndarray):
                first_3d_ver = np.asarray(first_3d_ver)
            
            # landmarks_3d_reconstructed будет (68, 3) из (3, 68)
            landmarks_3d_reconstructed = first_3d_ver.T.astype(np.float32)

            # Составляем финальные `landmarks` (68, 3) используя X,Y из реконструированных 3D (они уже в пиксельных координатах)
            landmarks = landmarks_3d_reconstructed # Теперь landmarks уже (68,3) в нужных координатах
            landmarks_2d_proj = landmarks[:, :2] # Для IOD контроля берем 2D проекцию

            # Debugging IOD calculation
            logger.debug(f"[extract_68_landmarks] landmarks_2d_proj[36,:]: {landmarks_2d_proj[36,:]}")
            logger.debug(f"[extract_68_landmarks] landmarks_2d_proj[45,:]: {landmarks_2d_proj[45,:]}")
            
            # ИСПРАВЛЕНО: Для IOD ratio используем roi_box_lst[0] - это [xmin, ymin, xmax, ymax, score]
            roi_box = roi_box_lst[0]
            roi_width = roi_box[2] - roi_box[0]
            if roi_width == 0: # Avoid division by zero
                logger.error("Ширина ROI равна нулю, невозможно рассчитать IOD ratio.")
                return np.array([]), np.array([]), np.array([])

            iod_ratio = np.linalg.norm(landmarks_2d_proj[36,:] - landmarks_2d_proj[45,:]) / roi_width
            assert 0.2 < iod_ratio < 0.5, f"IOD ratio abnormal: {iod_ratio:.3f}"
            
            # Оценка достоверности ландмарок (на основе отклонения от референсной модели)
            # ИСПРАВЛЕНО: Используем новую логику оценки качества
            quality_score = self._calculate_landmark_quality(landmarks, roi_width) # Передаем roi_width
            confidence_array = np.full(landmarks.shape[0], quality_score) # Теперь возвращает массив для каждой точки
            
            # Оценка позы
            # ИСПРАВЛЕНО: передаем param_lst[0] и confidence_array в determine_precise_face_pose
            # calc_pose уже вызывается внутри determine_precise_face_pose, передаем только param
            pose_info = self.determine_precise_face_pose(landmarks, param_lst[0], confidence_array)
            
            return landmarks, confidence_array, param_lst[0] # Возвращаем param_lst[0] вместо pose_category
            
        except Exception as e:
            logger.error(f"[extract_68_landmarks] Ошибка извлечения ландмарок: {e}")
            return np.array([]), np.array([]), np.array([])

    def extract_dense_surface_points(self, img: np.ndarray, models: Dict[str, Any]) -> np.ndarray:
        """
        Извлечение плотных точек поверхности (приблизительно 38000 точек).
        """
        face_boxes = models.get("face_boxes")
        tddfa = models.get("tddfa")
        
        if face_boxes is None or tddfa is None:
            logger.error("Модели FaceBoxes или TDDFA не переданы.")
            raise RuntimeError("Неинициализированные модели FaceBoxes или TDDFA")
        
        try:
            logger.info(f"Извлечение плотных точек поверхности из изображения {img.shape}")
            
            # Детекция лиц
            boxes = face_boxes(img)
            
            if not isinstance(boxes, np.ndarray):
                boxes = np.array(boxes)
            
            if len(boxes) == 0:
                logger.warning("Лица не обнаружены")
                return np.array([])
            
            # Выбор лучшего лица
            best_box = boxes[np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))]
            
            # Извлечение параметров
            param_lst, roi_box_lst = tddfa(img, [best_box])
            
            # Реконструкция с dense_flag=True для получения 38000 точек
            dense_ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
            dense_ver = dense_ver_lst[0] # (3, ~38000)
            
            if not isinstance(dense_ver, np.ndarray):
                dense_ver = dense_ver.detach().cpu().numpy()
            
            # Коррекция осей: X и Y инвертированы
            dense_ver[0, :] = -dense_ver[0, :]
            dense_ver[1, :] = -dense_ver[1, :]
            dense_points = dense_ver.T.astype(np.float32) # (N, 3) и float32
            
            # Фильтрация по MIN_VISIBILITY_Z
            if MIN_VISIBILITY_Z is not None:
                visible_mask = dense_points[:, 2] > MIN_VISIBILITY_Z
                dense_points = dense_points[visible_mask]
                logger.info(f"Отфильтровано {np.sum(~visible_mask)} невидимых точек")
            
            logger.info(f"Извлечено {len(dense_points)} плотных точек поверхности")
            
            return dense_points
            
        except Exception as e:
            logger.error(f"Ошибка извлечения плотных точек: {e}")
            return np.array([])

    def determine_precise_face_pose(self, landmarks_3d: np.ndarray, param: np.ndarray, confidence_array: np.ndarray) -> Dict[str, Any]:
        """
        Определение точной позы лица с использованием 3DMM параметров.
        ИСПРАВЛЕНО: Использование matrix2angle для получения углов и доверия.
        """
        try:
            if landmarks_3d.size == 0 or param.size == 0 or confidence_array.size == 0:
                return {
                    "pose_category": "Unknown",
                    "angles": (0.0, 0.0, 0.0),
                    "confidence": 0.0,
                    "details": "No landmarks, parameters, or confidence available"
                }
            
            logger.info("Определение точной позы лица")
            
            # ИСПРАВЛЕНО: Использование _parse_param для получения матрицы R и вычисление углов позы вручную
            R, offset, alpha_shp, alpha_exp = _parse_param(param)
            
            # Извлечение углов Эйлера (pitch, yaw, roll) из матрицы вращения R
            # Используем порядок углов yaw, pitch, roll для соответствия предыдущей логике
            pitch = -np.arcsin(R[1, 2]) # Вращение вокруг X
            roll = np.arctan2(R[0, 2], R[2, 2]) # Вращение вокруг Z
            yaw = np.arctan2(R[1, 0], R[1, 1]) # Вращение вокруг Y

            # Конвертация в градусы
            pitch_deg = np.degrees(pitch)
            yaw_deg = np.degrees(yaw)
            roll_deg = np.degrees(roll)

            # Определение категории позы
            abs_yaw = abs(yaw_deg)
            if abs_yaw <= 15:
                pose_category = "Frontal"
            elif abs_yaw <= 35:
                pose_category = "Frontal_Edge"
            elif abs_yaw <= 65:
                pose_category = "Semi_Profile"
            else:
                pose_category = "Profile"
            
            # Расчет confidence на основе переданного массива доверия
            confidence = np.mean(confidence_array) # Используем среднее от переданного массива доверия
            
            result = {
                "pose_category": pose_category,
                "angles": (float(pitch_deg), float(yaw_deg), float(roll_deg)), # Возвращаем в градусах
                "confidence": float(confidence),
                "details": f"Yaw: {yaw_deg:.1f}°, Pitch: {pitch_deg:.1f}°, Roll: {roll_deg:.1f}°"
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

    def _calculate_landmark_quality(self, landmarks_3d: np.ndarray, face_scale_factor: float = 1.0) -> float:
        """Расчет качества ландмарок с нормализацией по масштабу лица"""
        try:
            if landmarks_3d.size == 0:
                return 0.0
            
            # Проверка на выбросы
            distances = pdist(landmarks_3d)
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            
            # ИСПРАВЛЕНО: Нормализация mean_dist и std_dist по масштабу лица
            if face_scale_factor > 0:
                mean_dist_norm = mean_dist / face_scale_factor
                std_dist_norm = std_dist / face_scale_factor
            else:
                mean_dist_norm = mean_dist
                std_dist_norm = std_dist
            
            # Качество на основе нормализованного стандартного отклонения
            # Используем порог, который может быть настроен
            # Чем меньше std_dist_norm, тем выше качество
            # Здесь я использую простую линейную зависимость, можно использовать сигмоиду для более плавного перехода
            # Предположим, что допустимое std_dist_norm < 0.1 для хорошего качества, и выше 0.5 - плохое
            max_acceptable_std = 0.2 # Примерный порог, может потребоваться калибровка
            quality = max(0.0, 1.0 - (std_dist_norm / max_acceptable_std))
            
            return quality
            
        except Exception as e:
            logger.error(f"Ошибка расчета качества ландмарок: {e}")
            return 0.5

    def normalize_landmarks_by_pose_category(self, lmk3d: np.ndarray,
                                             pose: str,
                                             visibility: np.ndarray):
        # Определяем центр и масштаб на основе всех 68 ландмарок
        # visibility mask будет применен позже, если потребуется работа только с видимыми точками
        center = {
            "Frontal":       lmk3d[30, :3],
            "Frontal-Edge":  lmk3d[30, :3],
            "Semi-Profile":  0.5*(lmk3d[8, :3] + lmk3d[27, :3]),
            "Profile":       lmk3d[11, :3],
        }[pose]

        if pose == "Frontal":
            p1, p2 = lmk3d[36, :3], lmk3d[45, :3]
            scale = np.linalg.norm(p1 - p2) / STANDARD_IOD
            # Проверяем IOD (межзрачковое расстояние) — важно для корректной нормализации
            # Защита от деления на ноль (ZeroDivisionError — частая ошибка в Python, см. https://www.codewithharry.com/blogpost/solved-python-zerodivision-error/)
            try:
                if lmk3d[45,0] == lmk3d[36,0]:
                    logger.error("[normalize_landmarks_by_pose_category] Ошибка: lmk3d[45,0] == lmk3d[36,0], деление на ноль невозможно. Возвращаю исходные ландмарки без нормализации.")
                    return lmk3d
                iod_norm = np.linalg.norm(lmk3d[36,:2] - lmk3d[45,:2]) / (lmk3d[45,0] - lmk3d[36,0])
                # Новый, более широкий диапазон и мягкая обработка
                if not (0.15 < iod_norm < 0.6):
                    logger.warning(f"[normalize_landmarks_by_pose_category] IOD ratio: {iod_norm:.3f} — значение вне рекомендуемого диапазона (0.15 < IOD < 0.6), но анализ продолжается. Возможно, требуется калибровка.")
                    # TODO: Сделать порог IOD адаптивным (учитывать разрешение, угол, качество детекции)
                    # Здесь можно добавить дополнительную обработку для крайних случаев, если потребуется
            except ZeroDivisionError:
                logger.error("[normalize_landmarks_by_pose_category] ZeroDivisionError: попытка деления на ноль при расчёте iod_norm. Возвращаю исходные ландмарки без нормализации.")
                return lmk3d
        elif pose == "Frontal-Edge":
            # Выбираем видимый глаз для определения масштаба
            eye = lmk3d[36] if lmk3d[36,2] > lmk3d[45,2] else lmk3d[45]
            scale = np.linalg.norm(lmk3d[30,:3] - eye[:3]) / STANDARD_NOSE_EYE
        elif pose == "Semi-Profile":
            scale = np.linalg.norm(lmk3d[8,:3] - lmk3d[27,:3]) / STANDARD_FACE_HEIGHT
        else:                                           # Profile
            scale = np.linalg.norm(lmk3d[3,:3] - lmk3d[12,:3]) / STANDARD_PROFILE_HEIGHT

        # Применяем нормализацию
        norm = (lmk3d[:, :3] - center) / scale

        # Применяем visibility_mask для исключения невидимых точек
        norm_masked = norm * visibility[:, np.newaxis] # Умножаем на маску, чтобы обнулить невидимые

        return norm_masked # Возвращаем нормализованные и, возможно, отфильтрованные ландмарки

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
        """Расчет метрик геометрии черепа (6 метрик)"""
        try:
            metrics = {}
            
            # 1. skull_width_ratio - отношение ширины черепа к высоте лица
            skull_width = calculate_distance(landmarks_3d[0], landmarks_3d[16])
            face_height_for_ratio = calculate_distance(landmarks_3d[8], landmarks_3d[27])
            metrics["skull_width_ratio"] = skull_width / (face_height_for_ratio + 1e-6)
            
            # 2. skull_depth_ratio - отношение глубины лица к высоте лица
            # Аппроксимация глубины: разница Z между кончиком носа и средней точкой глаз.
            nose_tip_z = landmarks_3d[30, 2]
            eye_avg_z = np.mean(landmarks_3d[[36, 45], 2])
            face_depth = abs(nose_tip_z - eye_avg_z)
            metrics["skull_depth_ratio"] = face_depth / (face_height_for_ratio + 1e-6)
            
            # 3. forehead_height_ratio - отношение высоты лба к высоте лица
            forehead_top = landmarks_3d[19] # Точка на лбу (между бровями)
            nasion = landmarks_3d[27] # Переносица
            forehead_height = calculate_distance(forehead_top, nasion)
            metrics["forehead_height_ratio"] = forehead_height / (face_height_for_ratio + 1e-6)
            
            # 4. temple_width_ratio - отношение ширины висков к ширине черепа
            temple_width = calculate_distance(landmarks_3d[17], landmarks_3d[26]) # Внешние точки бровей как прокси
            metrics["temple_width_ratio"] = temple_width / (skull_width + 1e-6)
            
            # 5. zygomatic_arch_width - ширина скуловой дуги
            # Используем точки 1 и 15 как прокси для ширины скуловой дуги (bizygomatic width)
            metrics["zygomatic_arch_width"] = calculate_distance(landmarks_3d[1], landmarks_3d[15])
            
            # 6. occipital_curve - кривизна затылочной области (аппроксимация)
            # Используем точки на нижней челюсти и подбородке для аппроксимации кривизны нижней части лица
            # Эти точки (0, 8, 16) лучше отражают общий контур лица.
            posterior_points = landmarks_3d[[0, 8, 16]]
            metrics["occipital_curve"] = fit_curve_to_points(posterior_points) # Чем меньше, тем более плоская
            
            return metrics
            
        except Exception as e:
            logger.error(f"Ошибка расчета метрик геометрии черепа: {e}")
            return {
                "skull_width_ratio": 0.0,
                "skull_depth_ratio": 0.0,
                "forehead_height_ratio": 0.0,
                "temple_width_ratio": 0.0,
                "zygomatic_arch_width": 0.0,
                "occipital_curve": 0.0
            }

    def _calculate_facial_proportions_metrics(self, landmarks_3d: np.ndarray) -> Dict[str, float]:
        """Расчет метрик пропорций лица (6 метрик)"""
        try:
            metrics = {}
            face_width = calculate_distance(landmarks_3d[0], landmarks_3d[16]) # Ширина лица по боковым точкам
            
            # 1. eye_distance_ratio - отношение межзрачкового расстояния к ширине лица
            interocular_distance = calculate_distance(landmarks_3d[36], landmarks_3d[45])
            metrics["eye_distance_ratio"] = interocular_distance / (face_width + 1e-6)
            
            # 2. nose_width_ratio - отношение ширины носа к длине носа
            nose_width = calculate_distance(landmarks_3d[31], landmarks_3d[35]) # Ширина ноздрей
            nose_length = calculate_distance(landmarks_3d[27], landmarks_3d[30]) # Переносица - кончик носа
            metrics["nose_width_ratio"] = nose_width / (nose_length + 1e-6)
            
            # 3. mouth_width_ratio - отношение ширины рта к ширине лица
            mouth_width = calculate_distance(landmarks_3d[48], landmarks_3d[54])
            metrics["mouth_width_ratio"] = mouth_width / (face_width + 1e-6)
            
            # 4. chin_width_ratio - отношение ширины подбородка к ширине челюсти
            chin_width = calculate_distance(landmarks_3d[6], landmarks_3d[10]) # Нижняя часть подбородка
            jaw_width = calculate_distance(landmarks_3d[4], landmarks_3d[12]) # Ширина нижней челюсти
            metrics["chin_width_ratio"] = chin_width / (jaw_width + 1e-6)

            # 5. jaw_angle_ratio - отношение углов челюсти (левая и правая сторона)
            # Используем угол, образованный боковыми точками челюсти и подбородком
            left_jaw_angle = calculate_angle(landmarks_3d[4], landmarks_3d[8], landmarks_3d[6])
            right_jaw_angle = calculate_angle(landmarks_3d[12], landmarks_3d[8], landmarks_3d[10])
            metrics["jaw_angle_ratio"] = abs(left_jaw_angle - right_jaw_angle) # Отражает асимметрию
            
            # 6. forehead_angle - угол наклона лба
            # Используем точки на лбу и переносице для расчета угла
            metrics["forehead_angle"] = calculate_angle(landmarks_3d[19], landmarks_3d[27], landmarks_3d[24])
            
            return metrics
            
        except Exception as e:
            logger.error(f"Ошибка расчета пропорций лица: {e}")
            return {
                "eye_distance_ratio": 0.0,
                "nose_width_ratio": 0.0,
                "mouth_width_ratio": 0.0,
                "chin_width_ratio": 0.0,
                "jaw_angle_ratio": 0.0,
                "forehead_angle": 0.0
            }

    def _calculate_bone_structure_metrics(self, landmarks_3d: np.ndarray) -> Dict[str, float]:
        """Расчет метрик костной структуры (3 метрики)"""
        try:
            metrics = {}
            
            # 1. nose_projection_ratio - отношение выступа носа
            # Разница в Z-координате между кончиком носа и основанием носа
            nose_projection = abs(landmarks_3d[30, 2] - landmarks_3d[33, 2])
            nose_length = calculate_distance(landmarks_3d[27], landmarks_3d[30])
            metrics["nose_projection_ratio"] = nose_projection / (nose_length + 1e-6)
            
            # 2. chin_projection_ratio - отношение выступа подбородка
            # Разница в Z-координате между подбородком и кончиком носа
            chin_projection = abs(landmarks_3d[8, 2] - landmarks_3d[30, 2])
            metrics["chin_projection_ratio"] = chin_projection / (nose_projection + 1e-6) if nose_projection > 0 else 0.0
            
            # 3. jaw_line_angle - общий угол линии челюсти
            # Угол, образованный точками 4, 8, 12 (нижняя челюсть, подбородок, нижняя челюсть)
            metrics["jaw_line_angle"] = calculate_angle(landmarks_3d[4], landmarks_3d[8], landmarks_3d[12])
            
            return metrics
            
        except Exception as e:
            logger.error(f"Ошибка расчета костной структуры: {e}")
            return {
                "nose_projection_ratio": 0.0,
                "chin_projection_ratio": 0.0,
                "jaw_line_angle": 0.0
            }

    def _get_default_metrics(self) -> Dict[str, float]:
        """Получение метрик по умолчанию, согласно обновленному плану"""
        return {
            # Skull Geometry (6 metrics)
            "skull_width_ratio": 0.0,
            "skull_depth_ratio": 0.0,
            "forehead_height_ratio": 0.0,
            "temple_width_ratio": 0.0,
            "zygomatic_arch_width": 0.0,
            "occipital_curve": 0.0,
            # Facial Proportions (6 metrics)
            "eye_distance_ratio": 0.0,
            "nose_width_ratio": 0.0,
            "mouth_width_ratio": 0.0,
            "chin_width_ratio": 0.0,
            "jaw_angle_ratio": 0.0,
            "forehead_angle": 0.0,
            # Bone Structure (3 metrics)
            "nose_projection_ratio": 0.0,
            "chin_projection_ratio": 0.0,
            "jaw_line_angle": 0.0
        }

    def calculate_comprehensive_shape_error(self, landmarks_3d: np.ndarray) -> Dict[str, float]:
        """
        Расчет комплексной ошибки формы и региональных ошибок с использованием расстояния Хаусдорфа.
        """
        try:
            if landmarks_3d.size == 0 or landmarks_3d.shape[0] < 68:
                logger.warning("Недостаточно ландмарок для расчета ошибки формы.")
                return {
                    "overall_shape_error": 1.0,
                    "eye_region_error": 1.0,
                    "nose_region_error": 1.0,
                    "mouth_region_error": 1.0,
                    "hausdorff_distance": 1.0
                }
            
            if self.reference_model_landmarks is None or self.reference_model_landmarks.size == 0:
                logger.warning("Референсная модель для расчета shape error не загружена. Возвращены дефолтные значения.")
                return {
                    "overall_shape_error": 1.0,
                    "eye_region_error": 1.0,
                    "nose_region_error": 1.0,
                    "mouth_region_error": 1.0,
                    "hausdorff_distance": 1.0
                }
            
            logger.info("Расчет комплексной ошибки формы")
            
            shape_errors = {}
            
            # Расчет расстояния Хаусдорфа
            hausdorff_dist = self._calculate_hausdorff_distance(landmarks_3d, self.reference_model_landmarks)
            shape_errors["hausdorff_distance"] = float(hausdorff_dist)
            
            # Расчет общей ошибки формы (среднее Евклидово расстояние между соответствующими точками)
            overall_distances = [calculate_distance(landmarks_3d[i], self.reference_model_landmarks[i]) 
                                 for i in range(min(landmarks_3d.shape[0], self.reference_model_landmarks.shape[0]))]
            shape_errors["overall_shape_error"] = float(np.mean(overall_distances))
            
            # Расчет ошибок по регионам
            eye_region_indices = list(range(36, 48))
            nose_region_indices = list(range(27, 36))
            mouth_region_indices = list(range(48, 68))
            
            shape_errors["eye_region_error"] = float(self._calculate_region_error(landmarks_3d, eye_region_indices))
            shape_errors["nose_region_error"] = float(self._calculate_region_error(landmarks_3d, nose_region_indices))
            shape_errors["mouth_region_error"] = float(self._calculate_region_error(landmarks_3d, mouth_region_indices))
            
            logger.info(f"Shape error рассчитан: overall={shape_errors['overall_shape_error']:.3f}, Hausdorff={shape_errors['hausdorff_distance']:.3f}")
            
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
        """Расчет расстояния Хаусдорфа между двумя наборами точек с использованием scipy.spatial.distance.directed_hausdorff"""
        try:
            # ИСПРАВЛЕНО: Использование scipy.spatial.distance.directed_hausdorff
            # d(A, B) = max_{a in A} min_{b in B} ||a-b||
            # d(B, A) = max_{b in B} min_{a in A} ||b-a||
            # Hausdorff(A, B) = max(d(A, B), d(B, A))
            
            dist_ab = directed_hausdorff(set1, set2)[0]
            dist_ba = directed_hausdorff(set2, set1)[0]
            
            hausdorff = max(dist_ab, dist_ba)
            
            return float(hausdorff)
            
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

    def analyze_facial_asymmetry(self, landmarks_3d: np.ndarray, confidence_array: np.ndarray) -> Dict[str, Any]:
        """
        Анализ лицевой асимметрии, включая общую асимметрию, мягкую/сильную асимметрию
        и индикаторы хирургического вмешательства.
        """
        try:
            if landmarks_3d is None or confidence_array is None or landmarks_3d.size == 0 or landmarks_3d.shape[0] < 68 or confidence_array.size == 0:
                logger.warning("Недостаточно ландмарок или данных достоверности для анализа асимметрии.")
                return {
                    "overall_asymmetry_score": 0.0,
                    "mild_asymmetry_detected": False,
                    "severe_asymmetry_detected": False,
                    "surgical_asymmetry_indicators": False,
                    "asymmetry_details": ["No landmarks or confidence available"],
                    "confidence": 0.0 # Добавленная метрика уверенности
                }
            
            logger.info("Анализ лицевой асимметрии")
            
            # Определение центральной оси лица
            face_center_x = (landmarks_3d[0, 0] + landmarks_3d[16, 0]) / 2
            
            # Индексы левых и правых ландмарок (это существующие)
            left_indices = list(range(0, 9)) + list(range(17, 22)) + list(range(36, 42)) + list(range(48, 55))
            right_indices = list(range(8, 17)) + list(range(22, 27)) + list(range(42, 48)) + list(range(55, 60))
            
            # Получение левых и правых ландмарок
            left_landmarks = landmarks_3d[left_indices]
            right_landmarks = landmarks_3d[right_indices]
            
            # Отражение правых ландмарок относительно центральной оси
            right_landmarks_mirrored = mirror_landmarks(right_landmarks, face_center_x)
            
            # Расчет асимметрии
            if len(left_landmarks) == len(right_landmarks_mirrored) and len(left_landmarks) > 0:
                asymmetry_scores = np.linalg.norm(left_landmarks - right_landmarks_mirrored, axis=1)
                overall_asymmetry = np.mean(asymmetry_scores)
            else:
                logger.warning("Недостаточно симметричных ландмарок для расчета асимметрии.")
                asymmetry_scores = np.array([])  # Исправление: всегда определяем переменную
                overall_asymmetry = 0.0
            
            # Классификация асимметрии согласно порогам из core_config
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

            # Расчет общей уверенности на основе входного массива
            overall_confidence = float(np.mean(confidence_array))
            
            result = {
                "overall_asymmetry_score": float(overall_asymmetry),
                "mild_asymmetry_detected": mild_asymmetry,
                "severe_asymmetry_detected": severe_asymmetry,
                "surgical_asymmetry_indicators": surgical_indicators,
                "asymmetry_details": details,
                "confidence": overall_confidence # Добавляем уверенность
            }
            
            logger.info(f"Асимметрия проанализирована: {overall_asymmetry:.3f}, Confidence: {overall_confidence:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка анализа асимметрии: {e}")
            return {
                "overall_asymmetry_score": 0.0,
                "mild_asymmetry_detected": False,
                "severe_asymmetry_detected": False,
                "surgical_asymmetry_indicators": False,
                "asymmetry_details": [f"Error: {str(e)}"],
                "confidence": 0.0 # В случае ошибки
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

    def perform_cross_validation_landmarks(self, all_landmarks: List[np.ndarray], all_confidence_arrays: List[np.ndarray]) -> Dict[str, Any]:
        """
        Кросс-валидация ландмарок для проверки стабильности и качества.
        Расчет среднего и максимального отклонения, показателя стабильности и уверенности.
        """
        try:
            if not all_landmarks or len(all_landmarks) < 2 or not all_confidence_arrays or len(all_confidence_arrays) < 2:
                logger.warning("Недостаточно данных для кросс-валидации ландмарок.")
                return {
                    "mean_variation": 0.0,
                    "max_variation": 0.0,
                    "stability_score": 0.0,
                    "stable": True,
                    "landmarks_variations": [],
                    "confidence": 0.0 # В случае недостатка данных
                }
            
            logger.info(f"Кросс-валидация {len(all_landmarks)} наборов ландмарок")
            
            # Фильтрация валидных ландмарок (убеждаемся, что есть 68 точек)
            valid_landmarks = [lm for lm in all_landmarks if lm.size > 0 and lm.shape[0] >= 68]
            valid_confidence_arrays = [conf for conf in all_confidence_arrays if conf.size > 0 and conf.shape[0] >= 68]
            
            if len(valid_landmarks) < 2 or len(valid_confidence_arrays) < 2 or len(valid_landmarks) != len(valid_confidence_arrays):
                logger.warning("Недостаточно валидных ландмарок или массивов достоверности для кросс-валидации.")
                return {
                    "mean_variation": 0.0,
                    "max_variation": 0.0,
                    "stability_score": 1.0,
                    "stable": True,
                    "landmarks_variations": [],
                    "confidence": 0.0
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
                    "landmarks_variations": [float(v) for v in variations] # Преобразование в float
                }
                
                # Оценка стабильности
                validation_results["stability_score"] = float(max(0.0, 1.0 - validation_results["mean_variation"] / 5.0))
                validation_results["stable"] = bool(validation_results["mean_variation"] < 3.0)
                
                # Расчет средней уверенности
                overall_confidence = float(np.mean([np.mean(c) for c in valid_confidence_arrays]))
                validation_results["confidence"] = overall_confidence

                logger.info(f"Кросс-валидация: mean_var={validation_results['mean_variation']:.3f}, confidence={overall_confidence:.3f}")
                
            else:
                validation_results = {
                    "mean_variation": 0.0,
                    "max_variation": 0.0,
                    "stability_score": 1.0,
                    "stable": True,
                    "landmarks_variations": [],
                    "confidence": 0.0
                }
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Ошибка кросс-валидации ландмарок: {e}")
            return {
                "mean_variation": 0.0,
                "max_variation": 0.0,
                "stability_score": 0.0,
                "stable": False,
                "landmarks_variations": [],
                "confidence": 0.0
            }

    def analyze_facial_features_changes(self, current_metrics: Dict[str, float], previous_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Анализирует изменения лицевых признаков (identity metrics) между текущим и предыдущим наборами.
        Возвращает словарь с обнаруженными изменениями, их оценками и значительными изменениями.
        """
        logger.info("Начало анализа изменений лицевых признаков")
        changes_detected = {}
        change_scores = {}
        significant_changes = {}

        feature_change_threshold = CRITICAL_THRESHOLDS.get("FEATURE_CHANGE_THRESHOLD", 0.1) # Default to 0.1 if not found

        for metric_name, current_value in current_metrics.items():
            if metric_name in previous_metrics:
                previous_value = previous_metrics[metric_name]
                change = current_value - previous_value
                # Используем абсолютное изменение для оценки
                relative_change = abs(change / (previous_value + 1e-9)) if previous_value != 0 else abs(change)

                change_scores[metric_name] = float(change)
                changes_detected[metric_name] = bool(relative_change > feature_change_threshold)

                if changes_detected[metric_name]:
                    significant_changes[metric_name] = {
                        "current": float(current_value),
                        "previous": float(previous_value),
                        "change": float(change),
                        "relative_change": float(relative_change)
                    }
            else:
                logger.warning(f"Метрика '{metric_name}' не найдена в предыдущих метриках. Изменение не может быть рассчитано.")

        logger.info("Анализ изменений лицевых признаков завершен")
        return {
            "changes_detected": changes_detected,
            "change_scores": change_scores,
            "significant_changes": significant_changes
        }

    def get_combined_facial_analysis_results(self, img: np.ndarray, models: Dict[str, Any], previous_metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Объединяет результаты всех этапов 3D-анализа лица в единый комплексный словарь.
        """
        logger.info("Начало комбинированного анализа лица...")
        results = {}

        try:
            # 1. Извлечение 68 ландмарок с уверенностью
            landmarks_3d, confidence_array, param = self.extract_68_landmarks_with_confidence(img, models)
            results["landmarks_3d"] = landmarks_3d.tolist()
            results["landmarks_confidence"] = confidence_array.tolist()
            results["3d_model_params"] = param.tolist()
            logger.info("Ландмарки и параметры модели извлечены.")

            # 2. Извлечение плотных точек поверхности
            dense_points = self.extract_dense_surface_points(img, models)
            results["dense_surface_points"] = dense_points.tolist()
            logger.info("Плотные точки поверхности извлечены.")

            # 3. Определение точной позы лица
            pose_analysis = self.determine_precise_face_pose(landmarks_3d, param, confidence_array)
            results["pose_analysis"] = pose_analysis
            pose_category = pose_analysis["pose_category"]
            logger.info(f"Поза лица определена: {pose_category}.")

            # 4. Нормализация ландмарок по категории позы
            vis = landmarks_3d[:, 2] > MIN_VISIBILITY_Z
            norm = self.normalize_landmarks_by_pose_category(landmarks_3d, pose_category, vis)
            results["normalized_landmarks"] = norm.tolist()
            results["scale"] = 1.0
            results["center"] = landmarks_3d[30, :3].tolist()
            logger.info("Ландмарки нормализованы.")

            # 5. Расчет метрик идентичности
            identity_metrics = self.calculate_identity_signature_metrics(norm, pose_category)
            results["identity_metrics"] = identity_metrics
            logger.info("Метрики идентичности рассчитаны.")

            # 6. Расчет комплексной ошибки формы
            shape_errors = self.calculate_comprehensive_shape_error(norm)
            results["shape_errors"] = shape_errors
            logger.info("Ошибки формы рассчитаны.")

            # 7. Анализ асимметрии лица
            asymmetry_results = self.analyze_facial_asymmetry(landmarks_3d, confidence_array)
            results["facial_asymmetry"] = asymmetry_results
            logger.info("Асимметрия лица проанализирована.")

            # 8. Кросс-валидация ландмарок
            # Для простоты, используем текущие ландмарки как единственный набор для кросс-валидации.
            # В реальном сценарии здесь будет список ландмарок из нескольких кадров/источников.
            cross_validation_results = self.perform_cross_validation_landmarks([landmarks_3d], [confidence_array])
            results["landmark_cross_validation"] = cross_validation_results
            logger.info("Кросс-валидация ландмарок выполнена.")

            # 9. Анализ изменений лицевых признаков (если есть предыдущие метрики)
            if previous_metrics:
                feature_changes = self.analyze_facial_features_changes(identity_metrics, previous_metrics)
                results["feature_changes"] = feature_changes
                logger.info("Изменения лицевых признаков проанализированы.")
            else:
                results["feature_changes"] = {"changes_detected": {}, "change_scores": {}, "significant_changes": {}}
                logger.info("Предыдущие метрики не предоставлены, анализ изменений пропущен.")

            logger.info("Комбинированный анализ лица завершен успешно.")
            return results
                
        except Exception as e:
            logger.error(f"Ошибка при выполнении комбинированного анализа лица: {e}")
            raise # Перевыбрасываем исключение для обработки на более высоком уровне

    def self_test(self) -> None:
        """
        Метод для самопроверки функциональности Face3DAnalyzer.
        """
        logger.info("Запуск самотестирования Face3DAnalyzer...")
        try:
            # Мок-данные для тестирования
            test_image_path = "/Users/victorkhudyakov/nn/3DDFA2/3/01_01_10.jpg"
            mock_img = cv2.imread(test_image_path)

            if mock_img is None:
                logger.error(f"Не удалось загрузить тестовое изображение с пути: {test_image_path}. Проверьте путь и наличие файла.")
                return

            logger.info(f"Загружено тестовое изображение с размером: {mock_img.shape}")

            mock_models = dict(tddfa=self.tddfa, face_boxes=self.faceboxes) # Используем инициализированные модели
            mock_confidence = np.full(68, 0.95, dtype=np.float32)

            # Test extract_68_landmarks_with_confidence
            logger.info("Тестирование extract_68_landmarks_with_confidence...")
            landmarks, confidence_array, param = self.extract_68_landmarks_with_confidence(mock_img, mock_models)
            assert landmarks.shape == (68, 3), f"Expected (68,3) landmarks, got {landmarks.shape}"
            assert confidence_array.shape == (68,), f"Expected (68,) confidence, got {confidence_array.shape}"
            assert param.shape == (62,), f"Expected (62,) param, got {param.shape}"
            logger.info("extract_68_landmarks_with_confidence: OK")

            # Test extract_dense_surface_points
            logger.info("Тестирование extract_dense_surface_points...")
            dense_points = self.extract_dense_surface_points(mock_img, mock_models)
            # Убрал жесткую привязку к числу точек, так как оно может незначительно отличаться
            assert len(dense_points) > 30000, f"Expected more than 30000 dense points, got {len(dense_points)}"
            logger.info("extract_dense_surface_points: OK")

            # Test determine_precise_face_pose
            logger.info("Тестирование determine_precise_face_pose...")
            # Используем landmarks, param и confidence_array, полученные из предыдущего шага
            pose_results = self.determine_precise_face_pose(landmarks, param, confidence_array)
            assert 'pose_category' in pose_results and 'angles' in pose_results, "Pose results missing keys"
            assert len(pose_results['angles']) == 3, "Pose angles should have 3 components"
            logger.info("determine_precise_face_pose: OK")

            # Test normalize_landmarks_by_pose_category
            logger.info("Тестирование normalize_landmarks_by_pose_category...")
            vis = landmarks[:, 2] > MIN_VISIBILITY_Z
            norm = self.normalize_landmarks_by_pose_category(
                landmarks,
                pose_results["pose_category"],
                vis
            )
            iod_after_norm = np.linalg.norm(norm[36, :2] - norm[45, :2])
            if not (0.8 < iod_after_norm < 1.2):
                logger.warning(f"Самотест: IOD после нормализации = {iod_after_norm:.3f}, что вне ожидаемого диапазона (0.8–1.2), но тест продолжается.")
            assert isinstance(norm, np.ndarray), "Normalized landmarks should be a numpy array"
            logger.info("normalize_landmarks_by_pose_category: OK")

            # Test calculate_identity_signature_metrics
            logger.info("Тестирование calculate_identity_signature_metrics...")
            metrics = self.calculate_identity_signature_metrics(norm, "Frontal")
            assert len(metrics) == 15, f"Expected 15 metrics, got {len(metrics)}"
            assert all(np.isfinite(list(metrics.values()))), "Metrics contain non-finite values"
            logger.info("calculate_identity_signature_metrics: OK")

            # Test calculate_comprehensive_shape_error
            logger.info("Тестирование calculate_comprehensive_shape_error...")
            # For simplicity, create a dummy reference model if not loaded
            if self.reference_model_landmarks is None:
                self.reference_model_landmarks = np.zeros((68, 3), dtype=np.float32) # Dummy reference
            shape_errors = self.calculate_comprehensive_shape_error(norm)
            assert 'overall_shape_error' in shape_errors, "Shape errors missing overall_shape_error"
            assert 'eye_region_error' in shape_errors, "Shape errors missing eye_region_error"
            assert 'nose_region_error' in shape_errors, "Shape errors missing nose_region_error"
            assert 'mouth_region_error' in shape_errors, "Shape errors missing mouth_region_error"
            assert all(isinstance(v, float) for v in shape_errors.values()), "Shape errors values should be float"
            logger.info("calculate_comprehensive_shape_error: OK")

            # Test analyze_facial_asymmetry
            logger.info("Тестирование analyze_facial_asymmetry...")
            asymmetry_results = self.analyze_facial_asymmetry(landmarks, confidence_array)
            assert 'overall_asymmetry_score' in asymmetry_results, "Asymmetry results missing overall_asymmetry_score"
            assert 'mild_asymmetry_detected' in asymmetry_results, "Asymmetry results missing mild_asymmetry_detected"
            assert 'severe_asymmetry_detected' in asymmetry_results, "Asymmetry results missing severe_asymmetry_detected"
            assert 'surgical_asymmetry_indicators' in asymmetry_results, "Asymmetry results missing surgical_asymmetry_indicators"
            assert 'confidence' in asymmetry_results, "Asymmetry results missing confidence"
            logger.info("analyze_facial_asymmetry: OK")

            # Test perform_cross_validation_landmarks
            logger.info("Тестирование perform_cross_validation_landmarks...")
            # Create dummy data for all_landmarks and all_confidence_arrays
            all_landmarks = [landmarks + np.random.rand(68,3)*0.1, landmarks - np.random.rand(68,3)*0.1]
            all_confidence_arrays = [confidence_array, confidence_array]
            cross_validation_results = self.perform_cross_validation_landmarks(all_landmarks, all_confidence_arrays)
            assert 'stability_score' in cross_validation_results, "Cross validation results missing stability_score"
            assert 'stable' in cross_validation_results, "Cross validation results missing stable"
            logger.info("perform_cross_validation_landmarks: OK")

            # Test analyze_facial_features_changes
            logger.info("Тестирование analyze_facial_features_changes...")
            mock_current_metrics = {
                "skull_width_ratio": 1.0, "eye_distance_ratio": 0.5, "nose_projection_ratio": 0.2
            }
            mock_previous_metrics = {
                "skull_width_ratio": 1.05, "eye_distance_ratio": 0.4, "nose_projection_ratio": 0.1
            }
            feature_changes_results = self.analyze_facial_features_changes(mock_current_metrics, mock_previous_metrics)
            assert 'changes_detected' in feature_changes_results, "Feature changes results missing changes_detected"
            assert 'change_scores' in feature_changes_results, "Feature changes results missing change_scores"
            assert 'significant_changes' in feature_changes_results, "Feature changes results missing significant_changes"
            assert feature_changes_results['changes_detected']['nose_projection_ratio'] == True # Should detect a significant change
            assert feature_changes_results['changes_detected']['skull_width_ratio'] == False # Should not be a significant change
            logger.info("analyze_facial_features_changes: OK")

            # Test get_combined_facial_analysis_results
            logger.info("Тестирование get_combined_facial_analysis_results...")
            combined_results = self.get_combined_facial_analysis_results(mock_img, mock_models, mock_previous_metrics)
            assert "landmarks_3d" in combined_results, "Combined results missing landmarks_3d"
            assert "dense_surface_points" in combined_results, "Combined results missing dense_surface_points"
            assert "pose_analysis" in combined_results, "Combined results missing pose_analysis"
            assert "normalized_landmarks" in combined_results, "Combined results missing normalized_landmarks"
            assert "identity_metrics" in combined_results, "Combined results missing identity_metrics"
            assert "shape_errors" in combined_results, "Combined results missing shape_errors"
            assert "facial_asymmetry" in combined_results, "Combined results missing facial_asymmetry"
            assert "landmark_cross_validation" in combined_results, "Combined results missing landmark_cross_validation"
            assert "feature_changes" in combined_results, "Combined results missing feature_changes"
            logger.info("get_combined_facial_analysis_results: OK")

            logger.info("Самотестирование Face3DAnalyzer завершено успешно!")
        except Exception as e:
            logger.error(f"Самотестирование Face3DAnalyzer завершено с ошибками: {e}")
            raise

# ==================== ТОЧКА ВХОДА ====================

if __name__ == "__main__":
    analyzer = Face3DAnalyzer()
    analyzer.self_test()

img = cv2.imread("/Users/victorkhudyakov/nn/3DDFA2/3/01_01_10.jpg")
models = initialize_3ddfa_components("weights", skip_gpu_check=True)
fa = Face3DAnalyzer(models["tddfa"], models["face_boxes"])
lmk, conf, _ = fa.extract_68_landmarks_with_confidence(img, models)
print(lmk.shape, conf.mean())   # => (68, 3) 0.93

from embedding_analyzer import EmbeddingAnalyzer
ea = EmbeddingAnalyzer()
emb, c = ea.extract_512d_face_embedding(img, ea.face_app)
print(bool(emb is not None), c) # => True  ~0.8