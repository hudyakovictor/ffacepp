# face_3d_analyzer.py

# 3D анализ лица и извлечение ключевых точек

import numpy as np
import cv2
import pickle
import torch
import yaml
from typing import Tuple, Dict, List, Optional
import logging
import math
from scipy.spatial.distance import euclidean, pdist
from scipy.stats import mode
import os
from datetime import datetime


try:
    from core_config import USE_ONNX, ONNX_EXECUTION_PROVIDERS, IS_MACOS, IS_ARM64
    if USE_ONNX:
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'
        # ИСПРАВЛЕНО: Правильный импорт для структуры 3DDFA_V2
        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX
        FaceBoxes_Actual = FaceBoxes_ONNX
        TDDFA_Actual = TDDFA_ONNX
    else:
        from FaceBoxes import FaceBoxes
        from TDDFA import TDDFA
        FaceBoxes_Actual = FaceBoxes
        TDDFA_Actual = TDDFA
    
    from utils.pose import matrix2angle
    from utils.render import render
    from utils.functions import crop_img, get_suffix, parse_roi_box_from_bbox
    HAS_3DDFA = True
except ImportError as e:
    HAS_3DDFA = False
    logging.error(f"3DDFA_V2 не установлен: {e}")

from core_config import (
    TDDFA_CONFIG, BFM_MODEL, REQUIRED_3DDFA_FILES, 
    get_view_configs, get_identity_signature_metrics,
    MIN_VISIBILITY_Z, STANDARD_IOD, STANDARD_NOSE_EYE,
    STANDARD_FACE_HEIGHT, STANDARD_PROFILE_HEIGHT, EYE_REGION_ERROR_THRESHOLD,
    MILD_ASYMMETRY_THRESHOLD, SEVERE_ASYMMETRY_THRESHOLD,
    LANDMARK_MEAN_DISTANCE_THRESHOLD, LANDMARK_MAX_DISTANCE_THRESHOLD
)

def distance(p1, p2):
    """Вычисляет Евклидово расстояние между двумя точками."""
    return euclidean(p1, p2)

def calculate_angle(p1, p2, p3):
    """Вычисляет угол между тремя точками (p2 - вершина угла)."""
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

def mirror_landmarks(landmarks, center_x):
    """Зеркально отражает landmarks относительно центральной оси X."""
    mirrored = landmarks.copy()
    mirrored[:, 0] = 2 * center_x - mirrored[:, 0]
    return mirrored

def fit_curve(points):
    """Приближает кривую к набору точек для затылочной кривизны."""
    if len(points) < 2:
        return 0.0
    
    x = points[:, 0]
    y = points[:, 1]
    
    try:
        coefficients = np.polyfit(x, y, min(2, len(points) - 1))
        return np.sum(np.abs(coefficients))
    except np.linalg.LinAlgError:
        return 0.0

class Face3DAnalyzer:
    def __init__(self):
        self.tddfa_onnx = None
        self.tddfa_pytorch = None
        self.face_boxes = None
        self.dense_face_model = None  # Для 38000 точек
        self.reference_model_landmarks = None # Для shape error
        self.view_configs = get_view_configs()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.init_done = False
        
        if HAS_3DDFA:
            self.initialize_3ddfa_components()
        else:
            logging.warning("3DDFA_V2 недоступен. Функционал ограничен.")

        # Загрузка референсной 3DMM модели для shape error
        # Теперь мы получаем ландмарки из инициализированного объекта tddfa_pytorch
        self.reference_model_landmarks = None # Инициализируем как None по умолчанию

    def initialize_3ddfa_components(self):
        """Инициализация 3DDFA_V2 и FaceBoxes с проверкой GPU"""
        if self.init_done:
            logging.info("3DDFA_V2 компоненты уже инициализированы.")
            return

        if not HAS_3DDFA:
            logging.error("3DDFA_V2 не может быть инициализирован.")
            return

        try:
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_script_dir, 'configs', 'mb1_120x120.yml')
            
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")

            # Проверка существования BFM_MODEL
            if not os.path.exists(BFM_MODEL):
                raise FileNotFoundError(f"Файл BFM модели не найден: {BFM_MODEL}")

            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)

            self.face_boxes = FaceBoxes_Actual()
            
            if USE_ONNX:
                # Инициализация ONNX модели
                self.tddfa_onnx = TDDFA_Actual(
                    onnx_path=str(REQUIRED_3DDFA_FILES['onnx_model']),
                    device='cpu' if IS_MACOS and IS_ARM64 else self.device,
                    # ИСПРАВЛЕНО: Добавляем checkpoint_fp для корректной конвертации в ONNX, если ONNX файл отсутствует
                    checkpoint_fp=str(REQUIRED_3DDFA_FILES['weights_mb1_120x120']) 
                )
                self.tddfa_pytorch = None
                
            else:
                # Инициализация PyTorch модели
                self.tddfa_pytorch = TDDFA_Actual(
                    arch=cfg['arch'],
                    checkpoint_fp=str(REQUIRED_3DDFA_FILES['weights_mb1_120x120']),
                    bfm_fp=str(BFM_MODEL),
                    size=cfg['size'],
                    device=self.device
                )
                self.tddfa_onnx = None

            self.init_done = True
            logging.info("3DDFA_V2 и FaceBoxes успешно инициализированы.")

            if self.tddfa_pytorch: # Только если PyTorch модель инициализирована, пытаемся получить bfm
                target_bfm_source = self.tddfa_pytorch.bfm
                if 'shape_mu' in target_bfm_source and 'landmarks_idx' in target_bfm_source:
                    shape_mu = target_bfm_source['shape_mu']
                    landmarks_idx = target_bfm_source['landmarks_idx']
                    
                    if hasattr(shape_mu, 'cpu'):
                        shape_mu = shape_mu.cpu().numpy()
                    if hasattr(landmarks_idx, 'cpu'):
                        landmarks_idx = landmarks_idx.cpu().numpy()
                    
                    self.reference_model_landmarks = shape_mu[landmarks_idx, :].reshape(-1, 3)
                    logging.info("Референсные ландмарки BFM успешно загружены.")
                else:
                    logging.warning("Не удалось получить референсные ландмарки из BFM. Возможно, структура bfm некорректна для PyTorch модели.")
            else:
                logging.warning("Референсные ландмарки BFM не загружены, так как используется ONNX модель, которая не предоставляет прямой доступ к bfm или не требует его для shape error.")
                self.reference_model_landmarks = None

        except Exception as e:
            logging.error(f"Ошибка инициализации 3DDFA_V2: {e}")
            self.tddfa_onnx = None
            self.tddfa_pytorch = None
            self.face_boxes = None
            self.init_done = False
            self.reference_model_landmarks = None

    def extract_68_landmarks_with_confidence(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """Извлекает 68 ключевых 3D ландмарков с confidence scores"""
        if self.tddfa_onnx is None and self.tddfa_pytorch is None:
            raise RuntimeError("3DDFA_V2 модель не инициализирована. Извлечение ландмарков невозможно.")

        if self.face_boxes is None:
            raise RuntimeError("FaceBoxes детектор лиц не инициализирован. Извлечение ландмарков невозможно.")

        try:
            # Детекция лиц
            boxes = self.face_boxes(image)
            
            if not isinstance(boxes, np.ndarray):
                logging.warning(f"FaceBoxes вернул тип {type(boxes)}, ожидался np.ndarray. Попытка конвертации.")
                boxes = np.array(boxes)

            if boxes.shape[0] == 0:
                logging.warning("Лица не обнаружены.")
                return np.array([]), np.array([]), (0, 0)

            # Выбор самого большого лица
            boxes_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best_box = boxes[np.argmax(boxes_areas)]

            # Определяем, какую модель использовать
            tddfa_model = self.tddfa_onnx if USE_ONNX and self.tddfa_onnx else self.tddfa_pytorch
            
            if tddfa_model is None:
                raise RuntimeError("3DDFA_V2 модель не инициализирована. Извлечение ландмарков невозможно.")

            # Получение параметров с помощью активной модели
            logging.info(f"Вызов tddfa_model ({'ONNX' if USE_ONNX else 'PyTorch'}) с image.shape={image.shape}, best_box={best_box}")
            param_lst, roi_box_lst = tddfa_model(image, [best_box])
            
            # Убеждаемся, что param_lst и roi_box_lst являются массивами numpy
            param_lst = np.array(param_lst) if isinstance(param_lst, list) else param_lst
            roi_box_lst = np.array(roi_box_lst) if isinstance(roi_box_lst, list) else roi_box_lst

            # Реконструкция вершин с помощью той же активной модели
            ver = tddfa_model.recon_vers(param_lst, roi_box_lst, is_dense=False)[0]

            if not isinstance(ver, np.ndarray):
                logging.error(f"recon_vers вернул тип {type(ver)}, ожидался np.ndarray. Попытка конвертации.")
                ver = np.array(ver)

            if ver.size == 0:
                logging.warning("recon_vers вернул пустой массив.")
                return np.array([]), np.array([]), (0, 0)

            # ИСПРАВЛЕНО: Правильная корректировка координат
            ver = ver.T  # Транспонируем для получения (N, 3)
            ver[:, 1] = -ver[:, 1]  # Инвертируем только Y координаты

            # Оценка уверенности для каждой точки
            confidence_score = best_box[4] if len(best_box) > 4 else 0.9
            confidence_scores = np.ones(68) * confidence_score

            final_return_values = (ver, confidence_scores, image.shape[:2])
            logging.info(f"extract_68_landmarks_with_confidence готовится вернуть landmarks shape: {ver.shape}")
            
            return final_return_values

        except Exception as e:
            logging.error(f"Ошибка в extract_68_landmarks_with_confidence: {e}", exc_info=True)
            return np.array([]), np.array([]), (0, 0)

    def extract_dense_surface_points(self, image: np.ndarray) -> np.ndarray:
        """Извлекает 38000 точек поверхности для детального анализа масок"""
        if self.tddfa_onnx is None and self.tddfa_pytorch is None:
            raise RuntimeError("3DDFA_V2 модель не инициализирована. Извлечение плотных точек невозможно.")

        if self.face_boxes is None:
            raise RuntimeError("FaceBoxes детектор лиц не инициализирован. Извлечение плотных точек невозможно.")

        try:
            boxes = self.face_boxes(image)
            if boxes.shape[0] == 0:
                logging.warning("Лица не обнаружены для извлечения плотных точек.")
                return np.array([])

            best_box = boxes[np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))]

            # Определяем, какую модель использовать
            tddfa_model = self.tddfa_onnx if USE_ONNX and self.tddfa_onnx else self.tddfa_pytorch
            
            if tddfa_model is None:
                raise RuntimeError("3DDFA_V2 модель не инициализирована. Извлечение плотных точек невозможно.")

            # Получение параметров
            param_lst, roi_box_lst = tddfa_model(image, [best_box])
            
            # ИСПРАВЛЕНО: Правильный параметр is_dense=True
            dense_ver = tddfa_model.recon_vers(param_lst, roi_box_lst, is_dense=True)[0]

            # ИСПРАВЛЕНО: Правильная корректировка координат
            dense_ver = dense_ver.T  # Транспонируем
            dense_ver[:, 1] = -dense_ver[:, 1]  # Инвертируем Y

            return dense_ver

        except Exception as e:
            logging.error(f"Ошибка извлечения плотных точек поверхности: {e}")
            raise RuntimeError(f"Не удалось извлечь плотные точки: {e}")

    def determine_precise_face_pose(self, landmarks_3d: np.ndarray) -> Dict:
        """Определяет точный ракурс лица через PnP алгоритм."""
        if landmarks_3d.size == 0:
            return {
                'pose_category': 'Unknown', 
                'angles': (0, 0, 0), 
                'confidence': 0, 
                'details': 'No landmarks provided'
            }

        pose_analysis = self.calculate_precise_head_pose(landmarks_3d)
        
        if not pose_analysis['success']:
            return {
                'pose_category': 'Unknown', 
                'angles': (0, 0, 0), 
                'confidence': 0, 
                'details': 'Failed to calculate pose'
            }

        pitch, yaw, roll = pose_analysis['angles']
        confidence = 0.9

        # Классификация ракурса
        pose_category = 'Unknown'
        for category, config in self.view_configs.items():
            yaw_min, yaw_max = config['yaw_range']
            if yaw_min <= abs(yaw) <= yaw_max:
                pose_category = category
                break

        return {
            'pose_category': pose_category,
            'angles': (pitch, yaw, roll),
            'confidence': confidence,
            'details': f'Pitch: {pitch:.2f}°, Yaw: {yaw:.2f}°, Roll: {roll:.2f}°'
        }

    def normalize_landmarks_by_pose_category(self, landmarks_3d: np.ndarray, pose_category: str) -> np.ndarray:
        """Нормализация 3D landmarks по опорным точкам для каждого ракурса."""
        if landmarks_3d.size == 0 or pose_category not in self.view_configs:
            logging.warning(f"Невозможно нормализовать landmarks для категории: {pose_category}")
            return landmarks_3d

        config = self.view_configs[pose_category]
        normalized_landmarks = landmarks_3d.copy()

        # Центрирование
        center_point = np.mean(landmarks_3d, axis=0)

        # Расчет масштабного коэффициента
        scale_factor = 1.0
        
        if config['scale_metric'] == 'IOD':
            p1_idx, p2_idx = 36, 45
            dist_actual = distance(landmarks_3d[p1_idx], landmarks_3d[p2_idx])
            if dist_actual > 0:
                scale_factor = config['standard_value'] / dist_actual
                
        elif config['scale_metric'] == 'NOSE_EYE':
            nose_tip_idx = 30
            eye_center = (landmarks_3d[36] + landmarks_3d[45]) / 2
            dist_actual = distance(landmarks_3d[nose_tip_idx], eye_center)
            if dist_actual > 0:
                scale_factor = config['standard_value'] / dist_actual
                
        elif config['scale_metric'] == 'FACE_HEIGHT':
            bridge_nose_idx, chin_tip_idx = 27, 8
            dist_actual = distance(landmarks_3d[bridge_nose_idx], landmarks_3d[chin_tip_idx])
            if dist_actual > 0:
                scale_factor = config['standard_value'] / dist_actual
                
        elif config['scale_metric'] == 'PROFILE_HEIGHT':
            forehead_idx, jaw_idx = 19, 8
            dist_actual = distance(landmarks_3d[forehead_idx], landmarks_3d[jaw_idx])
            if dist_actual > 0:
                scale_factor = config['standard_value'] / dist_actual

        if scale_factor <= 0:
            logging.warning("Недопустимый scale_factor. Возврат исходных landmarks.")
            return landmarks_3d

        normalized_landmarks = (landmarks_3d - center_point) * scale_factor

        # Фильтрация невидимых точек
        # Removed: normalized_landmarks[normalized_landmarks[:, 2] < MIN_VISIBILITY_Z] = np.nan

        return normalized_landmarks

    def calculate_identity_signature_metrics(self, normalized_landmarks: np.ndarray, pose_category: str) -> Dict:
        """Вычисляет 15 ключевых метрик идентификации личности."""
        if normalized_landmarks.size == 0:
            return {}

        metrics = {}

        # Определяем ширину и высоту лица для нормализации
        # Эти значения должны быть рассчитаны из normalized_landmarks, не переданы извне как raw
        face_width = distance(normalized_landmarks[0], normalized_landmarks[16]) # Ширина по крайним точкам лица
        face_height = distance(normalized_landmarks[8], normalized_landmarks[27]) # Высота от подбородка до переносицы

        # Геометрия черепа (skull_geometry_signature)
        metrics['skull_width_ratio'] = distance(normalized_landmarks[0], normalized_landmarks[16]) / (face_width + 1e-6)
        metrics['temporal_bone_angle'] = calculate_angle(
    normalized_landmarks[0], normalized_landmarks[17], normalized_landmarks[1])
        metrics['zygomatic_arch_width'] = distance(normalized_landmarks[2], normalized_landmarks[14]) / (face_width + 1e-6)
        # Orbital depth: упрощенно, разница по Z между глазом и точкой на лбу/переносице
        metrics['orbital_depth'] = (np.mean(normalized_landmarks[[37,38,40,41],2]) - np.mean(normalized_landmarks[[19,24],2])) # Примерная глубина глазницы
        metrics['occipital_curve'] = np.std([normalized_landmarks[0,2], normalized_landmarks[8,2], normalized_landmarks[16,2]]) # Стандартное отклонение Z координат точек по контуру головы

        # Пропорции лица (facial_proportions_signature)
        # golden_ratio_deviation: Вызываем функцию для получения отклонений
        # Я предполагаю, что функция calculate_proportional_golden_ratios будет вызвана из metrics_calculator.py
        # в более высоком уровне, и сюда будет передано уже обработанное значение.
        # Поэтому я закомментирую вызов этой функции здесь, чтобы избежать прямой зависимости.
        # golden_ratios_data = calculate_proportional_golden_ratios(normalized_landmarks)
        # metrics['golden_ratio_deviation'] = golden_ratios_data.get('proportion_anomaly', False)

        # Цефалический индекс (ширина/длина черепа * 100)
        # Предполагаем, что 27 - переносица, 8 - подбородок, 0 - крайняя левая, 16 - крайняя правая
        skull_length = distance(normalized_landmarks[27], normalized_landmarks[8])
        skull_width = distance(normalized_landmarks[0], normalized_landmarks[16])
        if skull_length > 0:
            metrics['cephalic_index'] = (skull_width / skull_length) * 100
        else:
            metrics['cephalic_index'] = 0.0

        # nasolabial_angle: Угол между носогубным треугольником (точки 30, 33, 51)
        # 30: tip of nose, 33: subnasale, 51: upper lip (philtrum)
        metrics['nasolabial_angle'] = calculate_angle(normalized_landmarks[30], normalized_landmarks[33], normalized_landmarks[51])

        # orbital_index: Соотношение высоты и ширины глазницы
        # Левый глаз: ширина (36-39), высота (38-40)
        # Правый глаз: ширина (42-45), высота (43-47)
        left_orbital_width = distance(normalized_landmarks[36], normalized_landmarks[39])
        left_orbital_height = distance(normalized_landmarks[38], normalized_landmarks[40])
        right_orbital_width = distance(normalized_landmarks[42], normalized_landmarks[45])
        right_orbital_height = distance(normalized_landmarks[43], normalized_landmarks[47])

        orbital_index_left = left_orbital_height / (left_orbital_width + 1e-6)
        orbital_index_right = right_orbital_height / (right_orbital_width + 1e-6)
        metrics['orbital_index'] = (orbital_index_left + orbital_index_right) / 2

        metrics['forehead_height_ratio'] = distance(normalized_landmarks[27], normalized_landmarks[19]) / (face_height + 1e-6)
        # chin_projection_ratio: Проекция подбородка
        # Расстояние от точки подбородка (8) до проекции на вертикальную ось, проходящую через переносицу (27)
        # Относительно общей глубины лица
        chin_tip = normalized_landmarks[8]
        nasion_x = normalized_landmarks[27, 0]
        chin_projection_distance = abs(chin_tip[0] - nasion_x) # Горизонтальная проекция
        # Используем среднее расстояние от носа до подбородка как нормализатор, или глубину лица
        face_depth = distance(normalized_landmarks[27], normalized_landmarks[8]) + 1e-6 # Глубина лица от переносицы до подбородка по Z
        metrics['chin_projection_ratio'] = chin_projection_distance / face_depth

        # Костная структура (bone_structure_signature)
        # inter_pupillary_distance_ratio: Межзрачковое расстояние (39-42)
        metrics['inter_pupillary_distance_ratio'] = distance(normalized_landmarks[39], normalized_landmarks[42]) / (face_width + 1e-6)

        # Гониальный угол (угол нижней челюсти)
        # Индексы 4, 6, 8, 10, 12 - точки по контуру челюсти
        # Угол между линиями (4-6) и (6-8) для левой стороны
        # Угол между линиями (12-10) и (10-8) для правой стороны
        # Если landmarks[6] не является точкой gonion, нужно найти подходящие
        # Индексы: 5 (левый угол нижней челюсти), 8 (подбородок), 11 (правый угол нижней челюсти)
        gonial_angle_left = calculate_angle(normalized_landmarks[5], normalized_landmarks[8], normalized_landmarks[4])
        gonial_angle_right = calculate_angle(normalized_landmarks[11], normalized_landmarks[8], normalized_landmarks[12])
        metrics['gonial_angle_asymmetry'] = abs(gonial_angle_left - gonial_angle_right)

        # zygomatic_angle: Угол скуловой кости (например, угол, образованный точками 2, 8, 14)
        metrics['zygomatic_angle'] = calculate_angle(normalized_landmarks[2], normalized_landmarks[8], normalized_landmarks[14])

        jaw_vec1 = normalized_landmarks[5] - normalized_landmarks[8]
        jaw_vec2 = normalized_landmarks[11] - normalized_landmarks[8]
        jaw_angle = np.degrees(np.arccos(np.clip(
    np.dot(jaw_vec1, jaw_vec2) / (np.linalg.norm(jaw_vec1) * np.linalg.norm(jaw_vec2) + 1e-6), -1, 1)))
        metrics['jaw_angle_ratio'] = jaw_angle / 120.0 # Нормализация относительно среднего угла челюсти

        # mandibular_symphysis_angle: Угол подбородочного симфиза (точки 5, 8, 11)
        metrics['mandibular_symphysis_angle'] = calculate_angle(normalized_landmarks[5], normalized_landmarks[8], normalized_landmarks[11])

        return metrics

    def calculate_comprehensive_shape_error(self, landmarks_3d: np.ndarray, reference_model_landmarks: np.ndarray) -> Dict:
        """Детальный расчет shape error с анализом по зонам."""
        if landmarks_3d.size == 0 or reference_model_landmarks.size == 0 or landmarks_3d.shape != reference_model_landmarks.shape:
            return {
                'overall_shape_error': 0.0,
                'eye_region_error': 0.0,
                'nose_region_error': 0.0,
                'mouth_region_error': 0.0
            }

        # Общая ошибка формы
        overall_shape_error = np.sqrt(np.mean((landmarks_3d - reference_model_landmarks)**2))

        # Ошибки по регионам
        eye_region_error = np.sqrt(np.mean((landmarks_3d[36:48] - reference_model_landmarks[36:48])**2))
        nose_region_error = np.sqrt(np.mean((landmarks_3d[27:36] - reference_model_landmarks[27:36])**2))
        mouth_region_error = np.sqrt(np.mean((landmarks_3d[48:68] - reference_model_landmarks[48:68])**2))

        return {
            'overall_shape_error': overall_shape_error,
            'eye_region_error': eye_region_error,
            'nose_region_error': nose_region_error,
            'mouth_region_error': mouth_region_error
        }

    def analyze_cranial_bone_structure(self, landmarks_3d: np.ndarray) -> Dict:
        """Анализ костной структуры черепа."""
        if landmarks_3d.size == 0:
            return {}

        bone_metrics = {}

        # Ширина черепа
        bone_metrics['skull_width'] = distance(landmarks_3d[0], landmarks_3d[16])

        # Височные углы
        try:
            bone_metrics['temporal_bone_angle'] = calculate_angle(
                landmarks_3d[0], landmarks_3d[17], landmarks_3d[1]
            )
        except:
            bone_metrics['temporal_bone_angle'] = 0.0

        # Ширина скуловых дуг
        bone_metrics['zygomatic_arch_width'] = distance(landmarks_3d[2], landmarks_3d[14])

        # Глубина орбит
        bone_metrics['orbital_depth'] = np.mean(landmarks_3d[36:48, 2])

        # Длина черепа
        bone_metrics['skull_length'] = distance(landmarks_3d[3], landmarks_3d[12])

        # Тип черепа
        if 0.0 < bone_metrics['cephalic_index'] < 0.8:
            bone_metrics['skull_type'] = 'dolichocephalic'
        elif bone_metrics['cephalic_index'] > 0.85:
            bone_metrics['skull_type'] = 'brachycephalic'
        else:
            bone_metrics['skull_type'] = 'mesocephalic'

        # Орбитальный индекс
        left_orbital_width = distance(landmarks_3d[36], landmarks_3d[39])
        left_orbital_height = distance(landmarks_3d[37], landmarks_3d[41])
        if left_orbital_width > 0:
            bone_metrics['orbital_index'] = left_orbital_height / left_orbital_width
        else:
            bone_metrics['orbital_index'] = 0.0

        return bone_metrics

    def detect_facial_asymmetry_patterns(self, landmarks_3d: np.ndarray) -> Dict:
        """
        Анализ паттернов асимметрии лица.
        Выявляет естественную асимметрию и индикаторы хирургической асимметрии.
        Возвращает natural_asymmetry_coefficients и surgical_asymmetry_indicators.
        """
        asymmetry_results = {
            'overall_asymmetry_score': 0.0,
            'mild_asymmetry_detected': False,
            'severe_asymmetry_detected': False,
            'asymmetry_details': []
        }

        if landmarks_3d.size == 0 or len(landmarks_3d) < 68:
            asymmetry_results['asymmetry_details'].append("Недостаточно ландмарков для анализа асимметрии.")
            return asymmetry_results

        # Определяем центральную ось лица (по средней X координате между глазами или по переносице)
        center_x = (landmarks_3d[36, 0] + landmarks_3d[45, 0]) / 2 # Средняя X между внешними углами глаз

        # Определяем симметричные пары ландмарков для анализа
        # Это примерный список, его можно расширить
        left_indices = [0, 1, 2, 3, 4, 5, 6, 7, 17, 18, 19, 20, 21, 36, 37, 38, 39, 40, 41, 48, 49, 50, 59, 58, 67]
        right_indices = [16, 15, 14, 13, 12, 11, 10, 9, 26, 25, 24, 23, 22, 45, 44, 43, 42, 47, 46, 54, 53, 52, 55, 56, 57]

        if len(left_indices) != len(right_indices):
            asymmetry_results['asymmetry_details'].append("Ошибка: Количество левых и правых ландмарков не совпадает.")
            return asymmetry_results
        
        # Извлекаем соответствующие ландмарки
        left_landmarks = landmarks_3d[left_indices, :2] # Используем только XY для симметрии
        right_landmarks = landmarks_3d[right_indices, :2]

        # Зеркально отражаем правые ландмарки относительно центральной оси
        right_landmarks_mirrored = mirror_landmarks(right_landmarks, center_x)

        # Вычисляем асимметрию как Евклидово расстояние между фактическими левыми
        # и зеркально отраженными правыми ландмарками
        asymmetry_scores_per_point = np.linalg.norm(left_landmarks - right_landmarks_mirrored, axis=1)
        overall_asymmetry_score = np.mean(asymmetry_scores_per_point)

        asymmetry_results['overall_asymmetry_score'] = float(overall_asymmetry_score)

        # Классификация асимметрии
        if overall_asymmetry_score > SEVERE_ASYMMETRY_THRESHOLD:
            asymmetry_results['severe_asymmetry_detected'] = True
            asymmetry_results['asymmetry_details'].append("Обнаружена сильная асимметрия лица.")
        elif overall_asymmetry_score > MILD_ASYMMETRY_THRESHOLD:
            asymmetry_results['mild_asymmetry_detected'] = True
            asymmetry_results['asymmetry_details'].append("Обнаружена умеренная асимметрия лица.")
        else:
            asymmetry_results['asymmetry_details'].append("Асимметрия лица находится в пределах нормы.")

        # Дополнительный анализ для хирургических индикаторов
        # Хирургическая асимметрия часто проявляется как внезапные, локализованные
        # и неестественные изменения, которые не соответствуют естественной асимметрии.
        # Для этого потребуется сравнение с историческими данными или
        # более сложные модели.
        # Здесь мы просто проверяем наличие очень высоких отклонений для отдельных точек
        # которые могут указывать на локальные вмешательства.

        max_asymmetry_per_point = np.max(asymmetry_scores_per_point)
        if max_asymmetry_per_point > (SEVERE_ASYMMETRY_THRESHOLD * 2): # Произвольный порог для резких изменений
            asymmetry_results['surgical_asymmetry_indicators'] = True
            asymmetry_results['asymmetry_details'].append(f"Возможны признаки хирургического вмешательства: аномально высокое локальное отклонение ({max_asymmetry_per_point:.2f}).")
        else:
            asymmetry_results['surgical_asymmetry_indicators'] = False
        
        return asymmetry_results

    def adaptive_landmark_extraction(self, image: np.ndarray, quality_score: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Адаптивное извлечение landmarks в зависимости от качества изображения.
        Если quality_score ниже определенного порога, извлечение может быть пропущено
        или выполнены дополнительные шаги предварительной обработки.
        """
        min_quality_threshold = 0.6 # Примерный порог качества (можно вынести в core_config)

        if quality_score < min_quality_threshold:
            logging.warning(f"Качество изображения ({quality_score:.2f}) ниже порога ({min_quality_threshold:.2f}). Извлечение ландмарков может быть неточным или пропущено.")
            # Можно добавить логику для улучшения изображения или возвращать пустые результаты
            return np.array([]), np.array([])
        
        try:
            # Вызываем основную функцию извлечения ландмарков
            landmarks_3d, confidence_scores, _ = self.extract_68_landmarks_with_confidence(image)
            return landmarks_3d, confidence_scores
        except RuntimeError as e:
            logging.error(f"Ошибка при адаптивном извлечении ландмарков: {e}")
            return np.array([]), np.array([])

    def validate_landmark_consistency(self, landmarks_sequence: List[np.ndarray]) -> Dict:
        """
        Проверяет консистентность landmarks в последовательности.
        Анализирует стабильность положения ландмарков между последовательными кадрами
        и выявляет аномальные "скачки" или неконсистентность.
        Возвращает словарь с результатами валидации.
        """
        consistency_results = {
            'overall_consistency': 1.0, # 1.0 - высокая консистентность
            'inconsistencies_detected': False,
            'details': []
        }

        if not landmarks_sequence or len(landmarks_sequence) < 2:
            consistency_results['details'].append("Недостаточно данных для проверки консистентности ландмарков (требуется минимум 2 набора).")
            return consistency_results

        num_landmarks = landmarks_sequence[0].shape[0]
        if num_landmarks == 0:
            consistency_results['details'].append("Ландмарки пусты.")
            return consistency_results

        # Расчет среднего смещения для каждой точки между последовательными кадрами
        # и общего коэффициента вариации для каждой точки.
        landmark_displacements = [[] for _ in range(num_landmarks)]
        
        for i in range(1, len(landmarks_sequence)):
            prev_landmarks = landmarks_sequence[i-1]
            current_landmarks = landmarks_sequence[i]

            # Проверка на соответствие размеров
            if prev_landmarks.shape != current_landmarks.shape:
                consistency_results['inconsistencies_detected'] = True
                consistency_results['overall_consistency'] = 0.0
                consistency_results['details'].append(f"Несоответствие размеров ландмарков между кадром {i-1} и {i}.")
                return consistency_results

            # Расчет евклидова расстояния для каждой соответствующей точки
            for j in range(num_landmarks):
                dist = np.linalg.norm(current_landmarks[j] - prev_landmarks[j])
                landmark_displacements[j].append(dist)
        
        # Анализ смещений
        total_mean_displacement = 0.0
        inconsistent_landmarks = []

        for j in range(num_landmarks):
            if not landmark_displacements[j]:
                continue
            
            mean_disp = np.mean(landmark_displacements[j])
            std_disp = np.std(landmark_displacements[j])
            total_mean_displacement += mean_disp

            # Если среднее смещение слишком велико, или есть большие скачки (выбросы)
            if mean_disp > LANDMARK_MEAN_DISTANCE_THRESHOLD: # Порог из core_config
                inconsistent_landmarks.append(f"Ландмарк {j}: Среднее смещение ({mean_disp:.2f}px) превышает порог ({LANDMARK_MEAN_DISTANCE_THRESHOLD}px).")
                consistency_results['inconsistencies_detected'] = True

            # Детекция резких скачков (outliers) в смещениях
            if len(landmark_displacements[j]) > 2:
                q1, q3 = np.percentile(landmark_displacements[j], [25, 75])
                iqr = q3 - q1
                upper_bound = q3 + 1.5 * iqr
                # Или использовать абсолютный порог
                if np.max(landmark_displacements[j]) > LANDMARK_MAX_DISTANCE_THRESHOLD: # Порог из core_config
                    inconsistent_landmarks.append(f"Ландмарк {j}: Обнаружен резкий скачок ({np.max(landmark_displacements[j]):.2f}px) в смещении.")
                    consistency_results['inconsistencies_detected'] = True
        
        if inconsistent_landmarks:
            consistency_results['details'].extend(inconsistent_landmarks)
            consistency_results['overall_consistency'] = max(0.0, 1.0 - (total_mean_displacement / (num_landmarks * LANDMARK_MEAN_DISTANCE_THRESHOLD))) # Простое снижение балла
        
        # Если не обнаружено явных проблем, но среднее смещение все еще заметно
        elif total_mean_displacement / num_landmarks > (LANDMARK_MEAN_DISTANCE_THRESHOLD / 2):
            consistency_results['overall_consistency'] = max(0.0, 0.8 - (total_mean_displacement / (num_landmarks * LANDMARK_MEAN_DISTANCE_THRESHOLD)))
            consistency_results['details'].append(f"Общее среднее смещение ландмарков ({total_mean_displacement/num_landmarks:.2f}px) заметно, но в пределах нормы.")

        return consistency_results

    def calculate_precise_head_pose(self, landmarks_3d: np.ndarray) -> Dict:
        """Точное вычисление позы головы через PnP алгоритм."""
        if landmarks_3d.size == 0 or landmarks_3d.shape[1] != 3:
            return {'success': False, 'angles': (0, 0, 0)}

        # 3D модельные точки
        model_points = np.array([
            (0.0, 0.0, 0.0),           # Кончик носа
            (0.0, -330.0, -65.0),      # Подбородок
            (-225.0, 170.0, -135.0),   # Левый угол глаза
            (225.0, 170.0, -135.0),    # Правый угол глаза
            (-150.0, -150.0, -125.0),  # Левый угол рта
            (150.0, -150.0, -125.0)    # Правый угол рта
        ], dtype=np.float64)

        # Соответствующие 2D точки
        image_points = np.array([
            landmarks_3d[30, :2],  # Кончик носа
            landmarks_3d[8, :2],   # Подбородок
            landmarks_3d[36, :2],  # Левый угол глаза
            landmarks_3d[45, :2],  # Правый угол глаза
            landmarks_3d[48, :2],  # Левый угол рта
            landmarks_3d[54, :2]   # Правый угол рта
        ], dtype=np.float64)

        if np.any(np.isnan(image_points)) or np.any(np.isinf(image_points)): # Add check for inf too
            logging.warning("image_points contains NaN or Inf values, cannot calculate pose.")
            return {'success': False, 'angles': (0, 0, 0)}

        # Параметры камеры
        mean_coord = np.mean(landmarks_3d[:, :2], axis=0)
        focal_length = mean_coord[1] * 1.5
        
        # Check mean_coord and focal_length for NaN/Inf
        if np.any(np.isnan(mean_coord)) or np.any(np.isinf(mean_coord)) or \
           np.isnan(focal_length) or np.isinf(focal_length) or focal_length == 0:
            logging.warning("Calculated mean_coord or focal_length contains NaN/Inf or focal_length is zero, cannot calculate pose.")
            return {'success': False, 'angles': (0, 0, 0)}

        camera_matrix = np.array([
            [focal_length, 0, mean_coord[0]],
            [0, focal_length, mean_coord[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))

        # Решение PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            logging.warning("cv2.solvePnP failed to find a solution.")
            return {'success': False, 'angles': (0, 0, 0)}

        # Преобразование в углы Эйлера
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Check rotation_matrix for NaN/Inf before passing to _rotation_matrix_to_euler_angles
        if np.any(np.isnan(rotation_matrix)) or np.any(np.isinf(rotation_matrix)):
            logging.warning("Rotation matrix contains NaN or Inf values.")
            return {'success': False, 'angles': (0, 0, 0)}

        angles = self._rotation_matrix_to_euler_angles(rotation_matrix)
        
        # Ensure angles are not NaN/Inf
        if np.any(np.isnan(angles)) or np.any(np.isinf(angles)):
            logging.warning("Euler angles contain NaN or Inf values after conversion.")
            return {'success': False, 'angles': (0, 0, 0)}

        return {
            'success': True,
            'angles': angles,
            'rotation_matrix': rotation_vector, # Corrected: should be rotation_vector from solvePnP
            'translation_vector': translation_vector
        }
    
    def _rotation_matrix_to_euler_angles(self, R: np.ndarray) -> Tuple[float, float, float]:
        """Преобразование матрицы вращения в углы Эйлера."""
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])  # Pitch
            y = np.arctan2(-R[2, 0], sy)      # Yaw
            z = np.arctan2(R[1, 0], R[0, 0])  # Roll
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0

        # Check for NaN/Inf after calculations and before returning
        angles = (np.degrees(x), np.degrees(y), np.degrees(z))
        if np.any(np.isnan(angles)) or np.any(np.isinf(angles)):
            logging.warning(f"NaN or Inf detected in calculated Euler angles: {angles}. Returning (0.0, 0.0, 0.0) as fallback.")
            return (0.0, 0.0, 0.0)

        return angles

    def validate_landmarks_quality(self, landmarks_3d: np.ndarray, image_shape: Tuple[int, int]) -> Dict:
        """Валидация качества извлеченных landmarks."""
        if landmarks_3d.size == 0:
            return {'valid': False, 'reason': 'Empty landmarks array'}

        h, w = image_shape
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'quality_score': 1.0
        }

        # Проверка границ изображения
        x_coords = landmarks_3d[:, 0]
        y_coords = landmarks_3d[:, 1]
        
        out_of_bounds = np.sum((x_coords < 0) | (x_coords >= w) | (y_coords < 0) | (y_coords >= h))
        if out_of_bounds > 0:
            validation_results['warnings'].append(f'{out_of_bounds} landmarks вне границ изображения')
            validation_results['quality_score'] *= 0.9

        # Проверка анатомической корректности
        if len(landmarks_3d) >= 68:
            # Межзрачковое расстояние
            iod = distance(landmarks_3d[36], landmarks_3d[45])
            if iod < 20 or iod > 200:  # Разумные пределы в пикселях
                validation_results['warnings'].append(f'Подозрительное межзрачковое расстояние: {iod:.1f}px')
                validation_results['quality_score'] *= 0.8

            # Симметрия глаз
            left_eye_center = np.mean(landmarks_3d[36:42], axis=0)
            right_eye_center = np.mean(landmarks_3d[42:48], axis=0)
            eye_height_diff = abs(left_eye_center[1] - right_eye_center[1])
            
            if eye_height_diff > iod * 0.1:  # Более 10% от IOD
                validation_results['warnings'].append(f'Асимметрия глаз: {eye_height_diff:.1f}px')
                validation_results['quality_score'] *= 0.85

            # Проверка Z-координат
            z_coords = landmarks_3d[:, 2]
            if np.any(np.isnan(z_coords)) or np.any(np.isinf(z_coords)):
                validation_results['errors'].append('Некорректные Z-координаты (NaN или Inf)')
                validation_results['valid'] = False

        if validation_results['quality_score'] < 0.5:
            validation_results['valid'] = False
            validation_results['errors'].append('Качество landmarks ниже минимального порога')

        return validation_results

    def calculate_landmark_stability_metrics(self, landmarks_sequence: List[np.ndarray]) -> Dict:
        """Расчет метрик стабильности landmarks в последовательности."""
        if len(landmarks_sequence) < 2:
            return {'stability_score': 1.0, 'mean_drift': 0.0, 'max_drift': 0.0}

        stability_metrics = {
            'frame_to_frame_distances': [],
            'cumulative_drift': [],
            'stability_score': 0.0,
            'mean_drift': 0.0,
            'max_drift': 0.0
        }

        reference_landmarks = landmarks_sequence[0]
        cumulative_drift = 0.0

        for i in range(1, len(landmarks_sequence)):
            current_landmarks = landmarks_sequence[i]
            
            if current_landmarks.size == 0 or reference_landmarks.size == 0:
                continue

            # Покадровое расстояние
            frame_distances = np.linalg.norm(current_landmarks - landmarks_sequence[i-1], axis=1)
            mean_frame_distance = np.mean(frame_distances)
            stability_metrics['frame_to_frame_distances'].append(mean_frame_distance)

            # Кумулятивный дрейф от первого кадра
            drift_distances = np.linalg.norm(current_landmarks - reference_landmarks, axis=1)
            cumulative_drift = np.mean(drift_distances)
            stability_metrics['cumulative_drift'].append(cumulative_drift)

        if stability_metrics['frame_to_frame_distances']:
            stability_metrics['mean_drift'] = np.mean(stability_metrics['frame_to_frame_distances'])
            stability_metrics['max_drift'] = np.max(stability_metrics['frame_to_frame_distances'])
            
            # Стабильность обратно пропорциональна среднему дрейфу
            stability_metrics['stability_score'] = max(0.0, 1.0 - stability_metrics['mean_drift'] / 10.0)

        return stability_metrics

    def detect_landmark_outliers(self, landmarks_3d: np.ndarray) -> Dict:
        """Выявление выбросов среди landmarks."""
        if landmarks_3d.size == 0:
            return {'outliers': [], 'outlier_ratio': 0.0}

        outlier_detection = {
            'outliers': [],
            'outlier_ratio': 0.0,
            'outlier_indices': []
        }

        # Статистический анализ координат
        for axis in range(3):  # X, Y, Z
            coords = landmarks_3d[:, axis]
            mean_coord = np.mean(coords)
            std_coord = np.std(coords)
            
            # Z-score анализ
            z_scores = np.abs((coords - mean_coord) / (std_coord + 1e-8))
            outlier_mask = z_scores > 3.0  # 3-sigma правило
            
            outlier_indices = np.where(outlier_mask)[0]
            for idx in outlier_indices:
                if idx not in outlier_detection['outlier_indices']:
                    outlier_detection['outlier_indices'].append(idx)
                    outlier_detection['outliers'].append({
                        'landmark_index': idx,
                        'axis': ['X', 'Y', 'Z'][axis],
                        'value': coords[idx],
                        'z_score': z_scores[idx],
                        'coordinates': landmarks_3d[idx].tolist()
                    })

        outlier_detection['outlier_ratio'] = len(outlier_detection['outlier_indices']) / len(landmarks_3d)
        return outlier_detection

    def _analyze_geometric_consistency(self, landmarks_3d: np.ndarray) -> float:
        """Расчет балла геометрической консистентности landmarks с учетом треугольных неравенств."""
        if landmarks_3d.size == 0 or len(landmarks_3d) < 68:
            logging.warning("Недостаточно данных для анализа геометрической консистентности.")
            return 0.0

        consistency_scores = []

        # Проверка анатомических соотношений (существующая логика)
        try:
            # Соотношение ширины и высоты лица
            face_width = distance(landmarks_3d[0], landmarks_3d[16])
            face_height = distance(landmarks_3d[8], landmarks_3d[27])
            
            if face_width > 0 and face_height > 0:
                aspect_ratio = face_height / face_width
                # Нормальное соотношение 1.2-1.8
                if 1.2 <= aspect_ratio <= 1.8:
                    consistency_scores.append(1.0)
                else:
                    consistency_scores.append(max(0.0, 1.0 - abs(aspect_ratio - 1.5) / 0.5))

            # Симметрия носа
            nose_left = landmarks_3d[31]
            nose_right = landmarks_3d[35]
            nose_tip = landmarks_3d[30]
            
            nose_center_x = (nose_left[0] + nose_right[0]) / 2
            nose_tip_deviation = abs(nose_tip[0] - nose_center_x)
            nose_width = abs(nose_left[0] - nose_right[0])
            
            if nose_width > 0:
                nose_symmetry = max(0.0, 1.0 - nose_tip_deviation / (nose_width / 2))
                consistency_scores.append(nose_symmetry)

            # Проверка положения рта относительно носа
            mouth_center = np.mean(landmarks_3d[48:68], axis=0)
            mouth_nose_alignment = abs(mouth_center[0] - nose_tip[0])
            
            if face_width > 0:
                mouth_alignment_score = max(0.0, 1.0 - mouth_nose_alignment / (face_width * 0.1))
                consistency_scores.append(mouth_alignment_score)

            # Добавление проверки треугольных неравенств для ключевых областей
            # Пример: треугольник глаза (left_eye_outer, left_eye_inner, left_eyebrow_center)
            # Индексы landmarks: 36 (левый внешний уголок глаза), 39 (левый внутренний уголок глаза), 20 (центр левой брови)
            if len(landmarks_3d) >= 40: # Убедимся, что индексы существуют
                p1_eye = landmarks_3d[36]
                p2_eye = landmarks_3d[39]
                p3_eyebrow = landmarks_3d[20] # Центр левой брови

                dist1 = distance(p1_eye, p2_eye)
                dist2 = distance(p2_eye, p3_eyebrow)
                dist3 = distance(p3_eyebrow, p1_eye)

                # Проверка треугольного неравенства: a + b > c
                if dist1 + dist2 > dist3 and dist1 + dist3 > dist2 and dist2 + dist3 > dist1:
                    consistency_scores.append(1.0) # Соответствует треугольному неравенству
                else:
                    consistency_scores.append(0.0) # Не соответствует

            # Добавить больше таких проверок для других ключевых треугольников, например, вокруг рта, подбородка.
            # Например, треугольник вокруг носа (кончик, левое крыло, правое крыло): 30, 31, 35
            if len(landmarks_3d) >= 36:
                p1_nose = landmarks_3d[30] # Кончик носа
                p2_nose = landmarks_3d[31] # Левое крыло носа
                p3_nose = landmarks_3d[35] # Правое крыло носа

                dist1_n = distance(p1_nose, p2_nose)
                dist2_n = distance(p2_nose, p3_nose)
                dist3_n = distance(p3_nose, p1_nose)

                if dist1_n + dist2_n > dist3_n and dist1_n + dist3_n > dist2_n and dist2_n + dist3_n > dist1_n:
                    consistency_scores.append(1.0)
                else:
                    consistency_scores.append(0.0)


        except Exception as e:
            logging.warning(f"Ошибка при расчете геометрической консистентности: {e}")
            return 0.5 # Возвращаем средний балл в случае ошибки

        return np.mean(consistency_scores) if consistency_scores else 0.5

    def _analyze_temporal_consistency(self, temporal_context: Dict) -> float:
        """
        Анализирует временную консистентность 3D-метрик лица в краткосрочной перспективе.
        Оценивает стабильность ключевых метрик идентичности на основе последовательности кадров.
        Возвращает балл консистентности (0-1), где 1 - максимальная стабильность.
        """
        metrics_sequence = temporal_context.get('metrics_sequence', [])

        if not metrics_sequence or len(metrics_sequence) < 2:
            # Недостаточно данных для анализа временной консистентности,
            # считаем ее высокой по умолчанию.
            return 1.0

        # Собираем значения для каждой метрики по всей последовательности
        all_metric_values = {}
        # Получаем список ожидаемых метрик из core_config
        identity_signature_configs = get_identity_signature_metrics()
        for sig_type in identity_signature_configs.values():
            for metric_name in sig_type['metrics']:
                all_metric_values[metric_name] = []

        for metrics_dict in metrics_sequence:
            for metric_name, value in metrics_dict.items():
                if metric_name in all_metric_values:
                    all_metric_values[metric_name].append(value)

        consistency_scores = []
        for metric_name, values in all_metric_values.items():
            if len(values) < 2:
                consistency_scores.append(1.0) # Если только одно значение, считаем стабильным
                continue
            
            mean_val = np.mean(values)
            std_val = np.std(values)

            if mean_val == 0:
                consistency_scores.append(1.0 if std_val == 0 else 0.0)
                continue

            cv = std_val / mean_val
            
            # Преобразуем CV в балл консистентности от 0 до 1.
            # Используем функцию, чтобы 1 было при CV=0, и падало с ростом CV.
            consistency = 1 / (1 + cv * 5) # Коэффициент 5 можно настроить.

            consistency_scores.append(consistency)

        if not consistency_scores:
            return 0.0 # Если никаких метрик не удалось оценить

        return np.mean(consistency_scores)

    def estimate_landmark_uncertainty(self, landmarks_3d: np.ndarray, confidence_scores: np.ndarray) -> Dict:
        """Оценка неопределенности для каждого landmark."""
        if landmarks_3d.size == 0:
            return {'mean_uncertainty': 1.0, 'per_landmark_uncertainty': []}

        uncertainty_analysis = {
            'mean_uncertainty': 0.0,
            'per_landmark_uncertainty': [],
            'high_uncertainty_landmarks': [],
            'uncertainty_distribution': {}
        }

        # Базовая неопределенность на основе confidence scores
        base_uncertainty = 1.0 - confidence_scores

        # Дополнительная неопределенность на основе геометрии
        for i, landmark in enumerate(landmarks_3d):
            geometric_uncertainty = 0.0
            
            # Неопределенность на основе Z-координаты
            z_coord = landmark[2]
            if z_coord < MIN_VISIBILITY_Z:
                geometric_uncertainty += 0.3
            
            # Неопределенность на основе положения относительно границ
            # (требует знания размеров изображения, здесь упрощенно)
            
            total_uncertainty = min(1.0, base_uncertainty[i] + geometric_uncertainty)
            uncertainty_analysis['per_landmark_uncertainty'].append(total_uncertainty)
            
            if total_uncertainty > 0.7:
                uncertainty_analysis['high_uncertainty_landmarks'].append({
                    'index': i,
                    'uncertainty': total_uncertainty,
                    'coordinates': landmark.tolist()
                })

        uncertainty_analysis['mean_uncertainty'] = np.mean(uncertainty_analysis['per_landmark_uncertainty'])
        
        # Распределение неопределенности по диапазонам
        uncertainties = np.array(uncertainty_analysis['per_landmark_uncertainty'])
        uncertainty_analysis['uncertainty_distribution'] = {
            'low_uncertainty_count': np.sum(uncertainties < 0.3),
            'medium_uncertainty_count': np.sum((uncertainties >= 0.3) & (uncertainties < 0.7)),
            'high_uncertainty_count': np.sum(uncertainties >= 0.7)
        }

        return uncertainty_analysis

    def perform_cross_validation_landmarks(self, image: np.ndarray, iterations: int = 3) -> Dict:
        """Кросс-валидация извлечения landmarks для оценки стабильности."""
        if not HAS_3DDFA or (self.tddfa_onnx is None and self.tddfa_pytorch is None): # Уточнение проверки
            return {'stable': False, 'reason': '3DDFA_V2 недоступен'}

        validation_results = {
            'stable': True,
            'iterations_performed': 0,
            'mean_variation': 0.0,
            'max_variation': 0.0,
            'stability_score': 0.0,
            'landmarks_variations': []
        }

        all_landmarks = []
        
        for i in range(iterations):
            try:
                # Небольшие вариации в параметрах детекции для тестирования стабильности
                landmarks_3d, confidence_scores, _ = self.extract_68_landmarks_with_confidence(image)
                
                if landmarks_3d.size > 0:
                    all_landmarks.append(landmarks_3d)
                    validation_results['iterations_performed'] += 1
                    
            except Exception as e:
                logging.warning(f"Ошибка в итерации {i} кросс-валидации: {e}")
                continue

        if len(all_landmarks) < 2:
            validation_results['stable'] = False
            validation_results['reason'] = 'Недостаточно успешных извлечений для валидации'
            return validation_results

        # Анализ вариаций между итерациями
        variations = []
        reference_landmarks = all_landmarks[0]
        
        for landmarks in all_landmarks[1:]:
            if landmarks.shape == reference_landmarks.shape:
                variation = np.linalg.norm(landmarks - reference_landmarks, axis=1)
                variations.extend(variation)

        if variations:
            validation_results['mean_variation'] = np.mean(variations)
            validation_results['max_variation'] = np.max(variations)
            validation_results['landmarks_variations'] = variations
            
            # Стабильность обратно пропорциональна вариации
            validation_results['stability_score'] = max(0.0, 1.0 - validation_results['mean_variation'] / 5.0)
            
            if validation_results['mean_variation'] > 3.0:  # Порог в пикселях
                validation_results['stable'] = False

        return validation_results

    def generate_landmarks_quality_report(self, landmarks_3d: np.ndarray, confidence_scores: np.ndarray, 
                                        image_shape: Tuple[int, int]) -> Dict:
        """Генерация комплексного отчета о качестве landmarks."""
        quality_report = {
            'timestamp': datetime.now().isoformat(),
            'landmarks_count': len(landmarks_3d) if landmarks_3d.size > 0 else 0,
            'overall_quality_score': 0.0,
            'validation_results': {},
            'outlier_analysis': {},
            'geometric_consistency': 0.0,
            'uncertainty_analysis': {}, 
            'recommendations': []
        }

        if landmarks_3d.size == 0:
            quality_report['overall_quality_score'] = 0.0
            quality_report['recommendations'].append('Landmarks не извлечены - проверьте качество изображения')
            return quality_report

        # Валидация
        quality_report['validation_results'] = self.validate_landmarks_quality(landmarks_3d, image_shape)
        
        # Анализ выбросов
        quality_report['outlier_analysis'] = self.detect_landmark_outliers(landmarks_3d)
        
        # Геометрическая консистентность
        quality_report['geometric_consistency'] = self._analyze_geometric_consistency(landmarks_3d)
        
        # Анализ неопределенности
        quality_report['uncertainty_analysis'] = self.estimate_landmark_uncertainty(landmarks_3d, confidence_scores)

        # Расчет общего балла качества
        quality_components = [
            quality_report['validation_results']['quality_score'],
            1.0 - quality_report['outlier_analysis']['outlier_ratio'],
            quality_report['geometric_consistency'],
            1.0 - quality_report['uncertainty_analysis']['mean_uncertainty']
        ]
        
        quality_report['overall_quality_score'] = np.mean(quality_components)

        # Рекомендации
        if quality_report['overall_quality_score'] < 0.5:
            quality_report['recommendations'].append('Низкое качество landmarks - рассмотрите повторную обработку')
        
        if quality_report['outlier_analysis']['outlier_ratio'] > 0.1:
            quality_report['recommendations'].append('Высокий процент выбросов - проверьте освещение и четкость изображения')
        
        if quality_report['uncertainty_analysis']['mean_uncertainty'] > 0.6:
            quality_report['recommendations'].append('Высокая неопределенность - улучшите качество изображения')

        return quality_report

    def export_landmarks_for_verification(self, landmarks_3d: np.ndarray, output_path: str) -> bool:
        """Экспортирует 3D ландмарки в файл для внешней верификации."""
        try:
            # Пример сохранения в текстовый файл или CSV
            np.savetxt(output_path, landmarks_3d, delimiter=', ')
            logging.info(f"Ландмарки экспортированы в {output_path}")
            return True
        except Exception as e:
            logging.error(f"Ошибка экспорта ландмарков: {e}")
            return False
        
    def self_test(self):
        """Быстрый тест инициализации на M1"""
        import platform
        print(f"Платформа: {platform.system()} {platform.machine()}")
        print(f"ONNX провайдеры: {ONNX_EXECUTION_PROVIDERS}")
        
        # Тест с синтетическим изображением
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        try:
            landmarks, confidence, shape = self.extract_68_landmarks_with_confidence(test_image)
            print(f"✅ 3DDFA_V2 работает: {landmarks.shape if landmarks.size > 0 else 'No face detected'}")
        except Exception as e:
            print(f"❌ 3DDFA_V2 ошибка: {e}")

    # Добавьте в конец файла face_3d_analyzer.py:
    if __name__ == "__main__":
        analyzer = Face3DAnalyzer()
        analyzer.self_test()
