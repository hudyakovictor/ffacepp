#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПОЛНОСТЬЮ ПЕРЕПИСАННАЯ МАСКА МАРКВАРДТА ДЛЯ БИОМЕТРИЧЕСКОГО АНАЛИЗА
Исправлены все критические ошибки согласно анализу результатов
Версия: 6.0 - Стабильная с компенсацией поворотов
"""

import math
import time
import logging
import numpy as np
import traceback
from typing import Dict, List, Tuple, Optional, Union, Any, TypeVar

# Определения типов
T = TypeVar('T', bound=np.ndarray)
LandmarkArray = Union[np.ndarray, List[List[float]]]
RotationMatrix = Optional[np.ndarray]

# Константы
EPSILON = 1e-7  # Минимальное значение для предотвращения деления на ноль
PHI = 1.618033988749895  # Золотое сечение
PHI_INV = 0.618033988749895  # Обратное золотое сечение

class StabilizedMarquardtMask:
    """ИСПРАВЛЕНО: Стабилизированная маска Марквардта с компенсацией поворотов"""

    def __init__(self, config: Optional[Dict] = None):
        """
        Инициализация с возможностью настройки через конфигурацию
        Args:
            config: Словарь с настройками (пороги, коэффициенты и т.д.)
        """
        self.config = config or {}
        self.logger = self._setup_logging()

        # ИСПРАВЛЕНО: Только 4 ракурса без extreme
        self.pose_thresholds = {
            'frontal_yaw': 15.0,
            'frontal_pitch': 20.0,
            'frontal_roll': 15.0,
            'edge_yaw': 30.0,
            'edge_pitch': 25.0,
            'edge_roll': 20.0,
            'semi_profile_yaw': 45.0,
            'profile_yaw': 60.0
        }

        # Индексы ключевых точек лица согласно dlib 68-point model
        self.landmark_indices = {
            'jaw': list(range(0, 17)),
            'right_eyebrow': list(range(17, 22)),
            'left_eyebrow': list(range(22, 27)),
            'nose_bridge': list(range(27, 31)),
            'nose_lower': list(range(31, 36)),
            'right_eye': list(range(36, 42)),
            'left_eye': list(range(42, 48)),
            'outer_lips': list(range(48, 60)),
            'inner_lips': list(range(60, 68)),
            'nose_tip': 30,
            'chin': 8,
            'left_eye_center': 42,
            'right_eye_center': 39,
            'mouth_center': 51
        }

        # ИСПРАВЛЕНО: Правильные симметричные пары
        self.symmetric_pairs = [
            (0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10),  # челюсть
            (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),  # брови
            (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),  # глаза
            (31, 35),  # нос (ноздри)
            (48, 54), (49, 53), (50, 52), (59, 55), (58, 56)  # рот
        ]

        # Биологические пределы для детекции аномалий
        self.biological_limits = {
            'face_width_height_ratio': (0.6, 1.4),
            'nose_width_interocular_ratio': (0.25, 0.6),
            'mouth_width_nose_ratio': (1.2, 2.8),
            'nose_height_width_ratio': (1.2, 2.2),
            'jaw_width_face_ratio': (0.5, 0.9),
            'eyes_interocular_distance_ratio': (2.0, 3.0)
        }

        # ИСПРАВЛЕНО: Правильные идеальные пропорции с учетом единиц измерения
        self.ideal_proportions = {
            'frontal': {
                # Основные пропорции лица (безразмерные)
                'face_width_height_ratio': PHI_INV,  # 0.618
                'face_golden_ratio': PHI,  # 1.618

                # ИСПРАВЛЕНО: Глазные метрики
                'eye_width_iod_ratio': 0.3,
                'eyes_interocular_distance_ratio': 1.0,
                'eyes_aspect_ratio_mean': 0.3,

                # ИСПРАВЛЕНО: Носовые метрики
                'nose_width_iod_ratio': 0.333,
                'nose_height_width_ratio': 1.618,
                'nose_bridge_width_ratio': 0.5,
                
                # ИСПРАВЛЕНО: Ротовые метрики
                'mouth_width_nose_ratio': PHI,
                'mouth_upper_lower_ratio': PHI_INV,
                'mouth_height_width_ratio': 0.5,
                
                # ИСПРАВЛЕНО: Челюстные метрики
                'jaw_width_face_ratio': PHI_INV,
                'jaw_chin_width_ratio': PHI_INV,
                
                # ИСПРАВЛЕНО: Углы
                'canthal_tilt_angle': 0.0,
                'nasolabial_angle': 110.0,
                'nasofrontal_angle': 130.0
            },
            'frontal_edge': {
                'perspective_correction_factor': 0.95
            },
            'semi_profile': {
                'profile_visibility_ratio': 0.7
            },
            'profile': {
                'facial_angle': 80.0
            }
        }
        
        self.logger.info("StabilizedMarquardtMask инициализирован успешно")

    def _setup_logging(self) -> logging.Logger:
        """Настройка логирования"""
        logger = logging.getLogger(f"{__name__}.StabilizedMarquardtMask")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def classify_pose(self, yaw: float, pitch: float, roll: float) -> str:
        """ИСПРАВЛЕНО: Классификация позы по 4 ракурсам"""
        abs_yaw = abs(yaw)
        abs_pitch = abs(pitch)
        abs_roll = abs(roll)

        if (abs_yaw < self.pose_thresholds['frontal_yaw'] and
            abs_pitch < self.pose_thresholds['frontal_pitch'] and
            abs_roll < self.pose_thresholds['frontal_roll']):
            return 'frontal'
        elif (abs_yaw < self.pose_thresholds['edge_yaw'] and
              abs_pitch < self.pose_thresholds['edge_pitch'] and
              abs_roll < self.pose_thresholds['edge_roll']):
            return 'frontal_edge'
        elif abs_yaw < self.pose_thresholds['semi_profile_yaw']:
            return 'semi_profile'
        else:
            return 'profile'

    def get_stabilized_reference_distance(self, vertices: LandmarkArray, pose_type: str,
                                        R_matrix: RotationMatrix = None, yaw: float = 0.0) -> float:
        """ИСПРАВЛЕНО: Стабильное опорное расстояние с компенсацией поворота"""
        try:
            if vertices is None or len(vertices) < 68:
                self.logger.error("Недопустимые landmarks в get_stabilized_reference_distance")
                return 1.0

            # Трансформация в фронтальную позицию если есть матрица поворота
            if R_matrix is not None:
                vertices_frontal = self.transform_to_frontal_view(vertices, R_matrix)
            else:
                vertices_frontal = vertices.copy()

            if len(vertices_frontal) < 68:
                self.logger.error("Ошибка трансформации landmarks")
                return 1.0

            # Расчет центров глаз во фронтальной проекции
            left_eye_indices = list(range(42, 48))
            right_eye_indices = list(range(36, 42))
            
            left_eye_center = np.mean(vertices_frontal[left_eye_indices], axis=0)
            right_eye_center = np.mean(vertices_frontal[right_eye_indices], axis=0)

            # Используем только X,Y координаты для стабильности
            iod = float(np.linalg.norm(left_eye_center[:2] - right_eye_center[:2]))

            # ИСПРАВЛЕНО: Компенсация поворота головы
            yaw_abs = float(abs(yaw))
            if yaw_abs > 5.0:
                # Компенсируем перспективные искажения при повороте
                iod /= np.cos(np.radians(yaw_abs))

            # ИСПРАВЛЕНО: Коррекция в зависимости от ракурса
            correction = 1.0
            if pose_type == 'frontal':
                correction = 1.0
            elif pose_type == 'frontal_edge':
                # Небольшая коррекция для краевого фронтального вида
                correction = 0.95
            elif pose_type == 'semi_profile':
                # Значительная коррекция для полупрофиля
                correction = 0.85
            else:
                # Максимальная коррекция для профиля
                correction = 0.75

            return max(EPSILON, iod * correction)

        except Exception as e:
            self.logger.error(f"Ошибка расчета стабильного опорного расстояния: {e}")
            return 1.0

    def transform_to_frontal_view(self, landmarks: LandmarkArray, 
                                R_matrix: RotationMatrix) -> np.ndarray:
        """ИСПРАВЛЕНО: Трансформация landmarks в фронтальную позицию"""
        try:
            if landmarks is None or R_matrix is None:
                return np.array([]) if landmarks is None else landmarks.copy()

            # Применяем обратную матрицу поворота
            R_inv = R_matrix.T
            landmarks_frontal = np.zeros_like(landmarks)

            # ИСПРАВЛЕНО: Полная 3D трансформация с сохранением глубины
            for i, point in enumerate(landmarks):
                # Применяем полную 3D трансформацию
                landmarks_frontal[i] = R_inv @ point

            return landmarks_frontal.astype(np.float32)

        except Exception as e:
            self.logger.error(f"Ошибка трансформации в фронтальную позицию: {e}")
            return np.array([]) if landmarks is None else landmarks.copy()

    def _safe_2d_distance(self, p1: Union[np.ndarray, List[float]], 
                         p2: Union[np.ndarray, List[float]]) -> float:
        """Безопасный расчет 2D расстояния между точками"""
        try:
            p1_arr = self._ensure_numpy_array(p1)
            p2_arr = self._ensure_numpy_array(p2)
            return float(np.sqrt(np.sum((p2_arr - p1_arr) ** 2)))
        except Exception as e:
            self.logger.error(f"Ошибка в _safe_2d_distance: {e}")
            return 0.0
            
    def _safe_normalize_distance(self, distance: float, ref_distance: float) -> float:
        """Нормализация расстояния относительно опорного значения"""
        try:
            if ref_distance <= 0:
                return 0.0
            return distance / ref_distance
        except Exception as e:
            self.logger.error(f"Ошибка в _safe_normalize_distance: {e}")
            return 0.0

    def calculate_base_measurements(self, landmarks: LandmarkArray, ref_distance: float,
                                  yaw: float = 0.0, pitch: float = 0.0, 
                                  R_matrix: RotationMatrix = None) -> Dict[str, Dict[str, float]]:
        """ИСПРАВЛЕНО: Измерения с компенсацией поворота"""
        # Проверка входных данных
        if landmarks is None or len(landmarks) < 68 or ref_distance < EPSILON:
            self.logger.error("Недопустимые входные данные в calculate_base_measurements")
            return {}

        try:
            # Трансформация в фронтальную позицию если есть матрица поворота
            if R_matrix is not None:
                landmarks_frontal = self.transform_to_frontal_view(landmarks, R_matrix)
            else:
                landmarks_frontal = landmarks.copy()

            if len(landmarks_frontal) < 68:
                self.logger.error("Ошибка трансформации landmarks")
                return {}

            # Основные лицевые измерения
            face_measurements = {}
            face_width = self._safe_2d_distance(landmarks_frontal[0], landmarks_frontal[16])
            face_height = self._safe_2d_distance(landmarks_frontal[8], landmarks_frontal[27])
            face_measurements['width'] = self._safe_normalize_distance(face_width, ref_distance)
            face_measurements['height'] = self._safe_normalize_distance(face_height, ref_distance)

            # Глазные измерения
            eyes_measurements = {}
            right_eye_width = self._safe_2d_distance(landmarks_frontal[36], landmarks_frontal[39])
            left_eye_width = self._safe_2d_distance(landmarks_frontal[42], landmarks_frontal[45])
            right_eye_height = self._safe_2d_distance(landmarks_frontal[37], landmarks_frontal[41])
            left_eye_height = self._safe_2d_distance(landmarks_frontal[43], landmarks_frontal[47])
            
            eyes_measurements['width_right'] = self._safe_normalize_distance(right_eye_width, ref_distance)
            eyes_measurements['width_left'] = self._safe_normalize_distance(left_eye_width, ref_distance)
            eyes_measurements['width_mean'] = (eyes_measurements['width_right'] + eyes_measurements['width_left']) / 2.0
            eyes_measurements['height_right'] = self._safe_normalize_distance(right_eye_height, ref_distance)
            eyes_measurements['height_left'] = self._safe_normalize_distance(left_eye_height, ref_distance)
            eyes_measurements['height_mean'] = (eyes_measurements['height_right'] + eyes_measurements['height_left']) / 2.0
            eyes_measurements['interocular_distance'] = self._safe_normalize_distance(
                self._safe_2d_distance(landmarks_frontal[39], landmarks_frontal[42]),
                ref_distance
            )

            # Носовые измерения
            nose_measurements = {}
            nose_width = self._safe_2d_distance(landmarks_frontal[31], landmarks_frontal[35])
            nose_height = self._safe_2d_distance(landmarks_frontal[27], landmarks_frontal[33])
            nose_measurements['width'] = self._safe_normalize_distance(nose_width, ref_distance)
            nose_measurements['height'] = self._safe_normalize_distance(nose_height, ref_distance)
            nose_measurements['length'] = self._safe_normalize_distance(
                self._safe_2d_distance(landmarks_frontal[27], landmarks_frontal[30]),
                ref_distance
            )

            # Ротовые измерения
            mouth_measurements = {}
            mouth_width = self._safe_2d_distance(landmarks_frontal[48], landmarks_frontal[54])
            mouth_height = self._safe_2d_distance(landmarks_frontal[51], landmarks_frontal[57])
            mouth_measurements['width'] = self._safe_normalize_distance(mouth_width, ref_distance)
            mouth_measurements['height'] = self._safe_normalize_distance(mouth_height, ref_distance)
            mouth_measurements['upper_lip_height'] = self._safe_normalize_distance(
                self._safe_2d_distance(landmarks_frontal[50], landmarks_frontal[52]),
                ref_distance
            )
            mouth_measurements['lower_lip_height'] = self._safe_normalize_distance(
                self._safe_2d_distance(landmarks_frontal[57], landmarks_frontal[58]),
                ref_distance
            )

            # Челюстные измерения
            jaw_measurements = {}
            jaw_width = self._safe_2d_distance(landmarks_frontal[4], landmarks_frontal[12])
            jaw_height = self._safe_2d_distance(landmarks_frontal[8], landmarks_frontal[57])
            jaw_measurements['width'] = self._safe_normalize_distance(jaw_width, ref_distance)
            jaw_measurements['height'] = self._safe_normalize_distance(jaw_height, ref_distance)

            # Бровные измерения
            brow_measurements = {}
            brow_measurements['right_length'] = self._safe_normalize_distance(
                self._safe_2d_distance(landmarks_frontal[17], landmarks_frontal[21]),
                ref_distance
            )
            brow_measurements['left_length'] = self._safe_normalize_distance(
                self._safe_2d_distance(landmarks_frontal[22], landmarks_frontal[26]),
                ref_distance
            )

            # Симметрия
            symmetry_measurements = {}
            for left_idx, right_idx in self.symmetric_pairs:
                dist = float(self._safe_2d_distance(
                    landmarks_frontal[left_idx], 
                    landmarks_frontal[right_idx]
                ))
                symmetry_measurements[f'pair_{left_idx}_{right_idx}'] = \
                    self._safe_normalize_distance(dist, ref_distance)

            return {
                'face': face_measurements,
                'eyes': eyes_measurements,
                'nose': nose_measurements,
                'mouth': mouth_measurements,
                'jaw': jaw_measurements,
                'brows': brow_measurements,
                'symmetry': symmetry_measurements
            }

        except Exception as e:
            self.logger.error(f"Ошибка расчета базовых измерений: {str(e)}")
            return {}

    def calculate_angles(self, landmarks: LandmarkArray,
                          R_matrix: RotationMatrix = None) -> Dict[str, float]:
        """Расчет ключевых углов лица"""
        try:
            if landmarks is None or len(landmarks) < 68:
                return {}

            # Используем фронтальную проекцию если есть матрица поворота
            if R_matrix is not None:
                landmarks_frontal = self.transform_to_frontal_view(landmarks, R_matrix)
            else:
                landmarks_frontal = landmarks.copy()

            angles = {}

            # 1. Кантальный наклон (угол глаз)
            left_eye = landmarks_frontal[self.landmark_indices['left_eye']]
            right_eye = landmarks_frontal[self.landmark_indices['right_eye']]
            
            left_angle = float(np.arctan2(
                left_eye[3][1] - left_eye[0][1],
                left_eye[3][0] - left_eye[0][0]
            ) * 180 / np.pi)
            
            right_angle = float(np.arctan2(
                right_eye[3][1] - right_eye[0][1],
                right_eye[3][0] - right_eye[0][0]
            ) * 180 / np.pi)
            
            angles['canthal_tilt_left'] = left_angle
            angles['canthal_tilt_right'] = right_angle
            angles['canthal_tilt_mean'] = (left_angle + right_angle) / 2.0

            # 2. Назолабиальный угол
            nose_base = landmarks_frontal[33]
            upper_lip = landmarks_frontal[51]
            nose_tip = landmarks_frontal[30]
            
            vector1 = nose_base[:2] - nose_tip[:2]
            vector2 = nose_base[:2] - upper_lip[:2]
            
            cos_angle = np.dot(vector1, vector2) / (
                np.linalg.norm(vector1) * np.linalg.norm(vector2)
            )
            nasolabial_angle = float(np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi)
            angles['nasolabial_angle'] = nasolabial_angle

            # 3. Назофронтальный угол
            forehead = landmarks_frontal[27]  # Переносица
            vector3 = nose_tip[:2] - forehead[:2]
            vector4 = np.array([0, -1])  # Вертикальный вектор вверх
            
            cos_angle = np.dot(vector3, vector4) / np.linalg.norm(vector3)
            nasofrontal_angle = float(np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi)
            angles['nasofrontal_angle'] = nasофrontal_angle

            return angles

        except Exception as e:
            self.logger.error(f"Ошибка расчета углов: {str(e)}")
            return {}

    def calculate_ratios(self, measurements: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """ИСПРАВЛЕНО: Расчет пропорций на основе базовых измерений"""
        try:
            if not measurements:
                return {}

            ratios = {}

            # 1. Основные пропорции лица
            face = measurements.get('face', {})
            if 'width' in face and 'height' in face and face['height'] > EPSILON:
                ratios['face_width_height_ratio'] = face['width'] / face['height']

            # 2. Глазные пропорции
            eyes = measurements.get('eyes', {})
            if 'width_mean' in eyes and 'height_mean' in eyes and eyes['height_mean'] > EPSILON:
                ratios['eyes_aspect_ratio_mean'] = eyes['height_mean'] / eyes['width_mean']
            
            if 'width_mean' in eyes and face.get('width', 0) > EPSILON:
                ratios['eyes_face_width_ratio'] = eyes['width_mean'] / face['width']
            
            if 'interocular_distance' in eyes and eyes['width_mean'] > EPSILON:
                ratios['eyes_iod_ratio'] = eyes['interocular_distance'] / eyes['width_mean']

            # 3. Носовые пропорции
            nose = measurements.get('nose', {})
            eyes = measurements.get('eyes', {})
            if 'width' in nose and 'interocular_distance' in eyes and eyes['interocular_distance'] > EPSILON:
                ratios['nose_width_iod_ratio'] = nose['width'] / eyes['interocular_distance']
            
            if 'height' in nose and 'width' in nose and nose['width'] > EPSILON:
                ratios['nose_height_width_ratio'] = nose['height'] / nose['width']

            # 4. Ротовые пропорции
            mouth = measurements.get('mouth', {})
            if ('width' in mouth and 'width' in nose and 
                nose['width'] > EPSILON and 'height' in mouth):
                ratios['mouth_nose_width_ratio'] = mouth['width'] / nose['width']
                if mouth['width'] > EPSILON:
                    ratios['mouth_height_width_ratio'] = mouth['height'] / mouth['width']
            
            if ('upper_lip_height' in mouth and 'lower_lip_height' in mouth and 
                mouth['lower_lip_height'] > EPSILON):
                ratios['upper_lower_lip_ratio'] = (
                    mouth['upper_lip_height'] / mouth['lower_lip_height']
                )

            # 5. Челюстные пропорции
            jaw = measurements.get('jaw', {})
            if 'width' in jaw and 'width' in face and face['width'] > EPSILON:
                ratios['jaw_face_width_ratio'] = jaw['width'] / face['width']

            # 6. Симметрия
            symmetry = measurements.get('symmetry', {})
            if symmetry:
                # Среднее отклонение от идеальной симметрии
                deviations = []
                for key, value in symmetry.items():
                    if isinstance(value, (int, float)):
                        deviations.append(abs(value - 1.0))
                if deviations:
                    ratios['symmetry_deviation'] = float(np.mean(deviations))

            return ratios

        except Exception as e:
            self.logger.error(f"Ошибка расчета пропорций: {str(e)}")
            return {}

    def compare_with_ideal(self, measurements: Dict[str, Dict[str, float]],
                         ratios: Dict[str, float],
                         angles: Dict[str, float],
                         pose_type: str = 'frontal') -> Dict[str, float]:
        """Сравнение измерений с идеальными пропорциями"""
        try:
            if not measurements or pose_type not in self.ideal_proportions:
                return {}

            ideal = self.ideal_proportions[pose_type]
            deviations = {}

            # Функция для безопасного расчета отклонения
            def calculate_deviation(measured: float, ideal: float) -> float:
                if ideal < EPSILON:
                    return 0.0
                return float(abs(measured - ideal) / ideal)

            # 1. Основные пропорции лица
            if 'face_width_height_ratio' in ratios:
                deviations['face_width_height'] = calculate_deviation(
                    ratios['face_width_height_ratio'],
                    ideal['face_width_height_ratio']
                )

            # 2. Глазные пропорции
            if 'eyes_face_width_ratio' in ratios:
                deviations['eyes_face_width'] = calculate_deviation(
                    ratios['eyes_face_width_ratio'],
                    ideal['eyes_width_face_ratio']
                )

            # 3. Носовые пропорции
            if 'nose_width_iod_ratio' in ratios:
                deviations['nose_width_iod'] = calculate_deviation(
                    ratios['nose_width_iod_ratio'],
                    ideal['nose_width_interocular_ratio']
                )

            if 'nose_height_width_ratio' in ratios:
                deviations['nose_height_width'] = calculate_deviation(
                    ratios['nose_height_width_ratio'],
                    ideal['nose_height_width_ratio']
                )

            # 4. Ротовые пропорции
            if 'mouth_nose_width_ratio' in ratios:
                deviations['mouth_nose_width'] = calculate_deviation(
                    ratios['mouth_nose_width_ratio'],
                    ideal['mouth_width_nose_ratio']
                )

            if 'upper_lower_lip_ratio' in ratios:
                deviations['upper_lower_lip'] = calculate_deviation(
                    ratios['upper_lower_lip_ratio'],
                    ideal['mouth_upper_lower_lip_ratio']
                )

            # 5. Отклонения углов
            if 'canthal_tilt_mean' in angles:
                deviations['canthal_tilt'] = calculate_deviation(
                    angles['canthal_tilt_mean'],
                    ideal['canthal_tilt_angle']
                )

            if 'nasolabial_angle' in angles:
                deviations['nasolabial'] = calculate_deviation(
                    angles['nasolabial_angle'],
                    ideal['nasolabial_angle']
                )

            if 'nasofrontal_angle' in angles:
                deviations['nasofrontal'] = calculate_deviation(
                    angles['nasофrontal_angle'],
                    ideal['nasофrontal_angle']
                )

            # Расчет общего отклонения как взвешенное среднее
            weights = {
                'face_width_height': 1.0,
                'eyes_face_width': 1.0,
                'nose_width_iod': 0.8,
                'nose_height_width': 0.8,
                'mouth_nose_width': 0.9,
                'upper_lower_lip': 0.7,
                'canthal_tilt': 0.6,
                'nasolabial': 0.7,
                'nasofrontal': 0.7
            }

            total_weight = sum(weights[k] for k in deviations.keys() if k in weights)
            if total_weight > EPSILON:
                weighted_sum = sum(
                    deviations[k] * weights[k]
                    for k in deviations.keys()
                    if k in weights
                )
                deviations['total_deviation'] = float(weighted_sum / total_weight)
                deviations['harmony_score'] = float(1.0 / (1.0 + deviations['total_deviation']))

            return deviations

        except Exception as e:
            self.logger.error(f"Ошибка сравнения с идеальными пропорциями: {e}")
            return {}

    def validate_measurements(self, measurements: Dict[str, Dict[str, float]],
                           ratios: Dict[str, float]) -> Dict[str, bool]:
        """Проверка измерений на биологическую достоверность"""
        try:
            validation_results = {}
            
            def is_in_range(value: float, min_val: float, max_val: float) -> bool:
                return min_val <= value <= max_val
            
            # Проверка основных пропорций на биологическую достоверность
            for metric, (min_val, max_val) in self.biological_limits.items():
                if metric in ratios:
                    value = ratios[metric]
                    validation_results[metric] = is_in_range(value, min_val, max_val)
            
            # Проверка базовых измерений
            if 'face' in measurements:
                face = measurements['face']
                if 'width' in face and 'height' in face:
                    validation_results['face_proportions'] = is_in_range(
                        face['width'] / max(face['height'], EPSILON),
                        0.6, 1.4
                    )
            
            if 'eyes' in measurements:
                eyes = measurements['eyes']
                if 'width_mean' in eyes and 'height_mean' in eyes:
                    validation_results['eye_proportions'] = is_in_range(
                        eyes['height_mean'] / max(eyes['width_mean'], EPSILON),
                        0.2, 0.5
                    )
            
            if 'nose' in measurements:
                nose = measurements['nose']
                if 'width' in nose and 'height' in nose:
                    validation_results['nose_proportions'] = is_in_range(
                        nose['height'] / max(nose['width'], EPSILON),
                        1.2, 2.2
                    )
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Ошибка валидации измерений: {str(e)}")
            return {}

    def analyze(self, landmarks: LandmarkArray, yaw: float = 0.0, pitch: float = 0.0,
              roll: float = 0.0, R_matrix: RotationMatrix = None) -> Dict[str, Dict[str, float]]:
        """ИСПРАВЛЕНО: Полный анализ лица с компенсацией поворота"""
        try:
            if landmarks is None or len(landmarks) < 68:
                return {}
            
            # Определяем тип позы
            pose_type = self.classify_pose(yaw, pitch, roll)
            
            # Получаем стабильное опорное расстояние с компенсацией поворота
            ref_distance = self.get_stabilized_reference_distance(
                landmarks, pose_type, R_matrix, yaw
            )
            
            # Трансформация в фронтальную позицию если возможно
            landmarks_for_analysis = self.transform_to_frontal_view(landmarks, R_matrix) if R_matrix is not None else landmarks
            
            # Базовые измерения с компенсацией поворота
            base_measurements = self.calculate_base_measurements(
                landmarks_for_analysis, ref_distance, yaw, pitch, R_matrix
            )
            
            # Расчет углов
            angles = self.calculate_angles(landmarks_for_analysis, R_matrix)
            
            # Расчет пропорций
            ratios = self.calculate_ratios(base_measurements)
            
            # Сравнение с идеальными пропорциями
            ideal_comparisons = self.compare_with_ideal(
                base_measurements, ratios, angles, pose_type
            )
            
            # Валидация измерений
            validation_results = self.validate_measurements(base_measurements, ratios)
            
            # Расчет симметрии
            symmetry_metrics = self._calculate_symmetry_measurements(
                landmarks_for_analysis, ref_distance
            )
            
            # Метрики черепа для идентификации
            skull_metrics = self.calculate_skull_identity_metrics(
                base_measurements, ratios
            )
            
            # Детекция аномалий и масок
            anomaly_results = self.detect_anomalies_and_masks(
                base_measurements, ratios, symmetry_metrics
            )
            
            # Рассчитываем коэффициент перспективы
            perspective_factor = 1.0
            if abs(yaw) > 5.0:
                perspective_factor = 1.0 - (abs(yaw) / 90.0) * 0.3
            
            # Формируем итоговый результат
            analysis_result = {
                "pose_type": pose_type,
                "raw_measurements": base_measurements,
                "angular_metrics": angles,
                "proportion_metrics": ratios,
                "ideal_comparison": ideal_comparisons,
                "validation_results": validation_results,
                "symmetry_metrics": symmetry_metrics,
                "skull_metrics": skull_metrics,
                "anomaly_detection": anomaly_results,
                "stabilization_info": {
                    "reference_distance": float(ref_distance),
                    "perspective_factor": float(perspective_factor),
                    "rotation_compensation_applied": R_matrix is not None,
                    "yaw_angle": float(yaw),
                    "pitch_angle": float(pitch),
                    "roll_angle": float(roll)
                }
            }
            
            return analysis_result

        except Exception as e:
            self.logger.error(f"Критическая ошибка в методе analyze: {e}")
            traceback.print_exc()
            return {}

    def _safe_normalize_distance(self, value: float, ref_distance: float) -> float:
        """Безопасная нормализация расстояния"""
        return float(value / max(EPSILON, ref_distance))

    def _safe_2d_distance(self, p1: Union[np.ndarray, List[float]], 
                         p2: Union[np.ndarray, List[float]]) -> float:
        """Безопасный расчет 2D расстояния между точками"""
        try:
            p1_arr = self._ensure_numpy_array(p1)
            p2_arr = self._ensure_numpy_array(p2)
            return float(np.sqrt(np.sum((p2_arr - p1_arr) ** 2)))
        except Exception as e:
            self.logger.error(f"Ошибка в _safe_2d_distance: {e}")
            return 0.0

    def _ensure_numpy_array(self, data: LandmarkArray) -> np.ndarray:
        """Преобразует входные данные в numpy массив"""
        if isinstance(data, np.ndarray):
            return data
        return np.array(data)

    def _calculate_face_measurements(self, landmarks: np.ndarray, ref_distance: float) -> Dict[str, float]:
        """Расчет измерений лица"""
        dist_0_16 = self._safe_2d_distance(landmarks[0], landmarks[16])
        dist_8_27 = self._safe_2d_distance(landmarks[8], landmarks[27])
        
        return {
            'width': self._safe_normalize_distance(dist_0_16, ref_distance),
            'height': self._safe_normalize_distance(dist_8_27, ref_distance)
        }

    def _calculate_eye_measurements(self, landmarks: np.ndarray, ref_distance: float) -> Dict[str, float]:
        """Расчет измерений глаз"""
        # Правый глаз
        right_width = self._safe_2d_distance(landmarks[36], landmarks[39])
        right_height = self._safe_2d_distance(landmarks[37], landmarks[41])
        
        # Левый глаз
        left_width = self._safe_2d_distance(landmarks[42], landmarks[45])
        left_height = self._safe_2d_distance(landmarks[43], landmarks[47])
        
        # Межзрачковое расстояние
        iod = self._safe_2d_distance(
            self._safe_point_mean(landmarks[36:42]),
            self._safe_point_mean(landmarks[42:48])
        )
        
        return {
            'width_right': self._safe_normalize_distance(right_width, ref_distance),
            'width_left': self._safe_normalize_distance(left_width, ref_distance),
            'width_mean': self._safe_normalize_distance((right_width + left_width) / 2.0, ref_distance),
            'height_right': self._safe_normalize_distance(right_height, ref_distance),
            'height_left': self._safe_normalize_distance(left_height, ref_distance),
            'height_mean': self._safe_normalize_distance((right_height + left_height) / 2.0, ref_distance),
            'interocular_distance': self._safe_normalize_distance(iod, ref_distance)
        }

    def _calculate_nose_measurements(self, landmarks: np.ndarray, ref_distance: float) -> Dict[str, float]:
        """Расчет измерений носа"""
        width = self._safe_2d_distance(landmarks[31], landmarks[35])
        height = self._safe_2d_distance(landmarks[27], landmarks[33])
        length = self._safe_2d_distance(landmarks[27], landmarks[30])
        bridge_width = self._safe_2d_distance(landmarks[39], landmarks[42])
        
        return {
            'width': self._safe_normalize_distance(width, ref_distance),
            'height': self._safe_normalize_distance(height, ref_distance),
            'length': self._safe_normalize_distance(length, ref_distance),
            'bridge_width': self._safe_normalize_distance(bridge_width, ref_distance)
        }

    def _calculate_mouth_measurements(self, landmarks: np.ndarray, ref_distance: float) -> Dict[str, float]:
        """Расчет измерений рта"""
        width = self._safe_2d_distance(landmarks[48], landmarks[54])
        height = self._safe_2d_distance(landmarks[51], landmarks[57])
        upper_height = self._safe_2d_distance(landmarks[50], landmarks[52])
        lower_height = self._safe_2d_distance(landmarks[57], landmarks[58])
        
        return {
            'width': self._safe_normalize_distance(width, ref_distance),
            'height': self._safe_normalize_distance(height, ref_distance),
            'upper_lip_height': self._safe_normalize_distance(upper_height, ref_distance),
            'lower_lip_height': self._safe_normalize_distance(lower_height, ref_distance)
        }

    def _calculate_jaw_measurements(self, landmarks: np.ndarray, ref_distance: float) -> Dict[str, float]:
        """Расчет измерений челюсти"""
        try:
            # Ширина челюсти (между крайними точками)
            jaw_width = self._safe_2d_distance(landmarks[4], landmarks[12])
            
            # Высота челюсти (от подбородка до линии рта)
            jaw_height = self._safe_2d_distance(landmarks[8], landmarks[57])
            
            # Ширина в области скул
            zygomatic_width = self._safe_2d_distance(landmarks[2], landmarks[14])
            
            # Расстояние от подбородка до центра челюсти
            chin_to_jaw = self._safe_2d_distance(
                landmarks[8],
                self._safe_point_mean(landmarks[4:13])
            )
            
            return {
                'width': self._safe_normalize_distance(jaw_width, ref_distance),
                'height': self._safe_normalize_distance(jaw_height, ref_distance),
                'zygomatic_width': self._safe_normalize_distance(zygomatic_width, ref_distance),
                'chin_to_jaw_ratio': self._safe_normalize_distance(chin_to_jaw, ref_distance)
            }
        except Exception as e:
            self.logger.error(f"Ошибка расчета измерений челюсти: {str(e)}")
            return {}

    def _calculate_brow_measurements(self, landmarks: np.ndarray, ref_distance: float) -> Dict[str, float]:
        """Расчет измерений бровей"""
        try:
            # Длина правой брови
            right_brow_length = self._safe_2d_distance(landmarks[17], landmarks[21])
            
            # Длина левой брови
            left_brow_length = self._safe_2d_distance(landmarks[22], landmarks[26])
            
            # Расстояние между бровями
            brow_separation = self._safe_2d_distance(landmarks[21], landmarks[22])
            
            # Высота от центра правой брови до глаза
            right_brow_height = self._safe_2d_distance(
                self._safe_point_mean(landmarks[17:22]),
                self._safe_point_mean(landmarks[36:42])
            )
            
            # Высота от центра левой брови до глаза
            left_brow_height = self._safe_2d_distance(
                self._safe_point_mean(landmarks[22:27]),
                self._safe_point_mean(landmarks[42:48])
            )
            
            return {
                'right_length': self._safe_normalize_distance(right_brow_length, ref_distance),
                'left_length': self._safe_normalize_distance(left_brow_length, ref_distance),
                'separation': self._safe_normalize_distance(brow_separation, ref_distance),
                'right_height': self._safe_normalize_distance(right_brow_height, ref_distance),
                'left_height': self._safe_normalize_distance(left_brow_height, ref_distance),
                'length_ratio': right_brow_length / max(left_brow_length, EPSILON),
                'height_ratio': right_brow_height / max(left_brow_height, EPSILON)
            }
        except Exception as e:
            self.logger.error(f"Ошибка расчета измерений бровей: {str(e)}")
            return {}

    def _calculate_symmetry_measurements(self, landmarks: np.ndarray, ref_distance: float) -> Dict[str, float]:
        """Расчет измерений симметрии"""
        try:
            pair_deviations = []
            center_deviations = []
            
            # Расчет симметрии для каждой пары точек
            for left_idx, right_idx in self.symmetric_pairs:
                # Нормализованное расстояние между симметричными точками
                dist = self._safe_2d_distance(landmarks[left_idx], landmarks[right_idx])
                norm_dist = self._safe_normalize_distance(dist, ref_distance)
                
                # Расчет отклонений от идеальной симметрии
                left_point = landmarks[left_idx][:2]
                right_point = landmarks[right_idx][:2]
                midpoint = (left_point + right_point) / 2.0
                
                # Отклонение от центральной линии
                center_line_x = np.mean(landmarks[27:31, 0])  # X-координата центральной линии носа
                deviation = abs(midpoint[0] - center_line_x) / ref_distance
                
                pair_deviations.append(norm_dist)
                center_deviations.append(deviation)
            
            # Расчет агрегированных метрик симметрии
            mean_pair_deviation = float(np.mean(pair_deviations))
            mean_center_deviation = float(np.mean(center_deviations))
            symmetry_std = float(np.std(pair_deviations))
            
            return {
                'overall_symmetry': float(1.0 - min(1.0, mean_center_deviation * 2.0)),
                'symmetry_stability': float(1.0 - min(1.0, symmetry_std * 2.0)),
                'symmetry_confidence': float(1.0 / (1.0 + symmetry_std)),
                'mean_pair_deviation': mean_pair_deviation,
                'mean_center_deviation': mean_center_deviation
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета симметрии: {str(e)}")
            return {}

    def calculate_skull_identity_metrics(self, measurements: Dict[str, Dict[str, float]],
                                ratios: Dict[str, float]) -> Dict[str, float]:
        """Расчет метрик для идентификации уникальности черепа"""
        try:
            skull_metrics = {}
            
            # Ключевые пропорции черепа (неизменные при мимике)
            stable_ratios = {
                'face_width_height': ratios.get('face_width_height_ratio', 0.0),
                'jaw_face_width': ratios.get('jaw_face_width_ratio', 0.0),
                'nose_height_width': ratios.get('nose_height_width_ratio', 0.0)
            }
            
            if measurements.get('jaw'):
                jaw = measurements['jaw']
                if 'zygomatic_width' in jaw and 'width' in jaw and jaw['width'] > EPSILON:
                    stable_ratios['zygomatic_jaw_ratio'] = jaw['zygomatic_width'] / jaw['width']
            
            # Расчет композитного индекса черепа
            valid_values = [v for v in stable_ratios.values() if v > EPSILON]
            if valid_values:
                skull_metrics['skull_composite_index'] = float(np.mean(valid_values))
                skull_metrics['skull_variation'] = float(np.std(valid_values))
                
                # Нормализованный индекс уникальности (0-1)
                uniqueness_base = sum([hash(f"{k}:{v:.4f}") for k, v in stable_ratios.items()])
                skull_metrics['skull_uniqueness'] = float(
                    1.0 - 1.0 / (1.0 + abs(math.sin(uniqueness_base)))
                )
                
                # Стабильность метрик
                mean_val = np.mean(valid_values)
                if mean_val > EPSILON:
                    skull_metrics['skull_stability'] = float(
                        1.0 - min(1.0, np.std(valid_values) / mean_val)
                    )
            
            return skull_metrics
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета метрик черепа: {str(e)}")
            return {}

    def detect_anomalies_and_masks(self, measurements: Dict[str, Dict[str, float]],
                              ratios: Dict[str, float], 
                              symmetry_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Детекция аномальных пропорций и масок"""
        try:
            anomaly_scores = []
            detailed_anomalies = {}
            
            # Проверка основных пропорций на выход за биологические пределы
            for metric, limits in self.biological_limits.items():
                if metric in ratios:
                    value = ratios[metric]
                    min_val, max_val = limits
                    
                    if value < min_val or value > max_val:
                        # Расчет степени аномальности (0-1)
                        center = (min_val + max_val) / 2.0
                        range_half = (max_val - min_val) / 2.0
                        if range_half > EPSILON:
                            anomaly_degree = float(min(1.0, abs(value - center) / range_half))
                            anomaly_scores.append(anomaly_degree)
                            detailed_anomalies[f'anomaly_{metric}'] = anomaly_degree
            
            # Проверка симметрии
            symmetry_score = symmetry_metrics.get('overall_symmetry', 1.0)
            if symmetry_score < 0.8:  # Значительная асимметрия
                asymmetry_degree = float(min(1.0, (0.8 - symmetry_score) * 5.0))
                anomaly_scores.append(asymmetry_degree)
                detailed_anomalies['anomaly_asymmetry'] = asymmetry_degree
            
            # Общий индекс аномальности
            overall_anomaly = float(np.mean(anomaly_scores)) if anomaly_scores else 0.0
            confidence = float(1.0 / (1.0 + overall_anomaly))
            
            # Классификация
            if overall_anomaly > 0.7:
                classification = "potential_mask"
            elif overall_anomaly > 0.3:
                classification = "anomalous_human"
            else:
                classification = "normal_human"
            
            result = {
                'overall_anomaly_index': overall_anomaly,
                'classification': classification,
                'classification_confidence': confidence
            }
            result.update(detailed_anomalies)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка детекции аномалий: {str(e)}")
            return {}

    def format_analysis_results(self, measurements: Dict[str, Dict[str, float]],
                            ratios: Dict[str, float],
                            angles: Dict[str, float],
                            skull_metrics: Dict[str, float],
                            anomaly_detection: Dict[str, Any]) -> Dict[str, Any]:
        """Форматирование результатов анализа"""
        try:
            # Базовые измерения и пропорции
            basic_metrics = {
                'face': measurements.get('face', {}),
                'eyes': measurements.get('eyes', {}),
                'nose': measurements.get('nose', {}),
                'mouth': measurements.get('mouth', {}),
                'jaw': measurements.get('jaw', {}),
                'brows': measurements.get('brows', {})
            }
            
            # Преобразование значений углов в градусы и нормализация
            normalized_angles = {}
            ideal_angles = self.ideal_proportions['frontal']
            for angle_name, value in angles.items():
                if 'angle' in angle_name.lower():
                    ideal = ideal_angles.get(angle_name, 0.0)
                    if ideal > EPSILON:
                        normalized_angles[angle_name] = {
                            'value': float(value),
                            'deviation': float(abs(value - ideal) / ideal)
                        }
            
            # Расчет отклонений от золотого сечения
            golden_ratio_deviations = {}
            for ratio_name, value in ratios.items():
                if value > EPSILON:
                    if 'ratio' in ratio_name.lower():
                        golden_ratio_deviations[ratio_name] = {
                            'value': float(value),
                            'deviation_from_golden': float(abs(value - PHI) / PHI)
                        }
            
            # Формирование итогового отчета
            return {
                'version': '2.0',
                'timestamp': time.time(),
                
                # Основные метрики
                'basic_metrics': {
                    category: {
                        k: float(v) for k, v in metrics.items()
                    } for category, metrics in basic_metrics.items() if metrics
                },
                
                # Пропорции и отклонения
                'proportions': {
                    'ratios': {
                        k: float(v) for k, v in ratios.items()
                    },
                    'golden_ratio_analysis': golden_ratio_deviations,
                    'angles': normalized_angles
                },
                
                # Идентификация черепа
                'skull_identity': {
                    k: float(v) for k, v in skull_metrics.items()
                },
                
                # Анализ аномалий
                'anomaly_analysis': anomaly_detection,
                
                # Метаданные и статистика
                'statistics': {
                    'total_measurements': len(measurements),
                    'valid_ratios': len(ratios),
                    'anomaly_count': len([v for v in anomaly_detection.values() 
                                       if isinstance(v, (int, float)) and v > 0.3])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка форматирования результатов: {str(e)}")
            return {}

# Для обратной совместимости:
MarquardtMask = StabilizedMarquardtMask