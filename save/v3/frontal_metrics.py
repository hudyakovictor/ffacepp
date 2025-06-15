#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПОЛНОСТЬЮ ПЕРЕПИСАННЫЙ МОДУЛЬ АНАЛИЗА ФРОНТАЛЬНОГО РАКУРСА
Исправлены все критические ошибки согласно анализу результатов
Версия: 7.0 - Стабильная с компенсацией поворотов и масштабированием
"""
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import logging
from marquardt import MarquardtMask

# Константы
EPSILON = 1e-6
MIN_LANDMARKS_COUNT = 68
MIN_IOD = 0.1  # Минимальное межзрачковое расстояние
MAX_YAW_ANGLE = 45.0  # Максимальный угол поворота для надежного анализа

# Типы для аннотаций
LandmarkArray = Union[np.ndarray, None]
RotationMatrix = Union[np.ndarray, None]

class StabilizedFrontalAnalysisModule:
    """Стабилизированный модуль анализа фронтального ракурса"""
    
    def __init__(self):
        """Инициализация с улучшенными возможностями"""
        self.logger = self._setup_logging()
        
        # Инициализация улучшенной маски Марквардта
        self.marquardt_mask = MarquardtMask()
        
        # Биологические пределы для метрик
        self.biological_limits = {
            'face_width_height_ratio': (0.6, 1.4),
            'nose_width_face_ratio': (0.2, 0.4),
            'mouth_width_face_ratio': (0.4, 0.6),
            'eyes_distance_ratio': (0.4, 0.5),
            'jaw_width_face_ratio': (0.7, 0.9)
        }
        
        # Ключевые метрики черепа (неизменные при мимике)
        self.skull_metrics = [
            'face_width_height_ratio',
            'jaw_width_face_ratio',
            'zygomatic_width_ratio',
            'nose_bridge_ratio'
        ]

        # Симметричные пары точек сгруппированные по областям с весами
        self.symmetric_regions = {
            'jaw': [
                ((0, 16), 0.7),   # Челюсть - внешние точки
                ((4, 12), 0.8),   # Челюсть - углы
                ((2, 14), 0.6)    # Челюсть - промежуточные точки
            ],
            'brow': [
                ((17, 26), 0.8),  # Брови - внешние края
                ((19, 24), 0.9)   # Брови - центр
            ],
            'eye': [
                ((36, 45), 1.0),  # Глаза - внешние уголки
                ((39, 42), 1.0)   # Глаза - внутренние уголки
            ],
            'nose': [
                ((31, 35), 0.9)   # Нос - основание
            ],
            'mouth': [
                ((48, 54), 0.8),  # Рот - уголки
                ((51, 57), 0.8)   # Рот - центральные точки
            ]
        }
        
        self.logger.info("StabilizedFrontalAnalysisModule инициализирован успешно")

    def _setup_logging(self) -> logging.Logger:
        """Настройка логирования"""
        logger = logging.getLogger(f"{__name__}.StabilizedFrontalAnalysisModule")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def transform_to_canonical_pose(self, landmarks_3d: LandmarkArray, 
                               R_matrix: RotationMatrix) -> np.ndarray:
        """Трансформация в каноническую фронтальную позицию"""
        try:
            if landmarks_3d is None or R_matrix is None:
                return landmarks_3d if landmarks_3d is not None else np.array([])
            
            # Применяем обратную матрицу поворота для компенсации позы головы
            R_canonical = R_matrix.T
            canonical_landmarks = np.zeros_like(landmarks_3d)
            
            for i, point_3d in enumerate(landmarks_3d):
                # Полная 3D трансформация с сохранением глубины
                canonical_landmarks[i] = R_canonical @ point_3d
            
            return canonical_landmarks.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Ошибка трансформации в каноническую позицию: {e}")
            return np.array([]) if landmarks_3d is None else landmarks_3d

    def get_stabilized_reference_distance(self, landmarks_3d: LandmarkArray, 
                                        R_matrix: RotationMatrix, 
                                        s_scale: float) -> float:
        """Стабильное опорное расстояние с компенсацией поворота"""
        try:
            if landmarks_3d is None or len(landmarks_3d) < 48:
                return 1.0
            
            # Трансформируем в фронтальную позицию если есть матрица поворота
            if R_matrix is not None:
                frontal_landmarks = self.transform_to_canonical_pose(landmarks_3d, R_matrix)
            else:
                frontal_landmarks = landmarks_3d
            
            # Расчет центров глаз во фронтальной проекции
            left_eye_center = np.mean(frontal_landmarks[42:48], axis=0)
            right_eye_center = np.mean(frontal_landmarks[36:42], axis=0)
            
            # Используем только X,Y координаты для стабильности
            iod = float(np.linalg.norm(left_eye_center[:2] - right_eye_center[:2]))
            
            # Применяем масштабную коррекцию
            ref_distance = iod * s_scale
            
            return max(EPSILON, ref_distance)
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета опорного расстояния: {e}")
            return 1.0

    def calculate_stabilized_proportions(self, landmarks_3d: LandmarkArray, 
                                       R_matrix: RotationMatrix, 
                                       s_scale: float, 
                                       ref_distance: float) -> Dict[str, float]:
        """Расчет стабилизированных относительных пропорций"""
        try:
            if landmarks_3d is None:
                return {}
            
            # Трансформируем landmarks в каноническую позицию если есть матрица поворота
            if R_matrix is not None:
                canonical_landmarks = self.transform_to_canonical_pose(landmarks_3d, R_matrix)
            else:
                canonical_landmarks = landmarks_3d
            
            # Все измерения нормализуются относительно IOD
            def normalize_distance(p1_idx: int, p2_idx: int) -> float:
                dist = np.linalg.norm(canonical_landmarks[p1_idx][:2] - 
                                    canonical_landmarks[p2_idx][:2])
                return dist / ref_distance
            
            proportions = {}
            
            # Основные пропорции лица (безразмерные коэффициенты)
            face_width = normalize_distance(0, 16)  # Челюсть
            face_height = normalize_distance(27, 8)  # Переносица-подбородок
            proportions['face_width_height_ratio'] = face_width / face_height if face_height > EPSILON else 0.0
            
            # Глазные пропорции
            right_eye_width = normalize_distance(36, 39)
            left_eye_width = normalize_distance(42, 45)
            eye_width_mean = (right_eye_width + left_eye_width) / 2.0
            proportions['eye_width_iod_ratio'] = eye_width_mean  # Относительно IOD
            
            # Носовые пропорции
            nose_width = normalize_distance(31, 35)
            nose_height = normalize_distance(27, 33)
            proportions['nose_width_iod_ratio'] = nose_width
            proportions['nose_height_width_ratio'] = nose_height / nose_width if nose_width > EPSILON else 0.0
            
            # Добавляем метрики черепа
            skull_metrics = self._calculate_skull_ratios(canonical_landmarks, ref_distance)
            proportions.update(skull_metrics)
            
            return proportions
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета стабилизированных пропорций: {e}")
            return {}

    def _calculate_core_proportions(self, landmarks: np.ndarray, ref_distance: float) -> Dict[str, float]:
        """Расчет основных пропорций на стабилизированных landmarks"""
        try:
            def safe_ratio(num: float, den: float) -> float:
                """Безопасный расчет отношения с проверкой на ноль"""
                if abs(den) < EPSILON:
                    return 0.0
                ratio = num / den
                return float(ratio) if np.isfinite(ratio) else 0.0

            def point_distance_2d(p1_idx: int, p2_idx: int) -> float:
                """Расчет расстояния между точками в 2D"""
                dist = float(np.linalg.norm(landmarks[p1_idx, :2] - landmarks[p2_idx, :2]))
                return dist

            proportions = {}

            # 1. Базовые размеры лица (нормализованные)
            face_width = point_distance_2d(0, 16) / ref_distance
            face_height = point_distance_2d(8, 27) / ref_distance
            proportions['face_width_height_ratio'] = safe_ratio(face_width, face_height)

            # 2. Глазные пропорции
            right_eye_width = point_distance_2d(36, 39) / ref_distance
            left_eye_width = point_distance_2d(42, 45) / ref_distance
            eye_width_avg = (right_eye_width + left_eye_width) / 2.0
            proportions['eye_width_ratio'] = eye_width_avg

            # Межзрачковое расстояние уже нормализовано (равно 1.0)
            eye_separation = point_distance_2d(39, 42) / ref_distance
            proportions['eyes_separation_ratio'] = eye_separation

            # 3. Носовые пропорции
            nose_width = point_distance_2d(31, 35) / ref_distance
            nose_height = point_distance_2d(27, 33) / ref_distance
            nose_length = point_distance_2d(27, 30) / ref_distance

            proportions['nose_width_face_ratio'] = safe_ratio(nose_width, face_width)
            proportions['nose_height_width_ratio'] = safe_ratio(nose_height, nose_width)
            proportions['nose_length_height_ratio'] = safe_ratio(nose_length, face_height)

            # 4. Ротовые пропорции
            mouth_width = point_distance_2d(48, 54) / ref_distance
            mouth_height = point_distance_2d(51, 57) / ref_distance

            proportions['mouth_width_face_ratio'] = safe_ratio(mouth_width, face_width)
            proportions['mouth_height_width_ratio'] = safe_ratio(mouth_height, mouth_width)
            proportions['mouth_width_nose_ratio'] = safe_ratio(mouth_width, nose_width)

            # 5. Пропорции нижней части лица
            chin_height = point_distance_2d(8, 57) / ref_distance
            jaw_width = point_distance_2d(4, 12) / ref_distance

            proportions['chin_height_face_ratio'] = safe_ratio(chin_height, face_height)
            proportions['jaw_width_face_ratio'] = safe_ratio(jaw_width, face_width)

            return proportions

        except Exception as e:
            self.logger.error(f"Ошибка расчета основных пропорций: {e}")
            return {}

    def _calculate_skull_ratios(self, landmarks: np.ndarray, ref_distance: float) -> Dict[str, float]:
        """Расчет пропорций черепа (неизменные при мимике)"""
        try:
            if landmarks is None or len(landmarks) < 68:
                return {}

            def normalize_distance(p1_idx: int, p2_idx: int) -> float:
                try:
                    dist = np.linalg.norm(landmarks[p1_idx][:2] - landmarks[p2_idx][:2])
                    return float(dist / ref_distance) if ref_distance > EPSILON else 0.0
                except (IndexError, ValueError) as e:
                    self.logger.error(f"Ошибка при расчете расстояния между точками {p1_idx} и {p2_idx}: {e}")
                    return 0.0

            skull_proportions = {}

            # Скуловая ширина (между крайними точками скул)
            zygomatic_width = normalize_distance(0, 16)
            skull_proportions['zygomatic_width_ratio'] = zygomatic_width

            # Высота лба (от бровей до линии роста волос - аппроксимация)
            forehead_height = normalize_distance(19, 24)
            skull_proportions['forehead_height_ratio'] = forehead_height

            # Расстояние между внешними уголками глаз
            eye_corners_dist = normalize_distance(36, 45)
            skull_proportions['eye_corners_distance_ratio'] = eye_corners_dist

            # Ширина основания носа
            nose_base_width = normalize_distance(31, 35)
            skull_proportions['nose_base_width_ratio'] = nose_base_width

            # Расстояние между скуловыми точками
            cheekbone_dist = normalize_distance(2, 14)
            skull_proportions['cheekbone_width_ratio'] = cheekbone_dist

            return skull_proportions

        except Exception as e:
            self.logger.error(f"Ошибка расчета пропорций черепа: {e}")
            return {}

    def calculate_enhanced_symmetry_metrics(self, landmarks: LandmarkArray,
                                          ref_distance: float, 
                                          R_matrix: RotationMatrix = None) -> Dict[str, Any]:
        """Улучшенный расчет симметрии с компенсацией поворота и взвешиванием регионов"""
        try:
            if landmarks is None or len(landmarks) < 68:
                return {}

            # Трансформируем в каноническую позицию если есть матрица поворота
            canonical_landmarks = self.transform_to_canonical_pose(landmarks, R_matrix) if R_matrix is not None else landmarks

            # Инициализация результатов
            symmetry_metrics = {
                'overall_symmetry': 0.0,
                'regional_symmetry': {}
            }

            # Центральная линия лица (вертикаль через нос)
            nose_bridge = canonical_landmarks[27, :2]
            
            # Расчет симметрии по регионам
            total_weight = 0.0
            total_weighted_score = 0.0

            for region, pairs in self.symmetric_regions.items():
                region_scores = []
                region_total_weight = 0.0

                for (left_idx, right_idx), weight in pairs:
                    try:
                        # Координаты симметричных точек
                        left = canonical_landmarks[left_idx, :2]
                        right = canonical_landmarks[right_idx, :2]

                        # Расстояние до центральной линии
                        left_dist = np.abs(left[0] - nose_bridge[0])
                        right_dist = np.abs(right[0] - nose_bridge[0])

                        # Нормализованная асимметрия (0 - идеальная симметрия, 1 - максимальная асимметрия)
                        dist_diff = np.abs(left_dist - right_dist)
                        avg_dist = (left_dist + right_dist) / 2
                        if avg_dist > EPSILON:
                            asymmetry = dist_diff / avg_dist
                        else:
                            asymmetry = 0.0

                        # Ограничиваем значение асимметрии
                        asymmetry = min(1.0, asymmetry)
                        symmetry = 1.0 - asymmetry

                        region_scores.append(symmetry * weight)
                        region_total_weight += weight

                    except (IndexError, ValueError) as e:
                        self.logger.warning(f"Ошибка при расчете симметрии для пары {left_idx}-{right_idx}: {e}")
                        continue

                # Расчет средневзвешенной симметрии для региона
                if region_total_weight > EPSILON and region_scores:
                    region_symmetry = sum(region_scores) / region_total_weight
                    symmetry_metrics['regional_symmetry'][region] = float(region_symmetry)
                    total_weighted_score += region_symmetry * region_total_weight
                    total_weight += region_total_weight

            # Общая симметрия как взвешенное среднее по всем регионам
            if total_weight > EPSILON:
                symmetry_metrics['overall_symmetry'] = float(total_weighted_score / total_weight)

            return symmetry_metrics

        except Exception as e:
            self.logger.error(f"Ошибка при расчете симметрии: {e}")
            return {'overall_symmetry': 0.0, 'regional_symmetry': {}}

    def calculate_identity_metrics(self, landmarks: LandmarkArray,
                                ref_distance: float,
                                R_matrix: RotationMatrix = None) -> Dict[str, float]:
        """Расчет метрик идентичности на основе черепных пропорций"""
        try:
            if landmarks is None or len(landmarks) < 68:
                return {}

            # Трансформация в каноническую позицию если есть матрица поворота
            canonical_landmarks = self.transform_to_canonical_pose(landmarks, R_matrix) if R_matrix is not None else landmarks

            def safe_norm(a: np.ndarray, b: np.ndarray) -> float:
                """Безопасный расчет нормы с проверками"""
                try:
                    return float(np.linalg.norm(a - b))
                except:
                    return 0.0

            def safe_ratio(num: float, den: float) -> float:
                """Безопасный расчет отношения"""
                if abs(den) < EPSILON:
                    return 0.0
                return float(num / den)

            # Основные антропометрические точки и промеры
            identity_metrics = {}
            
            try:
                # Базовые размеры лица
                nasion = canonical_landmarks[27]  # Переносица
                gnathion = canonical_landmarks[8]  # Подбородок
                face_height = safe_norm(nasion, gnathion)

                zygion_l = canonical_landmarks[0]  # Левая скуловая
                zygion_r = canonical_landmarks[16]  # Правая скуловая
                face_width = safe_norm(zygion_l, zygion_r)

                gonion_l = canonical_landmarks[4]  # Левый угол челюсти
                gonion_r = canonical_landmarks[12]  # Правый угол челюсти
                jaw_width = safe_norm(gonion_l, gonion_r)

                # Нормализованные размеры
                face_height_norm = safe_ratio(face_height, ref_distance)
                face_width_norm = safe_ratio(face_width, ref_distance)
                jaw_width_norm = safe_ratio(jaw_width, ref_distance)

                # Основные пропорции
                facial_index = safe_ratio(face_height_norm, face_width_norm)
                jaw_face_ratio = safe_ratio(jaw_width_norm, face_width_norm)

                # Углы челюсти
                jaw_angle_l = self._calculate_angle(gonion_l, gnathion, nasion)
                jaw_angle_r = self._calculate_angle(gonion_r, gnathion, nasion)

                # Формирование основных метрик
                identity_metrics.update({
                    'facial_index': facial_index,
                    'jaw_face_ratio': jaw_face_ratio,
                    'jaw_angle_symmetry': float(1.0 - abs(jaw_angle_l - jaw_angle_r) / 180.0),
                    'face_width_norm': face_width_norm,
                    'face_height_norm': face_height_norm,
                    'jaw_width_norm': jaw_width_norm,
                    'left_jaw_angle': jaw_angle_l,
                    'right_jaw_angle': jaw_angle_r
                })

                # Дополнительные промеры
                eye_width_l = safe_norm(canonical_landmarks[36], canonical_landmarks[39])
                eye_width_r = safe_norm(canonical_landmarks[42], canonical_landmarks[45])
                nose_width = safe_norm(canonical_landmarks[31], canonical_landmarks[35])
                mouth_width = safe_norm(canonical_landmarks[48], canonical_landmarks[54])

                # Нормализованные дополнительные метрики
                eye_width_avg = (eye_width_l + eye_width_r) / 2.0
                identity_metrics.update({
                    'eye_ratio': safe_ratio(eye_width_avg, ref_distance),
                    'nose_ratio': safe_ratio(nose_width, ref_distance),
                    'mouth_ratio': safe_ratio(mouth_width, ref_distance),
                    'eye_symmetry': float(1.0 - abs(eye_width_l - eye_width_r) / max(eye_width_l, eye_width_r, EPSILON))
                })

            except Exception as e:
                self.logger.warning(f"Ошибка при расчете антропометрических точек: {e}")
                # ИНИЦИАЛИЗАЦИЯ переменных по умолчанию при ошибке
                jaw_angle_l = 0.0
                jaw_angle_r = 0.0
                face_width_norm = 1.0
                jaw_width_norm = 1.0
                face_height_norm = 1.0

            # Добавляем оценки стабильности черепа
            skull_stability = np.mean([
                1.0 - abs(jaw_angle_l - jaw_angle_r) / 180.0,  # Симметрия углов челюсти
                min(face_width_norm / jaw_width_norm if jaw_width_norm > EPSILON else 0.0, 1.0),  # Пропорции челюсти
                min(face_height_norm / face_width_norm if face_width_norm > EPSILON else 0.0, 1.0)  # Общие пропорции
            ])
            identity_metrics['skull_stability_index'] = float(skull_stability)

            return identity_metrics

        except Exception as e:
            self.logger.error(f"Ошибка расчета метрик идентичности: {e}")
            return {}

    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Расчет угла между тремя точками с проверками на корректность данных"""
        try:
            if p1 is None or p2 is None or p3 is None:
                return 0.0
                
            if not isinstance(p1, np.ndarray) or not isinstance(p2, np.ndarray) or not isinstance(p3, np.ndarray):
                return 0.0
                
            v1 = p1 - p2
            v2 = p3 - p2
            
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 < EPSILON or norm_v2 < EPSILON:
                return 0.0
                
            cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
            # Ограничиваем для численной стабильности
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            return float(np.degrees(np.arccos(cos_angle)))
            
        except Exception as e:
            self.logger.error(f"Ошибка при расчете угла: {e}")
            return 0.0

    def analyze(self, landmarks_3d: LandmarkArray, 
               yaw: float, 
               roll: float = 0.0,
               alpha_shp: Optional[np.ndarray] = None,
               alpha_exp: Optional[np.ndarray] = None,
               R_matrix: RotationMatrix = None, 
               t_vec: Optional[np.ndarray] = None,
               s_scale: float = 1.0,
               **kwargs) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Полный анализ лица с компенсацией поворотов. 
        Для анализа (метрики, симметрия и т.д.) используются нормализованные landmarks.
        Для визуализации (2d_sparse, 2d_dense) используйте исходные landmarks_3d до любых преобразований!
        Возвращает (результат, нормализованные landmarks).
        """
        try:
            if landmarks_3d is None or len(landmarks_3d) < MIN_LANDMARKS_COUNT:
                return self._create_error_result("Недостаточно landmarks"), np.zeros((0, 3), dtype=np.float32)

            # 1. Получаем стабильное опорное расстояние
            ref_distance = self.get_stabilized_reference_distance(
                landmarks_3d, R_matrix, s_scale
            )

            # 2. Рассчитываем стабилизированные пропорции
            stabilized_props = self.calculate_stabilized_proportions(
                landmarks_3d, R_matrix, s_scale, ref_distance
            )

            # 3. Анализ симметрии
            symmetry_metrics = self.calculate_enhanced_symmetry_metrics(
                landmarks_3d, ref_distance, R_matrix
            )

            # 4. Метрики идентичности
            identity_metrics = self.calculate_identity_metrics(
                landmarks_3d, ref_distance, R_matrix
            )

            # 5. Проверка на биологические пределы
            anomaly_scores = []
            for metric, (min_val, max_val) in self.biological_limits.items():
                if metric in stabilized_props:
                    value = stabilized_props[metric]
                    try:
                        deviation = abs(value - (min_val + max_val) / 2) / (max_val - min_val)
                        anomaly_scores.append(min(1.0, deviation))
                    except (ZeroDivisionError, ValueError) as e:
                        self.logger.warning(f"Ошибка при расчете отклонения для метрики {metric}: {e}")
                        continue

            # Формирование итогового результата
            result = {
                "stabilized_proportions": stabilized_props,
                "symmetry_metrics": symmetry_metrics,
                "identity_metrics": identity_metrics,
                "anomaly_score": float(np.mean(anomaly_scores) if anomaly_scores else 0.0),
                "pose_angles": {
                    "yaw": float(yaw),
                    "roll": float(roll),
                    "pitch": 0.0
                },
                "reference_distance": float(ref_distance),
                "stabilization_info": {
                    "rotation_compensated": R_matrix is not None,
                    "scale_applied": abs(s_scale - 1.0) > EPSILON
                },
                # Для визуализации landmarks используйте исходные (до нормализации)!
                "landmarks_for_visualization": np.array(landmarks_3d).copy()
            }

            # Добавляем оценки стабильности
            overall_score = np.mean([
                symmetry_metrics.get("overall_symmetry", 0.0),
                1.0 - (result["anomaly_score"] if result["anomaly_score"] < 1.0 else 1.0),
                identity_metrics.get("jaw_angle_symmetry", 0.0)
            ])

            result["stability_scores"] = {
                "overall_stability": float(overall_score),
                "symmetry_stability": float(symmetry_metrics.get("overall_symmetry", 0.0)),
                "proportion_stability": float(1.0 - result["anomaly_score"])
            }

            # Возвращаем также нормализованные landmarks (например, после компенсации поворота)
            normalized_landmarks = self.transform_to_canonical_pose(landmarks_3d, R_matrix) if R_matrix is not None else np.array(landmarks_3d)
            return result, normalized_landmarks

        except Exception as e:
            self.logger.error(f"Ошибка в analyze: {e}")
            return self._create_error_result(str(e)), np.zeros((0, 3), dtype=np.float32)

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Создание результата с ошибкой"""
        return {
            "error": error_message,
            "stabilized_proportions": {},
            "symmetry_metrics": {},
            "identity_metrics": {},
            "anomaly_score": 1.0,
            "pose_angles": {"yaw": 0.0, "roll": 0.0, "pitch": 0.0},
            "reference_distance": 1.0,
            "stabilization_info": {
                "rotation_compensated": False,
                "scale_applied": False
            }
        }

    def calculate_skull_identity_metrics(self, proportions: Dict[str, float]) -> Dict[str, float]:
        """Метрики для идентификации уникальности черепа"""
        skull_metrics = {}
        # Ключевые пропорции черепа (неизменные при мимике)
        skull_features = [
            'face_width_height_ratio',
            'nose_height_width_ratio',
            'jaw_width_face_ratio',
            'zygomatic_width_ratio'
        ]
        skull_values = [proportions.get(feature, 0.0) for feature in skull_features]
        skull_metrics['skull_composite_index'] = float(np.mean(skull_values))
        mean_val = np.mean(skull_values)
        skull_metrics['skull_variation_coefficient'] = float(np.std(skull_values) / mean_val) if mean_val > EPSILON else 0.0
        # Индекс уникальности (можно заменить на хэш или другую функцию)
        skull_metrics['skull_uniqueness_score'] = float(np.sum([abs(v - 1.0) for v in skull_values]))
        return skull_metrics

    def detect_anomalies_and_masks(self, proportions: Dict[str, float], symmetry_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Детекция аномальных пропорций и масок"""
        biological_ranges = {
            'face_width_height_ratio': (0.6, 1.4),
            'nose_width_iod_ratio': (0.25, 0.6),
            'eye_width_iod_ratio': (0.15, 0.25),
            'mouth_width_nose_ratio': (1.2, 2.5)
        }
        anomaly_scores = []
        detailed_anomalies = {}
        for metric, (min_val, max_val) in biological_ranges.items():
            if metric in proportions:
                value = proportions[metric]
                if value < min_val or value > max_val:
                    center = (min_val + max_val) / 2
                    range_size = max_val - min_val
                    anomaly_degree = min(1.0, abs(value - center) / range_size)
                    anomaly_scores.append(anomaly_degree)
                    detailed_anomalies[f'anomaly_{metric}'] = anomaly_degree
        overall_anomaly = float(np.mean(anomaly_scores)) if anomaly_scores else 0.0
        if overall_anomaly > 0.7:
            classification = "potential_mask"
            confidence = overall_anomaly
        elif overall_anomaly > 0.3:
            classification = "anomalous_human"
            confidence = overall_anomaly
        else:
            classification = "normal_human"
            confidence = 1.0 - overall_anomaly
        return {
            'overall_anomaly_index': overall_anomaly,
            'classification': classification,
            'classification_confidence': confidence,
            **detailed_anomalies
        }

    def format_analysis_results(self, proportions: Dict, skull_metrics: Dict, anomaly_detection: Dict, pose_side: str = "frontal") -> Dict:
        """Правильное форматирование результатов без абсолютных значений"""
        return {
            "facial_proportions": {
                f"{pose_side}_{key}": {
                    "ratio": float(value),
                    "deviation_from_golden": abs(value - 1.618) / 1.618 * 100
                } for key, value in proportions.items()
            },
            "skull_identity": {
                f"{pose_side}_skull_composite": skull_metrics.get('skull_composite_index', 0.0),
                f"{pose_side}_skull_uniqueness": skull_metrics.get('skull_uniqueness_score', 0.0),
                f"{pose_side}_skull_variation": skull_metrics.get('skull_variation_coefficient', 0.0)
            },
            "anomaly_analysis": {
                f"{pose_side}_anomaly_index": anomaly_detection.get('overall_anomaly_index', 0.0),
                f"{pose_side}_classification": anomaly_detection.get('classification', ''),
                f"{pose_side}_confidence": anomaly_detection.get('classification_confidence', 0.0)
            }
        }

# Для обратной совместимости:
FrontalAnalysisModule = StabilizedFrontalAnalysisModule
