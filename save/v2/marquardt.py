# marquardt.py
# Модуль расчета идеальных пропорций маски Марквардта для биометрического анализа
# Использует только 68 ключевых точек, доступных в 3DDFA_V2

import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from utils.pose import P2sRt, matrix2angle

# Математические константы золотого сечения
PHI = 1.618033988749895  # φ
PHI_INV = 0.618033988749895  # 1/φ = 0.618
PHI_SQ = 2.618033988749895  # φ² = φ + 1

# Epsilon для предотвращения деления на ноль
EPSILON = 1e-6

class MarquardtMask:
    """Расчет идеальных пропорций маски Марквардта на основе 68 ключевых точек"""
    
    def __init__(self):
        # Определение групп ключевых точек (совместимо с dlib/3DDFA_V2)
        self.landmark_groups = {
            'jaw': list(range(0, 17)),           # Контур челюсти (0-16)
            'right_eyebrow': list(range(17, 22)), # Правая бровь (17-21)
            'left_eyebrow': list(range(22, 27)),  # Левая бровь (22-26)
            'nose_bridge': list(range(27, 31)),   # Переносица (27-30)
            'nose_lower': list(range(31, 36)),    # Нижняя часть носа (31-35)
            'right_eye': list(range(36, 42)),     # Правый глаз (36-41)
            'left_eye': list(range(42, 48)),      # Левый глаз (42-47)
            'outer_lips': list(range(48, 60)),    # Внешний контур губ (48-59)
            'inner_lips': list(range(60, 68))     # Внутренний контур губ (60-67)
        }
        
        # Ключевые точки для каждого ракурса (по стабильности)
        self.stable_points = {
            'frontal': [27, 30, 33, 39, 42, 51, 57, 8],      # Высокостабильные для фронта
            'frontal_edge': [27, 30, 33, 51, 57, 8],         # Стабильные при повороте 10-20°
            'semi_profile': [27, 30, 33, 8, 48, 54],         # Стабильные при повороте 20-45°
            'profile': [27, 30, 33, 8, 0, 16]                # Стабильные в профиль 45-90°
        }
        
        self.ideal_proportions = self._calculate_ideal_proportions()
        self.golden_ratio = PHI # Добавляем для использования в calculate_marquardt_ratios
    
    def classify_angle(self, yaw: float) -> str:
        """Классификация ракурса по углу yaw"""
        abs_yaw = abs(yaw)
        if abs_yaw < 10:
            return 'frontal'
        elif abs_yaw < 20:
            return 'frontal_edge'
        elif abs_yaw < 45:
            return 'semi_profile'
        else:
            return 'profile'
    
    def get_reference_distance(self, vertices: np.ndarray, angle_type: str) -> float:
        """Получение опорного расстояния для нормализации"""
        try:
            if angle_type == 'frontal':
                dist = np.linalg.norm(vertices[39, :] - vertices[42, :])
                return dist if dist > EPSILON else EPSILON # Ensure not zero
            elif angle_type == 'frontal_edge':
                dist = np.linalg.norm(vertices[27, :] - vertices[33, :])
                return dist if dist > EPSILON else EPSILON
            elif angle_type == 'semi_profile':
                dist = abs(vertices[27, 1] - vertices[8, 1])
                return dist if dist > EPSILON else EPSILON
            else:  # profile
                dist = abs(vertices[27, 1] - vertices[8, 1])
                return dist if dist > EPSILON else EPSILON
        except (IndexError, ValueError):
            return EPSILON  # Fallback value
    
    def _calculate_ideal_proportions(self) -> Dict[str, Dict[str, float]]:
        """Расчет всех идеальных пропорций для каждого ракурса"""
        
        return {
            'frontal': {
                # ОСНОВНЫЕ ПРОПОРЦИИ ЛИЦА
                'face_width_height_ratio': 0.618,  # PHI_INV
                'face_golden_ratio': 1.618,            # PHI - Это, вероятно, общая золотая пропорция лица, не конкретная метрика
                'face_golden_ratio_height_width': 1.618, # Золотое сечение для высоты к ширине
                
                # ГЛАЗА (точки 36-47)
                'eye_width_face_ratio': 0.2,           # ширина глаза к ширине лица
                'interocular_distance_ratio': 1.0,     # расстояние между глазами = ширине глаза
                'right_eye_aspect_ratio': 0.3,         # Типичное идеальное значение EAR
                'left_eye_aspect_ratio': 0.3,          # Типичное идеальное значение EAR
                
                # НОС (точки 27-35)
                'nose_width_eye_distance': 1.0,        # ширина носа = расстоянию между глазами
                'nose_length_face_ratio': 0.333,       # длина носа = 1/3 высоты лица
                'nose_height_width_ratio': 1.6,        # Типичное идеальное соотношение (nasion-pronasale / alar_width)
                'columella_to_nose_length_ratio': 0.5, # Типичное идеальное соотношение (subnasale-pronasale / nasion-pronasale)
                
                # РОТ И ГУБЫ (точки 48-67)
                'mouth_width_nose_ratio': 1.618,         # PHI (ширина рта к ширине носа)
                'upper_lip_lower_ratio': 0.618,      # PHI_INV (верхняя губа к нижней)
                'mouth_height_width_ratio': 0.5,       # Типичное идеальное соотношение
                
                # ЧЕЛЮСТЬ И ПОДБОРОДОК (точки 0-16, 8)
                'jaw_width_face_ratio': 0.618,       # PHI_INV (ширина челюсти к ширине лица)
                'chin_width_mouth_ratio': 0.618,     # PHI_INV (ширина подбородка к ширине рта)
                'chin_projection_ratio': None,      # Установка None, так как это сложная 3D метрика
                'zygomatic_to_bigonial_ratio': 1.0,    # Идеальное соотношение ~1.0
                
                # БРОВИ (точки 17-26)
                'eyebrow_eye_distance_ratio': 0.618, # PHI_INV (расстояние бровь-глаз)
                'eyebrow_length_eye_ratio': 1.618,       # PHI (длина брови к длине глаза)
                
                # СИММЕТРИЯ (эти метрики идеально 0, поэтому их отклонение будет самим значением)
                'bilateral_symmetry_tolerance': None,   # Идеальная симметрия = 0
                'vertical_symmetry_ratio': None,        # Идеальная симметрия = 0
                'palpebral_symmetry_x': 0.0,           # Идеальная симметрия = 0
                'alar_width_ratio': 0.0,               # Идеальная симметрия = 0
                'midface_symmetry_index': 0.0,         # Идеальная симметрия = 0
                
                # ПОЗИЦИОННЫЕ МЕТРИКИ (некоторые из них также могут быть идеально 0)
                'philtrum_length_ratio': 0.2,          # Типичное идеальное значение (для справки)
                'canthal_tilt_normalized': 0.0,        # Идеальный наклон = 0 градусов
                'ocular_to_nasal_angle_degrees': 30.0, # Типичный угол (для справки)
                'nasolabial_angle_cos': 0.95,          # Типичный cos угла
                'upper_third_width_ratio': 1.0,        # Идеальное соотношение ~1.0
                
                # РАССТОЯНИЯ (для них нет фиксированных идеальных значений, поэтому None)
                'a_face_height': None,
                'b_forehead_to_eyes': None,
                'c_eyes_to_nose': None,
                'd_eyes_to_lips': None,
                'e_nose_width': None,
                'f_eye_span': None,
                'g_face_width': None,
                'i_nose_to_chin': None,
                'j_lips_to_chin': None,
                'k_mouth_width': None,
                'l_nose_to_lips': None,
                'ipd_interpupillary': None,
                'avg_eye_width': None
            },
            
            'frontal_edge': {
                # КОРРЕКЦИЯ НА ПОВОРОТ 10-20°
                'perspective_correction': 0.95,        # общая коррекция перспективы (возможно, оставить для анализа)
                'visible_face_ratio': 0.90,           # видимая часть лица
                
                # АСИММЕТРИЯ ПРИ ПОВОРОТЕ
                'eye_asymmetry_ratio': 0.92,          # асимметрия глаз
                'nostril_asymmetry_ratio': 0.88,      # асимметрия ноздрей
                'mouth_asymmetry_ratio': 0.95,        # асимметрия рта
                
                # СОХРАНЯЮЩИЕСЯ ПРОПОРЦИИ
                'nose_mouth_distance_ratio': PHI_INV,  # расстояние нос-рот
                'eye_nose_distance_ratio': PHI_INV,    # расстояние глаз-нос
                'vertical_thirds_ratio': 0.95,        # сохранение вертикальных третей
                
                # ПРОЕКЦИОННЫЕ ИЗМЕРЕНИЯ
                'nose_projection_ratio': 0.15,        # выступ носа
                'cheek_visibility_ratio': 0.80,       # видимость щеки
                'jaw_definition_ratio': 0.85,         # четкость линии челюсти
            },
            
            'semi_profile': {
                # ПОЛУПРОФИЛЬНЫЕ ПРОПОРЦИИ 20-45°
                'profile_emergence_ratio': 0.65,      # появление профильности
                'depth_width_ratio': 0.70,            # глубина к ширине
                
                # НОСОВЫЕ ХАРАКТЕРИСТИКИ
                'nose_bridge_prominence': 0.30,       # выступ переносицы
                'nasal_tip_projection': 0.60,         # проекция кончика носа
                'nostril_visibility_ratio': 0.75,     # видимость ноздрей
                
                # ГЛАЗНАЯ ОБЛАСТЬ
                'eye_socket_depth_ratio': 0.25,       # глубина глазницы
                'brow_prominence_ratio': 0.20,        # выступ брови
                
                # РОТО-ЧЕЛЮСТНАЯ ОБЛАСТЬ
                'mouth_projection_ratio': 0.40,       # проекция рта
                'jaw_angle_definition': 0.75,         # четкость угла челюсти
                'chin_profile_ratio': 0.65,           # профиль подбородка
                
                # СОХРАНЯЮЩИЕСЯ ЗОЛОТЫЕ ПРОПОРЦИИ
                'golden_triangle_ratio': PHI_INV,     # золотой треугольник
                'facial_thirds_maintenance': 0.90,    # сохранение третей
            },
            
            'profile': {
                # ПОЛНЫЙ ПРОФИЛЬ 45-90°
                'facial_angle': 80.0,                 # лицевой угол в градусах
                'profile_convexity': 0.0,             # выпуклость профиля
                
                # ПРОФИЛЬНЫЕ УГЛЫ (в градусах)
                'nasofrontal_angle': 120.0,           # угол лоб-нос
                'nasolabial_angle': 95.0,             # угол нос-губы
                'mentocervical_angle': 110.0,         # угол подбородок-шея
                
                # НОСОВОЙ ПРОФИЛЬ
                'nose_bridge_straightness': 0.95,     # прямота спинки носа
                'nasal_tip_rotation': 105.0,          # ротация кончика носа
                'nose_projection_profile': 0.67,      # выступ носа в профиль
                
                # ГУБНО-ПОДБОРОДОЧНАЯ ОБЛАСТЬ
                'upper_lip_protrusion': 0.40,        # выступ верхней губы
                'lower_lip_protrusion': 0.35,        # выступ нижней губы
                'chin_projection_profile': 0.65,      # выступ подбородка
                
                # ПРОФИЛЬНАЯ ЛИНИЯ
                'profile_smoothness': 0.90,          # плавность профиля
                'jawline_definition': 0.85,          # четкость линии челюсти
                
                # ПРОПОРЦИИ ГЛУБИНЫ
                'face_depth_height_ratio': PHI_INV,  # глубина к высоте лица
                'forehead_projection': 0.95,         # проекция лба
            }
        }
    
    def calculate_proportions(self, vertices: np.ndarray, angle_type: str, roll: float = 0.0) -> Dict[str, float]:
        """Расчет всех пропорций для конкретного ракурса"""
        
        ref_distance = self.get_reference_distance(vertices, angle_type)
        print(f"DEBUG: ref_distance for {angle_type} = {ref_distance}")
        # ref_distance уже гарантированно > EPSILON благодаря get_reference_distance
        
        # Calculate stable distances once for use in _calc_frontal_props and _calculate_additional_marquardt_metrics
        distances = self.calculate_stable_distances(vertices, ref_distance)

        if angle_type == 'frontal':
            basic_props = self._calc_frontal_props(vertices, ref_distance, distances)
        elif angle_type == 'frontal_edge':
            basic_props = self._calc_frontal_edge_props(vertices, ref_distance)
        elif angle_type == 'semi_profile':
            basic_props = self._calc_semi_profile_props(vertices, ref_distance)
        else:  # profile
            basic_props = self._calc_profile_props(vertices, ref_distance)

        additional_metrics = self._calculate_additional_marquardt_metrics(vertices, ref_distance, roll)
        
        # Объединяем основные пропорции и дополнительные метрики
        # Если ключи пересекаются, предпочтение отдается основным пропорциям
        all_metrics = {**additional_metrics, **basic_props}

        return all_metrics
    
    def _safe_divide(self, numerator: float, denominator: float) -> float:
        """Безопасное деление, возвращает 0.0, если знаменатель близок к нулю."""
        # Возвращаем очень большое число (или inf), чтобы не маскировать проблемы
        if abs(denominator) < EPSILON:
            if numerator > EPSILON:
                return float('inf')
            elif numerator < -EPSILON:
                return float('-inf')
            else:
                return 0.0 # if numerator is also zero
        return numerator / denominator

    def _calc_frontal_props(self, v: np.ndarray, ref: float, distances: Dict[str, float]) -> Dict[str, float]:
        """Расчет пропорций для фронтального ракурса"""
        props = {}
        
        try:
            # Основные размеры лица
            face_width = np.linalg.norm(v[0, :] - v[16, :])  # ширина челюсти
            face_height = abs(v[27, 1] - v[8, 1])  # высота лица
            props['face_width_height_ratio'] = self._safe_divide(face_width / ref, face_height / ref)
            
            # Добавление face_golden_ratio и face_golden_ratio_height_width
            if face_width > EPSILON and face_height > EPSILON:
                ratio1 = self._safe_divide(face_height, face_width)
                ratio2 = self._safe_divide(face_width, face_height)
                props['face_golden_ratio'] = max(ratio1, ratio2) # Берем большее из двух для общей золотой пропорции
                props['face_golden_ratio_height_width'] = ratio1
            
            # Глаза
            right_eye_width = np.linalg.norm(v[36, :] - v[39, :])  # правый глаз
            left_eye_width = np.linalg.norm(v[42, :] - v[45, :])   # левый глаз
            eye_distance = np.linalg.norm(v[39, :] - v[42, :])     # между глазами
            
            props['eye_width_face_ratio'] = self._safe_divide(((right_eye_width + left_eye_width) / 2), face_width)
            props['interocular_distance_ratio'] = self._safe_divide(eye_distance, ((right_eye_width + left_eye_width) / 2))
            
            # Нос
            nose_width = np.linalg.norm(v[31, :] - v[35, :])  # ширина носа
            nose_length = abs(v[27, 1] - v[33, 1])           # длина носа
            
            props['nose_width_eye_distance'] = self._safe_divide(nose_width, eye_distance)
            props['nose_length_face_ratio'] = self._safe_divide(nose_length, face_height)
            
            # Рот
            mouth_width = np.linalg.norm(v[48, :] - v[54, :])  # ширина рта
            upper_lip_height = abs(v[51, 1] - v[62, 1])       # высота верхней губы
            lower_lip_height = abs(v[57, 1] - v[66, 1])       # высота нижней губы
            
            props['mouth_width_nose_ratio'] = self._safe_divide(mouth_width, nose_width)
            props['upper_lip_lower_ratio'] = self._safe_divide(upper_lip_height, lower_lip_height)
            
            # Челюсть и подбородок - новые расчеты
            # jaw_width_face_ratio: ширина челюсти (v[0]-v[16]) к ширине лица (v[0]-v[16] же?)
            # Если "ширина лица" имеется в виду max width, то ее нужно найти.
            # Если имеется в виду g_face_width из stable_distances, то его нужно получить.
            # Пока используем face_width как ширину лица.
            props['jaw_width_face_ratio'] = self._safe_divide(face_width, face_width) # это будет 1.0, если face_width - это jaw_width
            
            # chin_width_mouth_ratio: ширина подбородка (v[6]-v[10]) к ширине рта
            # Теперь chin_width и k_mouth_width должны быть в distances
            chin_width_val = distances.get('chin_width', EPSILON)
            k_mouth_width_val = distances.get('k_mouth_width', EPSILON)
            props['chin_width_mouth_ratio'] = self._safe_divide(chin_width_val, k_mouth_width_val)
            
            # chin_projection_ratio (оставлено None в ideal_proportions)
            
        except Exception as e:
            print(f"Ошибка при расчете фронтальных пропорций: {e}")
            return {}
        return props
    
    def _calc_frontal_edge_props(self, v: np.ndarray, ref: float) -> Dict[str, float]:
        """Расчет пропорций для полуфронтального ракурса"""
        props = {}
        # В этом режиме нет стабильных пропорций для отчета.
        # Все метрики могут быть подвержены шуму или проекционным искажениям.
        return props
    
    def _calc_semi_profile_props(self, v: np.ndarray, ref: float) -> Dict[str, float]:
        """Расчет пропорций для полупрофильного ракурса"""
        props = {}
        # В этом режиме метрики также подвержены сильным искажениям из-за перспективы и поворота.
        # Не будем рассчитывать специфические пропорции.
        return props
    
    def _calc_profile_props(self, v: np.ndarray, ref: float) -> Dict[str, float]:
        """Расчет пропорций для профильного ракурса"""
        props = {}
        # В профильном режиме большинство метрик также подвержены значительным искажениям и шумам.
        # Не будем рассчитывать специфические пропорции.
        return props
    
    def _calculate_line_smoothness(self, points: List[np.ndarray]) -> float:
        """Расчет плавности линии через точки"""
        if len(points) < 3:
            return 1.0
        
        try:
            angles = []
            for i in range(1, len(points) - 1):
                v1 = points[i] - points[i-1]
                v2 = points[i+1] - points[i]
                
                mag_v1 = np.linalg.norm(v1)
                mag_v2 = np.linalg.norm(v2)

                if mag_v1 < EPSILON or mag_v2 < EPSILON: # Избегаем деления на ноль
                    continue
                
                cos_angle = np.dot(v1, v2) / (mag_v1 * mag_v2)
                angle = math.acos(np.clip(cos_angle, -1.0, 1.0))
                angles.append(angle)
            
            if not angles:
                return 1.0 # Если не удалось рассчитать углы
            
            # Чем меньше отклонение углов от 180°, тем плавнее линия
            avg_deviation = np.mean([abs(math.pi - angle) for angle in angles])
            smoothness = 1.0 - (avg_deviation / math.pi)
            
            return max(0.0, min(1.0, smoothness))
            
        except (ValueError, ZeroDivisionError):
            return 1.0
    
    def calculate_similarity_score(self, measured: Dict[str, float], angle_type: str) -> float:
        """Расчет коэффициента соответствия идеальным пропорциям"""
        
        ideal = self.ideal_proportions[angle_type]
        scores = []
        
        for metric, measured_value in measured.items():
            if metric in ideal:
                ideal_value = ideal[metric]
                
                if ideal_value is not None: # Пропускаем метрики без идеальных значений
                    if abs(ideal_value) < EPSILON: # Ideal is 0, measured is 0 -> perfect score
                        if abs(measured_value) < EPSILON:
                            scores.append(1.0)
                        else: # Ideal is 0, measured is not 0. Score is 0.
                            scores.append(0.0)
                    else:
                        deviation = abs(measured_value - ideal_value) / abs(ideal_value)
                        score = max(0.0, 1.0 - deviation)
                        scores.append(score)
        
        return np.mean(scores) if scores else 0.0
    
    def get_stable_landmarks(self, angle_type: str) -> List[int]:
        """Получение списка стабильных точек для ракурса"""
        return self.stable_points.get(angle_type, [])

    def extract_pose_parameters(self, param_3dmm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Извлечение параметров позы из 3DMM параметров"""
        if len(param_3dmm) < 12:
            return np.eye(3), np.zeros(3), 1.0

        # Create the 3x4 camera matrix P from the first 12 parameters
        P_camera = param_3dmm[:12].reshape(3, 4)
        
        # Decompose P_camera into scale (s), rotation matrix (R), and 3D translation (t3d)
        s, R_from_P2sRt, t3d = P2sRt(P_camera)
        
        # Use the rotation matrix R and scale s from P2sRt
        # Keep 2D translation t consistent with original code (param_3dmm[10:12])
        # t_2d = param_3dmm[10:12] 
        
        return R_from_P2sRt, t3d, s

    def extract_euler_angles(self, R: np.ndarray) -> List[float]:
        """Извлечение углов Эйлера (Pitch, Yaw, Roll) из матрицы поворота"""
        # Используем функцию matrix2angle из utils.pose, которая соответствует логике TDDFA
        # matrix2angle возвращает (yaw_rad, pitch_rad, roll_rad)
        yaw_rad, pitch_rad, roll_rad = matrix2angle(R)
        
        # Возвращаем углы в градусах в порядке [pitch, yaw, roll]
        return [math.degrees(pitch_rad), math.degrees(yaw_rad), math.degrees(roll_rad)]

    def normalize_3d_landmarks(self, landmarks_3d: np.ndarray, R: np.ndarray, t: np.ndarray, s: float) -> np.ndarray:
        """Нормализация 3D landmarks к канонической системе координат"""
        # Применяем обратное преобразование для устранения поворота
        R_inv = R.T  # обратная матрица поворота

        # Центрируем landmarks
        centroid = np.mean(landmarks_3d, axis=0)
        centered_landmarks = landmarks_3d - centroid

        # Применяем обратный поворот
        normalized_landmarks = np.dot(centered_landmarks, R_inv.T)

        # Нормализуем масштаб
        if s > EPSILON: # Проверка на ноль
            normalized_landmarks /= s

        return normalized_landmarks

    def calculate_stable_distances(self, landmarks_3d: np.ndarray, ref_distance: float = None) -> Dict[str, float]:
        """Расчет стабильных расстояний согласно маске Марквардта"""
        if len(landmarks_3d) < 68:
            return {}

        if ref_distance is None:
            # Для calculate_stable_distances всегда предполагаем фронтальный ракурс,
            # так как эти метрики обычно рассчитываются для сравнения с идеалом
            # фронтального лица, независимо от фактического поворота.<
            ref_distance = self.get_reference_distance(landmarks_3d, 'frontal')
            # ref_distance уже гарантированно > EPSILON
            
        distances = {}

        # Ключевые точки согласно 68-точечной модели
        # Глаза: 36-41 (левый), 42-47 (правый)
        left_eye_points = landmarks_3d[36:42]
        right_eye_points = landmarks_3d[42:48]
        left_eye_center = np.mean(left_eye_points, axis=0)
        right_eye_center = np.mean(right_eye_points, axis=0)

        # Нос: 27-35
        nose_bridge = landmarks_3d[27]
        nose_tip = landmarks_3d[30]
        nose_left = landmarks_3d[31]
        nose_right = landmarks_3d[35]

        # Рот: 48-67
        mouth_left = landmarks_3d[48]
        mouth_right = landmarks_3d[54]
        mouth_top = landmarks_3d[51]
        mouth_bottom = landmarks_3d[57]

        # Контур лица: 0-16
        face_left = landmarks_3d[0]
        face_right = landmarks_3d[16]
        chin = landmarks_3d[8]

        # Брови/лоб
        left_brow = landmarks_3d[19]
        right_brow = landmarks_3d[24]
        brow_center = (left_brow + right_brow) / 2

        # === БАЗОВЫЕ РАССТОЯНИЯ МАСКИ МАРКВАРДТА (НОРМАЛИЗОВАННЫЕ) ===
        # Все деления на ref_distance уже безопасны, так как ref_distance > EPSILON
        
        # a: Top of head to chin (приблизительно от бровей до подбородка)
        distances['a_face_height'] = self._safe_divide(np.linalg.norm(brow_center - chin), ref_distance)

        # b: Top of head to pupil (от бровей до центра глаз)
        eye_line_center = (left_eye_center + right_eye_center) / 2
        distances['b_forehead_to_eyes'] = self._safe_divide(np.linalg.norm(brow_center - eye_line_center), ref_distance)

        # c: Pupil to nosetip (от глаз до кончика носа)
        distances['c_eyes_to_nose'] = self._safe_divide(np.linalg.norm(eye_line_center - nose_tip), ref_distance)

        # d: Pupil to lip (от глаз до губ)
        distances['d_eyes_to_lips'] = self._safe_divide(np.linalg.norm(eye_line_center - mouth_top), ref_distance)

        # e: Width of nose (ширина носа)
        distances['e_nose_width'] = self._safe_divide(np.linalg.norm(nose_left - nose_right), ref_distance)

        # f: Outside difference between eyes (межглазное расстояние - внешние углы)
        distances['f_eye_span'] = self._safe_divide(np.linalg.norm(landmarks_3d[36] - landmarks_3d[45]), ref_distance)

        # g: Width of head (ширина лица)
        distances['g_face_width'] = self._safe_divide(np.linalg.norm(face_left - face_right), ref_distance)

        # h: Hairline to pupil (от линии волос до глаз - используем верх лба)
        distances['h_hairline_to_eyes'] = distances['b_forehead_to_eyes'] # Уже нормализовано

        # i: Nosetip to chin (от носа до подбородка)
        distances['i_nose_to_chin'] = self._safe_divide(np.linalg.norm(nose_tip - chin), ref_distance)

        # j: Lips to chin (от губ до подбородка)
        distances['j_lips_to_chin'] = self._safe_divide(np.linalg.norm(mouth_bottom - chin), ref_distance)

        # k: Length of lips (длина губ)
        distances['k_mouth_width'] = self._safe_divide(np.linalg.norm(mouth_left - mouth_right), ref_distance)

        # l: Nosetip to lips (от носа до губ)
        distances['l_nose_to_lips'] = self._safe_divide(np.linalg.norm(nose_tip - mouth_top), ref_distance)

        # Дополнительные важные расстояния
        # Межзрачковое расстояние (IPD)
        distances['ipd_interpupillary'] = self._safe_divide(np.linalg.norm(left_eye_center - right_eye_center), ref_distance)

        # Ширина глаз
        left_eye_width = np.linalg.norm(landmarks_3d[36] - landmarks_3d[39])
        right_eye_width = np.linalg.norm(landmarks_3d[42] - landmarks_3d[45])
        distances['avg_eye_width'] = self._safe_divide((left_eye_width + right_eye_width) / 2, ref_distance)

        # Ширина подбородка (v[6] - v[10])
        distances['chin_width'] = self._safe_divide(np.linalg.norm(landmarks_3d[6, :] - landmarks_3d[10, :]), ref_distance)

        return distances

    def calculate_marquardt_ratios(self, distances: Dict[str, float]) -> Dict[str, float]:
        """Расчет пропорций Марквардта на основе стабильных расстояний"""
        ratios = {}
        
        # Основные золотые пропорции, использующие нормализованные расстояния
        if 'a_face_height' in distances and 'g_face_width' in distances:
            # Face width to height ratio
            ratios['face_width_height_ratio'] = self._safe_divide(distances['g_face_width'], distances['a_face_height'])
            
            # Face golden ratio
            ratios['face_golden_ratio_height_width'] = self._safe_divide(distances['a_face_height'], distances['g_face_width'])
            
            # Общая золотая пропорция лица
            if distances['g_face_width'] > EPSILON and distances['a_face_height'] > EPSILON:
                ratio1 = self._safe_divide(distances['a_face_height'], distances['g_face_width'])
                ratio2 = self._safe_divide(distances['g_face_width'], distances['a_face_height'])
                ratios['face_golden_ratio'] = max(ratio1, ratio2)

        if 'avg_eye_width' in distances and 'g_face_width' in distances:
            ratios['eye_width_face_ratio'] = self._safe_divide(distances['avg_eye_width'], distances['g_face_width'])

        if 'ipd_interpupillary' in distances and 'avg_eye_width' in distances:
            ratios['interocular_distance_ratio'] = self._safe_divide(distances['ipd_interpupillary'], distances['avg_eye_width'])

        if 'e_nose_width' in distances and 'ipd_interpupillary' in distances:
            ratios['nose_width_eye_distance'] = self._safe_divide(distances['e_nose_width'], distances['ipd_interpupillary'])

        if 'i_nose_to_chin' in distances and 'a_face_height' in distances:
            ratios['nose_length_face_ratio'] = self._safe_divide(distances['i_nose_to_chin'], distances['a_face_height'])
        
        if 'k_mouth_width' in distances and 'e_nose_width' in distances:
            ratios['mouth_width_nose_ratio'] = self._safe_divide(distances['k_mouth_width'], distances['e_nose_width'])
        
        # Пропорция верхней к нижней губе (если доступны соответствующие метрики)
        # Эти метрики сейчас не рассчитываются в calculate_stable_distances напрямую,
        # поэтому пока оставим без реализации.
        # if 'upper_lip_height' in distances and 'lower_lip_height' in distances:
        #     if distances['lower_lip_height'] != 0:
        #         ratios['upper_lip_lower_ratio'] = distances['upper_lip_height'] / distances['lower_lip_height']

        return ratios

    def calculate_symmetry_metrics(self, landmarks_3d: np.ndarray) -> Dict[str, float]:
        """Расчет метрик симметрии лица.
        Включает оценку симметрии носа, глаз и рта.
        """
        return {}

    def calculate_geometric_invariants(self, landmarks_3d: np.ndarray) -> Dict[str, float]:
        """Расчет геометрических инвариантов лица."""
        return {}

    def calculate_morphological_stability(self, landmarks_3d: np.ndarray) -> Dict[str, float]:
        """Расчет индекса морфологической стабильности.
        Оценивает стабильность формы лица при небольших изменениях позы.
        """
        return {}

    def calculate_biometric_uniqueness(self, landmarks_3d: np.ndarray, distances: Dict[str, float]) -> Dict[str, float]:
        """Расчет индекса биометрической уникальности.
        Оценивает уникальность лица на основе комбинации стабильных метрик.
        """
        return {}

    # === Специализированные детекторы аномалий ===
    # Удалены все нестабильные метрики

    def _calculate_additional_marquardt_metrics(self, v: np.ndarray, ref_distance: float, roll: float = 0.0) -> Dict[str, float]:
        """
        Расчет 20 дополнительных биометрических метрик на основе 68-точечных лендмарков.
        Метрики нормализуются с помощью ref_distance.
        v: 3D landmarks (np.ndarray)
        ref_distance: опорное расстояние для нормализации
        """
        metrics = {}
        
        if len(v) < 68:
            print("ВНИМАНИЕ: Недостаточно landmarks для расчета дополнительных метрик.")
            return metrics
        
        # ref_distance уже гарантированно > EPSILON

        try:
            print("DEBUG: Before calling calculate_stable_distances in _calculate_additional_marquardt_metrics.")
            distances = self.calculate_stable_distances(v, ref_distance)
            metrics.update(distances)  # **ИСПОЛНЕНИЕ ПУНКТА 1: Добавляем distances в metrics**
            print(f"DEBUG: After calling calculate_stable_distances. distances = {distances}")
        except Exception as e:
            print(f"ОШИБКА: Исключение при расчете stable_distances: {e}")
            return metrics # Возвращаем рано, если есть ошибка

        if not distances: # Если stable_distances вернул пустой словарь (из-за ref_distance=0)
            print("ВНИМАНИЕ: stable_distances вернул пустой словарь. Пропускаем расчет метрик, зависящих от него.")
            return metrics

        # Убедимся, что все используемые индексы landmarks существуют
        # В этой функции v - это 68-точечные landmarks.

        # 1. outer_canthi_span_face_ratio: дистанция 36-45 ÷ g_face_width
        outer_canthi_span = np.linalg.norm(v[36, :] - v[45, :])
        face_width_g = np.linalg.norm(v[0, :] - v[16, :]) # g_face_width
        metrics['outer_canthi_span_face_ratio'] = self._safe_divide((outer_canthi_span / ref_distance), (face_width_g / ref_distance))

        # 2. inner_canthi_span_face_ratio: дистанция 39-42 ÷ g_face_width
        inner_canthi_span = np.linalg.norm(v[39, :] - v[42, :])
        metrics['inner_canthi_span_face_ratio'] = self._safe_divide((inner_canthi_span / ref_distance), (face_width_g / ref_distance))

        # 3. eye_line_height_ratio: средняя y-координата 39-42 - y бровей ÷ a_face_height
        # a_face_height (brow_center - chin)
        brow_center_y = (v[19, 1] + v[24, 1]) / 2 # средняя y правая/левая бровь
        eye_line_center_y = (v[39, 1] + v[42, 1]) / 2 # средняя y внутренних уголков глаз
        
        chin_y = v[8, 1]
        face_height_a = abs(brow_center_y - chin_y)
        
        metrics['eye_line_height_ratio'] = self._safe_divide((abs(eye_line_center_y - brow_center_y) / ref_distance), (face_height_a / ref_distance))
            
        # 4. glabella_to_subnasale_ratio: glabella–subnasale ÷ a_face_height
        # Glabella: приблизительно точка 27 (переносица - Corrected from 21 to 27)
        # Subnasale: приблизительно точка 33 (основание носа)
        glabella_to_subnasale = np.linalg.norm(v[27, :] - v[33, :]) # Corrected index
        metrics['glabella_to_subnasale_ratio'] = self._safe_divide((glabella_to_subnasale / ref_distance), (face_height_a / ref_distance))

        # 5. subnasale_to_menton_ratio: subnasale–pogonion ÷ a_face_height
        # Subnasale: приблизительно точка 33 (основание носа)
        # Menton/Pogonion: приблизительно точка 8 (подбородок)
        subnasale_to_menton = np.linalg.norm(v[33, :] - v[8, :])
        metrics['subnasale_to_menton_ratio'] = self._safe_divide((subnasale_to_menton / ref_distance), (face_height_a / ref_distance))

        # 6. zygomatic_to_bigonial_ratio: zygion-zygion ÷ gonion-gonion
        # Zygion: точки 1 и 15 (самые выступающие точки скул по бокам)
        # Gonion: точки 4 и 12 (углы нижней челюсти)
        zygion_span = np.linalg.norm(v[1, :] - v[15, :])
        gonion_span = np.linalg.norm(v[4, :] - v[12, :]) # Изменено с v[5]-v[11] на v[4]-v[12]
        metrics['zygomatic_to_bigonial_ratio'] = self._safe_divide((zygion_span / ref_distance), (gonion_span / ref_distance))

        # 7. philtrum_length_ratio: subnasale-labiale superius ÷ a_face_height
        # Labiale superius: точка 51 (верхняя точка верхней губы)
        philtrum_length = np.linalg.norm(v[33, :] - v[51, :])
        metrics['philtrum_length_ratio'] = self._safe_divide((philtrum_length / ref_distance), (face_height_a / ref_distance))

        # 8. eye_aspect_ratio: (высота 38-40) ÷ (ширина 36-39)
        # Для правого глаза:
        right_h1 = np.linalg.norm(v[37, :]-v[41, :]) # 37-41
        right_h2 = np.linalg.norm(v[38, :]-v[40, :]) # 38-40
        right_eye_width = np.linalg.norm(v[36, :] - v[39, :]) # 36-39
        metrics['right_eye_aspect_ratio'] = self._safe_divide(((right_h1 + right_h2) / 2), right_eye_width)
        
        # Для левого глаза:
        left_h1 = np.linalg.norm(v[43, :]-v[47, :]) # 43-47
        left_h2 = np.linalg.norm(v[44, :]-v[46, :]) # 44-46
        left_eye_width = np.linalg.norm(v[42, :] - v[45, :])
        metrics['left_eye_aspect_ratio'] = self._safe_divide(((left_h1 + left_h2) / 2), left_eye_width)

        # 9. nose_height_width_ratio: (nasion-pronasale) ÷ e_nose_width
        # Nasion: точка 27 (переносица)
        # Pronasale: точка 30 (кончик носа)
        # e_nose_width: (v[31, :] - v[35, :])
        nose_height = np.linalg.norm(v[27, :] - v[30, :]) # nose_height (nasion to pronasale)
        nose_width_e = np.linalg.norm(v[31, :] - v[35, :])
        metrics['nose_height_width_ratio'] = self._safe_divide((nose_height / ref_distance), (nose_width_e / ref_distance))

        # 10. columella_to_nose_length_ratio: subnasale-pronasale ÷ nasion-pronasale
        # Subnasale: точка 33
        # Pronasale: точка 30
        columella_pronasale = np.linalg.norm(v[33, :] - v[30, :])
        nasion_pronasale = np.linalg.norm(v[27, :] - v[30, :])
        metrics['columella_to_nose_length_ratio'] = self._safe_divide((columella_pronasale / ref_distance), (nasion_pronasale / ref_distance))

        # 11. mouth_height_width_ratio: (labiale superius-labiale inferius) ÷ k_mouth_width
        # Labiale superius: точка 51
        # Labiale inferius: точка 57
        # k_mouth_width: (v[48, :] - v[54, :])
        mouth_height = np.linalg.norm(v[51, :] - v[57, :])
        mouth_width_k = np.linalg.norm(v[48, :] - v[54, :])
        metrics['mouth_height_width_ratio'] = self._safe_divide((mouth_height / ref_distance), (mouth_width_k / ref_distance))

        # 12. canthal_tilt_normalized: угол между линией 36-45 и горизонталью
        # **ИСПОЛНЕНИЕ ПУНКТА 2: Оставляем чистый угол в градусах**
        p1_canthal = v[36, :2]
        p2_canthal = v[45, :2]
        delta_x_canthal = p2_canthal[0] - p1_canthal[0]
        delta_y_canthal = p2_canthal[1] - p1_canthal[1]
        
        angle_rad_canthal = math.atan2(delta_y_canthal, delta_x_canthal)
        metrics['canthal_tilt_normalized'] = math.degrees(angle_rad_canthal) # Изменено

        # 13. palpebral_symmetry_x = abs(x_left - x_right) / eye_distance
        # x_left и x_right - x-координаты внутренних уголков глаз (39 и 42)
        # eye_distance - расстояние между внутренними уголками глаз (39-42)
        metrics['palpebral_symmetry_x'] = self._safe_divide(abs(v[39, 0] - v[42, 0]), (distances.get('ipd_interpupillary', EPSILON) * ref_distance))

        # 14. alar_width_ratio = (dist(31–30)+dist(35–30))/(2*e_nose_width)
        left_alar_pronasale_dist = np.linalg.norm(v[31, :] - v[30, :])
        right_alar_pronasale_dist = np.linalg.norm(v[35, :] - v[30, :])
        nose_width_e = np.linalg.norm(v[31, :] - v[35, :])
        
        metrics['alar_width_ratio'] = self._safe_divide((left_alar_pronasale_dist + right_alar_pronasale_dist), (2 * nose_width_e))

        # 15. tragion_to_outer_canthi_ratio: Удалено.
        # 16. ear_span_face_width_ratio: Удалено.

        # 17. midface_symmetry_index: RMS разницы левых-правых 20 пар симм. точек ÷ g_face_width
        # **ИСПОЛНЕНИЕ ПУНКТА 3: Делим на ширину лица (g_face_width)**
        # Определим пары симметричных точек: (левая, правая)
        symmetric_pairs = [
            (0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), # Челюсть
            (17, 26), (18, 25), (19, 24), (20, 23), (21, 22), # Брови
            (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46), # Глаза
            (48, 54), (49, 53), (50, 52), (59, 55), (58, 56), # Внешние губы
            (60, 64), (61, 63), (67, 65) # Внутренние губы
        ]
        
        differences_x_squared = []
        
        for l_idx, r_idx in symmetric_pairs:
            # Вычисляем разницу по x-координатам
            diff_x = abs(v[l_idx, 0] - v[r_idx, 0])
            differences_x_squared.append(diff_x**2)

        if differences_x_squared:
            rms_diff_x = np.sqrt(np.mean(differences_x_squared))
            # Нормализуем на ширину лица g_face_width
            metrics['midface_symmetry_index'] = self._safe_divide(rms_diff_x, distances.get('g_face_width', EPSILON) * ref_distance) # Использование реального g_face_width
        else:
            metrics['midface_symmetry_index'] = 0.0

        # 18. ocular_to_nasal_angle: угол между линиями (36-42, 39-nasion)
        # 36: внешний уголок правого глаза
        # 42: внутренний уголок левого глаза
        # 39: внутренний уголок правого глаза
        # Nasion: точка 27
        
        # Вектор 1: 36 -> 42 (внешний правый к внутреннему левому)
        v1_ocular_nasal = v[42, :] - v[36, :]
        # Вектор 2: 39 -> 27 (внутренний правый к переносице)
        v2_ocular_nasal = v[27, :] - v[39, :]

        dot_product_ocular_nasal = np.dot(v1_ocular_nasal, v2_ocular_nasal)
        magnitude_v1_ocular_nasal = np.linalg.norm(v1_ocular_nasal)
        magnitude_v2_ocular_nasal = np.linalg.norm(v2_ocular_nasal)

        metrics['ocular_to_nasal_angle_degrees'] = self._safe_divide(math.degrees(math.acos(np.clip(self._safe_divide(dot_product_ocular_nasal, (magnitude_v1_ocular_nasal * magnitude_v2_ocular_nasal)), -1.0, 1.0))), 1.0) # Always define degrees as float.


        # 19. nasolabial_angle_cos: cos(угол subnasale-pronasale-labiale superius)
        # Subnasale: точка 33
        # Pronasale: точка 30
        # Labiale superius: точка 51
        
        # Вектор 1: pronasale -> subnasale
        v1_nasolabial = v[33, :] - v[30, :]
        # Вектор 2: pronasale -> labiale superius
        v2_nasolabial = v[51, :] - v[30, :]

        dot_product_nasolabial = np.dot(v1_nasolabial, v2_nasolabial)
        magnitude_v1_nasolabial = np.linalg.norm(v1_nasolabial)
        magnitude_v2_nasolabial = np.linalg.norm(v2_nasolabial)

        metrics['nasolabial_angle_cos'] = self._safe_divide(dot_product_nasolabial, (magnitude_v1_nasolabial * magnitude_v2_nasolabial))

        # 20. upper_third_width_ratio: glabella-zygion span ÷ g_face_width
        # Glabella (точка 27) не очень подходит для "glabella-zygion span".
        # "Glabella-zygion span" обычно относится к ширине верхней части лица.
        # Zygion: точки 1 и 15.
        # Ширина верхнего отдела лица: расстояние между zygion (скулами).
        # Используем v[1] и v[15] как zygion.
        upper_third_width = np.linalg.norm(v[1, :] - v[15, :]) # Это уже было как zygion_span
        
        metrics['upper_third_width_ratio'] = self._safe_divide((upper_third_width / ref_distance), (face_width_g / ref_distance))
        
        # 21. eyebrow_eye_distance_ratio: среднее расстояние бровь-глаз / avg_eye_width
        # Средняя точка брови: (v[19] + v[24]) / 2 (правая бровь), (v[22] + v[25]) / 2 (левая бровь)
        # Центр глаза: (v[39] + v[42]) / 2 (для межзрачкового)
        right_brow_center = (v[19, :] + v[20, :] + v[21, :]) / 3  # Усредняем 3 точки для правой брови
        left_brow_center = (v[22, :] + v[23, :] + v[24, :]) / 3   # Усредняем 3 точки для левой брови

        right_eye_center_inner = v[39, :]
        left_eye_center_inner = v[42, :]
        
        # Используем ближайшую точку глаза к брови для расстояния
        dist_right_brow_eye = np.linalg.norm(right_brow_center - right_eye_center_inner)
        dist_left_brow_eye = np.linalg.norm(left_brow_center - left_eye_center_inner)
        
        avg_brow_eye_distance = (dist_right_brow_eye + dist_left_brow_eye) / 2

        metrics['eyebrow_eye_distance_ratio'] = self._safe_divide((avg_brow_eye_distance / ref_distance), distances.get('avg_eye_width', EPSILON))

        # 22. eyebrow_length_eye_ratio: средняя длина брови / средняя длина глаза
        # Длина правой брови (от 17 до 21)
        len_right_brow = np.linalg.norm(v[17, :] - v[21, :])
        # Длина левой брови (от 22 до 26)
        len_left_brow = np.linalg.norm(v[22, :] - v[26, :])
        avg_brow_length = (len_right_brow + len_left_brow) / 2

        # Длина правого глаза (от 36 до 39)
        len_right_eye = np.linalg.norm(v[36, :] - v[39, :])
        # Длина левого глаза (от 42 до 45)
        len_left_eye = np.linalg.norm(v[42, :] - v[45, :])
        avg_eye_length = (len_right_eye + len_left_eye) / 2

        metrics['eyebrow_length_eye_ratio'] = self._safe_divide((avg_brow_length / ref_distance), (avg_eye_length / ref_distance))
        
        return metrics

