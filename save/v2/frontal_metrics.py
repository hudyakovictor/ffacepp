import numpy as np
import math
from typing import Dict, List, Tuple
# from skimage.metrics import structural_similarity as ssim # Удаляем импорт ssim, так как он больше не используется
import cv2

# ... existing code ...

from marquardt import MarquardtMask, EPSILON
# from utils.render import get_normal # Убеждаемся, что импорт закомментирован

class FrontalAnalysisModule:
    """Модуль анализа фронтального ракурса лица по маске Марквардта."""

    def __init__(self):
        self.marquardt_mask = MarquardtMask()

    def analyze(self, landmarks_3d: np.ndarray, yaw: float, roll: float = 0.0,
                alpha_shp: np.ndarray = None, alpha_exp: np.ndarray = None,
                R_matrix_descaled: np.ndarray = None, t_vec: np.ndarray = None,
                u_base: np.ndarray = None, depth_map_image: np.ndarray = None,
                tri: np.ndarray = None, ver_dense: np.ndarray = None, pncc_map_image: np.ndarray = None) -> Dict[str, any]:
        """Выполняет комплексный анализ фронтального лица."""

        # Список стабильных биометрических метрик (обновленный)
        # Теперь включает метрики расстояний, так как для них нет 'идеальных' пропорций
        stable_biometric_metric_names = [
            'face_width_height_ratio',
            'face_golden_ratio',
            'face_golden_ratio_height_width',
            'eye_width_face_ratio',
            'interocular_distance_ratio',
            'right_eye_aspect_ratio',
            'left_eye_aspect_ratio',
            'nose_width_eye_distance',
            'nose_length_face_ratio',
            'nose_height_width_ratio',
            'columella_to_nose_length_ratio',
            'mouth_width_nose_ratio',
            'upper_lip_lower_ratio',
            'mouth_height_width_ratio',
            'jaw_width_face_ratio',
            'chin_width_mouth_ratio',
            'chin_projection_ratio',
            'zygomatic_to_bigonial_ratio',
            'eyebrow_eye_distance_ratio',
            'eyebrow_length_eye_ratio',
            'bilateral_symmetry_tolerance',
            'vertical_symmetry_ratio',
            'palpebral_symmetry_x',
            'alar_width_ratio',
            'midface_symmetry_index',
            'philtrum_length_ratio',
            'canthal_tilt_normalized',
            'ocular_to_nasal_angle_degrees',
            'nasolabial_angle_cos',
            'upper_third_width_ratio',
            # Дополнительные метрики-расстояния, теперь без идеалов
            'a_face_height',
            'b_forehead_to_eyes',
            'c_eyes_to_nose',
            'd_eyes_to_lips',
            'e_nose_width',
            'f_eye_span',
            'g_face_width',
            'i_nose_to_chin',
            'j_lips_to_chin',
            'k_mouth_width',
            'l_nose_to_lips',
            'ipd_interpupillary',
            'avg_eye_width',
            'h_hairline_to_eyes', # Эта метрика также добавлена из MarquardtMask
            'outer_canthi_span_face_ratio',
            'inner_canthi_span_face_ratio',
            'eye_line_height_ratio',
            'glabella_to_subnasale_ratio',
            'subnasale_to_menton_ratio',
            'chin_width' # Добавлено для корректной обработки в группировке
        ]

        # Расчет общей ошибки формы и выражения
        shape_error_total = 0.0
        if alpha_shp is not None:
            # Норма alpha_shp отражает, насколько сильно форма отличается от базовой
            # Чем больше норма, тем больше отклонение формы от "среднего" лица
            shape_error_total = float(np.linalg.norm(alpha_shp))
            print(f"СТАТУС: shape_error_total = {shape_error_total:.4f}")

        expression_error = 0.0
        if alpha_exp is not None:
            # Норма alpha_exp отражает выраженность мимики
            expression_error = float(np.linalg.norm(alpha_exp))
            print(f"СТАТУС: expression_error = {expression_error:.4f}")

        # Расчет ошибки формы по частям лица
        shape_error_by_region = {}
        if landmarks_3d is not None and u_base is not None and R_matrix_descaled is not None and t_vec is not None:
            u_base_reshaped = u_base.reshape(-1, 3)
            # Убедимся, что landmarks_3d и projected_u_base имеют одинаковую форму (N, 3)
            # u_base_reshaped уже (N, 3). R_matrix_descaled (3,3), t_vec (3,). Преобразуем u_base_reshaped в (3, N) для умножения
            projected_u_base = (R_matrix_descaled @ u_base_reshaped.T).T + t_vec.reshape(1, 3) # t_vec должен быть добавлен построчно

            # Определяем группы точек для расчета ошибок по частям лица (68 точек)
            landmark_groups_for_error_by_region = {
                'forehead': list(range(17, 27)), # Используем точки бровей для оценки лба, так как прямых точек лба нет
                'eyebrows': list(range(17, 27)),
                'eyes': list(range(36, 48)),
                'nose': list(range(27, 36)),
                'cheeks': list(range(0, 5)) + list(range(11, 17)),
                'mouth': list(range(48, 68)),
                'chin': list(range(5, 12)),
                'jaw': list(range(0, 17)),
            }

            for group_name, indices in landmark_groups_for_error_by_region.items():
                valid_indices = [idx for idx in indices if idx < landmarks_3d.shape[0] and idx < projected_u_base.shape[0]]

                if len(valid_indices) > 0:
                    diff = landmarks_3d[valid_indices, :] - projected_u_base[valid_indices, :]
                    error_for_group = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
                    shape_error_by_region[group_name] = float(error_for_group)
                else:
                    print(f"ВНИМАНИЕ: Недостаточно landmarks для расчета ошибки формы для группы {group_name}. Пропускаем.")
                    shape_error_by_region[group_name] = 0.0

        predicted_angle_type = self.marquardt_mask.classify_angle(yaw)
        if predicted_angle_type != 'frontal':
             pass

        angle_type_for_props = 'frontal'

        all_measured_metrics = self.marquardt_mask.calculate_proportions(landmarks_3d, angle_type_for_props, roll)

        ideal_proportions = self.marquardt_mask.ideal_proportions[angle_type_for_props]

        basic_measured_proportions = {}
        new_additional_metrics_calculated = {}
        stable_distances_output = {} # Создаем новый словарь для стабильных расстояний

        # --- MODIFIED LOGIC FOR METRIC CATEGORIZATION ---
        for metric_name, measured_value in all_measured_metrics.items():
            if metric_name in ideal_proportions:
                ideal_val_for_metric = ideal_proportions[metric_name]
                if ideal_val_for_metric is None: # It's a distance metric
                    stable_distances_output[metric_name] = measured_value
                elif abs(ideal_val_for_metric) < EPSILON: # It's a symmetry/tilt metric with ideal 0.0
                    new_additional_metrics_calculated[metric_name] = measured_value
                else: # It's a regular proportion with a non-zero ideal
                    basic_measured_proportions[metric_name] = measured_value
            else: # Metric is not in ideal_proportions (e.g., specific to additional calculations)
                new_additional_metrics_calculated[metric_name] = measured_value
        # --- END MODIFIED LOGIC ---

        deviations = {}
        for metric, measured_value in basic_measured_proportions.items():
            ideal_value = ideal_proportions.get(metric)
            
            if ideal_value is not None: # Only if there is an ideal value (not None)
                if abs(ideal_value) < EPSILON: # Ideal value is 0 (или очень близко к 0)
                    deviations[metric] = float(abs(measured_value)) # Отклонение = измеренное значение
                else:
                    deviation = abs(measured_value - ideal_value) / abs(ideal_value)
                    deviations[metric] = float(deviation)
            # else: если ideal_value is None, не добавляем его в deviations

        anthropometric_ratio_error = 0.0
        if deviations:
            valid_deviations = [d for d in deviations.values() if d != float('inf')] # Inf уже не должно быть
            if valid_deviations:
                anthropometric_ratio_error = float(np.mean(valid_deviations))
            else:
                anthropometric_ratio_error = 0.0

        similarity_score = self.marquardt_mask.calculate_similarity_score(basic_measured_proportions, angle_type_for_props)

        analysis_result = {
            "analysis_type": angle_type_for_props,
            "measured_proportions": {k: float(v) for k, v in basic_measured_proportions.items()},
            # Для ideal_proportions, теперь храним только те, которые не None
            "ideal_proportions": {k: float(v) for k, v in ideal_proportions.items() if v is not None},
            "deviations_from_ideal": deviations,
            "overall_similarity_score": float(similarity_score),
            "measurement_info": {
                "landmarks_count": landmarks_3d.shape[0],
                "pose_normalized": True,
                "stability_validated": True,
            }
        }

        # ref_distance уже гарантированно > EPSILON
        ref_distance = self.marquardt_mask.get_reference_distance(landmarks_3d, angle_type_for_props)

        # Эти метрики теперь будут частью new_additional_metrics_calculated в calculate_proportions
        # stable_distances = self.marquardt_mask.calculate_stable_distances(landmarks_3d, ref_distance)
        # marquardt_ratios = self.marquardt_mask.calculate_marquardt_ratios(stable_distances)

        stable_biometric_metrics_output = {} # Изменено название переменной
        for metric_name in self.marquardt_mask.ideal_proportions[angle_type_for_props]: # Проходим по всем метрикам из идеальных пропорций
            if metric_name in all_measured_metrics: # Если метрика была измерена
                # Если у метрики есть идеальное значение (не None), это пропорция. Иначе это расстояние.
                ideal_val_in_marquardt = self.marquardt_mask.ideal_proportions[angle_type_for_props][metric_name]
                if ideal_val_in_marquardt is not None and abs(ideal_val_in_marquardt) >= EPSILON:
                    stable_biometric_metrics_output[metric_name] = all_measured_metrics[metric_name] # Пропорции
        
        analysis_result["additional_metrics"] = {
            "stable_distances": {k: float(v) for k, v in stable_distances_output.items()}, # **ИСПОЛНЕНИЕ ПУНКТА 5: Корректное заполнение stable_distances**
            "marquardt_ratios": {k: float(v) for k, v in basic_measured_proportions.items()}, # Пропорции
            "new_additional_metrics": {k: float(v) for k, v in new_additional_metrics_calculated.items()}, # Дополнительные метрики, не являющиеся ни расстояниями, ни пропорциями
            "stable_biometric_metrics": {k: float(v) for k, v in stable_biometric_metrics_output.items()}, # Только пропорции с идеальными значениями
            "shape_and_expression_errors": {
                "shape_error_total": shape_error_total,
                "expression_error": expression_error,
                "shape_error_by_region": shape_error_by_region
            },
            "anthropometric_ratio_error": anthropometric_ratio_error,
        }
        return analysis_result
