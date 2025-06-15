import numpy as np
import math
from typing import Dict, List, Tuple

# ... existing code ...

from marquardt import MarquardtMask

class FrontalEdgeAnalysisModule:
    """Модуль анализа полуфронтального ракурса лица по маске Марквардта."""

    def __init__(self):
        self.marquardt_mask = MarquardtMask()

    def analyze(self, landmarks_3d: np.ndarray, yaw: float, roll: float = 0.0) -> Dict[str, any]:
        """Выполняет комплексный анализ полуфронтального лица."""
        
        # Список 15 стабильных биометрических метрик (используем тот же список для консистентности)
        stable_biometric_metric_names = [
            'outer_canthi_span_face_ratio',
            'upper_third_width_ratio',
            'g_face_width',
            'inner_canthi_span_face_ratio',
            'ipd_interpupillary',
            'midface_symmetry_index',
            'f_eye_span',
            'eye_width_face_ratio',
            'b_forehead_to_eyes',
            'h_hairline_to_eyes',
            'e_nose_width',
            'nose_width_eye_distance',
            'interocular_distance_ratio',
            'avg_eye_width',
            'j_lips_to_chin'
        ]

        predicted_angle_type = self.marquardt_mask.classify_angle(yaw)
        # angle_type_for_props = 'frontal_edge' # Используем фактический тип для получения идеальных пропорций
        angle_type_for_props = predicted_angle_type if predicted_angle_type in self.marquardt_mask.ideal_proportions else 'frontal_edge'

        all_measured_metrics = self.marquardt_mask.calculate_proportions(landmarks_3d, angle_type_for_props, roll)

        ideal_proportions = self.marquardt_mask.ideal_proportions.get(angle_type_for_props, {})

        basic_measured_proportions = {}
        new_additional_metrics_calculated = {}

        for metric_name, measured_value in all_measured_metrics.items():
            if metric_name in ideal_proportions:
                basic_measured_proportions[metric_name] = measured_value
            else:
                new_additional_metrics_calculated[metric_name] = measured_value

        deviations = {}
        for metric, measured_value in basic_measured_proportions.items():
            ideal_value = ideal_proportions.get(metric, 0.0) # Используем .get для безопасного доступа
            if ideal_value != 0:
                deviation = abs(measured_value - ideal_value) / abs(ideal_value)
                deviations[metric] = float(deviation)
            else:
                deviations[metric] = float('inf') if measured_value != 0 else 0.0

        similarity_score = self.marquardt_mask.calculate_similarity_score(basic_measured_proportions, angle_type_for_props)

        analysis_result = {
            "analysis_type": angle_type_for_props,
            "measured_proportions": {k: float(v) for k, v in basic_measured_proportions.items()},
            "ideal_proportions": {k: float(v) for k, v in ideal_proportions.items()},
            "deviations_from_ideal": deviations,
            "overall_similarity_score": float(similarity_score),
            "measurement_info": {
                "landmarks_count": landmarks_3d.shape[0],
                "pose_normalized": True,
                "stability_validated": True,
            }
        }

        ref_distance = self.marquardt_mask.get_reference_distance(landmarks_3d, angle_type_for_props)
        stable_distances = self.marquardt_mask.calculate_stable_distances(landmarks_3d, ref_distance)
        marquardt_ratios = self.marquardt_mask.calculate_marquardt_ratios(stable_distances)

        stable_biometric_metrics = {}
        for metric_name in stable_biometric_metric_names:
            if metric_name in all_measured_metrics:
                stable_biometric_metrics[metric_name] = all_measured_metrics[metric_name]
            elif metric_name in stable_distances:
                stable_biometric_metrics[metric_name] = stable_distances[metric_name]
            elif metric_name in marquardt_ratios:
                stable_biometric_metrics[metric_name] = marquardt_ratios[metric_name]
            elif metric_name in new_additional_metrics_calculated:
                stable_biometric_metrics[metric_name] = new_additional_metrics_calculated[metric_name]

        analysis_result["additional_metrics"] = {
            "stable_distances": {k: float(v) for k, v in stable_distances.items()},
            "marquardt_ratios": {k: float(v) for k, v in marquardt_ratios.items()},
            "new_additional_metrics": {k: float(v) for k, v in new_additional_metrics_calculated.items()},
            "stable_biometric_metrics": {k: float(v) for k, v in stable_biometric_metrics.items()}
        }

        return analysis_result
