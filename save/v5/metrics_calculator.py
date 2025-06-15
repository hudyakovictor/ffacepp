import numpy as np
from scipy import stats
from typing import List, Dict
import random
import logging
from core_config import get_identity_signature_metrics # Импортируем для получения списка метрик

logger = logging.getLogger(__name__)

def calculate_temporal_angle(p1, p2, p3):
    """Рассчитывает угол между тремя 3D точками"""
    v1 = p1 - p2
    v2 = p3 - p2
    # Защита от деления на ноль для нормализации векторов
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0 # Или np.nan, в зависимости от требуемого поведения при нулевом векторе

    cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

def calculate_angle_between_3d_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Вычисляет угол между тремя точками в 3D пространстве."""
    vec1 = p1 - p2
    vec2 = p3 - p2
    cosine_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6) # Добавлено 1e-6 для защиты от деления на ноль
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1, 1)))
    return angle

def calculate_euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Вычисляет евклидово расстояние между двумя точками в 3D пространстве."""
    return np.linalg.norm(p1 - p2)

# =====================
# Нормализация метрик по стабильным опорным точкам
# =====================
def normalize_landmarks_by_robust_statistics(landmarks_3d: np.ndarray, reference_indices: List[int]) -> np.ndarray:
    """
    Нормализация ландмарков по робастным статистикам.
    Использует медиану и медианное абсолютное отклонение (MAD) для устойчивости к выбросам.
    landmarks_3d: np.ndarray [N, 3] - исходные ландмарки
    reference_indices: List[int] - индексы опорных точек
    Возвращает: нормализованные ландмарки [N, 3]
    """
    if landmarks_3d.size == 0 or len(reference_indices) < 3:
        raise ValueError("Недостаточно опорных точек для нормализации")
    
    reference_points = landmarks_3d[reference_indices]
    
    # Робастный центр (медиана вместо среднего)
    center = np.median(reference_points, axis=0)
    
    # Робастный масштаб (MAD - Median Absolute Deviation)
    deviations = np.abs(reference_points - center)
    scale = np.median(deviations)
    
    if scale < 1e-6:
        raise ValueError("Нулевой масштаб нормализации")
    
    normalized = (landmarks_3d - center) / scale
    return normalized

# =====================
# Вычисление 15 метрик идентификации для конкретного ракурса
# =====================
def calculate_identity_signature_for_pose(normalized_landmarks, pose_category):
    """
    Вычисляет подпись личности для конкретного ракурса:
    15 ключевых метрик, уникальных для каждого человека
    все нормализованы по стабильным анатомическим точкам
    normalized_landmarks: np.ndarray [N, 3]
    pose_category: строка ('Frontal', ...)
    Возвращает: dict с 15 метриками
    """
    metrics = {}

    # Проверка валидности входных данных
    if normalized_landmarks is None or len(normalized_landmarks) < 68:
        raise ValueError("Недостаточно landmarks для расчета метрик (ожидается 68).")
    
    # Безопасные вычисления для общих знаменателей
    face_height_raw = np.linalg.norm(normalized_landmarks[8] - normalized_landmarks[27])
    if face_height_raw < 1e-6:
        raise ValueError("Невозможно нормализовать: нулевое расстояние между опорными точками для высоты лица.")
    face_height = face_height_raw

    face_width_raw = np.linalg.norm(normalized_landmarks[1] - normalized_landmarks[15])
    if face_width_raw < 1e-6:
        raise ValueError("Невозможно нормализовать: нулевое расстояние между опорными точками для ширины лица.")
    face_width = face_width_raw

    # Геометрия черепа
    metrics['skull_width_ratio'] = np.linalg.norm(normalized_landmarks[1] - normalized_landmarks[15]) / (face_width + 1e-6)
    metrics['temporal_bone_angle'] = calculate_temporal_angle(
        normalized_landmarks[0], normalized_landmarks[17], normalized_landmarks[1]
    )
    metrics['zygomatic_arch_width'] = np.linalg.norm(normalized_landmarks[3] - normalized_landmarks[13]) / (face_height + 1e-6)
    metrics['orbital_depth'] = (np.mean(normalized_landmarks[[37,38,40,41],2]) - np.mean(normalized_landmarks[[19,24],2])) # Примерная глубина глазницы
    metrics['occipital_curvature'] = np.std([normalized_landmarks[0,2], normalized_landmarks[8,2], normalized_landmarks[16,2]]) # Исправлено: опечатка в названии метрики
    # Пропорции лица
    metrics['forehead_height_ratio'] = np.linalg.norm(normalized_landmarks[27] - normalized_landmarks[19]) / (face_height + 1e-6)
    metrics['nose_width_ratio'] = np.linalg.norm(normalized_landmarks[31] - normalized_landmarks[35]) / (face_width + 1e-6)
    metrics['eye_distance_ratio'] = np.linalg.norm(normalized_landmarks[36] - normalized_landmarks[45]) / (face_width + 1e-6)
    metrics['mouth_width_ratio'] = np.linalg.norm(normalized_landmarks[48] - normalized_landmarks[54]) / (face_width + 1e-6)
    metrics['chin_width_ratio'] = np.linalg.norm(normalized_landmarks[6] - normalized_landmarks[10]) / (face_width + 1e-6)
    # Костная структура
    metrics['temple_width_ratio'] = np.linalg.norm(normalized_landmarks[0] - normalized_landmarks[16]) / (face_width + 1e-6)
    metrics['cheekbone_width_ratio'] = np.linalg.norm(normalized_landmarks[3] - normalized_landmarks[13]) / (face_width + 1e-6)
    jaw_vec1 = normalized_landmarks[5] - normalized_landmarks[8]
    jaw_vec2 = normalized_landmarks[11] - normalized_landmarks[8]
    jaw_angle = np.degrees(np.arccos(np.clip(
        np.dot(jaw_vec1, jaw_vec2) / (np.linalg.norm(jaw_vec1) * np.linalg.norm(jaw_vec2) + 1e-6), -1, 1)))
    metrics['jaw_angle_ratio'] = jaw_angle / 120.0 # Нормализация относительно среднего угла челюсти
    # Индекс симметрии
    left = normalized_landmarks[[0,1,2,3,4,5,6,7,8,17,18,19,20,21,36,37,38,39,40,41,48,49,50,59,58,67]]
    right = normalized_landmarks[[16,15,14,13,12,11,10,9,8,26,25,24,23,22,45,44,43,42,47,46,54,53,52,55,56,57]]
    center_x = np.mean(normalized_landmarks[:,0])
    asymmetry = [np.linalg.norm(l - np.array([2*center_x - r[0], r[1], r[2]])) for l, r in zip(left, right)]
    metrics['facial_symmetry_index'] = 1.0 / (1.0 + np.mean(asymmetry))
    skull_depth = np.max(normalized_landmarks[:,2]) - np.min(normalized_landmarks[:,2])
    skull_width_for_depth_ratio = np.linalg.norm(normalized_landmarks[0] - normalized_landmarks[16]) # Ширина черепа для расчета skull_depth_ratio
    if skull_width_for_depth_ratio + 1e-6 < 1e-10: # Проверка на потенциальное деление на ноль
        metrics['skull_depth_ratio'] = 0.0 # Устанавливаем в 0.0 или другое осмысленное значение
    else:
        metrics['skull_depth_ratio'] = skull_depth / (skull_width_for_depth_ratio + 1e-6) # Исправлено: деление на ширину черепа

    # Пропорции лица (facial_proportions_signature)
    # golden_ratio_deviation: Вызываем функцию для получения отклонений
    golden_ratios_data = calculate_proportional_golden_ratios(normalized_landmarks)
    metrics['golden_ratio_deviation'] = golden_ratios_data.get('proportion_anomaly', False) # True/False или можно усреднить golden_deviations

    # nasolabial_angle: Угол между носогубным треугольником (точки 30, 33, 51)
    # 30: tip of nose, 33: subnasale, 51: upper lip (philtrum)
    metrics['nasolabial_angle'] = calculate_angle_between_3d_points(normalized_landmarks[30], normalized_landmarks[33], normalized_landmarks[51])

    # orbital_index: Соотношение высоты и ширины глазницы
    # Левый глаз: ширина (36-39), высота (38-40)
    # Правый глаз: ширина (42-45), высота (43-47)
    left_orbital_width = np.linalg.norm(normalized_landmarks[36] - normalized_landmarks[39])
    left_orbital_height = np.linalg.norm(normalized_landmarks[38] - normalized_landmarks[40])
    right_orbital_width = np.linalg.norm(normalized_landmarks[42] - normalized_landmarks[45])
    right_orbital_height = np.linalg.norm(normalized_landmarks[43] - normalized_landmarks[47])

    orbital_index_left = left_orbital_height / (left_orbital_width + 1e-6)
    orbital_index_right = right_orbital_height / (right_orbital_width + 1e-6)
    metrics['orbital_index'] = (orbital_index_left + orbital_index_right) / 2

    # chin_projection_ratio: Проекция подбородка
    # Расстояние от точки подбородка (8) до проекции на вертикальную ось, проходящую через переносицу (27)
    # Относительно общей глубины лица
    chin_tip = normalized_landmarks[8]
    nasion_x = normalized_landmarks[27, 0]
    chin_projection_distance = abs(chin_tip[0] - nasion_x) # Горизонтальная проекция
    # Используем среднее расстояние от носа до подбородка как нормализатор, или глубину лица
    face_depth = calculate_euclidean_distance(normalized_landmarks[27], normalized_landmarks[8]) + 1e-6 # Изменено на новую функцию
    metrics['chin_projection_ratio'] = chin_projection_distance / face_depth

    # Костная структура (bone_structure_signature)
    # inter_pupillary_distance_ratio: Межзрачковое расстояние (39-42)
    metrics['inter_pupillary_distance_ratio'] = calculate_euclidean_distance(normalized_landmarks[39], normalized_landmarks[42]) / (face_width + 1e-6) # Изменено на новую функцию

    # zygomatic_angle: Угол скуловой кости (например, угол, образованный точками 2, 8, 14)
    metrics['zygomatic_angle'] = calculate_angle_between_3d_points(normalized_landmarks[2], normalized_landmarks[8], normalized_landmarks[14]) # Изменено на новую функцию

    # Фильтруем метрики, чтобы возвращать только 15 ключевых
    required_metrics_names = []
    identity_signature_configs = get_identity_signature_metrics()
    for sig_type in identity_signature_configs.values():
        required_metrics_names.extend(sig_type['metrics'])

    final_metrics = {name: metrics[name] for name in required_metrics_names if name in metrics}
    
    return final_metrics

# =====================
# Метрики костной структуры черепа (неизменные после 25 лет)
# =====================
def calculate_cranial_stability_metrics(landmarks_3d: np.ndarray) -> Dict:
    """
    Метрики костной структуры черепа, неизменные после 25 лет:
    Цефалический индекс и его классификация по медицинским стандартам.
    landmarks_3d: np.ndarray [N, 3]
    Возвращает: dict с краниальными метриками
    """
    if landmarks_3d.shape[0] < 68:
        raise ValueError("Требуется 68 landmarks для краниального анализа")
    
    metrics = {}
    
    # Истинная ширина черепа (бизигоматическая ширина)
    # Используем точки скуловых дуг
    # Важно: для истинного цефалического индекса нужна максимальная ширина черепа,
    # а не бизигоматическая. Используем приближение по крайним точкам лица.
    skull_breadth = np.linalg.norm(landmarks_3d[0] - landmarks_3d[16]) # Приближение ширины черепа

    # Длина черепа (назион-опистокранион)
    nasion = landmarks_3d[27]  # Переносица
    # Аппроксимация затылочной точки через контур лица
    occiput_approx = landmarks_3d[8] + (landmarks_3d[8] - nasion) * 0.3
    cranial_length = np.linalg.norm(nasion - occiput_approx)
    
    # Цефалический индекс (медицински корректный)
    if cranial_length > 1e-6:
        cephalic_index = (skull_breadth / cranial_length) * 100
    else:
        cephalic_index = 0.0
    
    # Классификация по медицинским стандартам
    if cephalic_index < 75:
        skull_type = 'dolichocephalic'  # Длинноголовый
    elif cephalic_index > 80:
        skull_type = 'brachycephalic'   # Короткоголовый
    else:
        skull_type = 'mesocephalic'     # Средний тип
    
    metrics.update({
        'skull_breadth': skull_breadth, # Новое название
        'cranial_length': cranial_length,
        'cephalic_index': cephalic_index,
        'skull_type': skull_type
    })
    
    # Также сохраним исходную бизигоматическую ширину, если она используется для других метрик
    zygomatic_left = landmarks_3d[2]   # Левая скуловая дуга
    zygomatic_right = landmarks_3d[14] # Правая скуловая дуга
    metrics['bizygomatic_width'] = np.linalg.norm(zygomatic_left - zygomatic_right)
    
    return metrics

# =====================
# Пропорции золотого сечения по маске Марквардта
# =====================
def calculate_proportional_golden_ratios(landmarks):
    """
    Пропорции золотого сечения по маске Марквардта:
    facial_thirds, fifths, diagonal_proportions, все относительно базовых измерений
    landmarks: np.ndarray [N, 3]
    Возвращает: dict с пропорциями и отклонениями от золотого сечения
    """
    golden_ratio = 1.618
    ratios = {}

    if landmarks is None or len(landmarks) < 68: # Предполагаем, что для этих расчетов также нужно 68 landmarks
        raise ValueError("Недостаточно landmarks для расчета пропорций золотого сечения (ожидается 68).")

    # Вертикальные трети лица
    face_height_raw = np.linalg.norm(landmarks[8] - landmarks[27])
    if face_height_raw < 1e-6:
        raise ValueError("Невозможно рассчитать пропорции: нулевая высота лица.")
    face_height = face_height_raw

    upper_third = np.linalg.norm(landmarks[27] - landmarks[19])
    middle_third = np.linalg.norm(landmarks[19] - landmarks[33])
    lower_third = np.linalg.norm(landmarks[33] - landmarks[8])
    thirds = np.array([upper_third, middle_third, lower_third])
    ratios['facial_thirds'] = thirds / (face_height + 1e-6)
    
    # Горизонтальные пяты лица
    face_width_raw = np.linalg.norm(landmarks[1] - landmarks[15])
    if face_width_raw < 1e-6:
        raise ValueError("Невозможно рассчитать пропорции: нулевая ширина лица.")
    face_width = face_width_raw

    fifth = face_width / 5
    ratios['facial_fifths'] = fifth / (face_width + 1e-6)
    # Диагональные пропорции (пример: скуловая - подбородок)
    diag1 = np.linalg.norm(landmarks[3] - landmarks[8])
    diag2 = np.linalg.norm(landmarks[13] - landmarks[8])
    ratios['diagonal_proportions'] = np.array([diag1, diag2]) / (face_height + 1e-6)
    # Отклонения от золотого сечения
    
    # Избегаем деления на ноль, если middle_third или lower_third очень малы
    actual_ratios = []
    if middle_third > 1e-6: actual_ratios.append(upper_third / middle_third)
    if lower_third > 1e-6: actual_ratios.append(middle_third / lower_third)
    
    if len(actual_ratios) > 0:
        golden_deviations = np.abs(np.array(actual_ratios) - golden_ratio) / golden_ratio
        ratios['golden_deviations'] = golden_deviations
        ratios['proportion_anomaly'] = np.any(golden_deviations > 0.1)
    else:
        ratios['golden_deviations'] = np.array([0.0])
        ratios['proportion_anomaly'] = False

    return ratios

# =====================
# Балл биометрической уникальности набора метрик
# =====================
def calculate_biometric_uniqueness_score(metrics_set):
    """
    Вычисляет балл биометрической уникальности набора метрик:
    оценивает, насколько уникален данный набор характеристик
    metrics_set: np.ndarray или list
    Возвращает: float (0-1, где 1 — максимально уникально)
    """
    try:
        # Чем выше стандартное отклонение между метриками — тем менее уникально
        std = np.std(metrics_set)
        uniqueness = 1.0 / (1.0 + std)  # Чем меньше std, тем выше уникальность
        return uniqueness
    except Exception as e:
        print(f"Ошибка при вычислении уникальности: {e}")
        return 0.0

# =====================
# Корреляционный анализ между метриками
# =====================
def calculate_inter_metric_correlations(metrics_history):
    """
    Анализ корреляций между метриками для выявления аномальных паттернов
    metrics_history: dict {metric_name: [values]}
    Возвращает: матрица корреляций (numpy array)
    """
    try:
        metric_names = list(metrics_history.keys())
        values = np.array([metrics_history[name] for name in metric_names])
        corr_matrix = np.corrcoef(values)
        return corr_matrix, metric_names
    except Exception as e:
        print(f"Ошибка при анализе корреляций: {e}")
        return None, []

# =====================
# Bootstrap оценка доверительных интервалов для метрик
# =====================
def bootstrap_metric_confidence(metric_values: List[float], n_bootstrap: int = 1000, alpha: float = 0.05) -> Dict:
    """
    Выполняет bootstrap-анализ для оценки доверительных интервалов для средней метрики.
    
    Args:
        metric_values (List[float]): Список значений метрики.
        n_bootstrap (int): Количество bootstrap-выборок.
        alpha (float): Уровень значимости для доверительного интервала (например, 0.05 для 95% CI).

    Returns:
        Dict: Словарь с результатами bootstrap-анализа, включая среднее, стандартное отклонение,
              доверительный интервал и другие статистики.
    """
    if not metric_values:
        return {
            'mean': 0.0,
            'std': 0.0,
            'lower_ci': 0.0,
            'upper_ci': 0.0,
            'is_significant': False, # Недостаточно данных для значимости
            'bootstrap_means': []
        }
    
    data = np.array(metric_values)
    n_samples = len(data)
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        # Выборка с возвращением
        sample = random.choices(data, k=n_samples)
        bootstrap_means.append(np.mean(sample))
        
    bootstrap_means = np.array(bootstrap_means)
    
    # Основные статистики
    mean_estimate = np.mean(bootstrap_means)
    std_estimate = np.std(bootstrap_means)
    
    # Доверительный интервал (percentile method)
    lower_bound = np.percentile(bootstrap_means, (alpha / 2) * 100)
    upper_bound = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

    # Проверка на статистическую значимость (если 0 не входит в CI)
    is_significant = not (lower_bound <= 0.0 <= upper_bound)

    return {
        'mean': float(mean_estimate),
        'std': float(std_estimate),
        'lower_ci': float(lower_bound),
        'upper_ci': float(upper_bound),
        'is_significant': bool(is_significant),
        'bootstrap_means': bootstrap_means.tolist()
    }

# =====================
# Детекция выбросов в последовательности метрик
# =====================
def detect_metric_outliers(metric_sequence):
    """
    Выявление выбросов в последовательности метрик
    metric_sequence: list или np.ndarray
    Возвращает: индексы выбросов
    """
    try:
        metric_sequence = np.array(metric_sequence)
        z_scores = np.abs(stats.zscore(metric_sequence))
        outlier_indices = np.where(z_scores > 2.5)[0]
        return outlier_indices
    except Exception as e:
        print(f"Ошибка при детекции выбросов: {e}")
        return []

# =====================
# Балл стабильности метрики во времени
# =====================
def calculate_metric_stability_score(metric_history):
    """
    Вычисляет балл стабильности метрики во времени
    metric_history: list или np.ndarray
    Возвращает: float (0-1, где 1 — максимально стабильно)
    """
    try:
        metric_history = np.array(metric_history)
        if len(metric_history) < 2:
            return 1.0
        cv = np.std(metric_history) / (np.mean(metric_history) + 1e-6)
        # Чем меньше коэффициент вариации, тем выше стабильность
        stability = max(0, 1 - cv)
        return stability
    except Exception as e:
        print(f"Ошибка при расчёте стабильности: {e}")
        return 0.0