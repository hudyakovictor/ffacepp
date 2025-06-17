"""
MetricsCalculator - Калькулятор 15 метрик идентичности
Версия: 2.0
Дата: 2025-06-15
Исправлены все критические ошибки согласно правкам
"""

import numpy as np
import cv2
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import json
import math
from scipy.spatial.distance import euclidean
from scipy import stats

# --- ЦВЕТА КОНСОЛИ (Повторяются для каждого модуля, чтобы быть автономными) ---
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    PURPLE = "\033[95m"
    WHITE = "\033[97m"

# --- КАСТОМНЫЙ ФОРМАТТЕР ДЛЯ ЦВЕТНОГО ЛОГИРОВАНИЯ ---
class ColoredFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: f"{Colors.CYAN}%(levelname)s:{Colors.RESET} %(message)s",
        logging.INFO: f"{Colors.GREEN}%(levelname)s:{Colors.RESET} %(message)s",
        logging.WARNING: f"{Colors.YELLOW}%(levelname)s:{Colors.RESET} %(message)s",
        logging.ERROR: f"{Colors.RED}%(levelname)s:{Colors.RESET} %(message)s",
        logging.CRITICAL: f"{Colors.RED}{Colors.BOLD}%(levelname)s:{Colors.RESET} %(message)s"
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# Настройка логирования
log_file_handler = logging.FileHandler('logs/metricscalculator.log')
log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'))

console_handler = logging.StreamHandler()
console_handler.setFormatter(ColoredFormatter())

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        log_file_handler,
        console_handler
    ]
)
logger = logging.getLogger(__name__)

# Импорт конфигурации
try:
    from core_config import (
        CRITICAL_THRESHOLDS, CACHE_DIR, ERROR_CODES
    )
    logger.info(f"{Colors.GREEN}✔ Конфигурация успешно импортирована.{Colors.RESET}")
except ImportError as e:
    logger.critical(f"{Colors.RED}КРИТИЧЕСКАЯ ОШИБКА: Не удалось импортировать конфигурацию: {e}{Colors.RESET}")
    # Значения по умолчанию
    CRITICAL_THRESHOLDS = {"min_authenticity_score": 0.6}
    CACHE_DIR = Path("cache")
    ERROR_CODES = {"E001": "NO_FACE_DETECTED"}

# ==================== КОНСТАНТЫ МЕТРИК ====================

# ИСПРАВЛЕНО: 15 метрик идентичности в 3 группах согласно правкам
IDENTITY_METRICS_GROUPS = {
    "skull_geometry": [
        "skull_width_ratio",
        "temporal_bone_angle", 
        "zygomatic_arch_width",
        "orbital_depth",
        "occipital_curve"
    ],
    "facial_proportions": [
        "cephalic_index",
        "nasolabial_angle",
        "orbital_index",
        "forehead_height_ratio",
        "chin_projection_ratio"
    ],
    "bone_structure": [
        "interpupillary_distance_ratio",
        "gonial_angle_asymmetry",
        "zygomatic_angle",
        "jaw_angle_ratio",
        "mandibular_symphysis_angle"
    ]
}

# Все 15 метрик в одном списке
ALL_IDENTITY_METRICS = []
for group_metrics in IDENTITY_METRICS_GROUPS.values():
    ALL_IDENTITY_METRICS.extend(group_metrics)

# Индексы ландмарок для расчетов
LANDMARK_INDICES = {
    "jaw_line": list(range(0, 17)),
    "right_eyebrow": list(range(17, 22)),
    "left_eyebrow": list(range(22, 27)),
    "nose_bridge": list(range(27, 31)),
    "lower_nose": list(range(31, 36)),
    "right_eye": list(range(36, 42)),
    "left_eye": list(range(42, 48)),
    "outer_lip": list(range(48, 60)),
    "inner_lip": list(range(60, 68))
}

# ==================== ОСНОВНОЙ КЛАСС ====================

class MetricsCalculator:
    """
    ИСПРАВЛЕНО: Калькулятор 15 метрик идентичности
    Согласно правкам: 5+5+5 метрик в 3 группах
    """
    
    def __init__(self):
        logger.info(f"{Colors.BOLD}--- Инициализация MetricsCalculator (Расчетчика метрик) ---{Colors.RESET}")
        
        # Кэш вычислений
        self.metrics_cache = {}
        
        # Базовые линии для нормализации
        self.baseline_metrics = self._load_baseline_metrics()
        
        # Статистика вычислений
        self.calculation_stats = {
            "total_calculations": 0,
            "successful_calculations": 0,
            "failed_calculations": 0,
            "cache_hits": 0
        }
        
        logger.info(f"{Colors.BOLD}--- MetricsCalculator успешно инициализирован ---{Colors.RESET}")

    def _load_baseline_metrics(self) -> Dict[str, Dict[str, float]]:
        """Загрузка базовых линий метрик"""
        try:
            baseline_file = CACHE_DIR / "metrics_baselines.json"
            
            if baseline_file.exists():
                with open(baseline_file, 'r', encoding='utf-8') as f:
                    baselines = json.load(f)
                    logger.info(f"{Colors.GREEN}✔ Базовые линии метрик загружены из: {baseline_file}{Colors.RESET}")
                    return baselines
            else:
                logger.warning(f"{Colors.YELLOW}ПРЕДУПРЕЖДЕНИЕ: Файл базовых линий метрик не найден. Используются значения по умолчанию.{Colors.RESET}")
        except Exception as e:
            logger.error(f"{Colors.RED}ОШИБКА при загрузке базовых линий метрик: {e}{Colors.RESET}")
        
        # ИСПРАВЛЕНО: Значения по умолчанию для всех 15 метрик
        logger.info("Использование значений по умолчанию для базовых линий метрик.")
        return {
            # Геометрия черепа (5)
            "skull_width_ratio": {"mean": 0.75, "std": 0.05},
            "temporal_bone_angle": {"mean": 110.0, "std": 8.0},
            "zygomatic_arch_width": {"mean": 0.68, "std": 0.04},
            "orbital_depth": {"mean": 0.25, "std": 0.03},
            "occipital_curve": {"mean": 0.82, "std": 0.06},
            
            # Пропорции лица (5)
            "cephalic_index": {"mean": 78.5, "std": 4.2},
            "nasolabial_angle": {"mean": 102.0, "std": 8.5},
            "orbital_index": {"mean": 85.0, "std": 6.0},
            "forehead_height_ratio": {"mean": 0.35, "std": 0.04},
            "chin_projection_ratio": {"mean": 0.28, "std": 0.03},
            
            # Костная структура (5)
            "interpupillary_distance_ratio": {"mean": 0.32, "std": 0.02},
            "gonial_angle_asymmetry": {"mean": 2.5, "std": 1.8},
            "zygomatic_angle": {"mean": 125.0, "std": 7.0},
            "jaw_angle_ratio": {"mean": 0.85, "std": 0.05},
            "mandibular_symphysis_angle": {"mean": 75.0, "std": 5.5}
        }

    def calculate_identity_signature_metrics(self, landmarks_3d: np.ndarray, 
                                           pose_category: str = "frontal") -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Расчет 15 метрик идентичности
        Согласно правкам: calculateidentitysignaturemetrics с 15 метриками
        """
        if landmarks_3d.size == 0 or landmarks_3d.shape[0] < 68:
            logger.warning(f"{Colors.YELLOW}ПРЕДУПРЕЖДЕНИЕ: Недостаточно ландмарок (точек лица) для точного расчета метрик. Возможно, лицо не обнаружено или качество низкое.{Colors.RESET}")
            return {}
        
        try:
            logger.info(f"{Colors.CYAN}Начинаем расчет 15 метрик идентичности для позы: '{pose_category}'...")
            
            # Проверка кэша
            cache_key = f"{hash(landmarks_3d.tobytes())}_{pose_category}"
            if cache_key in self.metrics_cache:
                self.calculation_stats["cache_hits"] += 1
                logger.debug(f"Метрики найдены в кэше. Пропускаем расчет.")
                return self.metrics_cache[cache_key]
            
            self.calculation_stats["total_calculations"] += 1
            logger.debug(f"Метрики не найдены в кэше. Выполняем полный расчет.")
            
            # Расчет всех 15 метрик
            identity_metrics = {}
            
            logger.debug("Расчет метрик геометрии черепа...")
            # ГРУППА 1: Геометрия черепа (5 метрик)
            skull_metrics = self._calculate_skull_geometry_metrics(landmarks_3d)
            identity_metrics.update(skull_metrics)
            logger.debug(f"Рассчитано {len(skull_metrics)} метрик геометрии черепа.")
            
            logger.debug("Расчет метрик пропорций лица...")
            # ГРУППА 2: Пропорции лица (5 метрик)
            proportion_metrics = self._calculate_facial_proportion_metrics(landmarks_3d)
            identity_metrics.update(proportion_metrics)
            logger.debug(f"Рассчитано {len(proportion_metrics)} метрик пропорций лица.")
            
            logger.debug("Расчет метрик костной структуры...")
            # ГРУППА 3: Костная структура (5 метрик)
            bone_metrics = self._calculate_bone_structure_metrics(landmarks_3d)
            identity_metrics.update(bone_metrics)
            logger.debug(f"Рассчитано {len(bone_metrics)} метрик костной структуры.")
            
            # Нормализация метрик
            normalized_metrics = self._normalize_metrics(identity_metrics)
            
            # Результат
            result = {
                "raw_metrics": identity_metrics,
                "normalized_metrics": normalized_metrics,
                "pose_category": pose_category,
                "metrics_count": len(identity_metrics),
                "groups": {
                    "skull_geometry": {k: normalized_metrics[k] for k in IDENTITY_METRICS_GROUPS["skull_geometry"] if k in normalized_metrics},
                    "facial_proportions": {k: normalized_metrics[k] for k in IDENTITY_METRICS_GROUPS["facial_proportions"] if k in normalized_metrics},
                    "bone_structure": {k: normalized_metrics[k] for k in IDENTITY_METRICS_GROUPS["bone_structure"] if k in normalized_metrics}
                }
            }
            
            # Кэширование результата
            self.metrics_cache[cache_key] = result
            
            self.calculation_stats["successful_calculations"] += 1
            logger.info(f"{Colors.GREEN}✔ Успешно рассчитано {len(identity_metrics)} метрик идентичности.{Colors.RESET}")
            
            return result
            
        except Exception as e:
            logger.error(f"{Colors.RED}ОШИБКА при расчете метрик идентичности: {e}{Colors.RESET}")
            self.calculation_stats["failed_calculations"] += 1
            return {}

    def _calculate_skull_geometry_metrics(self, landmarks_3d: np.ndarray) -> Dict[str, float]:
        """
        ИСПРАВЛЕНО: Расчет 5 метрик геометрии черепа
        Согласно правкам: skull_width_ratio, temporal_bone_angle, zygomatic_arch_width, orbital_depth, occipital_curve
        """
        try:
            # logger.debug("Расчет метрик геометрии черепа") # Отключено, т.к. уже есть в calculate_identity_signature_metrics
            
            skull_metrics = {}
            
            # 1. ИСПРАВЛЕНО: skull_width_ratio - отношение ширины черепа к высоте лица
            # Ширина черепа: расстояние между височными точками (приблизительно точки 0 и 16)
            skull_width = euclidean(landmarks_3d[0], landmarks_3d[16])
            # Высота лица: от подбородка до лба
            face_height = euclidean(landmarks_3d[8], landmarks_3d[27])  # подбородок - переносица
            skull_metrics["skull_width_ratio"] = skull_width / (face_height + 1e-8)
            
            # 2. ИСПРАВЛЕНО: temporal_bone_angle - угол височной кости
            # Используем точки височной области (приблизительно 0, 1, 2 для правой стороны)
            if len(landmarks_3d) >= 17:
                vec1 = landmarks_3d[1] - landmarks_3d[0]
                vec2 = landmarks_3d[2] - landmarks_3d[1]
                temporal_angle = self._calculate_angle_between_vectors(vec1, vec2)
                skull_metrics["temporal_bone_angle"] = temporal_angle
            else:
                skull_metrics["temporal_bone_angle"] = 110.0  # Значение по умолчанию
            
            # 3. ИСПРАВЛЕНО: zygomatic_arch_width - ширина скуловых дуг
            # Используем точки скуловой области
            left_zygomatic = landmarks_3d[3]   # Левая скуловая точка
            right_zygomatic = landmarks_3d[13] # Правая скуловая точка
            zygomatic_width = euclidean(left_zygomatic, right_zygomatic)
            skull_metrics["zygomatic_arch_width"] = zygomatic_width / (face_height + 1e-8)
            
            # 4. ИСПРАВЛЕНО: orbital_depth - глубина орбит
            # Используем точки глаз для приблизительной оценки глубины
            left_eye_center = np.mean(landmarks_3d[42:48], axis=0)
            right_eye_center = np.mean(landmarks_3d[36:42], axis=0)
            nose_bridge = landmarks_3d[27]
            
            # Глубина как расстояние от центра глаз до переносицы
            left_orbital_depth = euclidean(left_eye_center, nose_bridge)
            right_orbital_depth = euclidean(right_eye_center, nose_bridge)
            avg_orbital_depth = (left_orbital_depth + right_orbital_depth) / 2
            skull_metrics["orbital_depth"] = avg_orbital_depth / (face_height + 1e-8)
            
            # 5. ИСПРАВЛЕНО: occipital_curve - кривизна затылочной области
            # Приблизительная оценка через кривизну контура лица
            jaw_points = landmarks_3d[0:17]  # Контур челюсти
            if len(jaw_points) >= 3:
                # Расчет кривизны через изменение углов
                curvatures = []
                for i in range(1, len(jaw_points) - 1):
                    vec1 = jaw_points[i] - jaw_points[i-1]
                    vec2 = jaw_points[i+1] - jaw_points[i]
                    angle = self._calculate_angle_between_vectors(vec1, vec2)
                    curvatures.append(abs(180 - angle))  # Отклонение от прямой линии
                
                avg_curvature = np.mean(curvatures) if curvatures else 0.0
                skull_metrics["occipital_curve"] = avg_curvature / 180.0  # Нормализация
            else:
                skull_metrics["occipital_curve"] = 0.5
            
            # logger.debug(f"Рассчитано {len(skull_metrics)} метрик геометрии черепа") # Отключено
            return skull_metrics
            
        except Exception as e:
            logger.error(f"{Colors.RED}ОШИБКА при расчете метрик геометрии черепа: {e}{Colors.RESET}")
            return {}

    def _calculate_facial_proportion_metrics(self, landmarks_3d: np.ndarray) -> Dict[str, float]:
        """
        ИСПРАВЛЕНО: Расчет 5 метрик пропорций лица
        Согласно правкам: cephalic_index, nasolabial_angle, orbital_index, forehead_height_ratio, chin_projection_ratio
        """
        try:
            # logger.debug("Расчет метрик пропорций лица") # Отключено
            
            proportion_metrics = {}
            
            # 1. ИСПРАВЛЕНО: cephalic_index - черепной индекс (ширина/длина * 100)
            skull_width = euclidean(landmarks_3d[0], landmarks_3d[16])
            # Длина черепа приблизительно через расстояние от лба до подбородка
            skull_length = euclidean(landmarks_3d[8], landmarks_3d[27])
            cephalic_index = (skull_width / (skull_length + 1e-8)) * 100
            proportion_metrics["cephalic_index"] = cephalic_index
            
            # 2. ИСПРАВЛЕНО: nasolabial_angle - носогубный угол
            # Используем точки носа и рта
            nose_tip = landmarks_3d[33]      # Кончик носа
            nose_base = landmarks_3d[51]     # Основание носа (центр верхней губы)
            mouth_corner = landmarks_3d[48]  # Уголок рта
            
            vec1 = nose_base - nose_tip
            vec2 = mouth_corner - nose_base
            nasolabial_angle = self._calculate_angle_between_vectors(vec1, vec2)
            proportion_metrics["nasolabial_angle"] = nasolabial_angle
            
            # 3. ИСПРАВЛЕНО: orbital_index - орбитальный индекс
            # Отношение высоты орбиты к ширине
            left_eye_width = euclidean(landmarks_3d[36], landmarks_3d[39])   # Ширина левого глаза
            left_eye_height = euclidean(landmarks_3d[37], landmarks_3d[41])  # Высота левого глаза
            right_eye_width = euclidean(landmarks_3d[42], landmarks_3d[45])  # Ширина правого глаза
            right_eye_height = euclidean(landmarks_3d[43], landmarks_3d[47]) # Высота правого глаза
            
            avg_eye_width = (left_eye_width + right_eye_width) / 2
            avg_eye_height = (left_eye_height + right_eye_height) / 2
            orbital_index = (avg_eye_height / (avg_eye_width + 1e-8)) * 100
            proportion_metrics["orbital_index"] = orbital_index
            
            # 4. ИСПРАВЛЕНО: forehead_height_ratio - отношение высоты лба к высоте лица
            # Высота лба: от бровей до линии волос (приблизительно)
            brow_center = np.mean(landmarks_3d[19:25], axis=0)  # Центр бровей
            # Приблизительная линия волос (выше бровей)
            forehead_top = brow_center.copy()
            forehead_top[1] -= 40  # Смещение вверх (приблизительно)
            
            forehead_height = euclidean(brow_center, forehead_top)
            face_height = euclidean(landmarks_3d[8], brow_center)  # От подбородка до бровей
            forehead_height_ratio = forehead_height / (face_height + 1e-8)
            proportion_metrics["forehead_height_ratio"] = forehead_height_ratio
            
            # 5. ИСПРАВЛЕНО: chin_projection_ratio - выступание подбородка
            chin_point = landmarks_3d[8]     # Точка подбородка
            mouth_center = landmarks_3d[51]  # Центр рта
            nose_base = landmarks_3d[33]     # Основание носа
            
            # Проекция подбородка относительно линии нос-рот
            chin_projection = euclidean(chin_point, mouth_center)
            face_depth = euclidean(nose_base, mouth_center)
            chin_projection_ratio = chin_projection / (face_depth + 1e-8)
            proportion_metrics["chin_projection_ratio"] = chin_projection_ratio
            
            # logger.debug(f"Рассчитано {len(proportion_metrics)} метрик пропорций лица") # Отключено
            return proportion_metrics
            
        except Exception as e:
            logger.error(f"{Colors.RED}ОШИБКА при расчете метрик пропорций лица: {e}{Colors.RESET}")
            return {}

    def _calculate_bone_structure_metrics(self, landmarks_3d: np.ndarray) -> Dict[str, float]:
        """
        ИСПРАВЛЕНО: Расчет 5 метрик костной структуры
        Согласно правкам: interpupillary_distance_ratio, gonial_angle_asymmetry, zygomatic_angle, jaw_angle_ratio, mandibular_symphysis_angle
        """
        try:
            # logger.debug("Расчет метрик костной структуры") # Отключено
            
            bone_metrics = {}
            
            # 1. ИСПРАВЛЕНО: interpupillary_distance_ratio - межзрачковое расстояние
            left_eye_center = np.mean(landmarks_3d[42:48], axis=0)
            right_eye_center = np.mean(landmarks_3d[36:42], axis=0)
            interpupillary_distance = euclidean(left_eye_center, right_eye_center)
            
            # Нормализация по ширине лица
            face_width = euclidean(landmarks_3d[0], landmarks_3d[16])
            ipd_ratio = interpupillary_distance / (face_width + 1e-8)
            bone_metrics["interpupillary_distance_ratio"] = ipd_ratio
            
            # 2. ИСПРАВЛЕНО: gonial_angle_asymmetry - асимметрия углов нижней челюсти
            # Углы нижней челюсти (gonial angles)
            left_gonial_points = [landmarks_3d[2], landmarks_3d[3], landmarks_3d[4]]
            right_gonial_points = [landmarks_3d[12], landmarks_3d[13], landmarks_3d[14]]
            
            # Расчет углов
            left_gonial_angle = self._calculate_angle_from_three_points(
                left_gonial_points[0], left_gonial_points[1], left_gonial_points[2]
            )
            right_gonial_angle = self._calculate_angle_from_three_points(
                right_gonial_points[0], right_gonial_points[1], right_gonial_points[2]
            )
            
            gonial_asymmetry = abs(left_gonial_angle - right_gonial_angle)
            bone_metrics["gonial_angle_asymmetry"] = gonial_asymmetry
            
            # 3. ИСПРАВЛЕНО: zygomatic_angle - угол скуловой кости
            # Используем точки скуловой области
            left_zygomatic = landmarks_3d[3]
            right_zygomatic = landmarks_3d[13]
            nose_bridge = landmarks_3d[27]
            
            # Угол между скуловыми точками и переносицей
            vec1 = left_zygomatic - nose_bridge
            vec2 = right_zygomatic - nose_bridge
            zygomatic_angle = self._calculate_angle_between_vectors(vec1, vec2)
            bone_metrics["zygomatic_angle"] = zygomatic_angle
            
            # 4. ИСПРАВЛЕНО: jaw_angle_ratio - отношение углов челюсти
            # Отношение ширины челюсти в разных точках
            upper_jaw_width = euclidean(landmarks_3d[3], landmarks_3d[13])   # Верхняя часть
            lower_jaw_width = euclidean(landmarks_3d[5], landmarks_3d[11])   # Нижняя часть
            jaw_angle_ratio = upper_jaw_width / (lower_jaw_width + 1e-8)
            bone_metrics["jaw_angle_ratio"] = jaw_angle_ratio
            
            # 5. ИСПРАВЛЕНО: mandibular_symphysis_angle - угол симфиза нижней челюсти
            # Угол в области подбородка
            chin_left = landmarks_3d[6]
            chin_center = landmarks_3d[8]
            chin_right = landmarks_3d[10]
            
            mandibular_angle = self._calculate_angle_from_three_points(
                chin_left, chin_center, chin_right
            )
            bone_metrics["mandibular_symphysis_angle"] = mandibular_angle
            
            # logger.debug(f"Рассчитано {len(bone_metrics)} метрик костной структуры") # Отключено
            return bone_metrics
            
        except Exception as e:
            logger.error(f"{Colors.RED}ОШИБКА при расчете метрик костной структуры: {e}{Colors.RESET}")
            return {}

    def _calculate_angle_between_vectors(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Расчет угла между двумя векторами в градусах"""
        try:
            # Нормализация векторов
            vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
            vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
            
            # Скалярное произведение
            dot_product = np.clip(np.dot(vec1_norm, vec2_norm), -1.0, 1.0)
            
            # Угол в радианах, затем в градусах
            angle_rad = np.arccos(dot_product)
            angle_deg = np.degrees(angle_rad)
            
            return float(angle_deg)
            
        except Exception as e:
            logger.error(f"{Colors.RED}ОШИБКА при расчете угла между векторами: {e}{Colors.RESET}")
            return 0.0

    def _calculate_angle_from_three_points(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Расчет угла в точке p2, образованного точками p1-p2-p3"""
        try:
            vec1 = p1 - p2
            vec2 = p3 - p2
            return self._calculate_angle_between_vectors(vec1, vec2)
            
        except Exception as e:
            logger.error(f"{Colors.RED}ОШИБКА при расчете угла из трех точек: {e}{Colors.RESET}")
            return 0.0

    def _normalize_metrics(self, raw_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        ИСПРАВЛЕНО: Нормализация метрик по базовым линиям
        Согласно правкам: z-score нормализация
        """
        try:
            logger.debug("Нормализация метрик...")
            normalized = {}
            
            for metric_name, value in raw_metrics.items():
                if metric_name in self.baseline_metrics:
                    baseline = self.baseline_metrics[metric_name]
                    mean = baseline["mean"]
                    std = baseline["std"]
                    
                    # Z-score нормализация
                    z_score = (value - mean) / (std + 1e-8)
                    
                    # Преобразование в диапазон [0, 1] через sigmoid
                    normalized_value = 1 / (1 + np.exp(-z_score))
                    normalized[metric_name] = float(normalized_value)
                else:
                    # Если нет базовой линии, оставляем как есть
                    normalized[metric_name] = float(value)
                    logger.warning(f"{Colors.YELLOW}ПРЕДУПРЕЖДЕНИЕ: Для метрики '{metric_name}' не найдена базовая линия. Нормализация не выполнена.{Colors.RESET}")
            
            logger.debug("Нормализация метрик завершена.")
            return normalized
            
        except Exception as e:
            logger.error(f"{Colors.RED}ОШИБКА при нормализации метрик: {e}{Colors.RESET}")
            return raw_metrics

    def calculate_metrics_similarity(self, metrics1: Dict[str, float], 
                                   metrics2: Dict[str, float]) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Расчет схожести между наборами метрик
        Согласно правкам: cosine similarity и euclidean distance
        """
        try:
            logger.info(f"{Colors.CYAN}Расчет схожести двух наборов метрик...{Colors.RESET}")
            
            # Общие метрики
            common_metrics = set(metrics1.keys()) & set(metrics2.keys())
            
            if not common_metrics:
                logger.warning(f"{Colors.YELLOW}ПРЕДУПРЕЖДЕНИЕ: Нет общих метрик для сравнения схожести. Возвращаем нулевую схожесть.{Colors.RESET}")
                return {"similarity": 0.0, "distance": float('inf'), "common_metrics": 0}
            
            # Векторы значений
            vec1 = np.array([metrics1[metric] for metric in common_metrics])
            vec2 = np.array([metrics2[metric] for metric in common_metrics])
            
            # Cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                cosine_similarity = 0.0
                logger.warning(f"{Colors.YELLOW}ПРЕДУПРЕЖДЕНИЕ: Один из векторов метрик равен нулю, косинусная схожесть будет 0.{Colors.RESET}")
            else:
                cosine_similarity = dot_product / (norm1 * norm2)
            
            # Euclidean distance
            euclidean_distance = np.linalg.norm(vec1 - vec2)
            
            # Нормализованное расстояние
            max_possible_distance = np.sqrt(len(common_metrics) * 2)  # Максимальное расстояние для нормализованных метрик
            normalized_distance = euclidean_distance / max_possible_distance
            
            # Общий балл схожести
            similarity_score = (cosine_similarity + (1 - normalized_distance)) / 2
            
            result = {
                "similarity": float(np.clip(similarity_score, 0.0, 1.0)),
                "cosine_similarity": float(cosine_similarity),
                "euclidean_distance": float(euclidean_distance),
                "normalized_distance": float(normalized_distance),
                "common_metrics": len(common_metrics),
                "total_metrics": len(set(metrics1.keys()) | set(metrics2.keys()))
            }
            
            logger.info(f"{Colors.GREEN}✔ Расчет схожести метрик завершен. Общая схожесть: {result['similarity']:.3f}{Colors.RESET}")
            return result
            
        except Exception as e:
            logger.error(f"{Colors.RED}ОШИБКА при расчете схожести метрик: {e}{Colors.RESET}")
            return {"similarity": 0.0, "distance": float('inf'), "common_metrics": 0}

    def validate_metrics_consistency(self, metrics_timeline: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Валидация согласованности метрик во времени
        Согласно правкам: coefficient of variation и trend analysis
        """
        try:
            logger.info(f"{Colors.CYAN}Проверка согласованности {len(metrics_timeline)} наборов метрик во времени...{Colors.RESET}")
            
            if len(metrics_timeline) < 2:
                logger.warning(f"{Colors.YELLOW}ПРЕДУПРЕЖДЕНИЕ: Недостаточно данных ({len(metrics_timeline)} точек) для анализа согласованности во времени. Требуется минимум 2 точки.{Colors.RESET}")
                return {"consistent": True, "reason": "Недостаточно данных"}
            
            consistency_results = {}
            
            # Анализ каждой метрики
            all_metrics = set()
            for metrics in metrics_timeline:
                all_metrics.update(metrics.keys())
            
            for metric_name in all_metrics:
                values = []
                for metrics in metrics_timeline:
                    if metric_name in metrics:
                        values.append(metrics[metric_name])
                
                if len(values) < 2:
                    logger.debug(f"Для метрики '{metric_name}' недостаточно точек для анализа тренда. Пропускаем.")
                    continue
                
                values_array = np.array(values)
                
                # Coefficient of variation
                mean_val = np.mean(values_array)
                std_val = np.std(values_array)
                cv = std_val / (mean_val + 1e-8)
                
                # Trend analysis
                x = np.arange(len(values))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, values_array)
                
                # Outlier detection (Z-score > 2.5)
                z_scores = np.abs(stats.zscore(values_array))
                outliers = np.sum(z_scores > 2.5)
                
                consistency_results[metric_name] = {
                    "coefficient_of_variation": float(cv),
                    "trend_slope": float(slope),
                    "trend_r_squared": float(r_value ** 2),
                    "trend_p_value": float(p_value),
                    "outliers_count": int(outliers),
                    "consistent": cv < 0.3 and outliers <= 1,  # Пороги согласованности
                    "values_count": len(values)
                }
            
            # Общая согласованность
            consistent_metrics = [result["consistent"] for result in consistency_results.values()]
            overall_consistency = np.mean(consistent_metrics) if consistent_metrics else 1.0
            
            result = {
                "overall_consistency": float(overall_consistency),
                "consistent": overall_consistency >= 0.7,
                "metrics_analysis": consistency_results,
                "total_metrics": len(consistency_results),
                "consistent_metrics": sum(consistent_metrics)
            }
            
            logger.info(f"{Colors.GREEN}✔ Проверка согласованности завершена. Общая согласованность: {overall_consistency:.3f}{Colors.RESET}")
            return result
            
        except Exception as e:
            logger.error(f"{Colors.RED}ОШИБКА при валидации согласованности метрик во времени: {e}{Colors.RESET}")
            return {"consistent": False, "reason": f"Ошибка: {str(e)}"}

    def get_metrics_statistics(self) -> Dict[str, Any]:
        """Получение статистики вычислений"""
        logger.info(f"{Colors.CYAN}Получение статистики MetricsCalculator...{Colors.RESET}")
        stats = {
            "calculation_stats": self.calculation_stats.copy(),
            "cache_size": len(self.metrics_cache),
            "supported_metrics": ALL_IDENTITY_METRICS.copy(),
            "metrics_groups": IDENTITY_METRICS_GROUPS.copy()
        }
        logger.info(f"{Colors.GREEN}✔ Статистика MetricsCalculator получена.{Colors.RESET}")
        return stats

    def clear_cache(self) -> None:
        """Очистка кэша вычислений"""
        self.metrics_cache.clear()
        logger.info(f"{Colors.YELLOW}Кэш вычислений метрик очищен.{Colors.RESET}")

    def save_baseline_metrics(self, baseline_file: str = "metrics_baselines.json") -> None:
        """Сохранение базовых линий метрик"""
        try:
            baseline_path = CACHE_DIR / baseline_file
            CACHE_DIR.mkdir(exist_ok=True)
            
            with open(baseline_path, 'w', encoding='utf-8') as f:
                json.dump(self.baseline_metrics, f, indent=2, ensure_ascii=False)
            
            logger.info(f"{Colors.GREEN}✔ Базовые линии метрик сохранены в: {baseline_path}{Colors.RESET}")
            
        except Exception as e:
            logger.error(f"{Colors.RED}ОШИБКА при сохранении базовых линий метрик: {e}{Colors.RESET}")

    def self_test(self) -> None:
        """Самотестирование модуля"""
        logger.info(f"{Colors.BOLD}\n=== Запуск самотестирования MetricsCalculator ==={Colors.RESET}")
        
        try:
            # Генерация тестовых ландмарок
            test_landmarks = np.random.rand(68, 3) * 100
            logger.info("Генерация тестовых 3D ландмарок для проверки...")
            
            # Тест расчета метрик
            metrics_result = self.calculate_identity_signature_metrics(test_landmarks, pose_category="test_pose")
            logger.info(f"Тест расчета метрик: успешно извлечено {len(metrics_result.get('raw_metrics', {}))} метрик.")
            if len(metrics_result.get('raw_metrics', {})) == 15: # Ожидаем 15 метрик
                logger.info(f"{Colors.GREEN}✔ Расчет всех 15 метрик: ПРОЙДЕН.{Colors.RESET}")
            else:
                logger.error(f"{Colors.RED}✖ Расчет метрик: ПРОВАЛЕН. Ожидалось 15, получено {len(metrics_result.get('raw_metrics', {}))}.{Colors.RESET}")
            
            # Тест схожести метрик
            metrics1 = metrics_result.get('normalized_metrics', {})
            metrics2 = {k: v + np.random.normal(0, 0.02) for k, v in metrics1.items()} # Небольшие изменения для схожести
            
            similarity = self.calculate_metrics_similarity(metrics1, metrics2)
            logger.info(f"Тест схожести метрик: Общая схожесть: {similarity['similarity']:.3f}")
            if similarity['similarity'] > 0.8: # Ожидаем высокую схожесть
                logger.info(f"{Colors.GREEN}✔ Тест схожести метрик: ПРОЙДЕН (высокая схожесть).{Colors.RESET}")
            else:
                logger.error(f"{Colors.RED}✖ Тест схожести метрик: ПРОВАЛЕН (низкая схожесть).{Colors.RESET}")
            
            # Тест согласованности
            timeline = [
                metrics1,
                {k: v + np.random.normal(0, 0.01) for k, v in metrics1.items()},
                {k: v + np.random.normal(0, 0.02) for k, v in metrics1.items()}
            ]
            consistency = self.validate_metrics_consistency(timeline)
            logger.info(f"Тест согласованности метрик во времени: Общая согласованность: {consistency['overall_consistency']:.3f}")
            if consistency['overall_consistency'] >= 0.7: # Ожидаем хорошую согласованность
                logger.info(f"{Colors.GREEN}✔ Тест согласованности метрик: ПРОЙДЕН (высокая согласованность).{Colors.RESET}")
            else:
                logger.error(f"{Colors.RED}✖ Тест согласованности метрик: ПРОВАЛЕН (низкая согласованность).{Colors.RESET}")
            
            # Статистика
            stats = self.get_metrics_statistics()
            logger.info(f"{Colors.CYAN}Статистика MetricsCalculator после теста: {Colors.RESET}")
            logger.info(f"  Всего вычислений: {stats['calculation_stats']['total_calculations']}")
            logger.info(f"  Успешно: {stats['calculation_stats']['successful_calculations']}")
            logger.info(f"  Ошибок: {stats['calculation_stats']['failed_calculations']}")
            logger.info(f"  Попаданий в кэш: {stats['calculation_stats']['cache_hits']}")
            logger.info(f"  Размер кэша: {stats['cache_size']}")
            
        except Exception as e:
            logger.critical(f"{Colors.RED}КРИТИЧЕСКАЯ ОШИБКА при самотестировании MetricsCalculator: {e}{Colors.RESET}")
        
        logger.info(f"{Colors.BOLD}=== Самотестирование MetricsCalculator завершено ==={Colors.RESET}\n")

# ==================== ТОЧКА ВХОДА ====================

if __name__ == "__main__":
    calculator = MetricsCalculator()
    calculator.self_test()