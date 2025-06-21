"""
MedicalValidator - Валидатор медицинской достоверности изменений лица
Версия: 2.0
Дата: 2025-06-21
ИСПРАВЛЕНО: Все критические ошибки согласно новому ТЗ
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta, date
from pathlib import Path
import json
import pickle
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import psutil
import time
from functools import lru_cache
import threading
from collections import OrderedDict, defaultdict

from core_config import get_config

# Настройка логирования
logger = logging.getLogger(__name__)

# === КОНСТАНТЫ МЕДИЦИНСКОЙ ВАЛИДАЦИИ ===

# Дата рождения Владимира Путина
PUTIN_BIRTH_DATE = date(1952, 10, 7)

# Модель старения
AGING_MODEL = {
    "elasticity_loss_per_year": 0.015,  # 1.5% потеря эластичности в год
    "tissue_sagging_per_year": 1.51,    # 1.51 мм опущение тканей в год
    "bone_stability_threshold": 25       # Возраст стабильности костей
}

# Пороги медицинской валидации
MEDICAL_VALIDATOR_THRESHOLDS = {
    "bone_metric_stability_threshold": 0.05,      # CV < 5% для костных метрик после 25 лет
    "soft_tissue_aging_slope_threshold": -0.02,   # Максимальный наклон старения мягких тканей
    "surgery_min_interval_days": 60,               # Минимальный интервал для хирургии
    "surgery_max_interval_days": 365,              # Максимальный интервал для хирургии
    "medical_event_correlation_days": 30,          # Окно корреляции с медицинскими событиями
    "physiological_change_limit": 0.02,            # Максимальное физиологическое изменение в год
    "aging_consistency_scale_factor": 2.0,         # Масштабирующий фактор для consistency
    "reverse_aging_threshold": -0.05,              # Порог обратного старения
    "rapid_change_threshold": 0.15,                # Порог быстрых изменений
    "bone_growth_threshold_age": 25,               # Возраст стабильности костей
    "ipd_variation_max": 0.02,                     # Максимальная вариация IPD после 25 лет
    "skull_stability_cv_threshold": 0.03,          # CV для стабильности черепа
    "temporal_bone_stability_threshold": 0.04,     # Стабильность височных костей
    "zygomatic_stability_threshold": 0.04,         # Стабильность скуловых костей
    "orbital_stability_threshold": 0.03            # Стабильность орбитальных метрик
}

# Критические пороги
CRITICAL_THRESHOLDS = {
    "aging_consistency_threshold": 0.6,
    "temporal_stability_threshold": 0.8
}

# Медицинские события для корреляции
DOCUMENTED_HEALTH_EVENTS = {
    "2000-03-26": {"type": "political_stress", "description": "Президентские выборы", "stress_level": "high"},
    "2004-03-14": {"type": "political_stress", "description": "Президентские выборы", "stress_level": "high"},
    "2008-05-07": {"type": "transition", "description": "Передача полномочий", "stress_level": "medium"},
    "2012-05-07": {"type": "political_stress", "description": "Президентские выборы", "stress_level": "high"},
    "2018-05-07": {"type": "political_stress", "description": "Президентские выборы", "stress_level": "high"},
    "2020-03-01": {"type": "health_crisis", "description": "Пандемия COVID-19", "stress_level": "very_high"},
    "2024-03-17": {"type": "political_stress", "description": "Президентские выборы", "stress_level": "high"}
}

# Медицинские константы старения
MEDICAL_AGING_CONSTANTS = {
    'collagen_degradation_rate': 0.01,     # 1% в год после 25 лет
    'elastin_loss_rate': 0.015,            # 1.5% в год после 30 лет
    'fat_redistribution_rate': 0.02,       # 2% в год после 35 лет
    'bone_density_loss': 0.005,            # 0.5% в год после 40 лет
    'muscle_tone_loss': 0.01,              # 1% в год после 30 лет
    'skin_thickness_reduction': 0.007      # 0.7% в год после 40 лет
}

# === ОСНОВНОЙ КЛАСС МЕДИЦИНСКОГО ВАЛИДАТОРА ===

class MedicalValidator:
    """
    Валидатор медицинской достоверности с полной функциональностью
    ИСПРАВЛЕНО: Все критические ошибки согласно новому ТЗ
    """

    def __init__(self):
        """Инициализация медицинского валидатора"""
        logger.info("Инициализация MedicalValidator")
        
        self.config = get_config()
        self.cache_dir = Path("./cache/medical_validator")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Медицинские события
        self.health_events = self._load_health_events()
        
        # Кэш результатов валидации
        self.validation_cache = {}
        
        # Модель старения
        self.aging_model = AGING_MODEL.copy()
        
        # Статистика валидации
        self.validation_stats = {
            'total_validations': 0,
            'consistent_identities': 0,
            'inconsistent_identities': 0,
            'bone_anomalies_detected': 0,
            'soft_tissue_anomalies': 0,
            'surgical_interventions_excluded': 0
        }
        
        # Блокировка для потокобезопасности
        self.validation_lock = threading.Lock()
        
        # Флаг калибровки
        self.calibrated = False
        
        logger.info("MedicalValidator инициализирован")

    def _load_health_events(self) -> Dict[datetime, Dict[str, str]]:
        """Загрузка документированных медицинских событий"""
        try:
            events = {}
            for date_str, event_data in DOCUMENTED_HEALTH_EVENTS.items():
                try:
                    event_date = datetime.strptime(date_str, "%Y-%m-%d")
                    events[event_date] = event_data
                except ValueError as e:
                    logger.warning(f"Неверный формат даты события: {date_str}, ошибка: {e}")
            
            logger.info(f"Загружено {len(events)} документированных медицинских событий")
            return events
        except Exception as e:
            logger.error(f"Ошибка загрузки медицинских событий: {e}")
            return {}

    def validate_aging_consistency_for_identity(self, identity_metrics_timeline: Dict[str, Dict[str, Any]],
                                              age_progression: List[float]) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Валидация согласованности старения для идентичности
        Согласно правкам: правильная aging model и валидация
        """
        if not identity_metrics_timeline or len(identity_metrics_timeline) < 2:
            logger.warning("Недостаточно данных для валидации согласованности старения")
            return {
                "overall_aging_consistency_score": 0.0,
                "anomalies_detected": False,
                "anomalies_details": ["Недостаточно данных для анализа"],
                "metric_deviations": {},
                "bone_stability_violations": [],
                "soft_tissue_anomalies": [],
                "medical_plausibility": 0.0
            }

        try:
            start_time = time.time()
            logger.info(f"Валидация согласованности старения для {len(identity_metrics_timeline)} временных точек")
            
            validation_results = {
                "overall_aging_consistency_score": 1.0,
                "anomalies_detected": False,
                "anomalies_details": [],
                "metric_deviations": {},
                "bone_stability_violations": [],
                "soft_tissue_anomalies": [],
                "medical_plausibility": 1.0,
                "processing_time_ms": 0.0
            }

            # Сортировка дат
            sorted_dates = sorted(identity_metrics_timeline.keys())
            
            # Получение базовых метрик (первая дата)
            first_date = sorted_dates[0]
            baseline_metrics = identity_metrics_timeline[first_date]
            
            if not baseline_metrics:
                logger.warning("Отсутствуют базовые метрики")
                validation_results["overall_aging_consistency_score"] = 0.0
                validation_results["anomalies_details"].append("Отсутствуют базовые метрики")
                return validation_results

            all_deviations = []
            bone_metrics = ["skull_width_ratio", "temporal_bone_angle", "zygomatic_arch_width", 
                          "interpupillary_distance_ratio", "orbital_depth"]
            soft_tissue_metrics = ["forehead_height_ratio", "nose_width_ratio", "mouth_width_ratio",
                                 "cheek_fullness_ratio", "nasolabial_depth"]

            # Анализ каждой временной точки
            for i in range(1, len(sorted_dates)):
                current_date = sorted_dates[i]
                current_metrics = identity_metrics_timeline[current_date]
                current_age = age_progression[i] if i < len(age_progression) else 50.0
                
                if not current_metrics:
                    logger.warning(f"Отсутствуют метрики для даты: {current_date}")
                    continue

                # Предсказание ожидаемых метрик на основе aging model
                predicted_metrics = self._predict_metrics_for_age(current_age, baseline_metrics)
                
                # Анализ отклонений
                deviations_for_period = {}
                period_deviations_sum = 0.0
                relevant_metrics_count = 0

                for metric_name, actual_value in current_metrics.items():
                    if metric_name in predicted_metrics:
                        predicted_value = predicted_metrics[metric_name]
                        
                        # Расчет отклонения
                        deviation = abs(actual_value - predicted_value)
                        normalized_deviation = deviation / (baseline_metrics.get(metric_name, 1.0) + 1e-6)
                        
                        deviations_for_period[metric_name] = normalized_deviation
                        all_deviations.append(normalized_deviation)
                        period_deviations_sum += normalized_deviation
                        relevant_metrics_count += 1

                        # Проверка на обратное старение для мягких тканей
                        if metric_name in soft_tissue_metrics:
                            if (actual_value - predicted_value) > MEDICAL_VALIDATOR_THRESHOLDS["reverse_aging_threshold"]:
                                validation_results["anomalies_detected"] = True
                                anomaly_detail = f"Обратное старение обнаружено {current_date}: {metric_name} = {actual_value - predicted_value:.4f}"
                                validation_results["anomalies_details"].append(anomaly_detail)
                                validation_results["soft_tissue_anomalies"].append({
                                    "date": current_date,
                                    "metric": metric_name,
                                    "type": "reverse_aging",
                                    "deviation": actual_value - predicted_value
                                })

                        # Проверка стабильности костных метрик после 25 лет
                        if metric_name in bone_metrics and current_age >= 25:
                            if abs(actual_value - predicted_value) > MEDICAL_VALIDATOR_THRESHOLDS["rapid_change_threshold"]:
                                validation_results["anomalies_detected"] = True
                                anomaly_detail = f"Нестабильность костной структуры {current_date}: {metric_name} = {actual_value - predicted_value:.4f}"
                                validation_results["anomalies_details"].append(anomaly_detail)
                                validation_results["bone_stability_violations"].append({
                                    "date": current_date,
                                    "metric": metric_name,
                                    "type": "bone_instability",
                                    "deviation": actual_value - predicted_value
                                })

                # Средняя девиация за период
                if relevant_metrics_count > 0:
                    avg_deviation_for_period = period_deviations_sum / relevant_metrics_count
                    validation_results["metric_deviations"][current_date] = deviations_for_period

            # Общая оценка согласованности
            if all_deviations:
                mean_normalized_deviation = np.mean(all_deviations)
                validation_results["overall_aging_consistency_score"] = max(
                    0.0, 1.0 - mean_normalized_deviation * MEDICAL_VALIDATOR_THRESHOLDS["aging_consistency_scale_factor"]
                )

            # Проверка общего порога
            if validation_results["overall_aging_consistency_score"] < CRITICAL_THRESHOLDS.get("aging_consistency_threshold", 0.6):
                validation_results["anomalies_detected"] = True
                validation_results["anomalies_details"].append(
                    f"Низкая общая согласованность старения: {validation_results['overall_aging_consistency_score']:.2f}"
                )

            # Расчет медицинской правдоподобности
            validation_results["medical_plausibility"] = self._calculate_medical_plausibility(validation_results)
            
            # Время обработки
            validation_results["processing_time_ms"] = (time.time() - start_time) * 1000
            
            # Обновление статистики
            self.validation_stats['total_validations'] += 1
            if validation_results["overall_aging_consistency_score"] >= 0.6:
                self.validation_stats['consistent_identities'] += 1
            else:
                self.validation_stats['inconsistent_identities'] += 1
            
            self.validation_stats['bone_anomalies_detected'] += len(validation_results["bone_stability_violations"])
            self.validation_stats['soft_tissue_anomalies'] += len(validation_results["soft_tissue_anomalies"])

            logger.info(f"Валидация согласованности завершена: score={validation_results['overall_aging_consistency_score']:.3f}")
            return validation_results

        except Exception as e:
            logger.error(f"Ошибка валидации согласованности старения: {e}")
            return {
                "overall_aging_consistency_score": 0.0,
                "anomalies_detected": True,
                "anomalies_details": [f"Ошибка валидации: {str(e)}"],
                "metric_deviations": {},
                "bone_stability_violations": [],
                "soft_tissue_anomalies": [],
                "medical_plausibility": 0.0,
                "processing_time_ms": 0.0
            }

    def _predict_metrics_for_age(self, age: float, baseline_metrics: Dict[str, float]) -> Dict[str, float]:
        """Предсказание метрик для заданного возраста на основе aging model"""
        try:
            predicted_metrics = {}
            
            for metric_name, baseline_value in baseline_metrics.items():
                predicted_value = baseline_value

                # Применение aging model согласно правкам
                if age >= AGING_MODEL["bone_stability_threshold"]:
                    # После 25 лет кости стабильны
                    bone_metrics = [
                        "skull_width_ratio", "temporal_bone_angle", "zygomatic_arch_width",
                        "orbital_depth", "interpupillary_distance_ratio"
                    ]

                    if metric_name in bone_metrics:
                        # Костные метрики остаются стабильными
                        predicted_value = baseline_value
                    else:
                        # Мягкие ткани изменяются согласно aging model
                        age_factor = max(0, age - 40)  # Изменения начинаются после 40 лет
                        elasticity_loss = age_factor * AGING_MODEL["elasticity_loss_per_year"]
                        tissue_sagging = age_factor * AGING_MODEL["tissue_sagging_per_year"] * 0.001  # Масштабирование
                        predicted_value = baseline_value * (1.0 - elasticity_loss) - tissue_sagging
                else:
                    # До 25 лет возможны изменения роста
                    growth_factor = 1.0 + (AGING_MODEL["bone_stability_threshold"] - age) * 0.001
                    predicted_value = baseline_value * growth_factor

                predicted_metrics[metric_name] = max(0.01, predicted_value)  # Минимальное значение

            return predicted_metrics

        except Exception as e:
            logger.error(f"Ошибка предсказания метрик: {e}")
            return baseline_metrics.copy()

    def _calculate_medical_plausibility(self, validation_results: Dict[str, Any]) -> float:
        """Расчет медицинской правдоподобности"""
        try:
            plausibility_factors = []
            
            # Фактор согласованности старения
            consistency_score = validation_results.get("overall_aging_consistency_score", 0.0)
            plausibility_factors.append(consistency_score)
            
            # Фактор костных аномалий
            bone_violations = len(validation_results.get("bone_stability_violations", []))
            bone_factor = max(0.0, 1.0 - bone_violations * 0.2)
            plausibility_factors.append(bone_factor)
            
            # Фактор мягких тканей
            soft_tissue_anomalies = len(validation_results.get("soft_tissue_anomalies", []))
            soft_tissue_factor = max(0.0, 1.0 - soft_tissue_anomalies * 0.15)
            plausibility_factors.append(soft_tissue_factor)
            
            # Общая медицинская правдоподобность
            medical_plausibility = np.mean(plausibility_factors) if plausibility_factors else 0.0
            
            return float(max(0.0, min(1.0, medical_plausibility)))
            
        except Exception as e:
            logger.error(f"Ошибка расчета медицинской правдоподобности: {e}")
            return 0.0

    def check_bone_structure_immutability(self, cranial_metrics_timeline: Dict[str, List[float]],
                                        ages: List[float]) -> Dict[str, Dict[str, Any]]:
        """
        ИСПРАВЛЕНО: Проверка неизменности костной структуры после 25 лет
        Согласно правкам: правильная валидация стабильности костей
        """
        if not cranial_metrics_timeline or not ages:
            logger.warning("Нет данных для проверки неизменности костной структуры")
            return {}

        try:
            logger.info(f"Проверка неизменности костной структуры для {len(cranial_metrics_timeline)} метрик")
            
            results = {}
            
            for metric, values in cranial_metrics_timeline.items():
                values = np.array(values)
                ages_array = np.array(ages)

                # Фильтрация данных после 25 лет
                adult_mask = ages_array >= MEDICAL_VALIDATOR_THRESHOLDS["bone_growth_threshold_age"]
                
                if np.sum(adult_mask) < 2:
                    results[metric] = {
                        "stable": True,
                        "reason": "Недостаточно данных для взрослого возраста",
                        "cv": 0.0,
                        "adult_data_points": int(np.sum(adult_mask)),
                        "mean": 0.0,
                        "std": 0.0,
                        "threshold": 0.0
                    }
                    continue

                adult_values = values[adult_mask]

                # Расчет coefficient of variation
                mean_val = np.mean(adult_values)
                std_val = np.std(adult_values)
                
                if mean_val != 0:
                    cv = std_val / mean_val
                else:
                    cv = 0.0 if std_val == 0 else float('inf')

                # Определение стабильности на основе CV
                stability_threshold = MEDICAL_VALIDATOR_THRESHOLDS["bone_metric_stability_threshold"]
                
                if metric == "skull_width_ratio":
                    stability_threshold = MEDICAL_VALIDATOR_THRESHOLDS["skull_stability_cv_threshold"]
                elif metric == "temporal_bone_angle":
                    stability_threshold = MEDICAL_VALIDATOR_THRESHOLDS["temporal_bone_stability_threshold"]
                elif metric == "zygomatic_arch_width":
                    stability_threshold = MEDICAL_VALIDATOR_THRESHOLDS["zygomatic_stability_threshold"]
                elif metric in ["orbital_depth", "orbital_index"]:
                    stability_threshold = MEDICAL_VALIDATOR_THRESHOLDS["orbital_stability_threshold"]

                is_stable = cv <= stability_threshold

                results[metric] = {
                    "stable": is_stable,
                    "cv": float(cv),
                    "mean": float(mean_val),
                    "std": float(std_val),
                    "threshold": stability_threshold,
                    "adult_data_points": int(np.sum(adult_mask)),
                    "reason": f"CV = {cv:.4f} {'<=' if is_stable else '>'} {stability_threshold:.4f}"
                }

                if not is_stable:
                    logger.warning(f"Нестабильность костной метрики {metric}: CV = {cv:.4f}")

            logger.info(f"Проверка костной структуры завершена для {len(results)} метрик")
            return results

        except Exception as e:
            logger.error(f"Ошибка проверки неизменности костной структуры: {e}")
            return {}

    def analyze_soft_tissue_aging_patterns(self, soft_tissue_metrics: Dict[str, List[float]],
                                         age_progression: List[float]) -> Dict[str, Dict[str, Any]]:
        """
        ИСПРАВЛЕНО: Анализ паттернов старения мягких тканей
        Согласно правкам: линейная регрессия и валидация наклона
        """
        if not soft_tissue_metrics or not age_progression:
            logger.warning("Нет данных для анализа паттернов старения мягких тканей")
            return {}

        try:
            logger.info(f"Анализ паттернов старения для {len(soft_tissue_metrics)} метрик мягких тканей")
            
            results = {}

            for metric, values in soft_tissue_metrics.items():
                values = np.array(values)
                ages = np.array(age_progression)

                # Убедимся, что длины arrays совпадают
                min_len = min(len(values), len(ages))
                values = values[:min_len]
                ages = ages[:min_len]

                if len(values) < 3 or len(ages) < 3:
                    results[metric] = {
                        "slope": 0.0,
                        "expected_negative": False,
                        "p_value": 1.0,
                        "r_squared": 0.0,
                        "biological_plausible": True,
                        "reason": "Недостаточно данных для анализа",
                        "intercept": 0.0,
                        "std_err": 0.0,
                        "significant": False
                    }
                    continue

                # Линейная регрессия
                slope, intercept, r_value, p_value, std_err = stats.linregress(ages, values)

                # Проверка биологической достоверности
                expected_negative = slope <= MEDICAL_VALIDATOR_THRESHOLDS["soft_tissue_aging_slope_threshold"]
                biological_plausible = True
                reason = "Нормальный паттерн старения"

                # Валидация наклона старения
                if abs(slope) > abs(MEDICAL_VALIDATOR_THRESHOLDS["soft_tissue_aging_slope_threshold"]) * 10:
                    biological_plausible = False
                    reason = f"Слишком быстрое изменение: {slope:.4f} в год"
                elif slope > 0 and metric in ["forehead_height_ratio", "nose_width_ratio"]:
                    # Положительный наклон для метрик, которые должны уменьшаться
                    biological_plausible = False
                    reason = f"Неожиданное увеличение метрики: {slope:.4f} в год"
                elif p_value > 0.05:
                    reason = f"Статистически незначимый тренд: p={p_value:.3f}"

                results[metric] = {
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "r_squared": float(r_value ** 2),
                    "p_value": float(p_value),
                    "std_err": float(std_err),
                    "expected_negative": expected_negative,
                    "biological_plausible": biological_plausible,
                    "reason": reason,
                    "significant": p_value < 0.05
                }

            logger.info(f"Анализ паттернов старения завершен для {len(results)} метрик")
            return results

        except Exception as e:
            logger.error(f"Ошибка анализа паттернов старения мягких тканей: {e}")
            return {}

    def exclude_surgical_hypotheses_by_timeline(self, metrics_changes: Dict[str, Dict[str, float]],
                                              time_intervals: Dict[str, int],
                                              medical_events: Optional[Dict[str, Dict[str, str]]] = None) -> Dict[str, Dict[str, Any]]:
        """
        ИСПРАВЛЕНО: Исключение хирургических гипотез по временной линии
        Согласно правкам: анализ 6-месячных интервалов и корреляция с событиями
        """
        if not metrics_changes:
            logger.warning("Нет данных изменений метрик для анализа хирургических гипотез")
            return {}

        # Использование переданных событий или загруженных по умолчанию
        events_to_use = medical_events if medical_events else self.health_events

        try:
            logger.info(f"Исключение хирургических гипотез для {len(metrics_changes)} периодов")
            
            results = {}

            for period, change in metrics_changes.items():
                interval = time_intervals.get(period, 0)

                # Проверка подходящего интервала для хирургии
                if interval < MEDICAL_VALIDATOR_THRESHOLDS["surgery_min_interval_days"]:
                    results[period] = {
                        "surgery_possible": False,
                        "reason": "Интервал слишком короткий для хирургического восстановления",
                        "interval_days": interval,
                        "event_found": False,
                        "correlated_events": [],
                        "significant_changes": []
                    }
                    continue

                elif interval > MEDICAL_VALIDATOR_THRESHOLDS["surgery_max_interval_days"]:
                    results[period] = {
                        "surgery_possible": False,
                        "reason": "Интервал слишком длинный для прямой связи с хирургией",
                        "interval_days": interval,
                        "event_found": False,
                        "correlated_events": [],
                        "significant_changes": []
                    }
                    continue

                # Поиск корреляции с медицинскими событиями
                event_found = False
                correlated_events = []

                # Преобразование периода в даты для сравнения
                try:
                    if isinstance(period, str) and '-' in period:
                        period_start_str, period_end_str = period.split(' - ')
                        period_start = datetime.strptime(period_start_str, "%Y-%m-%d")
                        period_end = datetime.strptime(period_end_str, "%Y-%m-%d")
                    else:
                        # Если период представлен одной датой
                        period_start = datetime.strptime(str(period), "%Y-%m-%d")
                        period_end = period_start + timedelta(days=interval)
                except (ValueError, AttributeError):
                    logger.warning(f"Не удалось распарсить период: {period}")
                    continue

                # Поиск событий в окне корреляции
                correlation_window = MEDICAL_VALIDATOR_THRESHOLDS["medical_event_correlation_days"]
                
                for event_date, event_info in events_to_use.items():
                    # Проверка попадания в окно корреляции
                    days_to_start = abs((event_date - period_start).days)
                    days_to_end = abs((event_date - period_end).days)
                    
                    if days_to_start <= correlation_window or days_to_end <= correlation_window:
                        event_found = True
                        correlated_events.append({
                            "date": event_date.strftime("%Y-%m-%d"),
                            "type": event_info.get("type", "unknown"),
                            "description": event_info.get("description", ""),
                            "days_to_period": min(days_to_start, days_to_end)
                        })

                # Анализ величины изменений
                significant_changes = []
                for metric_name, change_value in change.items():
                    if abs(change_value) > MEDICAL_VALIDATOR_THRESHOLDS["physiological_change_limit"]:
                        significant_changes.append({
                            "metric": metric_name,
                            "change": change_value,
                            "significant": True
                        })

                # Определение возможности хирургии
                surgery_possible = not event_found and len(significant_changes) > 0

                results[period] = {
                    "surgery_possible": surgery_possible,
                    "event_found": event_found,
                    "correlated_events": correlated_events,
                    "interval_days": interval,
                    "significant_changes": significant_changes,
                    "reason": self._generate_surgery_exclusion_reason(
                        surgery_possible, event_found, len(significant_changes), interval
                    )
                }

            # Обновление статистики
            excluded_surgeries = sum(1 for result in results.values() if not result["surgery_possible"])
            self.validation_stats['surgical_interventions_excluded'] += excluded_surgeries

            logger.info(f"Анализ хирургических гипотез завершен для {len(results)} периодов")
            return results

        except Exception as e:
            logger.error(f"Ошибка исключения хирургических гипотез: {e}")
            return {}

    def _generate_surgery_exclusion_reason(self, surgery_possible: bool, event_found: bool,
                                         significant_changes_count: int, interval_days: int) -> str:
        """Генерация объяснения исключения хирургии"""
        try:
            if not surgery_possible:
                if event_found:
                    return f"Изменения коррелируют с документированными медицинскими событиями (интервал: {interval_days} дней)"
                elif significant_changes_count == 0:
                    return f"Отсутствуют значительные изменения метрик (интервал: {interval_days} дней)"
                else:
                    return f"Интервал не подходит для хирургического восстановления: {interval_days} дней"
            else:
                return f"Возможно хирургическое вмешательство: {significant_changes_count} значительных изменений за {interval_days} дней"

        except Exception as e:
            logger.error(f"Ошибка генерации объяснения: {e}")
            return "Ошибка анализа"

    def validate_physiological_change_limits(self, metric_changes: Dict[str, List[float]],
                                           change_velocities: Dict[str, List[float]]) -> Dict[str, Dict[str, Any]]:
        """
        ИСПРАВЛЕНО: Валидация физиологических лимитов изменений
        Согласно правкам: проверка скоростей изменений
        """
        if not change_velocities:
            logger.warning("Нет данных скоростей изменений для валидации")
            return {}

        try:
            logger.info(f"Валидация физиологических лимитов для {len(change_velocities)} метрик")
            
            results = {}

            for metric, velocities in change_velocities.items():
                velocities = np.array(velocities)
                
                if velocities.size == 0:
                    results[metric] = {
                        "exceeds_limit": False,
                        "max_velocity": 0.0,
                        "mean_velocity": 0.0,
                        "std_velocity": 0.0,
                        "outlier_count": 0,
                        "reason": "Нет данных скоростей",
                        "physiological_limit": 0.0
                    }
                    continue

                # Проверка превышения физиологических лимитов
                physiological_limit = MEDICAL_VALIDATOR_THRESHOLDS["physiological_change_limit"]

                # Адаптация лимита для разных типов метрик
                if metric in ["skull_width_ratio", "temporal_bone_angle", "zygomatic_arch_width"]:
                    # Костные метрики - более строгие лимиты
                    physiological_limit *= 0.5
                elif metric in ["forehead_height_ratio", "nose_width_ratio", "mouth_width_ratio"]:
                    # Мягкие ткани - стандартные лимиты
                    physiological_limit *= 1.0

                exceeds_limit = np.any(np.abs(velocities) > physiological_limit)
                max_velocity = float(np.max(np.abs(velocities)))
                mean_velocity = float(np.mean(np.abs(velocities)))
                std_velocity = float(np.std(velocities))

                # Подсчет выбросов
                outlier_mask = np.abs(velocities) > physiological_limit
                outlier_count = int(np.sum(outlier_mask))

                # Генерация объяснения
                if exceeds_limit:
                    reason = f"Превышение физиологического лимита: max={max_velocity:.4f} > {physiological_limit:.4f}"
                    if outlier_count > 1:
                        reason += f" ({outlier_count} выбросов)"
                else:
                    reason = f"В пределах физиологических лимитов: max={max_velocity:.4f} <= {physiological_limit:.4f}"

                results[metric] = {
                    "exceeds_limit": exceeds_limit,
                    "max_velocity": max_velocity,
                    "mean_velocity": mean_velocity,
                    "std_velocity": std_velocity,
                    "outlier_count": outlier_count,
                    "physiological_limit": physiological_limit,
                    "reason": reason
                }

            logger.info(f"Валидация физиологических лимитов завершена для {len(results)} метрик")
            return results

        except Exception as e:
            logger.error(f"Ошибка валидации физиологических лимитов: {e}")
            return {}

    def correlate_anomalies_with_documented_health_events(self, anomaly_periods: List[Dict[str, Any]],
                                                        health_events: Optional[Dict[str, Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Корреляция аномалий с документированными медицинскими событиями
        Согласно правкам: временные окна и корреляционный анализ
        """
        if not anomaly_periods:
            logger.warning("Нет периодов аномалий для корреляции")
            return {}

        # Использование переданных событий или загруженных по умолчанию
        events_to_use = health_events if health_events else self.health_events

        try:
            logger.info(f"Корреляция {len(anomaly_periods)} аномалий с {len(events_to_use)} медицинскими событиями")
            
            correlations = {
                "total_correlations": 0,
                "correlated_anomalies": [],
                "uncorrelated_anomalies": [],
                "event_correlations": {},
                "correlation_statistics": {}
            }

            correlation_window = timedelta(days=MEDICAL_VALIDATOR_THRESHOLDS["medical_event_correlation_days"])

            for anomaly in anomaly_periods:
                anomaly_date_str = anomaly.get("date")
                if not anomaly_date_str:
                    continue

                try:
                    anomaly_date = datetime.strptime(anomaly_date_str, "%Y-%m-%d")
                except ValueError:
                    logger.warning(f"Неверный формат даты аномалии: {anomaly_date_str}")
                    continue

                # Поиск ближайших медицинских событий
                correlated_events = []
                for event_date, event_info in events_to_use.items():
                    time_diff = abs((anomaly_date - event_date).days)
                    if time_diff <= correlation_window.days:
                        correlated_events.append({
                            "event_date": event_date,
                            "event_info": event_info,
                            "days_difference": (anomaly_date - event_date).days,
                            "absolute_difference": time_diff,
                            "correlation_strength": self._calculate_correlation_strength(time_diff, correlation_window.days)
                        })

                if correlated_events:
                    # Выбор ближайшего события
                    closest_event = min(correlated_events, key=lambda x: x["absolute_difference"])
                    
                    correlation_entry = {
                        "anomaly": anomaly,
                        "correlated_event": closest_event,
                        "correlation_type": self._classify_correlation_type(closest_event["event_info"])
                    }

                    correlations["correlated_anomalies"].append(correlation_entry)
                    correlations["total_correlations"] += 1

                    # Группировка по событиям
                    event_key = closest_event["event_date"].strftime("%Y-%m-%d")
                    if event_key not in correlations["event_correlations"]:
                        correlations["event_correlations"][event_key] = {
                            "event_info": closest_event["event_info"],
                            "correlated_anomalies": []
                        }

                    correlations["event_correlations"][event_key]["correlated_anomalies"].append(correlation_entry)
                else:
                    correlations["uncorrelated_anomalies"].append(anomaly)

            # Статистики корреляций
            correlations["correlation_statistics"] = self._calculate_correlation_statistics(correlations)

            logger.info(f"Найдено {correlations['total_correlations']} корреляций аномалий с медицинскими событиями")
            return correlations

        except Exception as e:
            logger.error(f"Ошибка корреляции аномалий с медицинскими событиями: {e}")
            return {}

    def _calculate_correlation_strength(self, days_difference: int, max_window: int) -> str:
        """Расчет силы корреляции на основе временной близости"""
        try:
            ratio = days_difference / max_window

            if ratio <= 0.1:
                return "very_strong"
            elif ratio <= 0.3:
                return "strong"
            elif ratio <= 0.6:
                return "moderate"
            else:
                return "weak"

        except Exception as e:
            logger.error(f"Ошибка расчета силы корреляции: {e}")
            return "unknown"

    def _classify_correlation_type(self, event_info: Dict[str, str]) -> str:
        """Классификация типа корреляции на основе типа события"""
        try:
            event_type = event_info.get("type", "unknown")
            stress_level = event_info.get("stress_level", "unknown")

            if event_type == "health_crisis":
                return "direct_health_impact"
            elif event_type == "political_stress" and stress_level in ["high", "very_high"]:
                return "stress_related"
            elif event_type == "transition":
                return "lifestyle_change"
            else:
                return "indirect_correlation"

        except Exception as e:
            logger.error(f"Ошибка классификации корреляции: {e}")
            return "unknown"

    def _calculate_correlation_statistics(self, correlations: Dict[str, Any]) -> Dict[str, Any]:
        """Расчет статистик корреляций"""
        try:
            stats = {
                "correlation_rate": 0.0,
                "average_time_difference": 0.0,
                "most_correlated_event_type": None,
                "correlation_strength_distribution": {},
                "correlation_type_distribution": {}
            }

            if not correlations["correlated_anomalies"]:
                return stats

            # Распределение силы корреляций
            strength_counts = {}
            type_counts = {}
            time_differences = []

            for correlation in correlations["correlated_anomalies"]:
                strength = correlation["correlated_event"]["correlation_strength"]
                strength_counts[strength] = strength_counts.get(strength, 0) + 1

                corr_type = correlation["correlation_type"]
                type_counts[corr_type] = type_counts.get(corr_type, 0) + 1

                time_diff = abs(correlation["correlated_event"]["days_difference"])
                time_differences.append(time_diff)

            stats["correlation_strength_distribution"] = strength_counts
            stats["correlation_type_distribution"] = type_counts
            stats["average_time_difference"] = float(np.mean(time_differences))

            if type_counts:
                stats["most_correlated_event_type"] = max(type_counts, key=type_counts.get)

            # Общий коэффициент корреляции
            total_anomalies = len(correlations["correlated_anomalies"]) + len(correlations["uncorrelated_anomalies"])
            if total_anomalies > 0:
                stats["correlation_rate"] = len(correlations["correlated_anomalies"]) / total_anomalies

            return stats

        except Exception as e:
            logger.error(f"Ошибка расчета статистик корреляций: {e}")
            return {}

    def auto_generate_medical_report(self, validation_results: Dict[str, Any]) -> str:
        """
        ИСПРАВЛЕНО: Автогенерация медицинского отчета
        Согласно правкам: структурированный отчет с выводами
        """
        try:
            logger.info("Генерация медицинского отчета")
            
            report_sections = []

            # Заголовок отчета
            report_sections.append("# МЕДИЦИНСКИЙ ОТЧЕТ ВАЛИДАЦИИ")
            report_sections.append(f"Дата генерации: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_sections.append("")

            # Исполнительное резюме
            report_sections.append("## ИСПОЛНИТЕЛЬНОЕ РЕЗЮМЕ")
            
            aging_consistency = validation_results.get("aging_consistency", {})
            bone_stability = validation_results.get("bone_stability", {})
            soft_tissue_analysis = validation_results.get("soft_tissue_analysis", {})
            
            overall_score = aging_consistency.get("overall_aging_consistency_score", 0.0)

            if overall_score >= 0.8:
                conclusion = "ВЫСОКАЯ медицинская достоверность изменений"
            elif overall_score >= 0.6:
                conclusion = "УМЕРЕННАЯ медицинская достоверность изменений"
            else:
                conclusion = "НИЗКАЯ медицинская достоверность изменений"

            report_sections.append(f"**Общий вывод:** {conclusion}")
            report_sections.append(f"**Оценка согласованности старения:** {overall_score:.3f}")
            report_sections.append("")

            # Анализ костной структуры
            report_sections.append("## АНАЛИЗ КОСТНОЙ СТРУКТУРЫ")
            
            if bone_stability:
                stable_metrics = [metric for metric, data in bone_stability.items() if data.get("stable", False)]
                unstable_metrics = [metric for metric, data in bone_stability.items() if not data.get("stable", False)]
                
                report_sections.append(f"**Стабильные метрики:** {len(stable_metrics)}/{len(bone_stability)}")
                report_sections.append(f"**Нестабильные метрики:** {len(unstable_metrics)}")
                
                if unstable_metrics:
                    report_sections.append("**Детали нестабильности:**")
                    for metric in unstable_metrics:
                        data = bone_stability[metric]
                        report_sections.append(f"- {metric}: CV = {data.get('cv', 0):.4f} (порог: {data.get('threshold', 0):.4f})")
            else:
                report_sections.append("Данные анализа костной структуры отсутствуют")
            
            report_sections.append("")

            # Анализ мягких тканей
            report_sections.append("## АНАЛИЗ МЯГКИХ ТКАНЕЙ")
            
            if soft_tissue_analysis:
                plausible_metrics = [metric for metric, data in soft_tissue_analysis.items() if data.get("biological_plausible", False)]
                implausible_metrics = [metric for metric, data in soft_tissue_analysis.items() if not data.get("biological_plausible", False)]
                
                report_sections.append(f"**Биологически достоверные паттерны:** {len(plausible_metrics)}/{len(soft_tissue_analysis)}")
                report_sections.append(f"**Сомнительные паттерны:** {len(implausible_metrics)}")
                
                if implausible_metrics:
                    report_sections.append("**Детали сомнительных паттернов:**")
                    for metric in implausible_metrics:
                        data = soft_tissue_analysis[metric]
                        report_sections.append(f"- {metric}: {data.get('reason', 'Неизвестная причина')}")
            else:
                report_sections.append("Данные анализа мягких тканей отсутствуют")
            
            report_sections.append("")

            # Аномалии и рекомендации
            report_sections.append("## ВЫЯВЛЕННЫЕ АНОМАЛИИ")
            
            anomalies = aging_consistency.get("anomalies_details", [])
            if anomalies:
                for anomaly in anomalies:
                    report_sections.append(f"- {anomaly}")
            else:
                report_sections.append("Значительные аномалии не выявлены")
            
            report_sections.append("")

            # Рекомендации
            report_sections.append("## РЕКОМЕНДАЦИИ")
            
            if overall_score < 0.6:
                report_sections.append("- Требуется дополнительная медицинская экспертиза")
                report_sections.append("- Рекомендуется анализ дополнительных биометрических данных")
            elif len([metric for metric, data in bone_stability.items() if not data.get("stable", False)]) > 0 if bone_stability else False:
                report_sections.append("- Необходим углубленный анализ костной структуры")
            else:
                report_sections.append("- Изменения соответствуют естественному процессу старения")
            
            report_sections.append("")
            report_sections.append("---")
            report_sections.append("*Отчет сгенерирован автоматически системой медицинской валидации*")

            full_report = "\n".join(report_sections)
            
            logger.info("Медицинский отчет сгенерирован успешно")
            return full_report

        except Exception as e:
            logger.error(f"Ошибка генерации медицинского отчета: {e}")
            return f"Ошибка генерации отчета: {str(e)}"

    def calculate_biological_plausibility_score(self, changes_data: Dict[str, Any]) -> float:
        """
        ИСПРАВЛЕНО: Расчет оценки биологической достоверности
        Согласно правкам: комплексная оценка всех аспектов
        """
        try:
            logger.info("Расчет оценки биологической достоверности")
            
            if not changes_data:
                return 0.0

            plausibility_scores = []

            # Анализ костной стабильности
            bone_stability = changes_data.get("bone_stability", {})
            if bone_stability:
                stable_count = sum(1 for data in bone_stability.values() if data.get("stable", False))
                bone_score = stable_count / len(bone_stability)
                plausibility_scores.append(bone_score)

            # Анализ мягких тканей
            soft_tissue = changes_data.get("soft_tissue_analysis", {})
            if soft_tissue:
                plausible_count = sum(1 for data in soft_tissue.values() if data.get("biological_plausible", False))
                tissue_score = plausible_count / len(soft_tissue)
                plausibility_scores.append(tissue_score)

            # Анализ согласованности старения
            aging_consistency = changes_data.get("aging_consistency", {})
            if aging_consistency:
                consistency_score = aging_consistency.get("overall_aging_consistency_score", 0.0)
                plausibility_scores.append(consistency_score)

            # Анализ физиологических лимитов
            physiological_limits = changes_data.get("physiological_limits", {})
            if physiological_limits:
                within_limits_count = sum(1 for data in physiological_limits.values() if not data.get("exceeds_limit", True))
                limits_score = within_limits_count / len(physiological_limits)
                plausibility_scores.append(limits_score)

            # Общая оценка
            if plausibility_scores:
                overall_plausibility = np.mean(plausibility_scores)
            else:
                overall_plausibility = 0.5  # Нейтральная оценка при отсутствии данных

            logger.info(f"Биологическая достоверность: {overall_plausibility:.3f}")
            return float(np.clip(overall_plausibility, 0.0, 1.0))

        except Exception as e:
            logger.error(f"Ошибка расчета биологической достоверности: {e}")
            return 0.0

    def detect_surgical_intervention_evidence(self, temporal_metrics: Dict[str, List[float]], 
                                            dates: List[datetime]) -> Dict[str, Any]:
        """Детекция признаков хирургического вмешательства"""
        try:
            logger.info("Детекция признаков хирургического вмешательства")
            
            surgical_evidence = {
                "interventions_detected": [],
                "suspicious_periods": [],
                "rapid_changes": [],
                "overall_surgical_probability": 0.0
            }
            
            if len(dates) < 2:
                return surgical_evidence
            
            # Анализ быстрых изменений между соседними точками
            for i in range(1, len(dates)):
                time_diff = (dates[i] - dates[i-1]).days
                
                if 60 <= time_diff <= 365:  # Подходящий интервал для хирургии
                    rapid_changes = []
                    
                    for metric_name, values in temporal_metrics.items():
                        if i < len(values):
                            prev_value = values[i-1]
                            curr_value = values[i]
                            
                            if prev_value != 0:
                                change_rate = abs(curr_value - prev_value) / abs(prev_value)
                                
                                # Быстрое изменение > 10%
                                if change_rate > 0.1:
                                    rapid_changes.append({
                                        "metric": metric_name,
                                        "change_rate": change_rate,
                                        "prev_value": prev_value,
                                        "curr_value": curr_value
                                    })
                    
                    if len(rapid_changes) >= 2:  # Множественные быстрые изменения
                        surgical_evidence["interventions_detected"].append({
                            "date_range": (dates[i-1], dates[i]),
                            "time_interval_days": time_diff,
                            "changes": rapid_changes,
                            "confidence": min(1.0, len(rapid_changes) / 5.0)
                        })
            
            # Расчет общей вероятности хирургического вмешательства
            if surgical_evidence["interventions_detected"]:
                confidences = [intervention["confidence"] for intervention in surgical_evidence["interventions_detected"]]
                surgical_evidence["overall_surgical_probability"] = np.mean(confidences)
            
            logger.info(f"Обнаружено {len(surgical_evidence['interventions_detected'])} потенциальных хирургических вмешательств")
            return surgical_evidence
            
        except Exception as e:
            logger.error(f"Ошибка детекции хирургического вмешательства: {e}")
            return {
                "interventions_detected": [],
                "suspicious_periods": [],
                "rapid_changes": [],
                "overall_surgical_probability": 0.0
            }

    def validate_temporal_consistency(self, identity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Валидация временной консистентности идентичности"""
        try:
            logger.info("Валидация временной консистентности")
            
            consistency_result = {
                "is_temporally_consistent": True,
                "consistency_score": 1.0,
                "temporal_gaps": [],
                "inconsistency_periods": [],
                "recommendations": []
            }
            
            temporal_points = identity_data.get("temporal_points", [])
            if len(temporal_points) < 2:
                consistency_result["recommendations"].append("Недостаточно временных точек для валидации")
                return consistency_result
            
            # Сортировка по дате
            sorted_points = sorted(temporal_points, key=lambda p: p.get("date", datetime.min))
            
            # Анализ временных разрывов
            for i in range(1, len(sorted_points)):
                prev_date = sorted_points[i-1].get("date")
                curr_date = sorted_points[i].get("date")
                
                if prev_date and curr_date:
                    gap_days = (curr_date - prev_date).days
                    
                    if gap_days > 365:  # Разрыв больше года
                        consistency_result["temporal_gaps"].append({
                            "start_date": prev_date,
                            "end_date": curr_date,
                            "gap_days": gap_days,
                            "severity": "high" if gap_days > 730 else "medium"
                        })
            
            # Оценка общей консистентности
            if consistency_result["temporal_gaps"]:
                gap_penalty = len(consistency_result["temporal_gaps"]) * 0.1
                consistency_result["consistency_score"] = max(0.0, 1.0 - gap_penalty)
                
                if consistency_result["consistency_score"] < CRITICAL_THRESHOLDS["temporal_stability_threshold"]:
                    consistency_result["is_temporally_consistent"] = False
                    consistency_result["recommendations"].append("Обнаружены значительные временные разрывы")
            
            return consistency_result
            
        except Exception as e:
            logger.error(f"Ошибка валидации временной консистентности: {e}")
            return {
                "is_temporally_consistent": False,
                "consistency_score": 0.0,
                "temporal_gaps": [],
                "inconsistency_periods": [],
                "recommendations": [f"Ошибка валидации: {str(e)}"]
            }

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Получение статистики обработки"""
        stats = self.validation_stats.copy()
        
        # Добавление вычисляемых метрик
        if stats['total_validations'] > 0:
            stats['consistency_rate'] = stats['consistent_identities'] / stats['total_validations']
            stats['bone_anomaly_rate'] = stats['bone_anomalies_detected'] / stats['total_validations']
        else:
            stats['consistency_rate'] = 0.0
            stats['bone_anomaly_rate'] = 0.0
        
        # Информация о кэше
        stats['cache_info'] = {
            'validation_cache_size': len(self.validation_cache),
            'health_events_loaded': len(self.health_events)
        }
        
        # Информация о памяти
        process = psutil.Process()
        memory_info = process.memory_info()
        stats['memory_usage_mb'] = memory_info.rss / (1024 * 1024)
        
        return stats

    def clear_cache(self):
        """Очистка кэша валидации"""
        try:
            self.validation_cache.clear()
            logger.info("Кэш MedicalValidator очищен")
            
        except Exception as e:
            logger.error(f"Ошибка очистки кэша: {e}")

    def save_validation_cache(self, cache_file: str = "medical_validation_cache.pkl"):
        """Сохранение кэша валидации"""
        try:
            cache_path = self.cache_dir / cache_file
            
            cache_data = {
                'validation_cache': self.validation_cache,
                'health_events': self.health_events,
                'calibrated': self.calibrated,
                'validation_stats': self.validation_stats
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"Кэш валидации сохранен: {cache_path}")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения кэша валидации: {e}")

    def load_validation_cache(self, cache_file: str = "medical_validation_cache.pkl") -> bool:
        """Загрузка кэша валидации"""
        try:
            cache_path = self.cache_dir / cache_file
            
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                
                self.validation_cache = cache_data.get('validation_cache', {})
                self.health_events = cache_data.get('health_events', {})
                self.calibrated = cache_data.get('calibrated', False)
                self.validation_stats.update(cache_data.get('validation_stats', {}))
                
                logger.info(f"Кэш валидации загружен: {cache_path}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Ошибка загрузки кэша валидации: {e}")
            return False

    def detect_surgical_intervention_evidence(self, temporal_metrics: Dict[str, List[float]], 
                                            dates: List[datetime]) -> Dict[str, Any]:
        """Детекция признаков хирургического вмешательства"""
        try:
            logger.info("Детекция признаков хирургического вмешательства")
            
            surgical_evidence = {
                "interventions_detected": [],
                "suspicious_periods": [],
                "rapid_changes": [],
                "overall_surgical_probability": 0.0
            }
            
            if len(dates) < 2:
                return surgical_evidence
            
            # Анализ быстрых изменений между соседними точками
            for i in range(1, len(dates)):
                time_diff = (dates[i] - dates[i-1]).days
                
                if 60 <= time_diff <= 365:  # Подходящий интервал для хирургии
                    rapid_changes = []
                    
                    for metric_name, values in temporal_metrics.items():
                        if i < len(values):
                            prev_value = values[i-1]
                            curr_value = values[i]
                            
                            if prev_value != 0:
                                change_rate = abs(curr_value - prev_value) / abs(prev_value)
                                
                                # Быстрое изменение > 10%
                                if change_rate > 0.1:
                                    rapid_changes.append({
                                        "metric": metric_name,
                                        "change_rate": change_rate,
                                        "prev_value": prev_value,
                                        "curr_value": curr_value
                                    })
                    
                    if len(rapid_changes) >= 2:  # Множественные быстрые изменения
                        surgical_evidence["interventions_detected"].append({
                            "date_range": (dates[i-1], dates[i]),
                            "time_interval_days": time_diff,
                            "changes": rapid_changes,
                            "confidence": min(1.0, len(rapid_changes) / 5.0)
                        })
            
            # Расчет общей вероятности хирургического вмешательства
            if surgical_evidence["interventions_detected"]:
                confidences = [intervention["confidence"] for intervention in surgical_evidence["interventions_detected"]]
                surgical_evidence["overall_surgical_probability"] = np.mean(confidences)
            
            logger.info(f"Обнаружено {len(surgical_evidence['interventions_detected'])} потенциальных хирургических вмешательств")
            return surgical_evidence
            
        except Exception as e:
            logger.error(f"Ошибка детекции хирургического вмешательства: {e}")
            return {
                "interventions_detected": [],
                "suspicious_periods": [],
                "rapid_changes": [],
                "overall_surgical_probability": 0.0
            }

    def validate_temporal_consistency(self, identity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Валидация временной консистентности идентичности"""
        try:
            logger.info("Валидация временной консистентности")
            
            consistency_result = {
                "is_temporally_consistent": True,
                "consistency_score": 1.0,
                "temporal_gaps": [],
                "inconsistency_periods": [],
                "recommendations": []
            }
            
            temporal_points = identity_data.get("temporal_points", [])
            if len(temporal_points) < 2:
                consistency_result["recommendations"].append("Недостаточно временных точек для валидации")
                return consistency_result
            
            # Сортировка по дате
            sorted_points = sorted(temporal_points, key=lambda p: p.get("date", datetime.min))
            
            # Анализ временных разрывов
            for i in range(1, len(sorted_points)):
                prev_date = sorted_points[i-1].get("date")
                curr_date = sorted_points[i].get("date")
                
                if prev_date and curr_date:
                    gap_days = (curr_date - prev_date).days
                    
                    if gap_days > 365:  # Разрыв больше года
                        consistency_result["temporal_gaps"].append({
                            "start_date": prev_date,
                            "end_date": curr_date,
                            "gap_days": gap_days,
                            "severity": "high" if gap_days > 730 else "medium"
                        })
            
            # Оценка общей консистентности
            if consistency_result["temporal_gaps"]:
                gap_penalty = len(consistency_result["temporal_gaps"]) * 0.1
                consistency_result["consistency_score"] = max(0.0, 1.0 - gap_penalty)
                
                if consistency_result["consistency_score"] < CRITICAL_THRESHOLDS["temporal_stability_threshold"]:
                    consistency_result["is_temporally_consistent"] = False
                    consistency_result["recommendations"].append("Обнаружены значительные временные разрывы")
            
            return consistency_result
            
        except Exception as e:
            logger.error(f"Ошибка валидации временной консистентности: {e}")
            return {
                "is_temporally_consistent": False,
                "consistency_score": 0.0,
                "temporal_gaps": [],
                "inconsistency_periods": [],
                "recommendations": [f"Ошибка валидации: {str(e)}"]
            }

    def analyze_aging_rate_deviations(self, identity_timeline: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ отклонений скорости старения"""
        try:
            logger.info("Анализ отклонений скорости старения")
            
            deviation_analysis = {
                "overall_deviation_score": 0.0,
                "metric_deviations": {},
                "accelerated_aging_periods": [],
                "decelerated_aging_periods": [],
                "medical_explanations": []
            }
            
            temporal_points = identity_timeline.get("temporal_points", [])
            if len(temporal_points) < 3:
                return deviation_analysis
            
            # Сортировка по возрасту
            sorted_points = sorted(temporal_points, key=lambda p: p.get("age", 0))
            
            # Анализ каждой метрики
            for metric_name in sorted_points[0].get("metrics", {}):
                ages = [p.get("age", 0) for p in sorted_points]
                values = [p.get("metrics", {}).get(metric_name, 0) for p in sorted_points]
                
                if len(values) >= 3:
                    # Расчет скорости изменения
                    slopes = []
                    for i in range(2, len(values)):
                        age_diff = ages[i] - ages[i-2]
                        value_diff = values[i] - values[i-2]
                        
                        if age_diff > 0:
                            slope = value_diff / age_diff
                            slopes.append(slope)
                    
                    if slopes:
                        mean_slope = np.mean(slopes)
                        std_slope = np.std(slopes)
                        
                        # Ожидаемая скорость старения
                        expected_rate = self._get_expected_aging_rate(metric_name)
                        
                        # Отклонение от ожидаемой скорости
                        deviation = abs(mean_slope - expected_rate)
                        normalized_deviation = deviation / (abs(expected_rate) + 1e-6)
                        
                        deviation_analysis["metric_deviations"][metric_name] = {
                            "observed_rate": mean_slope,
                            "expected_rate": expected_rate,
                            "deviation": deviation,
                            "normalized_deviation": normalized_deviation,
                            "rate_variability": std_slope
                        }
            
            # Общая оценка отклонения
            if deviation_analysis["metric_deviations"]:
                deviations = [data["normalized_deviation"] for data in deviation_analysis["metric_deviations"].values()]
                deviation_analysis["overall_deviation_score"] = np.mean(deviations)
            
            return deviation_analysis
            
        except Exception as e:
            logger.error(f"Ошибка анализа отклонений скорости старения: {e}")
            return {
                "overall_deviation_score": 0.0,
                "metric_deviations": {},
                "accelerated_aging_periods": [],
                "decelerated_aging_periods": [],
                "medical_explanations": []
            }

    def _get_expected_aging_rate(self, metric_name: str) -> float:
        """Получение ожидаемой скорости старения для метрики"""
        try:
            expected_rates = {
                "forehead_height_ratio": -0.001,  # Уменьшение на 0.1% в год
                "nose_width_ratio": 0.0005,       # Увеличение на 0.05% в год
                "mouth_width_ratio": -0.0005,     # Уменьшение на 0.05% в год
                "cheek_fullness_ratio": -0.002,   # Уменьшение на 0.2% в год
                "nasolabial_depth": 0.001,        # Углубление на 0.1% в год
                "skull_width_ratio": 0.0,         # Стабильность после 25 лет
                "temporal_bone_angle": 0.0,       # Стабильность
                "zygomatic_arch_width": 0.0,      # Стабильность
                "interpupillary_distance_ratio": 0.0  # Стабильность
            }
            
            return expected_rates.get(metric_name, -0.001)  # По умолчанию небольшое уменьшение
            
        except Exception as e:
            logger.error(f"Ошибка получения ожидаемой скорости: {e}")
            return 0.0

    def calibrate_medical_thresholds(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Калибровка медицинских порогов на исторических данных"""
        try:
            logger.info(f"Калибровка медицинских порогов на {len(historical_data)} образцах")
            
            calibration_result = {
                "calibrated_thresholds": {},
                "calibration_quality": 0.0,
                "recommendations": []
            }
            
            if len(historical_data) < 50:
                calibration_result["recommendations"].append("Недостаточно данных для надежной калибровки")
                return calibration_result
            
            # Анализ распределений метрик
            metrics_distributions = defaultdict(list)
            
            for data_point in historical_data:
                for metric_name, value in data_point.get("metrics", {}).items():
                    metrics_distributions[metric_name].append(value)
            
            # Калибровка порогов для каждой метрики
            for metric_name, values in metrics_distributions.items():
                if len(values) >= 30:
                    # Статистические характеристики
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    
                    # Пороги на основе процентилей
                    p95 = np.percentile(values, 95)
                    p5 = np.percentile(values, 5)
                    
                    calibration_result["calibrated_thresholds"][metric_name] = {
                        "mean": mean_val,
                        "std": std_val,
                        "upper_threshold": p95,
                        "lower_threshold": p5,
                        "rapid_change_threshold": std_val * 3
                    }
            
            # Оценка качества калибровки
            quality_scores = []
            for metric_data in calibration_result["calibrated_thresholds"].values():
                # Качество на основе стабильности распределения
                quality = 1.0 / (1.0 + metric_data["std"] / (abs(metric_data["mean"]) + 1e-6))
                quality_scores.append(quality)
            
            calibration_result["calibration_quality"] = np.mean(quality_scores) if quality_scores else 0.0
            
            # Обновление внутренних порогов
            self._update_thresholds_from_calibration(calibration_result["calibrated_thresholds"])
            
            logger.info(f"Калибровка завершена. Качество: {calibration_result['calibration_quality']:.3f}")
            return calibration_result
            
        except Exception as e:
            logger.error(f"Ошибка калибровки медицинских порогов: {e}")
            return {
                "calibrated_thresholds": {},
                "calibration_quality": 0.0,
                "recommendations": [f"Ошибка калибровки: {str(e)}"]
            }

    def _update_thresholds_from_calibration(self, calibrated_thresholds: Dict[str, Dict[str, float]]):
        """Обновление порогов из результатов калибровки"""
        try:
            # Обновление порогов быстрых изменений
            rapid_change_thresholds = []
            for metric_data in calibrated_thresholds.values():
                rapid_change_thresholds.append(metric_data.get("rapid_change_threshold", 0.15))
            
            if rapid_change_thresholds:
                MEDICAL_VALIDATOR_THRESHOLDS["rapid_change_threshold"] = np.mean(rapid_change_thresholds)
            
            logger.info("Пороги обновлены из калибровки")
            
        except Exception as e:
            logger.error(f"Ошибка обновления порогов: {e}")

    def export_medical_validation_report(self, validation_results: Dict[str, Any], 
                                    output_format: str = "json") -> str:
        """Экспорт отчета медицинской валидации"""
        try:
            logger.info(f"Экспорт медицинского отчета в формате {output_format}")
            
            # Подготовка данных отчета
            report_data = {
                "metadata": {
                    "validation_date": datetime.now().isoformat(),
                    "validator_version": "2.0",
                    "putin_birth_date": PUTIN_BIRTH_DATE.isoformat(),
                    "medical_thresholds": MEDICAL_VALIDATOR_THRESHOLDS
                },
                "validation_results": validation_results,
                "summary": self._generate_validation_summary(validation_results)
            }
            
            # Определение пути выходного файла
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if output_format.lower() == "json":
                output_path = self.cache_dir / f"medical_validation_report_{timestamp}.json"
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
                    
            elif output_format.lower() == "html":
                output_path = self.cache_dir / f"medical_validation_report_{timestamp}.html"
                
                html_content = self._generate_html_medical_report(report_data)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                    
            else:
                raise ValueError(f"Неподдерживаемый формат: {output_format}")
            
            logger.info(f"Медицинский отчет экспортирован: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Ошибка экспорта медицинского отчета: {e}")
            return ""

    def _generate_validation_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация сводки валидации"""
        try:
            summary = {
                "overall_medical_plausibility": 0.0,
                "critical_findings": [],
                "bone_structure_violations": 0,
                "soft_tissue_anomalies": 0,
                "surgical_interventions_detected": 0,
                "temporal_inconsistencies": 0,
                "confidence_level": "unknown"
            }
            
            # Анализ костной структуры
            bone_stability = validation_results.get("bone_stability", {})
            if bone_stability:
                violations = sum(1 for data in bone_stability.values() if not data.get("stable", True))
                summary["bone_structure_violations"] = violations
                
                if violations > 0:
                    summary["critical_findings"].append(f"Нарушения стабильности костной структуры: {violations}")
            
            # Анализ мягких тканей
            soft_tissue = validation_results.get("soft_tissue_analysis", {})
            if soft_tissue:
                anomalies = sum(1 for data in soft_tissue.values() if not data.get("biological_plausible", True))
                summary["soft_tissue_anomalies"] = anomalies
                
                if anomalies > 0:
                    summary["critical_findings"].append(f"Аномалии мягких тканей: {anomalies}")
            
            # Анализ хирургических вмешательств
            surgical_evidence = validation_results.get("surgical_evidence", {})
            if surgical_evidence:
                interventions = len(surgical_evidence.get("interventions_detected", []))
                summary["surgical_interventions_detected"] = interventions
                
                if interventions > 0:
                    summary["critical_findings"].append(f"Потенциальные хирургические вмешательства: {interventions}")
            
            # Общая медицинская правдоподобность
            plausibility_scores = []
            
            aging_consistency = validation_results.get("aging_consistency", {})
            if aging_consistency:
                plausibility_scores.append(aging_consistency.get("overall_aging_consistency_score", 0.0))
            
            if bone_stability:
                stable_ratio = sum(1 for data in bone_stability.values() if data.get("stable", False)) / len(bone_stability)
                plausibility_scores.append(stable_ratio)
            
            if soft_tissue:
                plausible_ratio = sum(1 for data in soft_tissue.values() if data.get("biological_plausible", False)) / len(soft_tissue)
                plausibility_scores.append(plausible_ratio)
            
            if plausibility_scores:
                summary["overall_medical_plausibility"] = np.mean(plausibility_scores)
            
            # Уровень доверия
            if summary["overall_medical_plausibility"] >= 0.8:
                summary["confidence_level"] = "high"
            elif summary["overall_medical_plausibility"] >= 0.6:
                summary["confidence_level"] = "medium"
            else:
                summary["confidence_level"] = "low"
            
            return summary
            
        except Exception as e:
            logger.error(f"Ошибка генерации сводки валидации: {e}")
            return {
                "overall_medical_plausibility": 0.0,
                "critical_findings": ["Ошибка анализа"],
                "bone_structure_violations": 0,
                "soft_tissue_anomalies": 0,
                "surgical_interventions_detected": 0,
                "temporal_inconsistencies": 0,
                "confidence_level": "unknown"
            }

    def _generate_html_medical_report(self, report_data: Dict[str, Any]) -> str:
        """Генерация HTML медицинского отчета"""
        try:
            html_template = """
            <!DOCTYPE html>
            <html lang="ru">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Медицинский отчет валидации</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .header { background-color: #f8f9fa; padding: 20px; border-radius: 5px; }
                    .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                    .critical { background-color: #ffebee; border-color: #f44336; }
                    .warning { background-color: #fff3e0; border-color: #ff9800; }
                    .normal { background-color: #e8f5e8; border-color: #4caf50; }
                    .metric { margin: 10px 0; padding: 10px; background-color: #f5f5f5; border-radius: 3px; }
                    table { width: 100%; border-collapse: collapse; margin: 10px 0; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Медицинский отчет валидации</h1>
                    <p><strong>Дата валидации:</strong> {validation_date}</p>
                    <p><strong>Версия валидатора:</strong> {validator_version}</p>
                </div>
                
                <div class="section {summary_class}">
                    <h2>Сводка результатов</h2>
                    <p><strong>Общая медицинская правдоподобность:</strong> {overall_plausibility:.2f}</p>
                    <p><strong>Уровень доверия:</strong> {confidence_level}</p>
                    <p><strong>Нарушения костной структуры:</strong> {bone_violations}</p>
                    <p><strong>Аномалии мягких тканей:</strong> {soft_tissue_anomalies}</p>
                    {critical_findings_html}
                </div>
                
                {validation_sections}
                
            </body>
            </html>
            """
            
            # Подготовка данных для шаблона
            metadata = report_data.get("metadata", {})
            summary = report_data.get("summary", {})
            
            # Определение класса CSS для сводки
            confidence = summary.get("confidence_level", "unknown")
            if confidence == "high":
                summary_class = "normal"
            elif confidence == "medium":
                summary_class = "warning"
            else:
                summary_class = "critical"
            
            # Генерация списка критических находок
            critical_findings = summary.get("critical_findings", [])
            if critical_findings:
                critical_findings_html = "<p><strong>Критические находки:</strong></p><ul>"
                for finding in critical_findings:
                    critical_findings_html += f"<li>{finding}</li>"
                critical_findings_html += "</ul>"
            else:
                critical_findings_html = "<p><strong>Критических находок не обнаружено</strong></p>"
            
            # Заполнение шаблона
            html_content = html_template.format(
                validation_date=metadata.get("validation_date", "Неизвестно"),
                validator_version=metadata.get("validator_version", "Неизвестно"),
                summary_class=summary_class,
                overall_plausibility=summary.get("overall_medical_plausibility", 0.0),
                confidence_level=confidence,
                bone_violations=summary.get("bone_structure_violations", 0),
                soft_tissue_anomalies=summary.get("soft_tissue_anomalies", 0),
                critical_findings_html=critical_findings_html,
                validation_sections="<!-- Дополнительные секции валидации -->"
            )
            
            return html_content
            
        except Exception as e:
            logger.error(f"Ошибка генерации HTML отчета: {e}")
            return f"<html><body><h1>Ошибка генерации отчета</h1><p>{str(e)}</p></body></html>"

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Получение статистики обработки"""
        stats = self.validation_stats.copy()
        
        # Добавление вычисляемых метрик
        if stats['total_validations'] > 0:
            stats['consistency_rate'] = stats['consistent_identities'] / stats['total_validations']
            stats['bone_anomaly_rate'] = stats['bone_anomalies_detected'] / stats['total_validations']
        else:
            stats['consistency_rate'] = 0.0
            stats['bone_anomaly_rate'] = 0.0
        
        # Информация о кэше
        stats['cache_info'] = {
            'validation_cache_size': len(self.validation_cache),
            'health_events_loaded': len(self.health_events)
        }
        
        # Информация о памяти
        process = psutil.Process()
        memory_info = process.memory_info()
        stats['memory_usage_mb'] = memory_info.rss / (1024 * 1024)
        
        return stats

    def clear_cache(self):
        """Очистка кэша валидации"""
        try:
            self.validation_cache.clear()
            logger.info("Кэш MedicalValidator очищен")
            
        except Exception as e:
            logger.error(f"Ошибка очистки кэша: {e}")

    def self_test(self):
        """Самотестирование модуля"""
        logger.info("=== Самотестирование MedicalValidator ===")
        
        # Информация о параметрах
        logger.info(f"Aging model: {self.aging_model}")
        logger.info(f"Health events: {len(self.health_events)}")
        logger.info(f"Калиброван: {self.calibrated}")
        
        # Тестовые данные
        test_timeline = {
            "2020-01-01": {"skull_width_ratio": 0.8, "forehead_height_ratio": 0.35},
            "2021-01-01": {"skull_width_ratio": 0.81, "forehead_height_ratio": 0.34},
            "2022-01-01": {"skull_width_ratio": 0.805, "forehead_height_ratio": 0.33}
        }
        test_ages = [68.0, 69.0, 70.0]
        
        try:
            # Тест валидации согласованности старения
            aging_validation = self.validate_aging_consistency_for_identity(test_timeline, test_ages)
            logger.info(f"Тест валидации старения: score={aging_validation['overall_aging_consistency_score']:.3f}")
            
            # Тест проверки костной структуры
            cranial_metrics = {
                "skull_width_ratio": [0.8, 0.801, 0.802, 0.803],
                "temporal_bone_angle": [110.0, 110.1, 109.9, 110.2]
            }
            bone_stability = self.check_bone_structure_immutability(cranial_metrics, [68, 69, 70, 71])
            logger.info(f"Тест костной структуры: {len(bone_stability)} метрик проанализировано")
            
            # Тест анализа мягких тканей
            soft_tissue_metrics = {
                "forehead_height_ratio": [0.35, 0.34, 0.33, 0.32],
                "nose_width_ratio": [0.25, 0.24, 0.23, 0.22]
            }
            tissue_analysis = self.analyze_soft_tissue_aging_patterns(soft_tissue_metrics, test_ages)
            logger.info(f"Тест мягких тканей: {len(tissue_analysis)} метрик проанализировано")
            
            # Тест корреляции с медицинскими событиями
            test_anomalies = [
                {"date": "2020-03-15", "type": "geometric", "severity": "medium"},
                {"date": "2024-03-20", "type": "temporal", "severity": "high"}
            ]
            correlations = self.correlate_anomalies_with_documented_health_events(test_anomalies)
            logger.info(f"Тест корреляций: {correlations.get('total_correlations', 0)} корреляций найдено")
            
            # Тест генерации отчета
            validation_results = {
                "aging_consistency": aging_validation,
                "bone_stability": bone_stability,
                "soft_tissue_analysis": tissue_analysis
            }
            report = self.auto_generate_medical_report(validation_results)
            logger.info(f"Тест отчета: {len(report)} символов сгенерировано")
            
            # Тест экспорта отчета
            export_path = self.export_medical_validation_report(validation_results, "json")
            logger.info(f"Тест экспорта: {export_path}")
            
        except Exception as e:
            logger.error(f"Ошибка самотестирования: {e}")
        
        logger.info("=== Самотестирование завершено ===")

    # === ФУНКЦИИ САМОТЕСТИРОВАНИЯ ===

    def self_test():
        """Самотестирование модуля medical_validator"""
        try:
            logger.info("Запуск самотестирования medical_validator...")
            
            # Создание экземпляра валидатора
            validator = MedicalValidator()
            
            # Тест загрузки медицинских событий
            assert len(validator.health_events) > 0, "Медицинские события не загружены"
            
            # Тест предсказания метрик
            baseline_metrics = {"skull_width_ratio": 0.8, "forehead_height_ratio": 0.35}
            predicted = validator._predict_metrics_for_age(70.0, baseline_metrics)
            assert len(predicted) == len(baseline_metrics), "Неверное количество предсказанных метрик"
            
            # Тест расчета биологической правдоподобности
            test_data = {
                "bone_stability": {"skull_width_ratio": {"stable": True}},
                "soft_tissue_analysis": {"forehead_height_ratio": {"biological_plausible": True}}
            }
            plausibility = validator.calculate_biological_plausibility_score(test_data)
            assert 0.0 <= plausibility <= 1.0, "Неверный диапазон биологической правдоподобности"
            
            # Тест статистики
            stats = validator.get_processing_statistics()
            assert 'consistency_rate' in stats, "Отсутствует статистика консистентности"
            
            logger.info("Самотестирование medical_validator завершено успешно")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка самотестирования: {e}")
            return False

    # === ИНИЦИАЛИЗАЦИЯ ===

    if __name__ == "__main__":
        # Запуск самотестирования при прямом вызове модуля
        success = self_test()
        if success:
            print("✅ Модуль medical_validator работает корректно")
            
            # Демонстрация основной функциональности
            validator = MedicalValidator()
            print(f"📊 Медицинских событий: {len(validator.health_events)}")
            print(f"🔧 Модель старения: {validator.aging_model}")
            print(f"📏 Пороги валидации: {len(MEDICAL_VALIDATOR_THRESHOLDS)} параметров")
            print(f"💾 Кэш-директория: {validator.cache_dir}")
            print(f"🎛️ Калиброван: {validator.calibrated}")
        else:
            print("❌ Обнаружены ошибки в модуле medical_validator")
            exit(1)