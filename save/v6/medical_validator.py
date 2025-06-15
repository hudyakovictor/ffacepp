"""
MedicalValidator - Валидатор медицинской достоверности изменений лица
Версия: 2.0
Дата: 2025-06-15
Исправлены все критические ошибки согласно правкам
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

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('logs/medicalvalidator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Импорт конфигурации
try:
    from core_config import (
        PUTIN_BIRTH_DATE, AGING_MODEL, CRITICAL_THRESHOLDS,
        get_age_adjusted_thresholds, CACHE_DIR, ERROR_CODES
    )
    logger.info("Конфигурация успешно импортирована")
except ImportError as e:
    logger.error(f"Ошибка импорта конфигурации: {e}")
    # Значения по умолчанию
    PUTIN_BIRTH_DATE = date(1952, 10, 7)
    AGING_MODEL = {
        "elasticity_loss_per_year": 0.015,
        "tissue_sagging_per_year": 1.51,
        "bone_stability_threshold": 25
    }
    CRITICAL_THRESHOLDS = {"aging_consistency_threshold": 0.6}
    CACHE_DIR = Path("cache")
    ERROR_CODES = {"E006": "INVALID_DATE_FORMAT"}

# ==================== КОНСТАНТЫ МЕДИЦИНСКОЙ ВАЛИДАЦИИ ====================

# ИСПРАВЛЕНО: Пороги медицинской валидации согласно правкам
MEDICAL_VALIDATOR_THRESHOLDS = {
    "bone_metric_stability_threshold": 0.05,      # CV < 5% для костных метрик после 25 лет
    "soft_tissue_aging_slope_threshold": -0.02,   # Максимальный наклон старения мягких тканей
    "surgery_min_interval_days": 60,               # Минимальный интервал для хирургии
    "surgery_max_interval_days": 365,              # Максимальный интервал для хирургии
    "medical_event_correlation_days": 30,          # Окно корреляции с медицинскими событиями
    "physiological_change_limit": 0.02,           # Максимальное физиологическое изменение в год
    "aging_consistency_scale_factor": 2.0,        # Масштабирующий фактор для consistency
    "reverse_aging_threshold": -0.05,             # Порог обратного старения
    "rapid_change_threshold": 0.15,               # Порог быстрых изменений
    "bone_growth_threshold_age": 25,               # Возраст стабильности костей
    "ipd_variation_max": 0.02,                    # Максимальная вариация IPD после 25 лет
    "skull_stability_cv_threshold": 0.03,         # CV для стабильности черепа
    "temporal_bone_stability_threshold": 0.04,    # Стабильность височных костей
    "zygomatic_stability_threshold": 0.04,        # Стабильность скуловых костей
    "orbital_stability_threshold": 0.03           # Стабильность орбитальных метрик
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

# ==================== ОСНОВНОЙ КЛАСС ====================

class MedicalValidator:
    """
    Валидатор медицинской достоверности с полной функциональностью
    ИСПРАВЛЕНО: Все критические ошибки согласно правкам
    """
    
    def __init__(self):
        """Инициализация медицинского валидатора"""
        logger.info("Инициализация MedicalValidator")
        
        # Медицинские события
        self.health_events = self._load_health_events()
        
        # Кэш результатов валидации
        self.validation_cache = {}
        
        # Модель старения
        self.aging_model = AGING_MODEL.copy()
        
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
                "metric_deviations": {}
            }
        
        try:
            logger.info(f"Валидация согласованности старения для {len(identity_metrics_timeline)} временных точек")
            
            validation_results = {
                "overall_aging_consistency_score": 1.0,
                "anomalies_detected": False,
                "anomalies_details": [],
                "metric_deviations": {}
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
            
            # Анализ каждой временной точки
            for i in range(1, len(sorted_dates)):
                current_date = sorted_dates[i]
                current_metrics = identity_metrics_timeline[current_date]
                current_age = age_progression[i] if i < len(age_progression) else 50.0
                
                if not current_metrics:
                    logger.warning(f"Отсутствуют метрики для даты: {current_date}")
                    continue
                
                # ИСПРАВЛЕНО: Предсказание ожидаемых метрик на основе aging model
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
                        
                        # Нормализованное отклонение
                        normalized_deviation = deviation / (baseline_metrics.get(metric_name, 1.0) + 1e-6)
                        
                        deviations_for_period[metric_name] = normalized_deviation
                        all_deviations.append(normalized_deviation)
                        period_deviations_sum += normalized_deviation
                        relevant_metrics_count += 1
                        
                        # ИСПРАВЛЕНО: Проверка на обратное старение для мягких тканей
                        if metric_name in ["forehead_height_ratio", "nose_width_ratio", "mouth_width_ratio"]:
                            if (actual_value - predicted_value) > MEDICAL_VALIDATOR_THRESHOLDS["reverse_aging_threshold"]:
                                validation_results["anomalies_detected"] = True
                                validation_results["anomalies_details"].append(
                                    f"Обратное старение обнаружено {current_date}: {metric_name} = {actual_value - predicted_value:.4f}"
                                )
                        
                        # ИСПРАВЛЕНО: Проверка стабильности костных метрик после 25 лет
                        if metric_name in ["skull_width_ratio", "temporal_bone_angle", "zygomatic_arch_width", "interpupillary_distance_ratio"] and current_age >= 25:
                            if abs(actual_value - predicted_value) > MEDICAL_VALIDATOR_THRESHOLDS["rapid_change_threshold"]:
                                validation_results["anomalies_detected"] = True
                                validation_results["anomalies_details"].append(
                                    f"Нестабильность костной структуры {current_date}: {metric_name} = {actual_value - predicted_value:.4f}"
                                )
                
                # Средняя девиация за период
                if relevant_metrics_count > 0:
                    avg_deviation_for_period = period_deviations_sum / relevant_metrics_count
                    validation_results["metric_deviations"][current_date] = deviations_for_period
            
            # ИСПРАВЛЕНО: Общая оценка согласованности
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
            
            logger.info(f"Валидация согласованности завершена: score={validation_results['overall_aging_consistency_score']:.3f}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Ошибка валидации согласованности старения: {e}")
            return {
                "overall_aging_consistency_score": 0.0,
                "anomalies_detected": True,
                "anomalies_details": [f"Ошибка валидации: {str(e)}"],
                "metric_deviations": {}
            }

    def _predict_metrics_for_age(self, age: float, baseline_metrics: Dict[str, float]) -> Dict[str, float]:
        """Предсказание метрик для заданного возраста на основе aging model"""
        try:
            predicted_metrics = {}
            
            for metric_name, baseline_value in baseline_metrics.items():
                predicted_value = baseline_value
                
                # ИСПРАВЛЕНО: Применение aging model согласно правкам
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
                
                # ИСПРАВЛЕНО: Фильтрация данных после 25 лет
                adult_mask = ages_array >= MEDICAL_VALIDATOR_THRESHOLDS["bone_growth_threshold_age"]
                
                if np.sum(adult_mask) < 2:
                    results[metric] = {
                        "stable": True,
                        "reason": "Недостаточно данных для взрослого возраста",
                        "cv": 0.0,
                        "adult_data_points": int(np.sum(adult_mask))
                    }
                    continue
                
                adult_values = values[adult_mask]
                
                # ИСПРАВЛЕНО: Расчет coefficient of variation
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
                        "reason": "Недостаточно данных для анализа"
                    }
                    continue
                
                # ИСПРАВЛЕНО: Линейная регрессия
                slope, intercept, r_value, p_value, std_err = stats.linregress(ages, values)
                
                # Проверка биологической достоверности
                expected_negative = slope <= MEDICAL_VALIDATOR_THRESHOLDS["soft_tissue_aging_slope_threshold"]
                biological_plausible = True
                reason = "Нормальный паттерн старения"
                
                # ИСПРАВЛЕНО: Валидация наклона старения
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
                                              medical_events: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
        """
        ИСПРАВЛЕНО: Исключение хирургических гипотез по временной линии
        Согласно правкам: анализ 6-месячных интервалов и корреляция с событиями
        """
        if not metrics_changes:
            logger.warning("Нет данных изменений метрик для анализа хирургических гипотез")
            return {}
        
        try:
            logger.info(f"Исключение хирургических гипотез для {len(metrics_changes)} периодов")
            
            results = {}
            
            for period, change in metrics_changes.items():
                interval = time_intervals.get(period, 0)
                
                # ИСПРАВЛЕНО: Проверка подходящего интервала для хирургии
                if interval < MEDICAL_VALIDATOR_THRESHOLDS["surgery_min_interval_days"]:
                    results[period] = {
                        "surgery_possible": False,
                        "reason": "Интервал слишком короткий для хирургического восстановления",
                        "interval_days": interval,
                        "event_found": False
                    }
                    continue
                elif interval > MEDICAL_VALIDATOR_THRESHOLDS["surgery_max_interval_days"]:
                    results[period] = {
                        "surgery_possible": False,
                        "reason": "Интервал слишком длинный для прямой связи с хирургией",
                        "interval_days": interval,
                        "event_found": False
                    }
                    continue
                
                # ИСПРАВЛЕНО: Поиск корреляции с медицинскими событиями
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
                
                for event_date_str, event_info in medical_events.items():
                    try:
                        event_date = datetime.strptime(event_date_str, "%Y-%m-%d")
                        
                        # Проверка попадания в окно корреляции
                        days_to_start = abs((event_date - period_start).days)
                        days_to_end = abs((event_date - period_end).days)
                        
                        if days_to_start <= correlation_window or days_to_end <= correlation_window:
                            event_found = True
                            correlated_events.append({
                                "date": event_date_str,
                                "type": event_info.get("type", "unknown"),
                                "description": event_info.get("description", ""),
                                "days_to_period": min(days_to_start, days_to_end)
                            })
                    except ValueError:
                        continue
                
                # ИСПРАВЛЕНО: Анализ величины изменений
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
                        "reason": "Нет данных скоростей"
                    }
                    continue
                
                # ИСПРАВЛЕНО: Проверка превышения физиологических лимитов
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
            
            # ИСПРАВЛЕНО: Статистики корреляций
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
            elif len(unstable_metrics) > 0 if bone_stability else False:
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

    def save_validation_cache(self, cache_file: str = "medical_validation_cache.pkl") -> None:
        """Сохранение кэша валидации"""
        try:
            cache_path = CACHE_DIR / cache_file
            CACHE_DIR.mkdir(exist_ok=True)
            
            cache_data = {
                "validation_cache": self.validation_cache,
                "health_events": self.health_events,
                "calibrated": self.calibrated
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"Кэш валидации сохранен: {cache_path}")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения кэша валидации: {e}")

    def load_validation_cache(self, cache_file: str = "medical_validation_cache.pkl") -> None:
        """Загрузка кэша валидации"""
        try:
            cache_path = CACHE_DIR / cache_file
            
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                
                self.validation_cache = cache_data.get("validation_cache", {})
                self.health_events = cache_data.get("health_events", {})
                self.calibrated = cache_data.get("calibrated", False)
                
                logger.info(f"Кэш валидации загружен: {cache_path}")
            else:
                logger.info("Файл кэша валидации не найден, используется пустой кэш")
                
        except Exception as e:
            logger.error(f"Ошибка загрузки кэша валидации: {e}")

    def self_test(self) -> None:
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
            
        except Exception as e:
            logger.error(f"Ошибка самотестирования: {e}")
        
        logger.info("=== Самотестирование завершено ===")

# ==================== ТОЧКА ВХОДА ====================

if __name__ == "__main__":
    validator = MedicalValidator()
    validator.self_test()