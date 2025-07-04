"""
TemporalAnalyzer - Анализатор временных паттернов и старения лиц
Версия: 2.0
Дата: 2025-06-15
ИСПРАВЛЕНО: Все критические ошибки согласно правкам
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta, date
from pathlib import Path
import json
import pickle
from scipy import stats
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import ruptures as rpt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('logs/temporalanalyzer.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Импорт конфигурации
try:
    from core_config import (
        AGING_MODEL, CRITICAL_THRESHOLDS, PUTIN_BIRTH_DATE,
        START_ANALYSIS_DATE, END_ANALYSIS_DATE, CACHE_DIR,
        get_chronological_analysis_parameters, ERROR_CODES
    )
    logger.info("Конфигурация успешно импортирована")
except ImportError as e:
    logger.error(f"Ошибка импорта конфигурации: {e}")
    # Значения по умолчанию
    AGING_MODEL = {
        "elasticity_loss_per_year": 0.015,
        "tissue_sagging_per_year": 1.51,
        "bone_stability_threshold": 25
    }
    CRITICAL_THRESHOLDS = {"temporal_stability_threshold": 0.8}
    PUTIN_BIRTH_DATE = date(1952, 10, 7)
    START_ANALYSIS_DATE = datetime(1999, 1, 1)
    END_ANALYSIS_DATE = datetime(2025, 12, 31)
    CACHE_DIR = Path("cache")
    ERROR_CODES = {"E006": "INVALID_DATE_FORMAT"}

# ==================== КОНСТАНТЫ ВРЕМЕННОГО АНАЛИЗА ====================

TEMPORAL_ANALYSIS_PARAMS = {
    "min_data_points": 5,
    "max_gap_days": 180,
    "seasonal_period": 365,  # дней в году
    "changepoint_penalty": 10,
    "aging_tolerance_per_year": 0.02,
    "systematic_pattern_threshold": 0.15,
    "temporal_stability_threshold": 0.8,
    "medical_event_window_days": 30,
    "appearance_frequency_threshold": 0.1,
    "absence_threshold_days": 90,
    "z_score_threshold": 2.5
}

# Медицинские события для корреляции
MEDICAL_EVENTS = {
    "2000-03-26": {"type": "election", "description": "Президентские выборы"},
    "2004-03-14": {"type": "election", "description": "Президентские выборы"},
    "2008-05-07": {"type": "transition", "description": "Передача полномочий"},
    "2012-05-07": {"type": "election", "description": "Президентские выборы"},
    "2018-05-7": {"type": "election", "description": "Президентские выборы"},
    "2020-03-01": {"type": "health", "description": "Пандемия COVID-19"},
    "2024-03-17": {"type": "election", "description": "Президентские выборы"}
}

# ==================== ОСНОВНОЙ КЛАСС ====================

class TemporalAnalyzer:
    """
    Анализатор временных паттернов с полной функциональностью
    ИСПРАВЛЕНО: Все критические ошибки согласно правкам
    """

    def __init__(self):
        """Инициализация анализатора временных паттернов"""
        logger.info("Инициализация TemporalAnalyzer")
        
        # Модель старения
        self.aging_model = AGING_MODEL.copy()
        
        # Параметры анализа
        self.analysis_params = TEMPORAL_ANALYSIS_PARAMS.copy()
        
        # Кэш результатов
        self.analysis_cache = {}
        
        # Медицинские события
        self.medical_events = self._load_medical_events()
        
        # Флаг калибровки
        self.calibrated = False
        
        logger.info("TemporalAnalyzer инициализирован")

    def _load_medical_events(self) -> Dict[str, Dict[str, str]]:
        """Загрузка медицинских событий"""
        try:
            # Преобразование строковых дат в datetime
            events = {}
            for date_str, event_data in MEDICAL_EVENTS.items():
                try:
                    event_date = datetime.strptime(date_str, "%Y-%m-%d")
                    events[event_date] = event_data
                except ValueError as e:
                    logger.warning(f"Неверный формат даты события: {date_str}, ошибка: {e}")
            
            logger.info(f"Загружено {len(events)} медицинских событий")
            return events
        except Exception as e:
            logger.error(f"Ошибка загрузки медицинских событий: {e}")
            return {}

    def build_medical_aging_model(self, chronological_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Построение медицинской модели старения
        Согласно правкам: правильные коэффициенты и валидация
        """
        if not chronological_data or len(chronological_data) < self.analysis_params["min_data_points"]:
            logger.warning("Недостаточно данных для построения модели старения")
            return self._get_default_aging_model()

        try:
            logger.info(f"Построение медицинской модели старения на {len(chronological_data)} точках")
            
            # Подготовка данных
            df = pd.DataFrame(chronological_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # Расчет возраста на каждую дату
            birth_date = pd.to_datetime(PUTIN_BIRTH_DATE)
            df['age'] = (df['date'] - birth_date).dt.days / 365.25
            
            # ИСПРАВЛЕНО: Извлечение метрик для модели старения
            aging_metrics = self._extract_aging_metrics(df)
            
            if not aging_metrics:
                logger.warning("Не удалось извлечь метрики для модели старения")
                return self._get_default_aging_model()
            
            # ИСПРАВЛЕНО: Построение линейной модели для каждой метрики
            aging_model = {
                "model_type": "linear_regression",
                "base_age": float(df['age'].min()),
                "metrics_models": {},
                "validation_scores": {},
                "bone_stability_age": self.aging_model["bone_stability_threshold"],
                "elasticity_loss_rate": self.aging_model["elasticity_loss_per_year"],
                "tissue_sagging_rate": self.aging_model["tissue_sagging_per_year"]
            }
            
            for metric_name, values in aging_metrics.items():
                if len(values) >= self.analysis_params["min_data_points"]:
                    model_result = self._fit_aging_metric_model(df['age'].values, values, metric_name)
                    aging_model["metrics_models"][metric_name] = model_result
            
            # ИСПРАВЛЕНО: Валидация модели
            validation_result = self._validate_aging_model(aging_model, df)
            aging_model["validation"] = validation_result
            
            logger.info(f"Модель старения построена для {len(aging_model['metrics_models'])} метрик")
            return aging_model
            
        except Exception as e:
            logger.error(f"Ошибка построения модели старения: {e}")
            return self._get_default_aging_model()

    def _extract_aging_metrics(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Извлечение метрик для анализа старения"""
        try:
            aging_metrics = {}
            
            # Список метрик для анализа старения
            metric_columns = [
                'authenticity_score', 'shape_error', 'asymmetry_score',
                'texture_entropy', 'embedding_confidence', 'temporal_consistency'
            ]
            
            for metric in metric_columns:
                if metric in df.columns:
                    values = df[metric].dropna().tolist()
                    if len(values) >= self.analysis_params["min_data_points"]:
                        aging_metrics[metric] = values
            
            return aging_metrics
        except Exception as e:
            logger.error(f"Ошибка извлечения метрик старения: {e}")
            return {}

    def _fit_aging_metric_model(self, ages: np.ndarray, values: List[float], metric_name: str) -> Dict[str, Any]:
        """Построение модели для конкретной метрики старения"""
        try:
            X = ages.reshape(-1, 1)
            y = np.array(values)
            
            # Линейная регрессия
            model = LinearRegression()
            model.fit(X, y)
            
            # Предсказания
            y_pred = model.predict(X)
            
            # Метрики качества
            r2 = r2_score(y, y_pred)
            slope = float(model.coef_[0])
            intercept = float(model.intercept_)
            
            # Статистическая значимость
            _, p_value = stats.pearsonr(ages, y)
            
            return {
                "slope": slope,
                "intercept": intercept,
                "r2_score": r2,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "aging_rate_per_year": slope,
                "baseline_value": intercept
            }
            
        except Exception as e:
            logger.error(f"Ошибка построения модели для {metric_name}: {e}")
            return {
                "slope": 0.0,
                "intercept": 0.0,
                "r2_score": 0.0,
                "p_value": 1.0,
                "significant": False,
                "aging_rate_per_year": 0.0,
                "baseline_value": 0.0
            }

    def _validate_aging_model(self, aging_model: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Валидация модели старения"""
        try:
            validation_results = {
                "bone_stability_validated": False,
                "physiological_limits_validated": False,
                "temporal_consistency_validated": False,
                "overall_valid": False
            }
            
            # ИСПРАВЛЕНО: Валидация стабильности костей после 25 лет
            bone_stability_age = aging_model["bone_stability_age"]
            stable_age_data = df[df['age'] >= bone_stability_age]
            
            if len(stable_age_data) >= 3:
                # Проверка стабильности костных метрик
                bone_metrics = ['interpupillary_distance_ratio', 'skull_width_ratio']
                bone_stability_scores = []
                
                for metric in bone_metrics:
                    if metric in stable_age_data.columns:
                        metric_values = stable_age_data[metric].dropna()
                        if len(metric_values) >= 3:
                            cv = np.std(metric_values) / (np.mean(metric_values) + 1e-8)
                            bone_stability_scores.append(cv < 0.05)  # CV < 5%
                
                validation_results["bone_stability_validated"] = np.mean(bone_stability_scores) > 0.5
            
            # ИСПРАВЛЕНО: Валидация физиологических лимитов
            physiological_valid = True
            for metric_name, model_data in aging_model["metrics_models"].items():
                aging_rate = abs(model_data["aging_rate_per_year"])
                if aging_rate > self.analysis_params["aging_tolerance_per_year"] * 10:  # Максимум 20% в год
                    physiological_valid = False
                    break
            
            validation_results["physiological_limits_validated"] = physiological_valid
            
            # Валидация временной согласованности
            temporal_consistency = len([m for m in aging_model["metrics_models"].values() if m["significant"]]) / max(1, len(aging_model["metrics_models"]))
            validation_results["temporal_consistency_validated"] = temporal_consistency >= 0.6
            
            # Общая валидация
            validation_results["overall_valid"] = all([
                validation_results["bone_stability_validated"],
                validation_results["physiological_limits_validated"],
                validation_results["temporal_consistency_validated"]
            ])
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Ошибка валидации модели старения: {e}")
            return {
                "bone_stability_validated": False,
                "physiological_limits_validated": False,
                "temporal_consistency_validated": False,
                "overall_valid": False
            }

    def _get_default_aging_model(self) -> Dict[str, Any]:
        """Получение модели старения по умолчанию"""
        return {
            "model_type": "default",
            "base_age": 47.0,  # Возраст в 1999 году
            "metrics_models": {},
            "validation_scores": {},
            "bone_stability_age": self.aging_model["bone_stability_threshold"],
            "elasticity_loss_rate": self.aging_model["elasticity_loss_per_year"],
            "tissue_sagging_rate": self.aging_model["tissue_sagging_per_year"],
            "validation": {
                "bone_stability_validated": False,
                "physiological_limits_validated": False,
                "temporal_consistency_validated": False,
                "overall_valid": False
            }
        }

    def predict_expected_metrics_for_age(self, target_age: float, aging_model: Dict[str, Any]) -> Dict[str, float]:
        """
        ИСПРАВЛЕНО: Предсказание ожидаемых метрик для возраста
        Согласно правкам: предсказание всех метрик на основе модели
        """
        try:
            logger.info(f"Предсказание метрик для возраста: {target_age}")
            predicted_metrics = {}
            
            if aging_model["model_type"] == "default":
                # Базовые предсказания
                predicted_metrics = {
                    "authenticity_score": 0.85 - (target_age - 47) * 0.005,
                    "shape_error": 0.15 + (target_age - 47) * 0.002,
                    "asymmetry_score": 0.05 + (target_age - 47) * 0.001,
                    "texture_entropy": 7.5 - (target_age - 47) * 0.01,
                    "embedding_confidence": 0.9 - (target_age - 47) * 0.003
                }
            else:
                # Предсказания на основе построенной модели
                for metric_name, model_data in aging_model["metrics_models"].items():
                    predicted_value = model_data["intercept"] + model_data["slope"] * target_age
                    predicted_metrics[metric_name] = float(predicted_value)
            
            # ИСПРАВЛЕНО: Применение физиологических ограничений
            predicted_metrics = self._apply_physiological_constraints(predicted_metrics, target_age)
            
            logger.info(f"Предсказано {len(predicted_metrics)} метрик")
            return predicted_metrics
            
        except Exception as e:
            logger.error(f"Ошибка предсказания метрик: {e}")
            return {}

    def _apply_physiological_constraints(self, metrics: Dict[str, float], age: float) -> Dict[str, float]:
        """Применение физиологических ограничений к предсказанным метрикам"""
        try:
            constrained_metrics = metrics.copy()
            
            # Ограничения для различных метрик
            constraints = {
                "authenticity_score": (0.0, 1.0),
                "shape_error": (0.0, 1.0),
                "asymmetry_score": (0.0, 0.5),
                "texture_entropy": (4.0, 8.0),
                "embedding_confidence": (0.0, 1.0)
            }
            
            for metric_name, value in constrained_metrics.items():
                if metric_name in constraints:
                    min_val, max_val = constraints[metric_name]
                    constrained_metrics[metric_name] = np.clip(value, min_val, max_val)
            
            return constrained_metrics
        except Exception as e:
            logger.error(f"Ошибка применения ограничений: {e}")
            return metrics

    def detect_temporal_anomalies_in_metrics(self, chronological_data: List[Dict[str, Any]],
                                           aging_model: Dict[str, Any]) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Обнаружение временных аномалий в метриках
        Согласно правкам: Z-score, rapid change, reverse aging
        """
        if not chronological_data:
            logger.warning("Нет данных для анализа временных аномалий")
            return {}

        try:
            logger.info(f"Анализ временных аномалий в {len(chronological_data)} точках")
            
            # Подготовка данных
            df = pd.DataFrame(chronological_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # Расчет возраста
            birth_date = pd.to_datetime(PUTIN_BIRTH_DATE)
            df['age'] = (df['date'] - birth_date).dt.days / 365.25
            
            anomalies = {
                "z_score_anomalies": [],
                "rapid_change_anomalies": [],
                "reverse_aging_anomalies": [],
                "inconsistent_pattern_anomalies": [],
                "total_anomalies": 0
            }
            
            # Анализ каждой метрики
            metric_columns = [col for col in df.columns if col not in ['date', 'age', 'filepath']]
            
            for metric in metric_columns:
                if metric in df.columns and df[metric].notna().sum() >= 3:
                    metric_anomalies = self._detect_metric_anomalies(df, metric, aging_model)
                    
                    # Объединение аномалий
                    for anomaly_type in anomalies.keys():
                        if anomaly_type in metric_anomalies:
                            anomalies[anomaly_type].extend(metric_anomalies[anomaly_type])
            
            # Подсчет общего количества аномалий
            anomalies["total_anomalies"] = sum(len(anomalies[key]) for key in anomalies if key != "total_anomalies")
            
            logger.info(f"Обнаружено {anomalies['total_anomalies']} временных аномалий")
            return anomalies
            
        except Exception as e:
            logger.error(f"Ошибка анализа временных аномалий: {e}")
            return {}

    def _detect_metric_anomalies(self, df: pd.DataFrame, metric: str, aging_model: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """Обнаружение аномалий для конкретной метрики"""
        try:
            anomalies = {
                "z_score_anomalies": [],
                "rapid_change_anomalies": [],
                "reverse_aging_anomalies": [],
                "inconsistent_pattern_anomalies": []
            }
            
            values = df[metric].dropna()
            if len(values) < 3:
                return anomalies
            
            # ИСПРАВЛЕНО: Z-score аномалии
            z_scores = np.abs(stats.zscore(values))
            z_threshold = TEMPORAL_ANALYSIS_PARAMS.get("z_score_threshold", 2.5)
            z_anomaly_indices = np.where(z_scores > z_threshold)[0]
            
            for idx in z_anomaly_indices:
                anomalies["z_score_anomalies"].append({
                    "date": df.iloc[values.index[idx]]['date'],
                    "metric": metric,
                    "value": float(values.iloc[idx]),
                    "z_score": float(z_scores[idx]),
                    "severity": "high" if z_scores[idx] > 3.0 else "medium"
                })
            
            # ИСПРАВЛЕНО: Rapid change аномалии
            if len(values) > 1:
                diffs = np.diff(values)
                diff_threshold = np.std(diffs) * 3  # 3 стандартных отклонения
                rapid_indices = np.where(np.abs(diffs) > diff_threshold)[0]
                
                for idx in rapid_indices:
                    anomalies["rapid_change_anomalies"].append({
                        "date": df.iloc[values.index[idx+1]]['date'],
                        "metric": metric,
                        "change": float(diffs[idx]),
                        "threshold": float(diff_threshold),
                        "severity": "high" if abs(diffs[idx]) > diff_threshold * 1.5 else "medium"
                    })
            
            # ИСПРАВЛЕНО: Reverse aging аномалии
            if metric in aging_model.get("metrics_models", {}):
                model_data = aging_model["metrics_models"][metric]
                expected_slope = model_data["slope"]
                
                if len(values) > 2:
                    # Локальные тренды
                    window_size = min(5, len(values) // 2)
                    for i in range(len(values) - window_size + 1):
                        window_values = values.iloc[i:i+window_size]
                        window_ages = df.iloc[values.index[i:i+window_size]]['age']
                        
                        if len(window_values) >= 3:
                            slope, _, _, p_value, _ = stats.linregress(window_ages, window_values)
                            
                            # Обратное старение: slope противоположен ожидаемому
                            if expected_slope != 0 and np.sign(slope) != np.sign(expected_slope) and p_value < 0.1:
                                anomalies["reverse_aging_anomalies"].append({
                                    "date_start": df.iloc[values.index[i]]['date'],
                                    "date_end": df.iloc[values.index[i+window_size-1]]['date'],
                                    "metric": metric,
                                    "observed_slope": float(slope),
                                    "expected_slope": float(expected_slope),
                                    "p_value": float(p_value)
                                })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Ошибка анализа аномалий для {metric}: {e}")
            return {
                "z_score_anomalies": [],
                "rapid_change_anomalies": [],
                "reverse_aging_anomalies": [],
                "inconsistent_pattern_anomalies": []
            }

    def analyze_identity_switching_patterns(self, identity_timeline: Dict[str, Any]) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Анализ паттернов смены идентичности
        Согласно правкам: систематические интервалы и паттерны
        """
        if not identity_timeline:
            logger.warning("Нет данных временной линии для анализа смены идентичности")
            return {}

        try:
            logger.info(f"Анализ паттернов смены идентичности для {len(identity_timeline)} идентичностей")
            
            switching_analysis = {
                "systematic_patterns_detected": False,
                "switching_intervals": [],
                "identity_transitions": [],
                "regularity_score": 0.0,
                "suspicious_patterns": []
            }
            
            # Сортировка идентичностей по времени первого появления
            sorted_identities = sorted(
                identity_timeline.items(),
                key=lambda x: x[1].get("first_appearance", datetime.min)
            )
            
            # ИСПРАВЛЕНО: Анализ переходов между идентичностями
            for i in range(len(sorted_identities) - 1):
                current_identity = sorted_identities[i]
                next_identity = sorted_identities[i + 1]
                
                current_end = current_identity[1].get("last_appearance")
                next_start = next_identity[1].get("first_appearance")
                
                if current_end and next_start:
                    gap_days = (next_start - current_end).days
                    transition = {
                        "from_identity": current_identity[0],
                        "to_identity": next_identity[0],
                        "transition_date": next_start,
                        "gap_days": gap_days,
                        "overlap": gap_days < 0  # Отрицательный gap означает перекрытие
                    }
                    
                    switching_analysis["identity_transitions"].append(transition)
                    switching_analysis["switching_intervals"].append(gap_days)
            
            # ИСПРАВЛЕНО: Анализ систематических паттернов
            if len(switching_analysis["switching_intervals"]) >= 3:
                intervals = np.array(switching_analysis["switching_intervals"])
                
                # Регулярность интервалов
                interval_std = np.std(intervals)
                interval_mean = np.mean(intervals)
                cv = interval_std / (interval_mean + 1e-8)  # Coefficient of variation
                
                switching_analysis["regularity_score"] = max(0.0, 1.0 - cv)
                
                # Систематические паттерны
                if cv < self.analysis_params["systematic_pattern_threshold"]:
                    switching_analysis["systematic_patterns_detected"] = True
                    switching_analysis["suspicious_patterns"].append({
                        "type": "regular_intervals",
                        "description": f"Регулярные интервалы смены: {interval_mean:.1f} ± {interval_std:.1f} дней",
                        "coefficient_variation": cv
                    })
            
            # ИСПРАВЛЕНО: Поиск подозрительных паттернов
            suspicious_patterns = self._detect_suspicious_switching_patterns(switching_analysis["identity_transitions"])
            switching_analysis["suspicious_patterns"].extend(suspicious_patterns)
            
            logger.info(f"Анализ смены идентичности завершен. Систематические паттерны: {switching_analysis['systematic_patterns_detected']}")
            return switching_analysis
            
        except Exception as e:
            logger.error(f"Ошибка анализа паттернов смены идентичности: {e}")
            return {}

    def _detect_suspicious_switching_patterns(self, transitions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Обнаружение подозрительных паттернов смены"""
        try:
            suspicious = []
            
            if not transitions:
                return suspicious
            
            # Анализ частых перекрытий
            overlaps = [t for t in transitions if t.get("overlap", False)]
            if len(overlaps) / len(transitions) > 0.3:  # Более 30% перекрытий
                suspicious.append({
                    "type": "frequent_overlaps",
                    "description": f"Частые перекрытия идентичностей: {len(overlaps)}/{len(transitions)}",
                    "severity": "high"
                })
            
            # Анализ очень коротких интервалов
            short_intervals = [t for t in transitions if 0 <= t.get("gap_days", 0) <= 7]
            if len(short_intervals) > 2:
                suspicious.append({
                    "type": "rapid_switching",
                    "description": f"Быстрая смена идентичностей: {len(short_intervals)} переходов ≤ 7 дней",
                    "severity": "medium"
                })
            
            # Анализ периодичности
            if len(transitions) >= 4:
                gaps = [t.get("gap_days", 0) for t in transitions]
                
                # Поиск периодических паттернов
                for period in [30, 60, 90, 180, 365]:  # Месяц, 2 месяца, квартал, полгода, год
                    periodic_matches = sum(1 for gap in gaps if abs(gap % period) < 7 or abs(gap % period) > period - 7)
                    if periodic_matches >= len(gaps) * 0.6:  # 60% совпадений
                        suspicious.append({
                            "type": "periodic_pattern",
                            "description": f"Периодический паттерн смены каждые ~{period} дней",
                            "period": period,
                            "matches": periodic_matches,
                            "severity": "high"
                        })
            
            return suspicious
            
        except Exception as e:
            logger.error(f"Ошибка обнаружения подозрительных паттернов: {e}")
            return []

    def build_identity_appearance_timeline(self, cluster_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Построение временной линии появлений идентичностей
        Согласно правкам: activity periods, absence intervals
        """
        if not cluster_results or "cluster_metadata" not in cluster_results:
            logger.warning("Нет данных кластеризации для построения временной линии")
            return {}

        try:
            logger.info("Построение временной линии появлений идентичностей")
            timeline = {}
            
            for cluster_id, metadata in cluster_results["cluster_metadata"].items():
                items = metadata.get("items", [])
                
                # Фильтрация элементов с датами
                dated_items = [(item['date'], item) for item in items if item.get('date')]
                if not dated_items:
                    continue
                
                dated_items.sort(key=lambda x: x[0])
                dates = [item[0] for item in dated_items]
                
                # ИСПРАВЛЕНО: Анализ периодов активности
                activity_periods = self._analyze_activity_periods(dates)
                
                # ИСПРАВЛЕНО: Анализ интервалов отсутствия
                absence_intervals = self._analyze_absence_intervals(dates)
                
                # Статистики появлений
                appearance_stats = self._calculate_appearance_statistics(dates)
                
                timeline[f"Identity_{cluster_id}"] = {
                    "cluster_id": cluster_id,
                    "total_appearances": len(dated_items),
                    "date_range": {
                        "start": dates[0],
                        "end": dates[-1],
                        "span_days": (dates[-1] - dates[0]).days
                    },
                    "activity_periods": activity_periods,
                    "absence_intervals": absence_intervals,
                    "appearance_statistics": appearance_stats,
                    "cluster_metadata": metadata
                }
            
            logger.info(f"Временная линия построена для {len(timeline)} идентичностей")
            return timeline
            
        except Exception as e:
            logger.error(f"Ошибка построения временной линии: {e}")
            return {}

    def _analyze_activity_periods(self, dates: List[datetime]) -> List[Dict[str, Any]]:
        """Анализ периодов активности"""
        try:
            if len(dates) < 2:
                return []
            
            activity_periods = []
            current_period_start = dates[0]
            last_date = dates[0]
            max_gap = timedelta(days=self.analysis_params["absence_threshold_days"])
            
            for date in dates[1:]:
                gap = date - last_date
                
                if gap > max_gap:
                    # Завершение текущего периода
                    activity_periods.append({
                        "start_date": current_period_start,
                        "end_date": last_date,
                        "duration_days": (last_date - current_period_start).days,
                        "appearances_count": len([d for d in dates if current_period_start <= d <= last_date])
                    })
                    
                    # Начало нового периода
                    current_period_start = date
                
                last_date = date
            
            # Добавление последнего периода
            activity_periods.append({
                "start_date": current_period_start,
                "end_date": last_date,
                "duration_days": (last_date - current_period_start).days,
                "appearances_count": len([d for d in dates if current_period_start <= d <= last_date])
            })
            
            return activity_periods
            
        except Exception as e:
            logger.error(f"Ошибка анализа периодов активности: {e}")
            return []

    def _analyze_absence_intervals(self, dates: List[datetime]) -> List[Dict[str, Any]]:
        """Анализ интервалов отсутствия"""
        try:
            if len(dates) < 2:
                return []
            
            absence_intervals = []
            absence_threshold = timedelta(days=self.analysis_params["absence_threshold_days"])
            
            for i in range(len(dates) - 1):
                gap = dates[i + 1] - dates[i]
                
                if gap > absence_threshold:
                    absence_intervals.append({
                        "start_date": dates[i],
                        "end_date": dates[i + 1],
                        "duration_days": gap.days,
                        "severity": "long" if gap.days > 180 else "medium" if gap.days > 90 else "short"
                    })
            
            return absence_intervals
            
        except Exception as e:
            logger.error(f"Ошибка анализа интервалов отсутствия: {e}")
            return []

    def _calculate_appearance_statistics(self, dates: List[datetime]) -> Dict[str, Any]:
        """Расчет статистик появлений"""
        try:
            if not dates:
                return {}
            
            # Интервалы между появлениями
            intervals = [(dates[i+1] - dates[i]).days for i in range(len(dates) - 1)]
            
            stats_result = {
                "total_appearances": len(dates),
                "average_interval_days": np.mean(intervals) if intervals else 0,
                "median_interval_days": np.median(intervals) if intervals else 0,
                "std_interval_days": np.std(intervals) if intervals else 0,
                "min_interval_days": min(intervals) if intervals else 0,
                "max_interval_days": max(intervals) if intervals else 0,
                "regularity_coefficient": 1.0 - (np.std(intervals) / (np.mean(intervals) + 1e-8)) if intervals else 0
            }
            
            # Частота появлений по месяцам
            monthly_counts = {}
            for date in dates:
                month_key = date.strftime("%Y-%m")
                monthly_counts[month_key] = monthly_counts.get(month_key, 0) + 1
            
            stats_result["monthly_distribution"] = monthly_counts
            stats_result["active_months"] = len(monthly_counts)
            
            return stats_result
            
        except Exception as e:
            logger.error(f"Ошибка расчета статистик появлений: {e}")
            return {}

    def perform_seasonal_decomposition(self, chronological_data: List[Dict[str, Any]],
                                     metric_name: str = "authenticity_score") -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Сезонная декомпозиция временных рядов
        Согласно правкам: seasonal decomposition для выявления паттернов
        """
        if not chronological_data or len(chronological_data) < 24:  # Минимум 2 года данных
            logger.warning("Недостаточно данных для сезонной декомпозиции")
            return {}

        try:
            logger.info(f"Сезонная декомпозиция для метрики: {metric_name}")
            
            # Подготовка данных
            df = pd.DataFrame(chronological_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            if metric_name not in df.columns:
                logger.warning(f"Метрика {metric_name} не найдена в данных")
                return {}
            
            # Создание временного ряда с регулярными интервалами
            df_resampled = df.set_index('date').resample('M')[metric_name].mean().dropna()
            
            if len(df_resampled) < 24:
                logger.warning("Недостаточно месячных данных для декомпозиции")
                return {}
            
            # ИСПРАВЛЕНО: Сезонная декомпозиция
            decomposition = seasonal_decompose(
                df_resampled,
                model='additive',
                period=12,  # Годовая сезонность
                extrapolate_trend='freq'
            )
            
            # Извлечение компонентов
            seasonal_analysis = {
                "metric_name": metric_name,
                "data_points": len(df_resampled),
                "date_range": {
                    "start": df_resampled.index.min(),
                    "end": df_resampled.index.max()
                },
                "components": {
                    "trend": decomposition.trend.dropna().tolist(),
                    "seasonal": decomposition.seasonal.dropna().tolist(),
                    "residual": decomposition.resid.dropna().tolist(),
                    "observed": decomposition.observed.tolist()
                },
                "seasonal_strength": float(1 - np.var(decomposition.resid.dropna()) / np.var(decomposition.observed.dropna())),
                "trend_strength": float(1 - np.var(decomposition.resid.dropna()) / np.var(decomposition.trend.dropna() + decomposition.resid.dropna()))
            }
            
            # ИСПРАВЛЕНО: Анализ сезонных паттернов
            seasonal_patterns = self._analyze_seasonal_patterns(decomposition.seasonal)
            seasonal_analysis["seasonal_patterns"] = seasonal_patterns
            
            # Анализ тренда
            trend_analysis = self._analyze_trend_component(decomposition.trend.dropna())
            seasonal_analysis["trend_analysis"] = trend_analysis
            
            logger.info(f"Сезонная декомпозиция завершена. Сезонная сила: {seasonal_analysis['seasonal_strength']:.3f}")
            return seasonal_analysis
            
        except Exception as e:
            logger.error(f"Ошибка сезонной декомпозиции: {e}")
            return {}

    def _analyze_seasonal_patterns(self, seasonal_component: pd.Series) -> Dict[str, Any]:
        """Анализ сезонных паттернов"""
        try:
            # Группировка по месяцам
            monthly_seasonal = seasonal_component.groupby(seasonal_component.index.month).mean()
            
            patterns = {
                "monthly_effects": monthly_seasonal.to_dict(),
                "peak_month": int(monthly_seasonal.idxmax()),
                "trough_month": int(monthly_seasonal.idxmin()),
                "seasonal_amplitude": float(monthly_seasonal.max() - monthly_seasonal.min()),
                "significant_months": []
            }
            
            # Определение значимых месяцев
            threshold = np.std(monthly_seasonal) * 1.5
            for month, effect in monthly_seasonal.items():
                if abs(effect) > threshold:
                    patterns["significant_months"].append({
                        "month": month,
                        "effect": float(effect),
                        "direction": "positive" if effect > 0 else "negative"
                    })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Ошибка анализа сезонных паттернов: {e}")
            return {}

    def _analyze_trend_component(self, trend_component: pd.Series) -> Dict[str, Any]:
        """Анализ трендовой компоненты"""
        try:
            # Линейная регрессия тренда
            x = np.arange(len(trend_component))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, trend_component.values)
            
            trend_analysis = {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_value ** 2),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
                "strength": "strong" if abs(r_value) > 0.7 else "moderate" if abs(r_value) > 0.3 else "weak"
            }
            
            return trend_analysis
            
        except Exception as e:
            logger.error(f"Ошибка анализа тренда: {e}")
            return {}

    def detect_changepoints(self, chronological_data: List[Dict[str, Any]],
                          metric_name: str = "authenticity_score") -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Обнаружение точек изменения с ruptures
        Согласно правкам: changepoint detection для резких изменений
        """
        if not chronological_data or len(chronological_data) < 10:
            logger.warning("Недостаточно данных для обнаружения точек изменения")
            return {}

        try:
            logger.info(f"Обнаружение точек изменения для метрики: {metric_name}")
            
            # Подготовка данных
            df = pd.DataFrame(chronological_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            if metric_name not in df.columns:
                logger.warning(f"Метрика {metric_name} не найдена в данных")
                return {}
            
            # Извлечение временного ряда
            signal = df[metric_name].dropna().values
            dates = df.loc[df[metric_name].notna(), 'date'].values
            
            if len(signal) < 10:
                logger.warning("Недостаточно валидных данных для анализа")
                return {}
            
            # ИСПРАВЛЕНО: Обнаружение точек изменения с ruptures
            model = "rbf"  # Radial basis function model
            penalty = self.analysis_params["changepoint_penalty"]
            algo = rpt.Pelt(model=model).fit(signal)
            changepoints = algo.predict(pen=penalty)
            
            # Удаление последней точки (конец сигнала)
            if changepoints and changepoints[-1] == len(signal):
                changepoints = changepoints[:-1]
            
            changepoint_analysis = {
                "metric_name": metric_name,
                "total_changepoints": len(changepoints),
                "changepoints": [],
                "segments": []
            }
            
            # ИСПРАВЛЕНО: Анализ каждой точки изменения
            for i, cp_idx in enumerate(changepoints):
                if cp_idx < len(dates):
                    changepoint_date = pd.to_datetime(dates[cp_idx])
                    
                    # Анализ изменения
                    before_segment = signal[max(0, cp_idx-5):cp_idx]
                    after_segment = signal[cp_idx:min(len(signal), cp_idx+5)]
                    
                    if len(before_segment) > 0 and len(after_segment) > 0:
                        change_magnitude = np.mean(after_segment) - np.mean(before_segment)
                        
                        changepoint_info = {
                            "index": cp_idx,
                            "date": changepoint_date,
                            "change_magnitude": float(change_magnitude),
                            "direction": "increase" if change_magnitude > 0 else "decrease",
                            "significance": "high" if abs(change_magnitude) > np.std(signal) else "medium"
                        }
                        
                        # Корреляция с медицинскими событиями
                        medical_correlation = self._correlate_with_medical_events(changepoint_date)
                        if medical_correlation:
                            changepoint_info["medical_correlation"] = medical_correlation
                        
                        changepoint_analysis["changepoints"].append(changepoint_info)
            
            # Анализ сегментов между точками изменения
            segments = self._analyze_changepoint_segments(signal, changepoints, dates)
            changepoint_analysis["segments"] = segments
            
            logger.info(f"Обнаружено {len(changepoints)} точек изменения")
            return changepoint_analysis
            
        except Exception as e:
            logger.error(f"Ошибка обнаружения точек изменения: {e}")
            return {}

    def _correlate_with_medical_events(self, changepoint_date: datetime,
                                     window_days: int = 30) -> Optional[Dict[str, Any]]:
        """Корреляция точки изменения с медицинскими событиями"""
        try:
            window = timedelta(days=window_days)
            
            for event_date, event_info in self.medical_events.items():
                if abs((changepoint_date - event_date).days) <= window_days:
                    return {
                        "event_date": event_date,
                        "event_type": event_info["type"],
                        "event_description": event_info["description"],
                        "days_difference": (changepoint_date - event_date).days
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Ошибка корреляции с медицинскими событиями: {e}")
            return None

    def _analyze_changepoint_segments(self, signal: np.ndarray, changepoints: List[int],
                                    dates: np.ndarray) -> List[Dict[str, Any]]:
        """Анализ сегментов между точками изменения"""
        try:
            segments = []
            start_idx = 0
            
            for cp_idx in changepoints + [len(signal)]:
                if cp_idx > start_idx:
                    segment_signal = signal[start_idx:cp_idx]
                    segment_dates = dates[start_idx:cp_idx]
                    
                    if len(segment_signal) > 1:
                        # Статистики сегмента
                        segment_stats = {
                            "start_index": start_idx,
                            "end_index": cp_idx - 1,
                            "start_date": pd.to_datetime(segment_dates[0]),
                            "end_date": pd.to_datetime(segment_dates[-1]),
                            "duration_days": (pd.to_datetime(segment_dates[-1]) - pd.to_datetime(segment_dates[0])).days,
                            "mean_value": float(np.mean(segment_signal)),
                            "std_value": float(np.std(segment_signal)),
                            "trend_slope": 0.0,
                            "stability": "stable"
                        }
                        
                        # Анализ тренда в сегменте
                        if len(segment_signal) >= 3:
                            x = np.arange(len(segment_signal))
                            slope, _, r_value, p_value, _ = stats.linregress(x, segment_signal)
                            segment_stats["trend_slope"] = float(slope)
                            segment_stats["trend_r_squared"] = float(r_value ** 2)
                            segment_stats["trend_significant"] = p_value < 0.05
                            
                            # Классификация стабильности
                            if abs(slope) < np.std(segment_signal) * 0.1:
                                segment_stats["stability"] = "stable"
                            elif slope > 0:
                                segment_stats["stability"] = "increasing"
                            else:
                                segment_stats["stability"] = "decreasing"
                        
                        segments.append(segment_stats)
                
                start_idx = cp_idx
            
            return segments
            
        except Exception as e:
            logger.error(f"Ошибка анализа сегментов: {e}")
            return []

    def correlate_anomalies_with_medical_events(self, anomalies: Dict[str, Any]) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Корреляция аномалий с медицинскими событиями
        Согласно правкам: временные окна и корреляция
        """
        if not anomalies or not self.medical_events:
            logger.warning("Нет данных для корреляции аномалий с медицинскими событиями")
            return {}

        try:
            logger.info("Корреляция аномалий с медицинскими событиями")
            
            correlations = {
                "total_correlations": 0,
                "correlated_anomalies": [],
                "event_correlations": {},
                "correlation_statistics": {}
            }
            
            window_days = self.analysis_params["medical_event_window_days"]
            
            # Анализ каждого типа аномалий
            for anomaly_type, anomaly_list in anomalies.items():
                if anomaly_type == "total_anomalies" or not isinstance(anomaly_list, list):
                    continue
                
                for anomaly in anomaly_list:
                    anomaly_date = anomaly.get("date")
                    if not anomaly_date:
                        continue
                    
                    # Поиск ближайших медицинских событий
                    correlated_events = []
                    for event_date, event_info in self.medical_events.items():
                        days_diff = abs((anomaly_date - event_date).days)
                        if days_diff <= window_days:
                            correlated_events.append({
                                "event_date": event_date,
                                "event_info": event_info,
                                "days_difference": (anomaly_date - event_date).days,
                                "absolute_difference": days_diff
                            })
                    
                    if correlated_events:
                        # Выбор ближайшего события
                        closest_event = min(correlated_events, key=lambda x: x["absolute_difference"])
                        
                        correlation_entry = {
                            "anomaly_type": anomaly_type,
                            "anomaly_date": anomaly_date,
                            "anomaly_details": anomaly,
                            "correlated_event": closest_event,
                            "correlation_strength": self._calculate_correlation_strength(closest_event["absolute_difference"], window_days)
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
            
            if ratio <= 0.1:  # В пределах 10% окна
                return "very_strong"
            elif ratio <= 0.3:  # В пределах 30% окна
                return "strong"
            elif ratio <= 0.6:  # В пределах 60% окна
                return "moderate"
            else:
                return "weak"
                
        except Exception as e:
            logger.error(f"Ошибка расчета силы корреляции: {e}")
            return "unknown"

    def _calculate_correlation_statistics(self, correlations: Dict[str, Any]) -> Dict[str, Any]:
        """Расчет статистик корреляций"""
        try:
            stats = {
                "correlation_rate": 0.0,
                "average_time_difference": 0.0,
                "most_correlated_event_type": None,
                "correlation_strength_distribution": {}
            }
            
            if not correlations["correlated_anomalies"]:
                return stats
            
            # Распределение силы корреляций
            strength_counts = {}
            time_differences = []
            event_type_counts = {}
            
            for correlation in correlations["correlated_anomalies"]:
                strength = correlation["correlation_strength"]
                strength_counts[strength] = strength_counts.get(strength, 0) + 1
                
                time_diff = abs(correlation["correlated_event"]["days_difference"])
                time_differences.append(time_diff)
                
                event_type = correlation["correlated_event"]["event_info"]["type"]
                event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
            
            stats["correlation_strength_distribution"] = strength_counts
            stats["average_time_difference"] = float(np.mean(time_differences))
            
            if event_type_counts:
                stats["most_correlated_event_type"] = max(event_type_counts, key=event_type_counts.get)
            
            return stats
            
        except Exception as e:
            logger.error(f"Ошибка расчета статистик корреляций: {e}")
            return {}

    def save_analysis_cache(self, cache_file: str = "temporal_cache.pkl") -> None:
        """Сохранение кэша анализа"""
        try:
            cache_path = CACHE_DIR / cache_file
            CACHE_DIR.mkdir(exist_ok=True)
            
            with open(cache_path, 'wb') as f:
                pickle.dump(self.analysis_cache, f)
            
            logger.info(f"Кэш анализа сохранен: {cache_path}")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения кэша: {e}")

    def load_analysis_cache(self, cache_file: str = "temporal_cache.pkl") -> None:
        """Загрузка кэша анализа"""
        try:
            cache_path = CACHE_DIR / cache_file
            
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    self.analysis_cache = pickle.load(f)
                logger.info(f"Кэш анализа загружен: {cache_path}")
            else:
                logger.info("Файл кэша не найден, используется пустой кэш")
                
        except Exception as e:
            logger.error(f"Ошибка загрузки кэша: {e}")

    def calculate_putin_age_on_each_date(self, dates: List[datetime]) -> Dict[datetime, float]:
        """
        ИСПРАВЛЕНО: Расчет возраста Путина на каждую дату
        Согласно правкам: точный расчет от даты рождения 1952-10-07
        """
        try:
            logger.info(f"Расчет возраста для {len(dates)} дат")
            age_mapping = {}
            
            birth_date = datetime.combine(PUTIN_BIRTH_DATE, datetime.min.time())
            
            for date in dates:
                if isinstance(date, str):
                    # Парсинг строковых дат
                    try:
                        date = datetime.strptime(date, '%d_%m_%y')
                    except ValueError:
                        try:
                            date = datetime.strptime(date, '%Y-%m-%d')
                        except ValueError:
                            logger.warning(f"Неверный формат даты: {date}")
                            continue
                
                # Точный расчет возраста в годах
                age_delta = date - birth_date
                age_years = age_delta.days / 365.25  # Учет високосных годов
                
                age_mapping[date] = float(age_years)
            
            logger.info(f"Возраст рассчитан для {len(age_mapping)} дат")
            return age_mapping
            
        except Exception as e:
            logger.error(f"Ошибка расчета возраста: {e}")
            return {}

    def validate_aging_consistency_for_identity(self, identity_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Валидация консистентности старения для идентичности
        Согласно правкам: медицинская валидация старения
        """
        try:
            logger.info("Валидация консистентности старения")
            
            validation_result = {
                "is_consistent": True,
                "consistency_score": 1.0,
                "medical_plausibility": 1.0,
                "violations": [],
                "bone_structure_changes": [],
                "soft_tissue_changes": [],
                "temporal_gaps": [],
                "surgical_evidence": [],
                "aging_rate_analysis": {}
            }
            
            if not identity_data or "temporal_points" not in identity_data:
                validation_result["is_consistent"] = False
                validation_result["violations"].append("Недостаточно данных для валидации")
                return validation_result
            
            temporal_points = identity_data["temporal_points"]
            if len(temporal_points) < 2:
                validation_result["violations"].append("Менее 2 временных точек")
                return validation_result
            
            # Сортировка по возрасту
            sorted_points = sorted(temporal_points, key=lambda p: p.get("age", 0))
            
            # ИСПРАВЛЕНО: Анализ изменений костных структур
            bone_metrics = [
                "skull_width_ratio", "interpupillary_distance_ratio", 
                "zygomatic_arch_width", "mandible_width_ratio"
            ]
            
            for metric in bone_metrics:
                bone_changes = self._analyze_bone_metric_changes(sorted_points, metric)
                if bone_changes["significant_change"]:
                    validation_result["bone_structure_changes"].append(bone_changes)
                    validation_result["consistency_score"] *= 0.7
            
            # ИСПРАВЛЕНО: Анализ изменений мягких тканей
            soft_tissue_metrics = [
                "cheek_fullness_ratio", "nasolabial_depth", 
                "skin_texture_score", "facial_volume_ratio"
            ]
            
            for metric in soft_tissue_metrics:
                tissue_changes = self._analyze_soft_tissue_changes(sorted_points, metric)
                validation_result["soft_tissue_changes"].append(tissue_changes)
            
            # ИСПРАВЛЕНО: Анализ временных разрывов
            temporal_gaps = self._analyze_temporal_gaps(sorted_points)
            validation_result["temporal_gaps"] = temporal_gaps
            
            # ИСПРАВЛЕНО: Поиск признаков хирургического вмешательства
            surgical_evidence = self._detect_surgical_intervention_evidence(sorted_points)
            validation_result["surgical_evidence"] = surgical_evidence
            
            # ИСПРАВЛЕНО: Анализ скорости старения
            aging_analysis = self._analyze_aging_rates(sorted_points)
            validation_result["aging_rate_analysis"] = aging_analysis
            
            # Финальная оценка консистентности
            if len(validation_result["bone_structure_changes"]) > 2:
                validation_result["is_consistent"] = False
                validation_result["consistency_score"] *= 0.3
            
            if len(surgical_evidence) > 0:
                validation_result["medical_plausibility"] *= 0.5
            
            # Общая медицинская правдоподобность
            validation_result["medical_plausibility"] = min(
                validation_result["consistency_score"],
                validation_result["medical_plausibility"]
            )
            
            logger.info(f"Валидация завершена. Консистентность: {validation_result['consistency_score']:.3f}")
            return validation_result
            
        except Exception as e:
            logger.error(f"Ошибка валидации старения: {e}")
            return {
                "is_consistent": False,
                "consistency_score": 0.0,
                "medical_plausibility": 0.0,
                "error": str(e)
            }

    def _analyze_bone_metric_changes(self, temporal_points: List[Dict], metric_name: str) -> Dict[str, Any]:
        """Анализ изменений костных метрик"""
        try:
            values = []
            ages = []
            
            for point in temporal_points:
                if metric_name in point.get("metrics", {}):
                    values.append(point["metrics"][metric_name])
                    ages.append(point.get("age", 0))
            
            if len(values) < 2:
                return {
                    "metric_name": metric_name,
                    "significant_change": False,
                    "change_rate": 0.0,
                    "medical_explanation": "Недостаточно данных"
                }
            
            # Линейная регрессия для определения тренда
            slope, intercept, r_value, p_value, std_err = stats.linregress(ages, values)
            
            # Костные структуры должны быть стабильными после 25 лет
            max_allowed_change = 0.005  # 0.5% в год максимум
            significant_change = abs(slope) > max_allowed_change
            
            return {
                "metric_name": metric_name,
                "significant_change": significant_change,
                "change_rate": float(slope),
                "r_squared": float(r_value ** 2),
                "p_value": float(p_value),
                "medical_explanation": self._generate_bone_change_explanation(metric_name, slope, significant_change)
            }
            
        except Exception as e:
            logger.error(f"Ошибка анализа костной метрики {metric_name}: {e}")
            return {
                "metric_name": metric_name,
                "significant_change": False,
                "change_rate": 0.0,
                "medical_explanation": f"Ошибка анализа: {str(e)}"
            }

    def _analyze_soft_tissue_changes(self, temporal_points: List[Dict], metric_name: str) -> Dict[str, Any]:
        """Анализ изменений мягких тканей"""
        try:
            values = []
            ages = []
            
            for point in temporal_points:
                if metric_name in point.get("metrics", {}):
                    values.append(point["metrics"][metric_name])
                    ages.append(point.get("age", 0))
            
            if len(values) < 2:
                return {
                    "metric_name": metric_name,
                    "aging_rate": 0.0,
                    "expected_rate": 0.0,
                    "deviation": 0.0
                }
            
            # Анализ тренда
            slope, _, r_value, p_value, _ = stats.linregress(ages, values)
            
            # Ожидаемая скорость старения для мягких тканей
            expected_rates = {
                "cheek_fullness_ratio": -0.01,  # 1% потеря в год
                "nasolabial_depth": 0.02,       # 2% углубление в год
                "skin_texture_score": -0.015,   # 1.5% деградация в год
                "facial_volume_ratio": -0.008   # 0.8% потеря в год
            }
            
            expected_rate = expected_rates.get(metric_name, -0.01)
            deviation = abs(slope - expected_rate) / abs(expected_rate) if expected_rate != 0 else 0
            
            return {
                "metric_name": metric_name,
                "aging_rate": float(slope),
                "expected_rate": expected_rate,
                "deviation": float(deviation),
                "r_squared": float(r_value ** 2),
                "significant": p_value < 0.05
            }
            
        except Exception as e:
            logger.error(f"Ошибка анализа мягких тканей {metric_name}: {e}")
            return {
                "metric_name": metric_name,
                "aging_rate": 0.0,
                "expected_rate": 0.0,
                "deviation": 0.0
            }

    def _analyze_temporal_gaps(self, temporal_points: List[Dict]) -> List[Dict[str, Any]]:
        """Анализ временных разрывов"""
        try:
            gaps = []
            
            for i in range(len(temporal_points) - 1):
                current_date = temporal_points[i].get("date")
                next_date = temporal_points[i + 1].get("date")
                
                if current_date and next_date:
                    gap_days = (next_date - current_date).days
                    
                    if gap_days > self.analysis_params["max_gap_days"]:
                        gaps.append({
                            "start_date": current_date,
                            "end_date": next_date,
                            "gap_days": gap_days,
                            "severity": "critical" if gap_days > 365 else "moderate" if gap_days > 180 else "minor"
                        })
            
            return gaps
            
        except Exception as e:
            logger.error(f"Ошибка анализа временных разрывов: {e}")
            return []

    def _detect_surgical_intervention_evidence(self, temporal_points: List[Dict]) -> List[Dict[str, Any]]:
        """Детекция признаков хирургического вмешательства"""
        try:
            surgical_evidence = []
            
            for i in range(len(temporal_points) - 1):
                current_point = temporal_points[i]
                next_point = temporal_points[i + 1]
                
                time_diff = (next_point.get("date") - current_point.get("date")).days
                
                # Анализ быстрых изменений
                rapid_changes = []
                
                for metric in ["cheek_fullness_ratio", "nasolabial_depth", "facial_volume_ratio"]:
                    current_val = current_point.get("metrics", {}).get(metric, 0)
                    next_val = next_point.get("metrics", {}).get(metric, 0)
                    
                    if current_val != 0:
                        change_rate = abs(next_val - current_val) / current_val
                        
                        # Если изменение > 10% за короткий период
                        if change_rate > 0.1 and time_diff < 180:
                            rapid_changes.append({
                                "metric": metric,
                                "change_rate": change_rate,
                                "time_days": time_diff
                            })
                
                if len(rapid_changes) >= 2:  # Множественные быстрые изменения
                    surgical_evidence.append({
                        "date_range": (current_point.get("date"), next_point.get("date")),
                        "evidence_type": "rapid_multiple_changes",
                        "changes": rapid_changes,
                        "confidence": 0.8 if len(rapid_changes) >= 3 else 0.6
                    })
            
            return surgical_evidence
            
        except Exception as e:
            logger.error(f"Ошибка детекции хирургического вмешательства: {e}")
            return []

    def _analyze_aging_rates(self, temporal_points: List[Dict]) -> Dict[str, Any]:
        """Анализ скорости старения"""
        try:
            aging_analysis = {
                "overall_aging_rate": 0.0,
                "accelerated_periods": [],
                "decelerated_periods": [],
                "medical_consistency": True
            }
            
            if len(temporal_points) < 3:
                return aging_analysis
            
            # Анализ общей скорости старения
            ages = [p.get("age", 0) for p in temporal_points]
            overall_scores = []
            
            for point in temporal_points:
                metrics = point.get("metrics", {})
                # Комбинированный балл старения
                aging_score = (
                    metrics.get("skin_texture_score", 0.5) * 0.4 +
                    metrics.get("facial_volume_ratio", 0.5) * 0.3 +
                    metrics.get("cheek_fullness_ratio", 0.5) * 0.3
                )
                overall_scores.append(aging_score)
            
            if len(overall_scores) >= 3:
                slope, _, r_value, p_value, _ = stats.linregress(ages, overall_scores)
                aging_analysis["overall_aging_rate"] = float(slope)
                aging_analysis["r_squared"] = float(r_value ** 2)
                aging_analysis["significant"] = p_value < 0.05
                
                # Ожидаемая скорость старения: -1.5% в год после 40 лет
                expected_rate = -0.015
                deviation = abs(slope - expected_rate) / abs(expected_rate)
                
                aging_analysis["medical_consistency"] = deviation < 2.0  # В пределах 200% от нормы
            
            return aging_analysis
            
        except Exception as e:
            logger.error(f"Ошибка анализа скорости старения: {e}")
            return {
                "overall_aging_rate": 0.0,
                "accelerated_periods": [],
                "decelerated_periods": [],
                "medical_consistency": False
            }

    def _generate_bone_change_explanation(self, metric_name: str, change_rate: float, significant: bool) -> str:
        """Генерация медицинского объяснения изменений костей"""
        try:
            explanations = {
                "skull_width_ratio": "ширина черепа",
                "interpupillary_distance_ratio": "межзрачковое расстояние",
                "zygomatic_arch_width": "ширина скуловых дуг",
                "mandible_width_ratio": "ширина нижней челюсти"
            }
            
            metric_desc = explanations.get(metric_name, metric_name)
            
            if not significant:
                return f"Изменения {metric_desc} в пределах нормы ({change_rate*100:.2f}%/год)"
            
            if change_rate > 0:
                direction = "увеличение"
            else:
                direction = "уменьшение"
            
            return f"Значительное {direction} {metric_desc} ({change_rate*100:.2f}%/год), что медицински маловероятно для костных структур после 25 лет"
            
        except Exception as e:
            logger.error(f"Ошибка генерации объяснения: {e}")
            return f"Изменение {metric_name}: {change_rate*100:.2f}%/год"

    def calibrate_temporal_analysis_thresholds(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Калибровка порогов временного анализа
        Согласно правкам: автокалибровка на исторических данных
        """
        try:
            logger.info(f"Калибровка порогов на {len(historical_data)} образцах")
            
            calibration_result = {
                "calibrated_thresholds": {},
                "statistical_baselines": {},
                "confidence_intervals": {},
                "calibration_quality": 0.0
            }
            
            if len(historical_data) < 50:
                logger.warning("Недостаточно данных для надежной калибровки")
                return calibration_result
            
            # Подготовка данных
            df = pd.DataFrame(historical_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Калибровка порогов Z-score
            metrics_for_calibration = [
                'authenticity_score', 'shape_error', 'asymmetry_score',
                'texture_entropy', 'embedding_confidence'
            ]
            
            for metric in metrics_for_calibration:
                if metric in df.columns:
                    values = df[metric].dropna()
                    
                    if len(values) >= 30:
                        # Статистические характеристики
                        mean_val = float(np.mean(values))
                        std_val = float(np.std(values))
                        
                        # Калибровка порога аномалий (95-й процентиль)
                        anomaly_threshold = float(np.percentile(np.abs(stats.zscore(values)), 95))
                        
                        calibration_result["calibrated_thresholds"][metric] = {
                            "z_score_threshold": anomaly_threshold,
                            "rapid_change_threshold": std_val * 3,
                            "baseline_mean": mean_val,
                            "baseline_std": std_val
                        }
                        
                        # Доверительные интервалы
                        ci_lower = float(np.percentile(values, 2.5))
                        ci_upper = float(np.percentile(values, 97.5))
                        
                        calibration_result["confidence_intervals"][metric] = {
                            "lower_95": ci_lower,
                            "upper_95": ci_upper,
                            "range": ci_upper - ci_lower
                        }
            
            # Калибровка временных параметров
            if 'date' in df.columns and len(df) > 10:
                # Анализ типичных интервалов между снимками
                df_sorted = df.sort_values('date')
                intervals = []
                
                for i in range(len(df_sorted) - 1):
                    interval = (df_sorted.iloc[i+1]['date'] - df_sorted.iloc[i]['date']).days
                    intervals.append(interval)
                
                if intervals:
                    median_interval = float(np.median(intervals))
                    max_gap_calibrated = float(np.percentile(intervals, 90))  # 90-й процентиль
                    
                    calibration_result["calibrated_thresholds"]["temporal"] = {
                        "median_interval_days": median_interval,
                        "max_gap_days": max_gap_calibrated,
                        "absence_threshold_days": max_gap_calibrated * 1.5
                    }
            
            # Оценка качества калибровки
            quality_scores = []
            for metric_data in calibration_result["calibrated_thresholds"].values():
                if isinstance(metric_data, dict) and "baseline_std" in metric_data:
                    # Качество на основе стабильности стандартного отклонения
                    quality = 1.0 / (1.0 + metric_data["baseline_std"])
                    quality_scores.append(quality)
            
            calibration_result["calibration_quality"] = float(np.mean(quality_scores)) if quality_scores else 0.0
            
            # Обновление параметров анализа
            self._update_analysis_params_from_calibration(calibration_result["calibrated_thresholds"])
            
            logger.info(f"Калибровка завершена. Качество: {calibration_result['calibration_quality']:.3f}")
            return calibration_result
            
        except Exception as e:
            logger.error(f"Ошибка калибровки порогов: {e}")
            return {
                "calibrated_thresholds": {},
                "statistical_baselines": {},
                "confidence_intervals": {},
                "calibration_quality": 0.0,
                "error": str(e)
            }

    def _update_analysis_params_from_calibration(self, calibrated_thresholds: Dict[str, Any]) -> None:
        """Обновление параметров анализа из калибровки"""
        try:
            # Обновление Z-score порогов
            z_scores = []
            for metric_data in calibrated_thresholds.values():
                if isinstance(metric_data, dict) and "z_score_threshold" in metric_data:
                    z_scores.append(metric_data["z_score_threshold"])
            
            if z_scores:
                self.analysis_params["z_score_threshold"] = float(np.mean(z_scores))
            
            # Обновление временных параметров
            if "temporal" in calibrated_thresholds:
                temporal_params = calibrated_thresholds["temporal"]
                
                if "max_gap_days" in temporal_params:
                    self.analysis_params["max_gap_days"] = int(temporal_params["max_gap_days"])
                
                if "absence_threshold_days" in temporal_params:
                    self.analysis_params["absence_threshold_days"] = int(temporal_params["absence_threshold_days"])
            
            logger.info("Параметры анализа обновлены из калибровки")
            
        except Exception as e:
            logger.error(f"Ошибка обновления параметров: {e}")

    def export_temporal_analysis_report(self, analysis_results: Dict[str, Any], 
                                    output_format: str = "json") -> str:
        """
        ИСПРАВЛЕНО: Экспорт отчета временного анализа
        Согласно правкам: поддержка множественных форматов
        """
        try:
            logger.info(f"Экспорт отчета в формате {output_format}")
            
            # Подготовка данных отчета
            report_data = {
                "metadata": {
                    "analysis_date": datetime.now().isoformat(),
                    "analyzer_version": "2.0",
                    "putin_birth_date": PUTIN_BIRTH_DATE.isoformat(),
                    "analysis_parameters": self.analysis_params
                },
                "results": analysis_results,
                "summary": self._generate_analysis_summary(analysis_results)
            }
            
            # Определение пути выходного файла
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if output_format.lower() == "json":
                output_path = CACHE_DIR / f"temporal_analysis_report_{timestamp}.json"
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
                    
            elif output_format.lower() == "csv":
                output_path = CACHE_DIR / f"temporal_analysis_report_{timestamp}.csv"
                
                # Преобразование в табличный формат
                df = self._convert_results_to_dataframe(analysis_results)
                df.to_csv(output_path, index=False, encoding='utf-8')
                
            elif output_format.lower() == "html":
                output_path = CACHE_DIR / f"temporal_analysis_report_{timestamp}.html"
                
                html_content = self._generate_html_report(report_data)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                    
            else:
                raise ValueError(f"Неподдерживаемый формат: {output_format}")
            
            logger.info(f"Отчет экспортирован: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Ошибка экспорта отчета: {e}")
            return ""

    def _generate_analysis_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация сводки анализа"""
        try:
            summary = {
                "total_anomalies": 0,
                "critical_findings": [],
                "identity_switches": 0,
                "medical_inconsistencies": 0,
                "temporal_coverage": {},
                "confidence_score": 0.0
            }
            
            # Подсчет аномалий
            if "anomalies" in analysis_results:
                summary["total_anomalies"] = analysis_results["anomalies"].get("total_anomalies", 0)
            
            # Анализ переключений идентичности
            if "identity_switching" in analysis_results:
                switching_data = analysis_results["identity_switching"]
                summary["identity_switches"] = len(switching_data.get("identity_transitions", []))
                
                if switching_data.get("systematic_patterns_detected", False):
                    summary["critical_findings"].append("Систематические паттерны смены идентичности")
            
            # Медицинские несоответствия
            if "aging_validation" in analysis_results:
                validation = analysis_results["aging_validation"]
                if not validation.get("is_consistent", True):
                    summary["medical_inconsistencies"] += 1
                    summary["critical_findings"].append("Медицински неправдоподобное старение")
            
            # Временное покрытие
            if "timeline" in analysis_results:
                timeline = analysis_results["timeline"]
                dates = []
                
                for identity_data in timeline.values():
                    date_range = identity_data.get("date_range", {})
                    if "start" in date_range and "end" in date_range:
                        dates.extend([date_range["start"], date_range["end"]])
                
                if dates:
                    summary["temporal_coverage"] = {
                        "start_date": min(dates).isoformat() if dates else None,
                        "end_date": max(dates).isoformat() if dates else None,
                        "total_span_days": (max(dates) - min(dates)).days if len(dates) >= 2 else 0
                    }
            
            # Общий балл достоверности
            confidence_factors = []
            
            if summary["total_anomalies"] == 0:
                confidence_factors.append(1.0)
            else:
                confidence_factors.append(max(0.0, 1.0 - summary["total_anomalies"] / 100.0))
            
            if summary["medical_inconsistencies"] == 0:
                confidence_factors.append(1.0)
            else:
                confidence_factors.append(0.3)
            
            if summary["identity_switches"] <= 2:
                confidence_factors.append(1.0)
            else:
                confidence_factors.append(max(0.0, 1.0 - (summary["identity_switches"] - 2) / 10.0))
            
            summary["confidence_score"] = float(np.mean(confidence_factors)) if confidence_factors else 0.0
            
            return summary
            
        except Exception as e:
            logger.error(f"Ошибка генерации сводки: {e}")
            return {
                "total_anomalies": 0,
                "critical_findings": ["Ошибка анализа"],
                "identity_switches": 0,
                "medical_inconsistencies": 0,
                "temporal_coverage": {},
                "confidence_score": 0.0
            }

    def _convert_results_to_dataframe(self, analysis_results: Dict[str, Any]) -> pd.DataFrame:
        """Преобразование результатов в DataFrame для CSV экспорта"""
        try:
            rows = []
            
            # Извлечение аномалий
            if "anomalies" in analysis_results:
                anomalies = analysis_results["anomalies"]
                
                for anomaly_type, anomaly_list in anomalies.items():
                    if isinstance(anomaly_list, list):
                        for anomaly in anomaly_list:
                            row = {
                                "type": "anomaly",
                                "subtype": anomaly_type,
                                "date": anomaly.get("date", ""),
                                "metric": anomaly.get("metric", ""),
                                "value": anomaly.get("value", ""),
                                "severity": anomaly.get("severity", ""),
                                "description": str(anomaly)
                            }
                            rows.append(row)
            
            # Извлечение переключений идентичности
            if "identity_switching" in analysis_results:
                transitions = analysis_results["identity_switching"].get("identity_transitions", [])
                
                for transition in transitions:
                    row = {
                        "type": "identity_switch",
                        "subtype": "transition",
                        "date": transition.get("transition_date", ""),
                        "metric": "identity_change",
                        "value": f"{transition.get('from_identity', '')} -> {transition.get('to_identity', '')}",
                        "severity": "high" if transition.get("gap_days", 0) < 7 else "medium",
                        "description": f"Переход от {transition.get('from_identity', '')} к {transition.get('to_identity', '')} через {transition.get('gap_days', 0)} дней"
                    }
                    rows.append(row)
            
            if not rows:
                # Создание пустого DataFrame с нужными колонками
                return pd.DataFrame(columns=["type", "subtype", "date", "metric", "value", "severity", "description"])
            
            return pd.DataFrame(rows)
            
        except Exception as e:
            logger.error(f"Ошибка преобразования в DataFrame: {e}")
            return pd.DataFrame()

    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Генерация HTML отчета"""
        try:
            html_template = """
            <!DOCTYPE html>
            <html lang="ru">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Отчет временного анализа</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                    .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                    .critical { background-color: #ffebee; border-color: #f44336; }
                    .warning { background-color: #fff3e0; border-color: #ff9800; }
                    .normal { background-color: #e8f5e8; border-color: #4caf50; }
                    table { width: 100%; border-collapse: collapse; margin: 10px 0; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Отчет временного анализа</h1>
                    <p><strong>Дата анализа:</strong> {analysis_date}</p>
                    <p><strong>Версия анализатора:</strong> {analyzer_version}</p>
                </div>
                
                <div class="section {summary_class}">
                    <h2>Сводка</h2>
                    <p><strong>Общий балл достоверности:</strong> {confidence_score:.2f}</p>
                    <p><strong>Всего аномалий:</strong> {total_anomalies}</p>
                    <p><strong>Переключений идентичности:</strong> {identity_switches}</p>
                    <p><strong>Медицинских несоответствий:</strong> {medical_inconsistencies}</p>
                    {critical_findings_html}
                </div>
                
                {anomalies_section}
                {identity_section}
                {medical_section}
                
            </body>
            </html>
            """
            
            # Подготовка данных для шаблона
            metadata = report_data.get("metadata", {})
            summary = report_data.get("summary", {})
            
            # Определение класса CSS для сводки
            confidence = summary.get("confidence_score", 0.0)
            if confidence >= 0.7:
                summary_class = "normal"
            elif confidence >= 0.4:
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
                analysis_date=metadata.get("analysis_date", "Неизвестно"),
                analyzer_version=metadata.get("analyzer_version", "Неизвестно"),
                summary_class=summary_class,
                confidence_score=confidence,
                total_anomalies=summary.get("total_anomalies", 0),
                identity_switches=summary.get("identity_switches", 0),
                medical_inconsistencies=summary.get("medical_inconsistencies", 0),
                critical_findings_html=critical_findings_html,
                anomalies_section="<!-- Секция аномалий -->",
                identity_section="<!-- Секция идентичности -->",
                medical_section="<!-- Медицинская секция -->"
            )
            
            return html_content
            
        except Exception as e:
            logger.error(f"Ошибка генерации HTML отчета: {e}")
            return f"<html><body><h1>Ошибка генерации отчета</h1><p>{str(e)}</p></body></html>"

    def self_test(self) -> None:
        """Самотестирование модуля"""
        logger.info("=== Самотестирование TemporalAnalyzer ===")
        
        # Информация о параметрах
        logger.info(f"Aging model: {self.aging_model}")
        logger.info(f"Analysis params: {self.analysis_params}")
        logger.info(f"Medical events: {len(self.medical_events)}")
        
        # Тестовые данные
        test_dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='M')
        test_data = []
        
        for i, date in enumerate(test_dates):
            test_data.append({
                'date': date,
                'authenticity_score': 0.8 + 0.1 * np.sin(i * 0.5) + np.random.normal(0, 0.05),
                'shape_error': 0.2 + 0.05 * np.sin(i * 0.3) + np.random.normal(0, 0.02),
                'filepath': f'test_{i}.jpg'
            })
        
        try:
            # Тест модели старения
            aging_model = self.build_medical_aging_model(test_data)
            logger.info(f"Тест модели старения: {aging_model['model_type']}")
            
            # Тест предсказания метрик
            predicted = self.predict_expected_metrics_for_age(70.0, aging_model)
            logger.info(f"Тест предсказания: {len(predicted)} метрик")
            
            # Тест временных аномалий
            anomalies = self.detect_temporal_anomalies_in_metrics(test_data, aging_model)
            logger.info(f"Тест аномалий: {anomalies.get('total_anomalies', 0)} аномалий")
            
            # Тест сезонной декомпозиции
            seasonal = self.perform_seasonal_decomposition(test_data)
            logger.info(f"Тест сезонной декомпозиции: {len(seasonal)} результатов")
            
            # Тест точек изменения
            changepoints = self.detect_changepoints(test_data)
            logger.info(f"Тест точек изменения: {changepoints.get('total_changepoints', 0)} точек")
            
            # Тест корреляции с медицинскими событиями
            if anomalies:
                correlations = self.correlate_anomalies_with_medical_events(anomalies)
                logger.info(f"Тест корреляций: {correlations.get('total_correlations', 0)} корреляций")
            
            # Тест кластерных данных для timeline
            test_cluster_results = {
                "cluster_metadata": {
                    "0": {
                        "items": [
                            {"date": date, "filepath": f"test_{i}.jpg", "confidence": 0.9}
                            for i, date in enumerate(test_dates[:20])
                        ]
                    },
                    "1": {
                        "items": [
                            {"date": date, "filepath": f"test_{i}.jpg", "confidence": 0.8}
                            for i, date in enumerate(test_dates[20:])
                        ]
                    }
                }
            }
            
            # Тест построения временной линии
            timeline = self.build_identity_appearance_timeline(test_cluster_results)
            logger.info(f"Тест временной линии: {len(timeline)} идентичностей")
            
            # Тест анализа паттернов смены
            switching_patterns = self.analyze_identity_switching_patterns(timeline)
            logger.info(f"Тест паттернов смены: систематические={switching_patterns.get('systematic_patterns_detected', False)}")
            
            # Тест валидации старения
            test_identity_data = {
                "temporal_points": [
                    {
                        "date": test_dates[0],
                        "age": 67.5,
                        "metrics": {
                            "skull_width_ratio": 1.0,
                            "cheek_fullness_ratio": 0.8,
                            "skin_texture_score": 0.7
                        }
                    },
                    {
                        "date": test_dates[10],
                        "age": 68.0,
                        "metrics": {
                            "skull_width_ratio": 1.001,
                            "cheek_fullness_ratio": 0.78,
                            "skin_texture_score": 0.69
                        }
                    }
                ]
            }
            
            validation = self.validate_aging_consistency_for_identity(test_identity_data)
            logger.info(f"Тест валидации старения: консистентность={validation.get('is_consistent', False)}")
            
            # Тест калибровки
            calibration = self.calibrate_temporal_analysis_thresholds(test_data)
            logger.info(f"Тест калибровки: качество={calibration.get('calibration_quality', 0.0):.3f}")
            
            # Тест экспорта отчета
            test_results = {
                "anomalies": anomalies,
                "identity_switching": switching_patterns,
                "aging_validation": validation,
                "timeline": timeline
            }
            
            report_path = self.export_temporal_analysis_report(test_results, "json")
            logger.info(f"Тест экспорта отчета: {report_path}")
            
        except Exception as e:
            logger.error(f"Ошибка самотестирования: {e}")
        
        logger.info("=== Самотестирование завершено ===")

    # ==================== ТОЧКА ВХОДА ====================

    if __name__ == "__main__":
        analyzer = TemporalAnalyzer()
        analyzer.self_test()