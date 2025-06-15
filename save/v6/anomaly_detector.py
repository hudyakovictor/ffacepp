"""
AnomalyDetector - Детектор аномалий с байесовским анализом и каскадной верификацией
Версия: 2.0
Дата: 2025-06-15
Исправлены все критические ошибки согласно правкам
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
from scipy import stats
from scipy.spatial.distance import cosine
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_distances

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('logs/anomalydetector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Импорт конфигурации
try:
    from core_config import (
        AUTHENTICITY_WEIGHTS, MASK_DETECTION_LEVELS, CRITICAL_THRESHOLDS,
        BREAKTHROUGH_YEARS, AGING_MODEL, CACHE_DIR, ERROR_CODES,
        auto_calibrate_thresholds_historical_data
    )
    logger.info("Конфигурация успешно импортирована")
except ImportError as e:
    logger.error(f"Ошибка импорта конфигурации: {e}")
    # Значения по умолчанию
    AUTHENTICITY_WEIGHTS = {
        "geometry": 0.15, "embedding": 0.30, "texture": 0.10,
        "temporal_consistency": 0.15, "temporal_stability": 0.10,
        "aging_consistency": 0.10, "anomalies_score": 0.05, "liveness_score": 0.05
    }
    MASK_DETECTION_LEVELS = {}
    CRITICAL_THRESHOLDS = {"min_authenticity_score": 0.6}
    BREAKTHROUGH_YEARS = [2008, 2014, 2019, 2022]
    AGING_MODEL = {"elasticity_loss_per_year": 0.015}
    CACHE_DIR = Path("cache")
    ERROR_CODES = {"E005": "CLUSTERING_FAILED"}

# ==================== КОНСТАНТЫ ДЕТЕКЦИИ АНОМАЛИЙ ====================

# ИСПРАВЛЕНО: Пороги каскадной верификации согласно правкам
CASCADE_VERIFICATION_THRESHOLDS = {
    "geometry": 0.7,
    "embedding": 0.8,
    "texture": 0.6,
    "temporal": 0.7
}

# ИСПРАВЛЕНО: Продвинутые пороги аномалий
ANOMALY_DETECTION_ADVANCED_THRESHOLDS = {
    "mask_or_double_threshold": 0.3,
    "requires_analysis_threshold": 0.6,
    "surgery_min_interval_days": 60,
    "surgery_max_interval_days": 365,
    "mask_quality_jump_threshold": 0.15,
    "cross_source_critical_distance": 0.5,
    "swelling_metric_change_threshold": 0.1,
    "asymmetry_change_threshold": 0.05,
    "min_swelling_metrics_detected": 2,
    "surgery_swelling_weight": 0.3,
    "surgery_asymmetry_weight": 0.4,
    "surgery_healing_weight": 0.3,
    "surgery_short_interval_days": 90,
    "surgery_long_interval_days": 180,
    "surgery_short_interval_multiplier": 1.5,
    "surgery_long_interval_multiplier": 0.7,
    "healing_detection_interval_days": 30,
    "metric_stability_threshold": 0.7,
    "identity_stability_threshold": 0.8
}

# Байесовские априорные вероятности
BAYESIAN_PRIORS = {
    "same_person": 0.5,
    "different_person": 0.5
}

# Likelihood ratios для метрик
METRIC_BAYESIAN_PROPS = {
    "geometry_score": {
        "authentic_mean": 0.85,
        "non_authentic_mean": 0.25,
        "sensitivity": 7.0
    },
    "embedding_score": {
        "authentic_mean": 0.90,
        "non_authentic_mean": 0.10,
        "sensitivity": 9.0
    },
    "texture_score": {
        "authentic_mean": 0.75,
        "non_authentic_mean": 0.30,
        "sensitivity": 6.0
    },
    "temporal_consistency": {
        "authentic_mean": 0.80,
        "non_authentic_mean": 0.20,
        "sensitivity": 7.0
    }
}

# ==================== ОСНОВНОЙ КЛАСС ====================

class AnomalyDetector:
    """
    Детектор аномалий с полной функциональностью
    ИСПРАВЛЕНО: Все критические ошибки согласно правкам
    """
    
    def __init__(self):
        """Инициализация детектора аномалий"""
        logger.info("Инициализация AnomalyDetector")
        
        # Модели машинного обучения
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        
        # Байесовские априорные вероятности
        self.bayesian_priors = BAYESIAN_PRIORS.copy()
        
        # Кэш результатов
        self.analysis_cache = {}
        
        # Калиброванные пороги
        self.calibrated_thresholds = CRITICAL_THRESHOLDS.copy()
        
        # Флаг калибровки
        self.calibrated = False
        
        logger.info("AnomalyDetector инициализирован")

    def apply_bayesian_identity_analysis(self, evidence_dict: Dict[str, Dict], 
                                       prior_probabilities: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Байесовский анализ идентичности
        Согласно правкам: likelihood ratios для каждой метрики
        """
        if not evidence_dict:
            logger.warning("Пустые данные для байесовского анализа")
            return {}
        
        try:
            logger.info(f"Байесовский анализ для {len(evidence_dict)} идентичностей")
            
            # Использование переданных или стандартных априорных вероятностей
            priors = prior_probabilities if prior_probabilities else self.bayesian_priors
            
            bayesian_results = {}
            
            for identity_id, evidence in evidence_dict.items():
                logger.info(f"Анализ идентичности: {identity_id}")
                
                # ИСПРАВЛЕНО: Извлечение метрик из evidence
                geometry_evidence = evidence.get("geometry_score", 0.5)
                embedding_evidence = evidence.get("embedding_score", 0.5)
                texture_evidence = evidence.get("texture_score", 0.5)
                temporal_evidence = evidence.get("temporal_consistency", 0.5)
                
                # ИСПРАВЛЕНО: Расчет likelihood для одного человека
                likelihood_same = self._calculate_likelihood_same_person(
                    geometry_evidence, embedding_evidence, texture_evidence, temporal_evidence
                )
                
                # ИСПРАВЛЕНО: Расчет likelihood для разных людей
                likelihood_different = self._calculate_likelihood_different_person(
                    geometry_evidence, embedding_evidence, texture_evidence, temporal_evidence
                )
                
                # Байесовский расчет
                prior_same = priors.get("same_person", 0.5)
                prior_different = priors.get("different_person", 0.5)
                
                # Полная вероятность (evidence)
                evidence_total = likelihood_same * prior_same + likelihood_different * prior_different
                
                if evidence_total > 0:
                    # Апостериорные вероятности
                    posterior_same = (likelihood_same * prior_same) / evidence_total
                    posterior_different = (likelihood_different * prior_different) / evidence_total
                else:
                    posterior_same = 0.5
                    posterior_different = 0.5
                
                # Likelihood ratio
                likelihood_ratio = likelihood_same / likelihood_different if likelihood_different > 0 else float('inf')
                
                bayesian_results[identity_id] = {
                    "posterior_same_person": float(posterior_same),
                    "posterior_different_person": float(posterior_different),
                    "likelihood_ratio": float(likelihood_ratio),
                    "confidence": float(max(posterior_same, posterior_different)),
                    "decision": "same_person" if posterior_same > posterior_different else "different_person",
                    "evidence_strength": "strong" if likelihood_ratio > 10 or likelihood_ratio < 0.1 else "moderate"
                }
            
            logger.info(f"Байесовский анализ завершен для {len(bayesian_results)} идентичностей")
            return bayesian_results
            
        except Exception as e:
            logger.error(f"Ошибка байесовского анализа: {e}")
            return {}

    def _calculate_likelihood_same_person(self, geometry: float, embedding: float, 
                                        texture: float, temporal: float) -> float:
        """Расчет likelihood для одного человека"""
        try:
            # ИСПРАВЛЕНО: Комбинированный score с весами
            combined_score = (
                AUTHENTICITY_WEIGHTS["geometry"] * geometry +
                AUTHENTICITY_WEIGHTS["embedding"] * embedding +
                AUTHENTICITY_WEIGHTS["texture"] * texture +
                AUTHENTICITY_WEIGHTS["temporal_consistency"] * temporal
            )
            
            # Нормализация по сумме весов
            total_weight = (AUTHENTICITY_WEIGHTS["geometry"] + 
                          AUTHENTICITY_WEIGHTS["embedding"] + 
                          AUTHENTICITY_WEIGHTS["texture"] + 
                          AUTHENTICITY_WEIGHTS["temporal_consistency"])
            
            if total_weight > 0:
                combined_score /= total_weight
            
            # Сигмоидная функция для likelihood
            likelihood = 1 / (1 + np.exp(-5 * (combined_score - 0.7)))
            
            return float(likelihood)
            
        except Exception as e:
            logger.error(f"Ошибка расчета likelihood same person: {e}")
            return 0.5

    def _calculate_likelihood_different_person(self, geometry: float, embedding: float, 
                                             texture: float, temporal: float) -> float:
        """Расчет likelihood для разных людей"""
        try:
            # Комбинированный score
            combined_score = (
                AUTHENTICITY_WEIGHTS["geometry"] * geometry +
                AUTHENTICITY_WEIGHTS["embedding"] * embedding +
                AUTHENTICITY_WEIGHTS["texture"] * texture +
                AUTHENTICITY_WEIGHTS["temporal_consistency"] * temporal
            )
            
            # Нормализация
            total_weight = (AUTHENTICITY_WEIGHTS["geometry"] + 
                          AUTHENTICITY_WEIGHTS["embedding"] + 
                          AUTHENTICITY_WEIGHTS["texture"] + 
                          AUTHENTICITY_WEIGHTS["temporal_consistency"])
            
            if total_weight > 0:
                combined_score /= total_weight
            
            # Обратная сигмоидная функция
            likelihood = 1 / (1 + np.exp(-5 * (0.7 - combined_score)))
            
            return float(likelihood)
            
        except Exception as e:
            logger.error(f"Ошибка расчета likelihood different person: {e}")
            return 0.5

    def perform_cascade_verification(self, geometry_score: float, embedding_score: float, 
                                   texture_score: float, temporal_score: float) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Каскадная верификация
        Согласно правкам: несколько этапов верификации
        """
        try:
            logger.info("Выполнение каскадной верификации")
            
            # ИСПРАВЛЕНО: Проверка каждого уровня каскада
            geometry_passed = geometry_score >= CASCADE_VERIFICATION_THRESHOLDS["geometry"]
            embedding_passed = embedding_score >= CASCADE_VERIFICATION_THRESHOLDS["embedding"]
            texture_passed = texture_score >= CASCADE_VERIFICATION_THRESHOLDS["texture"]
            temporal_passed = temporal_score >= CASCADE_VERIFICATION_THRESHOLDS["temporal"]
            
            # Количество пройденных уровней
            levels_passed = sum([geometry_passed, embedding_passed, texture_passed, temporal_passed])
            
            # ИСПРАВЛЕНО: Финальная аутентичность с правильными весами
            final_authenticity = self.calculate_identity_authenticity_score(
                geometry_score, embedding_score, texture_score, temporal_score
            )
            
            # Интерпретация результата
            verification_result = self._interpret_authenticity_score(final_authenticity)
            
            # Confidence каскада
            cascade_confidence = levels_passed / 4.0
            
            result = {
                "geometry_passed": geometry_passed,
                "embedding_passed": embedding_passed,
                "texture_passed": texture_passed,
                "temporal_passed": temporal_passed,
                "levels_passed": levels_passed,
                "final_authenticity_score": final_authenticity,
                "verification_result": verification_result,
                "cascade_confidence": cascade_confidence,
                "all_levels_passed": levels_passed == 4
            }
            
            logger.info(f"Каскадная верификация: {levels_passed}/4 уровней пройдено")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка каскадной верификации: {e}")
            return {
                "geometry_passed": False,
                "embedding_passed": False,
                "texture_passed": False,
                "temporal_passed": False,
                "levels_passed": 0,
                "final_authenticity_score": 0.0,
                "verification_result": "error",
                "cascade_confidence": 0.0,
                "all_levels_passed": False
            }

    def calculate_identity_authenticity_score(self, geometry: float, embedding: float, 
                                            texture: float, temporal_consistency: float) -> float:
        """
        ИСПРАВЛЕНО: Расчет итогового authenticity score
        Согласно правкам: использование весов из coreconfig
        """
        try:
            logger.info("Расчет итогового authenticity score")
            
            # ИСПРАВЛЕНО: Использование весов из AUTHENTICITY_WEIGHTS
            authenticity = (
                AUTHENTICITY_WEIGHTS["geometry"] * geometry +
                AUTHENTICITY_WEIGHTS["embedding"] * embedding +
                AUTHENTICITY_WEIGHTS["texture"] * texture +
                AUTHENTICITY_WEIGHTS["temporal_consistency"] * temporal_consistency
            )
            
            # Нормализация по сумме весов
            total_weight = (AUTHENTICITY_WEIGHTS["geometry"] + 
                          AUTHENTICITY_WEIGHTS["embedding"] + 
                          AUTHENTICITY_WEIGHTS["texture"] + 
                          AUTHENTICITY_WEIGHTS["temporal_consistency"])
            
            if total_weight > 0:
                authenticity /= total_weight
            
            # Ограничение диапазона [0, 1]
            authenticity = np.clip(authenticity, 0.0, 1.0)
            
            logger.info(f"Итоговый authenticity score: {authenticity:.3f}")
            return float(authenticity)
            
        except Exception as e:
            logger.error(f"Ошибка расчета authenticity score: {e}")
            return 0.0

    def _interpret_authenticity_score(self, score: float) -> str:
        """Интерпретация authenticity score"""
        try:
            if score >= ANOMALY_DETECTION_ADVANCED_THRESHOLDS["requires_analysis_threshold"]:
                return "authentic_face"
            elif score >= ANOMALY_DETECTION_ADVANCED_THRESHOLDS["mask_or_double_threshold"]:
                return "requires_analysis"
            else:
                return "mask_or_double"
                
        except Exception as e:
            logger.error(f"Ошибка интерпретации score: {e}")
            return "unknown"

    def classify_mask_technology_level(self, texture_data: Dict[str, Any], 
                                     photo_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Классификация уровня технологии маски
        Согласно правкам: 5 уровней по годам с правильными параметрами
        """
        if not texture_data:
            logger.warning("Нет данных текстуры для классификации маски")
            return {"level": "Unknown", "confidence": 0.0, "reason": "Нет данных текстуры"}
        
        try:
            logger.info("Классификация уровня технологии маски")
            
            # Определение года фото
            if photo_date:
                photo_year = photo_date.year
            else:
                photo_year = 2020  # Значение по умолчанию
            
            # ИСПРАВЛЕНО: Извлечение агрегированных метрик
            avg_entropy = self._extract_average_metric(texture_data, "shannon_entropy", 6.5)
            avg_shape_error = self._extract_average_metric(texture_data, "shape_error", 0.3)
            avg_embedding_dist = self._extract_average_metric(texture_data, "embedding_distance", 0.5)
            
            # ИСПРАВЛЕНО: Классификация по уровням из MASK_DETECTION_LEVELS
            detected_level = None
            confidence = 0.0
            
            for level_name, level_data in MASK_DETECTION_LEVELS.items():
                year_start, year_end = level_data["years"]
                
                # Проверка временного диапазона
                if year_start <= photo_year <= year_end:
                    # Проверка метрик
                    entropy_match = avg_entropy <= level_data["entropy"]
                    shape_match = avg_shape_error >= level_data["shape_error"]
                    embedding_match = avg_embedding_dist >= level_data["embedding_dist"]
                    
                    # Расчет совпадений
                    matches = sum([entropy_match, shape_match, embedding_match])
                    match_confidence = matches / 3.0
                    
                    if match_confidence >= 0.6:  # Минимум 60% совпадений
                        detected_level = level_data
                        detected_level["level_name"] = level_name
                        confidence = match_confidence
                        break
            
            if detected_level:
                result = {
                    "level": detected_level["level_name"],
                    "confidence": float(confidence),
                    "reason": f"Энтропия: {avg_entropy:.2f}, Shape error: {avg_shape_error:.2f}",
                    "description": detected_level["description"],
                    "year_range": detected_level["years"],
                    "severity": self._calculate_mask_severity(detected_level, confidence)
                }
            else:
                result = {
                    "level": "Natural_Skin",
                    "confidence": 0.8,
                    "reason": f"Метрики соответствуют натуральной коже",
                    "description": "Натуральная кожа",
                    "year_range": None,
                    "severity": "none"
                }
            
            logger.info(f"Классификация маски: {result['level']}, confidence: {result['confidence']:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка классификации уровня маски: {e}")
            return {
                "level": "Error",
                "confidence": 0.0,
                "reason": f"Ошибка: {str(e)}",
                "description": "Ошибка анализа",
                "year_range": None,
                "severity": "unknown"
            }

    def _extract_average_metric(self, texture_data: Dict[str, Any], metric_name: str, default_value: float) -> float:
        """Извлечение среднего значения метрики из данных текстуры"""
        try:
            values = []
            
            # Поиск метрики во всех зонах
            for zone_name, zone_data in texture_data.items():
                if isinstance(zone_data, dict) and metric_name in zone_data:
                    values.append(zone_data[metric_name])
            
            return np.mean(values) if values else default_value
            
        except Exception as e:
            logger.error(f"Ошибка извлечения метрики {metric_name}: {e}")
            return default_value

    def _calculate_mask_severity(self, level_data: Dict[str, Any], confidence: float) -> str:
        """Расчет серьезности маски"""
        try:
            # Определение серьезности на основе уровня и confidence
            level_name = level_data.get("level_name", "")
            
            if "Level5" in level_name or "Advanced" in level_name:
                return "critical" if confidence > 0.8 else "high"
            elif "Level4" in level_name or "Professional" in level_name:
                return "high" if confidence > 0.7 else "medium"
            elif "Level3" in level_name or "Commercial" in level_name:
                return "medium" if confidence > 0.6 else "low"
            else:
                return "low"
                
        except Exception as e:
            logger.error(f"Ошибка расчета серьезности маски: {e}")
            return "unknown"

    def detect_surgical_intervention_evidence(self, metrics_sequence: List[Dict[str, Any]], 
                                            time_intervals: List[int]) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Обнаружение хирургических вмешательств
        Согласно правкам: анализ 2-4 месячных интервалов и 6-месячного восстановления
        """
        if not metrics_sequence or len(metrics_sequence) < 2:
            logger.warning("Недостаточно данных для анализа хирургических вмешательств")
            return {"surgical_evidence": False, "details": "Недостаточно данных"}
        
        try:
            logger.info(f"Анализ хирургических вмешательств в {len(metrics_sequence)} точках")
            
            surgical_evidence = {}
            
            for i in range(1, len(metrics_sequence)):
                current_metrics = metrics_sequence[i]
                previous_metrics = metrics_sequence[i-1]
                interval_days = time_intervals[i-1] if i-1 < len(time_intervals) else 90
                
                # ИСПРАВЛЕНО: Расчет изменений метрик
                changes = self._calculate_metric_changes(current_metrics, previous_metrics)
                
                # ИСПРАВЛЕНО: Обнаружение индикаторов отека
                swelling_indicators = self._detect_swelling_patterns(changes)
                
                # ИСПРАВЛЕНО: Обнаружение изменений асимметрии
                asymmetry_changes = self._detect_asymmetry_changes(changes)
                
                # ИСПРАВЛЕНО: Анализ динамики заживления
                healing_dynamics = self._analyze_healing_dynamics(changes, interval_days)
                
                # Проверка подходящего интервала для хирургии
                surgery_possible = (
                    ANOMALY_DETECTION_ADVANCED_THRESHOLDS["surgery_min_interval_days"] <= 
                    interval_days <= 
                    ANOMALY_DETECTION_ADVANCED_THRESHOLDS["surgery_max_interval_days"]
                )
                
                if any([swelling_indicators, asymmetry_changes]) and surgery_possible:
                    surgical_evidence[f"period_{i}"] = {
                        "swelling_detected": swelling_indicators,
                        "asymmetry_changes": asymmetry_changes,
                        "healing_dynamics": healing_dynamics,
                        "interval_sufficient": surgery_possible,
                        "surgery_likelihood": self._calculate_surgery_likelihood(
                            swelling_indicators, asymmetry_changes, healing_dynamics, interval_days
                        ),
                        "interval_days": interval_days
                    }
            
            # Общая оценка
            overall_evidence = len(surgical_evidence) > 0
            max_likelihood = max([evidence["surgery_likelihood"] for evidence in surgical_evidence.values()]) if surgical_evidence else 0.0
            
            result = {
                "surgical_evidence": overall_evidence,
                "max_likelihood": float(max_likelihood),
                "evidence_periods": surgical_evidence,
                "total_suspicious_periods": len(surgical_evidence),
                "recommendation": self._generate_surgical_recommendation(overall_evidence, max_likelihood)
            }
            
            logger.info(f"Анализ хирургии: evidence={overall_evidence}, likelihood={max_likelihood:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка анализа хирургических вмешательств: {e}")
            return {"surgical_evidence": False, "details": f"Ошибка: {str(e)}"}

    def _calculate_metric_changes(self, current: Dict[str, Any], previous: Dict[str, Any]) -> Dict[str, Any]:
        """Расчет изменений метрик"""
        try:
            changes = {}
            
            for metric_name in current.keys():
                if metric_name in previous:
                    current_val = current[metric_name]
                    previous_val = previous[metric_name]
                    
                    if isinstance(current_val, (int, float)) and isinstance(previous_val, (int, float)):
                        absolute_change = current_val - previous_val
                        relative_change = absolute_change / previous_val if previous_val != 0 else 0
                        
                        changes[metric_name] = {
                            "absolute_change": absolute_change,
                            "relative_change": relative_change,
                            "significant": abs(relative_change) > 0.1
                        }
            
            return changes
            
        except Exception as e:
            logger.error(f"Ошибка расчета изменений метрик: {e}")
            return {}

    def _detect_swelling_patterns(self, changes: Dict[str, Any]) -> bool:
        """Обнаружение паттернов отека"""
        try:
            # Метрики, указывающие на отек
            swelling_metrics = ["forehead_height_ratio", "nose_width_ratio", "mouth_width_ratio"]
            swelling_indicators = []
            
            for metric in swelling_metrics:
                if metric in changes:
                    relative_change = changes[metric].get("relative_change", 0)
                    if relative_change >= ANOMALY_DETECTION_ADVANCED_THRESHOLDS["swelling_metric_change_threshold"]:
                        swelling_indicators.append(True)
                    else:
                        swelling_indicators.append(False)
            
            # Требуется минимум 2 индикатора отека
            return sum(swelling_indicators) >= ANOMALY_DETECTION_ADVANCED_THRESHOLDS["min_swelling_metrics_detected"]
            
        except Exception as e:
            logger.error(f"Ошибка обнаружения отека: {e}")
            return False

    def _detect_asymmetry_changes(self, changes: Dict[str, Any]) -> bool:
        """Обнаружение изменений асимметрии"""
        try:
            # Проверка изменений в асимметрии
            if "gonial_angle_asymmetry" in changes:
                asymmetry_change = abs(changes["gonial_angle_asymmetry"].get("absolute_change", 0))
                if asymmetry_change >= ANOMALY_DETECTION_ADVANCED_THRESHOLDS["asymmetry_change_threshold"]:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Ошибка обнаружения изменений асимметрии: {e}")
            return False

    def _analyze_healing_dynamics(self, changes: Dict[str, Any], interval_days: int) -> Dict[str, Any]:
        """Анализ динамики заживления"""
        try:
            healing_indicators = {}
            
            # Проверка метрик заживления
            healing_metrics = ["texture_entropy", "shape_error", "asymmetry_score"]
            
            for metric in healing_metrics:
                if metric in changes:
                    absolute_change = changes[metric].get("absolute_change", 0)
                    
                    # Индикаторы заживления (уменьшение ошибок и улучшение метрик)
                    if interval_days >= ANOMALY_DETECTION_ADVANCED_THRESHOLDS["healing_detection_interval_days"]:
                        if metric in ["shape_error", "asymmetry_score"] and absolute_change < -0.05:
                            healing_indicators[metric] = True
                        elif metric == "texture_entropy" and absolute_change > 0.1:
                            healing_indicators[metric] = True
                        else:
                            healing_indicators[metric] = False
            
            return {
                "healing_detected": any(healing_indicators.values()),
                "details": healing_indicators
            }
            
        except Exception as e:
            logger.error(f"Ошибка анализа динамики заживления: {e}")
            return {"healing_detected": False, "details": {}}

    def _calculate_surgery_likelihood(self, swelling_detected: bool, asymmetry_changes: bool, 
                                    healing_dynamics: Dict[str, Any], interval_days: int) -> float:
        """Расчет вероятности хирургического вмешательства"""
        try:
            likelihood = 0.0
            
            # Веса индикаторов
            if swelling_detected:
                likelihood += ANOMALY_DETECTION_ADVANCED_THRESHOLDS["surgery_swelling_weight"]
            
            if asymmetry_changes:
                likelihood += ANOMALY_DETECTION_ADVANCED_THRESHOLDS["surgery_asymmetry_weight"]
            
            if healing_dynamics.get("healing_detected", False):
                likelihood += ANOMALY_DETECTION_ADVANCED_THRESHOLDS["surgery_healing_weight"]
            
            # Корректировка на основе интервала
            if interval_days <= ANOMALY_DETECTION_ADVANCED_THRESHOLDS["surgery_short_interval_days"]:
                likelihood *= ANOMALY_DETECTION_ADVANCED_THRESHOLDS["surgery_short_interval_multiplier"]
            elif interval_days >= ANOMALY_DETECTION_ADVANCED_THRESHOLDS["surgery_long_interval_days"]:
                likelihood *= ANOMALY_DETECTION_ADVANCED_THRESHOLDS["surgery_long_interval_multiplier"]
            
            return np.clip(likelihood, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Ошибка расчета вероятности хирургии: {e}")
            return 0.0

    def _generate_surgical_recommendation(self, evidence: bool, likelihood: float) -> str:
        """Генерация рекомендации по хирургическому вмешательству"""
        try:
            if not evidence:
                return "Признаки хирургического вмешательства не обнаружены"
            elif likelihood >= 0.8:
                return "Высокая вероятность хирургического вмешательства. Требуется детальный анализ"
            elif likelihood >= 0.6:
                return "Умеренная вероятность хирургического вмешательства. Рекомендуется дополнительная проверка"
            else:
                return "Низкая вероятность хирургического вмешательства. Возможны естественные изменения"
                
        except Exception as e:
            logger.error(f"Ошибка генерации рекомендации: {e}")
            return "Ошибка анализа"

    def perform_cross_source_verification(self, same_date_images: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Кросс-источниковая верификация
        Согласно правкам: анализ embedding и metrics consistency
        """
        if not same_date_images:
            logger.warning("Нет данных для кросс-источниковой верификации")
            return {}
        
        try:
            logger.info(f"Кросс-источниковая верификация для {len(same_date_images)} дат")
            
            cross_verification = {}
            
            # Ключевые метрики для кросс-валидации
            key_metrics = [
                "forehead_height_ratio", "nose_width_ratio", "mouth_width_ratio",
                "chin_width_ratio", "facial_symmetry_index", "skull_width_ratio",
                "temporal_bone_angle", "zygomatic_arch_width", "orbital_depth",
                "interpupillary_distance_ratio"
            ]
            
            for date, sources_data in same_date_images.items():
                if len(sources_data) < 2:
                    continue
                
                source_embeddings = []
                source_names = []
                source_metrics_list = []
                
                # Сбор данных по источникам
                for source_name, data in sources_data.items():
                    embedding = data.get("embedding")
                    if embedding is not None:
                        source_embeddings.append(embedding)
                        source_names.append(source_name)
                    
                    # Сбор метрик
                    current_metrics = {k: data.get(k) for k in key_metrics if k in data}
                    if current_metrics:
                        source_metrics_list.append(current_metrics)
                
                if len(source_embeddings) < 2:
                    continue
                
                # ИСПРАВЛЕНО: Анализ consistency эмбеддингов
                embedding_consistency_score = 1.0
                embedding_max_distance = 0.0
                embedding_mean_distance = 0.0
                embedding_critical_anomaly = False
                
                if len(source_embeddings) >= 2:
                    distance_matrix = cosine_distances(source_embeddings)
                    # Исключение диагонали (расстояние до самого себя)
                    off_diagonal_distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
                    
                    embedding_max_distance = np.max(off_diagonal_distances)
                    embedding_mean_distance = np.mean(off_diagonal_distances)
                    
                    # Критическая аномалия
                    embedding_critical_anomaly = embedding_max_distance >= ANOMALY_DETECTION_ADVANCED_THRESHOLDS["cross_source_critical_distance"]
                    embedding_consistency_score = 1.0 - embedding_mean_distance
                
                # ИСПРАВЛЕНО: Анализ consistency метрик
                metrics_consistency_score, metrics_inconsistency_details = self._calculate_metrics_consistency(
                    source_metrics_list, key_metrics
                )
                
                # Общая consistency
                overall_consistency_score = (embedding_consistency_score + metrics_consistency_score) / 2.0
                anomaly_detected = embedding_critical_anomaly or metrics_consistency_score < 0.7
                
                # Объяснение
                explanation = self._generate_cross_source_explanation(
                    embedding_critical_anomaly, embedding_max_distance, metrics_inconsistency_details
                )
                
                cross_verification[date] = {
                    "sources_analyzed": source_names,
                    "embedding_max_distance": float(embedding_max_distance),
                    "embedding_mean_distance": float(embedding_mean_distance),
                    "embedding_consistency_score": float(embedding_consistency_score),
                    "metrics_consistency_score": float(metrics_consistency_score),
                    "metrics_inconsistency_details": metrics_inconsistency_details,
                    "critical_anomaly_detected": anomaly_detected,
                    "overall_consistency_score": float(overall_consistency_score),
                    "explanation": explanation
                }
            
            logger.info(f"Кросс-источниковая верификация завершена для {len(cross_verification)} дат")
            return cross_verification
            
        except Exception as e:
            logger.error(f"Ошибка кросс-источниковой верификации: {e}")
            return {}

    def _calculate_metrics_consistency(self, source_metrics_list: List[Dict[str, Any]], 
                                     key_metrics: List[str]) -> Tuple[float, Dict[str, Any]]:
        """Расчет consistency метрик между источниками"""
        try:
            if not source_metrics_list or len(source_metrics_list) < 2:
                return 1.0, {}
            
            inconsistency_details = {}
            consistency_scores_per_metric = []
            
            for metric_name in key_metrics:
                values_for_metric = [metrics.get(metric_name) for metrics in source_metrics_list 
                                   if metrics.get(metric_name) is not None]
                
                if len(values_for_metric) < 2:
                    continue
                
                values_array = np.array(values_for_metric)
                
                # Расчет coefficient of variation
                mean_val = np.mean(values_array)
                std_val = np.std(values_array)
                
                if mean_val != 0:
                    cv = std_val / mean_val
                else:
                    cv = 0.0 if std_val == 0 else float('inf')
                
                # Consistency score (чем меньше CV, тем выше consistency)
                consistency_score = max(0.0, 1.0 - cv / 0.2)  # CV > 0.2 дает score = 0
                consistency_scores_per_metric.append(consistency_score)
                
                if consistency_score < 0.7:
                    inconsistency_details[metric_name] = {
                        "values": values_array.tolist(),
                        "mean": float(f"{mean_val:.4f}"),
                        "std": float(f"{std_val:.4f}"),
                        "cv": float(f"{cv:.4f}"),
                        "consistency_score": float(f"{consistency_score:.4f}"),
                        "status": "inconsistent"
                    }
                else:
                    inconsistency_details[metric_name] = {
                        "mean": float(f"{mean_val:.4f}"),
                        "std": float(f"{std_val:.4f}"),
                        "cv": float(f"{cv:.4f}"),
                        "consistency_score": float(f"{consistency_score:.4f}"),
                        "status": "consistent"
                    }
            
            overall_metrics_consistency = np.mean(consistency_scores_per_metric) if consistency_scores_per_metric else 1.0
            
            return overall_metrics_consistency, inconsistency_details
            
        except Exception as e:
            logger.error(f"Ошибка расчета consistency метрик: {e}")
            return 0.0, {}

    def _generate_cross_source_explanation(self, embedding_critical_anomaly: bool, 
                                         embedding_max_distance: float, 
                                         metrics_inconsistency_details: Dict[str, Any]) -> str:
        """Генерация объяснения кросс-источниковой верификации"""
        try:
            explanations = []
            
            # Анализ эмбеддингов
            if embedding_critical_anomaly:
                explanations.append(f"Критическое расхождение эмбеддингов: {embedding_max_distance:.2f}. "
                                  "Возможна подмена или использование маски.")
            elif embedding_max_distance > 0.4:
                explanations.append(f"Умеренное расхождение эмбеддингов: {embedding_max_distance:.2f}, "
                                  "требуется дополнительная проверка.")
            else:
                explanations.append(f"Эмбеддинги согласованы: {embedding_max_distance:.2f}.")
            
            # Анализ метрик
            inconsistent_metrics = [metric for metric, details in metrics_inconsistency_details.items() 
                                  if details["status"] == "inconsistent"]
            
            if inconsistent_metrics:
                explanations.append(f"Несогласованные метрики: {', '.join(inconsistent_metrics)}. "
                                  "Возможны технические различия между источниками.")
            else:
                explanations.append("Метрики между источниками согласованы.")
            
            return " ".join(explanations)
            
        except Exception as e:
            logger.error(f"Ошибка генерации объяснения: {e}")
            return "Ошибка анализа кросс-источниковой верификации."

    def auto_calibrate_detection_thresholds(self, historical_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        ИСПРАВЛЕНО: Автокалибровка порогов детекции
        Согласно правкам: auto-calibrated thresholds на исторических данных
        """
        if not historical_data:
            logger.warning("Нет исторических данных для автокалибровки")
            return self.calibrated_thresholds.copy()
        
        try:
            logger.info(f"Автокалибровка порогов на {len(historical_data)} образцах")
            
            # Использование функции из coreconfig
            calibrated_thresholds = auto_calibrate_thresholds_historical_data(historical_data)
            
            # Дополнительная калибровка специфичных для аномалий порогов
            authenticity_scores = [item.get("authenticity_score", 0.5) for item in historical_data]
            anomaly_scores = [item.get("anomaly_score", 0.0) for item in historical_data]
            
            if authenticity_scores:
                # Калибровка порога маски/двойника
                calibrated_thresholds["mask_or_double_threshold"] = np.percentile(authenticity_scores, 10)
                
                # Калибровка порога требующего анализа
                calibrated_thresholds["requires_analysis_threshold"] = np.percentile(authenticity_scores, 40)
            
            if anomaly_scores:
                # Калибровка порога аномалий
                calibrated_thresholds["anomaly_detection_threshold"] = np.percentile(anomaly_scores, 90)
            
            # Обновление внутренних порогов
            self.calibrated_thresholds.update(calibrated_thresholds)
            self.calibrated = True
            
            logger.info("Автокалибровка порогов завершена успешно")
            return calibrated_thresholds
            
        except Exception as e:
            logger.error(f"Ошибка автокалибровки порогов: {e}")
            return self.calibrated_thresholds.copy()

    def validate_identity_consistency(self, identity_metrics_over_time: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Валидация согласованности идентичности
        Согласно правкам: анализ temporal metrics и стабильности
        """
        if not identity_metrics_over_time:
            logger.warning("Нет данных для валидации согласованности идентичности")
            return {}
        
        try:
            logger.info(f"Валидация согласованности для {len(identity_metrics_over_time)} идентичностей")
            
            consistency_validation = {}
            
            for identity_id, metrics_timeline in identity_metrics_over_time.items():
                logger.info(f"Анализ идентичности: {identity_id}")
                
                # Группировка метрик по времени
                metric_series = {}
                dates_series = {}
                
                for timestamp_str, metrics in metrics_timeline.items():
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    
                    for metric_name, value in metrics.items():
                        if metric_name not in metric_series:
                            metric_series[metric_name] = []
                            dates_series[metric_name] = []
                        
                        metric_series[metric_name].append(value)
                        dates_series[metric_name].append(timestamp)
                
                # Анализ каждой метрики
                metric_consistency = {}
                
                for metric_name, values in metric_series.items():
                    if len(values) < 2:
                        metric_consistency[metric_name] = {
                            "coefficient_of_variation": 0.0,
                            "trend_slope": 0.0,
                            "outlier_count": 0,
                            "rapid_changes_detected": False,
                            "stability_score": 1.0,
                            "reason": "insufficient_data"
                        }
                        continue
                    
                    # Coefficient of variation
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    cv = std_val / mean_val if mean_val != 0 else (0.0 if std_val == 0 else float('inf'))
                    
                    # Trend slope
                    trend_slope = self._calculate_trend_slope(values)
                    
                    # Outliers (Z-score > 2.5)
                    z_scores = np.abs(stats.zscore(values))
                    outliers = np.sum(z_scores > 2.5)
                    
                    # Rapid changes
                    rapid_changes_detected = False
                    if len(values) > 1:
                        diffs = np.abs(np.diff(values))
                        if len(diffs) > 0 and np.std(diffs) > 0:
                            if np.any(diffs > 3 * np.std(diffs)):
                                rapid_changes_detected = True
                    
                    # Stability score
                    stability_score = max(0.0, 1.0 - cv / 0.5 - abs(trend_slope) / 0.2 - outliers * 0.1 - (0.2 if rapid_changes_detected else 0))
                    
                    # Reason
                    reason = "stable"
                    if stability_score < ANOMALY_DETECTION_ADVANCED_THRESHOLDS["metric_stability_threshold"]:
                        if cv > 0.5:
                            reason += ", high_variability"
                        if abs(trend_slope) > 0.1:
                            reason += ", significant_trend"
                        if outliers > 0:
                            reason += f", {outliers}_outliers"
                        if rapid_changes_detected:
                            reason += ", rapid_changes"
                    
                    metric_consistency[metric_name] = {
                        "coefficient_of_variation": f"{cv:.4f}",
                        "trend_slope": f"{trend_slope:.4f}",
                        "outlier_count": int(outliers),
                        "rapid_changes_detected": rapid_changes_detected,
                        "stability_score": f"{stability_score:.4f}",
                        "reason": reason
                    }
                
                # Общая оценка согласованности
                stability_scores = [float(mc["stability_score"]) for mc in metric_consistency.values() 
                                  if "stability_score" in mc]
                overall_consistency = np.mean(stability_scores) if stability_scores else 0.0
                
                consistency_validation[identity_id] = {
                    "metric_consistency": metric_consistency,
                    "overall_consistency_score": f"{overall_consistency:.4f}",
                    "identity_stable": overall_consistency >= ANOMALY_DETECTION_ADVANCED_THRESHOLDS["identity_stability_threshold"],
                    "anomalous_metrics": [
                        {"name": name, "score": float(mc["stability_score"]), "reason": mc["reason"]}
                        for name, mc in metric_consistency.items()
                        if float(mc["stability_score"]) < ANOMALY_DETECTION_ADVANCED_THRESHOLDS["metric_stability_threshold"]
                    ]
                }
            
            logger.info(f"Валидация согласованности завершена для {len(consistency_validation)} идентичностей")
            return consistency_validation
            
        except Exception as e:
            logger.error(f"Ошибка валидации согласованности идентичности: {e}")
            return {}

    def _calculate_trend_slope(self, values: List[float]) -> float:
        """Расчет наклона тренда"""
        try:
            if len(values) < 2:
                return 0.0
            
            x = np.arange(len(values))
            slope, _, _, _, _ = stats.linregress(x, values)
            
            return slope
            
        except Exception as e:
            logger.error(f"Ошибка расчета наклона тренда: {e}")
            return 0.0

    def save_analysis_cache(self, cache_file: str = "anomaly_cache.pkl") -> None:
        """Сохранение кэша анализа"""
        try:
            cache_path = CACHE_DIR / cache_file
            CACHE_DIR.mkdir(exist_ok=True)
            
            cache_data = {
                "analysis_cache": self.analysis_cache,
                "calibrated_thresholds": self.calibrated_thresholds,
                "calibrated": self.calibrated
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"Кэш анализа сохранен: {cache_path}")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения кэша: {e}")

    def load_analysis_cache(self, cache_file: str = "anomaly_cache.pkl") -> None:
        """Загрузка кэша анализа"""
        try:
            cache_path = CACHE_DIR / cache_file
            
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                
                self.analysis_cache = cache_data.get("analysis_cache", {})
                self.calibrated_thresholds = cache_data.get("calibrated_thresholds", CRITICAL_THRESHOLDS.copy())
                self.calibrated = cache_data.get("calibrated", False)
                
                logger.info(f"Кэш анализа загружен: {cache_path}")
            else:
                logger.info("Файл кэша не найден, используется пустой кэш")
                
        except Exception as e:
            logger.error(f"Ошибка загрузки кэша: {e}")

    def self_test(self) -> None:
        """Самотестирование модуля"""
        logger.info("=== Самотестирование AnomalyDetector ===")
        
        # Информация о параметрах
        logger.info(f"Байесовские априорные: {self.bayesian_priors}")
        logger.info(f"Калиброван: {self.calibrated}")
        logger.info(f"Пороги каскада: {CASCADE_VERIFICATION_THRESHOLDS}")
        
        # Тестовые данные
        test_evidence = {
            "identity_1": {
                "geometry_score": 0.8,
                "embedding_score": 0.9,
                "texture_score": 0.7,
                "temporal_consistency": 0.85
            },
            "identity_2": {
                "geometry_score": 0.3,
                "embedding_score": 0.2,
                "texture_score": 0.4,
                "temporal_consistency": 0.25
            }
        }
        
        try:
            # Тест байесовского анализа
            bayesian_results = self.apply_bayesian_identity_analysis(test_evidence)
            logger.info(f"Тест байесовского анализа: {len(bayesian_results)} результатов")
            
            # Тест каскадной верификации
            cascade_result = self.perform_cascade_verification(0.8, 0.9, 0.7, 0.85)
            logger.info(f"Тест каскадной верификации: {cascade_result['levels_passed']}/4 уровней")
            
            # Тест расчета authenticity score
            auth_score = self.calculate_identity_authenticity_score(0.8, 0.9, 0.7, 0.85)
            logger.info(f"Тест authenticity score: {auth_score:.3f}")
            
            # Тест классификации маски
            test_texture = {
                "forehead": {"shannon_entropy": 5.5, "shape_error": 0.4},
                "cheek": {"shannon_entropy": 5.2, "shape_error": 0.45}
            }
            mask_classification = self.classify_mask_technology_level(test_texture)
            logger.info(f"Тест классификации маски: {mask_classification['level']}")
            
            # Тест автокалибровки
            test_historical = [
                {"authenticity_score": 0.7, "anomaly_score": 0.1},
                {"authenticity_score": 0.8, "anomaly_score": 0.05},
                {"authenticity_score": 0.6, "anomaly_score": 0.2}
            ]
            calibrated = self.auto_calibrate_detection_thresholds(test_historical)
            logger.info(f"Тест автокалибровки: {len(calibrated)} порогов")
            
        except Exception as e:
            logger.error(f"Ошибка самотестирования: {e}")
        
        logger.info("=== Самотестирование завершено ===")

# ==================== ТОЧКА ВХОДА ====================

if __name__ == "__main__":
    detector = AnomalyDetector()
    detector.self_test()