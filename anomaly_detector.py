# anomaly_detector.py
import os
import json
import logging
import hashlib
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import euclidean, cosine
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import pickle
import time
import psutil
from functools import lru_cache
import threading
from collections import OrderedDict, defaultdict
import msgpack

from core_config import get_config

# Настройка логирования
logger = logging.getLogger(__name__)

# === КОНСТАНТЫ И КОНФИГУРАЦИЯ ===

# Веса для итогового показателя подлинности (0.3 + 0.3 + 0.2 + 0.2 = 1.0)
AUTHENTICITY_WEIGHTS = {
    'geometry': 0.3,
    'embedding': 0.3,
    'texture': 0.2,
    'temporal': 0.2
}

# Пороги для классификации
AUTHENTICITY_THRESHOLDS = {
    'fake_threshold': 0.3,      # < 0.3 = маска/двойник
    'suspicious_threshold': 0.7  # 0.3-0.7 = сомнительно, > 0.7 = подлинное
}

# Параметры байесовского анализа
BAYESIAN_PARAMS = {
    'prior_same_person': 0.5,
    'prior_different_person': 0.5,
    'likelihood_threshold': 0.1,
    'convergence_threshold': 0.01,
    'max_iterations': 100
}

# Параметры каскадной верификации
CASCADE_PARAMS = {
    'geometry_critical_threshold': 0.2,
    'embedding_critical_threshold': 0.25,
    'texture_critical_threshold': 0.3,
    'temporal_critical_threshold': 0.3,
    'cross_source_threshold': 0.5
}

# Параметры IsolationForest для детекции аномалий
ISOLATION_FOREST_PARAMS = {
    'contamination': 0.1,
    'random_state': 42,
    'n_estimators': 100,
    'max_samples': 'auto'
}

# === СТРУКТУРЫ ДАННЫХ ===

@dataclass
class AuthenticityScore:
    """Результат анализа подлинности"""
    image_id: str
    filepath: str
    
    # Индивидуальные баллы
    geometry_score: float = 0.0
    embedding_score: float = 0.0
    texture_score: float = 0.0
    temporal_score: float = 0.0
    
    # Итоговый балл
    overall_authenticity: float = 0.0
    
    # Классификация
    classification: str = "unknown"  # authentic, suspicious, fake
    confidence_level: float = 0.0
    
    # Байесовские вероятности
    posterior_same_person: float = 0.5
    posterior_different_person: float = 0.5
    
    # Флаги и предупреждения
    critical_flags: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    anomaly_flags: List[str] = field(default_factory=list)
    
    # Метаданные
    processing_time_ms: float = 0.0
    analysis_method: str = "cascade_bayesian"
    model_version: str = "v1.0"

@dataclass
class BayesianEvidence:
    """Байесовские доказательства для обновления вероятностей"""
    evidence_type: str  # geometry, embedding, texture, temporal
    likelihood_same: float
    likelihood_different: float
    confidence: float
    source_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CascadeResult:
    """Результат каскадной верификации"""
    stage: str
    passed: bool
    score: float
    threshold: float
    critical_failure: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CrossSourceAnalysis:
    """Результат кросс-источникового анализа"""
    date: datetime.date
    sources_count: int
    consistency_score: float
    max_distance: float
    mean_distance: float
    outlier_sources: List[str] = field(default_factory=list)
    critical_inconsistency: bool = False

# === ОСНОВНОЙ КЛАСС ДЕТЕКТОРА АНОМАЛИЙ ===

class AnomalyDetector:
    """Детектор аномалий для выявления масок и двойников"""
    
    def __init__(self):
        self.config = get_config()
        self.cache_dir = Path("./cache/anomaly_detector")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Кэш результатов
        self.authenticity_cache: Dict[str, AuthenticityScore] = {}
        self.bayesian_cache: Dict[str, Dict[str, float]] = {}
        
        # Модели для детекции аномалий
        self.isolation_forest = None
        self.scaler = StandardScaler()
        
        # Статистика
        self.processing_stats = {
            'total_analyzed': 0,
            'authentic_detected': 0,
            'suspicious_detected': 0,
            'fake_detected': 0,
            'critical_failures': 0,
            'bayesian_updates': 0
        }
        
        # Блокировка для потокобезопасности
        self.analysis_lock = threading.Lock()
        
        # Инициализация моделей
        self._initialize_anomaly_models()
        
        logger.info("AnomalyDetector инициализирован")

    def _initialize_anomaly_models(self):
        """Инициализация моделей детекции аномалий"""
        try:
            # Инициализация IsolationForest
            self.isolation_forest = IsolationForest(
                contamination=ISOLATION_FOREST_PARAMS['contamination'],
                random_state=ISOLATION_FOREST_PARAMS['random_state'],
                n_estimators=ISOLATION_FOREST_PARAMS['n_estimators'],
                max_samples=ISOLATION_FOREST_PARAMS['max_samples']
            )
            
            # Загрузка предобученных моделей если есть
            self._load_pretrained_models()
            
            logger.debug("Модели детекции аномалий инициализированы")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации моделей: {e}")
            raise

    def _load_pretrained_models(self):
        """Загрузка предобученных моделей"""
        try:
            models_path = self.cache_dir / "pretrained_models.pkl"
            if models_path.exists():
                with open(models_path, 'rb') as f:
                    models_data = pickle.load(f)
                
                if 'isolation_forest' in models_data:
                    self.isolation_forest = models_data['isolation_forest']
                if 'scaler' in models_data:
                    self.scaler = models_data['scaler']
                
                logger.info("Предобученные модели загружены")
                
        except Exception as e:
            logger.warning(f"Не удалось загрузить предобученные модели: {e}")

    def calculate_identity_authenticity_score(self, 
                                            geometry_metrics=None,
                                            embedding_package=None, 
                                            texture_package=None,
                                            temporal_analysis=None) -> Optional[AuthenticityScore]:
        """
        Основная функция расчета итогового показателя подлинности
        
        Args:
            geometry_metrics: Метрики геометрии лица
            embedding_package: Пакет эмбеддингов
            texture_package: Пакет текстурного анализа
            temporal_analysis: Результаты временного анализа
            
        Returns:
            Объект с результатами анализа подлинности
        """
        try:
            start_time = time.time()
            
            # Проверка входных данных
            if not any([geometry_metrics, embedding_package, texture_package, temporal_analysis]):
                logger.error("Отсутствуют входные данные для анализа")
                return None
            
            # Определение image_id
            image_id = self._extract_image_id(geometry_metrics, embedding_package, 
                                            texture_package, temporal_analysis)
            
            # Проверка кэша
            if image_id in self.authenticity_cache:
                cached_result = self.authenticity_cache[image_id]
                cached_result.processing_time_ms = (time.time() - start_time) * 1000
                return cached_result
            
            # Создание объекта результата
            result = AuthenticityScore(
                image_id=image_id,
                filepath=self._extract_filepath(geometry_metrics, embedding_package, 
                                              texture_package, temporal_analysis)
            )
            
            # Расчет индивидуальных баллов
            result.geometry_score = self._calculate_geometry_score(geometry_metrics)
            result.embedding_score = self._calculate_embedding_score(embedding_package)
            result.texture_score = self._calculate_texture_score(texture_package)
            result.temporal_score = self._calculate_temporal_score(temporal_analysis)
            
            # Каскадная верификация
            cascade_results = self.perform_cascade_verification(result)
            
            # Расчет итогового балла
            result.overall_authenticity = self._calculate_weighted_score(result)
            
            # Классификация результата
            result.classification = self._classify_authenticity(result.overall_authenticity)
            
            # Байесовский анализ
            bayesian_result = self.apply_bayesian_identity_analysis(result, 
                                                                   geometry_metrics,
                                                                   embedding_package,
                                                                   texture_package,
                                                                   temporal_analysis)
            
            result.posterior_same_person = bayesian_result.get('posterior_same_person', 0.5)
            result.posterior_different_person = bayesian_result.get('posterior_different_person', 0.5)
            
            # Детекция аномалий
            anomaly_flags = self._detect_statistical_anomalies(result)
            result.anomaly_flags.extend(anomaly_flags)
            
            # Расчет уровня достоверности
            result.confidence_level = self._calculate_confidence_level(result, cascade_results)
            
            # Метаданные обработки
            result.processing_time_ms = (time.time() - start_time) * 1000
            
            # Сохранение в кэш
            self.authenticity_cache[image_id] = result
            
            # Обновление статистики
            self._update_processing_stats(result)
            
            logger.debug(f"Анализ подлинности завершен за {result.processing_time_ms:.1f}мс")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка расчета подлинности: {e}")
            return None

    def _extract_image_id(self, *packages) -> str:
        """Извлечение image_id из любого доступного пакета"""
        for package in packages:
            if package and hasattr(package, 'image_id'):
                return package.image_id
        return hashlib.sha256(str(datetime.datetime.now()).encode()).hexdigest()

    def _extract_filepath(self, *packages) -> str:
        """Извлечение filepath из любого доступного пакета"""
        for package in packages:
            if package and hasattr(package, 'filepath'):
                return package.filepath
        return ""

    def _calculate_geometry_score(self, geometry_metrics) -> float:
        """Расчет балла геометрии"""
        try:
            if geometry_metrics is None:
                return 0.5  # Нейтральный балл при отсутствии данных
            
            # Базовый балл на основе достоверности ландмарков
            base_score = getattr(geometry_metrics, 'confidence_score', 0.5)
            
            # Штрафы за ошибки формы
            shape_error = getattr(geometry_metrics, 'shape_error', 0.0)
            eye_region_error = getattr(geometry_metrics, 'eye_region_error', 0.0)
            
            # Нормализация ошибок
            shape_penalty = min(shape_error / 100.0, 0.3)  # Максимальный штраф 30%
            eye_penalty = min(eye_region_error / 50.0, 0.2)  # Максимальный штраф 20%
            
            # Итоговый балл
            geometry_score = base_score - shape_penalty - eye_penalty
            
            return float(max(0.0, min(1.0, geometry_score)))
            
        except Exception as e:
            logger.error(f"Ошибка расчета геометрического балла: {e}")
            return 0.0

    def _calculate_embedding_score(self, embedding_package) -> float:
        """Расчет балла эмбеддингов"""
        try:
            if embedding_package is None:
                return 0.5
            
            # Базовый балл на основе достоверности извлечения
            base_score = getattr(embedding_package, 'extraction_confidence', 0.5)
            
            # Анализ кластерной принадлежности
            cluster_confidence = getattr(embedding_package, 'cluster_confidence', 0.5)
            
            # Проверка на аутлаеры
            is_outlier = getattr(embedding_package, 'is_outlier', False)
            outlier_penalty = 0.3 if is_outlier else 0.0
            
            # Итоговый балл
            embedding_score = (base_score + cluster_confidence) / 2.0 - outlier_penalty
            
            return float(max(0.0, min(1.0, embedding_score)))
            
        except Exception as e:
            logger.error(f"Ошибка расчета балла эмбеддингов: {e}")
            return 0.0

    def _calculate_texture_score(self, texture_package) -> float:
        """Расчет балла текстуры"""
        try:
            if texture_package is None:
                return 0.5
            
            # Базовый балл материала
            material_score = getattr(texture_package, 'material_authenticity_score', 0.5)
            
            # Штрафы за артефакты
            has_seam_artifacts = getattr(texture_package, 'seam_artifacts_detected', False)
            has_texture_transitions = getattr(texture_package, 'texture_transitions_detected', False)
            
            seam_penalty = 0.4 if has_seam_artifacts else 0.0
            transition_penalty = 0.3 if has_texture_transitions else 0.0
            
            # Анализ энтропии
            entropy_score = getattr(texture_package, 'entropy_score', 0.5)
            
            # Итоговый балл
            texture_score = (material_score + entropy_score) / 2.0 - seam_penalty - transition_penalty
            
            return float(max(0.0, min(1.0, texture_score)))
            
        except Exception as e:
            logger.error(f"Ошибка расчета балла текстуры: {e}")
            return 0.0

    def _calculate_temporal_score(self, temporal_analysis) -> float:
        """Расчет балла временной консистентности"""
        try:
            if temporal_analysis is None:
                return 0.5
            
            # Базовый балл консистентности
            consistency_score = getattr(temporal_analysis, 'consistency_score', 0.5)
            
            # Штрафы за аномалии
            temporal_anomalies = getattr(temporal_analysis, 'anomalies_detected', [])
            anomaly_penalty = min(len(temporal_anomalies) * 0.1, 0.4)
            
            # Анализ медицинской валидности
            medical_validity = getattr(temporal_analysis, 'medical_validity_score', 0.5)
            
            # Итоговый балл
            temporal_score = (consistency_score + medical_validity) / 2.0 - anomaly_penalty
            
            return float(max(0.0, min(1.0, temporal_score)))
            
        except Exception as e:
            logger.error(f"Ошибка расчета временного балла: {e}")
            return 0.0

    def _calculate_weighted_score(self, result: AuthenticityScore) -> float:
        """Расчет взвешенного итогового балла по формуле 0.3+0.3+0.2+0.2"""
        try:
            weighted_score = (
                result.geometry_score * AUTHENTICITY_WEIGHTS['geometry'] +
                result.embedding_score * AUTHENTICITY_WEIGHTS['embedding'] +
                result.texture_score * AUTHENTICITY_WEIGHTS['texture'] +
                result.temporal_score * AUTHENTICITY_WEIGHTS['temporal']
            )
            
            return float(max(0.0, min(1.0, weighted_score)))
            
        except Exception as e:
            logger.error(f"Ошибка расчета взвешенного балла: {e}")
            return 0.0

    def _classify_authenticity(self, overall_score: float) -> str:
        """Классификация результата по пороговым значениям"""
        if overall_score < AUTHENTICITY_THRESHOLDS['fake_threshold']:
            return "fake"
        elif overall_score < AUTHENTICITY_THRESHOLDS['suspicious_threshold']:
            return "suspicious"
        else:
            return "authentic"

    def perform_cascade_verification(self, result: AuthenticityScore) -> List[CascadeResult]:
        """
        Каскадная верификация с проверкой критических порогов
        
        Args:
            result: Результат анализа подлинности
            
        Returns:
            Список результатов каскадной проверки
        """
        try:
            cascade_results = []
            
            # Проверка геометрии
            geometry_result = CascadeResult(
                stage="geometry",
                passed=result.geometry_score >= CASCADE_PARAMS['geometry_critical_threshold'],
                score=result.geometry_score,
                threshold=CASCADE_PARAMS['geometry_critical_threshold'],
                critical_failure=result.geometry_score < CASCADE_PARAMS['geometry_critical_threshold']
            )
            cascade_results.append(geometry_result)
            
            if geometry_result.critical_failure:
                result.critical_flags.append("geometry_critical_failure")
                logger.warning(f"Критический провал геометрии: {result.geometry_score}")
            
            # Проверка эмбеддингов
            embedding_result = CascadeResult(
                stage="embedding",
                passed=result.embedding_score >= CASCADE_PARAMS['embedding_critical_threshold'],
                score=result.embedding_score,
                threshold=CASCADE_PARAMS['embedding_critical_threshold'],
                critical_failure=result.embedding_score < CASCADE_PARAMS['embedding_critical_threshold']
            )
            cascade_results.append(embedding_result)
            
            if embedding_result.critical_failure:
                result.critical_flags.append("embedding_critical_failure")
                logger.warning(f"Критический провал эмбеддингов: {result.embedding_score}")
            
            # Проверка текстуры
            texture_result = CascadeResult(
                stage="texture",
                passed=result.texture_score >= CASCADE_PARAMS['texture_critical_threshold'],
                score=result.texture_score,
                threshold=CASCADE_PARAMS['texture_critical_threshold'],
                critical_failure=result.texture_score < CASCADE_PARAMS['texture_critical_threshold']
            )
            cascade_results.append(texture_result)
            
            if texture_result.critical_failure:
                result.critical_flags.append("texture_critical_failure")
            
            # Проверка временной консистентности
            temporal_result = CascadeResult(
                stage="temporal",
                passed=result.temporal_score >= CASCADE_PARAMS['temporal_critical_threshold'],
                score=result.temporal_score,
                threshold=CASCADE_PARAMS['temporal_critical_threshold'],
                critical_failure=result.temporal_score < CASCADE_PARAMS['temporal_critical_threshold']
            )
            cascade_results.append(temporal_result)
            
            if temporal_result.critical_failure:
                result.critical_flags.append("temporal_critical_failure")
            
            # Подсчет критических провалов
            critical_failures = sum(1 for r in cascade_results if r.critical_failure)
            if critical_failures >= 2:
                result.critical_flags.append("multiple_critical_failures")
                result.overall_authenticity = min(result.overall_authenticity, 0.2)
            
            return cascade_results
            
        except Exception as e:
            logger.error(f"Ошибка каскадной верификации: {e}")
            return []

    def apply_bayesian_identity_analysis(self, result: AuthenticityScore, 
                                       geometry_metrics=None,
                                       embedding_package=None,
                                       texture_package=None,
                                       temporal_analysis=None) -> Dict[str, float]:
        """
        Байесовский анализ идентичности
        
        Args:
            result: Результат анализа подлинности
            geometry_metrics: Метрики геометрии
            embedding_package: Пакет эмбеддингов  
            texture_package: Пакет текстуры
            temporal_analysis: Временной анализ
            
        Returns:
            Словарь с байесовскими вероятностями
        """
        try:
            # Получение или инициализация prior вероятностей
            image_id = result.image_id
            if image_id not in self.bayesian_cache:
                self.bayesian_cache[image_id] = {
                    'prior_same_person': BAYESIAN_PARAMS['prior_same_person'],
                    'prior_different_person': BAYESIAN_PARAMS['prior_different_person']
                }
            
            priors = self.bayesian_cache[image_id]
            
            # Сбор доказательств
            evidence_list = []
            
            # Геометрические доказательства
            if geometry_metrics:
                geometry_evidence = self._calculate_geometry_likelihood(geometry_metrics)
                evidence_list.append(geometry_evidence)
            
            # Эмбеддинг доказательства
            if embedding_package:
                embedding_evidence = self._calculate_embedding_likelihood(embedding_package)
                evidence_list.append(embedding_evidence)
            
            # Текстурные доказательства
            if texture_package:
                texture_evidence = self._calculate_texture_likelihood(texture_package)
                evidence_list.append(texture_evidence)
            
            # Временные доказательства
            if temporal_analysis:
                temporal_evidence = self._calculate_temporal_likelihood(temporal_analysis)
                evidence_list.append(temporal_evidence)
            
            # Байесовское обновление
            posterior_same = priors['prior_same_person']
            posterior_different = priors['prior_different_person']
            
            for evidence in evidence_list:
                # Обновление по правилу Байеса
                numerator_same = posterior_same * evidence.likelihood_same
                numerator_different = posterior_different * evidence.likelihood_different
                
                denominator = numerator_same + numerator_different
                
                if denominator > 0:
                    posterior_same = numerator_same / denominator
                    posterior_different = numerator_different / denominator
            
            # Нормализация
            total = posterior_same + posterior_different
            if total > 0:
                posterior_same /= total
                posterior_different /= total
            
            # Обновление кэша
            self.bayesian_cache[image_id].update({
                'posterior_same_person': posterior_same,
                'posterior_different_person': posterior_different
            })
            
            self.processing_stats['bayesian_updates'] += 1
            
            return {
                'posterior_same_person': float(posterior_same),
                'posterior_different_person': float(posterior_different)
            }
            
        except Exception as e:
            logger.error(f"Ошибка байесовского анализа: {e}")
            return {
                'posterior_same_person': 0.5,
                'posterior_different_person': 0.5
            }

    def _calculate_geometry_likelihood(self, geometry_metrics) -> BayesianEvidence:
        """Расчет правдоподобия для геометрических данных"""
        try:
            confidence = getattr(geometry_metrics, 'confidence_score', 0.5)
            shape_error = getattr(geometry_metrics, 'shape_error', 0.0)
            
            # Высокая достоверность и низкая ошибка -> высокая вероятность same_person
            if confidence > 0.8 and shape_error < 20.0:
                likelihood_same = 0.9
                likelihood_different = 0.1
            elif confidence < 0.3 or shape_error > 80.0:
                likelihood_same = 0.1
                likelihood_different = 0.9
            else:
                # Линейная интерполяция
                likelihood_same = confidence * (1.0 - min(shape_error / 100.0, 0.8))
                likelihood_different = 1.0 - likelihood_same
            
            return BayesianEvidence(
                evidence_type="geometry",
                likelihood_same=likelihood_same,
                likelihood_different=likelihood_different,
                confidence=confidence,
                source_data={'shape_error': shape_error}
            )
            
        except Exception as e:
            logger.error(f"Ошибка расчета геометрического правдоподобия: {e}")
            return BayesianEvidence("geometry", 0.5, 0.5, 0.0)

    def _calculate_embedding_likelihood(self, embedding_package) -> BayesianEvidence:
        """Расчет правдоподобия для эмбеддингов"""
        try:
            cluster_confidence = getattr(embedding_package, 'cluster_confidence', 0.5)
            is_outlier = getattr(embedding_package, 'is_outlier', False)
            
            if is_outlier:
                likelihood_same = 0.2
                likelihood_different = 0.8
            elif cluster_confidence > 0.8:
                likelihood_same = 0.9
                likelihood_different = 0.1
            else:
                likelihood_same = cluster_confidence
                likelihood_different = 1.0 - cluster_confidence
            
            return BayesianEvidence(
                evidence_type="embedding",
                likelihood_same=likelihood_same,
                likelihood_different=likelihood_different,
                confidence=cluster_confidence,
                source_data={'is_outlier': is_outlier}
            )
            
        except Exception as e:
            logger.error(f"Ошибка расчета правдоподобия эмбеддингов: {e}")
            return BayesianEvidence("embedding", 0.5, 0.5, 0.0)

    def _calculate_texture_likelihood(self, texture_package) -> BayesianEvidence:
        """Расчет правдоподобия для текстуры"""
        try:
            material_score = getattr(texture_package, 'material_authenticity_score', 0.5)
            has_artifacts = getattr(texture_package, 'seam_artifacts_detected', False)
            
            if has_artifacts:
                likelihood_same = 0.1
                likelihood_different = 0.9
            elif material_score > 0.8:
                likelihood_same = 0.9
                likelihood_different = 0.1
            else:
                likelihood_same = material_score
                likelihood_different = 1.0 - material_score
            
            return BayesianEvidence(
                evidence_type="texture",
                likelihood_same=likelihood_same,
                likelihood_different=likelihood_different,
                confidence=material_score,
                source_data={'has_artifacts': has_artifacts}
            )
            
        except Exception as e:
            logger.error(f"Ошибка расчета текстурного правдоподобия: {e}")
            return BayesianEvidence("texture", 0.5, 0.5, 0.0)

    def _calculate_temporal_likelihood(self, temporal_analysis) -> BayesianEvidence:
        """Расчет правдоподобия для временных данных"""
        try:
            consistency_score = getattr(temporal_analysis, 'consistency_score', 0.5)
            anomalies_count = len(getattr(temporal_analysis, 'anomalies_detected', []))
            
            if anomalies_count > 3:
                likelihood_same = 0.2
                likelihood_different = 0.8
            elif consistency_score > 0.8:
                likelihood_same = 0.9
                likelihood_different = 0.1
            else:
                likelihood_same = consistency_score
                likelihood_different = 1.0 - consistency_score
            
            return BayesianEvidence(
                evidence_type="temporal",
                likelihood_same=likelihood_same,
                likelihood_different=likelihood_different,
                confidence=consistency_score,
                source_data={'anomalies_count': anomalies_count}
            )
            
        except Exception as e:
            logger.error(f"Ошибка расчета временного правдоподобия: {e}")
            return BayesianEvidence("temporal", 0.5, 0.5, 0.0)

    def _detect_statistical_anomalies(self, result: AuthenticityScore) -> List[str]:
        """Детекция статистических аномалий"""
        try:
            anomaly_flags = []
            
            # Создание вектора признаков
            features = np.array([[
                result.geometry_score,
                result.embedding_score,
                result.texture_score,
                result.temporal_score
            ]])
            
            # Нормализация признаков
            if hasattr(self.scaler, 'mean_'):
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features
            
            # Детекция аномалий через IsolationForest
            if self.isolation_forest is not None:
                try:
                    anomaly_score = self.isolation_forest.decision_function(features_scaled)[0]
                    is_anomaly = self.isolation_forest.predict(features_scaled)[0] == -1
                    
                    if is_anomaly:
                        anomaly_flags.append("statistical_anomaly")
                        if anomaly_score < -0.5:
                            anomaly_flags.append("severe_anomaly")
                            
                except Exception as e:
                    logger.warning(f"Ошибка IsolationForest: {e}")
            
            # Проверка экстремальных значений
            scores = [result.geometry_score, result.embedding_score, 
                     result.texture_score, result.temporal_score]
            
            if any(score < 0.1 for score in scores):
                anomaly_flags.append("extreme_low_score")
            
            if all(score > 0.95 for score in scores):
                anomaly_flags.append("suspiciously_perfect")
            
            # Проверка несбалансированности
            score_std = np.std(scores)
            if score_std > 0.4:
                anomaly_flags.append("high_score_variance")
            
            return anomaly_flags
            
        except Exception as e:
            logger.error(f"Ошибка детекции аномалий: {e}")
            return []

    def _calculate_confidence_level(self, result: AuthenticityScore, 
                                  cascade_results: List[CascadeResult]) -> float:
        """Расчет уровня достоверности анализа"""
        try:
            # Базовая достоверность на основе количества доступных данных
            available_scores = sum(1 for score in [result.geometry_score, result.embedding_score,
                                                 result.texture_score, result.temporal_score] if score > 0)
            
            base_confidence = available_scores / 4.0
            
            # Штрафы за критические провалы
            critical_failures = len(result.critical_flags)
            critical_penalty = min(critical_failures * 0.2, 0.6)
            
            # Бонус за согласованность результатов
            scores = [result.geometry_score, result.embedding_score, 
                     result.texture_score, result.temporal_score]
            consistency_bonus = 1.0 - np.std(scores) if np.std(scores) < 0.3 else 0.0
            consistency_bonus *= 0.2
            
            # Итоговая достоверность
            confidence = base_confidence - critical_penalty + consistency_bonus
            
            return float(max(0.0, min(1.0, confidence)))
            
        except Exception as e:
            logger.error(f"Ошибка расчета достоверности: {e}")
            return 0.5

    def perform_cross_source_verification(self, embeddings_by_source: Dict[str, List]) -> CrossSourceAnalysis:
        """
        Кросс-источниковая верификация эмбеддингов
        
        Args:
            embeddings_by_source: Словарь эмбеддингов по источникам
            
        Returns:
            Результат кросс-источникового анализа
        """
        try:
            if len(embeddings_by_source) < 2:
                logger.warning("Недостаточно источников для кросс-верификации")
                return CrossSourceAnalysis(
                    date=datetime.date.today(),
                    sources_count=len(embeddings_by_source),
                    consistency_score=1.0,
                    max_distance=0.0,
                    mean_distance=0.0
                )
            
            # Вычисление матрицы расстояний между источниками
            sources = list(embeddings_by_source.keys())
            embeddings = [np.mean(embeddings_by_source[source], axis=0) 
                         for source in sources]
            
            distances = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    dist = cosine(embeddings[i], embeddings[j])
                    distances.append(dist)
            
            if not distances:
                return CrossSourceAnalysis(
                    date=datetime.date.today(),
                    sources_count=len(embeddings_by_source),
                    consistency_score=1.0,
                    max_distance=0.0,
                    mean_distance=0.0
                )
            
            max_distance = max(distances)
            mean_distance = np.mean(distances)
            
            # Определение аутлаеров
            outlier_sources = []
            if max_distance > CASCADE_PARAMS['cross_source_threshold']:
                # Поиск источника, который наиболее отличается от остальных
                source_distances = []
                for i, source in enumerate(sources):
                    other_distances = []
                    for j, other_embedding in enumerate(embeddings):
                        if i != j:
                            dist = cosine(embeddings[i], other_embedding)
                            other_distances.append(dist)
                    source_distances.append((source, np.mean(other_distances)))
                
                # Источник с максимальным средним расстоянием
                outlier_source = max(source_distances, key=lambda x: x[1])
                if outlier_source[1] > CASCADE_PARAMS['cross_source_threshold']:
                    outlier_sources.append(outlier_source[0])
            
            # Расчет балла консистентности
            consistency_score = 1.0 - min(mean_distance / 0.5, 1.0)
            
            # Критическая несогласованность
            critical_inconsistency = max_distance > CASCADE_PARAMS['cross_source_threshold']
            
            return CrossSourceAnalysis(
                date=datetime.date.today(),
                sources_count=len(embeddings_by_source),
                consistency_score=consistency_score,
                max_distance=max_distance,
                mean_distance=mean_distance,
                outlier_sources=outlier_sources,
                critical_inconsistency=critical_inconsistency
            )
            
        except Exception as e:
            logger.error(f"Ошибка кросс-источниковой верификации: {e}")
            return CrossSourceAnalysis(
                date=datetime.date.today(),
                sources_count=0,
                consistency_score=0.0,
                max_distance=1.0,
                mean_distance=1.0,
                critical_inconsistency=True
            )

    def _update_processing_stats(self, result: AuthenticityScore):
        """Обновление статистики обработки"""
        try:
            self.processing_stats['total_analyzed'] += 1
            
            if result.classification == "authentic":
                self.processing_stats['authentic_detected'] += 1
            elif result.classification == "suspicious":
                self.processing_stats['suspicious_detected'] += 1
            elif result.classification == "fake":
                self.processing_stats['fake_detected'] += 1
            
            if result.critical_flags:
                self.processing_stats['critical_failures'] += 1
                
        except Exception as e:
            logger.error(f"Ошибка обновления статистики: {e}")

    def train_anomaly_models(self, training_data: List[AuthenticityScore]):
        """
        Обучение моделей детекции аномалий
        
        Args:
            training_data: Список результатов для обучения
        """
        try:
            if len(training_data) < 10:
                logger.warning("Недостаточно данных для обучения моделей")
                return
            
            # Подготовка признаков
            features = []
            for result in training_data:
                features.append([
                    result.geometry_score,
                    result.embedding_score,
                    result.texture_score,
                    result.temporal_score
                ])
            
            features = np.array(features)
            
            # Обучение нормализатора
            self.scaler.fit(features)
            features_scaled = self.scaler.transform(features)
            
            # Обучение IsolationForest
            self.isolation_forest.fit(features_scaled)
            
            # Сохранение обученных моделей
            self._save_trained_models()
            
            logger.info(f"Модели обучены на {len(training_data)} образцах")
            
        except Exception as e:
            logger.error(f"Ошибка обучения моделей: {e}")

    def _save_trained_models(self):
        """Сохранение обученных моделей"""
        try:
            models_data = {
                'isolation_forest': self.isolation_forest,
                'scaler': self.scaler,
                'training_timestamp': datetime.datetime.now().isoformat()
            }
            
            models_path = self.cache_dir / "pretrained_models.pkl"
            with open(models_path, 'wb') as f:
                pickle.dump(models_data, f)
            
            logger.info("Обученные модели сохранены")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения моделей: {e}")

    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики обработки
        
        Returns:
            Словарь со статистикой
        """
        stats = self.processing_stats.copy()
        
        if stats['total_analyzed'] > 0:
            stats['authentic_rate'] = stats['authentic_detected'] / stats['total_analyzed']
            stats['suspicious_rate'] = stats['suspicious_detected'] / stats['total_analyzed']
            stats['fake_rate'] = stats['fake_detected'] / stats['total_analyzed']
            stats['critical_failure_rate'] = stats['critical_failures'] / stats['total_analyzed']
        else:
            stats['authentic_rate'] = 0.0
            stats['suspicious_rate'] = 0.0
            stats['fake_rate'] = 0.0
            stats['critical_failure_rate'] = 0.0
        
        # Информация о моделях
        stats['models_info'] = {
            'isolation_forest_trained': self.isolation_forest is not None,
            'scaler_fitted': hasattr(self.scaler, 'mean_'),
            'bayesian_cache_size': len(self.bayesian_cache)
        }
        
        # Информация о памяти
        process = psutil.Process()
        memory_info = process.memory_info()
        stats['memory_usage_mb'] = memory_info.rss / (1024 * 1024)
        
        return stats

    def clear_cache(self):
        """Очистка кэша результатов"""
        try:
            self.authenticity_cache.clear()
            self.bayesian_cache.clear()
            logger.info("Кэш AnomalyDetector очищен")
            
        except Exception as e:
            logger.error(f"Ошибка очистки кэша: {e}")

# === ФУНКЦИИ САМОТЕСТИРОВАНИЯ ===

def self_test():
    """Самотестирование модуля anomaly_detector"""
    try:
        logger.info("Запуск самотестирования anomaly_detector...")
        
        # Создание экземпляра детектора
        detector = AnomalyDetector()
        
        # Создание тестовых данных
        class MockMetrics:
            def __init__(self):
                self.image_id = "test_image"
                self.filepath = "test.jpg"
                self.confidence_score = 0.8
                self.shape_error = 15.0
                self.eye_region_error = 8.0
        
        class MockEmbedding:
            def __init__(self):
                self.image_id = "test_image"
                self.extraction_confidence = 0.9
                self.cluster_confidence = 0.85
                self.is_outlier = False
        
        test_geometry = MockMetrics()
        test_embedding = MockEmbedding()
        
        # Тест расчета подлинности
        result = detector.calculate_identity_authenticity_score(
            geometry_metrics=test_geometry,
            embedding_package=test_embedding
        )
        
        assert result is not None, "Результат анализа не получен"
        assert 0.0 <= result.overall_authenticity <= 1.0, "Неверный диапазон подлинности"
        assert result.classification in ["authentic", "suspicious", "fake"], "Неверная классификация"
        
        # Тест каскадной верификации
        cascade_results = detector.perform_cascade_verification(result)
        assert len(cascade_results) > 0, "Каскадная верификация не выполнена"
        
        # Тест байесовского анализа
        bayesian_result = detector.apply_bayesian_identity_analysis(
            result, test_geometry, test_embedding
        )
        assert 'posterior_same_person' in bayesian_result, "Отсутствует байесовский результат"
        
        # Тест статистики
        stats = detector.get_processing_statistics()
        assert 'total_analyzed' in stats, "Отсутствует статистика"
        
        logger.info("Самотестирование anomaly_detector завершено успешно")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка самотестирования: {e}")
        return False

# === ИНИЦИАЛИЗАЦИЯ ===

if __name__ == "__main__":
    # Запуск самотестирования при прямом вызове модуля
    success = self_test()
    if success:
        print("✅ Модуль anomaly_detector работает корректно")
        
        # Демонстрация основной функциональности
        detector = AnomalyDetector()
        print(f"📊 Результатов в кэше: {len(detector.authenticity_cache)}")
        print(f"🔧 Байесовский кэш: {len(detector.bayesian_cache)}")
        print(f"📏 Веса формулы: {AUTHENTICITY_WEIGHTS}")
        print(f"🎯 Пороги классификации: {AUTHENTICITY_THRESHOLDS}")
        print(f"💾 Кэш-директория: {detector.cache_dir}")
    else:
        print("❌ Обнаружены ошибки в модуле anomaly_detector")
        exit(1)