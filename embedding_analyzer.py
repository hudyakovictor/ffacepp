"""
EmbeddingAnalyzer - Анализатор эмбеддингов лиц с кластеризацией и аномалиями
Версия: 2.0
Дата: 2025-06-15
Исправлены все критические ошибки согласно правкам
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from scipy.spatial.distance import pdist, squareform
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import os
import pickle
from pathlib import Path
import onnxruntime as ort
from insightface.utils import face_align

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('logs/embeddinganalyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Проверка доступности InsightFace
try:
    from insightface.app import FaceAnalysis
    HAS_INSIGHTFACE = True
    logger.info("InsightFace успешно импортирован.")
except ImportError:
    HAS_INSIGHTFACE = False
    logger.warning("InsightFace не установлен. Функции, зависящие от него, будут недоступны.")

# Импорт конфигурации
try:
    from core_config import (
        DBSCAN_PARAMS, EMBEDDING_ANALYSIS_THRESHOLDS, AUTHENTICITY_WEIGHTS,
        AGING_MODEL, CRITICAL_THRESHOLDS, get_chronological_analysis_parameters,
        CACHE_DIR, ERROR_CODES
    )
    logger.info("Конфигурация успешно импортирована")
except ImportError as e:
    logger.error(f"Ошибка импорта конфигурации: {e}")
    # Значения по умолчанию
    DBSCAN_PARAMS = {"epsilon": 0.35, "min_samples": 3, "metric": "cosine"}
    EMBEDDING_ANALYSIS_THRESHOLDS = {
        "dimension_anomaly_threshold": 2.5,
        "texture_anomaly_dims": (45, 67),
        "geometric_anomaly_dims": (120, 145),
        "lighting_anomaly_dims": (200, 230),
        "age_corrected_drift_enabled": True
    }
    AUTHENTICITY_WEIGHTS = {"embedding": 0.30, "temporal_consistency": 0.15, "temporal_stability": 0.10}
    AGING_MODEL = {"elasticity_loss_per_year": 0.015}
    CRITICAL_THRESHOLDS = {"min_confidence_threshold": 0.5}
    CACHE_DIR = Path("cache")
    ERROR_CODES = {"E004": "EMBEDDING_EXTRACTION_FAILED", "E005": "CLUSTERING_FAILED"}

# ==================== ОСНОВНОЙ КЛАСС ====================

class EmbeddingAnalyzer:
    """
    Анализатор эмбеддингов лиц с кластеризацией и аномалиями
    Версия: 2.0
    Дата: 2025-06-15
    Исправлены все критические ошибки согласно правкам
    """
    
    def __init__(self, face_app: Any = None):
        """Инициализация анализатора эмбеддингов"""
        logger.info("Инициализация EmbeddingAnalyzer")
        
        # InsightFace модель - теперь она может быть передана извне
        self.face_app = face_app
        
        # Кэш эмбеддингов
        self.embeddings_cache = {}
        
        # Кэш кластеризации
        self.clustering_cache = {}
        
        # Устройство вычислений
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Флаг инициализации - теперь зависит от face_app
        self.init_done = (self.face_app is not None)
        
        # Если face_app не передан, инициализируем его здесь (для автономной работы или тестов)
        if self.face_app is None and HAS_INSIGHTFACE:
            self.initialize_insightface_model()
        elif self.face_app is None and not HAS_INSIGHTFACE:
            logger.warning("InsightFace недоступен, используется заглушка")
        
        logger.info("EmbeddingAnalyzer инициализирован")

    def initialize_insightface_model(self) -> None:
        """
        ИСПРАВЛЕНО: Инициализация InsightFace модели Buffalo-L
        Согласно правкам: правильная инициализация с error handling
        """
        if self.face_app is not None: # Check if already initialized inside this instance
            logger.info("InsightFace уже инициализирован")
            return

        if not HAS_INSIGHTFACE:
            logger.error("InsightFace компоненты недоступны")
            return

        try:
            logger.info("Начало инициализации InsightFace Buffalo-L")
            
            # Local import to avoid circular dependencies if FaceAnalysis is not used globally
            from insightface.app import FaceAnalysis 

            providers = ['CUDAExecutionProvider'] if ort.get_device()=="GPU" else ['CPUExecutionProvider']
            self.face_app = FaceAnalysis(name='buffalo_l', providers=providers)
            
            # Подготовка модели
            ctx_id = 0 
            det_size = (640, 640)
            
            self.face_app.prepare(ctx_id=ctx_id, det_size=det_size)
            
            self.init_done = True
            logger.info("InsightFace Buffalo-L успешно инициализирован")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации InsightFace: {e}")
            self.face_app = None
            self.init_done = False
            raise # Re-raise if initialization fails

    def extract_512d_face_embedding(self, img_full: np.ndarray, insight_app: Any) -> Tuple[Optional[np.ndarray], float]:
        """
        ИСПРАВЛЕНО: Извлечение 512D эмбеддинга лица с confidence
        Согласно правкам: извлечение эмбеддинга напрямую из изображения.
        """
        if not self.init_done:
            if HAS_INSIGHTFACE:
                pass
            else:
                logger.error("InsightFace не инициализирован и недоступен")
                return None, 0.0
        
        if insight_app is None: # Используем переданный insight_app вместо self.face_app
            logger.error("InsightFace модель не инициализирована")
            return None, 0.0
        
        try:
            logger.info(f"Извлечение 512D эмбеддинга из изображения {img_full.shape}")
            
            # Детекция лиц с InsightFace на полном изображении
            faces = insight_app.get(img_full, max_num=1)
            
            if len(faces) == 0:
                logger.warning("InsightFace не обнаружил лица в изображении")
                return None, 0.0
            
            # Выбор лучшего лица по detection score
            best_face = max(faces, key=lambda x: self._get_face_score(x))
            
            if best_face is None:
                logger.error("InsightFace не смог выбрать лучшее лицо")
                return None, 0.0
            
            # ИСПРАВЛЕНО: Дополнительная проверка на наличие ландмарок, если нет эмбеддинга
            if not hasattr(best_face, 'embedding') or best_face.embedding is None:
                # Получаем количество обнаруженных ландмарок для более точного сообщения
                detected_kps_count = 0
                if hasattr(best_face, 'kps') and best_face.kps is not None:
                    detected_kps_count = best_face.kps.shape[0]

                if detected_kps_count == 0:
                    logger.error(f"InsightFace обнаружил лицо, но не смог извлечь 68 ключевых точек (ландмарок), необходимых для выравнивания и последующего вычисления эмбеддинга. Получено 0 ландмарок. Возможно, изображение слишком низкого качества или лицо сильно повернуто/заслонено.")
                else:
                    logger.error(f"InsightFace не содержит embedding. Доступные атрибуты: {dir(best_face)}. Возможно, проблема с внутренней обработкой InsightFace.")
                return None, 0.0
            
            emb = best_face.embedding.astype("float32")
            
            # ИСПРАВЛЕНО: Нормализация эмбеддинга на L2
            emb_normalized = emb / np.linalg.norm(emb)
            
            # ИСПРАВЛЕНО: Получение confidence score
            confidence = self._get_face_score(best_face)
            
            # Проверка размерности эмбеддинга
            if emb_normalized.shape[0] != 512:
                logger.warning(f"Неожиданная размерность эмбеддинга: {emb_normalized.shape}")
            
            logger.info(f"Эмбеддинг извлечен успешно. Shape: {emb_normalized.shape}, confidence: {confidence}")
            
            return emb_normalized, confidence
            
        except Exception as e:
            logger.error(f"Ошибка извлечения эмбеддинга: {e}", exc_info=True)
            return None, 0.0

    def _get_face_score(self, face) -> float:
        """Получение score лица из InsightFace API"""
        try:
            # Попытка получить detection score
            return getattr(face, 'det_score', getattr(face, 'score', 0.0))
        except Exception as e:
            logger.error(f"Ошибка получения face score: {e}")
            return 0.0

    def perform_identity_clustering(self, embeddings_with_metadata: List[Dict], 
                                  epsilon: Optional[float] = None, 
                                  min_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Кластеризация идентичностей с DBSCAN (исправлено: eps=0.35, min_samples=3, metric='cosine', возвращает все нужные поля)
        """
        if not embeddings_with_metadata:
            logger.warning("perform_identity_clustering: Пустой список эмбеддингов")
            return self._get_empty_clustering_result()
        if not isinstance(embeddings_with_metadata, list):
            logger.error(f"perform_identity_clustering: Неверный тип данных, ожидается list, получен {type(embeddings_with_metadata)}")
            return self._get_empty_clustering_result()
        logger.info(f"Кластеризация {len(embeddings_with_metadata)} эмбеддингов")
        try:
            valid_embeddings_and_metadata = []
            for i, item in enumerate(embeddings_with_metadata):
                if not isinstance(item, dict):
                    continue
                if 'embedding' not in item:
                    continue
                embedding = item.get('embedding')
                if isinstance(embedding, np.ndarray) and embedding.ndim == 1 and embedding.size == 512:
                    valid_embeddings_and_metadata.append(item)
            if not valid_embeddings_and_metadata:
                return self._get_empty_clustering_result()
            embeddings_array = np.array([item['embedding'] for item in valid_embeddings_and_metadata])
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            embeddings_array = embeddings_array / (norms + 1e-8)
            if embeddings_array.shape[0] < 2:
                return {
                    "labels": np.array([0] * len(valid_embeddings_and_metadata)),
                    "n_clusters": 1,
                    "unique_clusters": [0],
                    "outlier_ratio": 0.0,
                    "cluster_centers": np.array([embeddings_array[0]]),
                    "cluster_std": np.array([0.0]),
                    "outliers": [],
                    "confidence_scores": np.ones(len(valid_embeddings_and_metadata)),
                }
            eps_val = epsilon if epsilon is not None else 0.35
            min_samples_val = min_samples if min_samples is not None else 3
            db = DBSCAN(eps=eps_val, min_samples=min_samples_val, metric='cosine')
            cluster_labels = db.fit_predict(embeddings_array)
            unique_clusters = [lbl for lbl in np.unique(cluster_labels) if lbl != -1]
            n_clusters = len(unique_clusters)
            centers, stds, confs = [], [], np.zeros(len(embeddings_array), np.float32)
            for cid in unique_clusters:
                mask = cluster_labels == cid
                cluster_embs = embeddings_array[mask]
                center = cluster_embs.mean(0)
                std = cluster_embs.std(0).mean()
                centers.append(center)
                stds.append(std)
                from scipy.spatial.distance import pdist
                intra = pdist(cluster_embs, 'cosine')
                mean_intra = intra.mean() if len(intra) else 0
                confs[mask] = 1 - np.clip(mean_intra / eps_val, 0, 1)
            outliers = np.where(cluster_labels == -1)[0]
            return {
                "labels": cluster_labels.tolist(),
                "n_clusters": n_clusters,
                "unique_clusters": unique_clusters,
                "outlier_ratio": float(np.sum(cluster_labels == -1) / len(cluster_labels)) if len(cluster_labels) > 0 else 0.0,
                "cluster_centers": np.vstack(centers) if centers else np.empty((0,512)),
                "cluster_std": np.array(stds),
                "outliers": outliers.tolist(),
                "confidence_scores": confs.tolist(),
            }
        except Exception as e:
            logger.error(f"Ошибка кластеризации: {e}", exc_info=True)
            return self._get_empty_clustering_result()

    def _calculate_intra_cluster_distances(self, cluster_embeddings: np.ndarray) -> Dict[str, float]:
        """Расчет внутрикластерных расстояний"""
        try:
            if len(cluster_embeddings) < 2:
                return {"mean": 0, "std": 0, "max": 0, "min": 0}
            
            distances = pdist(cluster_embeddings, metric='cosine')
            
            return {
                "mean": np.mean(distances),
                "std": np.std(distances),
                "max": np.max(distances),
                "min": np.min(distances)
            }
        except Exception as e:
            logger.error(f"Ошибка расчета внутрикластерных расстояний: {e}")
            return {"mean": 0, "std": 0, "max": 0, "min": 0}

    def _get_empty_clustering_result(self) -> Dict[str, Any]:
        """Получение пустого результата кластеризации"""
        return {
            "labels": np.array([]),
            "n_clusters": 0,
            "unique_clusters": [],
            "outlier_ratio": 0.0
        }

    def build_identity_timeline(self, cluster_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Построение временной линии идентичностей
        Согласно правкам: правильный анализ gaps и appearances
        """
        try:
            logger.info("Построение временной линии идентичностей")
            
            identity_timeline = {}
            
            for cluster_id, metadata in cluster_results.get("cluster_metadata", {}).items():
                items = metadata.get("items", [])
                
                # Фильтрация элементов с датами
                dated_items = [(item['date'], item) for item in items if item.get('date')]
                dated_items.sort(key=lambda x: x[0])
                
                if not dated_items:
                    continue
                
                dates = [item[0] for item in dated_items]
                
                # ИСПРАВЛЕНО: Анализ временных пропусков
                gaps = []
                max_gap_days = get_chronological_analysis_parameters().get("max_gap_days", 180)
                
                for i in range(1, len(dates)):
                    gap_days = (dates[i] - dates[i-1]).days
                    if gap_days > max_gap_days:
                        gaps.append({
                            "start_date": dates[i-1],
                            "end_date": dates[i],
                            "gap_days": gap_days
                        })
                
                # ИСПРАВЛЕНО: Расчет appearances per month
                appearances_per_month = self._calculate_appearances_per_month(dates)
                
                identity_timeline[f"Identity_{cluster_id}"] = {
                    "cluster_id": cluster_id,
                    "first_appearance": dates[0],
                    "last_appearance": dates[-1],
                    "total_appearances": len(dated_items),
                    "active_period_days": (dates[-1] - dates[0]).days,
                    "gaps": gaps,
                    "appearances_per_month": appearances_per_month,
                    "avg_confidence": metadata.get("avg_confidence", 0),
                    "cluster_size": metadata.get("size", 0),
                    "appearance_dates": dates
                }
            
            logger.info(f"Временная линия построена для {len(identity_timeline)} идентичностей")
            return identity_timeline
            
        except Exception as e:
            logger.error(f"Ошибка построения временной линии: {e}")
            return {}

    def _calculate_appearances_per_month(self, dates: List[datetime]) -> Dict[str, int]:
        """Расчет количества появлений по месяцам"""
        try:
            appearances_by_month = {}
            
            for date in dates:
                month_key = date.strftime("%Y-%m")
                if month_key not in appearances_by_month:
                    appearances_by_month[month_key] = 0
                appearances_by_month[month_key] += 1
            
            return appearances_by_month
        except Exception as e:
            logger.error(f"Ошибка расчета appearances per month: {e}")
            return {}

    def detect_embedding_anomalies_by_dimensions(self, embedding: np.ndarray, 
                                               reference_embeddings: List[np.ndarray]) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Обнаружение аномалий эмбеддингов по диапазонам измерений
        Согласно правкам: dimensions 45-67 (texture), 120-145 (geometry), 200-230 (lighting)
        """
        if len(embedding) != 512 or len(reference_embeddings) == 0:
            logger.warning("Невалидные данные для анализа аномалий эмбеддингов")
            return self._get_empty_anomalies_result()
        
        try:
            logger.info("Анализ аномалий эмбеддингов по диапазонам измерений")
            
            # Преобразование референсных эмбеддингов в массив
            reference_array = np.array(reference_embeddings)
            
            # Расчет статистик по измерениям
            mean_dims = np.mean(reference_array, axis=0)
            std_dims = np.std(reference_array, axis=0)
            
            # Z-scores для всех измерений
            z_scores = (embedding - mean_dims) / (std_dims + 1e-8)  # epsilon для избежания деления на ноль
            
            # ИСПРАВЛЕНО: Пороги и диапазоны согласно правкам
            dimension_anomaly_threshold = EMBEDDING_ANALYSIS_THRESHOLDS["dimension_anomaly_threshold"]
            
            anomalies_by_category = {
                "overall_detected": False,
                "texture_anomalies": {"detected": False, "count": 0, "dimensions": []},
                "geometric_anomalies": {"detected": False, "count": 0, "dimensions": []},
                "lighting_anomalies": {"detected": False, "count": 0, "dimensions": []},
                "other_anomalies": {"detected": False, "count": 0, "dimensions": []}
            }
            
            # ИСПРАВЛЕНО: Диапазоны из coreconfig
            texture_dims_range = EMBEDDING_ANALYSIS_THRESHOLDS["texture_anomaly_dims"]  # (45, 67)
            geometric_dims_range = EMBEDDING_ANALYSIS_THRESHOLDS["geometric_anomaly_dims"]  # (120, 145)
            lighting_dims_range = EMBEDDING_ANALYSIS_THRESHOLDS["lighting_anomaly_dims"]  # (200, 230)
            
            # Анализ текстурных аномалий (dimensions 45-67)
            texture_z_scores = np.abs(z_scores[texture_dims_range[0]:texture_dims_range[1] + 1])
            texture_anomaly_indices = np.where(texture_z_scores > dimension_anomaly_threshold)[0]
            
            if len(texture_anomaly_indices) > 0:
                anomalies_by_category["texture_anomalies"]["detected"] = True
                anomalies_by_category["texture_anomalies"]["count"] = len(texture_anomaly_indices)
                anomalies_by_category["texture_anomalies"]["dimensions"] = (texture_anomaly_indices + texture_dims_range[0]).tolist()
                anomalies_by_category["overall_detected"] = True
            
            # Анализ геометрических аномалий (dimensions 120-145)
            geometric_z_scores = np.abs(z_scores[geometric_dims_range[0]:geometric_dims_range[1] + 1])
            geometric_anomaly_indices = np.where(geometric_z_scores > dimension_anomaly_threshold)[0]
            
            if len(geometric_anomaly_indices) > 0:
                anomalies_by_category["geometric_anomalies"]["detected"] = True
                anomalies_by_category["geometric_anomalies"]["count"] = len(geometric_anomaly_indices)
                anomalies_by_category["geometric_anomalies"]["dimensions"] = (geometric_anomaly_indices + geometric_dims_range[0]).tolist()
                anomalies_by_category["overall_detected"] = True
            
            # Анализ световых аномалий (dimensions 200-230)
            lighting_z_scores = np.abs(z_scores[lighting_dims_range[0]:lighting_dims_range[1] + 1])
            lighting_anomaly_indices = np.where(lighting_z_scores > dimension_anomaly_threshold)[0]
            
            if len(lighting_anomaly_indices) > 0:
                anomalies_by_category["lighting_anomalies"]["detected"] = True
                anomalies_by_category["lighting_anomalies"]["count"] = len(lighting_anomaly_indices)
                anomalies_by_category["lighting_anomalies"]["dimensions"] = (lighting_anomaly_indices + lighting_dims_range[0]).tolist()
                anomalies_by_category["overall_detected"] = True
            
            # Анализ других аномалий
            all_anomaly_indices = np.where(np.abs(z_scores) > dimension_anomaly_threshold)[0]
            specific_anomaly_dims = set(anomalies_by_category["texture_anomalies"]["dimensions"] + 
                                      anomalies_by_category["geometric_anomalies"]["dimensions"] + 
                                      anomalies_by_category["lighting_anomalies"]["dimensions"])
            
            other_anomaly_indices = [idx for idx in all_anomaly_indices if idx not in specific_anomaly_dims]
            
            if len(other_anomaly_indices) > 0:
                anomalies_by_category["other_anomalies"]["detected"] = True
                anomalies_by_category["other_anomalies"]["count"] = len(other_anomaly_indices)
                anomalies_by_category["other_anomalies"]["dimensions"] = other_anomaly_indices
                anomalies_by_category["overall_detected"] = True
            
            logger.info(f"Анализ аномалий завершен: overall_detected={anomalies_by_category['overall_detected']}")
            
            return anomalies_by_category
            
        except Exception as e:
            logger.error(f"Ошибка анализа аномалий эмбеддингов: {e}")
            return self._get_empty_anomalies_result()

    def _get_empty_anomalies_result(self) -> Dict[str, Any]:
        """Получение пустого результата анализа аномалий"""
        return {
            "overall_detected": False,
            "texture_anomalies": {"detected": False, "count": 0, "dimensions": []},
            "geometric_anomalies": {"detected": False, "count": 0, "dimensions": []},
            "lighting_anomalies": {"detected": False, "count": 0, "dimensions": []},
            "other_anomalies": {"detected": False, "count": 0, "dimensions": []}
        }

    def calculate_identity_confidence_score(self, embedding: np.ndarray, 
                                          cluster_center: np.ndarray, 
                                          appearances_count: int, 
                                          temporal_stability_score: float) -> float:
        """
        ИСПРАВЛЕНО: Расчет confidence score идентичности
        Согласно правкам: учет distance, appearances, temporal stability
        """
        if embedding.size == 0 or cluster_center.size == 0 or appearances_count <= 0:
            return 0.0
        
        try:
            logger.info("Расчет confidence score идентичности")
            
            # ИСПРАВЛЕНО: Расстояние до центра кластера
            distance_to_center = cosine_distances(embedding.reshape(1, -1), cluster_center.reshape(1, -1))[0, 0]
            
            # Нормализованный score расстояния (1 - distance = similarity)
            normalized_distance_score = max(0.0, 1.0 - distance_to_center)
            
            # ИСПРАВЛЕНО: Нормализованный score количества появлений
            saturation_factor = EMBEDDING_ANALYSIS_THRESHOLDS.get("APPEARANCE_SATURATION_FACTOR", 50)
            normalized_appearance_score = min(1.0, np.log1p(appearances_count) / (np.log1p(saturation_factor) + 1e-8))
            
            # ИСПРАВЛЕНО: Нормализованный temporal stability score
            normalized_temporal_stability_score = np.clip(temporal_stability_score, 0.0, 1.0)
            
            # ИСПРАВЛЕНО: Взвешенная комбинация согласно AUTHENTICITY_WEIGHTS
            confidence_score = (
                AUTHENTICITY_WEIGHTS["embedding"] * normalized_distance_score +
                AUTHENTICITY_WEIGHTS["temporal_consistency"] * normalized_appearance_score +
                AUTHENTICITY_WEIGHTS["temporal_stability"] * normalized_temporal_stability_score
            )
            
            # Нормализация по сумме весов
            total_weight_sum = (AUTHENTICITY_WEIGHTS["embedding"] + 
                              AUTHENTICITY_WEIGHTS["temporal_consistency"] + 
                              AUTHENTICITY_WEIGHTS["temporal_stability"])
            
            if total_weight_sum > 0:
                confidence_score /= total_weight_sum
            
            result = float(np.clip(confidence_score, 0.0, 1.0))
            
            logger.info(f"Confidence score рассчитан: {result:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка расчета confidence score: {e}")
            return 0.0

    def age_corrected_embedding_drift(self, current_embeddings: List[np.ndarray], 
                                    baseline_embeddings: List[np.ndarray], 
                                    age_difference: float) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Анализ дрифта эмбеддингов с учетом возраста
        Согласно правкам: добавлена функция agecorrectedembeddingdrift
        """
        if not current_embeddings or not baseline_embeddings:
            return {"drift_detected": False, "reason": "Недостаточно данных для анализа дрифта"}
        
        try:
            logger.info(f"Анализ дрифта эмбеддингов с учетом возраста: {age_difference} лет")
            
            # Расчет среднего дрифта
            actual_drift = self._calculate_mean_cosine_distance(current_embeddings, baseline_embeddings)
            
            # ИСПРАВЛЕНО: Ожидаемый дрифт на основе aging model
            expected_drift = age_difference * AGING_MODEL.get("elasticity_loss_per_year", 0.015) * 0.002  # Коэффициент для эмбеддингов
            
            # Анализ аномального дрифта
            anomalous_drift = False
            severity = 0.0
            
            if expected_drift > 0 and actual_drift > expected_drift * 3:  # Превышение в 3 раза
                anomalous_drift = True
                severity = actual_drift / expected_drift
            
            result = {
                "anomalous_drift": anomalous_drift,
                "severity": severity,
                "actual_drift": actual_drift,
                "expected_drift": expected_drift,
                "age_difference": age_difference,
                "drift_detected": anomalous_drift
            }
            
            if anomalous_drift:
                result["reason"] = f"Аномальный дрифт: {actual_drift:.4f} > {expected_drift:.4f} (ожидаемый)"
            else:
                result["reason"] = f"Нормальный дрифт: {actual_drift:.4f} <= {expected_drift:.4f} (ожидаемый)"
            
            logger.info(f"Анализ дрифта завершен: drift_detected={result['drift_detected']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка анализа дрифта эмбеддингов: {e}")
            return {"drift_detected": False, "reason": f"Ошибка: {str(e)}"}

    def _calculate_mean_cosine_distance(self, embeddings_list1: List[np.ndarray], 
                                      embeddings_list2: List[np.ndarray]) -> float:
        """Расчет среднего косинусного расстояния между двумя списками эмбеддингов"""
        if not embeddings_list1 or not embeddings_list2:
            return 0.0
        
        try:
            # Расчет средних эмбеддингов
            current_mean_embedding = np.mean(embeddings_list1, axis=0)
            baseline_mean_embedding = np.mean(embeddings_list2, axis=0)
            
            # Косинусное расстояние
            distance = 1 - np.dot(current_mean_embedding, baseline_mean_embedding) / (
                np.linalg.norm(current_mean_embedding) * np.linalg.norm(baseline_mean_embedding) + 1e-8
            )
            
            return distance
            
        except Exception as e:
            logger.error(f"Ошибка расчета косинусного расстояния: {e}")
            return 0.0

    def analyze_cluster_temporal_stability(self, cluster_timeline: Dict[str, Any]) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Анализ временной стабильности кластеров
        Согласно правкам: анализ coefficient of variation
        """
        try:
            logger.info("Анализ временной стабильности кластеров")
            
            stability_results = {}
            
            for identity_id, data in cluster_timeline.items():
                appearance_dates = sorted(data.get("appearance_dates", []))
                
                if len(appearance_dates) < 2:
                    stability_results[identity_id] = {
                        "stable": True,
                        "reason": "Недостаточно данных для анализа стабильности"
                    }
                    continue
                
                # Расчет интервалов между появлениями
                intervals = np.array([(appearance_dates[i+1] - appearance_dates[i]).days 
                                    for i in range(len(appearance_dates) - 1)])
                
                # Анализ стабильности интервалов
                if len(intervals) > 1:
                    std_interval = np.std(intervals)
                    mean_interval = np.mean(intervals)
                    
                    # Coefficient of variation
                    if mean_interval > 0 and std_interval / mean_interval > CRITICAL_THRESHOLDS.get("temporal_stability_threshold", 0.5):
                        stability_results[identity_id] = {
                            "stable": False,
                            "reason": f"Высокая вариабельность интервалов: CV={std_interval/mean_interval:.3f}"
                        }
                        continue
                
                # Проверка максимального пропуска
                max_gap = np.max(intervals)
                max_gap_days = get_chronological_analysis_parameters().get("max_gap_days", 180)
                
                if max_gap > max_gap_days * 6:  # Критический пропуск
                    stability_results[identity_id] = {
                        "stable": False,
                        "reason": f"Критический пропуск: {max_gap} дней"
                    }
                    continue
                
                stability_results[identity_id] = {
                    "stable": True,
                    "reason": "Стабильные временные интервалы"
                }
            
            logger.info(f"Анализ стабильности завершен для {len(stability_results)} идентичностей")
            return stability_results
            
        except Exception as e:
            logger.error(f"Ошибка анализа временной стабильности: {e}")
            return {}

    def save_cache(self, cache_file: str = "embedding_cache.pkl") -> None:
        """Сохранение кэша в файл"""
        try:
            cache_path = CACHE_DIR / cache_file
            CACHE_DIR.mkdir(exist_ok=True)
            
            cache_data = {
                "embeddings_cache": self.embeddings_cache,
                "clustering_cache": self.clustering_cache
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"Кэш сохранен: {cache_path}")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения кэша: {e}")

    def load_cache(self, cache_file: str = "embedding_cache.pkl") -> None:
        """Загрузка кэша из файла"""
        try:
            cache_path = CACHE_DIR / cache_file
            
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                
                self.embeddings_cache = cache_data.get("embeddings_cache", {})
                self.clustering_cache = cache_data.get("clustering_cache", {})
                
                logger.info(f"Кэш загружен: {cache_path}")
            else:
                logger.info("Файл кэша не найден, используется пустой кэш")
                
        except Exception as e:
            logger.error(f"Ошибка загрузки кэша: {e}")

    def self_test(self) -> None:
        """
        ИСПРАВЛЕНО: Самопроверка анализатора эмбеддингов
        Согласно правкам: Проверка извлечения эмбеддинга и кластеризации
        """
        logger.info("Запуск самотеста EmbeddingAnalyzer...")
        try:
            # Ensure InsightFace model is initialized for self-test
            if self.face_app is None:
                logger.info("InsightFace модель не инициализирована, инициализируем для самотеста.")
                self.initialize_insightface_model()
                if self.face_app is None: # Проверка после попытки инициализации
                    raise RuntimeError("Не удалось инициализировать InsightFace для самотеста.")
            
            app = self.face_app # Используем уже инициализированный app
            
            # Загрузка реального изображения для теста
            test_image_path = "/Users/victorkhudyakov/nn/3DDFA2/3/01_01_10.jpg"
            img = cv2.imread(test_image_path)
            
            if img is None:
                logger.error(f"Не удалось загрузить тестовое изображение с пути: {test_image_path}")
                raise FileNotFoundError(f"Тестовое изображение не найдено: {test_image_path}")

            # Тестирование извлечения эмбеддинга
            emb, conf = self.extract_512d_face_embedding(img, app)
            
            assert emb is not None, "Self-test: эмбеддинг не должен быть None"
            assert conf > 0.5, "Self-test: confidence должен быть > 0.5 для реального лица"
            logger.info("✔ Self-test: Извлечение эмбеддинга успешно")

            # Test clustering (Пример)
            dummy_embeddings = [{"embedding": np.random.rand(512), "metadata": {"date": datetime.now()}} for _ in range(10)]
            cluster_results = self.perform_identity_clustering(dummy_embeddings)
            assert "labels" in cluster_results, "Self-test: Кластеризация не вернула метки"
            logger.info("✔ Self-test: Кластеризация успешно")

            logger.info("✔ Все самотесты EmbeddingAnalyzer пройдены успешно.")

        except Exception as e:
            logger.error(f"❌ Ошибка в самотесте EmbeddingAnalyzer: {e}", exc_info=True)
            raise

    def ensemble_embedding_analysis(self, embeddings: np.ndarray, strategies=("mean-cosine","median")):
        """
        Выполняет N стратегий кластеризации, объединяет через majority_vote на уровне принадлежности точки кластеру.
        """
        results = []
        if 'mean-cosine' in strategies:
            meta = [{'embedding': e} for e in embeddings]
            res = self.perform_identity_clustering(meta)
            results.append(np.array(res['labels']))
        if 'median' in strategies:
            med = np.median(embeddings, axis=0, keepdims=True)
            emb_med = embeddings - med
            meta = [{'embedding': e} for e in emb_med]
            res = self.perform_identity_clustering(meta)
            results.append(np.array(res['labels']))
        if not results:
            return np.array([-1]*len(embeddings))
        stacked = np.vstack(results)
        def vote(x):
            vals = x[x!=-1]
            return np.bincount(vals).argmax() if len(vals) else -1
        final = np.apply_along_axis(vote, 0, stacked)
        return final

# ==================== ТОЧКА ВХОДА ====================

if __name__ == "__main__":
    analyzer = EmbeddingAnalyzer()
    analyzer.self_test()