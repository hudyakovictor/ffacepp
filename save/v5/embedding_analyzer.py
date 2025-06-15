# embedding_analyzer.py
# Анализ эмбеддингов лиц и кластеризация личностей

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from scipy.spatial.distance import pdist, squareform
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import os

try:
    import insightface
    from insightface.app import FaceAnalysis
except ImportError:
    raise ImportError("Библиотека 'insightface' не установлена. Пожалуйста, установите ее для работы с EmbeddingAnalyzer.")

from core_config import (
    INSIGHT_MODEL, DBSCAN_EPSILON, DBSCAN_MIN_SAMPLES, DBSCAN_METRIC, INSIGHT_FACE_DET_THRESHOLD,
    INSIGHT_CTX_ID, INSIGHT_FACE_DET_SIZE, EMBEDDING_ANALYSIS_THRESHOLDS, EMBEDDING_DRIFT_THRESHOLDS,
    AUTHENTICITY_WEIGHTS, \
    DISTANCE_METRIC_DEFAULT, AGING_MODEL_ENABLED, TEMPORAL_STABILITY_THRESHOLD, CONSECUTIVE_APPEARANCES_THRESHOLD,
    get_chronological_analysis_parameters
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmbeddingAnalyzer:
    def __init__(self):
        self.face_app = None
        self.embeddings_cache = {}

        # InsightFace модель инициализируется при первом использовании или по требованию.

    def initialize_insightface_model(self):
        """Инициализация InsightFace (Buffalo_L) с оптимизацией для портретов"""
        try:
            self.face_app = FaceAnalysis(name='buffalo_l')
            self.face_app.prepare(ctx_id=INSIGHT_CTX_ID, det_size=INSIGHT_FACE_DET_SIZE)
            print("InsightFace модель Buffalo_L инициализирована")
        except Exception as e:
            logging.error(f"Ошибка инициализации InsightFace: {e}")
            raise RuntimeError(f"Не удалось инициализировать InsightFace модель: {e}")

    def extract_512d_face_embedding(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Извлекает 512-мерный эмбеддинг - уникальный математический отпечаток лица"""
        if self.face_app is None:
            raise RuntimeError("InsightFace модель не инициализирована. Извлечение эмбеддинга невозможно.")

        try:
            logging.info(f"В extract_512d_face_embedding (перед app.get): image shape={image.shape}, dtype={image.dtype}")
            faces = self.face_app.get(image, max_num=1)
            
            if len(faces) == 0:
                logging.warning("InsightFace: Лица не обнаружены. Возвращаем пустые значения.")
                return np.array([]), 0.0

            # Берем лицо с наибольшей уверенностью
            best_face = max(faces, key=lambda x: self._get_face_score(x))

            if best_face is None:
                logging.error("InsightFace: Не удалось выбрать лучшее лицо из обнаруженных.")
                return np.array([]), 0.0

            # Извлечение и нормализация эмбеддинга
            if not hasattr(best_face, 'embedding') or best_face.embedding is None:
                logging.error(f"InsightFace: Обнаруженное лицо не содержит атрибута 'embedding'. best_face: {best_face}")
                return np.array([]), 0.0

            embedding = best_face.embedding
            
            # Проверка на нормализованный эмбеддинг
            if hasattr(best_face, 'normed_embedding') and best_face.normed_embedding is not None:
                embedding_normalized = best_face.normed_embedding
            else:
                embedding_normalized = embedding / (np.linalg.norm(embedding) + 1e-8)

            # ИСПРАВЛЕНО: Совместимость с разными версиями InsightFace API
            confidence = self._get_face_score(best_face)

            logging.info(f"В extract_512d_face_embedding: embedding_normalized Shape={embedding_normalized.shape}, confidence={confidence}")
            return embedding_normalized, confidence

        except Exception as e:
            logging.error(f"Ошибка извлечения эмбеддинга: {e}", exc_info=True)
            return np.array([]), 0.0

    def _get_face_score(self, face) -> float:
        """Получает score лица с учетом разных версий API"""
        # ИСПРАВЛЕНО: Совместимость с разными версиями InsightFace
        return getattr(face, 'det_score', getattr(face, 'score', 0.0))

    def calculate_embedding_distances_matrix(self, embeddings_list: List[np.ndarray]) -> np.ndarray:
        """Вычисляет матрицу косинусных расстояний между всеми эмбеддингами"""

        if len(embeddings_list) == 0:
            return np.array([])
        
        embeddings_array = np.array(embeddings_list)
        # Нормализация эмбеддингов для корректного косинусного расстояния
        normalized_embeddings = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        
        # Косинусное сходство через скалярное произведение
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        # Преобразование в расстояние
        distance_matrix = 1 - similarity_matrix
        
        return distance_matrix

    def perform_identity_clustering(self, embeddings_with_metadata: List[Dict], epsilon: Optional[float] = None, min_samples: Optional[int] = None) -> Dict:
        """Кластеризация DBSCAN для выявления уникальных личностей"""
        
        # Проверка входных данных
        if not embeddings_with_metadata:
            logging.warning("perform_identity_clustering: Нет данных для кластеризации.")
            return {}
        
        # Дополнительная проверка типов данных
        if not isinstance(embeddings_with_metadata, list):
            logging.error(f"perform_identity_clustering: Ожидался список, получен {type(embeddings_with_metadata)}")
            return {}
        
        # Проверяем каждый элемент
        for i, item in enumerate(embeddings_with_metadata):
            if not isinstance(item, dict):
                logging.error(f"perform_identity_clustering: Элемент {i} не является словарем: {type(item)}")
                return {}
            if 'embedding' not in item:
                logging.error(f"perform_identity_clustering: Элемент {i} не содержит ключ 'embedding'")
                return {}

        # Извлечение и валидация эмбеддингов вместе с метаданными
        valid_embeddings_and_metadata = []
        for item in embeddings_with_metadata:
            embedding = item.get('embedding')
            if isinstance(embedding, np.ndarray) and embedding.ndim == 1 and embedding.size == 512:
                valid_embeddings_and_metadata.append(item) # Сохраняем весь элемент
            else:
                file_path = item.get('file_path', 'Unknown')
                logging.warning(f"perform_identity_clustering: Пропущен некорректный эмбеддинг для файла {file_path}. Тип: {type(embedding)}, Размеры: {embedding.shape if isinstance(embedding, np.ndarray) else 'N/A'}.")

        if not valid_embeddings_and_metadata:
            logging.warning("perform_identity_clustering: После валидации не осталось действительных эмбеддингов.")
            return {}

        # Извлекаем эмбеддинги для кластеризации из валидных данных
        embeddings_array = np.array([item['embedding'] for item in valid_embeddings_and_metadata])

        # Явная L2-нормализация эмбеддингов
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        if norms.size == 0:
            logging.warning("perform_identity_clustering: Нормы эмбеддингов пусты. Возвращаем пустые результаты.")
            return {}

        # DBSCAN требует минимум 2 образца
        if embeddings_array.shape[0] < 2:
            logging.warning(f"perform_identity_clustering: Недостаточно эмбеддингов для кластеризации ({embeddings_array.shape[0]} < 2). Возвращаем пустые результаты.")
            return {
                'cluster_labels': np.array([]),
                'n_clusters': 0,
                'n_noise': embeddings_array.shape[0],
                'cluster_centers': {},
                'cluster_metadata': {},
                'outliers': valid_embeddings_and_metadata # Все валидные элементы считаются выбросами
            }

        # Параметры DBSCAN
        eps_val = epsilon if epsilon is not None else DBSCAN_EPSILON
        min_samples_val = min_samples if min_samples is not None else DBSCAN_MIN_SAMPLES

        dbscan = DBSCAN(
            eps=eps_val,
            min_samples=min_samples_val,
            metric=DBSCAN_METRIC
        )

        cluster_labels = dbscan.fit_predict(embeddings_array)

        # Анализ кластеров
        unique_clusters = set(cluster_labels)
        unique_clusters.discard(-1)  # Удаляем шум

        cluster_results = {
            'cluster_labels': cluster_labels,
            'n_clusters': len(unique_clusters),
            'n_noise': list(cluster_labels).count(-1),
            'cluster_centers': {},
            'cluster_metadata': {},
            'outliers': []
        }

        # Вычисление центров кластеров и метаданных
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            # Используем valid_embeddings_and_metadata для получения элементов кластера
            cluster_items = [valid_embeddings_and_metadata[i] for i in range(len(valid_embeddings_and_metadata)) if cluster_mask[i]]

            # Центр кластера
            cluster_center = np.mean(embeddings_array[cluster_mask], axis=0)
            cluster_results['cluster_centers'][str(cluster_id)] = cluster_center # Преобразуем cluster_id в строку

            # Метаданные кластера
            dates = [item.get('date') for item in cluster_items if item.get('date')]
            confidences = [item.get('confidence', 0) for item in cluster_items]

            cluster_results['cluster_metadata'][str(cluster_id)] = { # Преобразуем cluster_id в строку
                'size': len(cluster_items),
                'avg_confidence': np.mean(confidences) if confidences else 0.0,
                'first_appearance': min(dates) if dates else None,
                'last_appearance': max(dates) if dates else None,
                'intra_cluster_distances': self._calculate_intra_cluster_distances(embeddings_array[cluster_mask]),
                'items': cluster_items
            }

        # Выбросы (шум) - используем valid_embeddings_and_metadata
        noise_mask = cluster_labels == -1
        cluster_results['outliers'] = [valid_embeddings_and_metadata[i] for i in range(len(valid_embeddings_and_metadata)) if noise_mask[i]]

        return cluster_results

    def _calculate_intra_cluster_distances(self, cluster_embeddings: np.ndarray) -> Dict:
        """Вычисляет статистики расстояний внутри кластера"""

        if len(cluster_embeddings) < 2:
            return {'mean': 0, 'std': 0, 'max': 0, 'min': 0}

        distances = pdist(cluster_embeddings, metric='cosine')

        return {
            'mean': np.mean(distances),
            'std': np.std(distances),
            'max': np.max(distances),
            'min': np.min(distances)
        }

    def build_identity_timeline(self, cluster_results: Dict) -> Dict:
        """Строит временную линию появления каждой идентифицированной личности"""

        identity_timeline = {}

        for cluster_id, metadata in cluster_results.get('cluster_metadata', {}).items():
            items = metadata.get('items', [])

            # Сортировка по дате
            dated_items = [(item['date'], item) for item in items if item.get('date')]
            dated_items.sort(key=lambda x: x[0])

            if not dated_items:
                continue

            dates = [item[0] for item in dated_items]

            # Анализ временных промежутков
            gaps = []
            for i in range(1, len(dates)):
                gap_days = (dates[i] - dates[i-1]).days
                if gap_days > get_chronological_analysis_parameters()['max_gap_days']:  # Разрыв больше месяца
                    gaps.append({
                        'start_date': dates[i-1],
                        'end_date': dates[i],
                        'gap_days': gap_days
                    })

            identity_timeline[f'Identity_{cluster_id}'] = {
                'cluster_id': cluster_id,
                'first_appearance': dates[0],
                'last_appearance': dates[-1],
                'total_appearances': len(dated_items),
                'active_period_days': (dates[-1] - dates[0]).days,
                'gaps': gaps,
                'appearances_per_month': self._calculate_appearances_per_month(dates),
                'avg_confidence': metadata.get('avg_confidence', 0),
                'cluster_size': metadata.get('size', 0),
                'appearance_dates': dates
            }

        return identity_timeline

    def _calculate_appearances_per_month(self, dates: List[datetime]) -> Dict:
        """Вычисляет количество появлений по месяцам"""

        appearances_by_month = {}

        for date in dates:
            month_key = date.strftime('%Y-%m')
            if month_key not in appearances_by_month:
                appearances_by_month[month_key] = 0
            appearances_by_month[month_key] += 1

        return appearances_by_month

    def detect_embedding_anomalies_by_dimensions(self, embedding: np.ndarray, reference_embeddings: List[np.ndarray]) -> Dict:
        """Выявляет аномалии в специфических измерениях эмбеддинга:
        dimensions_45_67: текстурные аномалии (маски)
        dimensions_120_145: геометрические искажения
        dimensions_200_230: световые аномалии материалов
        """

        if len(embedding) != 512 or len(reference_embeddings) == 0:
            logger.warning("Недостаточно данных для детекции аномалий по измерениям эмбеддинга.")
            return {}
        
        reference_array = np.array(reference_embeddings)
        
        # Расчет среднего и стандартного отклонения для каждого измерения
        mean_dims = np.mean(reference_array, axis=0)
        std_dims = np.std(reference_array, axis=0)
        
        # Z-score для каждого измерения эмбеддинга
        # Добавляем epsilon для избежания деления на ноль, если std_dims близко к нулю
        z_scores = (embedding - mean_dims) / (std_dims + 1e-8)
        
        # Порог для аномалий в конкретных измерениях
        dimension_anomaly_threshold = EMBEDDING_ANALYSIS_THRESHOLDS['DIMENSION_ANOMALY_THRESHOLD']
        
        anomalies_by_category = {
            'overall_detected': False,
            'texture_anomalies': {'detected': False, 'count': 0, 'dimensions': []},
            'geometric_anomalies': {'detected': False, 'count': 0, 'dimensions': []},
            'lighting_anomalies': {'detected': False, 'count': 0, 'dimensions': []},
            'other_anomalies': {'detected': False, 'count': 0, 'dimensions': []}
        }

        # Получение диапазонов измерений из core_config
        texture_dims_range = EMBEDDING_ANALYSIS_THRESHOLDS['TEXTURE_ANOMALY_DIMS']
        geometric_dims_range = EMBEDDING_ANALYSIS_THRESHOLDS['GEOMETRIC_ANOMALY_DIMS']
        lighting_dims_range = EMBEDDING_ANALYSIS_THRESHOLDS['LIGHTING_ANOMALY_DIMS']

        # Анализ текстурных аномалий (dimensions 45-67)
        texture_z_scores = np.abs(z_scores[texture_dims_range[0]:texture_dims_range[1] + 1])
        texture_anomaly_indices = np.where(texture_z_scores > dimension_anomaly_threshold)[0]
        if len(texture_anomaly_indices) > 0:
            anomalies_by_category['texture_anomalies']['detected'] = True
            anomalies_by_category['texture_anomalies']['count'] = len(texture_anomaly_indices)
            anomalies_by_category['texture_anomalies']['dimensions'] = (texture_anomaly_indices + texture_dims_range[0]).tolist()
            anomalies_by_category['overall_detected'] = True

        # Анализ геометрических искажений (dimensions 120-145)
        geometric_z_scores = np.abs(z_scores[geometric_dims_range[0]:geometric_dims_range[1] + 1])
        geometric_anomaly_indices = np.where(geometric_z_scores > dimension_anomaly_threshold)[0]
        if len(geometric_anomaly_indices) > 0:
            anomalies_by_category['geometric_anomalies']['detected'] = True
            anomalies_by_category['geometric_anomalies']['count'] = len(geometric_anomaly_indices)
            anomalies_by_category['geometric_anomalies']['dimensions'] = (geometric_anomaly_indices + geometric_dims_range[0]).tolist()
            anomalies_by_category['overall_detected'] = True

        # Анализ световых аномалий материалов (dimensions 200-230)
        lighting_z_scores = np.abs(z_scores[lighting_dims_range[0]:lighting_dims_range[1] + 1])
        lighting_anomaly_indices = np.where(lighting_z_scores > dimension_anomaly_threshold)[0]
        if len(lighting_anomaly_indices) > 0:
            anomalies_by_category['lighting_anomalies']['detected'] = True
            anomalies_by_category['lighting_anomalies']['count'] = len(lighting_anomaly_indices)
            anomalies_by_category['lighting_anomalies']['dimensions'] = (lighting_anomaly_indices + lighting_dims_range[0]).tolist()
            anomalies_by_category['overall_detected'] = True

        # Общая детекция аномалий вне специфических диапазонов (если есть)
        all_anomaly_indices = np.where(np.abs(z_scores) > dimension_anomaly_threshold)[0]
        specific_anomaly_dims = set(anomalies_by_category['texture_anomalies']['dimensions'] + \
                                    anomalies_by_category['geometric_anomalies']['dimensions'] + \
                                    anomalies_by_category['lighting_anomalies']['dimensions'])
        
        other_anomaly_indices = [idx for idx in all_anomaly_indices if idx not in specific_anomaly_dims]
        if len(other_anomaly_indices) > 0:
            anomalies_by_category['other_anomalies']['detected'] = True
            anomalies_by_category['other_anomalies']['count'] = len(other_anomaly_indices)
            anomalies_by_category['other_anomalies']['dimensions'] = other_anomaly_indices
            anomalies_by_category['overall_detected'] = True

        return anomalies_by_category

    def calculate_identity_confidence_score(self, embedding: np.ndarray, cluster_center: np.ndarray, 
                                            appearances_count: int, temporal_stability_score: float) -> float:
        """Вычисляет уверенность в идентификации личности:
        учитывает расстояние до центра кластера, количество появлений, стабильность личности во времени.
        """
        if embedding.size == 0 or cluster_center.size == 0 or appearances_count <= 0:
            return 0.0

        # Косинусное расстояние до центра кластера (меньше = лучше)
        distance_to_center = cosine_distances(embedding.reshape(1, -1), cluster_center.reshape(1, -1))[0][0]
        
        # Нормализация расстояния: 0 (идеальное совпадение) -> 1.0 (максимальная уверенность по расстоянию)
        # 1 (полное различие) -> 0.0 (минимальная уверенность по расстоянию)
        normalized_distance_score = max(0.0, 1.0 - distance_to_center)

        # Влияние количества появлений (больше = лучше)
        # Используем логарифмическую шкалу для уменьшения влияния очень больших чисел,
        # так как после определенного числа появлений прирост уверенности замедляется.
        # Например, np.log1p(appearances_count) / np.log1p(50) - предположим, что 50 появлений дают насыщение
        # Можно настроить коэффициент насыщения через конфигурацию
        saturation_factor = EMBEDDING_ANALYSIS_THRESHOLDS.get('APPEARANCE_SATURATION_FACTOR', 50)
        normalized_appearance_score = min(1.0, np.log1p(appearances_count) / (np.log1p(saturation_factor) + 1e-8))

        # Влияние временной стабильности (выше = лучше)
        # temporal_stability_score уже должна быть нормализована в диапазоне [0, 1]
        normalized_temporal_stability_score = np.clip(temporal_stability_score, 0.0, 1.0)

        # Комбинированный балл уверенности с весами из конфигурации
        confidence_score = (
            AUTHENTICITY_WEIGHTS['embedding'] * normalized_distance_score + # Вес для сходства эмбеддингов
            AUTHENTICITY_WEIGHTS['temporal_consistency'] * normalized_appearance_score + # Используем этот вес для количества появлений
            AUTHENTICITY_WEIGHTS['temporal_stability'] * normalized_temporal_stability_score # Новый вес для временной стабильности
        )

        # Нормализация итогового балла уверенности
        total_weight_sum = (
            AUTHENTICITY_WEIGHTS['embedding'] + 
            AUTHENTICITY_WEIGHTS['temporal_consistency'] + # Суммируем веса
            AUTHENTICITY_WEIGHTS['temporal_stability']
        )
        
        if total_weight_sum > 0:
            confidence_score /= total_weight_sum

        return float(np.clip(confidence_score, 0.0, 1.0))

    def analyze_cluster_temporal_stability(self, cluster_timeline: Dict) -> Dict:
        """Анализирует временную стабильность кластера"""
        stability_results = {}

        for identity_id, data in cluster_timeline.items():
            appearance_dates = sorted(data['appearance_dates'])
            if len(appearance_dates) < 2:
                stability_results[identity_id] = {'stable': True, 'reason': 'Недостаточно данных для анализа.'}
                continue

            intervals = np.array([(appearance_dates[i+1] - appearance_dates[i]).days for i in range(len(appearance_dates) - 1)])
            
            # Отсутствие больших разрывов
            max_gap = np.max(intervals)
            if max_gap > get_chronological_analysis_parameters()['max_gap_days']: # Если есть разрыв более 6 месяцев
                stability_results[identity_id] = {'stable': False, 'reason': f'Большой временной разрыв ({max_gap} дней)'}
                continue

            # Низкая дисперсия интервалов (регулярность)
            if len(intervals) > 1:
                std_interval = np.std(intervals)
                mean_interval = np.mean(intervals)
                if mean_interval > 0 and (std_interval / mean_interval) > TEMPORAL_STABILITY_THRESHOLD['COEFFICIENT_OF_VARIATION_THRESHOLD']: # Коэффициент вариации
                    stability_results[identity_id] = {'stable': False, 'reason': 'Нерегулярные появления'}
                    continue
            
            stability_results[identity_id] = {'stable': True, 'reason': 'Стабильный временной паттерн'}

        return stability_results

    def detect_embedding_drift(self, current_embeddings: List[np.ndarray], baseline_embeddings: List[np.ndarray]) -> Dict:
        """Обнаруживает дрейф эмбеддингов относительно базовой линии"""
        if not current_embeddings or not baseline_embeddings:
            return {'drift_detected': False, 'reason': 'Недостаточно данных для анализа дрейфа.'}

        # Средний эмбеддинг текущей выборки
        current_mean_embedding = np.mean(current_embeddings, axis=0)
        # Средний эмбеддинг базовой линии
        baseline_mean_embedding = np.mean(baseline_embeddings, axis=0)

        # Косинусное расстояние между средними эмбеддингами
        distance = 1 - (np.dot(current_mean_embedding, baseline_mean_embedding) / \
                        (np.linalg.norm(current_mean_embedding) * np.linalg.norm(baseline_mean_embedding) + 1e-8))
        
        # Порог дрейфа (можно настроить)
        drift_threshold = EMBEDDING_DRIFT_THRESHOLDS['DRIFT_DISTANCE_THRESHOLD'] # Пример: косинусное расстояние > 0.1 считается дрейфом

        if distance > drift_threshold:
            return {
                'drift_detected': True,
                'distance': float(distance),
                'reason': f'Обнаружен значительный дрейф эмбеддингов (расстояние {distance:.4f} > {drift_threshold}).'
            }
        else:
            return {
                'drift_detected': False,
                'distance': float(distance),
                'reason': f'Дрейф эмбеддингов не обнаружен (расстояние {distance:.4f} <= {drift_threshold}).'
            }

    def _calculate_mean_cosine_distance(self, embeddings_list1: List[np.ndarray], embeddings_list2: List[np.ndarray]) -> float:
        """Рассчитывает среднее косинусное расстояние между двумя наборами эмбеддингов."""
        if not embeddings_list1 or not embeddings_list2:
            return 0.0
        
        distances = []
        # Для простоты, берем первый эмбеддинг из каждого списка для сравнения
        # В реальной системе нужно будет сопоставлять эмбеддинги по времени или ID
        min_len = min(len(embeddings_list1), len(embeddings_list2))
        for i in range(min_len):
            dist = cosine_distances(embeddings_list1[i].reshape(1, -1), embeddings_list2[i].reshape(1, -1))[0, 0]
            distances.append(dist)
            
        return np.mean(distances) if distances else 0.0

    def age_corrected_embedding_drift(self, current_embeddings: List[np.ndarray],\
                                     baseline_embeddings: List[np.ndarray],\
                                     age_difference: float) -> Dict:
        """Детекция дрейфа с учетом возрастных изменений"""
        
        # Расчет ожидаемого дрейфа. Коэффициент 0.002 (0.2% дрейфа в год) - примерный, может быть откалиброван.
        expected_drift = age_difference * 0.002
        
        # Расчет фактического дрейфа (среднее косинусное расстояние)
        actual_drift = self._calculate_mean_cosine_distance(current_embeddings, baseline_embeddings)
        
        # Проверка на превышение ожидаемого дрейфа в N раз (например, в 3 раза)
        anomalous_drift = False
        severity = 0.0

        if expected_drift > 0 and actual_drift > (expected_drift * 3):
            anomalous_drift = True
            severity = actual_drift / expected_drift
        
        return {
            'anomalous_drift': anomalous_drift,
            'severity': severity,
            'actual_drift': actual_drift,
            'expected_drift': expected_drift
        }
