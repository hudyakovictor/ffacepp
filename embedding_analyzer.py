# embedding_analyzer.py
import os
import json
import logging
import hashlib
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import onnxruntime as ort
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import zscore
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

# Размеры для InsightFace
INSIGHTFACE_INPUT_SIZE = (112, 112)
EMBEDDING_DIMENSION = 512

# Параметры DBSCAN для кластеризации
DBSCAN_EPS = 0.35
DBSCAN_MIN_SAMPLES = 3
DBSCAN_METRIC = 'cosine'

# Пороги для валидации эмбеддингов
EMBEDDING_DISTANCE_THRESHOLD = 0.35
CLUSTER_CONFIDENCE_THRESHOLD = 0.7
OUTLIER_THRESHOLD = 0.5

# Параметры предобработки для InsightFace
ARCFACE_SRC = np.array([
    [30.2946, 51.6963],
    [65.5318, 51.5014],
    [48.0252, 71.7366],
    [33.5493, 92.3655],
    [62.7299, 92.2041]
], dtype=np.float32)

# === СТРУКТУРЫ ДАННЫХ ===

@dataclass
class EmbeddingPackage:
    """Пакет данных с эмбеддингом лица"""
    image_id: str
    filepath: str
    embedding_vector: np.ndarray  # 512-мерный вектор
    extraction_confidence: float
    
    # Кластерная информация
    cluster_id: int = -1
    cluster_confidence: float = 0.0
    is_outlier: bool = False
    
    # Метрики качества
    alignment_quality: float = 0.0
    face_quality_score: float = 0.0
    
    # Статистика расстояний
    nearest_neighbor_distance: float = float('inf')
    mean_cluster_distance: float = float('inf')
    
    # Метаданные обработки
    processing_time_ms: float = 0.0
    model_version: str = "w600k_r50"
    device_used: str = "cpu"
    
    # Флаги качества
    quality_flags: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

@dataclass
class ClusterManifest:
    """Манифест кластера идентичности"""
    cluster_id: int
    center_embedding: np.ndarray
    member_count: int
    first_appearance: datetime.date
    last_appearance: datetime.date
    
    # Статистика кластера
    intra_cluster_variance: float
    cluster_radius: float
    stability_score: float
    
    # Временные характеристики
    appearance_gaps: List[int]  # дни между появлениями
    total_timespan_days: int
    
    # Качество кластера
    silhouette_score: float
    cohesion_score: float
    
    # Список участников
    member_image_ids: List[str] = field(default_factory=list)

@dataclass
class IdentityTimeline:
    """Временная линия появления идентичностей"""
    clusters: Dict[int, ClusterManifest]
    total_identities: int
    date_range: Tuple[datetime.date, datetime.date]
    
    # Статистика переключений
    identity_switches: List[Dict[str, Any]]
    switch_frequency: float
    
    # Анализ паттернов
    dominant_identity: int
    identity_distribution: Dict[int, float]

# === ОСНОВНОЙ КЛАСС АНАЛИЗАТОРА ЭМБЕДДИНГОВ ===

class EmbeddingAnalyzer:
    """Анализатор для извлечения и кластеризации эмбеддингов лиц"""
    
    def __init__(self):
        self.config = get_config()
        self.cache_dir = Path("./cache/embedding_analyzer")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Модели
        self.insightface_session = None
        self.scaler = StandardScaler()
        
        # Кэш результатов
        self.embedding_cache: Dict[str, EmbeddingPackage] = {}
        self.cluster_cache: Dict[str, ClusterManifest] = {}
        
        # Статистика
        self.processing_stats = {
            'total_processed': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'cache_hits': 0,
            'clusters_created': 0
        }
        
        # Блокировка для потокобезопасности
        self.extraction_lock = threading.Lock()
        
        # Инициализация модели
        self._initialize_insightface_model()
        
        logger.info("EmbeddingAnalyzer инициализирован")

    def _initialize_insightface_model(self):
        """Инициализация модели InsightFace"""
        try:
            model_path = self.config.get_model_path("insightface")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Модель InsightFace не найдена: {model_path}")
            
            # Настройка провайдеров для ONNX Runtime
            providers = []
            
            # Проверка доступности MPS (Apple Silicon)
            if torch.backends.mps.is_available():
                providers.append('CPUExecutionProvider')  # MPS пока не поддерживается ONNX Runtime
                logger.info("Используется CPU провайдер (MPS не поддерживается ONNX Runtime)")
            else:
                providers.append('CPUExecutionProvider')
            
            # Создание сессии ONNX Runtime
            self.insightface_session = ort.InferenceSession(
                model_path,
                providers=providers
            )
            
            # Проверка входных и выходных размерностей
            input_shape = self.insightface_session.get_inputs()[0].shape
            output_shape = self.insightface_session.get_outputs()[0].shape
            
            logger.info(f"InsightFace модель загружена: {input_shape} -> {output_shape}")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации InsightFace: {e}")
            raise

    def extract_512d_face_embedding(self, image: np.ndarray, 
                                  landmarks: np.ndarray) -> Optional[EmbeddingPackage]:
        """
        Извлечение 512-мерного эмбеддинга лица
        
        Args:
            image: Входное изображение
            landmarks: 68-точечные ландмарки лица
            
        Returns:
            Пакет с эмбеддингом или None при ошибке
        """
        try:
            start_time = time.time()
            
            # Генерация ID изображения
            image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
            image_id = hashlib.sha256(image_bytes).hexdigest()
            
            # Проверка кэша
            if image_id in self.embedding_cache:
                self.processing_stats['cache_hits'] += 1
                cached_result = self.embedding_cache[image_id]
                cached_result.processing_time_ms = (time.time() - start_time) * 1000
                return cached_result
            
            # Извлечение 5 ключевых точек из 68-точечной модели
            key_points = self._extract_5_keypoints(landmarks)
            
            # Аффинное выравнивание лица
            aligned_face, alignment_quality = self._align_face(image, key_points)
            
            if aligned_face is None:
                logger.warning("Не удалось выровнять лицо")
                self.processing_stats['failed_extractions'] += 1
                return None
            
            # Предобработка для InsightFace
            preprocessed = self._preprocess_for_insightface(aligned_face)
            
            # Извлечение эмбеддинга
            embedding_vector = self._run_insightface_inference(preprocessed)
            
            if embedding_vector is None:
                logger.error("Не удалось извлечь эмбеддинг")
                self.processing_stats['failed_extractions'] += 1
                return None
            
            # Нормализация эмбеддинга
            embedding_vector = self._normalize_embedding(embedding_vector)
            
            # Оценка качества лица
            face_quality_score = self._assess_face_quality(aligned_face)
            
            # Расчет достоверности извлечения
            extraction_confidence = self._calculate_extraction_confidence(
                alignment_quality, face_quality_score
            )
            
            # Создание пакета
            package = EmbeddingPackage(
                image_id=image_id,
                filepath="",  # Будет заполнено вызывающей функцией
                embedding_vector=embedding_vector,
                extraction_confidence=extraction_confidence,
                alignment_quality=alignment_quality,
                face_quality_score=face_quality_score,
                processing_time_ms=(time.time() - start_time) * 1000,
                device_used="cpu"
            )
            
            # Валидация результата
            self._validate_embedding_package(package)
            
            # Сохранение в кэш
            self.embedding_cache[image_id] = package
            
            self.processing_stats['successful_extractions'] += 1
            self.processing_stats['total_processed'] += 1
            
            logger.debug(f"Эмбеддинг извлечен за {package.processing_time_ms:.1f}мс")
            return package
            
        except Exception as e:
            logger.error(f"Ошибка извлечения эмбеддинга: {e}")
            self.processing_stats['failed_extractions'] += 1
            self.processing_stats['total_processed'] += 1
            return None

    def _extract_5_keypoints(self, landmarks_68: np.ndarray) -> np.ndarray:
        """Извлечение 5 ключевых точек из 68-точечной модели"""
        try:
            # Индексы для 5 ключевых точек InsightFace
            # Левый глаз, правый глаз, нос, левый угол рта, правый угол рта
            left_eye = np.mean(landmarks_68[42:48, :2], axis=0)  # Левый глаз
            right_eye = np.mean(landmarks_68[36:42, :2], axis=0)  # Правый глаз
            nose = landmarks_68[30, :2]  # Кончик носа
            left_mouth = landmarks_68[48, :2]  # Левый угол рта
            right_mouth = landmarks_68[54, :2]  # Правый угол рта
            
            keypoints = np.array([
                left_eye, right_eye, nose, left_mouth, right_mouth
            ], dtype=np.float32)
            
            return keypoints
            
        except Exception as e:
            logger.error(f"Ошибка извлечения ключевых точек: {e}")
            raise

    def _align_face(self, image: np.ndarray, keypoints: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """Аффинное выравнивание лица по 5 ключевым точкам"""
        try:
            # Расчет трансформационной матрицы
            tform = cv2.estimateAffinePartial2D(keypoints, ARCFACE_SRC)[0]
            
            if tform is None:
                logger.warning("Не удалось вычислить трансформационную матрицу")
                return None, 0.0
            
            # Применение трансформации
            aligned = cv2.warpAffine(
                image, tform, INSIGHTFACE_INPUT_SIZE, 
                borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
            
            # Оценка качества выравнивания
            alignment_quality = self._assess_alignment_quality(keypoints, tform)
            
            return aligned, alignment_quality
            
        except Exception as e:
            logger.error(f"Ошибка выравнивания лица: {e}")
            return None, 0.0

    def _assess_alignment_quality(self, keypoints: np.ndarray, tform: np.ndarray) -> float:
        """Оценка качества выравнивания"""
        try:
            # Применение трансформации к ключевым точкам
            keypoints_homogeneous = np.hstack([keypoints, np.ones((len(keypoints), 1))])
            transformed_keypoints = (tform @ keypoints_homogeneous.T).T
            
            # Расчет расстояний до эталонных точек
            distances = np.linalg.norm(transformed_keypoints - ARCFACE_SRC, axis=1)
            mean_distance = np.mean(distances)
            
            # Нормализация качества (чем меньше расстояние, тем лучше)
            quality = max(0.0, 1.0 - mean_distance / 10.0)
            
            return float(quality)
            
        except Exception as e:
            logger.error(f"Ошибка оценки качества выравнивания: {e}")
            return 0.0

    def _preprocess_for_insightface(self, image: np.ndarray) -> np.ndarray:
        """Предобработка изображения для InsightFace"""
        try:
            # Конвертация в RGB если нужно
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Нормализация к диапазону [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # Нормализация по каналам (ImageNet статистика)
            mean = np.array([0.5, 0.5, 0.5])
            std = np.array([0.5, 0.5, 0.5])
            
            image = (image - mean) / std
            
            # Изменение размерности для ONNX (NCHW)
            image = np.transpose(image, (2, 0, 1))
            image = np.expand_dims(image, axis=0)
            
            return image
            
        except Exception as e:
            logger.error(f"Ошибка предобработки: {e}")
            raise

    def _run_insightface_inference(self, preprocessed_image: np.ndarray) -> Optional[np.ndarray]:
        """Запуск инференса InsightFace"""
        try:
            if self.insightface_session is None:
                logger.error("Модель InsightFace не инициализирована")
                return None
            
            # Получение имени входного тензора
            input_name = self.insightface_session.get_inputs()[0].name
            
            # Запуск инференса
            outputs = self.insightface_session.run(None, {input_name: preprocessed_image})
            
            # Извлечение эмбеддинга
            embedding = outputs[0][0]  # Первый батч, первый элемент
            
            return embedding
            
        except Exception as e:
            logger.error(f"Ошибка инференса InsightFace: {e}")
            return None

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """L2-нормализация эмбеддинга"""
        try:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                return embedding / norm
            else:
                logger.warning("Нулевой эмбеддинг, нормализация невозможна")
                return embedding
                
        except Exception as e:
            logger.error(f"Ошибка нормализации эмбеддинга: {e}")
            return embedding

    def _assess_face_quality(self, face_image: np.ndarray) -> float:
        """Оценка качества лица"""
        try:
            # Конвертация в градации серого
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = face_image
            
            # Оценка резкости (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(1.0, laplacian_var / 100.0)
            
            # Оценка контраста
            contrast_score = np.std(gray) / 255.0
            
            # Оценка яркости
            brightness = np.mean(gray) / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2
            
            # Общий балл качества
            quality_score = (sharpness_score + contrast_score + brightness_score) / 3.0
            
            return float(max(0.0, min(1.0, quality_score)))
            
        except Exception as e:
            logger.error(f"Ошибка оценки качества лица: {e}")
            return 0.0

    def _calculate_extraction_confidence(self, alignment_quality: float, 
                                       face_quality: float) -> float:
        """Расчет достоверности извлечения эмбеддинга"""
        try:
            # Взвешенная комбинация факторов качества
            confidence = 0.6 * alignment_quality + 0.4 * face_quality
            
            return float(max(0.0, min(1.0, confidence)))
            
        except Exception as e:
            logger.error(f"Ошибка расчета достоверности: {e}")
            return 0.0

    def _validate_embedding_package(self, package: EmbeddingPackage):
        """Валидация пакета эмбеддинга"""
        warnings = []
        quality_flags = []
        
        try:
            # Проверка размерности эмбеддинга
            if len(package.embedding_vector) != EMBEDDING_DIMENSION:
                warnings.append(f"Неверная размерность эмбеддинга: {len(package.embedding_vector)}")
                quality_flags.append("wrong_dimension")
            
            # Проверка нормализации
            norm = np.linalg.norm(package.embedding_vector)
            if abs(norm - 1.0) > 0.01:
                warnings.append(f"Эмбеддинг не нормализован: норма = {norm:.3f}")
                quality_flags.append("not_normalized")
            
            # Проверка качества извлечения
            if package.extraction_confidence < CLUSTER_CONFIDENCE_THRESHOLD:
                warnings.append(f"Низкая достоверность извлечения: {package.extraction_confidence:.3f}")
                quality_flags.append("low_confidence")
            
            # Проверка качества выравнивания
            if package.alignment_quality < 0.5:
                warnings.append(f"Низкое качество выравнивания: {package.alignment_quality:.3f}")
                quality_flags.append("poor_alignment")
            
            package.warnings = warnings
            package.quality_flags = quality_flags
            
        except Exception as e:
            logger.error(f"Ошибка валидации пакета эмбеддинга: {e}")
            package.warnings = [f"Ошибка валидации: {str(e)}"]
            package.quality_flags = ["validation_error"]

    def calculate_embedding_distances_matrix(self, embeddings: List[EmbeddingPackage]) -> np.ndarray:
        """
        Расчет матрицы расстояний между эмбеддингами
        
        Args:
            embeddings: Список пакетов эмбеддингов
            
        Returns:
            Матрица косинусных расстояний
        """
        try:
            if len(embeddings) < 2:
                logger.warning("Недостаточно эмбеддингов для расчета матрицы расстояний")
                return np.array([[]])
            
            # Извлечение векторов
            vectors = np.array([pkg.embedding_vector for pkg in embeddings])
            
            # Расчет матрицы косинусных расстояний
            n = len(vectors)
            distance_matrix = np.zeros((n, n))
            
            for i in range(n):
                for j in range(i + 1, n):
                    dist = cosine(vectors[i], vectors[j])
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist
            
            logger.debug(f"Рассчитана матрица расстояний {n}x{n}")
            return distance_matrix
            
        except Exception as e:
            logger.error(f"Ошибка расчета матрицы расстояний: {e}")
            return np.array([[]])

    def perform_identity_clustering(self, embeddings: List[EmbeddingPackage]) -> Dict[int, ClusterManifest]:
        """
        Кластеризация эмбеддингов для выявления идентичностей
        
        Args:
            embeddings: Список пакетов эмбеддингов
            
        Returns:
            Словарь кластеров по ID
        """
        try:
            if len(embeddings) < DBSCAN_MIN_SAMPLES:
                logger.warning(f"Недостаточно эмбеддингов для кластеризации: {len(embeddings)}")
                return {}
            
            # Извлечение векторов
            vectors = np.array([pkg.embedding_vector for pkg in embeddings])
            
            # Кластеризация DBSCAN
            clustering = DBSCAN(
                eps=DBSCAN_EPS,
                min_samples=DBSCAN_MIN_SAMPLES,
                metric='cosine'
            )
            
            cluster_labels = clustering.fit_predict(vectors)
            
            # Обновление пакетов эмбеддингов
            for i, package in enumerate(embeddings):
                package.cluster_id = int(cluster_labels[i])
                package.is_outlier = (cluster_labels[i] == -1)
            
            # Создание манифестов кластеров
            clusters = {}
            unique_labels = set(cluster_labels)
            
            for label in unique_labels:
                if label == -1:  # Пропускаем выбросы
                    continue
                
                # Получение членов кластера
                cluster_members = [pkg for pkg, lbl in zip(embeddings, cluster_labels) if lbl == label]
                
                # Создание манифеста кластера
                manifest = self._create_cluster_manifest(label, cluster_members)
                clusters[label] = manifest
                
                # Обновление информации о кластере в пакетах
                for member in cluster_members:
                    member.cluster_confidence = manifest.cohesion_score
                    member.mean_cluster_distance = self._calculate_mean_distance_to_cluster(
                        member, cluster_members
                    )
            
            # Расчет расстояний до ближайших соседей
            self._calculate_nearest_neighbor_distances(embeddings)
            
            self.processing_stats['clusters_created'] = len(clusters)
            
            logger.info(f"Создано {len(clusters)} кластеров из {len(embeddings)} эмбеддингов")
            return clusters
            
        except Exception as e:
            logger.error(f"Ошибка кластеризации: {e}")
            return {}

    def _create_cluster_manifest(self, cluster_id: int, 
                               members: List[EmbeddingPackage]) -> ClusterManifest:
        """Создание манифеста кластера"""
        try:
            # Расчет центра кластера
            vectors = np.array([member.embedding_vector for member in members])
            center_embedding = np.mean(vectors, axis=0)
            center_embedding = center_embedding / np.linalg.norm(center_embedding)  # Нормализация
            
            # Расчет статистик кластера
            distances_to_center = [cosine(vec, center_embedding) for vec in vectors]
            intra_cluster_variance = float(np.var(distances_to_center))
            cluster_radius = float(np.max(distances_to_center))
            
            # Временные характеристики (заглушки, будут заполнены в temporal_analyzer)
            first_appearance = datetime.date.today()
            last_appearance = datetime.date.today()
            
            # Расчет качества кластера
            silhouette_score = self._calculate_silhouette_score(members, vectors)
            cohesion_score = 1.0 - intra_cluster_variance  # Простая метрика сплоченности
            
            # Стабильность кластера
            stability_score = self._calculate_cluster_stability(vectors)
            
            manifest = ClusterManifest(
                cluster_id=cluster_id,
                center_embedding=center_embedding,
                member_count=len(members),
                first_appearance=first_appearance,
                last_appearance=last_appearance,
                intra_cluster_variance=intra_cluster_variance,
                cluster_radius=cluster_radius,
                stability_score=stability_score,
                appearance_gaps=[],  # Будет заполнено в temporal_analyzer
                total_timespan_days=0,  # Будет заполнено в temporal_analyzer
                silhouette_score=silhouette_score,
                cohesion_score=cohesion_score,
                member_image_ids=[member.image_id for member in members]
            )
            
            return manifest
            
        except Exception as e:
            logger.error(f"Ошибка создания манифеста кластера: {e}")
            raise

    def _calculate_silhouette_score(self, members: List[EmbeddingPackage], 
                                  vectors: np.ndarray) -> float:
        """Расчет silhouette score для кластера"""
        try:
            if len(vectors) < 2:
                return 1.0  # Единственный элемент в кластере
            
            # Внутрикластерные расстояния
            intra_distances = []
            for i, vec in enumerate(vectors):
                other_vectors = np.delete(vectors, i, axis=0)
                distances = [cosine(vec, other_vec) for other_vec in other_vectors]
                intra_distances.append(np.mean(distances))
            
            # Упрощенный silhouette score (без межкластерных расстояний)
            mean_intra_distance = np.mean(intra_distances)
            silhouette = 1.0 - mean_intra_distance
            
            return float(max(0.0, min(1.0, silhouette)))
            
        except Exception as e:
            logger.error(f"Ошибка расчета silhouette score: {e}")
            return 0.0

    def _calculate_cluster_stability(self, vectors: np.ndarray) -> float:
        """Расчет стабильности кластера"""
        try:
            if len(vectors) < 2:
                return 1.0
            
            # Расчет центра
            center = np.mean(vectors, axis=0)
            center = center / np.linalg.norm(center)
            
            # Расстояния до центра
            distances = [cosine(vec, center) for vec in vectors]
            
            # Стабильность как обратная величина стандартного отклонения
            stability = 1.0 / (1.0 + np.std(distances))
            
            return float(stability)
            
        except Exception as e:
            logger.error(f"Ошибка расчета стабильности кластера: {e}")
            return 0.0

    def _calculate_mean_distance_to_cluster(self, target: EmbeddingPackage, 
                                          cluster_members: List[EmbeddingPackage]) -> float:
        """Расчет среднего расстояния до кластера"""
        try:
            distances = []
            for member in cluster_members:
                if member.image_id != target.image_id:
                    dist = cosine(target.embedding_vector, member.embedding_vector)
                    distances.append(dist)
            
            return float(np.mean(distances)) if distances else 0.0
            
        except Exception as e:
            logger.error(f"Ошибка расчета расстояния до кластера: {e}")
            return 0.0

    def _calculate_nearest_neighbor_distances(self, embeddings: List[EmbeddingPackage]):
        """Расчет расстояний до ближайших соседей"""
        try:
            for i, target in enumerate(embeddings):
                min_distance = float('inf')
                
                for j, other in enumerate(embeddings):
                    if i != j:
                        dist = cosine(target.embedding_vector, other.embedding_vector)
                        min_distance = min(min_distance, dist)
                
                target.nearest_neighbor_distance = min_distance
                
        except Exception as e:
            logger.error(f"Ошибка расчета расстояний до ближайших соседей: {e}")

    def build_identity_timeline(self, clusters: Dict[int, ClusterManifest], 
                              data_manager) -> IdentityTimeline:
        """
        Построение временной линии идентичностей
        
        Args:
            clusters: Словарь кластеров
            data_manager: Менеджер данных для получения дат
            
        Returns:
            Временная линия идентичностей
        """
        try:
            if not clusters:
                logger.warning("Нет кластеров для построения временной линии")
                return IdentityTimeline(
                    clusters={},
                    total_identities=0,
                    date_range=(datetime.date.today(), datetime.date.today()),
                    identity_switches=[],
                    switch_frequency=0.0,
                    dominant_identity=-1,
                    identity_distribution={}
                )
            
            # Обновление временных характеристик кластеров
            updated_clusters = {}
            all_dates = []
            
            for cluster_id, manifest in clusters.items():
                # Получение дат для членов кластера
                member_dates = []
                for image_id in manifest.member_image_ids:
                    # Здесь должен быть вызов к data_manager для получения даты
                    # Пока используем заглушку
                    member_dates.append(datetime.date.today())
                
                if member_dates:
                    member_dates.sort()
                    all_dates.extend(member_dates)
                    
                    # Обновление временных характеристик
                    manifest.first_appearance = member_dates[0]
                    manifest.last_appearance = member_dates[-1]
                    manifest.total_timespan_days = (member_dates[-1] - member_dates[0]).days
                    
                    # Расчет пропусков между появлениями
                    gaps = []
                    for i in range(1, len(member_dates)):
                        gap = (member_dates[i] - member_dates[i-1]).days
                        gaps.append(gap)
                    manifest.appearance_gaps = gaps
                
                updated_clusters[cluster_id] = manifest
            
            # Общая статистика
            if all_dates:
                all_dates.sort()
                date_range = (all_dates[0], all_dates[-1])
            else:
                date_range = (datetime.date.today(), datetime.date.today())
            
            # Анализ переключений идентичностей
            identity_switches = self._analyze_identity_switches(updated_clusters)
            
            # Расчет частоты переключений
            total_days = (date_range[1] - date_range[0]).days
            switch_frequency = len(identity_switches) / max(total_days, 1)
            
            # Определение доминирующей идентичности
            dominant_identity = max(updated_clusters.keys(), 
                                  key=lambda x: updated_clusters[x].member_count)
            
            # Распределение идентичностей
            total_members = sum(manifest.member_count for manifest in updated_clusters.values())
            identity_distribution = {
                cluster_id: manifest.member_count / total_members
                for cluster_id, manifest in updated_clusters.items()
            }
            
            timeline = IdentityTimeline(
                clusters=updated_clusters,
                total_identities=len(updated_clusters),
                date_range=date_range,
                identity_switches=identity_switches,
                switch_frequency=switch_frequency,
                dominant_identity=dominant_identity,
                identity_distribution=identity_distribution
            )
            
            logger.info(f"Построена временная линия с {len(updated_clusters)} идентичностями")
            return timeline
            
        except Exception as e:
            logger.error(f"Ошибка построения временной линии: {e}")
            return IdentityTimeline(
                clusters={},
                total_identities=0,
                date_range=(datetime.date.today(), datetime.date.today()),
                identity_switches=[],
                switch_frequency=0.0,
                dominant_identity=-1,
                identity_distribution={}
            )

    def _analyze_identity_switches(self, clusters: Dict[int, ClusterManifest]) -> List[Dict[str, Any]]:
        """Анализ переключений между идентичностями"""
        try:
            switches = []
            
            # Создание временной последовательности появлений
            appearances = []
            for cluster_id, manifest in clusters.items():
                appearances.append({
                    'date': manifest.first_appearance,
                    'cluster_id': cluster_id,
                    'type': 'first_appearance'
                })
                appearances.append({
                    'date': manifest.last_appearance,
                    'cluster_id': cluster_id,
                    'type': 'last_appearance'
                })
            
            # Сортировка по дате
            appearances.sort(key=lambda x: x['date'])
            
            # Поиск переключений
            current_identity = None
            for appearance in appearances:
                if appearance['type'] == 'first_appearance':
                    if current_identity is not None and current_identity != appearance['cluster_id']:
                        switches.append({
                            'date': appearance['date'],
                            'from_identity': current_identity,
                            'to_identity': appearance['cluster_id'],
                            'switch_type': 'transition'
                        })
                    current_identity = appearance['cluster_id']
            
            return switches
            
        except Exception as e:
            logger.error(f"Ошибка анализа переключений идентичностей: {e}")
            return []

    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики обработки
        
        Returns:
            Словарь со статистикой
        """
        stats = self.processing_stats.copy()
        
        if stats['total_processed'] > 0:
            stats['success_rate'] = stats['successful_extractions'] / stats['total_processed']
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_processed']
        else:
            stats['success_rate'] = 0.0
            stats['cache_hit_rate'] = 0.0
        
        # Информация о модели
        stats['model_info'] = {
            'model_loaded': self.insightface_session is not None,
            'embedding_dimension': EMBEDDING_DIMENSION,
            'dbscan_eps': DBSCAN_EPS,
            'dbscan_min_samples': DBSCAN_MIN_SAMPLES
        }
        
        # Информация о кэше
        stats['cache_info'] = {
            'embedding_cache_size': len(self.embedding_cache),
            'cluster_cache_size': len(self.cluster_cache)
        }
        
        # Информация о памяти
        process = psutil.Process()
        memory_info = process.memory_info()
        stats['memory_usage_mb'] = memory_info.rss / (1024 * 1024)
        
        return stats

    def clear_cache(self):
        """Очистка кэша результатов"""
        try:
            self.embedding_cache.clear()
            self.cluster_cache.clear()
            logger.info("Кэш EmbeddingAnalyzer очищен")
            
        except Exception as e:
            logger.error(f"Ошибка очистки кэша: {e}")

    def save_cache(self, cache_filename: str = "embedding_cache.pkl"):
        """Сохранение кэша на диск"""
        try:
            cache_path = self.cache_dir / cache_filename
            
            cache_data = {
                'embedding_cache': self.embedding_cache,
                'cluster_cache': self.cluster_cache,
                'processing_stats': self.processing_stats
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"Кэш сохранен: {cache_path}")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения кэша: {e}")

    def load_cache(self, cache_filename: str = "embedding_cache.pkl") -> bool:
        """Загрузка кэша с диска"""
        try:
            cache_path = self.cache_dir / cache_filename
            
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                
                self.embedding_cache = cache_data.get('embedding_cache', {})
                self.cluster_cache = cache_data.get('cluster_cache', {})
                self.processing_stats.update(cache_data.get('processing_stats', {}))
                
                logger.info(f"Кэш загружен: {cache_path}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Ошибка загрузки кэша: {e}")
            return False

# === ФУНКЦИИ САМОТЕСТИРОВАНИЯ ===

def self_test():
    """Самотестирование модуля embedding_analyzer"""
    try:
        logger.info("Запуск самотестирования embedding_analyzer...")
        
        # Создание экземпляра анализатора
        analyzer = EmbeddingAnalyzer()
        
        # Создание тестового изображения и ландмарков
        test_image = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        test_landmarks = np.random.rand(68, 3) * 100
        
        # Тест извлечения ключевых точек
        keypoints = analyzer._extract_5_keypoints(test_landmarks)
        assert keypoints.shape == (5, 2), "Неверная форма ключевых точек"
        
        # Тест предобработки
        preprocessed = analyzer._preprocess_for_insightface(test_image)
        assert preprocessed.shape == (1, 3, 112, 112), "Неверная форма предобработанного изображения"
        
        # Тест нормализации эмбеддинга
        test_embedding = np.random.rand(512)
        normalized = analyzer._normalize_embedding(test_embedding)
        assert abs(np.linalg.norm(normalized) - 1.0) < 0.01, "Эмбеддинг не нормализован"
        
        # Тест статистики
        stats = analyzer.get_processing_statistics()
        assert 'success_rate' in stats, "Отсутствует статистика"
        
        logger.info("Самотестирование embedding_analyzer завершено успешно")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка самотестирования: {e}")
        return False

# === ИНИЦИАЛИЗАЦИЯ ===

if __name__ == "__main__":
    # Запуск самотестирования при прямом вызове модуля
    success = self_test()
    if success:
        print("✅ Модуль embedding_analyzer работает корректно")
        
        # Демонстрация основной функциональности
        analyzer = EmbeddingAnalyzer()
        print(f"📊 Эмбеддингов в кэше: {len(analyzer.embedding_cache)}")
        print(f"🔧 Кластеров в кэше: {len(analyzer.cluster_cache)}")
        print(f"📏 Размерность эмбеддинга: {EMBEDDING_DIMENSION}")
        print(f"🎯 DBSCAN eps: {DBSCAN_EPS}")
        print(f"💾 Кэш-директория: {analyzer.cache_dir}")
    else:
        print("❌ Обнаружены ошибки в модуле embedding_analyzer")
        exit(1)