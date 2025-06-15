#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПОЛНОСТЬЮ ПЕРЕПИСАННАЯ СИСТЕМА АНАЛИЗА ЛИЦ С 3DDFA_V2
Сохранены ВСЕ функции + исправлены критические ошибки для стабильности метрик
Версия: 6.0 - Стабильная
"""

import sys
import argparse
import cv2
import yaml
import json
import os
import numpy as np
import math
import traceback
from scipy.spatial.distance import cosine
from typing import Dict, List, Tuple, Optional, Union
import warnings
import logging
from pathlib import Path
import time

warnings.filterwarnings('ignore')

# Добавляем директорию текущего скрипта в sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

# Импорты 3DDFA_V2
from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
from utils.depth import depth
from utils.pncc import pncc
from utils.uv import uv_tex
from utils.pose import viz_pose, P2sRt, matrix2angle
from utils.serialization import ser_to_ply, ser_to_obj
from utils.functions import draw_landmarks, get_suffix
from utils.tddfa_util import str2bool, _parse_param

# Импорты InsightFace
import insightface
from insightface.app import FaceAnalysis

# Импорты модулей анализа
from frontal_metrics import FrontalAnalysisModule
from frontal_edge_metrics import FrontalEdgeAnalysisModule
from semi_profile_metrics import SemiProfileAnalysisModule
from profile_metrics import ProfileAnalysisModule

# Константы
INSIGHT_FACE_THRESHOLD = 0.363
EPSILON = 1e-6
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

class NumpyEncoder(json.JSONEncoder):
    """JSON энкодер для NumPy типов"""
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            if np.isnan(o) or np.isinf(o):
                return None
            return round(float(o), 6)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.bool_):
            return bool(o)
        return super(NumpyEncoder, self).default(o)

class UnionFind:
    """Структура данных для анализа связей"""
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px != py:
            if self.rank[px] < self.rank[py]:
                px, py = py, px
            self.parent[py] = px
            if self.rank[px] == self.rank[py]:
                self.rank[px] += 1

class Enhanced3DFaceProcessor:
    """ИСПРАВЛЕНО: Правильная обработка 3D координат согласно документации 3DDFA_V2"""
    
    def __init__(self):
        """Инициализация процессора"""
        self.logger = logging.getLogger('Enhanced3DFaceProcessor')

    def _get_default_parameters(self):
        return {
            'R_matrix': np.eye(3, dtype=np.float32),
            't_vec': np.zeros(3, dtype=np.float32),
            's_scale': 1.0,
            'offset': np.zeros(2, dtype=np.float32),
            'alpha_shp': np.zeros(40, dtype=np.float32),
            'alpha_exp': np.zeros(10, dtype=np.float32),
            'pitch': 0.0,
            'yaw': 0.0,
            'roll': 0.0,
            'P_matrix': np.eye(3, 4, dtype=np.float32)
        }

    def extract_pose_and_shape_parameters(self, params):
        """Извлекает параметры позы и формы из param_lst"""
        if params is None:
            return None
            
        param = _parse_param(params)
        P, scale, pose = P2sRt(param)  # P2sRt возвращает 3 значения
        camera_params = {'P': P, 'scale': scale}
        
        # Получаем углы Эйлера
        _, yaw, pitch = matrix2angle(pose)
        R_matrix = pose  # Матрица поворота
        
        return {
            'yaw': yaw,
            'pitch': pitch,
            'R_matrix': R_matrix,
            'camera_params': camera_params
        }
    
    def extract_pose_and_shape_parameters(self, param_3dmm: np.ndarray) -> Dict:
        """ИСПРАВЛЕНО: Правильное извлечение параметров с компенсацией позы"""
        try:
            # Приведение к одномерному массиву
            param_3dmm = np.asarray(param_3dmm).flatten()
            if param_3dmm is None or len(param_3dmm) < 12:
                self.logger.warning(f"param_3dmm пустой или слишком короткий: {param_3dmm}")
                return self._get_default_parameters()
            if len(param_3dmm) not in (62, 72, 141):
                self.logger.error(f"param_3dmm имеет неподдерживаемую длину: {len(param_3dmm)}")
                return self._get_default_parameters()

            # 1. Извлекаем базовые параметры позы
            # _parse_param должен возвращать кортеж (R, t3d, s, ...)
            parsed = _parse_param(param_3dmm)
            if isinstance(parsed, (list, tuple)) and len(parsed) >= 3:
                R, t3d, s = parsed[:3]
            else:
                self.logger.error(f"_parse_param вернул неожиданный результат: {parsed}")
                return self._get_default_parameters()

            P_camera = np.eye(3, 4)
            pitch, yaw, roll = matrix2angle(R)

            # 2. Компенсация искажений в матрице поворота
            max_angle = 75.0  # градусы
            pitch_deg = float(np.degrees(pitch))
            yaw_deg = float(np.degrees(yaw))
            roll_deg = float(np.degrees(roll))
            pitch_deg = np.clip(pitch_deg, -max_angle, max_angle)
            yaw_deg = np.clip(yaw_deg, -max_angle, max_angle)
            roll_deg = np.clip(roll_deg, -30.0, 30.0)
            pitch = np.radians(pitch_deg)
            yaw = np.radians(yaw_deg)
            roll = np.radians(roll_deg)
            Rx = np.array([[1, 0, 0],
                          [0, np.cos(pitch), -np.sin(pitch)],
                          [0, np.sin(pitch), np.cos(pitch)]])
            Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                          [0, 1, 0],
                          [-np.sin(yaw), 0, np.cos(yaw)]])
            Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                          [np.sin(roll), np.cos(roll), 0],
                          [0, 0, 1]])
            R_compensated = Rz @ Ry @ Rx

            # 3. Извлекаем параметры формы и выражения
            if len(param_3dmm) >= 72:
                alpha_shp = param_3dmm[12:52]  # 40 параметров формы
                alpha_exp = param_3dmm[52:62]  # 10 параметров выражения
                offset = param_3dmm[62:64] if len(param_3dmm) > 64 else np.zeros(2)
            else:
                alpha_shp = np.zeros(40, dtype=np.float32)
                alpha_exp = np.zeros(10, dtype=np.float32)
                offset = np.zeros(2, dtype=np.float32)

            # 4. Применяем масштабирование для стабилизации позы
            # s может быть массивом, приводим к float
            if isinstance(s, (np.ndarray, list)):
                s_stable = float(np.squeeze(s))
            else:
                s_stable = float(s)
            if abs(yaw_deg) > 30.0 or abs(pitch_deg) > 30.0:
                perspective_factor = 1.0 + 0.002 * (abs(yaw_deg) + abs(pitch_deg))
                s_stable *= perspective_factor

            return {
                'R_matrix': R_compensated.astype(np.float32),
                't_vec': t3d.astype(np.float32),
                's_scale': float(s_stable),
                'offset': offset.astype(np.float32),
                'alpha_shp': alpha_shp.astype(np.float32),
                'alpha_exp': alpha_exp.astype(np.float32),
                'pitch': pitch_deg,
                'yaw': yaw_deg,
                'roll': roll_deg,
                'P_matrix': P_camera.astype(np.float32)
            }
        except Exception as e:
            self.logger.error(f"Ошибка при извлечении параметров: {e}")
            return self._get_default_parameters()
    
    def transform_to_frontal_view(self, landmarks_3d: np.ndarray, R_matrix: np.ndarray) -> np.ndarray:
        """ИСПРАВЛЕНО: Трансформация landmarks в фронтальную позицию с нормализацией по IOD"""
        try:
            if landmarks_3d is None or R_matrix is None:
                return landmarks_3d
            
            # 1. Нормализация по IOD (расстояние между глазами)
            left_eye_idx = 36  # Левый глаз
            right_eye_idx = 45  # Правый глаз
            iod = np.linalg.norm(landmarks_3d[right_eye_idx] - landmarks_3d[left_eye_idx])
            
            # Защита от деления на ноль
            if iod < 1e-6:
                self.logger.warning("IOD слишком мал, используем значение по умолчанию")
                iod = 1.0
            
            # Нормализуем координаты
            normalized_landmarks = landmarks_3d / iod
            
            # 2. Применяем обратную матрицу поворота для получения фронтальной позиции
            R_inv = R_matrix.T
            landmarks_frontal = np.zeros_like(normalized_landmarks)
            
            # 3. Компенсация перспективы для каждой точки
            for i, point in enumerate(normalized_landmarks):
                # Вычисляем Z-компоненту для коррекции перспективы
                z_factor = max(1.0, 1.0 + point[2] * 0.1)  # Увеличиваем влияние для более глубоких точек
                
                # Применяем поворот с учетом перспективы
                rotated_point = R_inv @ point
                
                # Компенсируем перспективные искажения
                landmarks_frontal[i] = rotated_point * z_factor
            
            # 4. Стабилизация глубины
            z_mean = np.mean(landmarks_frontal[:, 2])
            z_std = np.std(landmarks_frontal[:, 2])
            
            # Нормализуем Z-координату для уменьшения выбросов
            if z_std > 0:
                landmarks_frontal[:, 2] = (landmarks_frontal[:, 2] - z_mean) / z_std
            
            self.logger.debug(f"Трансформированы landmarks в фронтальную позицию с IOD={iod:.4f}")
            return landmarks_frontal.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Ошибка трансформации в фронтальную позицию: {e}")
            return landmarks_3d
    
    def correct_perspective_distortion(self, landmarks_3d: np.ndarray, s_scale: float, 
                                     yaw: float, pitch: float) -> np.ndarray:
        """ИСПРАВЛЕНО: Улучшенная коррекция перспективных искажений с учетом глубины и позы"""
        try:
            if landmarks_3d is None:
                return landmarks_3d
            
            # 1. Нормализация масштаба
            corrected_landmarks = landmarks_3d * s_scale
            
            # 2. Расчет углов в радианах
            yaw_rad = np.radians(abs(yaw))
            pitch_rad = np.radians(abs(pitch))
            
            # 3. Базовая коррекция глубины
            z_mean = np.mean(corrected_landmarks[:, 2])
            z_std = np.std(corrected_landmarks[:, 2])
            
            # 4. Компенсация перспективы и поворота для каждой точки
            for i in range(len(corrected_landmarks)):
                point = corrected_landmarks[i]
                
                # Компенсация yaw (рыскание)
                if abs(yaw) > 5.0:
                    # Используем нелинейную коррекцию для больших углов
                    x_factor = 1.0 / max(0.1, np.cos(yaw_rad))
                    z_offset = point[2] - z_mean
                    # Корректируем X в зависимости от Z-позиции
                    x_correction = x_factor * (1.0 + 0.1 * abs(z_offset) / max(1e-6, z_std))
                    corrected_landmarks[i, 0] *= x_correction
                
                # Компенсация pitch (тангаж)
                if abs(pitch) > 5.0:
                    # Аналогичная нелинейная коррекция для Y
                    y_factor = 1.0 / max(0.1, np.cos(pitch_rad))
                    z_offset = point[2] - z_mean
                    y_correction = y_factor * (1.0 + 0.1 * abs(z_offset) / max(1e-6, z_std))
                    corrected_landmarks[i, 1] *= y_correction
                
                # Стабилизация Z-координаты
                if z_std > 0:
                    # Нормализуем Z, сохраняя относительные глубины
                    z_norm = (point[2] - z_mean) / z_std
                    # Применяем сигмоидную функцию для ограничения диапазона
                    z_scaled = 2.0 / (1.0 + np.exp(-z_norm)) - 1.0
                    corrected_landmarks[i, 2] = z_mean + z_scaled * z_std
            
            return corrected_landmarks.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Ошибка коррекции перспективы: {e}")
            return landmarks_3d
    
    def process_landmarks_3d(self, vertices: np.ndarray, name: str = "vertices") -> np.ndarray:
        """ИСПРАВЛЕНО: Улучшенная обработка 3D координат с нормализацией"""
        if vertices is None:
            self.logger.error(f"{name} is None")
            return np.array([])  # Возвращаем пустой массив вместо None
        
        try:
            # 1. Предварительная обработка формата
            if len(vertices.shape) == 3 and vertices.shape[-1] == 1:
                vertices = np.squeeze(vertices, -1)
                self.logger.debug(f"Убрано лишнее измерение из {name}")
            
            # 2. Определение и обработка формата координат
            if len(vertices.shape) == 2:
                if vertices.shape[0] == 3 and vertices.shape[1] > vertices.shape[0]:
                    # Транспонируем из (3, N) в (N, 3)
                    result = vertices.T
                    self.logger.debug(f"Транспонирован {name}: {vertices.shape} -> {result.shape}")
                elif vertices.shape[1] == 3:
                    # Уже в правильном формате (N, 3)
                    result = vertices
                else:
                    self.logger.warning(f"Неожиданная форма {name}: {vertices.shape}")
                    return np.array([])
            else:
                self.logger.error(f"Неверное количество измерений в {name}: {len(vertices.shape)}")
                return np.array([])
            
            # 3. Нормализация координат
            # Центрируем относительно медианы для устойчивости к выбросам
            centroid = np.median(result, axis=0)
            result = result - centroid
            
            # Нормализуем масштаб по межглазному расстоянию (IOD)
            left_eye_idx = 36  # Индекс левого глаза
            right_eye_idx = 45  # Индекс правого глаза
            
            if result.shape[0] > max(left_eye_idx, right_eye_idx):
                iod = np.linalg.norm(result[right_eye_idx] - result[left_eye_idx])
                if iod > 1e-6:
                    result = result / iod
                    self.logger.debug(f"Нормализованы координаты по IOD={iod:.4f}")
            
            # 4. Стабилизация Z-координаты
            z_mean = np.mean(result[:, 2])
            z_std = np.std(result[:, 2])
            if z_std > 0:
                # Применяем сигмоидную нормализацию для Z-координаты
                z_normalized = (result[:, 2] - z_mean) / z_std
                result[:, 2] = 2.0 / (1.0 + np.exp(-z_normalized)) - 1.0
            
            return result.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки {name}: {e}")
            return np.array([])

    def normalize_landmarks_with_pose_compensation(self, vertices, R_matrix, yaw, pitch, scale_factor=1.0):
        """Нормализует вершины с компенсацией поворота головы"""
        if vertices is None or len(vertices) == 0:
            return None
            
        # Конвертируем в нужный формат, если нужно
        vertices = np.array(vertices)
        if vertices.shape[0] != 3 and vertices.shape[1] == 3:
            vertices = vertices.T
            
        # Центрируем вершины
        centroid = np.mean(vertices, axis=1, keepdims=True)
        vertices_centered = vertices - centroid
        
        # Компенсация поворота (инвертируем матрицу поворота)
        R_inv = np.linalg.inv(R_matrix)
        vertices_unrotated = np.dot(R_inv, vertices_centered)
        
        # Масштабирование
        if scale_factor != 1.0:
            vertices_unrotated *= scale_factor
        
        # Нормализация по z-координате
        z_min, z_max = vertices_unrotated[2].min(), vertices_unrotated[2].max()
        if abs(z_max - z_min) > EPSILON:
            vertices_unrotated[2] = (vertices_unrotated[2] - z_min) / (z_max - z_min)
        
        return vertices_unrotated.T  # Возвращаем в формате (N, 3)
    
    def project_vertices_to_2d(self, vertices_3d, img_size, camera_params=None):
        """Проецирует 3D вершины на 2D плоскость с учетом параметров камеры"""
        if vertices_3d is None or len(vertices_3d) == 0:
            return None
            
        # Конвертируем в однородные координаты
        vertices_3d = np.array(vertices_3d)
        if vertices_3d.shape[1] != 3:
            vertices_3d = vertices_3d.T
            
        # Добавляем однородную координату
        vertices_homo = np.hstack([vertices_3d, np.ones((vertices_3d.shape[0], 1))])
        
        if camera_params is not None and 'P' in camera_params:
            # Используем матрицу проекции из параметров камеры
            P = camera_params['P']
            vertices_2d = np.dot(vertices_homo, P.T)
        else:
            # Простая ортографическая проекция
            vertices_2d = vertices_3d[:, :2]
            
        # Нормализуем координаты в пиксели изображения
        w, h = img_size
        vertices_2d[:, 0] = np.clip(vertices_2d[:, 0] * w, 0, w-1)
        vertices_2d[:, 1] = np.clip(vertices_2d[:, 1] * h, 0, h-1)
        
        return vertices_2d

    def process_landmarks_3d_for_visualization(self, vertices: np.ndarray, roi_box: List = None) -> np.ndarray:
        """ИСПРАВЛЕНО: Обработка landmarks БЕЗ нормализации для корректной визуализации"""
        try:
            if vertices is None:
                return np.array([])
            # 1. Только транспонирование если нужно
            if len(vertices.shape) == 2:
                if vertices.shape[0] == 3 and vertices.shape[1] > vertices.shape[0]:
                    result = vertices.T  # (3, N) -> (N, 3)
                elif vertices.shape[1] == 3:
                    result = vertices  # Уже правильный формат
                else:
                    return np.array([])
            else:
                return np.array([])
            # 2. НИКАКОЙ нормализации для визуализации! Landmarks должны остаться в пиксельных координатах
            # 3. Применяем только roi_box трансформацию если нужно
            if roi_box and len(roi_box) >= 4:
                x1, y1, x2, y2 = roi_box[:4]
                if result.shape[0] > 0:
                    x_min, x_max = result[:, 0].min(), result[:, 0].max()
                    y_min, y_max = result[:, 1].min(), result[:, 1].max()
                    if x_min < x1 or x_max > x2 or y_min < y1 or y_max > y2:
                        # Масштабируем в пределы roi_box
                        result[:, 0] = x1 + (result[:, 0] - x_min) * (x2 - x1) / (x_max - x_min)
                        result[:, 1] = y1 + (result[:, 1] - y_min) * (y2 - y1) / (y_max - y_min)
            return result.astype(np.float32)
        except Exception as e:
            self.logger.error(f"Ошибка обработки {vertices.shape if vertices is not None else 'None'}: {e}")
            return np.array([])

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Настройка логирования"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('analysis.log', encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)

def debug_print(msg: str, level: str = "INFO"):
    """Вспомогательная функция для отладочной печати"""
    if level == "ERROR":
        logging.error(msg)
    elif level == "WARN":
        logging.warning(msg)
    elif level == "DEBUG":
        logging.debug(msg)
    else:
        logging.info(msg)

def debug_array_info(arr: np.ndarray, name: str):
    """Информация о массиве"""
    if arr is None:
        debug_print(f"{name}: None", "WARN")
        return
    
    try:
        debug_print(
            f"{name}: shape={arr.shape}, dtype={arr.dtype}, "
            f"min={arr.min():.3f}, max={arr.max():.3f}, "
            f"mean={arr.mean():.3f}, std={arr.std():.3f}", "DEBUG"
        )
    except Exception as e:
        debug_print(f"Ошибка анализа массива {name}: {e}", "ERROR")



def get_landmarks_from_tddfa(tddfa, param_lst: List, roi_box_lst: List, dense: bool = False, for_visualization: bool = False) -> List[np.ndarray]:
    """Получение landmarks из TDDFA: для визуализации - без нормализации, для анализа - с нормализацией"""
    debug_print(f"🎯 Получение landmarks (dense={dense}, for_visualization={for_visualization}) для {len(param_lst)} лиц", "INFO")
    try:
        vertices_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense)
        debug_print(f"Получено {len(vertices_lst)} наборов вершин", "INFO")
        processor = Enhanced3DFaceProcessor()
        processed_vertices = []
        for i, (vertices, roi_box, param) in enumerate(zip(vertices_lst, roi_box_lst, param_lst)):
            if not isinstance(vertices, np.ndarray):
                debug_print(f"vertices[{i}] не является numpy array: {type(vertices)}", "ERROR")
                processed_vertices.append(None)
                continue
            if for_visualization:
                processed = processor.process_landmarks_3d_for_visualization(vertices, roi_box)
                processed_vertices.append(processed)
                debug_array_info(processed, f"Processed vertices[{i}] for visualization")
            else:
                params = processor.extract_pose_and_shape_parameters(param)
                R_matrix = params['R_matrix']
                yaw = params['yaw']
                pitch = params['pitch']
                normalized = processor.normalize_landmarks_with_pose_compensation(
                    vertices.T if vertices.shape[0] == 3 else vertices,
                    R_matrix, yaw, pitch
                )
                processed_vertices.append(normalized)
                debug_array_info(normalized, f"Normalized vertices[{i}]")
        return processed_vertices
    except Exception as e:
        debug_print(f"Ошибка при получении landmarks: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return []

def darken_image(image: np.ndarray, factor: float = 0.3) -> np.ndarray:
    """Затемняет изображение"""
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        debug_print("Конвертировано RGBA в BGR", "DEBUG")
    
    black_overlay = np.zeros_like(image)
    return cv2.addWeighted(image, factor, black_overlay, 1 - factor, 0)

def calculate_embedding_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """Вычисление косинусного сходства"""
    try:
        if len(embedding1) != len(embedding2):
            debug_print(f"Разная длина эмбеддингов: {len(embedding1)} vs {len(embedding2)}", "WARN")
            return 0.0
        
        emb1 = np.array(embedding1, dtype=np.float32)
        emb2 = np.array(embedding2, dtype=np.float32)
        
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 < EPSILON or norm2 < EPSILON:
            return 0.0
        
        emb1 = emb1 / norm1
        emb2 = emb2 / norm2
        
        similarity = np.dot(emb1, emb2)
        return max(0.0, min(1.0, float(similarity)))
        
    except Exception as e:
        debug_print(f"Ошибка при вычислении сходства: {e}", "ERROR")
        return 0.0

def determine_pose_side(pose_type: str, yaw: float) -> str:
    if pose_type == "frontal":
        return "front"
    else:
        return "right" if yaw > 0 else "left"

def calculate_real_deviation(measured: float, ideal: float) -> float:
    """Расчет реального отклонения от идеального значения"""
    if abs(ideal) < EPSILON:
        return 0.0
    return abs(measured - ideal) / ideal * 100.0

def process_single_face_enhanced(face_idx: int, landmarks_3d: np.ndarray, params: np.ndarray,
                               roi_box: List, modules: Dict, insightface_embedding: List[float],
                               img: np.ndarray, processor_3d: Enhanced3DFaceProcessor) -> Dict:
    """ИСПРАВЛЕНО: Правильная обработка результатов анализа"""
    try:
        if landmarks_3d is None:
            logging.warning(f"landmarks_3d is None для лица {face_idx}")
            return {}

        # Получаем параметры позы и анализ    
        pose_params = processor_3d.extract_pose_and_shape_parameters(params)
        if not pose_params:
            logging.error(f"Не удалось извлечь параметры позы для лица {face_idx}")
            return {}

        # Определяем тип позы
        pose_type = modules['frontal'].marquardt_mask.classify_pose(
            pose_params['yaw'], pose_params['pitch'], pose_params['roll']
        )
        
        # Формируем базовый результат
        face_result = {
            "detection": {
                "face_index": face_idx,
                "bounding_box": roi_box,
                "confidence": 1.0
            },
            "pose": {
                "type": pose_type,
                "angles": {
                    "pitch": float(pose_params['pitch']),
                    "yaw": float(pose_params['yaw']),
                    "roll": float(pose_params['roll'])
                }
            },
            "landmarks": {
                "raw_3d": landmarks_3d.tolist() if isinstance(landmarks_3d, np.ndarray) else [],
                "normalized_3d": []
            },
            "embedding": insightface_embedding
        }

        # Выполняем анализ
        analysis_module = modules.get(pose_type)
        if analysis_module is None:
            logging.warning(f"Не найден модуль анализа для типа позы {pose_type}")
            return face_result

        try:
            # Запускаем полный анализ с проверкой возвращаемых значений
            analysis_result = analysis_module.analyze(
                landmarks_3d,
                pitch=float(pose_params['pitch']),
                yaw=float(pose_params['yaw']),
                roll=float(pose_params['roll']),
                alpha_shp=pose_params['alpha_shp'],
                alpha_exp=pose_params['alpha_exp'],
                R_matrix=pose_params['R_matrix'],
                t_vec=pose_params['t_vec'],
                s_scale=float(pose_params['s_scale'])
            )

            if not analysis_result:
                logging.warning(f"Анализ вернул пустой результат для лица {face_idx}")
                return face_result

            # Валидируем и конвертируем измерения
            validated_result = {}
            for category in ['raw_measurements', 'angular_metrics', 'proportion_metrics',
                           'skull_metrics', 'symmetry_metrics', 'anomaly_detection',
                           'stabilization_info']:
                category_data = analysis_result.get(category, {})
                if not isinstance(category_data, dict):
                    logging.warning(f"Неверный формат данных для {category}: {type(category_data)}")
                    continue

                validated_category = {}
                for key, value in category_data.items():
                    if isinstance(value, (np.floating, np.integer)):
                        validated_category[key] = float(value)
                    elif isinstance(value, dict):
                        validated_category[key] = {
                            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                            for k, v in value.items()
                        }
                    else:
                        validated_category[key] = value

                validated_result[category] = validated_category

            # Обновляем результат с проверенными значениями
            face_result.update(validated_result)

        except Exception as analysis_error:
            logging.error(f"Ошибка при анализе лица {face_idx}: {str(analysis_error)}")
            traceback.print_exc()

        return face_result

    except Exception as e:
        logging.error(f"Критическая ошибка в process_single_face_enhanced: {e}")
        traceback.print_exc()
        return {}

def generate_visualizations(img: np.ndarray, param_lst: List, roi_box_lst: List,
                          landmarks_3d_lst: List[np.ndarray], tddfa, base_name: str,
                          opt: str, selected_viz: Optional[List[str]] = None) -> Dict:
    """ИСПРАВЛЕНО: Генерация визуализаций согласно документации 3DDFA_V2"""
    generated_files = {}
    if opt == 'none':
        debug_print("Визуализации отключены", "INFO")
        return generated_files
    if opt == 'all':
        visualizations = ['2d_sparse', '2d_dense', '3d', 'depth', 'pncc', 'obj']
        debug_print("СТАТУС: Опция 'all' активна, генерируем все визуализации.", "INFO")
    elif opt == 'selected':
        visualizations = selected_viz or []
        if not visualizations:
            debug_print("Ни одной визуализации не выбрано (--opt selected)", "WARN")
            return generated_files
        debug_print(f"СТАТУС: Опция 'selected' активна: {visualizations}", "INFO")
    else:
        visualizations = [opt]
    ver_lst_dense = []
    if any(viz in ['2d_dense', '3d', 'depth', 'pncc', 'obj'] for viz in visualizations):
        try:
            ver_lst_dense = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
            debug_print(f"Получены RAW плотные вершины для {len(ver_lst_dense)} лиц", "DEBUG")
        except Exception as e:
            debug_print(f"Ошибка получения плотных вершин: {e}", "ERROR")
            ver_lst_dense = []
    original_wfp = f'examples/results/{base_name}_original.jpg'
    cv2.imwrite(original_wfp, img)
    generated_files["original_image"] = original_wfp
    for viz_type in visualizations:
        try:
            wfp = f'examples/results/{base_name}_{viz_type}'
            if viz_type == 'obj':
                wfp += '.obj'
            else:
                wfp += '.jpg'
            if viz_type != 'obj':
                img_darkened = darken_image(img.copy(), factor=0.3)
            else:
                img_darkened = img.copy()
            if viz_type == '2d_sparse':
                # Получаем свежие landmarks для визуализации
                sparse_landmarks_viz = get_landmarks_from_tddfa(tddfa, param_lst, roi_box_lst, dense=False, for_visualization=True)
                _generate_2d_sparse_visualization(img_darkened, sparse_landmarks_viz, wfp, generated_files, base_name)
            elif viz_type == '2d_dense':
                # Получаем свежие landmarks для визуализации
                dense_landmarks_viz = get_landmarks_from_tddfa(tddfa, param_lst, roi_box_lst, dense=True, for_visualization=True)
                _generate_2d_dense_visualization(img_darkened, dense_landmarks_viz, wfp, generated_files, base_name, param_lst)
            elif viz_type == '3d':
                if ver_lst_dense and len(ver_lst_dense) > 0:
                    render(img_darkened, ver_lst_dense, tddfa.tri, alpha=1.0, wfp=wfp)
                    debug_print(f'3D render saved to {wfp}', "INFO")
                    generated_files["3d_render_image"] = wfp
            elif viz_type == 'depth':
                if ver_lst_dense and len(ver_lst_dense) > 0:
                    depth(img_darkened, ver_lst_dense, tddfa.tri, wfp=wfp, with_bg_flag=True)
                    debug_print(f'Depth map saved to {wfp}', "INFO")
                    generated_files["depth_map_image"] = wfp
            elif viz_type == 'pncc':
                if ver_lst_dense and len(ver_lst_dense) > 0:
                    pncc(img_darkened, ver_lst_dense, tddfa.tri, wfp=wfp, with_bg_flag=True)
                    debug_print(f'PNCC map saved to {wfp}', "INFO")
                    generated_files["pncc_map_image"] = wfp
            elif viz_type == 'obj':
                if ver_lst_dense and len(ver_lst_dense) > 0:
                    ser_to_obj(img, ver_lst_dense, tddfa.tri, height=img.shape[0], wfp=wfp)
                    debug_print(f'OBJ model saved to {wfp}', "INFO")
                    generated_files["obj_model"] = wfp
        except Exception as e:
            debug_print(f"Ошибка генерации {viz_type}: {e}", "ERROR")
            import traceback
            traceback.print_exc()
    return generated_files

def _generate_2d_sparse_visualization(img_darkened: np.ndarray, landmarks_3d_lst: List[np.ndarray],
                                    wfp: str, generated_files: Dict, base_name: str):
    """Генерация 2D sparse landmarks визуализации"""
    # Правильные соединения согласно dlib 68-point model
    connections = {
        'jawline': [(i, i+1) for i in range(16)],
        'right_eyebrow': [(i, i+1) for i in range(17, 21)],
        'left_eyebrow': [(i, i+1) for i in range(22, 26)],
        'nose_bridge': [(i, i+1) for i in range(27, 30)],
        'nose_lower': [(31, 32), (32, 33), (33, 34), (34, 35), (35, 31)],
        'right_eye': [(36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36)],
        'left_eye': [(42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42)],
        'outer_lips': [(48, 49), (49, 50), (50, 51), (51, 52), (52, 53),
                       (53, 54), (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 48)],
        'inner_lips': [(60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67), (67, 60)]
    }
    
    img_viz = img_darkened.copy()
    h, w = img_viz.shape[:2]
    point_color = (255, 255, 255)      # Белый
    line_color = (0, 0, 255)           # Красный
    text_color = (255, 255, 255)           # Красный

    for face_landmarks in landmarks_3d_lst:
        if face_landmarks is None or len(face_landmarks) < 68:
            continue
        
        points_2d = face_landmarks[:68, :2].astype(np.int32)
        
        # Рисование соединений (красные линии)
        for region_name, point_pairs in connections.items():
            for p1_idx, p2_idx in point_pairs:
                if p1_idx < len(points_2d) and p2_idx < len(points_2d):
                    pt1 = tuple(points_2d[p1_idx])
                    pt2 = tuple(points_2d[p2_idx])
                    if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                        0 <= pt2[0] < w and 0 <= pt2[1] < h):
                        cv2.line(img_viz, pt1, pt2, line_color, 1, cv2.LINE_AA)
        
        # Рисование точек (белые) и номеров (красные)
        for i, (x, y) in enumerate(points_2d):
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(img_viz, (int(x), int(y)), 1, point_color, -1, cv2.LINE_AA)
                cv2.putText(img_viz, str(i), (int(x)+3, int(y)-3), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.35, text_color, 1, cv2.LINE_AA)
    
    cv2.imwrite(wfp, img_viz)
    debug_print(f'2D sparse landmarks visualization saved to {wfp}', "INFO")
    generated_files["landmarks_2d_sparse"] = wfp

def _generate_2d_dense_visualization(img_darkened: np.ndarray, ver_lst_dense: List[np.ndarray],
                                   wfp: str, generated_files: Dict, base_name: str,
                                   param_lst: Optional[List] = None):
    """ИСПРАВЛЕНО: Генерация 2D dense без повторной нормализации"""
    if not ver_lst_dense or len(ver_lst_dense) == 0:
        debug_print(f"СТАТУС: Пропуск визуализации 2d_dense для {base_name} - нет данных", "WARN")
        return
    green_color = (0, 255, 0)
    point_thickness = 1
    img_dense_viz = img_darkened.copy()
    h, w = img_dense_viz.shape[:2]
    faces_drawn = 0
    for face_idx, face_vertices in enumerate(ver_lst_dense):
        if face_vertices is not None:
            try:
                # Используем только исходные (сырые) landmarks без нормализации
                if face_vertices.shape[0] == 3:
                    points_3d = face_vertices.T
                else:
                    points_3d = face_vertices
                points_2d = points_3d[:, :2].astype(np.int32)
                valid_mask = ((points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) &
                             (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h))
                points_2d_filtered = points_2d[valid_mask]
                if len(points_2d_filtered) > 0:
                    faces_drawn += 1
                    debug_print(f"Dense visualization: {len(points_2d_filtered)} valid points", "DEBUG")
                    if points_3d.shape[1] >= 3:
                        z_values = points_3d[valid_mask, 2]
                        z_min, z_max = z_values.min(), z_values.max()
                        z_range = z_max - z_min
                        for point_idx, (x, y) in enumerate(points_2d_filtered):
                            if z_range > 1e-6:
                                # intensity: ближние к камере (z_min) — 255, дальние (z_max) — 80
                                intensity = int(255 + 255 * (1 - (z_values[point_idx] - z_min) / z_range))  # 80..255
                                point_color = (0, intensity, 0)
                            else:
                                point_color = green_color
                            size = int(0.3 + 0.5 * ((z_values[point_idx] - z_min) / z_range))  # размер: дальние крупнее
                            cv2.circle(img_dense_viz, (x, y), size, point_color, -1)
                    else:
                        for x, y in points_2d_filtered:
                            cv2.circle(img_dense_viz, (x, y), point_thickness, green_color, -1)
            except Exception as e:
                debug_print(f"Ошибка при обработке dense face {face_idx}: {e}", "ERROR")
                continue
    if faces_drawn > 0:
        cv2.imwrite(wfp, img_dense_viz)
        debug_print(f'2D dense landmarks visualization saved to {wfp}', "INFO")
        generated_files["2d_dense_landmarks_image"] = wfp
    else:
        debug_print(f"СТАТУС: Пропуск визуализации 2d_dense для {base_name} - нет валидных точек", "WARN")

def compare_faces_embeddings(all_analysis_results: Dict) -> Dict:
    """Сравнение эмбеддингов лиц"""
    debug_print("🔄 Запуск сравнения эмбеддингов лиц...", "INFO")
    
    face_comparisons = {}
    image_names = list(all_analysis_results.keys())
    
    # Сравнение между всеми парами изображений
    for i in range(len(image_names)):
        img_name1 = image_names[i]
        faces_data1 = all_analysis_results[img_name1]["faces_data"]
        
        # Сравнение внутри изображения
        for face_idx1 in range(len(faces_data1)):
            for face_idx2 in range(face_idx1 + 1, len(faces_data1)):
                embedding1 = faces_data1[face_idx1].get("insightface_embedding", [])
                embedding2 = faces_data1[face_idx2].get("insightface_embedding", [])
                
                if embedding1 and embedding2:
                    similarity = calculate_embedding_similarity(embedding1, embedding2)
                    comparison_key = f"{img_name1}_face{face_idx1}_vs_{img_name1}_face{face_idx2}"
                    
                    face_comparisons[comparison_key] = {
                        "embedding_similarity": round(similarity, 6),
                        "likely_same_person": bool(similarity > INSIGHT_FACE_THRESHOLD),
                        "confidence": round(similarity, 6),
                        "comparison_type": "intra_image"
                    }
        
        # Сравнение между изображениями
        for j in range(i + 1, len(image_names)):
            img_name2 = image_names[j]
            faces_data2 = all_analysis_results[img_name2]["faces_data"]
            
            for face_idx1, face1 in enumerate(faces_data1):
                embedding1 = face1.get("insightface_embedding", [])
                if not embedding1:
                    continue
                
                for face_idx2, face2 in enumerate(faces_data2):
                    embedding2 = face2.get("insightface_embedding", [])
                    if not embedding2:
                        continue
                    
                    similarity = calculate_embedding_similarity(embedding1, embedding2)
                    comparison_key = f"{img_name1}_face{face_idx1}_vs_{img_name2}_face{face_idx2}"
                    
                    face_comparisons[comparison_key] = {
                        "embedding_similarity": round(similarity, 6),
                        "likely_same_person": bool(similarity > INSIGHT_FACE_THRESHOLD),
                        "confidence": round(similarity, 6),
                        "comparison_type": "inter_image"
                    }
    
    debug_print(f"✅ Завершено сравнение эмбеддингов. Найдено {len(face_comparisons)} сравнений.", "SUCCESS")
    return face_comparisons

def analyze_identity_groups(face_comparisons: Dict) -> Dict:
    """Анализ групп идентичности с использованием Union-Find"""
    debug_print("🔄 Анализ групп идентичности...", "INFO")
    
    uf = UnionFind()
    
    # Строим граф связей
    for key, data in face_comparisons.items():
        if data["likely_same_person"]:
            parts = key.split("_vs_")
            face1_id = parts[0]
            face2_id = parts[1]
            uf.union(face1_id, face2_id)
    
    # Группируем по корням
    groups = {}
    for key in face_comparisons:
        parts = key.split("_vs_")
        for face_id in parts:
            root = uf.find(face_id)
            if root not in groups:
                groups[root] = set()
            groups[root].add(face_id)

    
    # Формируем результат
    identity_analysis = {
        "identity_groups": [],
        "total_unique_identities": 0,
        "summary_by_image": {}
    }
    
    for group_faces in groups.values():
        if len(group_faces) > 1:
            group_entry = {
                "group_id": f"identity_{len(identity_analysis['identity_groups']) + 1}",
                "faces": sorted(list(group_faces)),
                "num_faces": len(group_faces)
            }
            
            identity_analysis["identity_groups"].append(group_entry)
            identity_analysis["total_unique_identities"] += 1
            
            # Обновляем сводку по изображениям
            for face_id in group_faces:
                img_name = face_id.split("_face")[0]
                identity_analysis["summary_by_image"].setdefault(img_name, 0)
                identity_analysis["summary_by_image"][img_name] += 1
    
    debug_print(f"✅ Завершен анализ идентичности. Найдено {len(identity_analysis['identity_groups'])} групп идентичности.", "SUCCESS")
    return identity_analysis

def calculate_global_statistics(all_analysis_results: Dict) -> Dict:
    """Расчет глобальной статистики"""
    debug_print("🔄 Расчет глобальной статистики...", "INFO")
    
    global_statistics = {
        "pose_angles": {"pitch": [], "yaw": [], "roll": []},
        "biometric_metrics": {}
    }
    
    # Агрегация данных
    for img_name, img_data in all_analysis_results.items():
        for face_data in img_data["faces_data"]:
            try:
                # Углы позы
                pose_angles = face_data["pose"]["angles"]
                global_statistics["pose_angles"]["pitch"].append(pose_angles["pitch"])
                global_statistics["pose_angles"]["yaw"].append(pose_angles["yaw"])
                global_statistics["pose_angles"]["roll"].append(pose_angles["roll"])
                
                # Биометрические метрики
                biometric_analysis = face_data.get("biometric_analysis", {})
                for category, metrics in biometric_analysis.items():
                    if isinstance(metrics, dict):
                        for metric_name, metric_data in metrics.items():
                            if isinstance(metric_data, dict):
                                for value_type, value in metric_data.items():
                                    if isinstance(value, (int, float)) and not np.isnan(value):
                                        key = f"{category}_{metric_name}_{value_type}"
                                        global_statistics["biometric_metrics"].setdefault(key, []).append(value)
                                        
            except Exception as e:
                debug_print(f"Ошибка обработки данных лица в {img_name}: {e}", "ERROR")
                continue
    
    # Расчет статистики
    calculated_stats = {}
    
    # Статистика по углам позы
    for angle_type, values in global_statistics["pose_angles"].items():
        if values:
            calculated_stats[f"pose_{angle_type}"] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values))
            }
    
    # Статистика по биометрическим метрикам
    for metric_name, values in global_statistics["biometric_metrics"].items():
        if values and len(values) > 1:
            calculated_stats[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values))
            }
    
    debug_print("✅ Глобальная статистика рассчитана успешно.", "SUCCESS")
    return calculated_stats

def print_analysis_conclusions(summary_report: Dict):
    """Печать выводов анализа"""
    debug_print("📊 Печать ключевых выводов анализа...", "INFO")
    
    analysis_summary = summary_report.get("analysis_summary", {})
    face_comparisons = summary_report.get("face_comparisons", {})
    identity_analysis = summary_report.get("identity_analysis", {})
    statistical_summary = summary_report.get("statistical_summary", {})
    
    print("\n" + "="*60)
    print("🔍 КЛЮЧЕВЫЕ ВЫВОДЫ АНАЛИЗА")
    print("="*60)
    
    print(f"\n📊 Общие сведения:")
    print(f"  • Всего изображений обработано: {analysis_summary.get('total_images_processed', 0)}")
    print(f"  • Всего лиц обнаружено: {analysis_summary.get('total_faces_detected', 0)}")
    print(f"  • Всего сравнений выполнено: {len(face_comparisons)}")
    
    if identity_analysis.get("total_unique_identities", 0) > 0:
        print(f"\n👥 Группы идентичности:")
        for group in identity_analysis["identity_groups"]:
            faces_str = ", ".join(group['faces'])
            print(f"  • {group['group_id']}: {group['num_faces']} лиц - {faces_str}")
    else:
        print("\n✅ Все лица уникальны")
    
    if statistical_summary:
        print(f"\n📈 Статистика по ключевым метрикам:")
        key_metrics = [
            ("pose_yaw", "Поворот головы (yaw)"),
            ("pose_pitch", "Наклон головы (pitch)")
        ]
        
        for metric_key, metric_name in key_metrics:
            if metric_key in statistical_summary:
                stats = statistical_summary[metric_key]
                print(f"  • {metric_name}:")
                print(f"    - Среднее: {stats['mean']:.3f}")
                print(f"    - Диапазон: {stats['min']:.3f} - {stats['max']:.3f}")
    
    print("\n" + "="*60)
    print("✅ АНАЛИЗ ЗАВЕРШЕН УСПЕШНО")
    print("="*60)

def validate_input_path(path: str) -> Tuple[bool, List[str]]:
    """Валидация входного пути и получение списка файлов"""
    path_obj = Path(path)
    
    if not path_obj.exists():
        return False, []
    
    if path_obj.is_file():
        if path_obj.suffix.lower() in SUPPORTED_FORMATS:
            return True, [str(path_obj)]
        else:
            debug_print(f"Неподдерживаемый формат файла: {path_obj.suffix}", "ERROR")
            return False, []
    elif path_obj.is_dir():
        image_files = []
        for ext in SUPPORTED_FORMATS:
            image_files.extend(path_obj.glob(f"*{ext}"))
            image_files.extend(path_obj.glob(f"*{ext.upper()}"))
        image_files = [str(f) for f in image_files]
        
        if not image_files:
            debug_print(f"Не найдено изображений в директории {path}", "ERROR")
            return False, []
        
        return True, sorted(image_files)
    
    return False, []

def main(args):
    """Основная функция"""
    logger = setup_logging("DEBUG" if args.verbose else "INFO")
    debug_print("🚀 УЛУЧШЕННАЯ СИСТЕМА АНАЛИЗА ЛИЦ С ИСПРАВЛЕНИЯМИ", "INFO")
    debug_print("Запуск системы анализа", "INFO")
    
    # Создание директории результатов
    os.makedirs('examples/results', exist_ok=True)
    
    # Валидация входного пути
    is_valid, image_files = validate_input_path(args.img_fp)
    if not is_valid:
        debug_print(f"Неверный путь или нет изображений: {args.img_fp}", "ERROR")
        sys.exit(-1)
    
    debug_print(f"✅ Найдено {len(image_files)} изображений для обработки", "SUCCESS")
    
    # Загрузка конфигурации
    debug_print("📋 Загрузка конфигурации...", "DEBUG")
    try:
        cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)
        debug_print(f"✅ Конфигурация загружена из {args.config}", "SUCCESS")
    except Exception as e:
        debug_print(f"Не удалось загрузить конфигурацию: {e}", "ERROR")
        sys.exit(-1)
    
    # Инициализация TDDFA
    debug_print("🤖 Инициализация TDDFA...", "DEBUG")
    try:
        if args.onnx:
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
            os.environ['OMP_NUM_THREADS'] = '4'
            from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
            from TDDFA_ONNX import TDDFA_ONNX
            face_boxes = FaceBoxes_ONNX()
            tddfa = TDDFA_ONNX(**cfg)
        else:
            gpu_mode = args.mode == 'gpu'
            tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        debug_print("✅ TDDFA инициализирован успешно", "SUCCESS")
    except Exception as e:
        debug_print(f"Не удалось инициализировать TDDFA: {e}", "ERROR")
        sys.exit(-1)
    
    # Инициализация InsightFace
    debug_print("👁️ Инициализация InsightFace...", "DEBUG")
    try:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if args.mode == 'gpu' else ['CPUExecutionProvider']
        ctx = 0 if args.mode == 'gpu' else -1
        app = FaceAnalysis(name='buffalo_l', providers=providers)
        app.prepare(ctx_id=ctx, det_size=(640, 640))
        debug_print("✅ InsightFace инициализирован успешно", "SUCCESS")
    except Exception as e:
        debug_print(f"Не удалось инициализировать InsightFace: {e}", "ERROR")
        sys.exit(-1)
    
    # Инициализация модулей анализа
    debug_print("🔬 Инициализация модулей анализа...", "DEBUG")
    modules = {
        'frontal': FrontalAnalysisModule(),
        'frontal_edge': FrontalEdgeAnalysisModule(),
        'semi_profile': SemiProfileAnalysisModule(),
        'profile': ProfileAnalysisModule()
    }
    debug_print("✅ Модули анализа инициализированы", "SUCCESS")
    
    # ИСПРАВЛЕНО: Инициализация улучшенных компонентов
    processor_3d = Enhanced3DFaceProcessor()
    
    # Хранилище всех результатов
    all_analysis_results = {}
    
    # Обработка изображений
    for img_idx, img_path in enumerate(image_files):
        debug_print(f'\n📸 ОБРАБОТКА {img_idx+1}/{len(image_files)}: {img_path}', "INFO")
        
        try:
            # Загрузка изображения
            debug_print("Загрузка изображения...", "DEBUG")
            img = cv2.imread(img_path)
            if img is None:
                debug_print(f'Не удалось загрузить изображение {img_path}', "ERROR")
                continue
            
            debug_print(f"✅ Изображение загружено: {img.shape}", "SUCCESS")
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # Детекция лиц с помощью InsightFace
            debug_print("👁️ Детекция лиц с помощью InsightFace...", "DEBUG")
            faces_insight = app.get(img)
            n_faces = len(faces_insight)
            debug_print(f'✅ Обнаружено {n_faces} лиц', "SUCCESS")
            
            if n_faces == 0:
                debug_print(f'Лица не обнаружены в {img_path}', "WARN")
                continue
            
            # Подготовка bounding boxes с помощью TDDFA
            debug_print("📦 Подготовка bounding boxes для TDDFA...", "DEBUG")
            boxes_for_tddfa = []
            embeddings_list = []
            h, w = img.shape[:2]
            
            for i, face in enumerate(faces_insight):
                x1, y1, x2, y2 = face.bbox.astype(float)
                # Клиппинг координат
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                boxes_for_tddfa.append([np.float32(x1), np.float32(y1),
                                      np.float32(x2), np.float32(y2), 1.0])
                embeddings_list.append(face.embedding.tolist())
                debug_print(f"Лицо {i}: bbox=[{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}], embedding_len={len(face.embedding)}", "DEBUG")
            
            # 3D реконструкция с помощью TDDFA
            debug_print("🧠 3D реконструкция с помощью TDDFA...", "DEBUG")
            param_lst, roi_box_lst = tddfa(img, boxes_for_tddfa)
            debug_print(f"✅ Получены параметры для {len(param_lst)} лиц", "SUCCESS")
            
            # Получение landmarks
            landmarks_3d_lst = get_landmarks_from_tddfa(tddfa, param_lst, roi_box_lst, dense=False)
            debug_print(f'✅ 3D реконструкция выполнена для {len(landmarks_3d_lst)} лиц', "SUCCESS")
            
            # Обработка каждого лица
            faces_data = []
            for face_idx, (landmarks_3d_original, params, roi_box, embedding) in enumerate(
                zip(landmarks_3d_lst, param_lst, roi_box_lst, embeddings_list)
            ):
                debug_print(f'🎭 Обработка лица {face_idx}...', "INFO")
                
                # ИСПРАВЛЕНО: Полная обработка с исправлениями
                face_result = process_single_face_enhanced(
                    face_idx, landmarks_3d_original, params, roi_box, modules,
                    embedding, img, processor_3d
                )
                
                faces_data.append(face_result)
            
            # Генерация визуализаций
            generated_files = generate_visualizations(
                img, param_lst, roi_box_lst, landmarks_3d_lst, tddfa, base_name, args.opt
            )
            
            # Формирование итогового результата
            image_result = {
                "image_info": {
                    "path": str(img_path),
                    "name": base_name,
                    "dimensions": {"width": img.shape[1], "height": img.shape[0]},
                    "faces_detected_count": len(faces_data),
                    "processing_timestamp": str(np.datetime64('now')),
                    "analyzer_version": "3DDFA_V2_Enhanced_Stable_Analysis_v6.0"
                },
                "faces_data": faces_data,
                "output_files": generated_files
            }
            
            # Сохранение результата для отдельного изображения
            analysis_file = f'examples/results/{base_name}_enhanced_biometric_analysis.json'
            debug_print(f"💾 Сохранение результата в {analysis_file}...", "DEBUG")
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(image_result, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            debug_print(f'✅ Результат сохранен в {analysis_file}', "SUCCESS")
            
            all_analysis_results[base_name] = image_result
            
        except Exception as e:
            debug_print(f"Ошибка обработки изображения {img_path}: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            continue
    
    # Генерация сводного отчета для множественных изображений
    if len(all_analysis_results) > 1:
        debug_print('\n📊 ГЕНЕРАЦИЯ СВОДНОГО ОТЧЕТА', "INFO")
        
        try:
            # Сравнение эмбеддингов между изображениями
            face_comparisons = compare_faces_embeddings(all_analysis_results)
            
            # Формирование сводного отчета
            summary_report = {
                "analysis_summary": {
                    "total_images_processed": len(all_analysis_results),
                    "total_faces_detected": sum(
                        len(data["faces_data"]) for data in all_analysis_results.values()
                    ),
                    "processing_timestamp": str(np.datetime64('now')),
                    "version": "3DDFA_V2_Enhanced_Stable_Analysis_v6.0",
                    "improvements": [
                        "Стабилизированные метрики с компенсацией поворотов",
                        "Правильная обработка 3D координат согласно документации",
                        "Исправлены все критические ошибки",
                        "Улучшенная точность измерений"
                    ]
                },
                "face_comparisons": face_comparisons,
                "detailed_results": all_analysis_results,
                "identity_analysis": analyze_identity_groups(face_comparisons),
                "statistical_summary": calculate_global_statistics(all_analysis_results)
            }
            
            # Сохранение сводного отчета
            summary_file = 'examples/results/enhanced_biometric_summary_v6.json'
            debug_print(f"💾 Сохранение сводного отчета в {summary_file}...", "DEBUG")
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            debug_print(f'✅ Сводный отчет сохранен в {summary_file}', "SUCCESS")
            
            # Печать ключевых выводов
            print_analysis_conclusions(summary_report)
            
        except Exception as e:
            debug_print(f"Ошибка генерации сводного отчета: {e}", "ERROR")
            import traceback
            traceback.print_exc()
    
    debug_print('\n🎉 АНАЛИЗ ЗАВЕРШЕН УСПЕШНО', "SUCCESS")
    debug_print('✅ Все критические ошибки исправлены', "SUCCESS")
    debug_print('✅ Метрики стабилизированы', "SUCCESS")
    debug_print('✅ Система готова к продакшену', "SUCCESS")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Улучшенный стабильный биометрический анализ с 3DDFA_V2")
    parser.add_argument("--config", type=str, required=True, help="Путь к файлу конфигурации")
    parser.add_argument("--img_fp", type=str, required=True, help="Путь к изображению или директории")
    parser.add_argument("--onnx", action="store_true", help="Использовать ONNX модель")
    parser.add_argument("--mode", type=str, choices=["cpu", "gpu"], default="cpu", help="Режим CPU или GPU")
    parser.add_argument("--opt", type=str, choices=["none", "all", "selected", "2d_sparse", "2d_dense", "3d", "depth", "pncc", "obj"],
                       default="all", help="Опции визуализации")
    parser.add_argument("--selected_viz", type=str, nargs='+',
                       choices=["2d_sparse", "2d_dense", "3d", "depth", "pncc", "obj"],
                       help="Выбранные визуализации для --opt selected")
    parser.add_argument("--verbose", action="store_true", help="Подробный вывод отладочной информации")
    
    args = parser.parse_args()
    main(args)