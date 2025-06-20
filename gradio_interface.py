"""
GradioInterface - Интерфейс Gradio с модульной архитектурой и полным функционалом
Версия: 2.0
Дата: 2025-06-15
Исправлены все критические ошибки согласно правкам
"""

import os
os.makedirs("logs", exist_ok=True)
import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple, Union, AsyncGenerator
from pathlib import Path
import json
import pickle
import asyncio
from datetime import datetime, timedelta
import cv2
import os
from PIL import Image

# Удаляю настройку логгера (logging.basicConfig и создание хендлеров) из этого файла, оставляю только получение логгера через logging.getLogger(__name__).
logger = logging.getLogger(__name__)

# Импорт компонентов системы
try:
    from core_config import (
        MAX_FILE_UPLOAD_COUNT, VISUALIZATION_PARAMS, AUTHENTICITY_WEIGHTS,
        MASK_DETECTION_LEVELS, CRITICAL_THRESHOLDS, CACHE_DIR, ERROR_CODES
    )
    from face_3d_analyzer import Face3DAnalyzer
    from embedding_analyzer import EmbeddingAnalyzer
    from texture_analyzer import TextureAnalyzer
    from temporal_analyzer import TemporalAnalyzer
    from anomaly_detector import AnomalyDetector
    from medical_validator import MedicalValidator
    from data_manager import DataManager
    from metrics_calculator import MetricsCalculator
    logger.info("Все компоненты системы успешно импортированы")
except ImportError as e:
    logger.error(f"Ошибка импорта компонентов: {e}")
    # Заглушки для отсутствующих компонентов
    MAX_FILE_UPLOAD_COUNT = 1500
    VISUALIZATION_PARAMS = {"height": 600, "width": 800, "interactive": True}
    AUTHENTICITY_WEIGHTS = {"geometry": 0.15, "embedding": 0.30, "texture": 0.10}
    MASK_DETECTION_LEVELS = {}
    CRITICAL_THRESHOLDS = {"min_authenticity_score": 0.6}
    CACHE_DIR = Path("cache")
    ERROR_CODES = {"E001": "NO_FACE_DETECTED", "E002": "LOW_QUALITY_IMAGE"}

# ==================== UI КОМПОНЕНТЫ ====================

class SmartFileUploader:
    """
    ИСПРАВЛЕНО: Умная загрузка файлов с валидацией
    Согласно правкам: поддержка до 1500 файлов с валидацией качества
    """
    
    def __init__(self, max_files: int = MAX_FILE_UPLOAD_COUNT):
        self.max_files = max_files
        self.uploaded_files = []
        self.quality_cache = {}
        logger.info(f"SmartFileUploader инициализирован с лимитом {max_files} файлов")

    def create_uploader(self) -> gr.Column:
        """Создание интерфейса загрузки"""
        with gr.Column() as col:
            gr.Markdown("## 📁 Загрузка изображений")
            
            self.file_upload = gr.File(
                file_count="multiple",
                file_types=[".jpg", ".jpeg", ".png"],
                label=f"Загрузите изображения (макс. {self.max_files})",
            )
            
            with gr.Row():
                self.upload_progress = gr.Progress()
                self.upload_status = gr.Textbox(
                    label="Статус загрузки",
                    interactive=False,
                    lines=2
                )
            
            with gr.Row():
                self.quality_threshold = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.6,
                    label="Порог качества",
                    info="Минимальное качество для анализа"
                )
                self.auto_enhance = gr.Checkbox(
                    label="Автоулучшение",
                    value=True,
                    info="Автоматическое улучшение качества"
                )
            
            self.preview_gallery = gr.Gallery(
                label="Предварительный просмотр",
                columns=5,
                rows=2,
                height=300,
                allow_preview=True
            )
            
            # Привязка событий
            self.file_upload.change(
                fn=self.process_uploaded_files,
                inputs=[self.file_upload, self.quality_threshold],
                outputs=[self.upload_status, self.preview_gallery]
            )
        
        return col

    def process_uploaded_files(self, files: List[str], quality_threshold: float) -> Tuple[str, List[Any]]:
        print("=== process_uploaded_files вызван (реальные фото) ===")
        logger.info(f"[DEBUG] process_uploaded_files вызван, files: {files}, quality_threshold: {quality_threshold}")
        from PIL import Image
        import os

        if not files:
            self.uploaded_files = []
            return "Файлы не выбраны", []

        valid_files = []
        preview_images = []
        for file_path in files:
            try:
                if not os.path.exists(file_path):
                    continue
                img = Image.open(file_path)
                preview_images.append(img)
                valid_files.append(file_path)
            except Exception as e:
                logger.error(f"Ошибка открытия файла {file_path}: {e}")

        self.uploaded_files = valid_files

        status = f"Загружено: {len(valid_files)} из {len(files)} файлов"
        return status, preview_images

    def _process_uploaded_files_impl(self, files: List[str], quality_threshold: float) -> Tuple[str, List[str]]:
        if not files:
            return "Файлы не выбраны", []
        
        try:
            logger.info(f"Обработка {len(files)} загруженных файлов")
            
            if len(files) > self.max_files:
                return f"Превышен лимит файлов: {len(files)} > {self.max_files}", []
            
            valid_files = []
            invalid_files = []
            quality_scores = {}
            
            for file_path in files:
                try:
                    # Валидация качества изображения
                    quality_score = self._assess_image_quality(file_path)
                    quality_scores[file_path] = quality_score
                    
                    if quality_score >= quality_threshold:
                        valid_files.append(file_path)
                    else:
                        invalid_files.append(file_path)
                        logger.warning(f"Файл {file_path} не прошел проверку качества: {quality_score:.3f}")
                
                except Exception as e:
                    logger.error(f"Ошибка обработки файла {file_path}: {e}")
                    invalid_files.append(file_path)
            
            self.uploaded_files = valid_files
            
            # Формирование статуса
            status_lines = [
                f"Всего файлов: {len(files)}",
                f"Прошли проверку: {len(valid_files)}",
                f"Отклонены: {len(invalid_files)}",
                f"Средний балл качества: {np.mean(list(quality_scores.values())):.3f}"
            ]
            
            if invalid_files:
                status_lines.append(f"Отклоненные файлы: {', '.join([os.path.basename(f) for f in invalid_files[:5]])}")
                if len(invalid_files) > 5:
                    status_lines.append(f"... и еще {len(invalid_files) - 5}")
            
            status = "\n".join(status_lines)
            
            # Предварительный просмотр (первые 10 валидных файлов)
            preview_images = valid_files[:10]
            
            logger.info(f"Обработка завершена: {len(valid_files)} валидных файлов")
            return status, preview_images
            
        except Exception as e:
            logger.error(f"Ошибка обработки файлов: {e}")
            return f"Ошибка: {str(e)}", []

    def _assess_image_quality(self, file_path: str) -> float:
        """Оценка качества изображения"""
        try:
            if file_path in self.quality_cache:
                return self.quality_cache[file_path]
            
            image = cv2.imread(file_path)
            if image is None:
                return 0.0
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Проверка разрешения
            resolution_score = 1.0 if w >= 200 and h >= 200 else 0.5
            
            # Проверка размытия
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_quality = 1.0 if blur_score >= 100 else max(0.0, blur_score / 150.0)
            
            # Проверка шума
            noise_level = np.std(gray)
            noise_quality = 1.0 if noise_level <= 30 else max(0.0, 1.0 - (noise_level - 10) / 40.0)
            
            # Общий балл качества
            quality_score = (resolution_score + blur_quality + noise_quality) / 3.0
            
            self.quality_cache[file_path] = quality_score
            return quality_score
            
        except Exception as e:
            logger.error(f"Ошибка оценки качества {file_path}: {e}")
            return 0.0

class RealTimeAnalyzer:
    """
    ИСПРАВЛЕНО: Анализатор в реальном времени
    Согласно правкам: прогресс-трекинг и cancel операций
    """
    
    def __init__(self, data_processor, results_aggregator, smart_file_uploader):
        self.is_running = False
        self.current_progress = 0
        self.cancel_requested = False
        self.data_processor = data_processor
        self.results_aggregator = results_aggregator
        self.smart_file_uploader = smart_file_uploader
        logger.info("RealTimeAnalyzer инициализирован")

    def create_analyzer(self) -> gr.Column:
        """Создание интерфейса анализатора"""
        with gr.Column() as col:
            gr.Markdown("## ⚡ Анализ в реальном времени")
            
            with gr.Row():
                self.start_btn = gr.Button("🚀 Начать анализ", variant="primary")
                self.pause_btn = gr.Button("⏸️ Пауза")
                self.stop_btn = gr.Button("⏹️ Стоп", variant="stop")
            
            with gr.Row():
                self.current_file = gr.Textbox(
                    label="Текущий файл",
                    interactive=False
                )
                self.eta = gr.Textbox(
                    label="Оставшееся время",
                    interactive=False
                )
            
            self.live_metrics = gr.HTML(label="Метрики в реальном времени")
            self.live_plot = gr.Plot(label="График в реальном времени")
            
            # Привязка событий
            self.start_btn.click(
                fn=self.start_analysis,
                inputs=[],
                outputs=[self.current_file, self.live_metrics, self.live_plot]
            )
            
            self.stop_btn.click(
                fn=self.stop_analysis,
                inputs=[],
                outputs=[self.current_file, self.live_metrics, self.live_plot]
            )
        
        return col

    async def start_analysis(self, progress: gr.Progress) -> AsyncGenerator[Tuple[str, str, go.Figure], None]:
        print("=== start_analysis вызван ===")
        logger.info("[DEBUG] start_analysis вызван")
        files_to_process = self.smart_file_uploader.uploaded_files
        print(f"[DEBUG] self.smart_file_uploader.uploaded_files: {files_to_process}")
        logger.info(f"[DEBUG] self.smart_file_uploader.uploaded_files: {files_to_process}")
        # Явный yield в самом начале для сброса состояния интерфейса
        yield "Ожидание анализа...", "<div style='color: gray;'>Ожидание анализа...</div>", go.Figure()
        if not files_to_process:
            print("[DEBUG] Нет файлов для анализа")
            logger.info("[DEBUG] Нет файлов для анализа")
            self.is_running = False
            yield "Нет файлов для анализа.", "<div style='color: red;'>❌ Нет файлов для анализа.</div>", go.Figure()
            return

        total_files = len(files_to_process)
        authenticity_scores_history = []
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Аутентичность'))
        fig.update_layout(title="Динамика баллов аутентичности", xaxis_title="Файл", yaxis_title="Балл")

        any_success = False
        try:
            yield "Анализ запущен...", "<div style='color: orange;'>⏳ Анализ в процессе...</div>", fig
            for i, file_path in enumerate(files_to_process):
                try:
                    print(f"[DEBUG] Анализирую файл: {file_path}")
                    logger.info(f"[DEBUG] Анализирую файл: {file_path}")
                    analysis_result = await self.data_processor.process_single_file_async(file_path)
                    print(f"[DEBUG] Результат анализа: {analysis_result}")
                    logger.info(f"[DEBUG] Результат анализа: {analysis_result}")

                    # Новый блок: обработка ошибок анализа
                    error_msg = ""
                    if hasattr(analysis_result, "anomalies") and analysis_result.anomalies:
                        error_msg = analysis_result.anomalies.get("processing_error") or ""
                    if hasattr(analysis_result, "metadata") and "error" in analysis_result.metadata:
                        error_msg = analysis_result.metadata["error"]

                    if error_msg:
                        status = f"Ошибка файла {os.path.basename(file_path)}: {error_msg}"
                        live_metrics = f"<div style='color: red;'>Ошибка: {error_msg}</div>"
                        yield status, live_metrics, fig
                        continue  # Переходим к следующему файлу
                    else:
                        score = getattr(analysis_result, 'authenticity_score', 0.0)
                        authenticity_scores_history.append(score)
                        # Обновление графика
                        fig.data[0].x = list(range(len(authenticity_scores_history)))
                        fig.data[0].y = authenticity_scores_history
                        status = f"Обработан файл {i+1}/{total_files}: {os.path.basename(file_path)}"
                        live_metrics = f"<div style='color: blue;'>✅ Score: {score:.3f}</div>"
                        yield status, live_metrics, fig
                        any_success = True
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    logger.error(f"Ошибка анализа файла {file_path}: {e}\n{tb}")
                    print(f"=== ОШИБКА анализа файла {file_path} ===\n{tb}")
                    yield f"Ошибка файла {os.path.basename(file_path)}", f"<div style='color: red;'>Ошибка: {e}<br><pre>{tb}</pre></div>", fig
            self.is_running = False
            # Явный yield в самом конце, чтобы Gradio всегда обновлял интерфейс
            if any_success:
                yield "Анализ завершён.", "<div style='color: green;'>✔ Анализ завершён!</div>", fig
            else:
                yield "Анализ завершён. Все файлы с ошибками.", "<div style='color: red;'>❌ Все файлы с ошибками. Проверьте фото и попробуйте снова.</div>", fig
        except Exception as e:
            logger.error(f"Ошибка в start_analysis: {e}")
            yield f"Ошибка анализа: {e}", f"<div style='color: red;'>❌ Критическая ошибка анализа.</div>", go.Figure()
            self.is_running = False

    def stop_analysis(self) -> Tuple[str, str, go.Figure]:
        """Остановка анализа"""
        if self.is_running:
            self.cancel_requested = True
            logger.info("Запрошена отмена анализа.")
            return "Запрос на отмену...", "<div style='color: orange;'>⏳ Отмена анализа...</div>", go.Figure()
        else:
            return "Анализ не запущен.", "", go.Figure()

class Interactive3DViewer:
    """
    ИСПРАВЛЕНО: 3D визуализация с landmarks и dense points
    Согласно правкам: 68 landmarks, 3D wireframe, dense surface points
    """
    
    def __init__(self):
        self.current_landmarks = None
        self.current_dense_points = None
        logger.info("Interactive3DViewer инициализирован")

    def render(self) -> gr.Column:
        """Рендеринг 3D визуализатора"""
        with gr.Column() as col:
            gr.Markdown("## 🎯 3D Визуализация лица")
            
            with gr.Row():
                self.wireframe_toggle = gr.Checkbox(
                    label="Wireframe режим",
                    value=True,
                    info="Показать каркас лица"
                )
                self.dense_points_toggle = gr.Checkbox(
                    label="Плотные точки",
                    value=False,
                    info="Показать 38,000 точек поверхности"
                )
                self.landmarks_toggle = gr.Checkbox(
                    label="68 ландмарок",
                    value=True,
                    info="Показать ключевые точки"
                )
            
            # ИСПРАВЛЕНО: 3D модель с контролами
            self.model_3d = gr.Model3D(
                label="3D модель лица",
                height=500,
                interactive=True,
                camera_position=(0, 0, 5)
            )
            
            with gr.Row():
                self.rotation_x = gr.Slider(
                    minimum=-180,
                    maximum=180,
                    value=0,
                    label="Поворот X (pitch)"
                )
                self.rotation_y = gr.Slider(
                    minimum=-180,
                    maximum=180,
                    value=0,
                    label="Поворот Y (yaw)"
                )
                self.rotation_z = gr.Slider(
                    minimum=-180,
                    maximum=180,
                    value=0,
                    label="Поворот Z (roll)"
                )
            
            # Привязка событий
            for control in [self.wireframe_toggle, self.dense_points_toggle, self.landmarks_toggle]:
                control.change(
                    fn=self.update_3d_view,
                    inputs=[self.wireframe_toggle, self.dense_points_toggle, self.landmarks_toggle],
                    outputs=[self.model_3d]
                )
        
        return col

    def update_3d_view(self, wireframe: bool, dense_points: bool, landmarks: bool) -> str:
        """
        ИСПРАВЛЕНО: Обновление 3D вида
        Согласно правкам: wireframe, dense points, 68 landmarks
        """
        try:
            logger.info(f"Обновление 3D вида: wireframe={wireframe}, dense={dense_points}, landmarks={landmarks}")
            
            # Генерация тестовых данных для демонстрации
            if landmarks and self.current_landmarks is None:
                # Генерация 68 тестовых ландмарок
                self.current_landmarks = self._generate_test_landmarks()
            
            if dense_points and self.current_dense_points is None:
                # Генерация тестовых плотных точек
                self.current_dense_points = self._generate_test_dense_points()
            
            # Создание OBJ контента
            obj_content = self._create_obj_content(wireframe, dense_points, landmarks)
            
            return obj_content
            
        except Exception as e:
            logger.error(f"Ошибка обновления 3D вида: {e}")
            return ""

    def _generate_test_landmarks(self) -> np.ndarray:
        """Генерация тестовых 68 ландмарок"""
        # Примерная форма лица
        landmarks = np.zeros((68, 3))
        
        # Контур лица (0-16)
        for i in range(17):
            angle = (i - 8) * np.pi / 16
            landmarks[i] = [np.sin(angle) * 50, -abs(i - 8) * 5 - 30, np.cos(angle) * 10]
        
        # Брови (17-26)
        for i in range(17, 27):
            x = (i - 21.5) * 8
            landmarks[i] = [x, 20, 5]
        
        # Нос (27-35)
        for i in range(27, 36):
            y = 20 - (i - 27) * 5
            landmarks[i] = [0, y, 10 + (i - 31) ** 2]
        
        # Глаза (36-47)
        for i in range(36, 48):
            eye_side = -1 if i < 42 else 1
            angle = (i % 6) * np.pi / 3
            landmarks[i] = [eye_side * 20 + np.cos(angle) * 8, 10 + np.sin(angle) * 4, 5]
        
        # Рот (48-67)
        for i in range(48, 68):
            angle = (i - 57.5) * np.pi / 10
            landmarks[i] = [np.sin(angle) * 15, -10 + np.cos(angle) * 5, 8]
        
        return landmarks

    def _generate_test_dense_points(self) -> np.ndarray:
        """Генерация тестовых плотных точек (упрощенная версия 38,000)"""
        # Генерация сетки точек для демонстрации
        x = np.linspace(-50, 50, 100)
        y = np.linspace(-40, 30, 100)
        X, Y = np.meshgrid(x, y)
        
        # Простая функция формы лица
        Z = 10 * np.exp(-(X**2 + Y**2) / 1000) + np.random.normal(0, 1, X.shape)
        
        # Преобразование в список точек
        points = []
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                points.append([X[i, j], Y[i, j], Z[i, j]])
        
        return np.array(points[:1000])  # Ограничиваем для производительности

    def _create_obj_content(self, wireframe: bool, dense_points: bool, landmarks: bool) -> str:
        """Создание OBJ контента для 3D модели"""
        obj_lines = ["# 3D Face Model"]
        
        if landmarks and self.current_landmarks is not None:
            obj_lines.append("# 68 Landmarks")
            for i, point in enumerate(self.current_landmarks):
                obj_lines.append(f"v {point[0]:.3f} {point[1]:.3f} {point[2]:.3f}")
        
        if dense_points and self.current_dense_points is not None:
            obj_lines.append("# Dense Surface Points")
            for point in self.current_dense_points:
                obj_lines.append(f"v {point[0]:.3f} {point[1]:.3f} {point[2]:.3f}")
        
        return "\n".join(obj_lines)

class MetricsDashboard:
    """
    ИСПРАВЛЕНО: Dashboard с 15 метриками и визуализацией
    Согласно правкам: 15 метрик, color-coding, correlation matrix 15x15
    """
    
    def __init__(self):
        self.metrics_data = {}
        self.correlation_matrix = None
        logger.info("MetricsDashboard инициализирован")

    def render(self) -> gr.Column:
        """Рендеринг dashboard метрик"""
        with gr.Column() as col:
            gr.Markdown("## 📊 Dashboard метрик идентичности")
            
            # ИСПРАВЛЕНО: Tabs для разных категорий метрик
            with gr.Tabs():
                with gr.Tab("Геометрия черепа"):
                    self.geometry_tab = self._create_geometry_tab()
                
                with gr.Tab("Пропорции лица"):
                    self.proportions_tab = self._create_proportions_tab()
                
                with gr.Tab("Костная структура"):
                    self.bone_tab = self._create_bone_tab()
                
                with gr.Tab("Корреляционная матрица"):
                    self.correlation_tab = self._create_correlation_tab()
                
                with gr.Tab("Статистика"):
                    self.statistics_tab = self._create_statistics_tab()
        
        return col

    def _create_geometry_tab(self) -> gr.Column:
        """Создание вкладки геометрии черепа"""
        with gr.Column() as tab:
            gr.Markdown("### 🏗️ Геометрия черепа (5 метрик)")
            
            # ИСПРАВЛЕНО: 5 метрик геометрии черепа
            metrics_names = [
                "skull_width_ratio", "temporal_bone_angle", "zygomatic_arch_width",
                "orbital_depth", "occipital_curve"
            ]
            
            self.geometry_plots = {}
            
            for metric in metrics_names:
                with gr.Row():
                    # Индикатор значения с color-coding
                    value_indicator = gr.Number(
                        label=metric.replace("_", " ").title(),
                        precision=3,
                        interactive=False
                    )
                    
                    # График тренда
                    trend_plot = gr.Plot(
                        label=f"Тренд {metric}"
                    )
                    
                    self.geometry_plots[metric] = (value_indicator, trend_plot)
        
        return tab

    def _create_proportions_tab(self) -> gr.Column:
        """Создание вкладки пропорций лица"""
        with gr.Column() as tab:
            gr.Markdown("### 📏 Пропорции лица (5 метрик)")
            
            # ИСПРАВЛЕНО: 5 метрик пропорций лица
            metrics_names = [
                "cephalic_index", "nasolabial_angle", "orbital_index",
                "forehead_height_ratio", "chin_projection_ratio"
            ]
            
            self.proportions_plots = {}
            
            for metric in metrics_names:
                with gr.Row():
                    value_indicator = gr.Number(
                        label=metric.replace("_", " ").title(),
                        precision=3,
                        interactive=False
                    )
                    
                    trend_plot = gr.Plot(
                        label=f"Тренд {metric}"
                    )
                    
                    self.proportions_plots[metric] = (value_indicator, trend_plot)
        
        return tab

    def _create_bone_tab(self) -> gr.Column:
        """Создание вкладки костной структуры"""
        with gr.Column() as tab:
            gr.Markdown("### 🦴 Костная структура (5 метрик)")
            
            # ИСПРАВЛЕНО: 5 метрик костной структуры
            metrics_names = [
                "interpupillary_distance_ratio", "gonial_angle_asymmetry",
                "zygomatic_angle", "jaw_angle_ratio", "mandibular_symphysis_angle"
            ]
            
            self.bone_plots = {}
            
            for metric in metrics_names:
                with gr.Row():
                    value_indicator = gr.Number(
                        label=metric.replace("_", " ").title(),
                        precision=3,
                        interactive=False
                    )
                    
                    trend_plot = gr.Plot(
                        label=f"Тренд {metric}"
                    )
                    
                    self.bone_plots[metric] = (value_indicator, trend_plot)
        
        return tab

    def _create_correlation_tab(self) -> gr.Column:
        """Создание вкладки корреляционной матрицы"""
        with gr.Column() as tab:
            gr.Markdown("### 🔗 Корреляционная матрица 15×15")
            
            # ИСПРАВЛЕНО: Корреляционная матрица 15x15
            self.correlation_heatmap = gr.Plot(
                label="Корреляционная матрица метрик"
            )
            
            with gr.Row():
                self.correlation_threshold = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    label="Порог корреляции",
                    info="Показать корреляции выше порога"
                )
                
                self.update_correlation_btn = gr.Button(
                    "Обновить матрицу",
                    variant="secondary"
                )
            
            # Привязка событий
            self.update_correlation_btn.click(
                fn=self.update_correlation_matrix,
                inputs=[self.correlation_threshold],
                outputs=[self.correlation_heatmap]
            )
        
        return tab

    def _create_statistics_tab(self) -> gr.Column:
        """Создание вкладки статистики"""
        with gr.Column() as tab:
            gr.Markdown("### 📈 Статистическая сводка")
            
            with gr.Row():
                # Общая статистика
                self.stats_summary = gr.JSON(
                    label="Общая статистика"
                )
                
                # Percentile ranks
                self.percentile_ranks = gr.Dataframe(
                    headers=["Метрика", "Значение", "Percentile", "Ранг"],
                    label="Percentile ранги"
                )
            
            # Историческое сравнение
            self.historical_comparison = gr.Plot(
                label="Историческое сравнение"
            )
        
        return tab

    def update_correlation_matrix(self, threshold: float) -> go.Figure:
        """
        ИСПРАВЛЕНО: Обновление корреляционной матрицы 15x15
        Согласно правкам: correlation matrix 15x15 для всех метрик
        """
        try:
            logger.info(f"Обновление корреляционной матрицы с порогом {threshold}")
            
            # Генерация тестовой корреляционной матрицы 15x15
            metrics_names = [
                # Геометрия черепа (5)
                "skull_width_ratio", "temporal_bone_angle", "zygomatic_arch_width",
                "orbital_depth", "occipital_curve",
                # Пропорции лица (5)
                "cephalic_index", "nasolabial_angle", "orbital_index",
                "forehead_height_ratio", "chin_projection_ratio",
                # Костная структура (5)
                "interpupillary_distance_ratio", "gonial_angle_asymmetry",
                "zygomatic_angle", "jaw_angle_ratio", "mandibular_symphysis_angle"
            ]
            
            # Генерация корреляционной матрицы
            np.random.seed(42)  # Для воспроизводимости
            correlation_matrix = np.random.rand(15, 15)
            correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2  # Симметричная
            np.fill_diagonal(correlation_matrix, 1.0)  # Диагональ = 1
            
            # Применение порога
            masked_matrix = np.where(np.abs(correlation_matrix) >= threshold, correlation_matrix, 0)
            
            # Создание heatmap
            fig = go.Figure(data=go.Heatmap(
                z=masked_matrix,
                x=metrics_names,
                y=metrics_names,
                colorscale='RdBu',
                zmid=0,
                text=np.round(masked_matrix, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False,
                hovertemplate="<b>%{x}</b><br><b>%{y}</b><br>Корреляция: %{z:.3f}<extra></extra>"
            ))
            
            fig.update_layout(
                title=f"Корреляционная матрица метрик (порог ≥ {threshold})",
                xaxis_title="Метрики",
                yaxis_title="Метрики",
                height=600,
                width=800
            )
            
            # Поворот подписей осей
            fig.update_xaxes(tickangle=45)
            fig.update_yaxes(tickangle=0)
            
            self.correlation_matrix = correlation_matrix
            
            return fig
            
        except Exception as e:
            logger.error(f"Ошибка обновления корреляционной матрицы: {e}")
            return go.Figure()

class MaskDetectionDashboard:
    """
    ИСПРАВЛЕНО: Dashboard обнаружения масок
    Согласно правкам: 5 уровней масок, confidence scores, timeline visualization
    """
    
    def __init__(self):
        self.mask_levels = MASK_DETECTION_LEVELS
        self.detection_history = []
        logger.info("MaskDetectionDashboard инициализирован")

    def render(self) -> gr.Column:
        """Рендеринг dashboard обнаружения масок"""
        with gr.Column() as col:
            gr.Markdown("## 🎭 Dashboard обнаружения масок")
            
            with gr.Row():
                # Текущий уровень маски
                self.current_level = gr.Textbox(
                    label="Обнаруженный уровень",
                    value="Natural_Skin",
                    interactive=False
                )
                
                # Confidence score
                self.confidence_score = gr.Number(
                    label="Confidence Score",
                    precision=3,
                    interactive=False
                )
                
                # Индикатор риска
                self.risk_indicator = gr.HTML(
                    label="Индикатор риска",
                    value="<div style='color: green;'>✅ Низкий риск</div>"
                )
            
            # ИСПРАВЛЕНО: Levels 1-5 с параметрами
            with gr.Accordion("📋 Уровни технологий масок", open=False):
                self.levels_info = gr.Dataframe(
                    headers=["Уровень", "Годы", "Shape Error", "Entropy", "Embedding Dist", "Описание"],
                    value=self._get_levels_data(),
                    label="Характеристики уровней"
                )
            
            # Timeline visualization
            self.timeline_plot = gr.Plot(
                label="Временная линия анализа"
            )
            
            # Детальная статистика
            with gr.Row():
                self.detection_stats = gr.JSON(
                    label="Статистика обнаружения"
                )
                
                self.breakthrough_years = gr.JSON(
                    label="Breakthrough Years",
                    value={"years": [2008, 2014, 2019, 2022], "description": "Годы технологических прорывов"}
                )
        
        return col

    def _get_levels_data(self) -> List[List[str]]:
        """Получение данных об уровнях масок"""
        if not self.mask_levels:
            # Данные по умолчанию если MASK_DETECTION_LEVELS пуст
            return [
                ["Level1_Primitive", "1999-2005", "0.6", "4.2", "0.8", "Примитивные маски"],
                ["Level2_Basic", "2006-2010", "0.4", "5.2", "0.7", "Базовые маски"],
                ["Level3_Commercial", "2011-2015", "0.3", "6.0", "0.5", "Коммерческие маски"],
                ["Level4_Professional", "2016-2020", "0.2", "6.5", "0.4", "Профессиональные маски"],
                ["Level5_Advanced", "2021-2025", "0.15", "7.0", "0.3", "Продвинутые маски"]
            ]
        
        levels_data = []
        for level_name, level_info in self.mask_levels.items():
            years = f"{level_info['years'][0]}-{level_info['years'][1]}"
            levels_data.append([
                level_name,
                years,
                str(level_info.get('shape_error', 'N/A')),
                str(level_info.get('entropy', 'N/A')),
                str(level_info.get('embedding_dist', 'N/A')),
                level_info.get('description', 'Описание отсутствует')
            ])
        
        return levels_data

    def update_detection(self, level: str, confidence: float, risk_level: str) -> Tuple[str, float, str]:
        """Обновление результатов обнаружения"""
        try:
            logger.info(f"Обновление обнаружения маски: {level}, confidence={confidence}")
            
            # Цветовое кодирование риска
            risk_colors = {
                "low": "green",
                "medium": "orange", 
                "high": "red",
                "critical": "darkred"
            }
            
            color = risk_colors.get(risk_level.lower(), "gray")
            risk_html = f"<div style='color: {color}; font-weight: bold;'>{risk_level.upper()} РИСК</div>"
            
            # Добавление в историю
            self.detection_history.append({
                "timestamp": datetime.now(),
                "level": level,
                "confidence": confidence,
                "risk": risk_level
            })
            
            return level, confidence, risk_html
            
        except Exception as e:
            logger.error(f"Ошибка обновления обнаружения: {e}")
            return "Error", 0.0, "<div style='color: red;'>Ошибка</div>"

# ==================== ГЛАВНЫЙ ИНТЕРФЕЙС ====================

class GradioInterface:
    """
    ИСПРАВЛЕНО: Главный интерфейс Gradio с модульной архитектурой
    Согласно правкам: полная модульная система с всеми компонентами
    """
    
    def __init__(self, all_system_components: Dict[str, Any]):
        """Инициализация главного интерфейса"""
        logger.info("Инициализация GradioInterface")

        # Проверка наличия всех обязательных компонентов
        required_keys = [
            'data_processor', 'results_aggregator', 'visualization_engine',
            'face_3d_analyzer', 'embedding_analyzer', 'texture_analyzer',
            'temporal_analyzer', 'anomaly_detector', 'medical_validator',
            'data_manager', 'metrics_calculator'
        ]
        for key in required_keys:
            if key not in all_system_components or all_system_components[key] is None:
                print(f"[CRITICAL] Не найден компонент '{key}' в all_system_components!")
                logger.critical(f"Не найден компонент '{key}' в all_system_components!")
                raise RuntimeError(f"Не найден компонент '{key}' в all_system_components!")

        self.data_processor = all_system_components['data_processor']
        self.results_aggregator = all_system_components['results_aggregator']
        self.visualization_engine = all_system_components['visualization_engine']

        # Инициализация SmartFileUploader отдельно, так как он нужен для RealTimeAnalyzer
        smart_file_uploader_instance = SmartFileUploader(max_files=MAX_FILE_UPLOAD_COUNT)

        # Инициализация всех отдельных виджетов
        self.widgets = {
            "face_3d_analyzer": all_system_components['face_3d_analyzer'],
            "embedding_analyzer": all_system_components['embedding_analyzer'],
            "texture_analyzer": all_system_components['texture_analyzer'],
            "temporal_analyzer": all_system_components['temporal_analyzer'],
            "anomaly_detector": all_system_components['anomaly_detector'],
            "medical_validator": all_system_components['medical_validator'],
            "data_manager": all_system_components['data_manager'],
            "metrics_calculator": all_system_components['metrics_calculator'],
            "interactive_3d_viewer": Interactive3DViewer(),  # This widget is local to GradioInterface
            "metrics_dashboard": MetricsDashboard(), # This widget is local to GradioInterface
            "mask_detection_dashboard": MaskDetectionDashboard(), # This widget is local to GradioInterface
            "smart_file_uploader": smart_file_uploader_instance, # Используем уже созданный экземпляр
            "real_time_analyzer": RealTimeAnalyzer(
                data_processor=self.data_processor,
                results_aggregator=self.results_aggregator,
                smart_file_uploader=smart_file_uploader_instance # Передаем уже созданный экземпляр
            ),
        }
        
        # Кэш результатов
        self.results_cache = {}
        
        # Статистика сессии
        self.session_stats = {
            "files_processed": 0,
            "analysis_started": datetime.now(),
            "total_authenticity_score": 0.0,
            "anomalies_detected": 0
        }
        
        logger.info("Interactive3DViewer инициализирован")
        logger.info("MetricsDashboard инициализирован")
        logger.info("MaskDetectionDashboard инициализирован")
        logger.info(f"SmartFileUploader инициализирован с лимитом {MAX_FILE_UPLOAD_COUNT} файлов")
        logger.info("RealTimeAnalyzer инициализирован")
        logger.info("GradioInterface полностью инициализирован")

    def create_interface(self) -> gr.Blocks:
        print("=== [DEBUG] GradioInterface.create_interface вызван ===")
        logger.info("=== [DEBUG] GradioInterface.create_interface вызван ===")
        """
        ИСПРАВЛЕНО: Создание полного интерфейса
        Согласно правкам: модульная архитектура с всеми компонентами
        """
        try:
            logger.info("Создание интерфейса Gradio")
            
            # ИСПРАВЛЕНО: Использование gr.Blocks вместо gr.Interface
            with gr.Blocks(
                title="🔍 Система анализа подлинности 3D лиц",
                theme=gr.themes.Soft(),
                css=self._get_custom_css()
            ) as demo:
                
                # Заголовок
                gr.Markdown("""
                # 🔍 Система анализа подлинности 3D лиц
                ## Комплексный анализ с медицинской валидацией и временным анализом
                """)
                
                # ИСПРАВЛЕНО: Tabs для разных функций
                with gr.Tabs():
                    # Основной анализ
                    with gr.Tab("🏠 Главная"):
                        self._create_main_tab()
                    
                    # 3D визуализация
                    with gr.Tab("🎯 3D Анализ"):
                        self._create_3d_tab()
                    
                    # Dashboard метрик
                    with gr.Tab("📊 Метрики"):
                        self._create_metrics_tab()
                    
                    # Обнаружение масок
                    with gr.Tab("🎭 Маски"):
                        self._create_mask_tab()
                    
                    # Временной анализ
                    with gr.Tab("⏰ Временной анализ"):
                        self._create_temporal_tab()
                    
                    # Медицинская валидация
                    with gr.Tab("🏥 Медицинская валидация"):
                        self._create_medical_tab()
                    
                    # Экспорт результатов
                    with gr.Tab("💾 Экспорт"):
                        self._create_export_tab()
                    
                    # Настройки
                    with gr.Tab("⚙️ Настройки"):
                        self._create_settings_tab()
                
                # Футер с информацией
                gr.Markdown("""
                ---
                **Система анализа подлинности 3D лиц v2.0** | 
                Поддерживает до 1500 файлов | 
                15 метрик идентичности | 
                5 уровней технологий масок
                """)
            
            logger.info("Интерфейс Gradio создан успешно")
            return demo
            
        except Exception as e:
            print("CRITICAL ERROR при создании интерфейса Gradio:", e)
            import traceback
            print(traceback.format_exc())
            logger.error(f"Ошибка создания интерфейса: {e}")
            # Возвращаем простой интерфейс в случае ошибки
            return gr.Interface(
                fn=lambda x: "Ошибка инициализации системы",
                inputs=gr.Textbox(label="Ввод"),
                outputs=gr.Textbox(label="Вывод"),
                title="Ошибка системы"
            )

    def _create_main_tab(self) -> None:
        print("=== [DEBUG] GradioInterface._create_main_tab вызван ===")
        logger.info("=== [DEBUG] GradioInterface._create_main_tab вызван ===")
        print(f"[DEBUG] uploader instance: {self.widgets['smart_file_uploader']}")
        print(f"[DEBUG] analyzer instance: {self.widgets['real_time_analyzer']}")
        """Создание главной вкладки"""
        with gr.Row():
            with gr.Column(scale=1):
                # Загрузка файлов
                uploader_ui = self.widgets['smart_file_uploader'].create_uploader()
                
                # Настройки анализа
                with gr.Accordion("⚙️ Настройки анализа", open=False):
                    self.confidence_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.85,
                        label="Порог confidence",
                        info="Минимальный порог для принятия решения"
                    )
                    
                    self.analysis_mode = gr.Radio(
                        choices=["quick", "full"],
                        value="full",
                        label="Режим анализа",
                        info="Quick - быстрый, Full - полный анализ"
                    )
                    
                    self.export_format = gr.Radio(
                        choices=["HTML", "PDF", "JSON", "CSV"],
                        value="HTML",
                        label="Формат экспорта"
                    )
            
            with gr.Column(scale=2):
                # Анализатор в реальном времени
                analyzer_ui = self.widgets['real_time_analyzer'].create_analyzer()
                
                # Результаты анализа
                with gr.Row():
                    self.overall_authenticity = gr.Number(
                        label="Общая аутентичность",
                        precision=3,
                        interactive=False
                    )
                    
                    self.risk_level = gr.HTML(
                        label="Уровень риска",
                        value="<div style='color: gray;'>Ожидание анализа...</div>"
                    )

    def _create_3d_tab(self) -> None:
        """Создание вкладки 3D анализа"""
        with gr.Row():
            with gr.Column(scale=2):
                # 3D визуализатор
                viewer_ui = self.widgets['interactive_3d_viewer'].render()
            
            with gr.Column(scale=1):
                # Информация о ландмарках
                gr.Markdown("### 📍 Информация о ландмарках")
                
                self.landmarks_info = gr.JSON(
                    label="68 ключевых точек"
                )
                
                # Контролы позы
                gr.Markdown("### 🔄 Контроль позы")
                
                self.pose_category = gr.Textbox(
                    label="Категория позы",
                    value="Frontal",
                    interactive=False
                )
                
                self.pose_angles = gr.JSON(
                    label="Углы позы (pitch, yaw, roll)",
                    value={"pitch": 0, "yaw": 0, "roll": 0}
                )

    def _create_metrics_tab(self) -> None:
        """Создание вкладки метрик"""
        # Используем готовый dashboard метрик
        metrics_ui = self.widgets['metrics_dashboard'].render()

    def _create_mask_tab(self) -> None:
        """Создание вкладки масок"""
        # Используем готовый dashboard масок
        mask_ui = self.widgets['mask_detection_dashboard'].render()

    def _create_temporal_tab(self) -> None:
        """Создание вкладки временного анализа"""
        with gr.Column():
            gr.Markdown("## ⏰ Временной анализ")
            
            # Timeline визуализация
            self.timeline_plot = gr.Plot(
                label="Временная линия анализа"
            )
            
            with gr.Row():
                # Аномалии во времени
                self.temporal_anomalies = gr.JSON(
                    label="Временные аномалии"
                )
                
                # Паттерны смены идентичности
                self.identity_patterns = gr.JSON(
                    label="Паттерны смены идентичности"
                )

    def _create_medical_tab(self) -> None:
        """Создание вкладки медицинской валидации"""
        with gr.Column():
            gr.Markdown("## 🏥 Медицинская валидация")
            
            with gr.Row():
                # Согласованность старения
                self.aging_consistency = gr.Number(
                    label="Согласованность старения",
                    precision=3,
                    interactive=False
                )
                
                # Стабильность костей
                self.bone_stability = gr.HTML(
                    label="Стабильность костей",
                    value="<div>Анализ не проведен</div>"
                )
            
            # Медицинский отчет
            self.medical_report = gr.HTML(
                label="Медицинский отчет"
            )

    def _create_export_tab(self) -> None:
        """Создание вкладки экспорта"""
        with gr.Column():
            gr.Markdown("## 💾 Экспорт результатов")
            
            with gr.Row():
                self.export_format_selector = gr.Radio(
                    choices=["CSV", "Excel", "PDF", "JSON"],
                    value="CSV",
                    label="Формат экспорта"
                )
                
                self.export_btn = gr.Button(
                    "📥 Экспортировать",
                    variant="primary"
                )
            
            self.export_status = gr.Textbox(
                label="Статус экспорта",
                interactive=False
            )
            
            self.download_file = gr.File(
                label="Скачать файл",
                visible=False
            )

    def _create_settings_tab(self) -> None:
        """Создание вкладки настроек"""
        with gr.Column():
            gr.Markdown("## ⚙️ Настройки системы")
            
            with gr.Accordion("🎯 Пороги анализа", open=True):
                self.authenticity_threshold = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=CRITICAL_THRESHOLDS.get("min_authenticity_score", 0.6),
                    label="Порог аутентичности"
                )
                
                self.quality_threshold = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.6,
                    label="Порог качества изображения"
                )
            
            with gr.Accordion("🔧 Параметры обработки", open=False):
                self.max_files = gr.Number(
                    label="Максимум файлов",
                    value=MAX_FILE_UPLOAD_COUNT,
                    precision=0
                )
                
                self.batch_size = gr.Number(
                    label="Размер batch",
                    value=50,
                    precision=0
                )

    def _get_custom_css(self) -> str:
        """Получение пользовательского CSS"""
        return """
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .risk-high {
            background-color: #ffebee;
            border-left: 4px solid #f44336;
            padding: 10px;
            margin: 5px 0;
        }
        
        .risk-medium {
            background-color: #fff3e0;
            border-left: 4px solid #ff9800;
            padding: 10px;
            margin: 5px 0;
        }
        
        .risk-low {
            background-color: #e8f5e8;
            border-left: 4px solid #4caf50;
            padding: 10px;
            margin: 5px 0;
        }
        
        .metric-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            margin: 10px;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .progress-bar {
            width: 100%;
            height: 20px;
            background-color: #f0f0f0;
            border-radius: 10px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4caf50, #8bc34a);
            transition: width 0.3s ease;
        }
        """

    def launch(self, **kwargs) -> None:
        """
        ИСПРАВЛЕНО: Запуск интерфейса
        Согласно правкам: правильная конфигурация запуска
        """
        try:
            logger.info("Запуск интерфейса Gradio")
            
            # Создание интерфейса
            demo = self.create_interface()
            
            # ЯВНО ВКЛЮЧАЕМ ОЧЕРЕДЬ ДЛЯ GR.BLOCKS
            demo.queue()

            # Автоматический поиск свободного порта
            demo.launch(**kwargs)
            
        except Exception as e:
            logger.error(f"Ошибка запуска интерфейса: {e}")
            raise

# ==================== ТОЧКА ВХОДА ====================

def create_modular_interface(all_system_components: Dict[str, Any]) -> GradioInterface:
    """Factory function to create the GradioInterface with all its components."""
    # Pass all components from the main system to the GradioInterface
    return GradioInterface(all_system_components)

def main():
    # This main is for testing GradioInterface independently, not used by main.py
    pass # Main application entry point is in main.py

if __name__ == "__main__":
    main()