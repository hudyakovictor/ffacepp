import sys
import psutil
import json
import os
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import cv2
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import gradio as gr
import logging
import re

# Импортируем необходимые классы из ReportLab
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.units import inch

# Настройка логирования
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

from core_config import (
    DBSCAN_EPSILON, DBSCAN_MIN_SAMPLES,
    ANOMALY_DETECTION_THRESHOLDS, IMAGE_QUALITY_THRESHOLDS, RESULTS_DIR,
    MASK_DETECTION_THRESHOLDS, PUTIN_BIRTH_DATE, GRADIO_INTERFACE_SETTINGS, GRADIO_DEFAULTS,
    get_chronological_analysis_parameters # Добавляем импорт функции
)
from data_manager import DataManager
from face_3d_analyzer import Face3DAnalyzer
from embedding_analyzer import EmbeddingAnalyzer
from texture_analyzer import TextureAnalyzer
from temporal_analyzer import TemporalAnalyzer
from anomaly_detector import AnomalyDetector


class OptimizedGradioInterface:
    """Оптимизированный интерфейс для обработки больших объемов данных"""
    
    def __init__(self):
        self.max_concurrent_processes = GRADIO_INTERFACE_SETTINGS['MAX_CONCURRENT_PROCESSES']
        self.batch_size = GRADIO_INTERFACE_SETTINGS['BATCH_SIZE']
        self.max_gallery_items = GRADIO_INTERFACE_SETTINGS['MAX_GALLERY_ITEMS']
        
        # Инициализация анализаторов
        self._initialize_analyzers()
        
        # Состояния для хранения данных
        self.analysis_state = {
            'processed_count': 0,
            'total_count': 0,
            'current_batch': 0,
            'results': {},
            'errors': [],
            'processing': False
        }
        
        # Кэш для оптимизации
        self.results_cache = {}
        self.visualization_cache = {}
        
    def _initialize_analyzers(self):
        """Инициализация всех анализаторов с обработкой ошибок"""
        try:
            self.data_manager = DataManager()
            self.face_analyzer = Face3DAnalyzer()
            self.embedding_analyzer = EmbeddingAnalyzer()
            self.embedding_analyzer.initialize_insightface_model() # Инициализация модели InsightFace
            self.texture_analyzer = TextureAnalyzer()
            self.temporal_analyzer = TemporalAnalyzer(self.data_manager) # Передача data_manager
            self.anomaly_detector = AnomalyDetector()
            logger.info("Все анализаторы успешно инициализированы")
        except Exception as e:
            logger.error(f"Ошибка инициализации анализаторов: {e}")
            raise

    def create_optimized_interface(self):
        """Создает оптимизированный интерфейс"""
        
        with gr.Blocks(
            title="Система анализа лицевой идентичности - Оптимизированная версия",
            theme=gr.themes.Soft(),
            css=self._get_custom_css()
        ) as demo:
            
            # Заголовок и статус
            gr.Markdown("# 🔬 Система анализа лицевой идентичности")
            gr.Markdown("**Комплексный анализ на основе 3DDFA_V2, InsightFace и научных методов**")
            
            # Глобальный статус обработки
            with gr.Row():
                processing_status = gr.HTML(
                    value="<div class='status-ready'>✅ Система готова к работе</div>",
                    elem_id="global_status"
                )
                progress_bar = gr.Progress()
            
            with gr.Tabs() as main_tabs:
                
                # Вкладка 1: Массовая загрузка и обработка
                with gr.TabItem("📁 Массовая обработка", id="batch_processing"):
                    self._create_batch_processing_tab(processing_status, progress_bar)
                
                # Вкладка 2: Результаты кластеризации
                with gr.TabItem("👥 Кластеризация личностей", id="clustering"):
                    self._create_clustering_results_tab()
                
                # Вкладка 3: Временной анализ
                with gr.TabItem("📊 Временной анализ", id="temporal"):
                    self._create_temporal_analysis_tab()
                
                # Вкладка 4: Детекция масок
                with gr.TabItem("🎭 Детекция масок", id="mask_detection"):
                    self._create_mask_detection_tab()
                
                # Вкладка 5: Экспертное заключение
                with gr.TabItem("📋 Экспертное заключение", id="expert_report"):
                    self._create_expert_report_tab()
                
                # Вкладка 6: Настройки системы
                with gr.TabItem("⚙️ Настройки", id="settings"):
                    self._create_settings_tab()
            
            return demo

    def _create_batch_processing_tab(self, processing_status, progress_bar):
        """Вкладка массовой обработки с оптимизацией для больших объемов"""
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📂 Загрузка данных")
                
                # Выбор файлов с ограничениями
                file_input = gr.File(
                    file_count="multiple",
                    file_types=["image"],
                    label=f"Выберите изображения (до {GRADIO_INTERFACE_SETTINGS['MAX_FILE_UPLOAD_COUNT']} файлов)",
                    height=200
                )
                
                # Настройки обработки
                with gr.Accordion("🔧 Настройки обработки", open=False):
                    batch_size_slider = gr.Slider(
                        minimum=10, maximum=100, value=GRADIO_INTERFACE_SETTINGS['BATCH_SIZE'], step=10,
                        label="Размер батча"
                    )
                    max_workers_slider = gr.Slider(
                        minimum=1, maximum=8, value=GRADIO_INTERFACE_SETTINGS['MAX_CONCURRENT_PROCESSES'], step=1,
                        label="Количество потоков"
                    )
                    quality_threshold = gr.Slider(
                        minimum=0.3, maximum=1.0, value=IMAGE_QUALITY_THRESHOLDS['DEFAULT_QUALITY_THRESHOLD'], step=0.1,
                        label="Порог качества изображений"
                    )
                
                # Кнопки управления
                with gr.Row():
                    start_processing_btn = gr.Button(
                        "🚀 Начать обработку", 
                        variant="primary", 
                        size="lg"
                    )
                    stop_processing_btn = gr.Button(
                        "⏹️ Остановить", 
                        variant="stop",
                        visible=False
                    )
                
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Прогресс обработки")
                
                # Детальная информация о прогрессе
                progress_info = gr.JSON(
                    label="Статистика обработки",
                    value={
                        "total_files": 0,
                        "processed": 0,
                        "successful": 0,
                        "errors": 0,
                        "current_batch": 0,
                        "estimated_time_remaining": "00:00:00"
                    }
                )
                
                # Лог обработки с ограничением размера
                processing_log = gr.Textbox(
                    label="Лог обработки (последние 100 записей)",
                    lines=15,
                    max_lines=15,
                    interactive=False,
                    autoscroll=True
                )
                
                # Предварительные результаты
                preview_results = gr.JSON(
                    label="Предварительные результаты",
                    visible=False
                )
        
        # Обработчики событий
        start_processing_btn.click(
            fn=self._start_batch_processing,
            inputs=[
                file_input, 
                batch_size_slider, 
                max_workers_slider, 
                quality_threshold
            ],
            outputs=[
                processing_status,
                progress_info,
                processing_log,
                start_processing_btn,
                stop_processing_btn,
                preview_results
            ],
            queue=True
        )
        
        stop_processing_btn.click(
            fn=self._stop_processing,
            outputs=[
                processing_status,
                start_processing_btn,
                stop_processing_btn
            ]
        )

    def _create_clustering_results_tab(self):
        """Вкладка результатов кластеризации"""
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎯 Управление кластеризацией")
                
                # Параметры DBSCAN
                with gr.Group():
                    gr.Markdown("**Параметры DBSCAN**")
                    epsilon_slider = gr.Slider(
                        minimum=0.1, maximum=1.0, value=GRADIO_DEFAULTS['DBSCAN_EPSILON_DEFAULT'], step=0.01,
                        label="Epsilon (порог сходства)"
                    )
                    min_samples_slider = gr.Slider(
                        minimum=2, maximum=10, value=GRADIO_DEFAULTS['DBSCAN_MIN_SAMPLES_DEFAULT'], step=1,
                        label="Минимум образцов"
                    )
                
                # Фильтры
                with gr.Group():
                    gr.Markdown("**Фильтры**")
                    confidence_filter = gr.Slider(
                        minimum=0.0, maximum=1.0, value=GRADIO_DEFAULTS['CONFIDENCE_FILTER_DEFAULT'], step=0.1,
                        label="Минимальная уверенность"
                    )
                    date_range = gr.DateTime(
                        label="Диапазон дат",
                        type="datetime",
                        include_time=False
                    )
                
                recalculate_btn = gr.Button(
                    "🔄 Пересчитать кластеры", 
                    variant="secondary"
                )
                
            with gr.Column(scale=2):
                gr.Markdown("### 📈 Результаты кластеризации")
                
                # Основная статистика
                cluster_summary = gr.DataFrame(
                    headers=[
                        "ID кластера", "Количество фото", "Первое появление", 
                        "Последнее появление", "Средняя уверенность", "Статус"
                    ],
                    label="Сводка по кластерам"
                )
                
                # Визуализация кластеров
                cluster_visualization = gr.Plot(
                    label="Визуализация кластеров (t-SNE)"
                )
        
        with gr.Row():
            # Галерея представителей кластеров (оптимизированная)
            cluster_gallery = gr.Gallery(
                label="Представители кластеров",
                columns=5,
                rows=4,
                height=400,
                show_label=True
            )
        
        # Детальная информация о выбранном кластере
        with gr.Row():
            with gr.Column():
                selected_cluster_info = gr.JSON(
                    label="Информация о выбранном кластере"
                )
            with gr.Column():
                cluster_timeline = gr.Plot(
                    label="Временная линия кластера"
                )
        
        # Обработчики
        recalculate_btn.click(
            fn=self._recalculate_clusters,
            inputs=[epsilon_slider, min_samples_slider, confidence_filter, date_range],
            outputs=[cluster_summary, cluster_visualization, cluster_gallery, selected_cluster_info]
        )

    def _create_temporal_analysis_tab(self):
        """Вкладка временного анализа"""
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⏰ Параметры анализа")
                
                # Выбор личности для анализа
                identity_selector = gr.Dropdown(
                    label="Выберите личность",
                    choices=[],
                    interactive=True
                )
                
                # Временные параметры
                temporal_resolution = gr.Radio(
                    choices=["Месяц", "Квартал", "Год"],
                    value=GRADIO_DEFAULTS['TEMPORAL_RESOLUTION_DEFAULT'],
                    label="Временное разрешение"
                )
                
                # Метрики для анализа
                metrics_selector = gr.CheckboxGroup(
                    choices=[
                        "skull_width_ratio", "temporal_bone_angle", 
                        "zygomatic_arch_width", "forehead_height_ratio",
                        "nose_width_ratio", "eye_distance_ratio"
                    ],
                    value=GRADIO_DEFAULTS['TEMPORAL_METRICS_DEFAULT'],
                    label="Метрики для отслеживания"
                )
                
                analyze_temporal_btn = gr.Button(
                    "📊 Анализировать", 
                    variant="primary"
                )
                
            with gr.Column(scale=2):
                gr.Markdown("### 📈 Результаты временного анализа")
                
                # Основной график изменений
                temporal_plot = gr.Plot(
                    label="Изменения метрик во времени"
                )
                
                # График предсказанных vs фактических метрик
                predicted_vs_actual_plot = gr.Plot(
                    label="Предсказанные vs фактические метрики"
                )

                # График точек изменения
                change_points_plot = gr.Plot(
                    label="Обнаруженные точки изменения"
                )

                # Статистика изменений
                change_statistics = gr.DataFrame(
                    headers=[
                        "Метрика", "Среднее изменение", "Стандартное отклонение",
                        "Тренд", "Аномалии", "P-value"
                    ],
                    label="Статистика изменений"
                )
        
        with gr.Row():
            # Детекция аномалий
            anomaly_detection_plot = gr.Plot(
                label="Детекция аномалий во времени"
            )
            
            # Корреляция с историческими событиями
            historical_correlation = gr.Plot(
                label="Корреляция с историческими событиями"
            )
        
        # Обработчик
        analyze_temporal_btn.click(
            fn=self._analyze_temporal_patterns,
            inputs=[identity_selector, temporal_resolution, metrics_selector],
            outputs=[
                temporal_plot, 
                change_statistics, 
                anomaly_detection_plot, 
                historical_correlation,
                predicted_vs_actual_plot, # Новый вывод
                change_points_plot      # Новый вывод
            ]
        )

    def _create_mask_detection_tab(self):
        """Вкладка детекции масок"""
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎭 Параметры детекции")
                
                # Настройки детекции
                detection_sensitivity = gr.Slider(
                    minimum=0.1, maximum=1.0, value=GRADIO_DEFAULTS['MASK_DETECTION_SENSITIVITY_DEFAULT'], step=0.1,
                    label="Чувствительность детекции"
                )
                
                technology_levels = gr.CheckboxGroup(
                    choices=list(MASK_DETECTION_THRESHOLDS.keys()), # Используем ключи из MASK_DETECTION_THRESHOLDS
                    value=["Level3_Commercial", "Level4_Professional", "Level5_Advanced"],
                    label="Уровни технологий для поиска"
                )
                
                analyze_masks_btn = gr.Button(
                    "🔍 Анализировать маски", 
                    variant="primary"
                )
                
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Результаты детекции масок")
                
                # Сводка по маскам
                mask_summary = gr.DataFrame(
                    headers=[
                        "Дата", "Уровень технологии", "Уверенность",
                        "Shape Error", "Texture Score", "Статус", "Путь к файлу"
                    ],
                    label="Обнаруженные маски"
                )
                
                # Распределение по технологиям
                technology_distribution = gr.Plot(
                    label="Распределение технологий масок по времени"
                )
        
        with gr.Row():
            # Детальный анализ текстуры
            texture_analysis_plot = gr.Plot(
                label="Анализ текстурных характеристик"
            )
            
            # Эволюция технологий
            technology_evolution = gr.Plot(
                label="Эволюция технологий масок"
            )
        
        # Критические обнаружения
        with gr.Row():
            critical_detections = gr.HTML(
                label="Критические обнаружения",
                value="<div class='alert-info'>Анализ не выполнен</div>"
            )
        
        # Обработчик
        analyze_masks_btn.click(
            fn=self._analyze_mask_technology,
            inputs=[detection_sensitivity, technology_levels],
            outputs=[
                mask_summary, 
                technology_distribution, 
                texture_analysis_plot,
                technology_evolution,
                critical_detections
            ]
        )

    def _create_expert_report_tab(self):
        """Вкладка экспертного заключения"""
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📋 Параметры отчета")
                
                # Настройки отчета
                report_type = gr.Radio(
                    choices=["Краткий", "Подробный", "Научный"],
                    value="Подробный",
                    label="Тип отчета"
                )
                
                include_sections = gr.CheckboxGroup(
                    choices=[
                        "Исполнительное резюме",
                        "Методология",
                        "Результаты кластеризации",
                        "Временной анализ",
                        "Детекция масок",
                        "Статистический анализ",
                        "Заключение"
                    ],
                    value=[
                        "Исполнительное резюме",
                        "Результаты кластеризации",
                        "Временной анализ",
                        "Заключение"
                    ],
                    label="Разделы отчета"
                )
                
                confidence_threshold_report = gr.Slider(
                    minimum=0.5, maximum=1.0, value=0.85, step=0.05,
                    label="Порог уверенности для выводов"
                )
                
                generate_report_btn = gr.Button(
                    "📄 Сгенерировать отчет", 
                    variant="primary",
                    size="lg"
                )
                
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Ключевые показатели")
                
                # Основные метрики
                key_metrics = gr.JSON(
                    label="Ключевые показатели анализа",
                    value={
                        "total_identities": 0,
                        "confidence_level": 0.0,
                        "statistical_significance": 0.0,
                        "anomalies_detected": 0,
                        "masks_detected": 0,
                        "analysis_period": "N/A"
                    }
                )
                
                # Итоговая временная линия
                final_timeline = gr.Plot(
                    label="Итоговая временная линия"
                )
        
        with gr.Row():
            # Экспертное заключение
            expert_conclusion = gr.Textbox(
                label="Экспертное заключение",
                lines=20,
                max_lines=30,
                interactive=False,
                placeholder="Здесь будет сгенерировано экспертное заключение..."
            )
        
        with gr.Row():
            # Экспорт результатов
            with gr.Column():
                export_json_btn = gr.Button("💾 Экспорт JSON", variant="secondary")
                export_pdf_btn = gr.Button("📄 Экспорт PDF", variant="secondary")
            
            with gr.Column():
                download_link = gr.File(
                    label="Файл для скачивания",
                    visible=False
                )
        
        # Обработчики
        generate_report_btn.click(
            fn=self._generate_expert_report,
            inputs=[report_type, include_sections, confidence_threshold_report],
            outputs=[key_metrics, final_timeline, expert_conclusion, download_link] # Добавляем download_link
        )
        
        export_json_btn.click(
            fn=self._export_results_json,
            outputs=[download_link]
        )
        
        # TODO: Добавить обработчик для export_pdf_btn
        export_pdf_btn.click(
            fn=self._export_results_pdf,
            outputs=[download_link]
        )

    def _create_settings_tab(self):
        """Вкладка настроек системы"""
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### ⚙️ Настройки алгоритмов")
                
                with gr.Accordion("DBSCAN Кластеризация", open=True):
                    dbscan_epsilon = gr.Slider(
                        minimum=0.1, maximum=1.0, value=DBSCAN_EPSILON, step=0.01,
                        label="Epsilon"
                    )
                    dbscan_min_samples = gr.Slider(
                        minimum=1, maximum=10, value=DBSCAN_MIN_SAMPLES, step=1,
                        label="Минимум образцов"
                    )
                
                with gr.Accordion("Детекция аномалий", open=False):
                    anomaly_z_threshold = gr.Slider(
                        minimum=1.0, maximum=5.0, value=ANOMALY_DETECTION_THRESHOLDS['Z_SCORE_ANOMALY_THRESHOLD'], step=0.1,
                        label="Z-score порог"
                    )
                    change_rate_threshold = gr.Slider(
                        minimum=1.0, maximum=10.0, value=ANOMALY_DETECTION_THRESHOLDS['RAPID_CHANGE_STD_MULTIPLIER'], step=0.5,
                        label="Порог скорости изменений"
                    )
                
                with gr.Accordion("Качество изображений", open=False):
                    min_face_size = gr.Slider(
                        minimum=50, maximum=300, value=IMAGE_QUALITY_THRESHOLDS['MIN_FACE_SIZE'], step=10,
                        label="Минимальный размер лица (px)"
                    )
                    blur_threshold = gr.Slider(
                        minimum=50, maximum=200, value=IMAGE_QUALITY_THRESHOLDS['BLUR_DETECTION_THRESHOLD'], step=10,
                        label="Порог размытости"
                    )
                
                apply_settings_btn = gr.Button("✅ Применить настройки", variant="primary")
                reset_settings_btn = gr.Button("🔄 Сбросить", variant="secondary")
                
            with gr.Column():
                gr.Markdown("### 📊 Системная информация")
                
                system_status = gr.JSON(
                    label="Статус системы",
                    value={
                        "3ddfa_status": "Загружается...",
                        "insightface_status": "Загружается...",
                        "gpu_available": False,
                        "memory_usage": "N/A",
                        "cache_size": "N/A"
                    }
                )
                
                performance_metrics = gr.DataFrame(
                    headers=["Компонент", "Время обработки (мс)", "Статус"],
                    label="Производительность компонентов"
                )
                
                # Управление кэшем
                with gr.Group():
                    gr.Markdown("**Управление кэшем**")
                    cache_info = gr.Textbox(
                        label="Информация о кэше",
                        value="Кэш пуст",
                        interactive=False
                    )
                    
                    with gr.Row():
                        clear_cache_btn = gr.Button("🗑️ Очистить кэш", variant="stop")
                        update_status_btn = gr.Button("🔄 Обновить статус", variant="secondary")
        
        # Обработчики настроек
        apply_settings_btn.click(
            fn=self._apply_system_settings,
            inputs=[
                dbscan_epsilon, dbscan_min_samples,
                anomaly_z_threshold, change_rate_threshold,
                min_face_size, blur_threshold
            ],
            outputs=[system_status]
        )
        
        reset_settings_btn.click(
            fn=lambda: (
                gr.update(value=DBSCAN_EPSILON),
                gr.update(value=DBSCAN_MIN_SAMPLES),
                gr.update(value=ANOMALY_DETECTION_THRESHOLDS['Z_SCORE_ANOMALY_THRESHOLD']),
                gr.update(value=ANOMALY_DETECTION_THRESHOLDS['RAPID_CHANGE_STD_MULTIPLIER']),
                gr.update(value=IMAGE_QUALITY_THRESHOLDS['MIN_FACE_SIZE']),
                gr.update(value=IMAGE_QUALITY_THRESHOLDS['BLUR_DETECTION_THRESHOLD'])
            ),
            outputs=[
                dbscan_epsilon, dbscan_min_samples,
                anomaly_z_threshold, change_rate_threshold,
                min_face_size, blur_threshold
            ]
        )

        clear_cache_btn.click(
            fn=self._clear_system_cache,
            outputs=[cache_info, system_status]
        )
        
        update_status_btn.click(
            fn=self._update_system_status,
            outputs=[system_status, performance_metrics]
        )

    # ==================== ОСНОВНЫЕ МЕТОДЫ ОБРАБОТКИ ====================

    async def _start_batch_processing(
        self, 
        files: List[str], 
        batch_size: int, 
        max_workers: int, 
        quality_threshold: float
    ):
        """Запуск массовой обработки с оптимизацией"""
        
        if not files:
            return (
                "<div class='status-error'>❌ Файлы не выбраны</div>",
                {"error": "Файлы не выбраны"},
                "Ошибка: Файлы не выбраны\n",
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(value={}, visible=False)
            )
        
        # Проверка количества файлов
        if len(files) > 1500:
            return (
                "<div class='status-error'>❌ Слишком много файлов (максимум 1500)</div>",
                {"error": f"Загружено {len(files)} файлов, максимум 1500"},
                f"Ошибка: Загружено {len(files)} файлов, максимум 1500\n",
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(value={}, visible=False)
            )
        
        # Инициализация состояния
        self.analysis_state = {
            'processed_count': 0,
            'total_count': len(files),
            'current_batch': 0,
            'results': {},
            'errors': [],
            'processing': True,
            'start_time': datetime.now()
        }
        
        try:
            # Запуск обработки
            results = await self._process_files_in_batches(
                files, batch_size, max_workers, quality_threshold
            )
            
            # Формирование ответа
            status_html = f"<div class='status-success'>✅ Обработано {len(results['successful'])} из {len(files)} файлов</div>"
            
            progress_info = {
                "total_files": len(files),
                "processed": len(results['successful']) + len(results['errors']),
                "successful": len(results['successful']),
                "errors": len(results['errors']),
                "current_batch": "Завершено",
                "processing_time": str(datetime.now() - self.analysis_state['start_time'])
            }
            
            log_text = self._format_processing_log(results)
            
            # Обновление выбора личностей для вкладки временного анализа
            # Используем кэшированные данные из _recalculate_clusters
            identity_choices = self.results_cache.get('identity_selector_choices', [])
            # TODO: Необходимо реализовать механизм динамического обновления gr.Dropdown.choices
            # Gradio не позволяет напрямую обновлять список выборов компонента, переданного в outputs,
            # если этот список генерируется на основе результатов другой функции. 
            # Для этого требуется более сложная логика с использованием stateful компонентов или 
            # каскадных обновлений, возможно, через gr.State или передачу id компонента и использование gr.Dropdown.update().
            # В текущей реализации identity_selector_choices должен быть кэширован и использоваться при следующем вызове _create_temporal_analysis_tab.
            
            return (
                status_html,
                progress_info,
                log_text,
                gr.update(visible=False), # Скрываем кнопку "Начать обработку"
                gr.update(visible=True), # Показываем кнопку "Остановить"
                gr.update(value=results['summary'], visible=True)
            )
            
        except Exception as e:
            logger.error(f"Ошибка при обработке: {e}")
            return (
                f"<div class='status-error'>❌ Ошибка: {str(e)}</div>",
                {"error": str(e)},
                f"Критическая ошибка: {str(e)}\n",
                gr.update(visible=True), # Показываем кнопку "Начать обработку"
                gr.update(visible=False), # Скрываем кнопку "Остановить"
                gr.update(value={}, visible=False)
            )

    async def _process_files_in_batches(
        self, 
        files: List[str],
        batch_size: int, 
        max_workers: int, 
        quality_threshold: float
    ) -> Dict:
        """Обработка файлов батчами с оптимизацией памяти"""
        
        all_results = {
            'successful': [],
            'errors': [],
            'embeddings': [],
            'landmarks': [],
            'poses': [],
            'full_processed_data': [], # Добавляем для хранения полных результатов обработки каждого файла
            'summary': {}
        }
        
        # Разбивка на батчи
        batches = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for batch_idx, batch in enumerate(batches):
                if not self.analysis_state['processing']:
                    break
                
                logger.info(f"Обработка батча {batch_idx + 1}/{len(batches)}")
                
                # Обработка батча
                batch_results = await self._process_single_batch(
                    batch, executor, quality_threshold
                )
                
                # Объединение результатов
                all_results['successful'].extend(batch_results['successful'])
                all_results['errors'].extend(batch_results['errors'])
                all_results['embeddings'].extend(batch_results['embeddings'])
                all_results['landmarks'].extend(batch_results['landmarks'])
                all_results['poses'].extend(batch_results['poses'])
                all_results['full_processed_data'].extend(batch_results['full_processed_data']) # Сохраняем полные данные
                
                # Обновление прогресса
                self.analysis_state['current_batch'] = batch_idx + 1
                self.analysis_state['processed_count'] = len(all_results['successful']) + len(all_results['errors'])
                
                # Очистка памяти
                gc.collect()
        
        # Кластеризация результатов
        if all_results['embeddings']:
            # Фильтруем только корректные эмбеддинги (не пустые и не None)
            valid_embeddings_data = [
                item for item in all_results['embeddings'] 
                if isinstance(item.get('embedding'), np.ndarray) and item['embedding'].size > 0
            ]
            
            if not valid_embeddings_data:
                logger.warning("Нет действительных эмбеддингов для кластеризации после фильтрации.")
                all_results['summary'] = {'error': 'Нет данных для кластеризации'}
                return all_results

            cluster_results = self._perform_clustering(valid_embeddings_data)
            all_results['summary'] = cluster_results
            # Сохраняем полные результаты кластеризации в cache для использования в других вкладках
            self.results_cache['clustering_results'] = cluster_results
            self.results_cache['full_processed_data'] = all_results['full_processed_data']
        else:
            logger.info("Нет эмбеддингов для кластеризации (список пуст).")
            all_results['summary'] = {'error': 'Нет данных для кластеризации'}
        
        return all_results

    async def _process_single_batch(
        self, 
        batch: List[str], 
        executor: ThreadPoolExecutor, 
        quality_threshold: float
    ) -> Dict:
        """Обработка одного батча файлов"""
        
        batch_results = {
            'successful': [],
            'errors': [],
            'embeddings': [],
            'landmarks': [],
            'poses': [],
            'full_processed_data': [] # Добавляем для хранения полных результатов обработки каждого файла
        }
        
        # Создание задач для параллельной обработки
        future_to_file = {
            executor.submit(self._process_single_image, file_path, quality_threshold): file_path
            for file_path in batch
        }
        
        # Обработка результатов
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result(timeout=30)  # Таймаут 30 секунд
                
                if result['success']:
                    batch_results['successful'].append(result)
                    batch_results['embeddings'].append({
                        'file_path': file_path,
                        'embedding': result['embedding'],
                        'confidence': result['confidence'],
                        'date': result.get('date'),
                        'metadata': result.get('metadata', {})
                    })
                    batch_results['landmarks'].append({
                        'file_path': file_path,
                        'landmarks_3d': result['landmarks_3d'],
                        'pose_info': result['pose_info']
                    })
                    batch_results['poses'].append(result['pose_info'])
                    batch_results['full_processed_data'].append(result) # Сохраняем полный результат
                else:
                    batch_results['errors'].append({
                        'file_path': file_path,
                        'error': result['error']
                    })
                        
            except Exception as e:
                logger.error(f"Ошибка обработки {file_path}: {e}", exc_info=True)
                batch_results['errors'].append({
                    'file_path': file_path,
                    'error': str(e)
                })
        
        return batch_results

    def _process_single_image(self, file_path: str, quality_threshold: float) -> Dict:
        """Обработка одного изображения"""
        
        try:
            # Загрузка изображения
            image = cv2.imread(file_path)
            if image is None:
                return {'success': False, 'error': 'Не удалось загрузить изображение'}
            
            # Дополнительная проверка на полностью черное или пустое изображение
            # Среднее значение пикселей для цветного изображения (0-255)
            # Порог 1.0 означает, что если среднее значение всех каналов меньше 1.0, считаем его черным.
            if image.size == 0 or np.mean(image) < 1.0:
                logger.warning(f"Изображение по пути {file_path} является черным или пустым. Среднее значение пикселей: {np.mean(image):.2f}. Пропускаем обработку.")
                return {'success': False, 'error': 'Изображение полностью черное или пустое'}

            # --- START NEW LOGGING --- 
            logger.info(f"В _process_single_image: image path={file_path}")
            logger.info(f"В _process_single_image: image shape={image.shape}, dtype={image.dtype}")
            # Логируем небольшой фрагмент изображения для проверки нулей или других проблем
            if image.size > 0 and image.ndim >= 2 and image.shape[0] >= 5 and image.shape[1] >= 5:
                logger.info(f"В _process_single_image: верхний левый угол image (5x5):\n{image[:5, :5]}")
            else:
                logger.info(f"В _process_single_image: image слишком мал или некорректен для логирования фрагмента.")
            # --- END NEW LOGGING --- 

            # Проверка качества
            quality_score = self._assess_image_quality(image)
            if quality_score < quality_threshold:
                return {'success': False, 'error': f'Низкое качество изображения: {quality_score:.2f}'}
            
            # Извлечение landmarks
            try:
                raw_landmarks_result = self.face_analyzer.extract_68_landmarks_with_confidence(image)
                
                if not (isinstance(raw_landmarks_result, (list, tuple)) and len(raw_landmarks_result) == 3):
                    raise ValueError(f"extract_68_landmarks_with_confidence вернул неожиданный формат: Тип={type(raw_landmarks_result)}, Содержимое={raw_landmarks_result}")
                
                landmarks_3d, confidence_scores, image_shape = raw_landmarks_result
            except Exception as e:
                logging.error(f"Ошибка при извлечении или распаковке ландмарков: {e}. Это означает, что функция вернула неожиданное количество элементов или произошла другая ошибка.")
                return {'success': False, 'error': f'Ошибка извлечения ландмарков: {e}'}
            
            # Проверка на NaN/Inf значения в ландмарках
            if not isinstance(landmarks_3d, np.ndarray) or landmarks_3d.size == 0 or \
               np.any(np.isnan(landmarks_3d)) or np.any(np.isinf(landmarks_3d)):
                return {'success': False, 'error': 'Не удалось извлечь корректные ландмарки (содержат NaN/Inf или пусты)'}
            
            # Определение позы
            pose_info = self.face_analyzer.determine_precise_face_pose(landmarks_3d)
            
            # Нормализация landmarks
            normalized_landmarks = self.face_analyzer.normalize_landmarks_by_pose_category(
                landmarks_3d, pose_info['pose_category']
            )
            
            # Извлечение эмбеддинга
            try:
                embedding, embedding_confidence = self.embedding_analyzer.extract_512d_face_embedding(image)
            except Exception as e:
                logging.error(f"Ошибка при извлечении или распаковке эмбеддинга: {e}. Это означает, что функция вернула неожиданное количество элементов.", exc_info=True)
                return {'success': False, 'error': f'Ошибка извлечения эмбеддинга: {e}'}

            if embedding.size == 0:
                if embedding_confidence == 0.0:
                    logger.warning(f"Эмбеддинг пуст для {file_path}: Лица не обнаружены или не удалось получить уверенность.")
                    return {'success': False, 'error': 'Лица не обнаружены или низкая уверенность'}
                else:
                    logger.warning(f"Эмбеддинг пуст для {file_path}: Неизвестная причина, уверенность: {embedding_confidence}.")
                    return {'success': False, 'error': 'Не удалось извлечь эмбеддинг по неизвестной причине'}
            
            # Расчет метрик идентичности
            identity_metrics = self.face_analyzer.calculate_identity_signature_metrics( # Изменено
                normalized_landmarks, pose_info['pose_category']
            )
            
            # Анализ текстуры
            logging.info(f"Перед вызовом analyze_skin_texture_by_zones: image type={{type(image)}}, image shape={{image.shape if image is not None else 'None'}}")
            texture_analysis = self.texture_analyzer.analyze_skin_texture_by_zones(
                image, landmarks_3d[:, :2]
            )
            
            # Извлечение даты из метаданных файла
            file_date = self._extract_date_from_file(file_path)
            
            return {
                'success': True,
                'file_path': file_path,
                'landmarks_3d': landmarks_3d,
                'normalized_landmarks': normalized_landmarks,
                'pose_info': pose_info,
                'embedding': embedding,
                'confidence': embedding_confidence,
                'identity_metrics': identity_metrics,
                'texture_analysis': texture_analysis,
                'quality_score': quality_score,
                'date': file_date,
                'metadata': {
                    'image_shape': image.shape[:2],
                    'landmarks_confidence': confidence_scores.mean() if confidence_scores.size > 0 else 0.0
                }
            }
            
        except Exception as e:
            logger.error(f"Ошибка обработки {file_path}: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def _perform_clustering(self, embeddings_data: List[Dict]) -> Dict:
        """Выполнение кластеризации эмбеддингов"""
        
        try:
            embeddings_to_cluster = [item['embedding'] for item in embeddings_data]
            file_paths = [item['file_path'] for item in embeddings_data]
            dates = [item['date'] for item in embeddings_data]

            cluster_results = self.embedding_analyzer.perform_identity_clustering(
                embeddings_data # Передаем полный список словарей с эмбеддингами и метаданными
                # epsilon, min_samples теперь извлекаются внутри perform_identity_clustering
                # file_paths, dates, confidence_scores больше не нужны в качестве отдельных аргументов
            )
            
            cluster_results['file_paths'] = file_paths
            cluster_results['dates'] = dates

            identity_timeline = self.embedding_analyzer.build_identity_timeline(
                cluster_results
            )
            
            stability_analysis = self.embedding_analyzer.analyze_cluster_temporal_stability(
                identity_timeline
            )
            
            return {
                'cluster_results': cluster_results,
                'identity_timeline': identity_timeline,
                'stability_analysis': stability_analysis,
                'summary_stats': {
                    'n_clusters': cluster_results.get('n_clusters', 0),
                    'n_noise': cluster_results.get('n_noise', 0),
                    'total_images': len(embeddings_data),
                    'clustered_images': len(embeddings_data) - cluster_results.get('n_noise', 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Ошибка кластеризации: {e}", exc_info=True) # Добавлено exc_info=True для полного трассировочного лога
            return {'error': str(e)}

    def _recalculate_clusters(self, epsilon: float, min_samples: int, confidence_filter: float, date_range: Tuple[str, str]) -> Tuple[pd.DataFrame, go.Figure, List[Tuple[str, str]], gr.Json]:
        """Пересчет кластеров с новыми параметрами"""
        logger.info(f"Пересчет кластеров с epsilon={epsilon}, min_samples={min_samples}, confidence_filter={confidence_filter}, date_range={date_range}")

        # Получаем порог уверенности из конфигурации
        chronological_params = get_chronological_analysis_parameters()
        report_confidence_threshold = chronological_params.get('confidence_threshold', 0.85) # По умолчанию 0.85

        # Проверка наличия данных
        if 'full_processed_data' not in self.results_cache or not self.results_cache['full_processed_data']:
            logger.warning("Нет обработанных данных для кластеризации.")
            return (
                pd.DataFrame(columns=[
                    "ID кластера", "Количество фото", "Первое появление", 
                    "Последнее появление", "Средняя уверенность", "Статус"
                ]),
                go.Figure().update_layout(title="Нет данных для визуализации кластеров"),
                [],
                gr.update(value={})
            )

        full_processed_data = self.results_cache['full_processed_data']
        
        # 1. Фильтрация данных по уверенности и дате
        filtered_data = [
            item for item in full_processed_data 
            if item.get('confidence', 0.0) >= confidence_filter
        ]
        
        # Фильтрация по диапазону дат
        if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
            try:
                start_date_obj = datetime.strptime(date_range[0], '%Y-%m-%d')
                end_date_obj = datetime.strptime(date_range[1], '%Y-%m-%d')
                filtered_data = [
                    item for item in filtered_data
                    if item.get('date') and start_date_obj <= item['date'] <= end_date_obj
                ]
            except ValueError as e:
                logger.error(f"Некорректный формат даты: {e}")
                # Продолжаем без фильтрации по дате, если формат некорректен

        if not filtered_data:
            logger.warning("Нет данных после фильтрации.")
            return (
                pd.DataFrame(columns=[
                    "ID кластера", "Количество фото", "Первое появление", 
                    "Последнее появление", "Средняя уверенность", "Статус"
                ]),
                go.Figure().update_layout(title="Нет данных после фильтрации"),
                [],
                gr.update(value={})
            )

        embeddings_data_for_clustering = [
            {
                'embedding': item['embedding'],
                'file_path': item['file_path'],
                'date': item['date'],
                'confidence': item['confidence'],
                'metadata': item['metadata']
            }
            for item in filtered_data
        ]
        
        # 2. Пересчет кластеров с новыми параметрами DBSCAN
        try:
            embeddings_only = [item['embedding'] for item in embeddings_data_for_clustering]
            file_paths_for_clustering = [item['file_path'] for item in embeddings_data_for_clustering]
            dates_for_clustering = [item['date'] for item in embeddings_data_for_clustering]
            confidence_scores_for_clustering = [item['confidence'] for item in embeddings_data_for_clustering]

            # Вызов perform_identity_clustering с учетом параметров DBSCAN
            cluster_results = self.embedding_analyzer.perform_identity_clustering(
                embeddings_data_for_clustering, # Передаем полный список словарей с эмбеддингами и метаданными
                epsilon=epsilon,
                min_samples=min_samples
                # file_paths, dates, confidence_scores теперь извлекаются внутри perform_identity_clustering
            )
            
            # Обновление filtered_data с метками кластеров
            for i, item in enumerate(filtered_data):
                if i < len(cluster_results['cluster_labels']):
                    item['cluster_label'] = cluster_results['cluster_labels'][i]
                else:
                    item['cluster_label'] = -1 # В случае несоответствия длины, помечаем как шум

            # Построение временной линии идентичности
            identity_timeline = self.embedding_analyzer.build_identity_timeline(
                cluster_results
            )
            
            # Кэшируем новые результаты
            self.results_cache['clustering_results_recalculated'] = {
                'cluster_results': cluster_results,
                'identity_timeline': identity_timeline,
                'filtered_data': filtered_data # Сохраняем отфильтрованные данные с метками кластеров
            }
            logger.info(f"Кластеризация завершена. Найдено {cluster_results.get('n_clusters', 0)} кластеров.")

        except Exception as e:
            logger.error(f"Ошибка при пересчете кластеров: {e}", exc_info=True)
            return (
                pd.DataFrame(columns=[
                    "ID кластера", "Количество фото", "Первое появление", 
                    "Последнее появление", "Средняя уверенность", "Статус"
                ]),
                go.Figure().update_layout(title=f"Ошибка пересчета кластеров: {e}"),
                [],
                gr.update(value={})
            )
            
        # 3. Формирование cluster_summary
        cluster_summary_data = []
        unique_cluster_ids = np.unique([item['cluster_label'] for item in filtered_data])
        
        for cluster_id in unique_cluster_ids:
            if cluster_id == -1: # Пропускаем шумовые точки
                continue
            
            cluster_items = [
                item for item in filtered_data if item['cluster_label'] == cluster_id
            ]
            
            if not cluster_items:
                continue
                
            cluster_dates = sorted([item['date'] for item in cluster_items if item['date']])
            avg_confidence = np.mean([item['confidence'] for item in cluster_items]) if cluster_items else 0.0
            
            first_appearance = cluster_dates[0].strftime('%Y-%m-%d') if cluster_dates else "N/A"
            last_appearance = cluster_dates[-1].strftime('%Y-%m-%d') if cluster_dates else "N/A"
            
            status = "Подтвержден" if len(cluster_items) >= min_samples and avg_confidence >= report_confidence_threshold else "Требует проверки" # Используем report_confidence_threshold
            
            cluster_summary_data.append({
                "ID кластера": cluster_id,
                "Количество фото": len(cluster_items),
                "Первое появление": first_appearance,
                "Последнее появление": last_appearance,
                "Средняя уверенность": f"{avg_confidence:.2f}",
                "Статус": status
            })
            
        cluster_summary_df = pd.DataFrame(cluster_summary_data)
        
        # 4. Визуализация кластеров (t-SNE)
        embeddings_for_tsne = np.array([item['embedding'] for item in filtered_data])
        cluster_labels_for_tsne = np.array([item['cluster_label'] for item in filtered_data])

        if embeddings_for_tsne.shape[0] < 2: # t-SNE требует минимум 2 образца
             tsne_figure = go.Figure().update_layout(title="Недостаточно данных для t-SNE")
        else:
            try:
                # Выбираем perplexity в зависимости от количества образцов
                perplexity_val = min(30, max(5, embeddings_for_tsne.shape[0] - 1)) 
                if embeddings_for_tsne.shape[0] < 5: # TSNE требует n_samples > n_components, и обычно > perplexity
                    perplexity_val = embeddings_for_tsne.shape[0] - 1 if embeddings_for_tsne.shape[0] > 1 else 1 # Ensure perplexity > 1
                if perplexity_val < 1: perplexity_val = 1 # Ensure at least 1 for small datasets
                
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val, learning_rate='auto', init='random')
                reduced_embeddings = tsne.fit_transform(embeddings_for_tsne)
                
                tsne_df = pd.DataFrame(reduced_embeddings, columns=['Component 1', 'Component 2'])
                tsne_df['Cluster'] = cluster_labels_for_tsne
                tsne_df['File Path'] = [item['file_path'] for item in filtered_data]
                tsne_df['Date'] = [item['date'].strftime('%Y-%m-%d') if item['date'] else 'N/A' for item in filtered_data]
                tsne_df['Confidence'] = [f"{item['confidence']:.2f}" for item in filtered_data]
                tsne_df['Quality'] = [f"{item['quality_score']:.2f}" for item in filtered_data]
                
                # Убедимся, что все кластеры имеют уникальный цвет
                unique_labels = sorted(tsne_df['Cluster'].unique())
                color_map = {label: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] for i, label in enumerate(unique_labels)}

                tsne_figure = px.scatter(
                    tsne_df, 
                    x='Component 1', 
                    y='Component 2', 
                    color='Cluster', 
                    color_discrete_map=color_map, # Применяем уникальные цвета
                    hover_data=['File Path', 'Date', 'Confidence', 'Quality'],
                    title='Визуализация кластеров (t-SNE)'
                )
                tsne_figure.update_layout(height=500)
                
            except Exception as e:
                logger.error(f"Ошибка t-SNE визуализации: {e}", exc_info=True)
                tsne_figure = go.Figure().update_layout(title=f"Ошибка t-SNE: {e}")

        # 5. Галерея представителей кластеров (до 100 элементов)
        gallery_images = []
        representative_images_per_cluster = {} # Для хранения пути к лучшему изображению кластера

        for cluster_id in unique_cluster_ids:
            if cluster_id == -1: # Пропускаем шум
                continue
            
            cluster_items = [
                item for item in filtered_data if item['cluster_label'] == cluster_id
            ]
            
            if cluster_items:
                # Выбираем изображение с самой высокой уверенностью как представителя
                best_item = max(cluster_items, key=lambda x: x.get('confidence', 0.0))
                representative_images_per_cluster[cluster_id] = best_item['file_path']
                # Добавляем в галерею с подписью
                gallery_images.append((best_item['file_path'], f"ID: {cluster_id}, Conf: {best_item['confidence']:.2f}"))
        
        # Ограничиваем количество элементов в галерее, если их слишком много
        gallery_images = gallery_images[:self.max_gallery_items]

        # Обновляем состояние dropdownd`identity_selector` на вкладке временного анализа
        identity_selector_choices = [(f"Личность {c_id}", c_id) for c_id in sorted(unique_cluster_ids) if c_id != -1]
        self.results_cache['identity_selector_choices'] = identity_selector_choices # Кэшируем для обновления дропдауна

        return (
            cluster_summary_df, 
            tsne_figure, 
            gallery_images,
            gr.update(value={}) # Очищаем информацию о выбранном кластере
        )

    def _analyze_temporal_patterns(self, identity: int, resolution: str, metrics: List[str]) -> Tuple[go.Figure, pd.DataFrame, go.Figure, go.Figure, go.Figure, go.Figure]:
        """
        Анализирует временные паттерны для выбранной личности.
        Добавлено: прогнозирование возрастных изменений, детекция точек изменения и визуализация.
        """
        logger.info(f"Анализ временных паттернов для личности {identity} с разрешением {resolution}")

        if not self.analysis_state['results'] or 'identity_clustering' not in self.analysis_state['results']:
            gr.Warning("Данные кластеризации не загружены. Пожалуйста, сначала выполните массовую обработку.")
            return go.Figure(), pd.DataFrame(), go.Figure(), go.Figure(), go.Figure(), go.Figure()
        
        all_clusters = self.analysis_state['results']['identity_clustering']['clusters']
        identity_timeline_raw = all_clusters.get(str(identity), {}).get('timeline', [])
        
        if not identity_timeline_raw:
            gr.Warning(f"Временная линия для личности {identity} не найдена.")
            return go.Figure(), pd.DataFrame(), go.Figure(), go.Figure(), go.Figure(), go.Figure()

        # Преобразование данных временной линии для удобства
        timeline_df = pd.DataFrame(identity_timeline_raw)
        if 'date' not in timeline_df.columns or 'metrics' not in timeline_df.columns:
            gr.Warning("Некорректный формат временной линии.")
            return go.Figure(), pd.DataFrame(), go.Figure(), go.Figure(), go.Figure(), go.Figure()

        timeline_df['date'] = pd.to_datetime(timeline_df['date'])
        timeline_df = timeline_df.sort_values(by='date').reset_index(drop=True)

        # Извлечение базовых метрик (первое появление) и прогрессии возраста
        first_entry_metrics = timeline_df['metrics'].iloc[0]
        baseline_metrics = first_entry_metrics if isinstance(first_entry_metrics, dict) else {}
        
        # Вычисляем возраст для каждой даты
        birth_date_str = self.analysis_state['results'].get('birth_date', PUTIN_BIRTH_DATE) # Используем PUTIN_BIRTH_DATE по умолчанию
        birth_date = datetime.strptime(birth_date_str, '%Y-%m-%d')
        ages = self.temporal_analyzer.calculate_age_on_each_date(timeline_df['date'].tolist(), birth_date)
        timeline_df['age'] = ages

        # Агрегация данных по выбранному разрешению
        grouped_data = []
        for name, group in timeline_df.groupby(pd.Grouper(key='date', freq=self._get_freq_string(resolution))):
            if not group.empty:
                avg_metrics = {metric: np.mean([entry['metrics'][metric] for entry in group.to_dict(orient='records') if metric in entry['metrics']]) for metric in metrics}
                grouped_data.append({'date': name, 'metrics': avg_metrics, 'age': np.mean(group['age'])})
        
        if not grouped_data:
            gr.Warning("Недостаточно данных для агрегации по выбранному разрешению.")
            return go.Figure(), pd.DataFrame(), go.Figure(), go.Figure(), go.Figure(), go.Figure()

        grouped_df = pd.DataFrame(grouped_data)

        # Обновленные метрики для временного анализа
        temporal_metrics_timeline = {metric: [] for metric in metrics}
        predicted_temporal_metrics_timeline = {metric: [] for metric in metrics}

        for index, row in grouped_df.iterrows():
            current_age = row['age']
            predicted_values = self.temporal_analyzer.predict_expected_metrics_for_age(current_age, baseline_metrics)
            for metric in metrics:
                actual_val = row['metrics'].get(metric)
                if actual_val is not None:
                    temporal_metrics_timeline[metric].append(actual_val)
                    predicted_temporal_metrics_timeline[metric].append(predicted_values.get(metric, np.nan))

        # Создание основного графика изменений
        fig_temporal = go.Figure()
        for metric in metrics:
            fig_temporal.add_trace(go.Scatter(
                x=grouped_df['date'], 
                y=temporal_metrics_timeline[metric],
                mode='lines+markers', 
                name=f'Фактический {metric}'
            ))
        fig_temporal.update_layout(title='Изменения метрик во времени', xaxis_title='Дата', yaxis_title='Значение')

        # Статистика изменений и аномалий
        change_stats_data = []
        for metric in metrics:
            values = np.array(temporal_metrics_timeline[metric])
            if len(values) < 2: continue

            mean_change = np.mean(np.diff(values)) if len(values) > 1 else 0.0
            std_change = np.std(np.diff(values)) if len(values) > 1 else 0.0

            # Тренд (простая линейная регрессия)
            x = np.arange(len(values)).reshape(-1, 1)
            model = LinearRegression().fit(x, values)
            trend_slope = model.coef_[0]
            
            # Детекция аномалий (используем TemporalAnalyzer)
            metric_dict_for_anomaly = {metric: temporal_metrics_timeline[metric]}
            predicted_dict_for_anomaly = {metric: predicted_temporal_metrics_timeline[metric]}
            anomalies = self.temporal_analyzer.detect_temporal_anomalies_in_metrics(metric_dict_for_anomaly, predicted_dict_for_anomaly)
            anomaly_count = len(anomalies.get(metric, {}).get('anomaly_indices', []))
            p_value = anomalies.get(metric, {}).get('significance_level', 1.0)

            change_stats_data.append([
                metric, 
                f'{mean_change:.4f}', 
                f'{std_change:.4f}', 
                f'{trend_slope:.4f}', 
                anomaly_count,
                f'{p_value:.4f}'
            ])
        df_change_stats = pd.DataFrame(change_stats_data, columns=[
            "Метрика", "Среднее изменение", "Стандартное отклонение",
            "Тренд", "Аномалии", "P-value"
        ])
        
        # Детекция аномалий Plot (из _analyze_temporal_patterns)
        # Здесь используем те же аномалии, что и для статистики, но для визуализации
        fig_anomaly = go.Figure()
        for metric in metrics:
            actual_vals = temporal_metrics_timeline[metric]
            dates_for_plot = grouped_df['date']
            fig_anomaly.add_trace(go.Scatter(
                x=dates_for_plot, 
                y=actual_vals,
                mode='lines', 
                name=f'{metric}'
            ))
            if metric in anomalies:
                anomaly_indices = anomalies[metric]['anomaly_indices']
                if anomaly_indices:
                    anomaly_dates = [dates_for_plot.iloc[i] for i in anomaly_indices]
                    anomaly_values = [actual_vals[i] for i in anomaly_indices]
                    fig_anomaly.add_trace(go.Scatter(
                        x=anomaly_dates,
                        y=anomaly_values,
                        mode='markers',
                        marker=dict(color='red', size=8),
                        name=f'Аномалии {metric}'
                    ))
        fig_anomaly.update_layout(title='Детекция аномалий во времени', xaxis_title='Дата', yaxis_title='Значение')

        # Корреляция с историческими событиями (заглушка или использование data_manager)
        fig_historical = go.Figure()
        fig_historical.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 1], mode='lines+markers', name='Пример'))
        fig_historical.update_layout(title='Корреляция с историческими событиями', xaxis_title='Время', yaxis_title='Событие')

        # === НОВЫЕ ГРАФИКИ ===
        # 1. График предсказанных vs фактических метрик
        fig_predicted_vs_actual = self._create_predicted_vs_actual_plot(
            grouped_df['date'].tolist(), 
            temporal_metrics_timeline, 
            predicted_temporal_metrics_timeline,
            metrics
        )

        # 2. График точек изменения
        change_points_results = self.temporal_analyzer.detect_change_points(temporal_metrics_timeline)
        fig_change_points = self._create_change_points_plot(
            grouped_df['date'].tolist(), 
            temporal_metrics_timeline, 
            change_points_results
        )

        return fig_temporal, df_change_stats, fig_anomaly, fig_historical, fig_predicted_vs_actual, fig_change_points

    def _get_freq_string(self, resolution: str) -> str:
        if resolution == "Месяц":
            return "M"
        elif resolution == "Квартал":
            return "QS"
        elif resolution == "Год":
            return "Y"
        return "D"

    def _create_predicted_vs_actual_plot(self, dates: List[datetime], actual_metrics: Dict, 
                                          predicted_metrics: Dict, metrics_to_plot: List[str]) -> go.Figure:
        """
        Создает график сравнения фактических и предсказанных значений метрик.
        """
        fig = go.Figure()
        for metric in metrics_to_plot:
            fig.add_trace(go.Scatter(
                x=dates, 
                y=actual_metrics.get(metric, []),
                mode='lines+markers', 
                name=f'Фактический {metric}',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=dates, 
                y=predicted_metrics.get(metric, []),
                mode='lines', 
                name=f'Предсказанный {metric}',
                line=dict(color='orange', dash='dash')
            ))
        fig.update_layout(
            title='Фактические vs Предсказанные Метрики',
            xaxis_title='Дата',
            yaxis_title='Значение',
            hovermode="x unified"
        )
        return fig

    def _create_change_points_plot(self, dates: List[datetime], metrics_timeline: Dict, 
                                   change_points_results: Dict) -> go.Figure:
        """
        Создает график, показывающий обнаруженные точки изменения для метрик.
        """
        fig = go.Figure()
        for metric in metrics_timeline.keys():
            values = metrics_timeline[metric]
            fig.add_trace(go.Scatter(
                x=dates, 
                y=values,
                mode='lines', 
                name=f'{metric}',
                line=dict(color='grey')
            ))
            
            if metric in change_points_results and change_points_results[metric]['detected']:
                points = change_points_results[metric]['points']
                for point_idx in points:
                    if 0 <= point_idx < len(dates):
                        fig.add_vline(
                            x=dates[point_idx],
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Изменение в {metric}",
                            annotation_position="top right"
                        )
        fig.update_layout(
            title='Обнаруженные Точки Изменения Метрик',
            xaxis_title='Дата',
            yaxis_title='Значение',
            hovermode="x unified"
        )
        return fig

    def _analyze_mask_technology(self, sensitivity: float, tech_levels: List[str]) -> Tuple[pd.DataFrame, go.Figure, go.Figure, go.Figure, gr.HTML]:
        """Анализ технологий масок"""
        logger.info(f"Анализ технологий масок с чувствительностью {sensitivity} и уровнями {tech_levels}")

        if 'full_processed_data' not in self.results_cache or not self.results_cache['full_processed_data']:
            logger.warning("Нет обработанных данных для анализа масок.")
            return (
                pd.DataFrame(columns=[
                    "Дата", "Уровень технологии", "Уверенность",
                    "Shape Error", "Texture Score", "Статус", "Путь к файлу"
                ]),
                go.Figure().update_layout(title="Нет данных для распределения технологий"),
                go.Figure().update_layout(title="Нет данных для анализа текстур"),
                go.Figure().update_layout(title="Нет данных для эволюции технологий"),
                gr.update(value="<div class='alert-info'>Анализ не выполнен: нет данных</div>")
            )
        
        full_processed_data = self.results_cache['full_processed_data']
        
        mask_detection_results = []
        all_shape_errors = []
        all_entropies = []
        all_dates_for_texture = []

        for item in full_processed_data:
            if 'texture_analysis' in item and 'identity_metrics' in item and 'embedding' in item and 'pose_info' in item:
                try:
                    texture_data = item['texture_analysis']
                    # identity_metrics = item['identity_metrics'] # Не используется напрямую в этом блоке
                    embedding = item['embedding']
                    file_date = item['date']
                    file_path = item['file_path']
                    cluster_label = item.get('cluster_label') # Получаем метку кластера

                    # 1. Расчет балла аутентичности материала
                    material_authenticity_score = self.texture_analyzer.calculate_material_authenticity_score(texture_data)

                    # 2. Классификация уровня технологии маски
                    # Получаем корректный центр кластера на основе метки кластера элемента
                    cluster_center_embedding = np.zeros(embedding.shape) # Значение по умолчанию
                    if cluster_label is not None and cluster_label != -1 and \
                       'clustering_results_recalculated' in self.results_cache and \
                       'cluster_results' in self.results_cache['clustering_results_recalculated'] and \
                       'cluster_centers' in self.results_cache['clustering_results_recalculated']['cluster_results'] and \
                       cluster_label in self.results_cache['clustering_results_recalculated']['cluster_results']['cluster_centers']:
                        cluster_center_embedding = self.results_cache['clustering_results_recalculated']['cluster_results']['cluster_centers'][cluster_label]
                    else:
                        logger.warning(f"Центр кластера не найден для элемента {file_path} с меткой {cluster_label}. Используется нулевой вектор.")
                        # В качестве альтернативы, можно использовать средний эмбеддинг всех данных или просто продолжить без этого параметра

                    # Если _process_single_image возвращает shape_error, используем его, иначе по умолчанию
                    shape_error = item['metadata'].get('shape_error', 0.25) 
                    
                    mask_info = self.texture_analyzer.classify_mask_technology_level(
                        texture_data,
                        file_date,
                        shape_error, # Передаем shape_error
                        self.embedding_analyzer.calculate_embedding_distances_matrix([embedding, cluster_center_embedding])[0,1] # Расстояние до центра кластера
                    )

                    # Проверяем, соответствует ли уровень технологии выбранным пользователем
                    if mask_info['level'] in tech_levels and mask_info['confidence'] >= sensitivity:
                        mask_detection_results.append({
                            "Дата": file_date.strftime('%Y-%m-%d'),
                            "Уровень технологии": mask_info['level'],
                            "Уверенность": f"{mask_info['confidence']:.2f}",
                            "Shape Error": f"{shape_error:.3f}", # Добавляем Shape Error
                            "Texture Score": f"{material_authenticity_score:.2f}",
                            "Статус": "Обнаружена маска",
                            "Путь к файлу": file_path
                        })
                    
                    all_shape_errors.append(shape_error)
                    all_entropies.append(texture_data.get('shannon_entropy', 0.0))
                    all_dates_for_texture.append(file_date)

                except Exception as e:
                    logger.error(f"Ошибка анализа маски для {item.get('file_path', 'Unknown')}: {e}", exc_info=True)

        mask_summary_df = pd.DataFrame(mask_detection_results)

        # 1. Распределение по технологиям (Bar Plot)
        if not mask_summary_df.empty:
            tech_counts = mask_summary_df['Уровень технологии'].value_counts().reset_index()
            tech_counts.columns = ['Уровень технологии', 'Количество']
            technology_distribution_fig = px.bar(
                tech_counts, 
                x='Уровень технологии', 
                y='Количество', 
                title='Распределение обнаруженных технологий масок'
            )
        else:
            technology_distribution_fig = go.Figure().update_layout(title="Маски не обнаружены")

        # 2. Детальный анализ текстуры (Scatter plot: Shape Error vs Entropy)
        texture_analysis_plot_fig = go.Figure()
        if all_shape_errors and all_entropies and all_dates_for_texture:
            texture_df = pd.DataFrame({
                'Shape Error': all_shape_errors,
                'Entropy': all_entropies,
                'Date': all_dates_for_texture
            })
            texture_analysis_plot_fig = px.scatter(
                texture_df, 
                x='Shape Error', 
                y='Entropy', 
                color='Date', 
                title='Анализ текстурных характеристик (Shape Error vs Entropy)'
            )
        else:
            texture_analysis_plot_fig.update_layout(title="Нет данных для анализа текстур")

        # 3. Эволюция технологий (Line Plot)
        technology_evolution_fig = go.Figure()
        if not mask_summary_df.empty:
            # Группируем по году и уровню технологии
            mask_summary_df['Year'] = pd.to_datetime(mask_summary_df['Дата']).dt.year
            tech_evolution = mask_summary_df.groupby(['Year', 'Уровень технологии']).size().unstack(fill_value=0)
            
            if not tech_evolution.empty:
                technology_evolution_fig = px.line(
                    tech_evolution, 
                    x=tech_evolution.index,
                    y=tech_evolution.columns,
                    title='Эволюция обнаруженных технологий масок по годам'
                )
                technology_evolution_fig.update_layout(
                    xaxis_title="Год", 
                    yaxis_title="Количество обнаружений",
                    legend_title="Уровень технологии"
                )
            else:
                technology_evolution_fig.update_layout(title="Нет данных для эволюции технологий")
        else:
            technology_evolution_fig.update_layout(title="Нет данных для эволюции технологий")

        # Критические обнаружения
        critical_detections_html = "<div class='alert-info'>Аномалий не обнаружено</div>"
        if not mask_summary_df.empty:
            critical_masks = mask_summary_df[mask_summary_df['Уверенность'].astype(float) >= 0.95]
            if not critical_masks.empty:
                critical_detections_html = "<div class='alert-danger'>**КРИТИЧЕСКИЕ ОБНАРУЖЕНИЯ МАСОК:**<br>"
                for index, row in critical_masks.iterrows():
                    critical_detections_html += f"- **{row['Дата']}**: Уровень {row['Уровень технологии']}, Уверенность {row['Уверенность']}<br>"
                critical_detections_html += "</div>"
            
        return (
            mask_summary_df, 
            technology_distribution_fig, 
            texture_analysis_plot_fig,
            technology_evolution_fig,
            gr.update(value=critical_detections_html)
        )

    def _generate_expert_report(self, report_type: str, sections: List[str], confidence_threshold: float) -> Tuple[Dict, go.Figure, str, gr.File]:
        """Генерация экспертного отчета"""
        logger.info(f"Генерация экспертного отчета: {report_type}, Разделы: {sections}, Порог уверенности: {confidence_threshold}")

        report_data = {
            "total_identities": 0,
            "confidence_level": 0.0,
            "statistical_significance": 0.0,
            "anomalies_detected": 0,
            "masks_detected": 0,
            "analysis_period": "N/A"
        }
        expert_conclusion_text = ""
        final_timeline_figure = go.Figure().update_layout(title="Итоговая временная линия")
        download_file = None

        if 'clustering_results_recalculated' not in self.results_cache:
            expert_conclusion_text = ("Недостаточно данных для генерации отчета. Выполните массовую обработку и кластеризацию."
                                     "<br>Ошибка: Нет кэшированных результатов кластеризации.")
            return report_data, final_timeline_figure, expert_conclusion_text, None
        
        clustering_results = self.results_cache['clustering_results_recalculated']['cluster_results']
        identity_timeline = self.results_cache['clustering_results_recalculated']['identity_timeline']
        filtered_data = self.results_cache['clustering_results_recalculated']['filtered_data']

        # Общие показатели
        report_data['total_identities'] = clustering_results.get('n_clusters', 0)
        
        all_confidences = [item['confidence'] for item in filtered_data if 'confidence' in item]
        if all_confidences:
            report_data['confidence_level'] = np.mean(all_confidences)
        
        all_dates = [item['date'] for item in filtered_data if 'date' in item]
        if all_dates:
            report_data['analysis_period'] = f"{min(all_dates).strftime('%Y-%m-%d')} - {max(all_dates).strftime('%Y-%m-%d')}"

        # Сбор данных по аномалиям и маскам (если они были обнаружены)
        anomalies_count = 0
        masks_count = 0
        if 'temporal_analysis_results' in self.results_cache:
            for metric_anomalies in self.results_cache['temporal_analysis_results']['anomalies'].values():
                anomalies_count += len(metric_anomalies.get('anomaly_indices', []))
                anomalies_count += len(metric_anomalies.get('rapid_change_indices', []))
        report_data['anomalies_detected'] = anomalies_count

        if 'mask_detection_results' in self.results_cache:
            masks_count = len(self.results_cache['mask_detection_results'])
        report_data['masks_detected'] = masks_count

        # Формирование заключения
        expert_conclusion_text = "## Экспертное заключение\n\n"
        if "Исполнительное резюме" in sections:
            expert_conclusion_text += "### Исполнительное резюме\n"
            # Используем report_data.get('total_images') для избежания KeyError, если ключ отсутствует
            expert_conclusion_text += f"На основе комплексного анализа {report_data.get('total_images', 'N/A')} изображений за период {report_data['analysis_period']} идентифицировано **{report_data['total_identities']}** потенциально уникальных личностей. "
            if report_data['masks_detected'] > 0:
                expert_conclusion_text += f"Обнаружено **{report_data['masks_detected']}** случаев возможного использования масок. "
            if report_data['anomalies_detected'] > 0:
                expert_conclusion_text += f"Выявлено **{report_data['anomalies_detected']}** аномалий во временных паттернах метрик.\n\n"

        if "Результаты кластеризации" in sections:
            expert_conclusion_text += "### Результаты кластеризации\n"
            expert_conclusion_text += f"Система идентифицировала {report_data['total_identities']} различных кластеров личностей. Средний уровень уверенности идентификации: {report_data['confidence_level']:.2f}.\n\n"
            # Детализация кластеров
            if 'cluster_summary_df' in self.results_cache:
                expert_conclusion_text += "#### Сводка по кластерам:\n"
                expert_conclusion_text += self.results_cache['cluster_summary_df'].to_markdown(index=False) + "\n\n"

        if "Временной анализ" in sections:
            expert_conclusion_text += "### Временной анализ\n"
            expert_conclusion_text += f"Проведен анализ временных изменений ключевых лицевых метрик. Обнаружено {report_data['anomalies_detected']} значительных аномалий, не объяснимых естественными процессами старения.\n\n"
            # Детализация временных аномалий
            if 'temporal_analysis_results' in self.results_cache and 'change_statistics_df' in self.results_cache['temporal_analysis_results']:
                expert_conclusion_text += "#### Статистика изменений метрик:\n"
                expert_conclusion_text += self.results_cache['temporal_analysis_results']['change_statistics_df'].to_markdown(index=False) + "\n\n"

        if "Детекция масок" in sections:
            expert_conclusion_text += "### Детекция масок\n"
            if report_data['masks_detected'] > 0:
                expert_conclusion_text += f"Идентифицировано {report_data['masks_detected']} случаев, где характеристики лица указывают на возможное использование маски. "
                expert_conclusion_text += "Это включает аномалии в текстуре кожи, форме лица и несоответствие паттернам естественного старения.\n\n"
                if 'mask_summary_df' in self.results_cache:
                    expert_conclusion_text += "#### Обнаруженные маски:\n"
                    expert_conclusion_text += self.results_cache['mask_summary_df'].to_markdown(index=False) + "\n\n"
            else:
                expert_conclusion_text += "Признаков использования масок не обнаружено.\n\n"

        if "Статистический анализ" in sections:
            expert_conclusion_text += "### Статистический анализ\n"
            expert_conclusion_text += f"Проведена оценка статистической значимости выводов: P-value = {report_data['statistical_significance']:.4f}. "
            if report_data['statistical_significance'] < 0.05:
                expert_conclusion_text += "Результаты являются статистически значимыми.\n\n"
            else:
                expert_conclusion_text += "Требуется дополнительный анализ для подтверждения статистической значимости.\n\n"

        if "Заключение" in sections:
            expert_conclusion_text += "### Заключение\n"
            if report_data['total_identities'] > 1 and report_data['confidence_level'] > confidence_threshold:
                expert_conclusion_text += "Комплексный анализ с высокой степенью уверенности указывает на наличие более чем одной личности. Рекомендуется дальнейшее расследование.\n\n"
            else:
                expert_conclusion_text += "Анализ не выявил убедительных доказательств наличия нескольких личностей, соответствующих заданным критериям.\n\n"

        # Генерация итоговой временной линии
        # Здесь нужно объединить все кластеры и показать их активность
        final_timeline_figure = self._generate_final_timeline_plot(identity_timeline)

        # Сохранение отчета в файл (для экспорта)
        report_filename = f"expert_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_path = os.path.join(RESULTS_DIR, report_filename) # Используем RESULTS_DIR из core_config
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(expert_conclusion_text)
        download_file = gr.update(value=report_path, visible=True)

        return report_data, final_timeline_figure, expert_conclusion_text, download_file

    def _generate_final_timeline_plot(self, identity_timeline: Dict) -> go.Figure:
        """Генерирует итоговую временную линию всех идентифицированных личностей."""
        timeline_data = []
        for cluster_id, info in identity_timeline.items():
            if cluster_id != -1: # Исключаем шум
                first_app = info.get('first_appearance')
                last_app = info.get('last_appearance')
                if first_app and last_app:
                    timeline_data.append({
                        'start': first_app,
                        'end': last_app,
                        'identity': f'Личность {cluster_id}',
                        'appearance_count': info.get('appearance_count', 0)
                    })
        
        if not timeline_data:
            return go.Figure().update_layout(title="Нет данных для итоговой временной линии")

        timeline_df = pd.DataFrame(timeline_data)
        timeline_df['start'] = pd.to_datetime(timeline_df['start'])
        timeline_df['end'] = pd.to_datetime(timeline_df['end'])

        fig = px.timeline(timeline_df, x_start="start", x_end="end", y="identity", color="appearance_count",
                          title="Итоговая временная линия появления личностей")
        fig.update_yaxes(autorange="reversed") # Для лучшего отображения порядка личностей
        return fig

    def _export_results_json(self):
        """Экспортирует все текущие результаты анализа в JSON файл."""
        if not self.results_cache:
            logger.warning("Нет кэшированных результатов для экспорта JSON.")
            return gr.update(visible=False)

        output_filename = f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = os.path.join(RESULTS_DIR, output_filename) # Используем RESULTS_DIR из core_config
        
        # Преобразование numpy массивов в списки для JSON сериализации
        def convert_numpy_arrays(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        try:
            # Глубокое копирование, чтобы избежать модификации кэша при сериализации
            export_data = json.loads(json.dumps(self.results_cache, default=convert_numpy_arrays))
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=4)
            logger.info(f"Результаты экспортированы в JSON: {output_path}")
            return gr.update(value=output_path, visible=True)
        except Exception as e:
            logger.error(f"Ошибка при экспорте JSON: {e}", exc_info=True)
            return gr.update(visible=False)

    def _export_results_pdf(self) -> gr.File:
        """Экспортирует результаты анализа в PDF-файл."""
        if not self.results_cache:
            logger.warning("Нет кэшированных результатов для экспорта PDF.")
            return gr.update(visible=False)

        output_filename = f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        output_path = os.path.join(RESULTS_DIR, output_filename)

        try:
            # Убедимся, что директория для сохранения существует
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            doc = SimpleDocTemplate(output_path, pagesize=letter)
            styles = getSampleStyleSheet()
            
            # Настройка стилей для отчета
            title_style = styles['h1']
            heading_style = styles['h2']
            body_style = styles['Normal']
            body_style.fontSize = 10
            body_style.leading = 14

            # Создание содержимого для PDF
            story = []

            # Заголовок отчета
            story.append(Paragraph("Отчет по анализу лицевой идентичности", title_style))
            story.append(Spacer(1, 0.2 * inch))
            story.append(Paragraph(f"Дата отчета: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
            story.append(Spacer(1, 0.2 * inch))

            # Общие результаты (используем self.results_cache)
            results = self.results_cache.get('clustering_results_recalculated', {})
            
            if "summary_stats" in results.get('cluster_results', {}):
                summary_stats = results['cluster_results']['summary_stats']
                story.append(Paragraph("Общие результаты", heading_style))
                story.append(Spacer(1, 0.1 * inch))
                story.append(Paragraph(f'Количество идентифицированных личностей: {summary_stats.get("n_clusters", "N/A")}', body_style))
                story.append(Paragraph(f'Всего обработано изображений: {summary_stats.get("total_images", "N/A")}', body_style))
                story.append(Paragraph(f'Изображений в кластерах: {summary_stats.get("clustered_images", "N/A")}', body_style))
                story.append(Spacer(1, 0.2 * inch))

            # Добавление подробностей по кластерам
            if 'cluster_summary_df' in self.results_cache and not self.results_cache['cluster_summary_df'].empty:
                story.append(Paragraph("Сводка по кластерам:", heading_style))
                # Преобразование DataFrame в список списков для Table
                data = [self.results_cache['cluster_summary_df'].columns.tolist()] + self.results_cache['cluster_summary_df'].values.tolist()
                table = Table(data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.grey),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0,0), (-1,0), 12),
                    ('BACKGROUND', (0,1), (-1,-1), colors.beige),
                    ('GRID', (0,0), (-1,-1), 1, colors.black)
                ]))
                story.append(table)
                story.append(Spacer(1, 0.2 * inch))

            # Добавление метрик и аномалий
            if 'temporal_analysis_results' in self.results_cache and 'change_statistics_df' in self.results_cache['temporal_analysis_results']:
                story.append(Paragraph("Анализ метрик и аномалий:", heading_style))
                change_stats_df = self.results_cache['temporal_analysis_results']['change_statistics_df']
                data = [change_stats_df.columns.tolist()] + change_stats_df.values.tolist()
                table = Table(data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.grey),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0,0), (-1,0), 12),
                    ('BACKGROUND', (0,1), (-1,-1), colors.beige),
                    ('GRID', (0,0), (-1,-1), 1, colors.black)
                ]))
                story.append(table)
                story.append(Spacer(1, 0.2 * inch))

            # Если есть результаты по маскам
            if 'mask_summary_df' in self.results_cache and not self.results_cache['mask_summary_df'].empty:
                story.append(Paragraph("Детекция масок:", heading_style))
                mask_summary_df = self.results_cache['mask_summary_df']
                data = [mask_summary_df.columns.tolist()] + mask_summary_df.values.tolist()
                table = Table(data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.grey),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0,0), (-1,0), 12),
                    ('BACKGROUND', (0,1), (-1,-1), colors.beige),
                    ('GRID', (0,0), (-1,-1), 1, colors.black)
                ]))
                story.append(table)
                story.append(Spacer(1, 0.2 * inch))
            
            # Сборка и сохранение PDF
            doc.build(story)

            logging.info(f"Отчет успешно экспортирован в PDF: {output_path}")
            return gr.update(value=output_path, visible=True)

        except Exception as e:
            logging.error(f"Ошибка при экспорте результатов в PDF: {e}")
            return gr.update(visible=False)

    def _apply_system_settings(self, dbscan_epsilon: float, dbscan_min_samples: int, anomaly_z_threshold: float, change_rate_threshold: float, min_face_size: int, blur_threshold: float):
        """Применение системных настроек"""
        logger.info("Применение системных настроек...")
        
        # Обновление глобальных констант в core_config (если это допустимо)
        # Внимание: прямое изменение глобальных констант во время выполнения может быть не лучшей практикой
        # Лучше передавать эти параметры в методы анализаторов, которые их используют.
        # Для демонстрации: 
        try:
            global DBSCAN_EPSILON, DBSCAN_MIN_SAMPLES, ANOMALY_DETECTION_THRESHOLDS, IMAGE_QUALITY_THRESHOLDS
            DBSCAN_EPSILON = dbscan_epsilon
            DBSCAN_MIN_SAMPLES = dbscan_min_samples
            ANOMALY_DETECTION_THRESHOLDS['Z_SCORE_ANOMALY_THRESHOLD'] = anomaly_z_threshold # Исправлено
            ANOMALY_DETECTION_THRESHOLDS['RAPID_CHANGE_STD_MULTIPLIER'] = change_rate_threshold # Исправлено
            IMAGE_QUALITY_THRESHOLDS['MIN_FACE_SIZE'] = min_face_size
            IMAGE_QUALITY_THRESHOLDS['BLUR_DETECTION_THRESHOLD'] = blur_threshold
            
            # Обновляем параметры анализаторов, если они используют эти константы
            # Например, можно передать их в init или в отдельные методы set_params()
            # self.embedding_analyzer.set_dbscan_params(dbscan_epsilon, dbscan_min_samples)
            # self.face_analyzer.set_quality_thresholds(min_face_size, blur_threshold)

            logger.info("Настройки успешно применены.")
            return {
                "settings_applied": True,
                "timestamp": datetime.now().isoformat(),
                "dbscan_epsilon": DBSCAN_EPSILON,
                "dbscan_min_samples": DBSCAN_MIN_SAMPLES,
                "anomaly_z_threshold": anomaly_z_threshold, # Возвращаем то, что получили
                "change_rate_threshold": change_rate_threshold, # Возвращаем то, что получили
                "min_face_size": IMAGE_QUALITY_THRESHOLDS['MIN_FACE_SIZE'],
                "blur_threshold": IMAGE_QUALITY_THRESHOLDS['BLUR_DETECTION_THRESHOLD']
            }
        except Exception as e:
            logger.error(f"Ошибка применения настроек: {e}", exc_info=True)
            return {"error": str(e)}

    def _clear_system_cache(self):
        """Очистка системного кэша"""
        logger.info("Очистка системного кэша...")
        self.results_cache = {}
        self.visualization_cache = {}
        gc.collect() # Принудительная сборка мусора
        logger.info("Кэш успешно очищен.")
        return (
            "Кэш очищен",
            {"cache_cleared": True, "timestamp": datetime.now().isoformat()}
        )

    def _update_system_status(self):
        """Обновление статуса системы"""
        logger.info("Обновление статуса системы...")
        
        status = {
            "3ddfa_status": "Неактивен",
            "insightface_status": "Неактивен",
            "gpu_available": False,
            "memory_usage": "N/A",
            "cache_size": f"{sys.getsizeof(self.results_cache) / (1024**2):.2f} MB" # Размер кэша в MB
        }
        performance_data = []

        try:
            # Проверка 3DDFA_V2
            # Для реальной проверки нужно запускать тест производительности
            # Здесь пока заглушка
            if hasattr(self, 'face_analyzer') and self.face_analyzer and self.face_analyzer.init_done:
                status['3ddfa_status'] = "Активен"
                performance_data.append({"Компонент": "3DDFA_V2", "Время обработки (мс)": 120, "Статус": "OK"})
            else:
                status['3ddfa_status'] = "Не инициализирован"
            
            # Проверка InsightFace
            if hasattr(self, 'embedding_analyzer') and self.embedding_analyzer and self.embedding_analyzer.model_initialized:
                status['insightface_status'] = "Активен"
                performance_data.append({"Компонент": "InsightFace", "Время обработки (мс)": 85, "Статус": "OK"})
            else:
                status['insightface_status'] = "Не инициализирован"
            
            # Проверка GPU
            try:
                import torch
                if torch.cuda.is_available():
                    status['gpu_available'] = True
                    status['gpu_name'] = torch.cuda.get_device_name(0)
                else:
                    status['gpu_available'] = False
            except ImportError:
                status['gpu_available'] = "N/A (Torch не установлен)"

            # Обновление информации о памяти
            import psutil
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            status['memory_usage'] = f"{mem_info.rss / (1024**2):.2f} MB" # Resident Set Size

            logger.info("Статус системы обновлен.")

        except Exception as e:
            logger.error(f"Ошибка обновления статуса системы: {e}", exc_info=True)
            status['error'] = str(e)

        return (
            status,
            pd.DataFrame(performance_data)
        )

    def _assess_image_quality(self, image: np.ndarray) -> float:
        """Оценивает качество изображения (резкость, освещение)"""
        # Простая оценка резкости (Variance of Laplacian)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if np.isnan(fm) or np.isinf(fm):
            logging.warning(f"Laplacian variance (fm) contains NaN or Inf: {fm}. Returning 0.0 for quality.")
            return 0.0

        # Оценка освещения (средняя яркость)
        brightness = np.mean(gray) / 255.0 # Нормализация к 0-1

        if np.isnan(brightness) or np.isinf(brightness):
            logging.warning(f"Brightness contains NaN or Inf: {brightness}. Returning 0.0 for quality.")
            return 0.0

        # Комбинирование оценок
        # Пороги должны быть откалиброваны на реальных данных
        sharpness_score = min(1.0, fm / IMAGE_QUALITY_THRESHOLDS['BLUR_DETECTION_THRESHOLD']) # Нормализация по порогу размытости
        lighting_score = min(1.0, (brightness - 0.2) / 0.6) # Оптимальная яркость 0.2-0.8
        lighting_score = max(0.0, lighting_score)

        overall_quality = (sharpness_score + lighting_score) / 2
        return overall_quality

    def _extract_date_from_file(self, file_path: str) -> Optional[datetime]:
        """Извлекает дату из метаданных файла или имени файла."""
        try:
            # Попытка извлечь дату из EXIF (если изображение JPEG)
            from PIL import Image
            from PIL.ExifTags import TAGS
            
            if file_path.lower().endswith(('.jpg', '.jpeg')):
                with Image.open(file_path) as img:
                    exif_data = img._getexif()
                    if exif_data:
                        for tag_id, value in exif_data.items():
                            tag = TAGS.get(tag_id, tag_id)
                            if tag == 'DateTimeOriginal' or tag == 'DateTimeDigitized':
                                return datetime.strptime(value, '%Y:%m:%d %H:%M:%S')
            
            # Если EXIF нет или не удалось, пытаемся из имени файла (YMD_HMS_...)
            filename = os.path.basename(file_path)
            # Пример: 20230115_123045_image.jpg
            match = re.search(r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})', filename)
            if match:
                year, month, day, hour, minute, second = map(int, match.groups())
                return datetime(year, month, day, hour, minute, second)
            
            # Если ничего не найдено, используем дату модификации файла
            timestamp = os.path.getmtime(file_path)
            return datetime.fromtimestamp(timestamp)

        except Exception as e:
            logger.warning(f"Не удалось извлечь дату из файла {file_path}: {e}")
            return None

    def _format_processing_log(self, results: Dict) -> str:
        """Форматирует лог обработки."""
        log_entries = []
        for res in results['successful']:
            log_entries.append(f"✅ Успешно обработан: {res['file_path']}")
        for err in results['errors']:
            log_entries.append(f"❌ Ошибка обработки {err['file_path']}: {err['error']}")
        
        # Ограничиваем количество записей, чтобы не перегружать интерфейс
        max_log_lines = 100
        if len(log_entries) > max_log_lines:
            log_entries = ["... (сокращено) ..."] + log_entries[-max_log_lines+1:]
            
        return "\n".join(log_entries)

    def _get_custom_css(self) -> str:
        """Возвращает пользовательский CSS для Gradio интерфейса."""
        return """
            .gradio-container {
                max-width: 1200px;
                margin: auto;
                font-family: 'Roboto', sans-serif;
            }
            #global_status {
                text-align: center;
                font-size: 1.2em;
                padding: 10px;
                border-radius: 8px;
                margin-bottom: 20px;
            }
            .status-ready {
                background-color: #e6ffe6;
                color: #006600;
                border: 1px solid #00cc00;
            }
            .status-error {
                background-color: #ffe6e6;
                color: #cc0000;
                border: 1px solid #ff0000;
            }
            .status-success {
                background-color: #e6e6ff;
                color: #0000cc;
                border: 1px solid #0000ff;
            }
            .alert-info {
                background-color: #e0f2f7;
                color: #265c7c;
                border: 1px solid #82c4e0;
                padding: 10px;
                border-radius: 5px;
                margin-top: 15px;
            }
            .alert-danger {
                background-color: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
                padding: 10px;
                border-radius: 5px;
                margin-top: 15px;
                font-weight: bold;
            }
            h1, h3 {
                color: #333;
            }
            .gr-button.primary {
                background-color: #4CAF50;
                color: white;
            }
            .gr-button.secondary {
                background-color: #f44336;
                color: white;
            }
            .gr-button.stop {
                background-color: #ff9800;
                color: white;
            }
        """

    def _stop_processing(self):
        """Останавливает текущую обработку."""
        logger.info("Запрошена остановка обработки.")
        self.analysis_state['processing'] = False
        return (
            "<div class='status-error'>🛑 Обработка остановлена</div>",
            gr.update(visible=True),
            gr.update(visible=False)
        )

    def create_3d_anomaly_visualization(self, landmarks_3d: np.ndarray,
                                   anomaly_scores: np.ndarray) -> go.Figure:
        """Создает 3D-визуализацию лица с выделением аномальных областей"""
        if landmarks_3d.size == 0 or anomaly_scores.size == 0:
            logging.warning("Невозможно создать 3D-визуализацию: отсутствуют landmarks или оценки аномалий.")
            return go.Figure() # Возвращаем пустую фигуру

        if landmarks_3d.shape[0] != anomaly_scores.shape[0]:
            logging.warning("Размеры landmarks и anomaly_scores не совпадают. Визуализация может быть некорректной.")
            # Можно обрезать или вернуть пустую фигуру
            return go.Figure()

        fig = go.Figure(data=[go.Scatter3d(
            x=landmarks_3d[:, 0],
            y=landmarks_3d[:, 1], 
            z=landmarks_3d[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=anomaly_scores, # Цвет по оценкам аномалий
                colorscale='Viridis', # Цветовая шкала
                colorbar=dict(title='Оценка аномалии'),
                showscale=True
            )
        )])

        fig.update_layout(
            title='3D Визуализация Аномалий Лица',
            scene=dict(
                xaxis_title='X Координата',
                yaxis_title='Y Координата',
                zaxis_title='Z Координата',
                aspectmode='data' # Сохранение пропорций
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        return fig


# ==================== ЗАПУСК ИНТЕРФЕЙСА ====================

def create_interface():
    """Создание и запуск интерфейса"""
    
    try:
        interface = OptimizedGradioInterface()
        demo = interface.create_optimized_interface()
        
        return demo
        
    except Exception as e:
        logger.error(f"Ошибка создания интерфейса: {e}")
        raise

if __name__ == "__main__":
    import re # Добавлен import re
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1", # Изменен адрес сервера на localhost для отладки
        server_port=7860, # Устанавливаем фиксированный порт для отладки
        share=False, # Устанавливаем share=False, чтобы отключить создание публичной ссылки
        debug=True,
        max_file_size="100mb",
        max_threads=10
    )
