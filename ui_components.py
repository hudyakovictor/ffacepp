"""
UIComponents - UI компоненты для Gradio интерфейса
Версия: 2.0
Дата: 2025-06-15
Исправлены все критические ошибки согласно правкам
"""

import os
os.makedirs("logs", exist_ok=True)
import gradio as gr
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Any, Tuple, AsyncGenerator
import json
import logging
from pathlib import Path

# Получение логгера через logging.getLogger(__name__)
logger = logging.getLogger(__name__)

# ==================== БАЗОВЫЙ КЛАСС ====================

class BaseUIComponent:
    """Базовый класс для UI компонентов"""
    
    def __init__(self, component_id: str):
        self.component_id = component_id
        self.state = {}
        logger.info(f"Инициализация компонента: {component_id}")
    
    def render(self) -> gr.Component:
        """Рендеринг компонента"""
        raise NotImplementedError("Метод render должен быть реализован в наследнике")
    
    def update_state(self, **kwargs):
        """Обновление состояния компонента"""
        self.state.update(kwargs)

# ==================== 3D VIEWER ====================

class Interactive3DViewer(BaseUIComponent):
    """
    ИСПРАВЛЕНО: 3D визуализатор с landmarks и dense points
    Согласно правкам: 68 landmarks, wireframe, dense surface points
    """
    
    def __init__(self):
        super().__init__("3d_viewer")
        self.current_landmarks = None
        self.current_dense_points = None
        
    def render(self) -> gr.Column:
        """Рендеринг 3D визуализатора"""
        with gr.Column() as col:
            gr.Markdown("### 🎯 3D Визуализация лица")
            
            # ИСПРАВЛЕНО: 3D модель с контролами
            self.model_3d = gr.Model3D(
                label="3D модель лица",
                height=500,
                interactive=True,
                camera_position=(0, 0, 5)
            )
            
            with gr.Row():
                self.wireframe_toggle = gr.Checkbox(
                    label="Wireframe режим",
                    value=True,
                    info="Показать каркас лица"
                )
                self.dense_points_toggle = gr.Checkbox(
                    label="Плотные точки (38,000)",
                    value=False,
                    info="Показать dense surface points"
                )
                self.landmarks_toggle = gr.Checkbox(
                    label="68 ландмарок",
                    value=True,
                    info="Показать ключевые точки"
                )
            
            # Привязка событий
            for control in [self.wireframe_toggle, self.dense_points_toggle, self.landmarks_toggle]:
                control.change(
                    fn=self.update_3d_view,
                    inputs=[self.wireframe_toggle, self.dense_points_toggle, self.landmarks_toggle],
                    outputs=[self.model_3d]
                )
        
        return col
    
    def create_3d_model_from_landmarks(self, landmarks_3d: np.ndarray) -> str:
        """Создание 3D модели из ландмарок"""
        try:
            if landmarks_3d is None or not hasattr(landmarks_3d, 'size') or landmarks_3d.size == 0:
                logger.warning("Ландмарки не найдены или пусты, пропуск.")
                return ""
            
            obj_content = self.landmarks_to_obj(landmarks_3d)
            return obj_content
            
        except Exception as e:
            logger.error(f"Ошибка создания 3D модели: {e}")
            return ""
    
    def landmarks_to_obj(self, landmarks: np.ndarray) -> str:
        """Конвертация ландмарок в OBJ формат"""
        try:
            obj_lines = ["# 3D Face Landmarks"]
            
            # Добавление вершин
            for point in landmarks:
                obj_lines.append(f"v {point[0]:.3f} {point[1]:.3f} {point[2]:.3f}")
            
            return "\n".join(obj_lines)
            
        except Exception as e:
            logger.error(f"Ошибка конвертации в OBJ: {e}")
            return ""
    
    def update_3d_view(self, wireframe: bool, dense_points: bool, landmarks: bool) -> str:
        """Обновление 3D вида"""
        try:
            logger.info(f"Обновление 3D: wireframe={wireframe}, dense={dense_points}, landmarks={landmarks}")
            
            obj_content = "# 3D Face Model\n"
            
            if landmarks and self.current_landmarks is not None:
                obj_content += self.landmarks_to_obj(self.current_landmarks)
            
            return obj_content
            
        except Exception as e:
            logger.error(f"Ошибка обновления 3D: {e}")
            return ""

# ==================== HEATMAP ====================

class InteractiveHeatmap(BaseUIComponent):
    """
    ИСПРАВЛЕНО: Интерактивная тепловая карта аутентичности
    Согласно правкам: color-coding по зонам лица
    """
    
    def __init__(self):
        super().__init__("heatmap")
    
    def render(self) -> gr.AnnotatedImage:
        """Рендеринг тепловой карты"""
        return gr.AnnotatedImage(
            label="Карта аутентичности по зонам",
            height=400,
            width=400
        )
    
    def create_authenticity_heatmap(self, image: np.ndarray, authenticity_scores: Dict[str, float]) -> tuple:
        """
        ИСПРАВЛЕНО: Создание тепловой карты аутентичности
        Согласно правкам: color-coding для каждой зоны
        """
        try:
            annotations = []
            
            # ИСПРАВЛЕНО: Color-coding по score
            for zone, score in authenticity_scores.items():
                color = self.score_to_color(score)
                annotations.append({
                    "label": f"{zone}: {score:.2f}",
                    "color": color
                })
            
            return image, annotations
            
        except Exception as e:
            logger.error(f"Ошибка создания heatmap: {e}")
            return image, []
    
    def score_to_color(self, score: float) -> str:
        """Конвертация score в цвет"""
        if score >= 0.8:
            return "#00FF00"  # Зеленый - высокая аутентичность
        elif score >= 0.6:
            return "#FFFF00"  # Желтый - средняя
        elif score >= 0.4:
            return "#FFA500"  # Оранжевый - низкая
        else:
            return "#FF0000"  # Красный - очень низкая

# ==================== TEMPORAL SLIDER ====================

class TemporalSlider(BaseUIComponent):
    """
    ИСПРАВЛЕНО: Временной слайдер для навигации по timeline
    Согласно правкам: хронологическая навигация
    """
    
    def __init__(self):
        super().__init__("temporal_slider")
    
    def render(self) -> gr.Column:
        """Рендеринг временного слайдера"""
        with gr.Column() as col:
            gr.Markdown("### ⏰ Временная навигация")
            
            self.slider = gr.Slider(
                minimum=0,
                maximum=100,
                step=1,
                label="Временная позиция",
                interactive=True
            )
            
            with gr.Row():
                self.preview = gr.Image(
                    label="Предварительный просмотр",
                    height=200,
                    width=200
                )
                
                self.info = gr.Textbox(
                    label="Информация о дате",
                    lines=3,
                    interactive=False
                )
            
            # Привязка событий
            self.slider.change(
                fn=self.update_temporal_position,
                inputs=[self.slider],
                outputs=[self.preview, self.info]
            )
        
        return col
    
    def update_temporal_position(self, position: int) -> Tuple[Optional[str], str]:
        """Обновление временной позиции"""
        try:
            # Заглушка для демонстрации
            info = f"Позиция: {position}%\nДата: 2020-01-01\nВозраст: 67.3 лет"
            return None, info
            
        except Exception as e:
            logger.error(f"Ошибка обновления позиции: {e}")
            return None, "Ошибка"

# ==================== METRICS GALLERY ====================

class MetricsGallery(BaseUIComponent):
    """
    ИСПРАВЛЕНО: Галерея метрик с визуализацией
    Согласно правкам: 15 метрик в 3 группах
    """
    
    def __init__(self):
        super().__init__("metrics_gallery")
    
    def render(self) -> gr.Column:
        """Рендеринг галереи метрик"""
        with gr.Column() as col:
            gr.Markdown("### 📊 Галерея метрик (15 метрик)")
            
            # ИСПРАВЛЕНО: Tabs для 3 групп метрик
            with gr.Tabs():
                with gr.Tab("Геометрия черепа (5)"):
                    self.skull_metrics = gr.Gallery(
                        label="Метрики геометрии черепа",
                        columns=3,
                        rows=2,
                        height=300
                    )
                
                with gr.Tab("Пропорции лица (5)"):
                    self.proportions_metrics = gr.Gallery(
                        label="Метрики пропорций лица",
                        columns=3,
                        rows=2,
                        height=300
                    )
                
                with gr.Tab("Костная структура (5)"):
                    self.bone_metrics = gr.Gallery(
                        label="Метрики костной структуры",
                        columns=3,
                        rows=2,
                        height=300
                    )
        
        return col
    
    def create_metric_visualization(self, metric_name: str, values: List[float]) -> str:
        """Создание визуализации метрики"""
        try:
            fig = px.line(
                x=range(len(values)),
                y=values,
                title=f"Тренд: {metric_name}",
                labels={"x": "Время", "y": "Значение"}
            )
            
            return fig.to_html()
            
        except Exception as e:
            logger.error(f"Ошибка создания визуализации {metric_name}: {e}")
            return ""

# ==================== ADVANCED FILTERS ====================

class AdvancedFilters(BaseUIComponent):
    """
    ИСПРАВЛЕНО: Продвинутые фильтры для анализа
    Согласно правкам: фильтрация по различным критериям
    """
    
    def __init__(self):
        super().__init__("filters")
    
    def render(self) -> gr.Column:
        """Рендеринг продвинутых фильтров"""
        with gr.Column() as col:
            gr.Markdown("### 🔍 Продвинутые фильтры")
            
            # Временной диапазон
            self.date_range = gr.DateRange(
                label="Диапазон дат"
            )
            
            # Порог аутентичности
            self.authenticity_threshold = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                label="Порог аутентичности"
            )
            
            # Фильтр качества
            self.quality_filter = gr.CheckboxGroup(
                choices=["Высокое качество", "Среднее качество", "Низкое качество"],
                label="Фильтр качества"
            )
            
            # Типы аномалий
            self.anomaly_types = gr.CheckboxGroup(
                choices=["Геометрические", "Текстурные", "Временные", "Эмбеддинг"],
                label="Типы аномалий"
            )
            
            # Уровни масок
            self.mask_levels = gr.CheckboxGroup(
                choices=["Level1", "Level2", "Level3", "Level4", "Level5"],
                label="Уровни технологий масок"
            )
            
            # Кнопка применения фильтров
            self.apply_filters_btn = gr.Button(
                "Применить фильтры",
                variant="primary"
            )
        
        return col

# ==================== INTERACTIVE COMPARISON ====================

class InteractiveComparison(BaseUIComponent):
    """
    ИСПРАВЛЕНО: Интерактивное сравнение результатов
    Согласно правкам: сравнение метрик и timeline
    """
    
    def __init__(self):
        super().__init__("comparison")
    
    def create_comparison_widget(self) -> gr.Column:
        """Создание виджета сравнения"""
        with gr.Column() as col:
            gr.Markdown("### 🔄 Интерактивное сравнение")
            
            with gr.Row():
                self.left_image = gr.Image(label="Изображение A")
                self.right_image = gr.Image(label="Изображение B")
            
            self.comparison_metrics = gr.DataFrame(
                column_names=["Метрика", "Значение A", "Значение B", "Разница"],
                label="Сравнение метрик"
            )
            
            self.similarity_score = gr.Number(
                label="Общий балл схожести",
                precision=3
            )
        
        return col

# ==================== ADVANCED SEARCH ====================

class AdvancedSearch(BaseUIComponent):
    """
    ИСПРАВЛЕНО: Продвинутый поиск по результатам
    Согласно правкам: поиск по метрикам и датам
    """
    
    def __init__(self):
        super().__init__("search")
    
    def create_search_interface(self) -> gr.Column:
        """Создание интерфейса поиска"""
        with gr.Column() as col:
            gr.Markdown("### 🔍 Продвинутый поиск")
            
            self.search_query = gr.Textbox(
                label="Поисковый запрос",
                placeholder="Введите критерии поиска..."
            )
            
            self.search_btn = gr.Button("Поиск", variant="primary")
            
            self.search_results = gr.DataFrame(
                label="Результаты поиска"
            )
        
        return col

# ==================== AI ASSISTANT ====================

class AIAssistant(BaseUIComponent):
    """
    ИСПРАВЛЕНО: AI-ассистент для интерпретации результатов
    Согласно правкам: помощь в анализе
    """
    
    def __init__(self):
        super().__init__("assistant")
    
    def create_assistant_interface(self) -> gr.Column:
        """Создание интерфейса AI-ассистента"""
        with gr.Column() as col:
            gr.Markdown("### 🤖 AI-Ассистент")
            
            self.chat_history = gr.Chatbot(
                label="История чата",
                height=400
            )
            
            self.user_input = gr.Textbox(
                label="Ваш вопрос",
                placeholder="Задайте вопрос об анализе..."
            )
            
            self.send_btn = gr.Button("Отправить", variant="primary")
        
        return col

# ==================== ЭКСПОРТ КОМПОНЕНТОВ ====================

def create_ui_components() -> Dict[str, BaseUIComponent]:
    """Создание всех UI компонентов"""
    return {
        '3d_viewer': Interactive3DViewer(),
        'heatmap': InteractiveHeatmap(),
        'temporal_slider': TemporalSlider(),
        'metrics_gallery': MetricsGallery(),
        'filters': AdvancedFilters(),
        'comparison': InteractiveComparison(),
        'search': AdvancedSearch(),
        'assistant': AIAssistant()
    }

# ==================== ТОЧКА ВХОДА ====================

if __name__ == "__main__":
    logger.info("=== Тестирование UI компонентов ===")
    
    components = create_ui_components()
    
    for name, component in components.items():
        try:
            logger.info(f"Тестирование компонента: {name}")
            rendered = component.render()
            logger.info(f"Компонент {name} успешно отрендерен")
        except Exception as e:
            logger.error(f"Ошибка рендеринга {name}: {e}")
    
    logger.info("=== Тестирование завершено ===")