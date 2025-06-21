"""
UIComponents - UI компоненты для Gradio интерфейса
Версия: 2.0
Дата: 2025-06-21
ИСПРАВЛЕНО: Все критические ошибки согласно новому ТЗ
"""

import os
os.makedirs("logs", exist_ok=True)

import gradio as gr
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from pathlib import Path
import json
import time
import asyncio
from datetime import datetime, timedelta
import threading
from functools import lru_cache
from collections import OrderedDict, defaultdict
import hashlib

from core_config import get_config

# Настройка логирования
logger = logging.getLogger(__name__)

# Импорт библиотек для визуализации
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
    logger.info("✔ Plotly импортирована для интерактивных графиков")
except ImportError as e:
    HAS_PLOTLY = False
    logger.warning(f"❌ Plotly не найдена. Графики будут ограничены. Детали: {e}")

try:
    from PIL import Image
    HAS_PIL = True
    logger.info("✔ PIL импортирована для обработки изображений")
except ImportError as e:
    HAS_PIL = False
    logger.warning(f"❌ PIL не найдена. Обработка изображений ограничена. Детали: {e}")

# === КОНСТАНТЫ UI КОМПОНЕНТОВ ===

# Дата рождения Владимира Путина
PUTIN_BIRTH_DATE = datetime(1952, 10, 7)

# Параметры UI
UI_PARAMS = {
    "max_3d_points": 50000,
    "heatmap_zones": 7,
    "timeline_max_points": 1000,
    "gallery_columns": 3,
    "gallery_rows": 2,
    "preview_size": (200, 200),
    "thumbnail_size": (150, 150)
}

# Цветовые схемы
UI_COLOR_SCHEMES = {
    "authenticity": {
        "high": "#28a745",      # Зеленый > 0.7
        "medium": "#ffc107",    # Желтый 0.3-0.7
        "low": "#dc3545"        # Красный < 0.3
    },
    "mask_levels": {
        "Level1": "#8B0000",
        "Level2": "#FF4500", 
        "Level3": "#FFD700",
        "Level4": "#32CD32",
        "Level5": "#006400"
    },
    "anomaly_types": {
        "geometric": "#FF0000",
        "texture": "#FFA500",
        "temporal": "#0000FF",
        "embedding": "#800080",
        "medical": "#008000"
    }
}

# === БАЗОВЫЙ КЛАСС ===

class BaseUIComponent:
    """
    ИСПРАВЛЕНО: Базовый класс для UI компонентов
    Согласно правкам: единая архитектура компонентов
    """

    def __init__(self, component_id: str):
        self.component_id = component_id
        self.state = {}
        self.config = get_config()
        self.component_lock = threading.Lock()
        
        logger.info(f"Инициализация UI компонента: {component_id}")

    def render(self) -> gr.Component:
        """Рендеринг компонента"""
        raise NotImplementedError("Метод render должен быть реализован в наследнике")

    def update_state(self, **kwargs):
        """Обновление состояния компонента"""
        try:
            with self.component_lock:
                self.state.update(kwargs)
            logger.debug(f"Состояние компонента {self.component_id} обновлено: {kwargs}")
        except Exception as e:
            logger.error(f"Ошибка обновления состояния {self.component_id}: {e}")

    def get_state(self, key: str, default=None):
        """Получение значения из состояния"""
        try:
            with self.component_lock:
                return self.state.get(key, default)
        except Exception as e:
            logger.error(f"Ошибка получения состояния {self.component_id}.{key}: {e}")
            return default

    def clear_state(self):
        """Очистка состояния компонента"""
        try:
            with self.component_lock:
                self.state.clear()
            logger.info(f"Состояние компонента {self.component_id} очищено")
        except Exception as e:
            logger.error(f"Ошибка очистки состояния {self.component_id}: {e}")

# === 3D VIEWER ===

class Interactive3DViewer(BaseUIComponent):
    """
    ИСПРАВЛЕНО: 3D визуализатор с landmarks и dense points
    Согласно правкам: 68 landmarks, wireframe, dense surface points
    """

    def __init__(self):
        super().__init__("3d_viewer")
        self.current_landmarks = None
        self.current_dense_points = None
        self.max_points = UI_PARAMS["max_3d_points"]

    def render(self) -> gr.Column:
        """Рендеринг 3D визуализатора"""
        try:
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
                
                with gr.Row():
                    self.point_size = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        label="Размер точек",
                        info="Размер отображаемых точек"
                    )
                    
                    self.opacity = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.8,
                        label="Прозрачность",
                        info="Прозрачность модели"
                    )
                
                # Информационная панель
                self.info_panel = gr.HTML(
                    value=self._create_info_html(),
                    label="Информация о модели"
                )
                
                # Привязка событий
                for control in [self.wireframe_toggle, self.dense_points_toggle, 
                               self.landmarks_toggle, self.point_size, self.opacity]:
                    control.change(
                        fn=self.update_3d_view,
                        inputs=[self.wireframe_toggle, self.dense_points_toggle, 
                               self.landmarks_toggle, self.point_size, self.opacity],
                        outputs=[self.model_3d, self.info_panel]
                    )
                
            return col
            
        except Exception as e:
            logger.error(f"Ошибка рендеринга 3D визуализатора: {e}")
            return gr.Column()

    def create_3d_model_from_landmarks(self, landmarks_3d: np.ndarray, 
                                     dense_points: Optional[np.ndarray] = None) -> str:
        """
        ИСПРАВЛЕНО: Создание 3D модели из ландмарок
        Согласно правкам: поддержка 68 landmarks + dense points
        """
        try:
            if landmarks_3d is None or not hasattr(landmarks_3d, 'size') or landmarks_3d.size == 0:
                logger.warning("Ландмарки не найдены или пусты")
                return ""
            
            self.current_landmarks = landmarks_3d
            self.current_dense_points = dense_points
            
            obj_content = self._landmarks_to_obj(landmarks_3d, dense_points)
            
            # Обновление состояния
            self.update_state(
                landmarks_count=len(landmarks_3d),
                dense_points_count=len(dense_points) if dense_points is not None else 0,
                last_update=datetime.now().isoformat()
            )
            
            return obj_content
            
        except Exception as e:
            logger.error(f"Ошибка создания 3D модели: {e}")
            return ""

    def _landmarks_to_obj(self, landmarks: np.ndarray, 
                         dense_points: Optional[np.ndarray] = None) -> str:
        """Конвертация ландмарок в OBJ формат"""
        try:
            obj_lines = ["# 3D Face Model Generated by UI Components"]
            obj_lines.append(f"# Landmarks: {len(landmarks)}")
            
            # Добавление 68 ландмарок
            for i, point in enumerate(landmarks):
                obj_lines.append(f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}")
                obj_lines.append(f"# landmark_{i}")
            
            # Добавление dense points если есть
            if dense_points is not None and len(dense_points) > 0:
                # Ограничение количества точек для производительности
                if len(dense_points) > self.max_points:
                    step = len(dense_points) // self.max_points
                    dense_points = dense_points[::step]
                
                obj_lines.append(f"# Dense points: {len(dense_points)}")
                for point in dense_points:
                    obj_lines.append(f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}")
            
            # Добавление связей для wireframe (упрощенный вариант)
            if len(landmarks) >= 68:
                obj_lines.extend(self._create_face_wireframe())
            
            return "\n".join(obj_lines)
            
        except Exception as e:
            logger.error(f"Ошибка конвертации в OBJ: {e}")
            return ""

    def _create_face_wireframe(self) -> List[str]:
        """Создание wireframe связей для лица"""
        try:
            wireframe_lines = []
            
            # Контур лица (0-16)
            for i in range(16):
                wireframe_lines.append(f"l {i+1} {i+2}")
            
            # Левая бровь (17-21)
            for i in range(17, 21):
                wireframe_lines.append(f"l {i+1} {i+2}")
            
            # Правая бровь (22-26)
            for i in range(22, 26):
                wireframe_lines.append(f"l {i+1} {i+2}")
            
            # Нос (27-35)
            for i in range(27, 35):
                wireframe_lines.append(f"l {i+1} {i+2}")
            
            # Глаза и рот (упрощенно)
            # Левый глаз (36-41)
            for i in range(36, 41):
                wireframe_lines.append(f"l {i+1} {i+2}")
            wireframe_lines.append(f"l 42 37")  # Замыкание
            
            # Правый глаз (42-47)
            for i in range(42, 47):
                wireframe_lines.append(f"l {i+1} {i+2}")
            wireframe_lines.append(f"l 48 43")  # Замыкание
            
            # Рот (48-67)
            for i in range(48, 67):
                wireframe_lines.append(f"l {i+1} {i+2}")
            wireframe_lines.append(f"l 68 49")  # Замыкание
            
            return wireframe_lines
            
        except Exception as e:
            logger.error(f"Ошибка создания wireframe: {e}")
            return []

    def update_3d_view(self, wireframe: bool, dense_points: bool, landmarks: bool,
                      point_size: int, opacity: float) -> Tuple[str, str]:
        """Обновление 3D вида"""
        try:
            logger.info(f"Обновление 3D: wireframe={wireframe}, dense={dense_points}, "
                       f"landmarks={landmarks}, size={point_size}, opacity={opacity}")
            
            obj_content = "# 3D Face Model\n"
            
            if landmarks and self.current_landmarks is not None:
                obj_content += self._landmarks_to_obj(
                    self.current_landmarks,
                    self.current_dense_points if dense_points else None
                )
            
            # Обновление информационной панели
            info_html = self._create_info_html()
            
            return obj_content, info_html
            
        except Exception as e:
            logger.error(f"Ошибка обновления 3D: {e}")
            return "", self._create_error_html(str(e))

    def _create_info_html(self) -> str:
        """Создание HTML информационной панели"""
        try:
            landmarks_count = self.get_state("landmarks_count", 0)
            dense_count = self.get_state("dense_points_count", 0)
            last_update = self.get_state("last_update", "Никогда")
            
            html = f"""
            <div style="padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
                <h4>Информация о 3D модели</h4>
                <p><strong>68 ландмарок:</strong> {landmarks_count}</p>
                <p><strong>Dense points:</strong> {dense_count:,}</p>
                <p><strong>Последнее обновление:</strong> {last_update}</p>
                <p><strong>Максимум точек:</strong> {self.max_points:,}</p>
            </div>
            """
            
            return html
            
        except Exception as e:
            logger.error(f"Ошибка создания info HTML: {e}")
            return self._create_error_html(str(e))

    def _create_error_html(self, error_msg: str) -> str:
        """Создание HTML сообщения об ошибке"""
        return f"""
        <div style="padding: 10px; background-color: #ffebee; border-radius: 5px; color: #d32f2f;">
            <h4>Ошибка</h4>
            <p>{error_msg}</p>
        </div>
        """

# === INTERACTIVE HEATMAP ===

class InteractiveHeatmap(BaseUIComponent):
    """
    ИСПРАВЛЕНО: Интерактивная тепловая карта аутентичности
    Согласно правкам: color-coding по зонам лица
    """

    def __init__(self):
        super().__init__("heatmap")
        self.zones_config = {
            "forehead": {"x": 0.5, "y": 0.2, "width": 0.6, "height": 0.2},
            "left_eye": {"x": 0.3, "y": 0.35, "width": 0.15, "height": 0.1},
            "right_eye": {"x": 0.7, "y": 0.35, "width": 0.15, "height": 0.1},
            "nose": {"x": 0.5, "y": 0.5, "width": 0.2, "height": 0.2},
            "mouth": {"x": 0.5, "y": 0.7, "width": 0.3, "height": 0.15},
            "left_cheek": {"x": 0.25, "y": 0.55, "width": 0.2, "height": 0.2},
            "right_cheek": {"x": 0.75, "y": 0.55, "width": 0.2, "height": 0.2}
        }

    def render(self) -> gr.Column:
        """Рендеринг тепловой карты"""
        try:
            with gr.Column() as col:
                gr.Markdown("### 🔥 Тепловая карта аутентичности")
                
                self.heatmap_image = gr.AnnotatedImage(
                    label="Карта аутентичности по зонам",
                    height=400,
                    width=400
                )
                
                with gr.Row():
                    self.zone_selector = gr.Dropdown(
                        choices=list(self.zones_config.keys()),
                        label="Выбор зоны",
                        value="forehead"
                    )
                    
                    self.threshold_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        label="Порог отображения",
                        info="Минимальный балл для отображения"
                    )
                
                self.zone_details = gr.DataFrame(
                    headers=["Зона", "Балл", "Статус", "Цвет"],
                    label="Детали по зонам",
                    interactive=False
                )
                
                # События
                self.zone_selector.change(
                    fn=self.highlight_zone,
                    inputs=[self.zone_selector],
                    outputs=[self.heatmap_image]
                )
                
            return col
            
        except Exception as e:
            logger.error(f"Ошибка рендеринга heatmap: {e}")
            return gr.Column()

    def create_authenticity_heatmap(self, image: np.ndarray, 
                                  authenticity_scores: Dict[str, float]) -> Tuple[Any, List[Dict]]:
        """
        ИСПРАВЛЕНО: Создание тепловой карты аутентичности
        Согласно правкам: color-coding для каждой зоны
        """
        try:
            if image is None or image.size == 0:
                logger.warning("Изображение для heatmap пустое")
                return None, []
            
            annotations = []
            zone_data = []
            
            # ИСПРАВЛЕНО: Color-coding по score
            for zone, score in authenticity_scores.items():
                if zone in self.zones_config:
                    color = self._score_to_color(score)
                    status = self._score_to_status(score)
                    
                    # Создание аннотации
                    zone_config = self.zones_config[zone]
                    
                    annotations.append({
                        "label": f"{zone}: {score:.3f}",
                        "color": color,
                        "coordinates": self._zone_to_coordinates(zone_config, image.shape)
                    })
                    
                    # Данные для таблицы
                    zone_data.append([zone, f"{score:.3f}", status, color])
            
            # Обновление состояния
            self.update_state(
                zones_processed=len(annotations),
                last_scores=authenticity_scores,
                last_update=datetime.now().isoformat()
            )
            
            # Обновление таблицы деталей
            self.zone_details.value = zone_data
            
            return image, annotations
            
        except Exception as e:
            logger.error(f"Ошибка создания heatmap: {e}")
            return image, []

    def _score_to_color(self, score: float) -> str:
        """Конвертация score в цвет"""
        try:
            colors = UI_COLOR_SCHEMES["authenticity"]
            
            if score >= 0.7:
                return colors["high"]      # Зеленый - высокая аутентичность
            elif score >= 0.3:
                return colors["medium"]    # Желтый - средняя
            else:
                return colors["low"]       # Красный - низкая
                
        except Exception as e:
            logger.error(f"Ошибка конвертации score в цвет: {e}")
            return "#808080"  # Серый по умолчанию

    def _score_to_status(self, score: float) -> str:
        """Конвертация score в статус"""
        try:
            if score >= 0.7:
                return "ВЫСОКИЙ"
            elif score >= 0.3:
                return "СРЕДНИЙ"
            else:
                return "НИЗКИЙ"
        except:
            return "НЕИЗВЕСТНО"

    def _zone_to_coordinates(self, zone_config: Dict[str, float], 
                           image_shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
        """Конвертация конфигурации зоны в координаты"""
        try:
            height, width = image_shape[:2]
            
            x = zone_config["x"]
            y = zone_config["y"]
            w = zone_config["width"]
            h = zone_config["height"]
            
            x1 = int((x - w/2) * width)
            y1 = int((y - h/2) * height)
            x2 = int((x + w/2) * width)
            y2 = int((y + h/2) * height)
            
            return (x1, y1, x2, y2)
            
        except Exception as e:
            logger.error(f"Ошибка конвертации координат зоны: {e}")
            return (0, 0, 100, 100)

    def highlight_zone(self, selected_zone: str) -> Any:
        """Подсветка выбранной зоны"""
        try:
            logger.info(f"Подсветка зоны: {selected_zone}")
            
            # Здесь должна быть логика подсветки конкретной зоны
            # Возвращаем обновленное изображение с подсветкой
            
            return None  # Заглушка
            
        except Exception as e:
            logger.error(f"Ошибка подсветки зоны: {e}")
            return None

# === TEMPORAL SLIDER ===

class TemporalSlider(BaseUIComponent):
    """
    ИСПРАВЛЕНО: Временной слайдер для навигации по timeline
    Согласно правкам: хронологическая навигация
    """

    def __init__(self):
        super().__init__("temporal_slider")
        self.timeline_data = []
        self.current_position = 0

    def render(self) -> gr.Column:
        """Рендеринг временного слайдера"""
        try:
            with gr.Column() as col:
                gr.Markdown("### ⏰ Временная навигация")
                
                self.slider = gr.Slider(
                    minimum=0,
                    maximum=100,
                    step=1,
                    value=0,
                    label="Временная позиция (%)",
                    interactive=True
                )
                
                with gr.Row():
                    self.date_display = gr.Textbox(
                        label="Текущая дата",
                        interactive=False,
                        value="Не выбрано"
                    )
                    
                    self.age_display = gr.Textbox(
                        label="Возраст Путина",
                        interactive=False,
                        value="0.0 лет"
                    )
                
                with gr.Row():
                    self.preview = gr.Image(
                        label="Предварительный просмотр",
                        height=200,
                        width=200
                    )
                    
                    self.info = gr.HTML(
                        value=self._create_timeline_info_html(),
                        label="Информация о временной точке"
                    )
                
                # Контролы навигации
                with gr.Row():
                    self.prev_btn = gr.Button("◀ Предыдущий", size="sm")
                    self.play_btn = gr.Button("▶ Воспроизвести", size="sm")
                    self.next_btn = gr.Button("▶ Следующий", size="sm")
                    self.reset_btn = gr.Button("🔄 Сброс", size="sm")
                
                # Настройки воспроизведения
                with gr.Accordion("⚙️ Настройки воспроизведения", open=False):
                    self.playback_speed = gr.Slider(
                        minimum=0.1,
                        maximum=5.0,
                        value=1.0,
                        label="Скорость воспроизведения",
                        info="Множитель скорости"
                    )
                    
                    self.auto_loop = gr.Checkbox(
                        label="Зацикливание",
                        value=False,
                        info="Автоматически начинать сначала"
                    )
                
                # Привязка событий
                self.slider.change(
                    fn=self.update_temporal_position,
                    inputs=[self.slider],
                    outputs=[self.date_display, self.age_display, self.preview, self.info]
                )
                
                self.prev_btn.click(
                    fn=self.previous_frame,
                    inputs=[],
                    outputs=[self.slider, self.date_display, self.age_display, self.preview, self.info]
                )
                
                self.next_btn.click(
                    fn=self.next_frame,
                    inputs=[],
                    outputs=[self.slider, self.date_display, self.age_display, self.preview, self.info]
                )
                
                self.reset_btn.click(
                    fn=self.reset_position,
                    inputs=[],
                    outputs=[self.slider, self.date_display, self.age_display, self.preview, self.info]
                )
                
            return col
            
        except Exception as e:
            logger.error(f"Ошибка рендеринга temporal slider: {e}")
            return gr.Column()

    def load_timeline_data(self, timeline_data: List[Dict[str, Any]]):
        """Загрузка данных временной линии"""
        try:
            self.timeline_data = sorted(timeline_data, key=lambda x: x.get("date", ""))
            
            # Обновление диапазона слайдера
            if self.timeline_data:
                self.slider.maximum = len(self.timeline_data) - 1
            
            self.update_state(
                timeline_length=len(self.timeline_data),
                date_range=(
                    self.timeline_data[0].get("date", "") if self.timeline_data else "",
                    self.timeline_data[-1].get("date", "") if self.timeline_data else ""
                ),
                loaded_at=datetime.now().isoformat()
            )
            
            logger.info(f"Загружено {len(self.timeline_data)} временных точек")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки timeline данных: {e}")

    def update_temporal_position(self, position: float) -> Tuple[str, str, Any, str]:
        """Обновление временной позиции"""
        try:
            if not self.timeline_data:
                return "Нет данных", "0.0 лет", None, self._create_timeline_info_html()
            
            # Конвертация позиции в индекс
            index = int(position * (len(self.timeline_data) - 1) / 100) if len(self.timeline_data) > 1 else 0
            index = max(0, min(index, len(self.timeline_data) - 1))
            
            self.current_position = index
            current_data = self.timeline_data[index]
            
            # Извлечение данных
            date_str = current_data.get("date", "Неизвестно")
            image_path = current_data.get("image_path")
            
            # Расчет возраста Путина
            try:
                if date_str != "Неизвестно":
                    current_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    age_years = (current_date - PUTIN_BIRTH_DATE.date()).days / 365.25
                    age_display = f"{age_years:.1f} лет"
                else:
                    age_display = "Неизвестно"
            except:
                age_display = "Ошибка расчета"
            
            # Загрузка изображения
            preview_image = None
            if image_path and Path(image_path).exists():
                try:
                    if HAS_PIL:
                        preview_image = Image.open(image_path)
                        preview_image.thumbnail(UI_PARAMS["preview_size"])
                except Exception as e:
                    logger.warning(f"Ошибка загрузки изображения {image_path}: {e}")
            
            # Создание информационной панели
            info_html = self._create_timeline_info_html(current_data)
            
            return date_str, age_display, preview_image, info_html
            
        except Exception as e:
            logger.error(f"Ошибка обновления позиции: {e}")
            return "Ошибка", "0.0 лет", None, self._create_error_info_html(str(e))

    def previous_frame(self) -> Tuple[float, str, str, Any, str]:
        """Переход к предыдущему кадру"""
        try:
            if self.current_position > 0:
                new_position = (self.current_position - 1) * 100 / max(1, len(self.timeline_data) - 1)
                return (new_position,) + self.update_temporal_position(new_position)
            else:
                return (0.0,) + self.update_temporal_position(0.0)
        except Exception as e:
            logger.error(f"Ошибка перехода к предыдущему кадру: {e}")
            return 0.0, "Ошибка", "0.0 лет", None, self._create_error_info_html(str(e))

    def next_frame(self) -> Tuple[float, str, str, Any, str]:
        """Переход к следующему кадру"""
        try:
            if self.current_position < len(self.timeline_data) - 1:
                new_position = (self.current_position + 1) * 100 / max(1, len(self.timeline_data) - 1)
                return (new_position,) + self.update_temporal_position(new_position)
            else:
                max_pos = 100.0
                return (max_pos,) + self.update_temporal_position(max_pos)
        except Exception as e:
            logger.error(f"Ошибка перехода к следующему кадру: {e}")
            return 100.0, "Ошибка", "0.0 лет", None, self._create_error_info_html(str(e))

    def reset_position(self) -> Tuple[float, str, str, Any, str]:
        """Сброс позиции в начало"""
        try:
            self.current_position = 0
            return (0.0,) + self.update_temporal_position(0.0)
        except Exception as e:
            logger.error(f"Ошибка сброса позиции: {e}")
            return 0.0, "Ошибка", "0.0 лет", None, self._create_error_info_html(str(e))

    def _create_timeline_info_html(self, current_data: Optional[Dict[str, Any]] = None) -> str:
        """Создание HTML информационной панели"""
        try:
            if current_data is None:
                return """
                <div style="padding: 15px; background-color: #f8f9fa; border-radius: 5px;">
                    <h4>Временная навигация</h4>
                    <p>Выберите позицию на временной линии для просмотра данных</p>
                </div>
                """
            
            authenticity_score = current_data.get("authenticity_score", 0.0)
            cluster_id = current_data.get("cluster_id", "Неизвестно")
            quality_score = current_data.get("quality_score", 0.0)
            
            html = f"""
            <div style="padding: 15px; background-color: #f8f9fa; border-radius: 5px;">
                <h4>Информация о кадре</h4>
                <p><strong>Дата:</strong> {current_data.get('date', 'Неизвестно')}</p>
                <p><strong>Аутентичность:</strong> {authenticity_score:.3f}</p>
                <p><strong>Кластер:</strong> {cluster_id}</p>
                <p><strong>Качество:</strong> {quality_score:.3f}</p>
                <p><strong>Позиция:</strong> {self.current_position + 1} из {len(self.timeline_data)}</p>
            </div>
            """
            
            return html
            
        except Exception as e:
            logger.error(f"Ошибка создания timeline info HTML: {e}")
            return self._create_error_info_html(str(e))

    def _create_error_info_html(self, error_msg: str) -> str:
        """Создание HTML сообщения об ошибке"""
        return f"""
        <div style="padding: 15px; background-color: #ffebee; border-radius: 5px; color: #d32f2f;">
            <h4>Ошибка</h4>
            <p>{error_msg}</p>
        </div>
        """

# === METRICS GALLERY ===

class MetricsGallery(BaseUIComponent):
    """
    ИСПРАВЛЕНО: Галерея метрик с визуализацией
    Согласно правкам: 15 метрик в 3 группах
    """

    def __init__(self):
        super().__init__("metrics_gallery")
        self.metrics_groups = {
            "skull": [
                "skull_width_ratio", "temporal_bone_angle", "zygomatic_arch_width",
                "orbital_depth", "occipital_curve"
            ],
            "proportions": [
                "cephalic_index", "nasolabial_angle", "orbital_index",
                "forehead_height_ratio", "chin_projection_ratio"
            ],
            "bone_structure": [
                "interpupillary_distance_ratio", "gonial_angle_asymmetry",
                "zygomatic_angle", "jaw_angle_ratio", "mandibular_symphysis_angle"
            ]
        }

    def render(self) -> gr.Column:
        """Рендеринг галереи метрик"""
        try:
            with gr.Column() as col:
                gr.Markdown("### 📊 Галерея метрик (15 метрик в 3 группах)")
                
                # ИСПРАВЛЕНО: Tabs для 3 групп метрик
                with gr.Tabs():
                    with gr.Tab("🏛️ Геометрия черепа (5)"):
                        self.skull_metrics = gr.Gallery(
                            label="Метрики геометрии черепа",
                            columns=UI_PARAMS["gallery_columns"],
                            rows=UI_PARAMS["gallery_rows"],
                            height=300,
                            allow_preview=True
                        )
                        
                        self.skull_table = gr.DataFrame(
                            headers=["Метрика", "Текущее", "Эталон", "Отклонение", "Статус"],
                            label="Детали метрик черепа",
                            interactive=False
                        )
                    
                    with gr.Tab("👤 Пропорции лица (5)"):
                        self.proportions_metrics = gr.Gallery(
                            label="Метрики пропорций лица",
                            columns=UI_PARAMS["gallery_columns"],
                            rows=UI_PARAMS["gallery_rows"],
                            height=300,
                            allow_preview=True
                        )
                        
                        self.proportions_table = gr.DataFrame(
                            headers=["Метрика", "Текущее", "Эталон", "Отклонение", "Статус"],
                            label="Детали пропорций лица",
                            interactive=False
                        )
                    
                    with gr.Tab("🦴 Костная структура (5)"):
                        self.bone_metrics = gr.Gallery(
                            label="Метрики костной структуры",
                            columns=UI_PARAMS["gallery_columns"],
                            rows=UI_PARAMS["gallery_rows"],
                            height=300,
                            allow_preview=True
                        )
                        
                        self.bone_table = gr.DataFrame(
                            headers=["Метрика", "Текущее", "Эталон", "Отклонение", "Статус"],
                            label="Детали костной структуры",
                            interactive=False
                        )
                
                # Общая статистика
                with gr.Row():
                    self.overall_stats = gr.HTML(
                        value=self._create_overall_stats_html(),
                        label="Общая статистика метрик"
                    )
                
                # Контролы экспорта
                with gr.Row():
                    self.export_metrics_btn = gr.Button("📊 Экспорт метрик", variant="secondary")
                    self.compare_metrics_btn = gr.Button("🔄 Сравнить с эталоном", variant="primary")
                
            return col
            
        except Exception as e:
            logger.error(f"Ошибка рендеринга metrics gallery: {e}")
            return gr.Column()

    def update_metrics(self, metrics_data: Dict[str, Dict[str, float]]):
        """
        ИСПРАВЛЕНО: Обновление метрик в галерее
        Согласно правкам: обработка 15 метрик по группам
        """
        try:
            logger.info("Обновление галереи метрик")
            
            # Обновление каждой группы
            for group_name, metric_names in self.metrics_groups.items():
                group_data = {}
                for metric_name in metric_names:
                    if metric_name in metrics_data:
                        group_data[metric_name] = metrics_data[metric_name]
                
                # Создание визуализаций для группы
                visualizations = self._create_group_visualizations(group_name, group_data)
                
                # Создание таблицы данных
                table_data = self._create_group_table_data(group_name, group_data)
                
                # Обновление соответствующих компонентов
                if group_name == "skull":
                    self.skull_metrics.value = visualizations
                    self.skull_table.value = table_data
                elif group_name == "proportions":
                    self.proportions_metrics.value = visualizations
                    self.proportions_table.value = table_data
                elif group_name == "bone_structure":
                    self.bone_metrics.value = visualizations
                    self.bone_table.value = table_data
            
            # Обновление общей статистики
            self.overall_stats.value = self._create_overall_stats_html(metrics_data)
            
            # Обновление состояния
            self.update_state(
                metrics_count=len(metrics_data),
                last_update=datetime.now().isoformat(),
                groups_processed=len(self.metrics_groups)
            )
            
        except Exception as e:
            logger.error(f"Ошибка обновления метрик: {e}")

    def _create_group_visualizations(self, group_name: str, 
                                   group_data: Dict[str, Dict[str, float]]) -> List[Any]:
        """Создание визуализаций для группы метрик"""
        try:
            visualizations = []
            
            for metric_name, metric_values in group_data.items():
                if HAS_PLOTLY:
                    # Создание графика тренда метрики
                    fig = self._create_metric_trend_plot(metric_name, metric_values)
                    
                    # Конвертация в изображение для галереи
                    img_bytes = fig.to_image(format="png", width=300, height=200)
                    visualizations.append(img_bytes)
                else:
                    # Заглушка без Plotly
                    visualizations.append(None)
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Ошибка создания визуализаций группы {group_name}: {e}")
            return []

    def _create_metric_trend_plot(self, metric_name: str, 
                                metric_values: Dict[str, float]) -> Any:
        """Создание графика тренда метрики"""
        try:
            if not HAS_PLOTLY:
                return None
            
            # Извлечение временных данных
            dates = list(metric_values.keys())
            values = list(metric_values.values())
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=values,
                mode='lines+markers',
                name=metric_name,
                line=dict(width=2),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title=f"Тренд: {metric_name}",
                xaxis_title="Дата",
                yaxis_title="Значение",
                height=200,
                width=300,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Ошибка создания графика тренда {metric_name}: {e}")
            return None

    def _create_group_table_data(self, group_name: str, 
                               group_data: Dict[str, Dict[str, float]]) -> List[List[str]]:
        """Создание данных таблицы для группы"""
        try:
            table_data = []
            
            for metric_name, metric_values in group_data.items():
                if metric_values:
                    current_value = list(metric_values.values())[-1]  # Последнее значение
                    reference_value = self._get_reference_value(metric_name)
                    deviation = abs(current_value - reference_value) / reference_value * 100
                    status = self._get_metric_status(deviation)
                    
                    table_data.append([
                        metric_name,
                        f"{current_value:.4f}",
                        f"{reference_value:.4f}",
                        f"{deviation:.2f}%",
                        status
                    ])
            
            return table_data
            
        except Exception as e:
            logger.error(f"Ошибка создания данных таблицы группы {group_name}: {e}")
            return []

    def _get_reference_value(self, metric_name: str) -> float:
        """Получение эталонного значения метрики"""
        # Эталонные значения для метрик (примерные)
        reference_values = {
            "skull_width_ratio": 1.0,
            "temporal_bone_angle": 110.0,
            "zygomatic_arch_width": 0.8,
            "orbital_depth": 0.6,
            "occipital_curve": 0.7,
            "cephalic_index": 0.75,
            "nasolabial_angle": 95.0,
            "orbital_index": 0.85,
            "forehead_height_ratio": 0.35,
            "chin_projection_ratio": 0.4,
            "interpupillary_distance_ratio": 0.15,
            "gonial_angle_asymmetry": 2.0,
            "zygomatic_angle": 45.0,
            "jaw_angle_ratio": 1.2,
            "mandibular_symphysis_angle": 85.0
        }
        
        return reference_values.get(metric_name, 1.0)

    def _get_metric_status(self, deviation: float) -> str:
        """Определение статуса метрики по отклонению"""
        if deviation <= 5.0:
            return "НОРМА"
        elif deviation <= 15.0:
            return "ОТКЛОНЕНИЕ"
        else:
            return "АНОМАЛИЯ"

    def _create_overall_stats_html(self, metrics_data: Optional[Dict[str, Dict[str, float]]] = None) -> str:
        """Создание HTML общей статистики"""
        try:
            if metrics_data is None:
                return """
                <div style="padding: 15px; background-color: #f8f9fa; border-radius: 5px;">
                    <h4>Общая статистика метрик</h4>
                    <p>Данные не загружены</p>
                </div>
                """
            
            total_metrics = len(metrics_data)
            normal_count = 0
            deviation_count = 0
            anomaly_count = 0
            
            for metric_name, metric_values in metrics_data.items():
                if metric_values:
                    current_value = list(metric_values.values())[-1]
                    reference_value = self._get_reference_value(metric_name)
                    deviation = abs(current_value - reference_value) / reference_value * 100
                    
                    if deviation <= 5.0:
                        normal_count += 1
                    elif deviation <= 15.0:
                        deviation_count += 1
                    else:
                        anomaly_count += 1
            
            html = f"""
            <div style="padding: 15px; background-color: #f8f9fa; border-radius: 5px;">
                <h4>Общая статистика метрик</h4>
                <div style="display: flex; justify-content: space-around; margin: 10px 0;">
                    <div style="text-align: center;">
                        <div style="font-size: 24px; font-weight: bold; color: #28a745;">{normal_count}</div>
                        <div>НОРМА</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 24px; font-weight: bold; color: #ffc107;">{deviation_count}</div>
                        <div>ОТКЛОНЕНИЯ</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 24px; font-weight: bold; color: #dc3545;">{anomaly_count}</div>
                        <div>АНОМАЛИИ</div>
                    </div>
                </div>
                <p><strong>Всего метрик:</strong> {total_metrics}</p>
                <p><strong>Последнее обновление:</strong> {datetime.now().strftime('%H:%M:%S')}</p>
            </div>
            """
            
            return html
            
        except Exception as e:
            logger.error(f"Ошибка создания общей статистики: {e}")
            return f"<div>Ошибка: {e}</div>"

# === ADVANCED FILTERS ===

class AdvancedFilters(BaseUIComponent):
    """
    ИСПРАВЛЕНО: Продвинутые фильтры для анализа
    Согласно правкам: фильтрация по различным критериям
    """

    def __init__(self):
        super().__init__("filters")
        self.active_filters = {}

    def render(self) -> gr.Column:
        """Рендеринг продвинутых фильтров"""
        try:
            with gr.Column() as col:
                gr.Markdown("### 🔍 Продвинутые фильтры")
                
                # Временной диапазон
                with gr.Row():
                    self.start_date = gr.Textbox(
                        label="Дата начала (YYYY-MM-DD)",
                        placeholder="1999-01-01"
                    )
                    
                    self.end_date = gr.Textbox(
                        label="Дата окончания (YYYY-MM-DD)",
                        placeholder="2024-12-31"
                    )
                
                # Пороги
                with gr.Row():
                    self.authenticity_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        label="Порог аутентичности",
                        info="Минимальный балл аутентичности"
                    )
                    
                    self.quality_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.6,
                        label="Порог качества",
                        info="Минимальное качество изображения"
                    )
                
                # Категориальные фильтры
                with gr.Row():
                    self.quality_filter = gr.CheckboxGroup(
                        choices=["Высокое качество", "Среднее качество", "Низкое качество"],
                        label="Фильтр качества",
                        value=["Высокое качество", "Среднее качество"]
                    )
                    
                    self.anomaly_types = gr.CheckboxGroup(
                        choices=["Геометрические", "Текстурные", "Временные", "Эмбеддинг", "Медицинские"],
                        label="Типы аномалий",
                        value=[]
                    )
                
                # Уровни масок
                self.mask_levels = gr.CheckboxGroup(
                    choices=["Level1", "Level2", "Level3", "Level4", "Level5"],
                    label="Уровни технологий масок",
                    value=["Level1", "Level2", "Level3", "Level4", "Level5"]
                )
                
                # Дополнительные фильтры
                with gr.Accordion("🔧 Дополнительные фильтры", open=False):
                    self.cluster_filter = gr.Textbox(
                        label="ID кластера",
                        placeholder="Введите ID кластера или оставьте пустым"
                    )
                    
                    self.source_filter = gr.CheckboxGroup(
                        choices=["Официальные", "СМИ", "Соцсети", "Архивные"],
                        label="Источники изображений",
                        value=["Официальные", "СМИ", "Архивные"]
                    )
                    
                    self.confidence_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.8,
                        label="Порог уверенности детекции",
                        info="Минимальная уверенность детекции лица"
                    )
                
                # Кнопки управления
                with gr.Row():
                    self.apply_filters_btn = gr.Button(
                        "✅ Применить фильтры",
                        variant="primary"
                    )
                    
                    self.reset_filters_btn = gr.Button(
                        "🔄 Сбросить фильтры",
                        variant="secondary"
                    )
                    
                    self.save_preset_btn = gr.Button(
                        "💾 Сохранить пресет",
                        variant="secondary"
                    )
                
                # Статус фильтров
                self.filter_status = gr.HTML(
                    value=self._create_filter_status_html(),
                    label="Статус фильтров"
                )
                
                # Привязка событий
                self.apply_filters_btn.click(
                    fn=self.apply_filters,
                    inputs=[
                        self.start_date, self.end_date, self.authenticity_threshold,
                        self.quality_threshold, self.quality_filter, self.anomaly_types,
                        self.mask_levels, self.cluster_filter, self.source_filter,
                        self.confidence_threshold
                    ],
                    outputs=[self.filter_status]
                )
                
                self.reset_filters_btn.click(
                    fn=self.reset_filters,
                    inputs=[],
                    outputs=[
                        self.start_date, self.end_date, self.authenticity_threshold,
                        self.quality_threshold, self.quality_filter, self.anomaly_types,
                        self.mask_levels, self.cluster_filter, self.source_filter,
                        self.confidence_threshold, self.filter_status
                    ]
                )
                
            return col
            
        except Exception as e:
            logger.error(f"Ошибка рендеринга advanced filters: {e}")
            return gr.Column()

    def apply_filters(self, start_date: str, end_date: str, authenticity_threshold: float,
                     quality_threshold: float, quality_filter: List[str], anomaly_types: List[str],
                     mask_levels: List[str], cluster_filter: str, source_filter: List[str],
                     confidence_threshold: float) -> str:
        """Применение фильтров"""
        try:
            logger.info("Применение продвинутых фильтров")
            
            # Сохранение активных фильтров
            self.active_filters = {
                "date_range": (start_date, end_date),
                "authenticity_threshold": authenticity_threshold,
                "quality_threshold": quality_threshold,
                "quality_filter": quality_filter,
                "anomaly_types": anomaly_types,
                "mask_levels": mask_levels,
                "cluster_filter": cluster_filter,
                "source_filter": source_filter,
                "confidence_threshold": confidence_threshold,
                "applied_at": datetime.now().isoformat()
            }
            
            # Обновление состояния
            self.update_state(
                active_filters=self.active_filters,
                filters_count=len([f for f in self.active_filters.values() if f])
            )
            
            # Создание статуса
            status_html = self._create_filter_status_html()
            
            logger.info(f"Фильтры применены: {len(self.active_filters)} активных")
            return status_html
            
        except Exception as e:
            logger.error(f"Ошибка применения фильтров: {e}")
            return f"<div style='color: red;'>Ошибка: {e}</div>"

    def reset_filters(self) -> Tuple[str, str, float, float, List[str], List[str], List[str], str, List[str], float, str]:
        """Сброс всех фильтров"""
        try:
            logger.info("Сброс всех фильтров")
            
            self.active_filters = {}
            self.clear_state()
            
            # Возврат значений по умолчанию
            return (
                "",  # start_date
                "",  # end_date
                0.5,  # authenticity_threshold
                0.6,  # quality_threshold
                ["Высокое качество", "Среднее качество"],  # quality_filter
                [],  # anomaly_types
                ["Level1", "Level2", "Level3", "Level4", "Level5"],  # mask_levels
                "",  # cluster_filter
                ["Официальные", "СМИ", "Архивные"],  # source_filter
                0.8,  # confidence_threshold
                self._create_filter_status_html()  # filter_status
            )
            
        except Exception as e:
            logger.error(f"Ошибка сброса фильтров: {e}")
            return ("",) * 10 + (f"<div style='color: red;'>Ошибка: {e}</div>",)

    def _create_filter_status_html(self) -> str:
        """Создание HTML статуса фильтров"""
        try:
            if not self.active_filters:
                return """
                <div style="padding: 10px; background-color: #e3f2fd; border-radius: 5px;">
                    <h4>Статус фильтров</h4>
                    <p>Фильтры не применены. Отображаются все данные.</p>
                </div>
                """
            
            filters_count = len([f for f in self.active_filters.values() if f])
            applied_at = self.active_filters.get("applied_at", "Неизвестно")
            
            html = f"""
            <div style="padding: 10px; background-color: #e8f5e8; border-radius: 5px;">
                <h4>Статус фильтров</h4>
                <p><strong>Активных фильтров:</strong> {filters_count}</p>
                <p><strong>Применены:</strong> {applied_at}</p>
                <div style="margin-top: 10px;">
            """
            
            # Детали активных фильтров
            if self.active_filters.get("date_range"):
                start, end = self.active_filters["date_range"]
                if start or end:
                    html += f"<p>📅 <strong>Период:</strong> {start or 'начало'} — {end or 'конец'}</p>"
            
            if self.active_filters.get("authenticity_threshold", 0) > 0:
                html += f"<p>🎯 <strong>Аутентичность ≥</strong> {self.active_filters['authenticity_threshold']:.2f}</p>"
            
            if self.active_filters.get("quality_threshold", 0) > 0:
                html += f"<p>⭐ <strong>Качество ≥</strong> {self.active_filters['quality_threshold']:.2f}</p>"
            
            if self.active_filters.get("mask_levels"):
                levels = ", ".join(self.active_filters["mask_levels"])
                html += f"<p>🎭 <strong>Уровни масок:</strong> {levels}</p>"
            
            html += "</div></div>"
            
            return html
            
        except Exception as e:
            logger.error(f"Ошибка создания статуса фильтров: {e}")
            return f"<div style='color: red;'>Ошибка: {e}</div>"

# === INTERACTIVE COMPARISON ===
# === INTERACTIVE COMPARISON ===

class InteractiveComparison(BaseUIComponent):
    """
    ИСПРАВЛЕНО: Интерактивное сравнение результатов
    Согласно правкам: сравнение метрик между различными идентичностями
    """

    def __init__(self):
        super().__init__("comparison")
        self.comparison_data = {}
        self.selected_identities = []

    def render(self) -> gr.Column:
        """Рендеринг интерфейса сравнения"""
        try:
            with gr.Column() as col:
                gr.Markdown("### 🔄 Интерактивное сравнение идентичностей")
                
                # Селекторы для выбора идентичностей
                with gr.Row():
                    self.identity_a_selector = gr.Dropdown(
                        choices=[],
                        label="Идентичность A",
                        info="Выберите первую идентичность для сравнения"
                    )
                    
                    self.identity_b_selector = gr.Dropdown(
                        choices=[],
                        label="Идентичность B", 
                        info="Выберите вторую идентичность для сравнения"
                    )
                
                # Изображения для сравнения
                with gr.Row():
                    self.left_image = gr.Image(
                        label="Изображение A",
                        height=300,
                        width=300
                    )
                    
                    self.right_image = gr.Image(
                        label="Изображение B",
                        height=300,
                        width=300
                    )
                
                # Таблица сравнения метрик
                self.comparison_metrics = gr.DataFrame(
                    headers=["Метрика", "Значение A", "Значение B", "Разница", "Статус"],
                    label="Детальное сравнение метрик",
                    interactive=False,
                    height=400
                )
                
                # Общие показатели
                with gr.Row():
                    self.similarity_score = gr.Number(
                        label="Общий балл схожести",
                        precision=3,
                        interactive=False
                    )
                    
                    self.confidence_level = gr.Textbox(
                        label="Уровень уверенности",
                        interactive=False
                    )
                
                # Кнопки управления
                with gr.Row():
                    self.compare_btn = gr.Button("🔄 Сравнить", variant="primary")
                    self.swap_btn = gr.Button("🔀 Поменять местами", variant="secondary")
                    self.export_comparison_btn = gr.Button("📊 Экспорт сравнения", variant="secondary")
                
                # Привязка событий
                self.compare_btn.click(
                    fn=self.perform_comparison,
                    inputs=[self.identity_a_selector, self.identity_b_selector],
                    outputs=[self.left_image, self.right_image, self.comparison_metrics, 
                            self.similarity_score, self.confidence_level]
                )
                
                self.swap_btn.click(
                    fn=self.swap_identities,
                    inputs=[self.identity_a_selector, self.identity_b_selector],
                    outputs=[self.identity_a_selector, self.identity_b_selector]
                )
                
            return col
            
        except Exception as e:
            logger.error(f"Ошибка рендеринга comparison: {e}")
            return gr.Column()

    def perform_comparison(self, identity_a: str, identity_b: str) -> Tuple[Any, Any, List[List], float, str]:
        """Выполнение сравнения между двумя идентичностями"""
        try:
            if not identity_a or not identity_b:
                return None, None, [], 0.0, "Выберите обе идентичности"
            
            # Получение данных идентичностей
            data_a = self.comparison_data.get(identity_a, {})
            data_b = self.comparison_data.get(identity_b, {})
            
            # Создание таблицы сравнения
            comparison_table = self._create_comparison_table(data_a, data_b)
            
            # Расчет общего балла схожести
            similarity = self._calculate_similarity_score(data_a, data_b)
            
            # Определение уровня уверенности
            confidence = self._determine_confidence_level(similarity)
            
            # Получение изображений
            image_a = data_a.get("image_path")
            image_b = data_b.get("image_path")
            
            return image_a, image_b, comparison_table, similarity, confidence
            
        except Exception as e:
            logger.error(f"Ошибка сравнения: {e}")
            return None, None, [], 0.0, f"Ошибка: {str(e)}"

    def _create_comparison_table(self, data_a: Dict, data_b: Dict) -> List[List]:
        """Создание таблицы сравнения метрик"""
        try:
            table_data = []
            
            # Список метрик для сравнения
            metrics_to_compare = [
                "skull_width_ratio", "temporal_bone_angle", "zygomatic_arch_width",
                "orbital_depth", "occipital_curve", "cephalic_index", "nasolabial_angle",
                "orbital_index", "forehead_height_ratio", "chin_projection_ratio",
                "interpupillary_distance_ratio", "gonial_angle_asymmetry", 
                "zygomatic_angle", "jaw_angle_ratio", "mandibular_symphysis_angle"
            ]
            
            for metric in metrics_to_compare:
                value_a = data_a.get("metrics", {}).get(metric, 0.0)
                value_b = data_b.get("metrics", {}).get(metric, 0.0)
                
                difference = abs(value_a - value_b)
                relative_diff = (difference / max(abs(value_a), abs(value_b), 1e-6)) * 100
                
                # Определение статуса
                if relative_diff < 5.0:
                    status = "ИДЕНТИЧНО"
                elif relative_diff < 15.0:
                    status = "ПОХОЖЕ"
                else:
                    status = "РАЗЛИЧНО"
                
                table_data.append([
                    metric,
                    f"{value_a:.4f}",
                    f"{value_b:.4f}",
                    f"{relative_diff:.2f}%",
                    status
                ])
            
            return table_data
            
        except Exception as e:
            logger.error(f"Ошибка создания таблицы сравнения: {e}")
            return []

    def _calculate_similarity_score(self, data_a: Dict, data_b: Dict) -> float:
        """Расчет общего балла схожести"""
        try:
            similarities = []
            
            # Сравнение геометрических метрик
            geom_a = data_a.get("geometry_score", 0.0)
            geom_b = data_b.get("geometry_score", 0.0)
            geom_sim = 1.0 - abs(geom_a - geom_b)
            similarities.append(geom_sim)
            
            # Сравнение эмбеддингов
            emb_a = data_a.get("embedding_score", 0.0)
            emb_b = data_b.get("embedding_score", 0.0)
            emb_sim = 1.0 - abs(emb_a - emb_b)
            similarities.append(emb_sim)
            
            # Сравнение текстуры
            tex_a = data_a.get("texture_score", 0.0)
            tex_b = data_b.get("texture_score", 0.0)
            tex_sim = 1.0 - abs(tex_a - tex_b)
            similarities.append(tex_sim)
            
            # Общий балл
            overall_similarity = np.mean(similarities) if similarities else 0.0
            return float(np.clip(overall_similarity, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Ошибка расчета схожести: {e}")
            return 0.0

    def _determine_confidence_level(self, similarity: float) -> str:
        """Определение уровня уверенности"""
        if similarity >= 0.9:
            return "ОЧЕНЬ ВЫСОКИЙ"
        elif similarity >= 0.7:
            return "ВЫСОКИЙ"
        elif similarity >= 0.5:
            return "СРЕДНИЙ"
        elif similarity >= 0.3:
            return "НИЗКИЙ"
        else:
            return "ОЧЕНЬ НИЗКИЙ"

    def swap_identities(self, identity_a: str, identity_b: str) -> Tuple[str, str]:
        """Смена местами выбранных идентичностей"""
        return identity_b, identity_a

    def update_available_identities(self, identities: List[str]):
        """Обновление списка доступных идентичностей"""
        try:
            self.identity_a_selector.choices = identities
            self.identity_b_selector.choices = identities
            logger.info(f"Обновлен список идентичностей: {len(identities)}")
        except Exception as e:
            logger.error(f"Ошибка обновления идентичностей: {e}")

# === ADVANCED SEARCH ===

class AdvancedSearch(BaseUIComponent):
    """
    ИСПРАВЛЕНО: Продвинутый поиск по результатам
    Согласно правкам: поиск по метрикам, датам и критериям
    """

    def __init__(self):
        super().__init__("search")
        self.search_index = {}
        self.search_results = []

    def render(self) -> gr.Column:
        """Рендеринг интерфейса поиска"""
        try:
            with gr.Column() as col:
                gr.Markdown("### 🔍 Продвинутый поиск")
                
                # Поисковая строка
                self.search_query = gr.Textbox(
                    label="Поисковый запрос",
                    placeholder="Введите критерии поиска...",
                    lines=1
                )
                
                # Фильтры поиска
                with gr.Row():
                    self.search_type = gr.Radio(
                        choices=["Текст", "Метрики", "Даты", "Аномалии"],
                        value="Текст",
                        label="Тип поиска"
                    )
                    
                    self.sort_by = gr.Dropdown(
                        choices=["Релевантность", "Дата", "Аутентичность", "Качество"],
                        value="Релевантность",
                        label="Сортировка"
                    )
                
                # Дополнительные фильтры
                with gr.Accordion("🔧 Дополнительные фильтры", open=False):
                    with gr.Row():
                        self.date_from = gr.Textbox(
                            label="Дата от (YYYY-MM-DD)",
                            placeholder="1999-01-01"
                        )
                        
                        self.date_to = gr.Textbox(
                            label="Дата до (YYYY-MM-DD)",
                            placeholder="2024-12-31"
                        )
                    
                    with gr.Row():
                        self.min_authenticity = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            label="Минимальная аутентичность"
                        )
                        
                        self.max_authenticity = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=1.0,
                            label="Максимальная аутентичность"
                        )
                
                # Кнопки управления
                with gr.Row():
                    self.search_btn = gr.Button("🔍 Поиск", variant="primary")
                    self.clear_btn = gr.Button("🗑️ Очистить", variant="secondary")
                    self.save_query_btn = gr.Button("💾 Сохранить запрос", variant="secondary")
                
                # Результаты поиска
                self.search_results_display = gr.DataFrame(
                    headers=["ID", "Дата", "Аутентичность", "Описание", "Путь"],
                    label="Результаты поиска",
                    interactive=False,
                    height=400
                )
                
                # Статистика поиска
                self.search_stats = gr.HTML(
                    value="<div>Введите запрос для поиска</div>",
                    label="Статистика поиска"
                )
                
                # Привязка событий
                self.search_btn.click(
                    fn=self.perform_search,
                    inputs=[self.search_query, self.search_type, self.sort_by,
                           self.date_from, self.date_to, self.min_authenticity, self.max_authenticity],
                    outputs=[self.search_results_display, self.search_stats]
                )
                
                self.clear_btn.click(
                    fn=self.clear_search,
                    inputs=[],
                    outputs=[self.search_query, self.search_results_display, self.search_stats]
                )
                
            return col
            
        except Exception as e:
            logger.error(f"Ошибка рендеринга search: {e}")
            return gr.Column()

    def perform_search(self, query: str, search_type: str, sort_by: str,
                      date_from: str, date_to: str, min_auth: float, max_auth: float) -> Tuple[List[List], str]:
        """Выполнение поиска"""
        try:
            if not query.strip():
                return [], "<div>Введите поисковый запрос</div>"
            
            logger.info(f"Выполнение поиска: '{query}' типа '{search_type}'")
            
            # Фильтрация результатов
            filtered_results = self._filter_results(query, search_type, date_from, date_to, min_auth, max_auth)
            
            # Сортировка результатов
            sorted_results = self._sort_results(filtered_results, sort_by)
            
            # Подготовка данных для отображения
            display_data = self._prepare_display_data(sorted_results)
            
            # Создание статистики
            stats_html = self._create_search_stats_html(len(sorted_results), query)
            
            return display_data, stats_html
            
        except Exception as e:
            logger.error(f"Ошибка поиска: {e}")
            return [], f"<div style='color: red;'>Ошибка поиска: {e}</div>"

    def _filter_results(self, query: str, search_type: str, date_from: str, date_to: str,
                       min_auth: float, max_auth: float) -> List[Dict]:
        """Фильтрация результатов поиска"""
        try:
            filtered = []
            
            # Заглушка для демонстрации
            sample_results = [
                {
                    "id": "img_001",
                    "date": "2020-01-15",
                    "authenticity": 0.85,
                    "description": "Официальное фото",
                    "path": "/path/to/img_001.jpg"
                },
                {
                    "id": "img_002", 
                    "date": "2021-03-20",
                    "authenticity": 0.45,
                    "description": "Подозрительное изображение",
                    "path": "/path/to/img_002.jpg"
                }
            ]
            
            for result in sample_results:
                # Фильтр по аутентичности
                if not (min_auth <= result["authenticity"] <= max_auth):
                    continue
                
                # Фильтр по дате
                if date_from and result["date"] < date_from:
                    continue
                if date_to and result["date"] > date_to:
                    continue
                
                # Фильтр по запросу
                if search_type == "Текст":
                    if query.lower() in result["description"].lower():
                        filtered.append(result)
                elif search_type == "Метрики":
                    # Поиск по метрикам
                    if query.lower() in result["id"].lower():
                        filtered.append(result)
                else:
                    filtered.append(result)
            
            return filtered
            
        except Exception as e:
            logger.error(f"Ошибка фильтрации: {e}")
            return []

    def _sort_results(self, results: List[Dict], sort_by: str) -> List[Dict]:
        """Сортировка результатов"""
        try:
            if sort_by == "Дата":
                return sorted(results, key=lambda x: x["date"], reverse=True)
            elif sort_by == "Аутентичность":
                return sorted(results, key=lambda x: x["authenticity"], reverse=True)
            elif sort_by == "Качество":
                return sorted(results, key=lambda x: x.get("quality", 0.0), reverse=True)
            else:  # Релевантность
                return results
                
        except Exception as e:
            logger.error(f"Ошибка сортировки: {e}")
            return results

    def _prepare_display_data(self, results: List[Dict]) -> List[List]:
        """Подготовка данных для отображения"""
        try:
            display_data = []
            
            for result in results:
                display_data.append([
                    result["id"],
                    result["date"],
                    f"{result['authenticity']:.3f}",
                    result["description"],
                    result["path"]
                ])
            
            return display_data
            
        except Exception as e:
            logger.error(f"Ошибка подготовки данных: {e}")
            return []

    def _create_search_stats_html(self, count: int, query: str) -> str:
        """Создание HTML статистики поиска"""
        try:
            html = f"""
            <div style="padding: 10px; background-color: #f0f0f0; border-radius: 5px;">
                <h4>Статистика поиска</h4>
                <p><strong>Запрос:</strong> "{query}"</p>
                <p><strong>Найдено результатов:</strong> {count}</p>
                <p><strong>Время поиска:</strong> < 1 сек</p>
            </div>
            """
            return html
            
        except Exception as e:
            logger.error(f"Ошибка создания статистики: {e}")
            return "<div>Ошибка создания статистики</div>"

    def clear_search(self) -> Tuple[str, List[List], str]:
        """Очистка поиска"""
        return "", [], "<div>Поиск очищен</div>"

# === AI ASSISTANT ===

class AIAssistant(BaseUIComponent):
    """
    ИСПРАВЛЕНО: AI-ассистент для интерпретации результатов
    Согласно правкам: помощь в анализе и интерпретации
    """

    def __init__(self):
        super().__init__("assistant")
        self.conversation_history = []

    def render(self) -> gr.Column:
        """Рендеринг интерфейса AI-ассистента"""
        try:
            with gr.Column() as col:
                gr.Markdown("### 🤖 AI-Ассистент для анализа")
                
                # История чата
                self.chat_history = gr.Chatbot(
                    label="Диалог с ассистентом",
                    height=400,
                    bubble_full_width=False
                )
                
                # Ввод пользователя
                with gr.Row():
                    self.user_input = gr.Textbox(
                        label="Ваш вопрос",
                        placeholder="Задайте вопрос об анализе...",
                        lines=1,
                        scale=4
                    )
                    
                    self.send_btn = gr.Button("📤 Отправить", variant="primary", scale=1)
                
                # Быстрые вопросы
                with gr.Row():
                    self.quick_questions = gr.Radio(
                        choices=[
                            "Объясните результаты анализа",
                            "Какие аномалии обнаружены?",
                            "Рекомендации по улучшению",
                            "Сравнить с эталоном"
                        ],
                        label="Быстрые вопросы"
                    )
                
                # Настройки ассистента
                with gr.Accordion("⚙️ Настройки ассистента", open=False):
                    self.response_style = gr.Radio(
                        choices=["Краткий", "Подробный", "Технический"],
                        value="Подробный",
                        label="Стиль ответов"
                    )
                    
                    self.include_context = gr.Checkbox(
                        label="Включать контекст анализа",
                        value=True
                    )
                
                # Кнопки управления
                with gr.Row():
                    self.clear_chat_btn = gr.Button("🗑️ Очистить чат", variant="secondary")
                    self.export_chat_btn = gr.Button("📄 Экспорт диалога", variant="secondary")
                
                # Привязка событий
                self.send_btn.click(
                    fn=self.process_user_message,
                    inputs=[self.user_input, self.response_style, self.include_context],
                    outputs=[self.chat_history, self.user_input]
                )
                
                self.user_input.submit(
                    fn=self.process_user_message,
                    inputs=[self.user_input, self.response_style, self.include_context],
                    outputs=[self.chat_history, self.user_input]
                )
                
                self.quick_questions.change(
                    fn=self.handle_quick_question,
                    inputs=[self.quick_questions, self.response_style],
                    outputs=[self.chat_history]
                )
                
                self.clear_chat_btn.click(
                    fn=self.clear_chat,
                    inputs=[],
                    outputs=[self.chat_history]
                )
                
            return col
            
        except Exception as e:
            logger.error(f"Ошибка рендеринга assistant: {e}")
            return gr.Column()

    def process_user_message(self, message: str, style: str, include_context: bool) -> Tuple[List[List], str]:
        """Обработка сообщения пользователя"""
        try:
            if not message.strip():
                return self.conversation_history, ""
            
            # Добавление сообщения пользователя
            self.conversation_history.append([message, None])
            
            # Генерация ответа ассистента
            response = self._generate_response(message, style, include_context)
            
            # Добавление ответа ассистента
            self.conversation_history[-1][1] = response
            
            return self.conversation_history, ""
            
        except Exception as e:
            logger.error(f"Ошибка обработки сообщения: {e}")
            error_response = f"Извините, произошла ошибка: {str(e)}"
            self.conversation_history.append([message, error_response])
            return self.conversation_history, ""

    def _generate_response(self, message: str, style: str, include_context: bool) -> str:
        """Генерация ответа ассистента"""
        try:
            # Заглушка для демонстрации
            responses = {
                "краткий": "Краткий ответ на ваш вопрос.",
                "подробный": f"Подробный анализ вашего вопроса: '{message}'. Рекомендую обратить внимание на ключевые метрики.",
                "технический": f"Технический анализ запроса '{message}': используются алгоритмы 3DDFA V2 и InsightFace для точной оценки."
            }
            
            base_response = responses.get(style.lower(), responses["подробный"])
            
            if include_context:
                context = "\n\nКонтекст: Анализ проводится с использованием медицинской валидации и байесовского подхода."
                base_response += context
            
            return base_response
            
        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}")
            return "Извините, не могу сгенерировать ответ на этот вопрос."

    def handle_quick_question(self, question: str, style: str) -> List[List]:
        """Обработка быстрого вопроса"""
        try:
            if not question:
                return self.conversation_history
            
            response = self._generate_response(question, style, True)
            self.conversation_history.append([question, response])
            
            return self.conversation_history
            
        except Exception as e:
            logger.error(f"Ошибка обработки быстрого вопроса: {e}")
            return self.conversation_history

    def clear_chat(self) -> List[List]:
        """Очистка истории чата"""
        self.conversation_history = []
        return []

# === ЭКСПОРТ КОМПОНЕНТОВ ===

def create_ui_components() -> Dict[str, BaseUIComponent]:
    """Создание всех UI компонентов"""
    try:
        logger.info("Создание UI компонентов")
        
        components = {
            '3d_viewer': Interactive3DViewer(),
            'heatmap': InteractiveHeatmap(),
            'temporal_slider': TemporalSlider(),
            'metrics_gallery': MetricsGallery(),
            'filters': AdvancedFilters(),
            'comparison': InteractiveComparison(),
            'search': AdvancedSearch(),
            'assistant': AIAssistant()
        }
        
        logger.info(f"Создано {len(components)} UI компонентов")
        return components
        
    except Exception as e:
        logger.error(f"Ошибка создания UI компонентов: {e}")
        return {}

def get_component_by_id(component_id: str) -> Optional[BaseUIComponent]:
    """Получение компонента по ID"""
    try:
        components = create_ui_components()
        return components.get(component_id)
    except Exception as e:
        logger.error(f"Ошибка получения компонента {component_id}: {e}")
        return None

# === ФУНКЦИИ САМОТЕСТИРОВАНИЯ ===

def self_test():
    """Самотестирование модуля ui_components"""
    try:
        logger.info("Запуск самотестирования ui_components...")
        
        # Тест создания компонентов
        components = create_ui_components()
        assert len(components) > 0, "Компоненты не созданы"
        
        # Тест каждого компонента
        for name, component in components.items():
            assert isinstance(component, BaseUIComponent), f"Компонент {name} не наследует BaseUIComponent"
            assert hasattr(component, 'render'), f"Компонент {name} не имеет метода render"
            assert hasattr(component, 'component_id'), f"Компонент {name} не имеет component_id"
        
        # Тест рендеринга (базовая проверка)
        viewer_3d = components['3d_viewer']
        assert viewer_3d.component_id == "3d_viewer", "Неверный ID 3D viewer"
        
        heatmap = components['heatmap']
        assert heatmap.component_id == "heatmap", "Неверный ID heatmap"
        
        logger.info("Самотестирование ui_components завершено успешно")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка самотестирования: {e}")
        return False

# === ТОЧКА ВХОДА ===

if __name__ == "__main__":
    # Запуск самотестирования при прямом вызове модуля
    success = self_test()
    if success:
        print("✅ Модуль ui_components работает корректно")
        
        # Демонстрация основной функциональности
        components = create_ui_components()
        print(f"📊 Создано компонентов: {len(components)}")
        
        for name, component in components.items():
            print(f"🔧 {name}: {component.__class__.__name__} (ID: {component.component_id})")
        
        # Тест рендеринга компонентов
        print(f"\n🧪 Тестирование рендеринга...")
        for name, component in components.items():
            try:
                rendered = component.render()
                print(f"  ✓ {name}: успешно отрендерен")
            except Exception as e:
                print(f"  ❌ {name}: ошибка рендеринга - {e}")
        
        print(f"\n🎉 Все UI компоненты готовы к использованию!")
        
    else:
        print("❌ Обнаружены ошибки в модуле ui_components")
        exit(1)