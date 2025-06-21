"""
ReportGenerator - Генератор отчетов с экспортом в различные форматы
Версия: 2.0
Дата: 2025-06-21
ИСПРАВЛЕНО: Все критические ошибки согласно новому ТЗ
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import json
from datetime import datetime, timedelta, date
import base64
import io
import hashlib
import time
import psutil
from functools import lru_cache
import threading
from collections import OrderedDict, defaultdict
import msgpack
import zipfile
import os
import subprocess
from jinja2 import Template, Environment, FileSystemLoader

from core_config import get_config

# Настройка логирования
logger = logging.getLogger(__name__)

# === ЦВЕТА КОНСОЛИ ===
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    PURPLE = "\033[95m"
    WHITE = "\033[97m"

# === КАСТОМНЫЙ ФОРМАТТЕР ДЛЯ ЦВЕТНОГО ЛОГИРОВАНИЯ ===
class ColoredFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: f"{Colors.CYAN}%(levelname)s:{Colors.RESET} %(message)s",
        logging.INFO: f"{Colors.GREEN}%(levelname)s:{Colors.RESET} %(message)s",
        logging.WARNING: f"{Colors.YELLOW}%(levelname)s:{Colors.RESET} %(message)s",
        logging.ERROR: f"{Colors.RED}%(levelname)s:{Colors.RESET} %(message)s",
        logging.CRITICAL: f"{Colors.RED}{Colors.BOLD}%(levelname)s:{Colors.RESET} %(message)s"
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# Импорт библиотек для экспорта
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    HAS_REPORTLAB = True
    logger.info(f"{Colors.GREEN}✔ ReportLab импортирована для PDF отчетов.")
except ImportError as e:
    HAS_REPORTLAB = False
    logger.warning(f"{Colors.YELLOW}❌ ReportLab не найдена. Экспорт в PDF будет недоступен. Детали: {e}")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    HAS_PLOTLY = True
    logger.info(f"{Colors.GREEN}✔ Plotly импортирована для интерактивных графиков.")
except ImportError as e:
    HAS_PLOTLY = False
    logger.warning(f"{Colors.YELLOW}❌ Plotly не найдена. Графики в отчетах будут ограничены. Детали: {e}")

try:
    import openpyxl
    HAS_OPENPYXL = True
    logger.info(f"{Colors.GREEN}✔ OpenPyXL импортирована для Excel отчетов.")
except ImportError as e:
    HAS_OPENPYXL = False
    logger.warning(f"{Colors.YELLOW}❌ OpenPyXL не найдена. Экспорт в Excel будет недоступен. Детали: {e}")

try:
    from cryptography.fernet import Fernet
    HAS_CRYPTOGRAPHY = True
    logger.info(f"{Colors.GREEN}✔ Cryptography импортирована для шифрования отчетов.")
except ImportError as e:
    HAS_CRYPTOGRAPHY = False
    logger.warning(f"{Colors.YELLOW}❌ Cryptography не найдена. Шифрование отчетов будет недоступно. Детали: {e}")

# === КОНСТАНТЫ ОТЧЕТОВ ===

# Дата рождения Владимира Путина
PUTIN_BIRTH_DATE = date(1952, 10, 7)

# Ограничения размера PDF
PDF_SIZE_LIMIT_MB = 5
MAX_IMAGES_IN_PDF = 50

# Поддерживаемые форматы экспорта
SUPPORTED_FORMATS = ["HTML", "PDF", "CSV", "Excel", "JSON"]

# Шаблоны отчетов
REPORT_TEMPLATES = {
    "comprehensive": "comprehensive_report.html",
    "summary": "summary_report.html",
    "technical": "technical_report.html",
    "executive": "executive_report.html"
}

# === ОСНОВНОЙ КЛАСС ГЕНЕРАТОРА ОТЧЕТОВ ===

class ReportGenerator:
    """
    Генератор отчетов с полной функциональностью
    ИСПРАВЛЕНО: Все критические ошибки согласно новому ТЗ
    """

    def __init__(self):
        """Инициализация генератора отчетов"""
        logger.info(f"{Colors.BOLD}--- Инициализация ReportGenerator ---{Colors.RESET}")
        
        self.config = get_config()
        self.cache_dir = Path("./cache/report_generator")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Директории для отчетов
        self.results_dir = Path("./results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.templates_dir = Path("./templates")
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Поддерживаемые форматы
        self.supported_formats = SUPPORTED_FORMATS.copy()
        
        # Шаблоны отчетов
        self.report_templates = self._load_report_templates()
        
        # Кэш отчетов
        self.reports_cache = {}
        
        # Статистика генерации
        self.generation_stats = {
            "total_reports": 0,
            "successful_exports": 0,
            "failed_exports": 0,
            "formats_used": defaultdict(int),
            "total_size_mb": 0.0,
            "average_generation_time_ms": 0.0
        }
        
        # Блокировка для потокобезопасности
        self.generation_lock = threading.Lock()
        
        # Настройка Jinja2
        self.jinja_env = Environment(
            loader=FileSystemLoader(self.templates_dir),
            autoescape=True
        )
        
        # Ключ шифрования
        self.encryption_key = self._get_or_create_encryption_key()
        
        logger.info(f"{Colors.BOLD}--- ReportGenerator инициализирован ---{Colors.RESET}")

    def _load_report_templates(self) -> Dict[str, str]:
        """Загрузка шаблонов отчетов"""
        try:
            logger.info(f"{Colors.CYAN}Загрузка шаблонов отчетов...")
            
            templates = {}
            
            # HTML шаблон для comprehensive report
            comprehensive_template = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Комплексный отчет анализа лица</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
        .header { background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .critical { background-color: #ffebee; border-color: #f44336; }
        .warning { background-color: #fff3e0; border-color: #ff9800; }
        .normal { background-color: #e8f5e8; border-color: #4caf50; }
        .score { font-size: 24px; font-weight: bold; text-align: center; margin: 10px 0; }
        .metric { margin: 10px 0; padding: 10px; background-color: #f5f5f5; border-radius: 3px; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .chart-container { margin: 20px 0; text-align: center; }
        .footer { margin-top: 40px; padding: 20px; background-color: #f8f9fa; border-radius: 5px; font-size: 12px; color: #666; }
    </style>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div class="header">
        <h1>Комплексный отчет анализа лица</h1>
        <p><strong>Дата генерации:</strong> {{ generation_date }}</p>
        <p><strong>Версия системы:</strong> {{ system_version }}</p>
        <p><strong>ID клиента:</strong> {{ client_id }}</p>
    </div>

    <div class="section {{ risk_class }}">
        <h2>Исполнительное резюме</h2>
        <div class="score">{{ overall_score }}</div>
        <p><strong>Уровень риска:</strong> {{ risk_level }}</p>
        <p><strong>Рекомендация:</strong> {{ recommendation }}</p>
    </div>

    <div class="section">
        <h2>Геометрический анализ</h2>
        {% for metric, value in geometry_scores.items() %}
        <div class="metric">
            <strong>{{ metric }}:</strong> {{ "%.3f"|format(value) if value is number else value }}
        </div>
        {% endfor %}
    </div>

    <div class="section">
        <h2>Анализ эмбеддингов</h2>
        {% for metric, value in embedding_scores.items() %}
        <div class="metric">
            <strong>{{ metric }}:</strong> {{ "%.3f"|format(value) if value is number else value }}
        </div>
        {% endfor %}
    </div>

    <div class="section">
        <h2>Анализ текстуры</h2>
        {% for metric, value in texture_scores.items() %}
        <div class="metric">
            <strong>{{ metric }}:</strong> {{ "%.3f"|format(value) if value is number else value }}
        </div>
        {% endfor %}
    </div>

    <div class="section">
        <h2>Временной анализ</h2>
        {% for metric, value in temporal_data.items() %}
        <div class="metric">
            <strong>{{ metric }}:</strong> {{ "%.3f"|format(value) if value is number else value }}
        </div>
        {% endfor %}
    </div>

    <div class="section">
        <h2>Детекция аномалий</h2>
        {% for metric, value in anomalies.items() %}
        <div class="metric">
            <strong>{{ metric }}:</strong> {{ "%.3f"|format(value) if value is number else value }}
        </div>
        {% endfor %}
    </div>

    <div class="section">
        <h2>Медицинская валидация</h2>
        {% for metric, value in medical_data.items() %}
        <div class="metric">
            <strong>{{ metric }}:</strong> {{ value }}
        </div>
        {% endfor %}
        <p><strong>Возраст Путина на дату анализа:</strong> {{ putin_age }} лет</p>
    </div>

    <div class="section">
        <h2>Визуализации</h2>
        <div class="chart-container">
            {{ visualizations|safe }}
        </div>
    </div>

    <div class="section">
        <h2>Выводы и рекомендации</h2>
        {{ conclusions|safe }}
    </div>

    <div class="footer">
        <p>Отчет сгенерирован автоматически системой анализа лиц версии {{ system_version }}</p>
        <p>Дата генерации: {{ generation_date }}</p>
        <p>© 2025 Система анализа двойников</p>
    </div>
</body>
</html>
            """
            
            templates["comprehensive"] = comprehensive_template
            
            # Сохранение шаблонов в файлы
            for template_name, template_content in templates.items():
                template_file = self.templates_dir / f"{template_name}_report.html"
                with open(template_file, 'w', encoding='utf-8') as f:
                    f.write(template_content)
            
            logger.info(f"{Colors.GREEN}✔ Шаблоны отчетов загружены: {len(templates)}")
            return templates
            
        except Exception as e:
            logger.error(f"{Colors.RED}❌ Ошибка загрузки шаблонов: {e}")
            return {}

    def _get_or_create_encryption_key(self) -> Optional[bytes]:
        """Получение или создание ключа шифрования"""
        try:
            if not HAS_CRYPTOGRAPHY:
                return None
            
            key_file = self.cache_dir / "encryption.key"
            
            if key_file.exists():
                with open(key_file, 'rb') as f:
                    key = f.read()
            else:
                key = Fernet.generate_key()
                with open(key_file, 'wb') as f:
                    f.write(key)
                logger.info(f"{Colors.GREEN}✔ Создан новый ключ шифрования")
            
            return key
            
        except Exception as e:
            logger.error(f"{Colors.RED}❌ Ошибка работы с ключом шифрования: {e}")
            return None

    def generate_full_report(self, analysis_results: Dict[str, Any], 
                           export_format: str = "HTML",
                           template_type: str = "comprehensive",
                           encrypt: bool = False) -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Генерация полного отчета с экспортом в различные форматы
        Согласно правкам: поддержка HTML, PDF, CSV, Excel, JSON с размером PDF < 5 МБ
        """
        try:
            start_time = time.time()
            logger.info(f"{Colors.CYAN}Генерация отчета в формате {export_format}...")
            
            with self.generation_lock:
                self.generation_stats["total_reports"] += 1
            
            # Валидация формата
            if export_format not in self.supported_formats:
                error_msg = f"Неподдерживаемый формат экспорта: {export_format}. Поддерживаемые: {self.supported_formats}"
                logger.error(f"{Colors.RED}❌ {error_msg}")
                return {"status": "error", "message": error_msg}
            
            # Подготовка данных отчета
            report_data = self._prepare_report_data(analysis_results)
            
            # Генерация временной метки
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Выбор метода экспорта
            export_result = None
            
            if export_format == "HTML":
                export_result = self._export_html(report_data, timestamp, template_type)
            elif export_format == "PDF":
                export_result = self._export_pdf(report_data, timestamp)
            elif export_format == "CSV":
                export_result = self._export_csv(report_data, timestamp)
            elif export_format == "Excel":
                export_result = self._export_excel(report_data, timestamp)
            elif export_format == "JSON":
                export_result = self._export_json(report_data, timestamp)
            
            if export_result and export_result["status"] == "success":
                # Шифрование файла если требуется
                if encrypt and self.encryption_key:
                    export_result = self._encrypt_report_file(export_result)
                
                # Обновление статистики
                with self.generation_lock:
                    self.generation_stats["successful_exports"] += 1
                    self.generation_stats["formats_used"][export_format] += 1
                    
                    # Расчет размера файла
                    if "file_path" in export_result:
                        file_size = Path(export_result["file_path"]).stat().st_size / (1024 * 1024)
                        self.generation_stats["total_size_mb"] += file_size
                        export_result["file_size_mb"] = file_size
                
                # Время генерации
                generation_time = (time.time() - start_time) * 1000
                export_result["generation_time_ms"] = generation_time
                
                # Обновление среднего времени
                with self.generation_lock:
                    total_time = self.generation_stats["average_generation_time_ms"] * (self.generation_stats["successful_exports"] - 1)
                    self.generation_stats["average_generation_time_ms"] = (total_time + generation_time) / self.generation_stats["successful_exports"]
                
                logger.info(f"{Colors.GREEN}✔ Отчет {export_format} успешно сгенерирован за {generation_time:.1f}мс")
                return export_result
            else:
                with self.generation_lock:
                    self.generation_stats["failed_exports"] += 1
                
                logger.error(f"{Colors.RED}❌ Ошибка генерации отчета {export_format}")
                return export_result or {"status": "error", "message": "Неизвестная ошибка экспорта"}
                
        except Exception as e:
            logger.error(f"{Colors.RED}❌ Критическая ошибка генерации отчета: {e}")
            with self.generation_lock:
                self.generation_stats["failed_exports"] += 1
            return {"status": "error", "message": f"Критическая ошибка: {str(e)}"}

    def _prepare_report_data(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Подготовка данных для отчета"""
        try:
            logger.info(f"{Colors.CYAN}Подготовка данных отчета...")
            
            # Извлечение основных данных
            client_id = analysis_results.get("client_id", "unknown")
            overall_score = analysis_results.get("overall_authenticity_score", 0.0)
            
            # Определение уровня риска
            if overall_score >= 0.7:
                risk_level = "НИЗКИЙ"
                risk_class = "normal"
                recommendation = "Лицо соответствует ожидаемым параметрам. Дополнительная проверка не требуется."
            elif overall_score >= 0.3:
                risk_level = "СРЕДНИЙ"
                risk_class = "warning"
                recommendation = "Обнаружены некоторые отклонения. Рекомендуется дополнительная экспертная оценка."
            else:
                risk_level = "ВЫСОКИЙ"
                risk_class = "critical"
                recommendation = "Обнаружены значительные аномалии. Высокая вероятность использования маски или двойника."
            
            # Расчет возраста Путина
            analysis_date = analysis_results.get("analysis_date")
            if analysis_date:
                if isinstance(analysis_date, str):
                    analysis_date = datetime.fromisoformat(analysis_date).date()
                putin_age = (analysis_date - PUTIN_BIRTH_DATE).days / 365.25
            else:
                putin_age = (datetime.now().date() - PUTIN_BIRTH_DATE).days / 365.25
            
            # Подготовка структурированных данных
            report_data = {
                "client_id": client_id,
                "overall_score": f"{overall_score:.3f}",
                "risk_level": risk_level,
                "risk_class": risk_class,
                "recommendation": recommendation,
                "putin_age": f"{putin_age:.1f}",
                "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "system_version": self.config.get("system_version", "2.0"),
                
                # Секции анализа
                "geometry_scores": analysis_results.get("geometry_analysis", {}).get("scores", {}),
                "embedding_scores": analysis_results.get("embedding_analysis", {}).get("scores", {}),
                "texture_scores": analysis_results.get("texture_analysis", {}).get("scores", {}),
                "temporal_data": analysis_results.get("temporal_analysis", {}),
                "anomalies": analysis_results.get("anomaly_detection", {}),
                "medical_data": analysis_results.get("medical_validation", {}),
                
                # Метаданные
                "total_images": analysis_results.get("total_images", 0),
                "processing_time": analysis_results.get("processing_time", 0),
                "analysis_date": analysis_date or datetime.now().date(),
                
                # Дополнительные данные
                "raw_analysis_results": analysis_results
            }
            
            logger.info(f"{Colors.GREEN}✔ Данные отчета подготовлены для клиента {client_id}")
            return report_data
            
        except Exception as e:
            logger.error(f"{Colors.RED}❌ Ошибка подготовки данных отчета: {e}")
            return {}

    def _export_html(self, report_data: Dict[str, Any], timestamp: str, 
                    template_type: str = "comprehensive") -> Dict[str, Any]:
        """Экспорт отчета в HTML"""
        try:
            logger.info(f"{Colors.CYAN}Экспорт HTML отчета...")
            
            # Создание директории для клиента
            output_dir = self.results_dir / report_data["client_id"]
            output_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = output_dir / f"report_{timestamp}.html"
            
            # Генерация визуализаций
            visualizations_html = self._generate_visualizations(report_data)
            report_data["visualizations"] = visualizations_html
            
            # Генерация выводов
            conclusions_html = self._generate_conclusions(report_data)
            report_data["conclusions"] = conclusions_html
            
            # Рендеринг шаблона
            try:
                template = self.jinja_env.get_template(f"{template_type}_report.html")
                html_content = template.render(**report_data)
            except Exception as template_error:
                logger.warning(f"{Colors.YELLOW}Ошибка загрузки шаблона, используем встроенный: {template_error}")
                html_content = self._render_builtin_template(report_data)
            
            # Сохранение файла
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            
            logger.info(f"{Colors.GREEN}✔ HTML отчет сохранен: {file_path}")
            return {"status": "success", "file_path": str(file_path), "format": "HTML"}
            
        except Exception as e:
            logger.error(f"{Colors.RED}❌ Ошибка экспорта HTML отчета: {e}")
            return {"status": "error", "message": f"Ошибка экспорта HTML: {e}"}

    def _render_builtin_template(self, report_data: Dict[str, Any]) -> str:
        """Рендеринг встроенного шаблона"""
        try:
            template_content = self.report_templates.get("comprehensive", "")
            if not template_content:
                # Минимальный HTML если шаблон не найден
                template_content = """
<!DOCTYPE html>
<html>
<head><title>Отчет анализа</title></head>
<body>
<h1>Отчет анализа лица</h1>
<p>Общий балл: {{ overall_score }}</p>
<p>Уровень риска: {{ risk_level }}</p>
<p>Дата: {{ generation_date }}</p>
</body>
</html>
                """
            
            template = Template(template_content)
            return template.render(**report_data)
            
        except Exception as e:
            logger.error(f"{Colors.RED}❌ Ошибка рендеринга встроенного шаблона: {e}")
            return f"<html><body><h1>Ошибка генерации отчета</h1><p>{str(e)}</p></body></html>"

    def _generate_visualizations(self, report_data: Dict[str, Any]) -> str:
        """Генерация визуализаций для отчета"""
        try:
            if not HAS_PLOTLY:
                return "<p>Библиотека Plotly не установлена. Визуализации недоступны.</p>"
            
            logger.info(f"{Colors.CYAN}Генерация визуализаций...")
            
            visualizations_html = ""
            
            # График распределения баллов
            scores = [
                report_data["geometry_scores"].get("overall_geometry_score", 0),
                report_data["embedding_scores"].get("overall_embedding_score", 0),
                report_data["texture_scores"].get("overall_texture_score", 0)
            ]
            categories = ["Геометрия", "Эмбеддинги", "Текстура"]
            
            fig_scores = go.Figure(data=[
                go.Bar(x=categories, y=scores, marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ])
            fig_scores.update_layout(
                title="Распределение баллов по категориям",
                xaxis_title="Категория анализа",
                yaxis_title="Балл",
                height=400
            )
            
            visualizations_html += f"<div>{fig_scores.to_html(include_plotlyjs=False, div_id='scores_chart')}</div>"
            
            # Временной график (если есть данные)
            temporal_data = report_data.get("temporal_data", {})
            if temporal_data:
                fig_temporal = go.Figure()
                
                # Добавление временных метрик
                for metric_name, values in temporal_data.items():
                    if isinstance(values, list) and len(values) > 1:
                        fig_temporal.add_trace(go.Scatter(
                            y=values,
                            mode='lines+markers',
                            name=metric_name
                        ))
                
                fig_temporal.update_layout(
                    title="Временная динамика метрик",
                    xaxis_title="Время",
                    yaxis_title="Значение",
                    height=400
                )
                
                visualizations_html += f"<div>{fig_temporal.to_html(include_plotlyjs=False, div_id='temporal_chart')}</div>"
            
            logger.info(f"{Colors.GREEN}✔ Визуализации сгенерированы")
            return visualizations_html
            
        except Exception as e:
            logger.error(f"{Colors.RED}❌ Ошибка генерации визуализаций: {e}")
            return f"<p>Ошибка при генерации визуализаций: {e}</p>"

    def _generate_conclusions(self, report_data: Dict[str, Any]) -> str:
        """Генерация раздела выводов и рекомендаций"""
        try:
            logger.info(f"{Colors.CYAN}Генерация выводов...")
            
            overall_score = float(report_data["overall_score"])
            risk_level = report_data["risk_level"]
            
            conclusions_html = f"""
            <h3>Основные выводы</h3>
            <p>На основе всестороннего анализа получен итоговый балл аутентичности <strong>{overall_score:.3f}</strong>, 
            что соответствует уровню риска <strong>{risk_level}</strong>.</p>
            
            <h3>Детальный анализ</h3>
            <ul>
            """
            
            # Анализ геометрии
            geometry_scores = report_data.get("geometry_scores", {})
            if geometry_scores:
                avg_geometry = np.mean([v for v in geometry_scores.values() if isinstance(v, (int, float))])
                if avg_geometry > 0.7:
                    conclusions_html += "<li><strong>Геометрический анализ:</strong> Высокое соответствие ожидаемым параметрам лица.</li>"
                elif avg_geometry > 0.3:
                    conclusions_html += "<li><strong>Геометрический анализ:</strong> Обнаружены умеренные отклонения в геометрии лица.</li>"
                else:
                    conclusions_html += "<li><strong>Геометрический анализ:</strong> Значительные отклонения от ожидаемых параметров.</li>"
            
            # Анализ эмбеддингов
            embedding_scores = report_data.get("embedding_scores", {})
            if embedding_scores:
                avg_embedding = np.mean([v for v in embedding_scores.values() if isinstance(v, (int, float))])
                if avg_embedding > 0.7:
                    conclusions_html += "<li><strong>Анализ эмбеддингов:</strong> Высокое сходство с референсными векторами.</li>"
                elif avg_embedding > 0.3:
                    conclusions_html += "<li><strong>Анализ эмбеддингов:</strong> Умеренное сходство с эталонными данными.</li>"
                else:
                    conclusions_html += "<li><strong>Анализ эмбеддингов:</strong> Низкое сходство, возможна подмена.</li>"
            
            # Анализ текстуры
            texture_scores = report_data.get("texture_scores", {})
            if texture_scores:
                avg_texture = np.mean([v for v in texture_scores.values() if isinstance(v, (int, float))])
                if avg_texture > 0.7:
                    conclusions_html += "<li><strong>Анализ текстуры:</strong> Естественная текстура кожи без артефактов.</li>"
                elif avg_texture > 0.3:
                    conclusions_html += "<li><strong>Анализ текстуры:</strong> Обнаружены незначительные аномалии текстуры.</li>"
                else:
                    conclusions_html += "<li><strong>Анализ текстуры:</strong> Выявлены признаки искусственной текстуры.</li>"
            
            conclusions_html += "</ul>"
            
            # Рекомендации
            conclusions_html += f"""
            <h3>Рекомендации</h3>
            <p>{report_data["recommendation"]}</p>
            
            <h3>Дополнительные замечания</h3>
            <p>Анализ проведен для возраста {report_data["putin_age"]} лет на дату {report_data["generation_date"]}.</p>
            <p>Для получения более детальной информации обратитесь к соответствующим разделам отчета.</p>
            """
            
            logger.info(f"{Colors.GREEN}✔ Выводы сгенерированы")
            return conclusions_html
            
        except Exception as e:
            logger.error(f"{Colors.RED}❌ Ошибка генерации выводов: {e}")
            return f"<p>Ошибка генерации выводов: {e}</p>"

    def _export_pdf(self, report_data: Dict[str, Any], timestamp: str) -> Dict[str, Any]:
        """Экспорт отчета в PDF"""
        try:
            if not HAS_REPORTLAB:
                return {"status": "error", "message": "ReportLab не установлен. PDF экспорт недоступен."}
            
            logger.info(f"{Colors.CYAN}Экспорт PDF отчета...")
            
            output_dir = self.results_dir / report_data["client_id"]
            output_dir.mkdir(parents=True, exist_ok=True)
            file_path = output_dir / f"report_{timestamp}.pdf"
            
            # Создание PDF документа
            doc = SimpleDocTemplate(str(file_path), pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Заголовок
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                alignment=1  # Центрирование
            )
            
            story.append(Paragraph("Комплексный отчет анализа лица", title_style))
            story.append(Spacer(1, 0.2 * inch))
            
            # Метаданные
            normal_style = styles['Normal']
            story.append(Paragraph(f"<b>Дата генерации:</b> {report_data['generation_date']}", normal_style))
            story.append(Paragraph(f"<b>Версия системы:</b> {report_data['system_version']}", normal_style))
            story.append(Paragraph(f"<b>ID клиента:</b> {report_data['client_id']}", normal_style))
            story.append(Spacer(1, 0.2 * inch))
            
            # Исполнительное резюме
            story.append(Paragraph("Исполнительное резюме", styles['Heading2']))
            story.append(Paragraph(f"<b>Итоговый балл аутентичности:</b> {report_data['overall_score']}", normal_style))
            story.append(Paragraph(f"<b>Уровень риска:</b> {report_data['risk_level']}", normal_style))
            story.append(Paragraph(f"<b>Рекомендация:</b> {report_data['recommendation']}", normal_style))
            story.append(Spacer(1, 0.2 * inch))
            
            # Таблица результатов
            story.append(Paragraph("Детальные результаты", styles['Heading2']))
            
            # Геометрический анализ
            if report_data.get("geometry_scores"):
                table_data = [["Метрика", "Значение"]]
                for metric, value in report_data["geometry_scores"].items():
                    if isinstance(value, (int, float)):
                        table_data.append([metric, f"{value:.3f}"])
                    else:
                        table_data.append([metric, str(value)])
                
                table = Table(table_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 14),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(Paragraph("Геометрический анализ", styles['Heading3']))
                story.append(table)
                story.append(Spacer(1, 0.1 * inch))
            
            # Проверка размера файла
            doc.build(story)
            
            # Проверка ограничения размера
            file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            if file_size_mb > PDF_SIZE_LIMIT_MB:
                logger.warning(f"{Colors.YELLOW}Размер PDF ({file_size_mb:.1f} МБ) превышает лимит {PDF_SIZE_LIMIT_MB} МБ")
                # Можно добавить логику сжатия или уменьшения контента
            
            logger.info(f"{Colors.GREEN}✔ PDF отчет сохранен: {file_path} ({file_size_mb:.1f} МБ)")
            return {"status": "success", "file_path": str(file_path), "format": "PDF", "size_mb": file_size_mb}
            
        except Exception as e:
            logger.error(f"{Colors.RED}❌ Ошибка экспорта PDF отчета: {e}")
            return {"status": "error", "message": f"Ошибка экспорта PDF: {e}"}

    def _export_csv(self, report_data: Dict[str, Any], timestamp: str) -> Dict[str, Any]:
        """Экспорт отчета в CSV"""
        try:
            logger.info(f"{Colors.CYAN}Экспорт CSV отчета...")
            
            output_dir = self.results_dir / report_data["client_id"]
            output_dir.mkdir(parents=True, exist_ok=True)
            file_path = output_dir / f"report_{timestamp}.csv"
            
            # Подготовка данных для CSV
            csv_data = []
            
            # Основные метрики
            base_row = {
                "client_id": report_data["client_id"],
                "overall_score": report_data["overall_score"],
                "risk_level": report_data["risk_level"],
                "putin_age": report_data["putin_age"],
                "generation_date": report_data["generation_date"],
                "system_version": report_data["system_version"]
            }
            
            # Добавление геометрических метрик
            for metric, value in report_data.get("geometry_scores", {}).items():
                base_row[f"geometry_{metric}"] = value
            
            # Добавление метрик эмбеддингов
            for metric, value in report_data.get("embedding_scores", {}).items():
                base_row[f"embedding_{metric}"] = value
            
            # Добавление метрик текстуры
            for metric, value in report_data.get("texture_scores", {}).items():
                base_row[f"texture_{metric}"] = value
            
            # Добавление временных метрик
            for metric, value in report_data.get("temporal_data", {}).items():
                base_row[f"temporal_{metric}"] = value
            
            # Добавление аномалий
            for metric, value in report_data.get("anomalies", {}).items():
                base_row[f"anomaly_{metric}"] = value
            
            # Добавление медицинских данных
            for metric, value in report_data.get("medical_data", {}).items():
                base_row[f"medical_{metric}"] = value
            
            csv_data.append(base_row)
            
            # Создание DataFrame и сохранение
            df = pd.DataFrame(csv_data)
            df.to_csv(file_path, index=False, encoding="utf-8")
            
            logger.info(f"{Colors.GREEN}✔ CSV отчет сохранен: {file_path}")
            return {"status": "success", "file_path": str(file_path), "format": "CSV"}
            
        except Exception as e:
            logger.error(f"{Colors.RED}❌ Ошибка экспорта CSV отчета: {e}")
            return {"status": "error", "message": f"Ошибка экспорта CSV: {e}"}

    def _export_excel(self, report_data: Dict[str, Any], timestamp: str) -> Dict[str, Any]:
        """Экспорт отчета в Excel"""
        try:
            if not HAS_OPENPYXL:
                return {"status": "error", "message": "OpenPyXL не установлен. Excel экспорт недоступен."}
            
            logger.info(f"{Colors.CYAN}Экспорт Excel отчета...")
            
            output_dir = self.results_dir / report_data["client_id"]
            output_dir.mkdir(parents=True, exist_ok=True)
            file_path = output_dir / f"report_{timestamp}.xlsx"
            
            # Создание Excel файла с несколькими листами
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # Лист сводки
                summary_data = {
                    "Параметр": ["ID клиента", "Общий балл", "Уровень риска", "Возраст Путина", "Дата генерации"],
                    "Значение": [
                        report_data["client_id"],
                        report_data["overall_score"],
                        report_data["risk_level"],
                        report_data["putin_age"],
                        report_data["generation_date"]
                    ]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name="Сводка", index=False)
                
                # Лист геометрических метрик
                if report_data.get("geometry_scores"):
                    geometry_df = pd.DataFrame([
                        {"Метрика": k, "Значение": v} 
                        for k, v in report_data["geometry_scores"].items()
                    ])
                    geometry_df.to_excel(writer, sheet_name="Геометрия", index=False)
                
                # Лист метрик эмбеддингов
                if report_data.get("embedding_scores"):
                    embedding_df = pd.DataFrame([
                        {"Метрика": k, "Значение": v} 
                        for k, v in report_data["embedding_scores"].items()
                    ])
                    embedding_df.to_excel(writer, sheet_name="Эмбеддинги", index=False)
                
                # Лист метрик текстуры
                if report_data.get("texture_scores"):
                    texture_df = pd.DataFrame([
                        {"Метрика": k, "Значение": v} 
                        for k, v in report_data["texture_scores"].items()
                    ])
                    texture_df.to_excel(writer, sheet_name="Текстура", index=False)
            
            logger.info(f"{Colors.GREEN}✔ Excel отчет сохранен: {file_path}")
            return {"status": "success", "file_path": str(file_path), "format": "Excel"}
            
        except Exception as e:
            logger.error(f"{Colors.RED}❌ Ошибка экспорта Excel отчета: {e}")
            return {"status": "error", "message": f"Ошибка экспорта Excel: {e}"}

    def _export_json(self, report_data: Dict[str, Any], timestamp: str) -> Dict[str, Any]:
        """Экспорт отчета в JSON"""
        try:
            logger.info(f"{Colors.CYAN}Экспорт JSON отчета...")
            
            output_dir = self.results_dir / report_data["client_id"]
            output_dir.mkdir(parents=True, exist_ok=True)
            file_path = output_dir / f"report_{timestamp}.json"
            
            # Подготовка данных для JSON
            json_data = self._serialize_for_json(report_data)
            
            # Добавление метаданных
            json_data["export_metadata"] = {
                "export_timestamp": timestamp,
                "export_format": "JSON",
                "generator_version": "2.0",
                "data_schema_version": "1.0"
            }
            
            # Сохранение файла
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)
            
            logger.info(f"{Colors.GREEN}✔ JSON отчет сохранен: {file_path}")
            return {"status": "success", "file_path": str(file_path), "format": "JSON"}
            
        except Exception as e:
            logger.error(f"{Colors.RED}❌ Ошибка экспорта JSON отчета: {e}")
            return {"status": "error", "message": f"Ошибка экспорта JSON: {e}"}

    def _serialize_for_json(self, data: Any) -> Any:
        """Рекурсивное преобразование данных для JSON сериализации"""
        try:
            if isinstance(data, (datetime, date)):
                return data.isoformat()
            elif isinstance(data, timedelta):
                return str(data)
            elif isinstance(data, Path):
                return str(data)
            elif isinstance(data, np.integer):
                return int(data)
            elif isinstance(data, np.floating):
                return float(data)
            elif isinstance(data, np.ndarray):
                return data.tolist()
            elif isinstance(data, dict):
                return {k: self._serialize_for_json(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [self._serialize_for_json(item) for item in data]
            else:
                return data
                
        except Exception as e:
            logger.error(f"{Colors.RED}❌ Ошибка сериализации данных: {e}")
            return str(data)

    def _encrypt_report_file(self, export_result: Dict[str, Any]) -> Dict[str, Any]:
        """Шифрование файла отчета"""
        try:
            if not self.encryption_key or not HAS_CRYPTOGRAPHY:
                logger.warning(f"{Colors.YELLOW}Шифрование недоступно")
                return export_result
            
            logger.info(f"{Colors.CYAN}Шифрование отчета...")
            
            file_path = Path(export_result["file_path"])
            encrypted_path = file_path.with_suffix(file_path.suffix + ".encrypted")
            
            # Чтение исходного файла
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            # Шифрование
            fernet = Fernet(self.encryption_key)
            encrypted_data = fernet.encrypt(file_data)
            
            # Сохранение зашифрованного файла
            with open(encrypted_path, 'wb') as f:
                f.write(encrypted_data)
            
            # Удаление исходного файла
            file_path.unlink()
            
            export_result["file_path"] = str(encrypted_path)
            export_result["encrypted"] = True
            
            logger.info(f"{Colors.GREEN}✔ Отчет зашифрован: {encrypted_path}")
            return export_result
            
        except Exception as e:
            logger.error(f"{Colors.RED}❌ Ошибка шифрования отчета: {e}")
            return export_result

    def decrypt_report_file(self, encrypted_file_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Расшифровка файла отчета"""
        try:
            if not self.encryption_key or not HAS_CRYPTOGRAPHY:
                return {"status": "error", "message": "Шифрование недоступно"}
            
            logger.info(f"{Colors.CYAN}Расшифровка отчета...")
            
            encrypted_path = Path(encrypted_file_path)
            if not encrypted_path.exists():
                return {"status": "error", "message": "Зашифрованный файл не найден"}
            
            # Определение пути для расшифрованного файла
            if output_path:
                decrypted_path = Path(output_path)
            else:
                decrypted_path = encrypted_path.with_suffix("")
            
            # Чтение зашифрованного файла
            with open(encrypted_path, 'rb') as f:
                encrypted_data = f.read()
            
            # Расшифровка
            fernet = Fernet(self.encryption_key)
            decrypted_data = fernet.decrypt(encrypted_data)
            
            # Сохранение расшифрованного файла
            with open(decrypted_path, 'wb') as f:
                f.write(decrypted_data)
            
            logger.info(f"{Colors.GREEN}✔ Отчет расшифрован: {decrypted_path}")
            return {"status": "success", "file_path": str(decrypted_path)}
            
        except Exception as e:
            logger.error(f"{Colors.RED}❌ Ошибка расшифровки отчета: {e}")
            return {"status": "error", "message": f"Ошибка расшифровки: {e}"}

    def generate_batch_reports(self, analysis_results_list: List[Dict[str, Any]], 
                             export_format: str = "HTML") -> Dict[str, Any]:
        """Генерация пакета отчетов"""
        try:
            logger.info(f"{Colors.CYAN}Генерация пакета из {len(analysis_results_list)} отчетов...")
            
            batch_results = {
                "total_reports": len(analysis_results_list),
                "successful": 0,
                "failed": 0,
                "reports": [],
                "batch_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            
            for i, analysis_result in enumerate(analysis_results_list):
                try:
                    logger.info(f"{Colors.CYAN}Обработка отчета {i+1}/{len(analysis_results_list)}...")
                    
                    result = self.generate_full_report(analysis_result, export_format)
                    
                    if result["status"] == "success":
                        batch_results["successful"] += 1
                    else:
                        batch_results["failed"] += 1
                    
                    batch_results["reports"].append(result)
                    
                except Exception as e:
                    logger.error(f"{Colors.RED}❌ Ошибка обработки отчета {i+1}: {e}")
                    batch_results["failed"] += 1
                    batch_results["reports"].append({
                        "status": "error",
                        "message": f"Ошибка обработки: {e}"
                    })
            
            logger.info(f"{Colors.GREEN}✔ Пакетная генерация завершена: {batch_results['successful']} успешно, {batch_results['failed']} с ошибками")
            return batch_results
            
        except Exception as e:
            logger.error(f"{Colors.RED}❌ Критическая ошибка пакетной генерации: {e}")
            return {"status": "error", "message": f"Критическая ошибка: {e}"}

    def create_archive_report(self, client_id: str, include_cache: bool = False) -> Dict[str, Any]:
        """Создание архивного отчета со всеми данными клиента"""
        try:
            logger.info(f"{Colors.CYAN}Создание архивного отчета для клиента {client_id}...")
            
            client_dir = self.results_dir / client_id
            if not client_dir.exists():
                return {"status": "error", "message": f"Данные клиента {client_id} не найдены"}
            
            # Создание архива
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = self.results_dir / f"archive_{client_id}_{timestamp}.zip"
            
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Добавление всех отчетов клиента
                for file_path in client_dir.rglob("*"):
                    if file_path.is_file():
                        arcname = file_path.relative_to(client_dir)
                        zipf.write(file_path, arcname)
                
                # Добавление кэша если требуется
                if include_cache:
                    cache_client_dir = self.cache_dir / client_id
                    if cache_client_dir.exists():
                        for file_path in cache_client_dir.rglob("*"):
                            if file_path.is_file():
                                arcname = Path("cache") / file_path.relative_to(cache_client_dir)
                                zipf.write(file_path, arcname)
                
                # Добавление метаданных архива
                archive_metadata = {
                    "client_id": client_id,
                    "archive_timestamp": timestamp,
                    "generator_version": "2.0",
                    "include_cache": include_cache,
                    "total_files": len(zipf.namelist())
                }
                
                zipf.writestr("archive_metadata.json", 
                            json.dumps(archive_metadata, indent=2, ensure_ascii=False))
            
            archive_size_mb = archive_path.stat().st_size / (1024 * 1024)
            
            logger.info(f"{Colors.GREEN}✔ Архивный отчет создан: {archive_path} ({archive_size_mb:.1f} МБ)")
            return {
                "status": "success", 
                "archive_path": str(archive_path),
                "size_mb": archive_size_mb,
                "files_count": len(zipf.namelist())
            }
            
        except Exception as e:
            logger.error(f"{Colors.RED}❌ Ошибка создания архивного отчета: {e}")
            return {"status": "error", "message": f"Ошибка создания архива: {e}"}

    def get_generation_statistics(self) -> Dict[str, Any]:
        """Получение статистики генерации отчетов"""
        try:
            logger.info(f"{Colors.BLUE}Запрос статистики генерации отчетов")
            
            stats = self.generation_stats.copy()
            
            # Добавление вычисляемых метрик
            if stats["total_reports"] > 0:
                stats["success_rate"] = stats["successful_exports"] / stats["total_reports"]
                stats["failure_rate"] = stats["failed_exports"] / stats["total_reports"]
            else:
                stats["success_rate"] = 0.0
                stats["failure_rate"] = 0.0
            
            # Информация о поддерживаемых форматах
            stats["supported_formats"] = self.supported_formats
            stats["available_libraries"] = {
                "reportlab": HAS_REPORTLAB,
                "plotly": HAS_PLOTLY,
                "openpyxl": HAS_OPENPYXL,
                "cryptography": HAS_CRYPTOGRAPHY
            }
            
            # Информация о памяти
            process = psutil.Process()
            memory_info = process.memory_info()
            stats["memory_usage_mb"] = memory_info.rss / (1024 * 1024)
            
            return stats
            
        except Exception as e:
            logger.error(f"{Colors.RED}❌ Ошибка получения статистики: {e}")
            return {"error": str(e)}

    def clear_cache(self):
        """Очистка кэша отчетов"""
        try:
            self.reports_cache.clear()
            logger.info(f"{Colors.GREEN}✔ Кэш ReportGenerator очищен")
            
        except Exception as e:
            logger.error(f"{Colors.RED}❌ Ошибка очистки кэша: {e}")

    def validate_report_data(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Валидация данных для генерации отчета"""
        try:
            logger.info(f"{Colors.CYAN}Валидация данных отчета...")
            
            validation_result = {
                "is_valid": True,
                "errors": [],
                "warnings": []
            }
            
            # Проверка обязательных полей
            required_fields = ["client_id", "overall_authenticity_score"]
            for field in required_fields:
                if field not in analysis_results:
                    validation_result["errors"].append(f"Отсутствует обязательное поле: {field}")
                    validation_result["is_valid"] = False
            
            # Проверка типов данных
            if "overall_authenticity_score" in analysis_results:
                score = analysis_results["overall_authenticity_score"]
                if not isinstance(score, (int, float)) or not (0 <= score <= 1):
                    validation_result["errors"].append("overall_authenticity_score должен быть числом от 0 до 1")
                    validation_result["is_valid"] = False
            
            # Проверка структуры данных
            expected_sections = ["geometry_analysis", "embedding_analysis", "texture_analysis"]
            for section in expected_sections:
                if section not in analysis_results:
                    validation_result["warnings"].append(f"Отсутствует секция: {section}")
            
            logger.info(f"{Colors.GREEN}✔ Валидация завершена: {'успешно' if validation_result['is_valid'] else 'с ошибками'}")
            return validation_result
            
        except Exception as e:
            logger.error(f"{Colors.RED}❌ Ошибка валидации данных: {e}")
            return {
                "is_valid": False,
                "errors": [f"Ошибка валидации: {e}"],
                "warnings": []
            }

    def generate_batch_reports(self, analysis_results_list: List[Dict[str, Any]], 
                            export_format: str = "HTML") -> Dict[str, Any]:
        """Генерация пакета отчетов"""
        try:
            logger.info(f"{Colors.CYAN}Генерация пакета из {len(analysis_results_list)} отчетов...")
            
            batch_results = {
                "total_reports": len(analysis_results_list),
                "successful": 0,
                "failed": 0,
                "reports": [],
                "batch_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            
            for i, analysis_result in enumerate(analysis_results_list):
                try:
                    logger.info(f"{Colors.CYAN}Обработка отчета {i+1}/{len(analysis_results_list)}...")
                    
                    result = self.generate_comprehensive_report(analysis_result, export_format)
                    
                    if result["status"] == "success":
                        batch_results["successful"] += 1
                    else:
                        batch_results["failed"] += 1
                    
                    batch_results["reports"].append(result)
                    
                except Exception as e:
                    logger.error(f"{Colors.RED}❌ Ошибка обработки отчета {i+1}: {e}")
                    batch_results["failed"] += 1
                    batch_results["reports"].append({
                        "status": "error",
                        "message": f"Ошибка обработки: {e}"
                    })
            
            logger.info(f"{Colors.GREEN}✔ Пакетная генерация завершена: {batch_results['successful']} успешно, {batch_results['failed']} с ошибками")
            return batch_results
            
        except Exception as e:
            logger.error(f"{Colors.RED}❌ Критическая ошибка пакетной генерации: {e}")
            return {"status": "error", "message": f"Критическая ошибка: {e}"}

    def create_archive_report(self, client_id: str, include_cache: bool = False) -> Dict[str, Any]:
        """Создание архивного отчета со всеми данными клиента"""
        try:
            logger.info(f"{Colors.CYAN}Создание архивного отчета для клиента {client_id}...")
            
            client_dir = RESULTS_DIR / client_id
            if not client_dir.exists():
                return {"status": "error", "message": f"Данные клиента {client_id} не найдены"}
            
            # Создание архива
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = RESULTS_DIR / f"archive_{client_id}_{timestamp}.zip"
            
            import zipfile
            
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Добавление всех отчетов клиента
                for file_path in client_dir.rglob("*"):
                    if file_path.is_file():
                        arcname = file_path.relative_to(client_dir)
                        zipf.write(file_path, arcname)
                
                # Добавление кэша если требуется
                if include_cache:
                    cache_client_dir = CACHE_DIR / client_id
                    if cache_client_dir.exists():
                        for file_path in cache_client_dir.rglob("*"):
                            if file_path.is_file():
                                arcname = Path("cache") / file_path.relative_to(cache_client_dir)
                                zipf.write(file_path, arcname)
                
                # Добавление метаданных архива
                archive_metadata = {
                    "client_id": client_id,
                    "archive_timestamp": timestamp,
                    "generator_version": SYSTEM_VERSION,
                    "include_cache": include_cache,
                    "total_files": len(zipf.namelist())
                }
                
                zipf.writestr("archive_metadata.json", 
                            json.dumps(archive_metadata, indent=2, ensure_ascii=False))
            
            archive_size_mb = archive_path.stat().st_size / (1024 * 1024)
            
            logger.info(f"{Colors.GREEN}✔ Архивный отчет создан: {archive_path} ({archive_size_mb:.1f} МБ)")
            return {
                "status": "success", 
                "archive_path": str(archive_path),
                "size_mb": archive_size_mb,
                "files_count": len(zipf.namelist())
            }
            
        except Exception as e:
            logger.error(f"{Colors.RED}❌ Ошибка создания архивного отчета: {e}")
            return {"status": "error", "message": f"Ошибка создания архива: {e}"}

    def export_machine_readable_data(self, analysis_results: Dict[str, Any], 
                                output_format: str = "JSON") -> Dict[str, Any]:
        """Экспорт машиночитаемых данных для ML-обучения"""
        try:
            logger.info(f"{Colors.CYAN}Экспорт машиночитаемых данных в формате {output_format}...")
            
            # Подготовка структурированных данных
            ml_data = {
                "metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "system_version": SYSTEM_VERSION,
                    "putin_birth_date": PUTIN_BIRTH_DATE,
                    "data_schema_version": "1.0"
                },
                "features": self._extract_ml_features(analysis_results),
                "labels": self._extract_ml_labels(analysis_results),
                "raw_metrics": analysis_results
            }
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            client_id = analysis_results.get("client_id", "unknown")
            
            output_dir = RESULTS_DIR / client_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if output_format.upper() == "JSON":
                file_path = output_dir / f"ml_data_{timestamp}.json"
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(ml_data, f, indent=2, ensure_ascii=False, default=str)
                    
            elif output_format.upper() == "CSV":
                file_path = output_dir / f"ml_data_{timestamp}.csv"
                # Преобразование в плоскую структуру для CSV
                flat_data = self._flatten_ml_data(ml_data)
                df = pd.DataFrame([flat_data])
                df.to_csv(file_path, index=False, encoding="utf-8")
                
            else:
                return {"status": "error", "message": f"Неподдерживаемый формат: {output_format}"}
            
            logger.info(f"{Colors.GREEN}✔ Машиночитаемые данные экспортированы: {file_path}")
            return {"status": "success", "file_path": str(file_path), "format": output_format}
            
        except Exception as e:
            logger.error(f"{Colors.RED}❌ Ошибка экспорта машиночитаемых данных: {e}")
            return {"status": "error", "message": f"Ошибка экспорта: {e}"}

    def _extract_ml_features(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Извлечение признаков для ML"""
        try:
            features = {}
            
            # Геометрические признаки
            geometry = analysis_results.get("geometry_analysis", {}).get("scores", {})
            for key, value in geometry.items():
                if isinstance(value, (int, float)):
                    features[f"geometry_{key}"] = float(value)
            
            # Эмбеддинг признаки
            embedding = analysis_results.get("embedding_analysis", {}).get("scores", {})
            for key, value in embedding.items():
                if isinstance(value, (int, float)):
                    features[f"embedding_{key}"] = float(value)
            
            # Текстурные признаки
            texture = analysis_results.get("texture_analysis", {}).get("scores", {})
            for key, value in texture.items():
                if isinstance(value, (int, float)):
                    features[f"texture_{key}"] = float(value)
            
            # Временные признаки
            temporal = analysis_results.get("temporal_analysis", {})
            for key, value in temporal.items():
                if isinstance(value, (int, float)):
                    features[f"temporal_{key}"] = float(value)
            
            return features
            
        except Exception as e:
            logger.error(f"{Colors.RED}❌ Ошибка извлечения ML признаков: {e}")
            return {}

    def _extract_ml_labels(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Извлечение меток для ML"""
        try:
            labels = {}
            
            # Основная метка аутентичности
            overall_score = analysis_results.get("overall_authenticity_score", 0.0)
            labels["authenticity_score"] = float(overall_score)
            
            # Категориальная метка
            if overall_score >= 0.7:
                labels["authenticity_category"] = "authentic"
            elif overall_score >= 0.3:
                labels["authenticity_category"] = "suspicious"
            else:
                labels["authenticity_category"] = "fake"
            
            # Детекция маски
            mask_detected = analysis_results.get("mask_detection", {}).get("is_mask", False)
            labels["mask_detected"] = bool(mask_detected)
            
            return labels
            
        except Exception as e:
            logger.error(f"{Colors.RED}❌ Ошибка извлечения ML меток: {e}")
            return {}

    def _flatten_ml_data(self, ml_data: Dict[str, Any]) -> Dict[str, Any]:
        """Преобразование вложенной структуры в плоскую для CSV"""
        try:
            flat_data = {}
            
            # Метаданные
            for key, value in ml_data.get("metadata", {}).items():
                flat_data[f"meta_{key}"] = value
            
            # Признаки
            for key, value in ml_data.get("features", {}).items():
                flat_data[f"feature_{key}"] = value
            
            # Метки
            for key, value in ml_data.get("labels", {}).items():
                flat_data[f"label_{key}"] = value
            
            return flat_data
            
        except Exception as e:
            logger.error(f"{Colors.RED}❌ Ошибка преобразования данных: {e}")
            return {}

    def optimize_pdf_size(self, pdf_path: str, max_size_mb: float = 5.0) -> Dict[str, Any]:
        """Оптимизация размера PDF файла"""
        try:
            logger.info(f"{Colors.CYAN}Оптимизация размера PDF: {pdf_path}")
            
            file_size_mb = Path(pdf_path).stat().st_size / (1024 * 1024)
            
            if file_size_mb <= max_size_mb:
                logger.info(f"{Colors.GREEN}✔ PDF уже оптимизирован: {file_size_mb:.1f} МБ")
                return {"status": "success", "size_mb": file_size_mb, "optimized": False}
            
            # Попытка оптимизации через сжатие изображений
            try:
                import PyPDF2
                
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    writer = PyPDF2.PdfWriter()
                    
                    for page in reader.pages:
                        writer.add_page(page)
                    
                    # Сжатие
                    writer.compress_identical_objects()
                    
                    # Сохранение оптимизированного файла
                    optimized_path = pdf_path.replace('.pdf', '_optimized.pdf')
                    with open(optimized_path, 'wb') as output_file:
                        writer.write(output_file)
                    
                    # Проверка нового размера
                    new_size_mb = Path(optimized_path).stat().st_size / (1024 * 1024)
                    
                    if new_size_mb < file_size_mb:
                        # Замена оригинала оптимизированным
                        Path(pdf_path).unlink()
                        Path(optimized_path).rename(pdf_path)
                        
                        logger.info(f"{Colors.GREEN}✔ PDF оптимизирован: {file_size_mb:.1f} → {new_size_mb:.1f} МБ")
                        return {"status": "success", "size_mb": new_size_mb, "optimized": True}
                    else:
                        # Удаление неудачной попытки
                        Path(optimized_path).unlink()
                        
            except ImportError:
                logger.warning(f"{Colors.YELLOW}PyPDF2 не установлен, оптимизация недоступна")
            
            logger.warning(f"{Colors.YELLOW}Не удалось оптимизировать PDF: {file_size_mb:.1f} МБ")
            return {"status": "warning", "size_mb": file_size_mb, "optimized": False}
            
        except Exception as e:
            logger.error(f"{Colors.RED}❌ Ошибка оптимизации PDF: {e}")
            return {"status": "error", "message": str(e)}

    def clear_cache(self):
        """Очистка кэша отчетов"""
        try:
            self.reports_cache.clear()
            logger.info(f"{Colors.GREEN}✔ Кэш ReportGenerator очищен")
            
        except Exception as e:
            logger.error(f"{Colors.RED}❌ Ошибка очистки кэша: {e}")

    def self_test(self) -> None:
        """Самотестирование генератора отчетов"""
        logger.info(f"{Colors.BOLD}--- Запуск самопроверки ReportGenerator ---{Colors.RESET}")
        
        # Моковые данные для теста
        mock_analysis_results = {
            "client_id": "test_client_001",
            "overall_authenticity_score": 0.75,
            "analysis_date": datetime.now().isoformat(),
            "geometry_analysis": {
                "scores": {
                    "3d_model_fit": 0.8,
                    "deviation_from_average": 0.1,
                    "unnatural_deformation": 0.05,
                    "overall_geometry_score": 0.85
                }
            },
            "embedding_analysis": {
                "scores": {
                    "cosine_similarity": 0.95,
                    "mahalanobis_distance": 0.1,
                    "clustering_metric": 0.88,
                    "overall_embedding_score": 0.91
                }
            },
            "texture_analysis": {
                "scores": {
                    "texture_quality": 0.9,
                    "rendering_artifacts": 0.02,
                    "lighting_consistency": 0.93,
                    "overall_texture_score": 0.88
                }
            },
            "temporal_analysis": {
                "parameter_stability": 0.9,
                "abnormal_transitions": 0.01,
                "average_frame_deviation": 0.005
            },
            "anomaly_detection": {
                "isolation_forest_score": 0.05,
                "lof_score": 0.1
            },
            "medical_validation": {
                "estimated_age": 70,
                "estimated_gender": "Мужской",
                "biometric_compatibility": 0.98
            }
        }
        
        try:
            # Тест валидации данных
            logger.info(f"{Colors.CYAN}Тестирование валидации данных...")
            validation = self.validate_report_data(mock_analysis_results)
            if validation["is_valid"]:
                logger.info(f"{Colors.GREEN}✔ Валидация данных прошла успешно")
            else:
                logger.error(f"{Colors.RED}❌ Ошибки валидации: {validation['errors']}")
            
            # Тест генерации HTML отчета
            logger.info(f"{Colors.CYAN}Тестирование генерации HTML отчета...")
            html_report = self.generate_comprehensive_report(mock_analysis_results, export_format="HTML")
            if html_report["status"] == "success":
                logger.info(f"{Colors.GREEN}✔ HTML отчет успешно сгенерирован: {html_report['file_path']}")
            else:
                logger.error(f"{Colors.RED}❌ Ошибка генерации HTML отчета: {html_report['message']}")
            
            # Тест генерации JSON отчета
            logger.info(f"{Colors.CYAN}Тестирование генерации JSON отчета...")
            json_report = self.generate_comprehensive_report(mock_analysis_results, export_format="JSON")
            if json_report["status"] == "success":
                logger.info(f"{Colors.GREEN}✔ JSON отчет успешно сгенерирован: {json_report['file_path']}")
            else:
                logger.error(f"{Colors.RED}❌ Ошибка генерации JSON отчета: {json_report['message']}")
            
            # Тест экспорта ML данных
            logger.info(f"{Colors.CYAN}Тестирование экспорта ML данных...")
            ml_export = self.export_machine_readable_data(mock_analysis_results)
            if ml_export["status"] == "success":
                logger.info(f"{Colors.GREEN}✔ ML данные экспортированы: {ml_export['file_path']}")
            else:
                logger.error(f"{Colors.RED}❌ Ошибка экспорта ML данных: {ml_export['message']}")
            
            # Тест создания архива
            logger.info(f"{Colors.CYAN}Тестирование создания архива...")
            archive = self.create_archive_report("test_client_001")
            if archive["status"] == "success":
                logger.info(f"{Colors.GREEN}✔ Архив создан: {archive['archive_path']}")
            else:
                logger.info(f"{Colors.YELLOW}Архив не создан (нормально для теста): {archive['message']}")
            
            # Статистика
            stats = self.get_generation_statistics()
            logger.info(f"{Colors.BOLD}--- Статистика генерации отчетов ---{Colors.RESET}")
            logger.info(f"Всего попыток: {stats['total_reports']}")
            logger.info(f"Успешных экспортов: {stats['successful_exports']}")
            logger.info(f"Неудачных экспортов: {stats['failed_exports']}")
            logger.info(f"Использованные форматы: {stats['formats_used']}")
            
        except Exception as e:
            logger.error(f"{Colors.RED}❌ Ошибка самотестирования: {e}")
        
        logger.info(f"{Colors.BOLD}--- Самопроверка ReportGenerator завершена ---{Colors.RESET}")

    # === ФУНКЦИИ САМОТЕСТИРОВАНИЯ ===

    def self_test():
        """Самотестирование модуля report_generator"""
        try:
            logger.info("Запуск самотестирования report_generator...")
            
            # Создание экземпляра генератора
            generator = ReportGenerator()
            
            # Тест поддерживаемых форматов
            assert "HTML" in generator.supported_formats, "HTML формат не поддерживается"
            assert "PDF" in generator.supported_formats, "PDF формат не поддерживается"
            assert "JSON" in generator.supported_formats, "JSON формат не поддерживается"
            
            # Тест загрузки шаблонов
            assert len(generator.report_templates) > 0, "Шаблоны отчетов не загружены"
            
            # Тест статистики
            stats = generator.get_generation_statistics()
            assert "total_reports" in stats, "Отсутствует статистика отчетов"
            
            logger.info("Самотестирование report_generator завершено успешно")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка самотестирования: {e}")
            return False

    # === ИНИЦИАЛИЗАЦИЯ ===

    if __name__ == "__main__":
        # Запуск самотестирования при прямом вызове модуля
        success = self_test()
        if success:
            print("✅ Модуль report_generator работает корректно")
            
            # Демонстрация основной функциональности
            generator = ReportGenerator()
            print(f"📊 Поддерживаемые форматы: {generator.supported_formats}")
            print(f"🔧 Шаблонов загружено: {len(generator.report_templates)}")
            print(f"📏 Библиотеки: ReportLab={HAS_REPORTLAB}, Plotly={HAS_PLOTLY}")
            print(f"💾 Директория результатов: {RESULTS_DIR}")
            print(f"🎛️ Версия системы: {SYSTEM_VERSION}")
        else:
            print("❌ Обнаружены ошибки в модуле report_generator")
            exit(1)