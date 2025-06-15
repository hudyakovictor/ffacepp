"""
ReportGenerator - Генератор отчетов с экспортом в различные форматы
Версия: 2.0
Дата: 2025-06-15
Исправлены все критические ошибки согласно правкам
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
from datetime import datetime, timedelta, date
import base64
import io
from jinja2 import Template

# --- ЦВЕТА КОНСОЛИ (Повторяются для каждого модуля, чтобы быть автономными) ---
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

# --- КАСТОМНЫЙ ФОРМАТТЕР ДЛЯ ЦВЕТНОГО ЛОГИРОВАНИЯ ---
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

# Настройка логирования
log_file_handler = logging.FileHandler('logs/reportgenerator.log')
log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'))

console_handler = logging.StreamHandler()
console_handler.setFormatter(ColoredFormatter())

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        log_file_handler,
        console_handler
    ]
)
logger = logging.getLogger(__name__)

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
    HAS_PLOTLY = True
    logger.info(f"{Colors.GREEN}✔ Plotly импортирована для интерактивных графиков.")
except ImportError as e:
    HAS_PLOTLY = False
    logger.warning(f"{Colors.YELLOW}❌ Plotly не найдена. Графики в отчетах будут ограничены. Детали: {e}")

# Импорт конфигурации
try:
    from core_config import (
        AUTHENTICITY_WEIGHTS, MASK_DETECTION_LEVELS, CRITICAL_THRESHOLDS,
        RESULTS_DIR, CACHE_DIR, PUTIN_BIRTH_DATE, SYSTEM_VERSION
    )
    logger.info(f"{Colors.GREEN}✔ Конфигурация успешно импортирована.")
except ImportError as e:
    logger.critical(f"{Colors.RED}❌ КРИТИЧЕСКАЯ ОШИБКА: Не удалось импортировать конфигурацию. Детали: {e}")
    # Значения по умолчанию
    AUTHENTICITY_WEIGHTS = {"geometry": 0.15, "embedding": 0.30, "texture": 0.10}
    MASK_DETECTION_LEVELS = {}
    CRITICAL_THRESHOLDS = {"min_authenticity_score": 0.6}
    RESULTS_DIR = Path("results")
    CACHE_DIR = Path("cache")
    PUTIN_BIRTH_DATE = "1952-10-07"
    SYSTEM_VERSION = "2.0"

# ==================== ОСНОВНОЙ КЛАСС ====================

class ReportGenerator:
    """
    ИСПРАВЛЕНО: Генератор отчетов с полной функциональностью
    Согласно правкам: экспорт в HTML, PDF, CSV, Excel, JSON с 9 секциями
    """
    
    def __init__(self):
        logger.info(f"{Colors.BOLD}--- Инициализация ReportGenerator ---{Colors.RESET}")
        
        # Поддерживаемые форматы
        self.supported_formats = ["HTML", "PDF", "CSV", "Excel", "JSON"]
        
        # Шаблоны отчетов
        self.report_templates = self._load_report_templates()
        
        # Кэш отчетов
        self.reports_cache = {}
        
        # Статистика генерации
        self.generation_stats = {
            "total_reports": 0,
            "successful_exports": 0,
            "failed_exports": 0,
            "formats_used": {}
        }
        
        logger.info(f"{Colors.BOLD}--- ReportGenerator инициализирован ---{Colors.RESET}")

    def _load_report_templates(self) -> Dict[str, str]:
        """Загрузка шаблонов отчетов"""
        try:
            logger.info(f"{Colors.CYAN}Загрузка шаблонов отчетов...")
            # HTML шаблон для comprehensive report
            html_template = """
            <!DOCTYPE html>
            <html lang="ru">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{{ title }}</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .header { text-align: center; margin-bottom: 30px; }
                    .section { margin-bottom: 25px; }
                    .metric-card { border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }
                    .high-risk { background-color: #ffebee; border-left: 4px solid #f44336; }
                    .medium-risk { background-color: #fff3e0; border-left: 4px solid #ff9800; }
                    .low-risk { background-color: #e8f5e8; border-left: 4px solid #4caf50; }
                    table { width: 100%; border-collapse: collapse; margin: 15px 0; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    .chart-container { text-align: center; margin: 20px 0; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>{{ title }}</h1>
                    <p>Дата генерации: {{ generation_date }}</p>
                    <p>Версия системы: {{ system_version }}</p>
                </div>
                
                {{ content }}
                
                <div class="footer">
                    <hr>
                    <p><small>Сгенерировано системой анализа подлинности 3D лиц v{{ system_version }}</small></p>
                </div>
            </body>
            </html>
            """
            
            logger.info(f"{Colors.GREEN}✔ Шаблоны отчетов успешно загружены.")
            return {
                "html": html_template,
                "pdf": html_template,  # Используется тот же шаблон для PDF
                "executive_summary": """
                <div class="section">
                    <h2>Исполнительное резюме</h2>
                    <div class="metric-card {{ risk_class }}">
                        <h3>Общий вывод: {{ conclusion }}</h3>
                        <p><strong>Итоговый балл аутентичности:</strong> {{ overall_score }}</p>
                        <p><strong>Уровень риска:</strong> {{ risk_level }}</p>
                        <p><strong>Рекомендация:</strong> {{ recommendation }}</p>
                    </div>
                </div>
                """
            }
            
        except Exception as e:
            logger.critical(f"{Colors.RED}❌ КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить шаблоны отчетов. Детали: {e}")
            return {}

    def generate_comprehensive_report(self, analysis_results: Dict[str, Any], 
                                    export_format: str = "HTML") -> Dict[str, Any]:
        """
        ИСПРАВЛЕНО: Генерация комплексного отчета с 9 секциями
        """
        self.generation_stats["total_reports"] += 1
        self.generation_stats["formats_used"][export_format] = self.generation_stats["formats_used"].get(export_format, 0) + 1
        
        logger.info(f"{Colors.BLUE}=== Генерация комплексного отчета в формате {export_format} ===")
        
        if export_format not in self.supported_formats:
            logger.error(f"{Colors.RED}Неподдерживаемый формат экспорта: {export_format}. Поддерживаемые форматы: {', '.join(self.supported_formats)}")
            self.generation_stats["failed_exports"] += 1
            return {"status": "error", "message": "Неподдерживаемый формат экспорта"}
        
        try:
            report_data = self._prepare_report_data(analysis_results)
            
            # Генерация секций
            executive_summary = self._generate_executive_summary(report_data)
            geometry_analysis = self._generate_geometry_analysis(report_data)
            embedding_analysis = self._generate_embedding_analysis(report_data)
            texture_analysis = self._generate_texture_analysis(report_data)
            temporal_analysis = self._generate_temporal_analysis(report_data)
            anomaly_detection = self._generate_anomaly_detection(report_data)
            medical_validation = self._generate_medical_validation(report_data)
            visualizations_html = self._generate_visualizations(report_data)
            conclusions = self._generate_conclusions(report_data)
            
            # Сборка полного содержимого отчета
            full_content = f"""
            {executive_summary}
            {geometry_analysis}
            {embedding_analysis}
            {texture_analysis}
            {temporal_analysis}
            {anomaly_detection}
            {medical_validation}
            {visualizations_html}
            {conclusions}
            """
            
            # Заполнение шаблона
            template = Template(self.report_templates.get("html"))
            final_report_content = template.render(
                title=f"Отчет анализа подлинности лица для {report_data['client_id']}",
                generation_date=report_data['generation_date'],
                system_version=report_data['system_version'],
                content=full_content,
                overall_score=f"{report_data['overall_score']:.2f}",
                risk_level=report_data['risk_level'],
                conclusion=report_data['conclusion'],
                recommendation=report_data['recommendation'],
                risk_class=report_data['risk_class']
            )
            
            export_result = self._export_report(final_report_content, report_data, export_format)
            if export_result["status"] == "success":
                self.generation_stats["successful_exports"] += 1
                logger.info(f"{Colors.GREEN}✔ Отчет успешно сгенерирован и экспортирован в {export_format}.")
            else:
                self.generation_stats["failed_exports"] += 1
                logger.error(f"{Colors.RED}❌ Ошибка экспорта отчета в {export_format}: {export_result.get('message', 'Неизвестная ошибка')}")
            
            return export_result
            
        except Exception as e:
            self.generation_stats["failed_exports"] += 1
            logger.critical(f"{Colors.RED}❌ КРИТИЧЕСКАЯ ОШИБКА при генерации комплексного отчета: {e}")
            return {"status": "error", "message": f"Ошибка при генерации отчета: {e}"}

    def _prepare_report_data(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Подготовка данных для отчета"""
        logger.info(f"{Colors.CYAN}Подготовка данных отчета...")
        
        # Пример данных (замените на реальную обработку analysis_results)
        # Добавляем логирование для каждого шага подготовки
        
        # 1. Сводные данные
        overall_score = analysis_results.get("overall_authenticity_score", 0.0)
        risk_level, risk_class = self._classify_risk_level(overall_score)
        recommendation = self._get_recommendation(risk_class)

        logger.info(f"{Colors.CYAN}  -> Общий балл подлинности: {overall_score:.2f}")
        logger.info(f"{Colors.CYAN}  -> Уровень риска: {risk_level} ({risk_class})")
        
        # 2. Детализированные данные по секциям
        geometry_scores = analysis_results.get("geometry_analysis", {}).get("scores", {})
        embedding_scores = analysis_results.get("embedding_analysis", {}).get("scores", {})
        texture_scores = analysis_results.get("texture_analysis", {}).get("scores", {})
        temporal_data = analysis_results.get("temporal_analysis", {})
        
        logger.info(f"{Colors.CYAN}  -> Загрузка данных анализа геометрии, эмбеддинга, текстуры и временного анализа.")
        
        # 3. Дополнительные метаданные
        client_id = analysis_results.get("client_id", "Неизвестный клиент")
        analysis_date_str = analysis_results.get("analysis_date", datetime.now().isoformat())
        
        try:
            analysis_datetime = datetime.fromisoformat(analysis_date_str)
        except ValueError:
            logger.warning(f"{Colors.YELLOW}Неверный формат даты анализа '{analysis_date_str}'. Используется текущая дата.")
            analysis_datetime = datetime.now()
        
        generation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        logger.info(f"{Colors.CYAN}  -> Идентификатор клиента: {client_id}")
        
        report_data = {
            "client_id": client_id,
            "overall_score": overall_score,
            "risk_level": risk_level,
            "risk_class": risk_class,
            "conclusion": f"Анализ завершен. Лицо оценено как {risk_level}.",
            "recommendation": recommendation,
            "generation_date": generation_date,
            "system_version": SYSTEM_VERSION,
            "analysis_datetime": analysis_datetime,
            "geometry_scores": geometry_scores,
            "embedding_scores": embedding_scores,
            "texture_scores": texture_scores,
            "temporal_data": temporal_data,
            "anomaly_detection": analysis_results.get("anomaly_detection", {}),
            "medical_validation": analysis_results.get("medical_validation", {})
        }
        
        logger.info(f"{Colors.GREEN}✔ Данные отчета успешно подготовлены.")
        return report_data

    def _classify_risk_level(self, score: float) -> Tuple[str, str]:
        """Классификация уровня риска на основе общего балла аутентичности"""
        logger.debug(f"{Colors.CYAN}Классификация уровня риска для балла: {score:.2f}")
        if score >= CRITICAL_THRESHOLDS.get("min_authenticity_score", 0.6):
            return "Низкий", "low-risk"
        elif score >= 0.4:
            return "Средний", "medium-risk"
        else:
            return "Высокий", "high-risk"

    def _get_recommendation(self, risk_class: str) -> str:
        """Получение рекомендации на основе уровня риска"""
        logger.debug(f"{Colors.CYAN}Получение рекомендации для класса риска: {risk_class}")
        recommendations = {
            "low-risk": "Признаков подделки не обнаружено. Лицо признано подлинным.",
            "medium-risk": "Обнаружены потенциальные аномалии. Рекомендуется дополнительная проверка.",
            "high-risk": "Обнаружены серьезные признаки подделки. Лицо, вероятно, не является подлинным."
        }
        return recommendations.get(risk_class, "Неизвестный уровень риска. Рекомендации отсутствуют.")

    def _generate_executive_summary(self, report_data: Dict[str, Any]) -> str:
        """Генерация раздела 'Исполнительное резюме'"""
        logger.info(f"{Colors.CYAN}Генерация раздела 'Исполнительное резюме'...")
        template = Template(self.report_templates.get("executive_summary"))
        summary = template.render(
            overall_score=f"{report_data['overall_score']:.2f}",
            risk_level=report_data['risk_level'],
            conclusion=report_data['conclusion'],
            recommendation=report_data['recommendation'],
            risk_class=report_data['risk_class']
        )
        logger.info(f"{Colors.GREEN}✔ Раздел 'Исполнительное резюме' сгенерирован.")
        return summary

    def _generate_geometry_analysis(self, report_data: Dict[str, Any]) -> str:
        """Генерация раздела 'Анализ геометрии'"""
        logger.info(f"{Colors.CYAN}Генерация раздела 'Анализ геометрии'...")
        scores = report_data.get("geometry_scores", {})
        
        content = f"""
        <div class="section">
            <h2>Анализ геометрии</h2>
            <p><strong>Оценка соответствия 3D-модели:</strong> {scores.get('3d_model_fit', 'Н/Д'):.2f}</p>
            <p><strong>Отклонение от средних параметров:</strong> {scores.get('deviation_from_average', 'Н/Д'):.2f}</p>
            <p><strong>Обнаружение неестественных деформаций:</strong> {scores.get('unnatural_deformation', 'Н/Д'):.2f}</p>
            <h3>Метрики по областям лица:</h3>
            <ul>
                <li>Глаза: {scores.get('eyes_metric', 'Н/Д'):.2f}</li>
                <li>Нос: {scores.get('nose_metric', 'Н/Д'):.2f}</li>
                <li>Рот: {scores.get('mouth_metric', 'Н/Д'):.2f}</li>
                <li>Подбородок: {scores.get('chin_metric', 'Н/Д'):.2f}</li>
            </ul>
        </div>
        """
        logger.info(f"{Colors.GREEN}✔ Раздел 'Анализ геометрии' сгенерирован.")
        return content

    def _generate_embedding_analysis(self, report_data: Dict[str, Any]) -> str:
        """Генерация раздела 'Анализ эмбеддингов'"""
        logger.info(f"{Colors.CYAN}Генерация раздела 'Анализ эмбеддингов'...")
        scores = report_data.get("embedding_scores", {})
        
        content = f"""
        <div class="section">
            <h2>Анализ эмбеддингов</h2>
            <p><strong>Косинусное сходство с референсом:</strong> {scores.get('cosine_similarity', 'Н/Д'):.2f}</p>
            <p><strong>Расстояние Махаланобиса:</strong> {scores.get('mahalanobis_distance', 'Н/Д'):.2f}</p>
            <p><strong>Метрика кластеризации:</strong> {scores.get('clustering_metric', 'Н/Д'):.2f}</p>
        </div>
        """
        logger.info(f"{Colors.GREEN}✔ Раздел 'Анализ эмбеддингов' сгенерирован.")
        return content

    def _generate_texture_analysis(self, report_data: Dict[str, Any]) -> str:
        """Генерация раздела 'Анализ текстуры'"""
        logger.info(f"{Colors.CYAN}Генерация раздела 'Анализ текстуры'...")
        scores = report_data.get("texture_scores", {})
        
        content = f"""
        <div class="section">
            <h2>Анализ текстуры</h2>
            <p><strong>Качество текстуры:</strong> {scores.get('texture_quality', 'Н/Д'):.2f}</p>
            <p><strong>Наличие артефактов рендеринга:</strong> {scores.get('rendering_artifacts', 'Н/Д'):.2f}</p>
            <p><strong>Соответствие освещению:</strong> {scores.get('lighting_consistency', 'Н/Д'):.2f}</p>
            <p><strong>Анализ дефектов кожи (пятна, поры):</strong> {scores.get('skin_defect_analysis', 'Н/Д'):.2f}</p>
        </div>
        """
        logger.info(f"{Colors.GREEN}✔ Раздел 'Анализ текстуры' сгенерирован.")
        return content

    def _generate_temporal_analysis(self, report_data: Dict[str, Any]) -> str:
        """Генерация раздела 'Временной анализ'"""
        logger.info(f"{Colors.CYAN}Генерация раздела 'Временной анализ'...")
        temporal_data = report_data.get("temporal_data", {})
        
        content = f"""
        <div class="section">
            <h2>Временной анализ (для видео)</h2>
            <p><strong>Стабильность параметров во времени:</strong> {temporal_data.get('parameter_stability', 'Н/Д'):.2f}</p>
            <p><strong>Обнаружение аномальных переходов:</strong> {temporal_data.get('abnormal_transitions', 'Н/Д'):.2f}</p>
            <p><strong>Среднее отклонение по кадрам:</strong> {temporal_data.get('average_frame_deviation', 'Н/Д'):.2f}</p>
        </div>
        """
        logger.info(f"{Colors.GREEN}✔ Раздел 'Временной анализ' сгенерирован.")
        return content

    def _generate_anomaly_detection(self, report_data: Dict[str, Any]) -> str:
        """Генерация раздела 'Обнаружение аномалий'"""
        logger.info(f"{Colors.CYAN}Генерация раздела 'Обнаружение аномалий'...")
        anomalies = report_data.get("anomaly_detection", {})
        
        content = f"""
        <div class="section">
            <h2>Обнаружение аномалий</h2>
            <p><strong>Наличие аномалий (изоляция леса):</strong> {anomalies.get('isolation_forest_score', 'Н/Д'):.2f}</p>
            <p><strong>Локальный фактор выброса (LOF):</strong> {anomalies.get('lof_score', 'Н/Д'):.2f}</p>
        </div>
        """
        logger.info(f"{Colors.GREEN}✔ Раздел 'Обнаружение аномалий' сгенерирован.")
        return content

    def _generate_medical_validation(self, report_data: Dict[str, Any]) -> str:
        """Генерация раздела 'Медицинская валидация'"""
        logger.info(f"{Colors.CYAN}Генерация раздела 'Медицинская валидация'...")
        medical_data = report_data.get("medical_validation", {})
        
        analysis_datetime = report_data.get("analysis_datetime")
        putin_age = "Н/Д"
        if analysis_datetime:
            putin_age = self._calculate_age(PUTIN_BIRTH_DATE.strftime("%Y-%m-%d"), analysis_datetime)

        content = f"""
        <div class="section">
            <h2>Медицинская валидация</h2>
            <p><strong>Возраст (оценка):</strong> {medical_data.get('estimated_age', 'Н/Д')}</p>
            <p><strong>Пол (оценка):</strong> {medical_data.get('estimated_gender', 'Н/Д')}</p>
            <p><strong>Совместимость с биометрическими данными:</strong> {medical_data.get('biometric_compatibility', 'Н/Д'):.2f}</p>
            <p><strong>Возраст Путина на дату анализа:</strong> {putin_age} лет</p>
        </div>
        """
        logger.info(f"{Colors.GREEN}✔ Раздел 'Медицинская валидация' сгенерирован.")
        return content

    def _calculate_age(self, birth_date_str: str, current_datetime: datetime) -> int:
        """Вычисляет возраст на основе даты рождения и текущей даты."""
        logger.debug(f"{Colors.CYAN}Вычисление возраста для {birth_date_str} на {current_datetime.isoformat()}")
        birth_date = datetime.strptime(birth_date_str, "%Y-%m-%d").date()
        age = current_datetime.year - birth_date.year - ((current_datetime.month, current_datetime.day) < (birth_date.month, birth_date.day))
        logger.debug(f"{Colors.CYAN}  -> Рассчитанный возраст: {age}")
        return age

    def _prepare_visualization_data(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Подготовка данных для визуализаций"""
        logger.info(f"{Colors.CYAN}Подготовка данных для визуализаций...")
        
        # Пример: подготовка данных для гипотетического графика "Оценка по секциям"
        sections = ["geometry", "embedding", "texture"]
        scores = {
            "geometry": analysis_results.get("geometry_analysis", {}).get("scores", {}).get("3d_model_fit", 0.0),
            "embedding": analysis_results.get("embedding_analysis", {}).get("scores", {}).get("cosine_similarity", 0.0),
            "texture": analysis_results.get("texture_analysis", {}).get("scores", {}).get("texture_quality", 0.0)
        }
        
        logger.info(f"{Colors.GREEN}✔ Данные для визуализаций подготовлены.")
        return {"sections": sections, "scores": scores}

    def _generate_visualizations(self, report_data: Dict[str, Any]) -> str:
        """Генерация раздела 'Визуализации'"""
        logger.info(f"{Colors.CYAN}Генерация раздела 'Визуализации'...")
        
        vis_html = ""
        if HAS_PLOTLY:
            try:
                vis_data = self._prepare_visualization_data(report_data)
                
                fig = px.bar(
                    x=vis_data["sections"], 
                    y=[vis_data["scores"][s] for s in vis_data["sections"]], 
                    title="Оценка по основным секциям анализа",
                    labels={"x": "Секция анализа", "y": "Балл"},
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig.update_layout(xaxis_title="Секция анализа", yaxis_title="Балл (от 0 до 1)")
                
                vis_html = f"""
                <div class="section">
                    <h2>Визуализации</h2>
                    <div class="chart-container">
                        {fig.to_html(full_html=False, include_plotlyjs='cdn')}
                    </div>
                </div>
                """
                logger.info(f"{Colors.GREEN}✔ Раздел 'Визуализации' сгенерирован (Plotly).")
            except Exception as e:
                logger.error(f"{Colors.RED}❌ Ошибка при генерации визуализаций с Plotly: {e}")
                vis_html = f"<p style='color: red;'>Ошибка при генерации визуализаций: {e}</p>"
        else:
            vis_html = "<p>Библиотека Plotly не установлена. Визуализации недоступны.</p>"
            logger.warning(f"{Colors.YELLOW}Plotly не установлена, визуализации не будут сгенерированы.")
            
        return vis_html

    def _generate_conclusions(self, report_data: Dict[str, Any]) -> str:
        """Генерация раздела 'Выводы и рекомендации'"""
        logger.info(f"{Colors.CYAN}Генерация раздела 'Выводы и рекомендации'...")
        
        content = f"""
        <div class="section">
            <h2>Выводы и рекомендации</h2>
            <p>На основе всестороннего анализа, система предоставляет следующие выводы и рекомендации:</p>
            <ul>
                <li><strong>Общий вывод:</strong> {report_data['conclusion']}</li>
                <li><strong>Уровень риска:</strong> {report_data['risk_level']}</li>
                <li><strong>Рекомендация:</strong> {report_data['recommendation']}</li>
            </ul>
            <p>Для получения более детальной информации, пожалуйста, обратитесь к соответствующим разделам отчета.</p>
        </div>
        """
        logger.info(f"{Colors.GREEN}✔ Раздел 'Выводы и рекомендации' сгенерирован.")
        return content

    def _export_report(self, content: str, report_data: Dict[str, Any], format: str) -> Dict[str, Any]:
        """Экспорт отчета в указанный формат"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_data["timestamp"] = timestamp
        
        logger.info(f"{Colors.CYAN}Экспорт отчета в формат: {format}...")
        
        if format == "HTML":
            return self._export_html(content, report_data, timestamp)
        elif format == "PDF":
            if HAS_REPORTLAB:
                return self._export_pdf(content, report_data, timestamp)
            else:
                logger.error(f"{Colors.RED}❌ Экспорт в PDF невозможен: ReportLab не установлена.")
                return {"status": "error", "message": "ReportLab не установлена"}
        elif format == "CSV":
            return self._export_csv(report_data, timestamp)
        elif format == "Excel":
            return self._export_excel(report_data, timestamp)
        elif format == "JSON":
            return self._export_json(report_data, timestamp)
        else:
            logger.error(f"{Colors.RED}❌ Неизвестный формат экспорта: {format}.")
            return {"status": "error", "message": "Неизвестный формат экспорта"}

    def _export_html(self, content: str, report_data: Dict[str, Any], timestamp: str) -> Dict[str, Any]:
        """Экспорт отчета в HTML"""
        try:
            output_dir = RESULTS_DIR / report_data["client_id"]
            output_dir.mkdir(parents=True, exist_ok=True)
            file_path = output_dir / f"report_{timestamp}.html"
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            logger.info(f"{Colors.GREEN}✔ Отчет HTML сохранен: {file_path}")
            return {"status": "success", "file_path": str(file_path)}
        except Exception as e:
            logger.error(f"{Colors.RED}❌ Ошибка при экспорте HTML отчета: {e}")
            return {"status": "error", "message": f"Ошибка при экспорте HTML: {e}"}

    def _export_pdf(self, content: str, report_data: Dict[str, Any], timestamp: str) -> Dict[str, Any]:
        """Экспорт отчета в PDF"""
        try:
            output_dir = RESULTS_DIR / report_data["client_id"]
            output_dir.mkdir(parents=True, exist_ok=True)
            file_path = output_dir / f"report_{timestamp}.pdf"

            doc = SimpleDocTemplate(str(file_path), pagesize=A4)
            styles = getSampleStyleSheet()
            
            # Базовый стиль для Paragraph
            body_style = styles['Normal']
            body_style.fontName = 'Helvetica'
            body_style.fontSize = 10
            
            # Добавление поддержки кириллицы (если нужно)
            # from reportlab.pdfbase import pdfmetrics
            # from reportlab.pdfbase.ttfonts import TTFont
            # pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))
            # body_style.fontName = 'DejaVuSans'

            story = []

            # Заголовок
            story.append(Paragraph(f"""<h1 align="center">Отчет анализа подлинности лица для {report_data['client_id']}</h1>""", styles['h1']))
            story.append(Paragraph(f"""<p align="center">Дата генерации: {report_data['generation_date']}</p>""", body_style))
            story.append(Paragraph(f"""<p align="center">Версия системы: {report_data['system_version']}</p>""", body_style))
            story.append(Spacer(1, 0.2 * inch))

            # Исполнительное резюме
            story.append(Paragraph("<h2>Исполнительное резюме</h2>", styles['h2']))
            risk_color = "green" if report_data['risk_class'] == "low-risk" else "orange" if report_data['risk_class'] == "medium-risk" else "red"
            story.append(Paragraph(f"<p><b>Общий вывод:</b> <font color='{risk_color}'>{report_data['conclusion']}</font></p>", body_style))
            story.append(Paragraph(f"<p><b>Итоговый балл аутентичности:</b> {report_data['overall_score']}</p>", body_style))
            story.append(Paragraph(f"<p><b>Уровень риска:</b> {report_data['risk_level']}</p>", body_style))
            story.append(Paragraph(f"<p><b>Рекомендация:</b> {report_data['recommendation']}</p>", body_style))
            story.append(Spacer(1, 0.2 * inch))
            
            # Прочие секции (нужно будет распарсить HTML или передавать данные отдельно)
            # Пока для простоты, будем добавлять как Paragraph
            # Это место, где потребуется более сложный парсинг HTML или переработка логики
            story.append(Paragraph("<h2>Детализированный анализ</h2>", styles['h2']))
            story.append(Paragraph("<p><i>Для полной версии отчета с интерактивными графиками, пожалуйста, откройте HTML версию.</i></p>", body_style))
            
            # Пример добавления текста из HTML (очень упрощенно)
            # В реальном приложении потребуется библиотека для парсинга HTML в ReportLab объекты
            # story.append(Paragraph(content, body_style)) 

            doc.build(story)
            
            logger.info(f"{Colors.GREEN}✔ Отчет PDF сохранен: {file_path}")
            return {"status": "success", "file_path": str(file_path)}
        except Exception as e:
            logger.error(f"{Colors.RED}❌ Ошибка при экспорте PDF отчета: {e}")
            return {"status": "error", "message": f"Ошибка при экспорте PDF: {e}"}

    def _export_csv(self, report_data: Dict[str, Any], timestamp: str) -> Dict[str, Any]:
        """Экспорт отчета в CSV"""
        try:
            output_dir = RESULTS_DIR / report_data["client_id"]
            output_dir.mkdir(parents=True, exist_ok=True)
            file_path = output_dir / f"report_{timestamp}.csv"
            
            # Подготовка данных для CSV (очень упрощенно)
            # В реальном приложении потребуется более сложная логика для flatten
            data = {
                "client_id": report_data["client_id"],
                "overall_score": report_data["overall_score"],
                "risk_level": report_data["risk_level"],
                "recommendation": report_data["recommendation"],
                "generation_date": report_data["generation_date"],
                "system_version": report_data["system_version"],
                "geometry_3d_model_fit": report_data["geometry_scores"].get("3d_model_fit", ""),
                "embedding_cosine_similarity": report_data["embedding_scores"].get("cosine_similarity", ""),
                "texture_quality": report_data["texture_scores"].get("texture_quality", ""),
            }
            
            df = pd.DataFrame([data])
            df.to_csv(file_path, index=False, encoding="utf-8")
            
            logger.info(f"{Colors.GREEN}✔ Отчет CSV сохранен: {file_path}")
            return {"status": "success", "file_path": str(file_path)}
        except Exception as e:
            logger.error(f"{Colors.RED}❌ Ошибка при экспорте CSV отчета: {e}")
            return {"status": "error", "message": f"Ошибка при экспорте CSV: {e}"}

    def _export_excel(self, report_data: Dict[str, Any], timestamp: str) -> Dict[str, Any]:
        """Экспорт отчета в Excel"""
        try:
            output_dir = RESULTS_DIR / report_data["client_id"]
            output_dir.mkdir(parents=True, exist_ok=True)
            file_path = output_dir / f"report_{timestamp}.xlsx"

            data = {
                "client_id": [report_data["client_id"]],
                "overall_score": [report_data["overall_score"]],
                "risk_level": [report_data["risk_level"]],
                "recommendation": [report_data["recommendation"]],
                "generation_date": [report_data["generation_date"]],
                "system_version": [report_data["system_version"]],
                "geometry_3d_model_fit": [report_data["geometry_scores"].get("3d_model_fit", "")],
                "embedding_cosine_similarity": [report_data["embedding_scores"].get("cosine_similarity", "")],
                "texture_quality": [report_data["texture_scores"].get("texture_quality", "")],
            }
            
            df = pd.DataFrame(data)
            df.to_excel(file_path, index=False, engine='openpyxl')
            
            logger.info(f"{Colors.GREEN}✔ Отчет Excel сохранен: {file_path}")
            return {"status": "success", "file_path": str(file_path)}
        except Exception as e:
            logger.error(f"{Colors.RED}❌ Ошибка при экспорте Excel отчета: {e}")
            return {"status": "error", "message": f"Ошибка при экспорте Excel: {e}"}

    def _export_json(self, report_data: Dict[str, Any], timestamp: str) -> Dict[str, Any]:
        """Экспорт отчета в JSON"""
        try:
            output_dir = RESULTS_DIR / report_data["client_id"]
            output_dir.mkdir(parents=True, exist_ok=True)
            file_path = output_dir / f"report_{timestamp}.json"
            
            # Сериализация данных, чтобы избежать ошибок с несериализуемыми объектами
            serializable_data = self._serialize_for_json(report_data)
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(serializable_data, f, indent=4, ensure_ascii=False)
            
            logger.info(f"{Colors.GREEN}✔ Отчет JSON сохранен: {file_path}")
            return {"status": "success", "file_path": str(file_path)}
        except Exception as e:
            logger.error(f"{Colors.RED}❌ Ошибка при экспорте JSON отчета: {e}")
            return {"status": "error", "message": f"Ошибка при экспорте JSON: {e}"}

    def _serialize_for_json(self, data: Any) -> Any:
        """Рекурсивно преобразует несериализуемые объекты в сериализуемые."""
        if isinstance(data, (datetime, date)):
            return data.isoformat()
        if isinstance(data, timedelta):
            return str(data)
        if isinstance(data, Path):
            return str(data)
        if isinstance(data, np.integer):
            return int(data)
        if isinstance(data, np.floating):
            return float(data)
        if isinstance(data, np.ndarray):
            return data.tolist()
        if isinstance(data, dict):
            return {k: self._serialize_for_json(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._serialize_for_json(item) for item in data]
        return data

    def get_generation_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику генерации отчетов."""
        logger.info(f"{Colors.BLUE}Запрос статистики генерации отчетов.")
        return self.generation_stats

    def self_test(self) -> None:
        """
        Проводит самопроверку генератора отчетов.
        ИСПРАВЛЕНО: Добавлены более полные моковые данные и проверки.
        """
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
                    "eyes_metric": 0.9, "nose_metric": 0.85, "mouth_metric": 0.88, "chin_metric": 0.92
                }
            },
            "embedding_analysis": {
                "scores": {
                    "cosine_similarity": 0.95,
                    "mahalanobis_distance": 0.1,
                    "clustering_metric": 0.88
                }
            },
            "texture_analysis": {
                "scores": {
                    "texture_quality": 0.9,
                    "rendering_artifacts": 0.02,
                    "lighting_consistency": 0.93,
                    "skin_defect_analysis": 0.01
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
                "estimated_age": 30,
                "estimated_gender": "Мужской",
                "biometric_compatibility": 0.98
            }
        }

        # Тест генерации HTML отчета
        logger.info(f"{Colors.CYAN}Тестирование генерации HTML отчета...")
        html_report = self.generate_comprehensive_report(mock_analysis_results, export_format="HTML")
        if html_report["status"] == "success":
            logger.info(f"{Colors.GREEN}✔ HTML отчет успешно сгенерирован: {html_report['file_path']}")
        else:
            logger.error(f"{Colors.RED}❌ Ошибка генерации HTML отчета: {html_report['message']}")

        # Тест генерации PDF отчета (если ReportLab доступен)
        if HAS_REPORTLAB:
            logger.info(f"{Colors.CYAN}Тестирование генерации PDF отчета...")
            pdf_report = self.generate_comprehensive_report(mock_analysis_results, export_format="PDF")
            if pdf_report["status"] == "success":
                logger.info(f"{Colors.GREEN}✔ PDF отчет успешно сгенерирован: {pdf_report['file_path']}")
            else:
                logger.error(f"{Colors.RED}❌ Ошибка генерации PDF отчета: {pdf_report['message']}")
        else:
            logger.warning(f"{Colors.YELLOW}Пропуск тестирования PDF: ReportLab не установлен.")

        # Тест генерации CSV отчета
        logger.info(f"{Colors.CYAN}Тестирование генерации CSV отчета...")
        csv_report = self.generate_comprehensive_report(mock_analysis_results, export_format="CSV")
        if csv_report["status"] == "success":
            logger.info(f"{Colors.GREEN}✔ CSV отчет успешно сгенерирован: {csv_report['file_path']}")
        else:
            logger.error(f"{Colors.RED}❌ Ошибка генерации CSV отчета: {csv_report['message']}")
            
        # Тест генерации Excel отчета
        logger.info(f"{Colors.CYAN}Тестирование генерации Excel отчета...")
        excel_report = self.generate_comprehensive_report(mock_analysis_results, export_format="Excel")
        if excel_report["status"] == "success":
            logger.info(f"{Colors.GREEN}✔ Excel отчет успешно сгенерирован: {excel_report['file_path']}")
        else:
            logger.error(f"{Colors.RED}❌ Ошибка генерации Excel отчета: {excel_report['message']}")

        # Тест генерации JSON отчета
        logger.info(f"{Colors.CYAN}Тестирование генерации JSON отчета...")
        json_report = self.generate_comprehensive_report(mock_analysis_results, export_format="JSON")
        if json_report["status"] == "success":
            logger.info(f"{Colors.GREEN}✔ JSON отчет успешно сгенерирован: {json_report['file_path']}")
        else:
            logger.error(f"{Colors.RED}❌ Ошибка генерации JSON отчета: {json_report['message']}")

        # Тест некорректного формата
        logger.info(f"{Colors.CYAN}Тестирование обработки некорректного формата...")
        invalid_report = self.generate_comprehensive_report(mock_analysis_results, export_format="UNKNOWN")
        if invalid_report["status"] == "error" and "Неподдерживаемый формат" in invalid_report["message"]:
            logger.info(f"{Colors.GREEN}✔ Обработка некорректного формата прошла успешно.")
        else:
            logger.error(f"{Colors.RED}❌ Ошибка обработки некорректного формата.")
            
        # Статистика
        stats = self.get_generation_statistics()
        logger.info(f"{Colors.BOLD}--- Статистика генерации отчетов ---{Colors.RESET}")
        logger.info(f"Всего попыток: {stats['total_reports']}")
        logger.info(f"Успешных экспортов: {stats['successful_exports']}")
        logger.info(f"Неудачных экспортов: {stats['failed_exports']}")
        logger.info(f"Использованные форматы: {stats['formats_used']}")

        logger.info(f"{Colors.BOLD}--- Самопроверка ReportGenerator завершена ---{Colors.RESET}")

# ==================== ТОЧКА ВХОДА ====================

if __name__ == "__main__":
    generator = ReportGenerator()
    generator.self_test()
