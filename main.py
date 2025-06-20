"""
Main - Главный модуль запуска системы анализа подлинности 3D лиц
Версия: 2.0
Дата: 2025-06-15
Полная интеграция всех модулей с ОРИГИНАЛЬНЫМИ названиями
"""

import os
import sys
import logging
import asyncio
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import signal
import traceback
import json
import numpy as np
import warnings
import uuid
import psutil
import time
import math

# === ГАРАНТИРОВАННОЕ СОЗДАНИЕ ПАПКИ ДЛЯ ЛОГОВ ===
os.makedirs("logs", exist_ok=True)

# === ДОБАВЛЕНО: ColoredFormatter для цветного логирования ===
class ColoredFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: "\033[96m%(levelname)s:\033[0m %(message)s",
        logging.INFO: "\033[92m%(levelname)s:\033[0m %(message)s",
        logging.WARNING: "\033[93m%(levelname)s:\033[0m %(message)s",
        logging.ERROR: "\033[91m%(levelname)s:\033[0m %(message)s",
        logging.CRITICAL: "\033[91m\033[1m%(levelname)s:\033[0m %(message)s"
    }
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# --- ЦВЕТА КОНСОЛИ ---
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

# === ГЕНЕРАЦИЯ run_uuid ===
RUN_UUID = str(uuid.uuid4())

# === Кастомный JSONL-логгер ===
class JSONLFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "mod": record.module,
            "func": record.funcName,
            "msg": record.getMessage(),
            "duration_ms": getattr(record, "duration_ms", None),
            "mem_mb": psutil.Process().memory_info().rss // 1e6,
            "run_uuid": RUN_UUID
        }
        return json.dumps(log_record, ensure_ascii=False)

# Настройка root-логгера
logger = logging.getLogger()
for h in logger.handlers:
    logger.removeHandler(h)
file_handler = logging.FileHandler(f"logs/session_{RUN_UUID}.jsonl", mode="a", encoding="utf-8")
file_handler.setFormatter(JSONLFormatter())
logger.addHandler(file_handler)

log_file_handler = logging.FileHandler('logs/main.log')
log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'))
logger.addHandler(log_file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(ColoredFormatter())
logger.addHandler(console_handler)

logger.setLevel(logging.INFO)

# Автоматическое повышение уровня логирования при ошибках
class AutoDebugHandler(logging.Handler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_counter = 0
    def emit(self, record):
        if record.levelno >= logging.ERROR:
            logger.setLevel(logging.DEBUG)
            self.debug_counter = 20
        elif self.debug_counter > 0:
            self.debug_counter -= 1
            if self.debug_counter == 0:
                logger.setLevel(logging.INFO)

logger.addHandler(AutoDebugHandler())

# ИСПРАВЛЕНО: Импорт всех модулей с ОРИГИНАЛЬНЫМИ названиями
try:
    from core_config import (
        validate_configuration, save_configuration_snapshot,
        PUTIN_BIRTH_DATE, START_ANALYSIS_DATE, END_ANALYSIS_DATE,
        DATA_DIR, RESULTS_DIR, CACHE_DIR, LOGS_DIR, CONFIG_DIR,
        MAX_FILE_UPLOAD_COUNT, ERROR_CODES, SYSTEM_VERSION,
        AUTHENTICITY_WEIGHTS, CRITICAL_THRESHOLDS, MIN_VISIBILITY_Z
    )
    print("DEBUG: Импортирован core_config")
    from face_3d_analyzer import Face3DAnalyzer, initialize_3ddfa_components
    print("DEBUG: Импортирован face_3d_analyzer")
    from embedding_analyzer import EmbeddingAnalyzer
    print("DEBUG: Импортирован embedding_analyzer")
    from texture_analyzer import TextureAnalyzer
    print("DEBUG: Импортирован texture_analyzer")
    from temporal_analyzer import TemporalAnalyzer
    print("DEBUG: Импортирован temporal_analyzer")
    from anomaly_detector import AnomalyDetector
    print("DEBUG: Импортирован anomaly_detector")
    from medical_validator import MedicalValidator
    print("DEBUG: Импортирован medical_validator")
    from data_manager import DataManager
    print("DEBUG: Импортирован data_manager")
    from metrics_calculator import MetricsCalculator
    print("DEBUG: Импортирован metrics_calculator")
    from data_processing import DataProcessor, ResultsAggregator, AnalysisResult
    print("DEBUG: Импортирован data_processing")
    from gradio_interface import GradioInterface, create_modular_interface
    print("DEBUG: Импортирован gradio_interface")
    from visualization_engine import VisualizationEngine
    print("DEBUG: Импортирован visualization_engine")
    from report_generator import ReportGenerator
    print("DEBUG: Импортирован report_generator")
    print("DEBUG: Все модули системы успешно импортированы с оригинальными названиями")
except ImportError as e:
    print(f"CRITICAL ERROR: Не удалось импортировать модули системы: {e}")
    print(traceback.format_exc())
    sys.exit(1)

# ==================== КОНСТАНТЫ СИСТЕМЫ ====================

SYSTEM_NAME = "Система анализа подлинности 3D лиц"
AUTHOR = "AI Research Team"
BUILD_DATE = "2025-06-15"

# Режимы работы системы
OPERATION_MODES = {
    "gui": "Графический интерфейс Gradio",
    "cli": "Командная строка",
    "api": "API сервер",
    "batch": "Пакетная обработка",
    "test": "Режим тестирования"
}

# ==================== ГЛАВНЫЙ КЛАСС СИСТЕМЫ ====================

class FaceAuthenticitySystem:
    """
    Главный класс системы анализа подлинности 3D лиц
    ИСПРАВЛЕНО: Интеграция с оригинальными названиями модулей
    """
    
    def __init__(self, run_self_tests: bool = True):
        logger.info(f"{Colors.BOLD}--- Инициализация системы '{SYSTEM_NAME}' v{SYSTEM_VERSION} ---{Colors.RESET}")
        
        # Флаги состояния системы
        self.initialized = False
        self.running = False
        self.shutdown_requested = False
        
        # Компоненты системы
        self.components = {}
        
        # Статистика работы
        self.session_stats = {
            "start_time": datetime.now(),
            "files_processed": 0,
            "analyses_completed": 0,
            "errors_encountered": 0,
            "uptime_seconds": 0
        }
        
        # Инициализация компонентов
        self._initialize_system_components(run_self_tests)
        
        logger.info(f"{Colors.BOLD}--- Система успешно инициализирована и готова к работе ---{Colors.RESET}")

    def _initialize_system_components(self, run_self_tests: bool) -> None:
        """ИСПРАВЛЕНО: Инициализация всех компонентов с оригинальными названиями"""
        try:
            logger.info(f"\n{Colors.CYAN}### Шаг 1: Инициализация компонентов системы...{Colors.RESET}")
            
            # 1. Валидация конфигурации
            logger.info("Проверка конфигурации...")
            if not validate_configuration():
                logger.critical("КРИТИЧЕСКАЯ ОШИБКА: Конфигурация системы содержит ошибки! Исправьте перед запуском.")
                raise RuntimeError("Конфигурация системы содержит ошибки")
            logger.info("Конфигурация в порядке.")
            
            # 2. Создание необходимых директорий
            logger.info("Проверка и создание необходимых папок...")
            self._ensure_directories_exist()
            logger.info("Все нужные папки созданы.")
            
            # 3. ИСПРАВЛЕНО: data_manager (первым)
            logger.info("Инициализация 'Менеджера данных' (data_manager)...")
            self.components['data_manager'] = DataManager()
            
            # 4. ИСПРАВЛЕНО: metrics_calculator
            logger.info("Инициализация 'Расчетчика метрик' (metrics_calculator)...")
            self.components['metrics_calculator'] = MetricsCalculator()
            
            # 5. ИСПРАВЛЕНО: face_3d_analyzer
            logger.info("Инициализация '3D Анализатора лица' (face_3d_analyzer)...")
            from face_3d_analyzer import initialize_3ddfa_components
            tddfa_models = initialize_3ddfa_components(str(CONFIG_DIR), skip_gpu_check=run_self_tests)
            self.components['face_3d_analyzer'] = Face3DAnalyzer(tddfa_model=tddfa_models['tddfa'], face_boxes_model=tddfa_models['face_boxes'])
            
            # 6. ИСПРАВЛЕНО: embedding_analyzer
            logger.info("Инициализация 'Анализатора эмбеддингов' (embedding_analyzer)...")
            # Инициализируем EmbeddingAnalyzer без face_app, он инициализируется внутри себя
            temp_embedding_analyzer = EmbeddingAnalyzer()
            # Затем получаем инициализированный face_app из него
            insightface_app_instance = temp_embedding_analyzer.face_app
            # И теперь инициализируем финальный экземпляр, передавая ему face_app
            self.components['embedding_analyzer'] = EmbeddingAnalyzer(face_app=insightface_app_instance)
            
            # 7. ИСПРАВЛЕНО: texture_analyzer
            logger.info("Инициализация 'Анализатора текстуры' (texture_analyzer)...")
            self.components['texture_analyzer'] = TextureAnalyzer()
            
            # 8. ИСПРАВЛЕНО: temporal_analyzer
            logger.info("Инициализация 'Временного анализатора' (temporal_analyzer)...")
            self.components['temporal_analyzer'] = TemporalAnalyzer()
            
            # 9. ИСПРАВЛЕНО: anomaly_detector
            logger.info("Инициализация 'Детектора аномалий' (anomaly_detector)...")
            self.components['anomaly_detector'] = AnomalyDetector()
            
            # 10. ИСПРАВЛЕНО: medical_validator
            logger.info("Инициализация 'Медицинского валидатора' (medical_validator)...")
            self.components['medical_validator'] = MedicalValidator()
            
            # 11. ИСПРАВЛЕНО: data_processing (DataProcessor)
            logger.info("Инициализация 'Обработчика данных' (data_processor)...")
            self.components['data_processor'] = DataProcessor(
                face_analyzer=self.components['face_3d_analyzer'],
                embedding_analyzer=self.components['embedding_analyzer'],
                texture_analyzer=self.components['texture_analyzer'],
                temporal_analyzer=self.components['temporal_analyzer'],
                data_manager=self.components['data_manager'],
                metrics_calculator=self.components['metrics_calculator']
            )
            
            # 12. ИСПРАВЛЕНО: visualization_engine
            logger.info("Инициализация 'Движка визуализации' (visualization_engine)...")
            self.components['visualization_engine'] = VisualizationEngine()
            
            # 13. ИСПРАВЛЕНО: report_generator
            logger.info("Инициализация 'Генератора отчетов' (report_generator)...")
            self.components['report_generator'] = ReportGenerator()
            
            # 14. ИСПРАВЛЕНО: ResultsAggregator из data_processing
            logger.info("Инициализация 'Агрегатора результатов' (results_aggregator)...")
            self.components['results_aggregator'] = ResultsAggregator()
            
            # 17. ИСПРАВЛЕНО: gradio_interface (последним)
            logger.info("Инициализация 'Интерфейса Gradio' (gradio_interface)...")
            print("=== gradio_interface импортирован ===")
            self.components['gradio_interface'] = create_modular_interface(
                all_system_components=self.components
            )
            print("=== интерфейс создан ===")
            
            logger.info(f"\n{Colors.CYAN}### Шаг 2: Загрузка кэшей системы...{Colors.RESET}")
            # Загрузка кэшей
            self._load_system_caches()
            
            if run_self_tests:
                logger.info(f"\n{Colors.CYAN}### Шаг 3: Запуск самотестирования компонентов...{Colors.RESET}")
                self._run_component_self_tests()
            else:
                logger.info(f"\n{Colors.YELLOW}### Шаг 3: Самотестирование компонентов пропущено (режим GUI).{Colors.RESET}")
            
            self.initialized = True
            logger.info(f"\n{Colors.BOLD}{Colors.GREEN}✔ Все компоненты системы успешно инициализированы!{Colors.RESET}")
            
        except Exception as e:
            logger.critical(f"{Colors.RED}КРИТИЧЕСКАЯ ОШИБКА инициализации системы: {e}{Colors.RESET}")
            logger.debug(f"Трассировка ошибки: {traceback.format_exc()}")
            raise

    def _ensure_directories_exist(self) -> None:
        """Создание необходимых директорий"""
        try:
            directories = [DATA_DIR, RESULTS_DIR, CACHE_DIR, LOGS_DIR]
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Директория создана/проверена: {directory}")
            
            logger.info("Все необходимые директории созданы.")
            
        except Exception as e:
            logger.critical(f"{Colors.RED}ОШИБКА: Не удалось создать нужные папки: {e}{Colors.RESET}")
            raise

    def _load_system_caches(self) -> None:
        """Загрузка кэшей всех компонентов"""
        try:
            logger.info("Загрузка кэшей компонентов...")
            
            # ИСПРАВЛЕНО: Загрузка кэшей с оригинальными названиями
            cache_methods = [
                ('data_manager', 'load_cache'),
                ('embedding_analyzer', 'load_cache'),
                ('texture_analyzer', 'load_analysis_cache'),
                ('temporal_analyzer', 'load_analysis_cache'),
                ('anomaly_detector', 'load_analysis_cache'),
                ('medical_validator', 'load_validation_cache'),
                ('visualization_engine', 'load_plot_cache'),
                ('data_processor', 'load_results_from_cache')
            ]
            
            for component_name, method_name in cache_methods:
                if component_name in self.components:
                    component = self.components[component_name]
                    if hasattr(component, method_name):
                        try:
                            getattr(component, method_name)()
                            logger.debug(f"Кэш загружен для {component_name}")
                        except Exception as e:
                            logger.warning(f"{Colors.YELLOW}ПРЕДУПРЕЖДЕНИЕ: Ошибка загрузки кэша для '{component_name}': {e}{Colors.RESET}")
            
            logger.info(f"{Colors.GREEN}✔ Загрузка кэшей завершена.{Colors.RESET}")
            
        except Exception as e:
            logger.critical(f"{Colors.RED}КРИТИЧЕСКАЯ ОШИБКА при загрузке кэшей: {e}{Colors.RESET}")
            raise

    def _run_component_self_tests(self) -> None:
        """Запуск самотестирования всех компонентов"""
        try:
            logger.info("Запуск самотестирования компонентов...")
            
            test_results = {}
            
            for component_name, component in self.components.items():
                if hasattr(component, 'self_test'):
                    try:
                        logger.info(f"Тестирование '{component_name}'...")
                        component.self_test()
                        test_results[component_name] = "PASSED"
                        logger.info(f"{Colors.GREEN}✔ Тест '{component_name}' пройден.{Colors.RESET}")
                    except Exception as e:
                        logger.error(f"{Colors.RED}ОШИБКА: Тест '{component_name}' не пройден: {e}{Colors.RESET}")
                        test_results[component_name] = f"FAILED: {str(e)}"
                else:
                    test_results[component_name] = "NO_TEST_METHOD"
                    logger.debug(f"Для '{component_name}' нет метода самотестирования.")
            
            # Вывод результатов тестирования
            logger.info(f"\n{Colors.CYAN}### Результаты самотестирования всех компонентов:{Colors.RESET}")
            for component, result in test_results.items():
                if result == "PASSED":
                    logger.info(f"  {Colors.GREEN}✔ {component}: Пройдено{Colors.RESET}")
                elif result.startswith("FAILED"):
                    logger.error(f"  {Colors.RED}✖ {component}: ОШИБКА ({result.split(':', 1)[1].strip()}){Colors.RESET}")
                else:
                    logger.info(f"  {Colors.YELLOW}- {component}: Без теста{Colors.RESET}")
            
            # Проверка критических компонентов
            critical_components = ['face_3d_analyzer', 'embedding_analyzer', 'data_manager', 'metrics_calculator']
            failed_critical = [comp for comp in critical_components 
                             if test_results.get(comp, "").startswith("FAILED")]
            
            if failed_critical:
                logger.warning(f"{Colors.YELLOW}ПРЕДУПРЕЖДЕНИЕ: Некоторые критические компоненты не прошли тестирование: {', '.join(failed_critical)}{Colors.RESET}")
            
            logger.info(f"{Colors.GREEN}✔ Самотестирование завершено.{Colors.RESET}")
            
        except Exception as e:
            logger.critical(f"{Colors.RED}КРИТИЧЕСКАЯ ОШИБКА при выполнении самотестирования: {e}{Colors.RESET}")
            raise

    def run_gui_mode(self, **kwargs) -> None:
        """Запуск в режиме GUI"""
        if not self.initialized:
            logger.critical(f"{Colors.RED}ОШИБКА: Система не инициализирована. Запуск GUI невозможен.{Colors.RESET}")
            raise RuntimeError("Система не инициализирована")
        
        try:
            logger.info(f"\n{Colors.CYAN}### Запуск системы в режиме ГРАФИЧЕСКОГО ИНТЕРФЕЙСА (GUI)...{Colors.RESET}")
            
            # Параметры запуска по умолчанию
            launch_params = {
                "server_name": "0.0.0.0",
                "server_port": None,
                "share": False,
                "debug": False, # Изменено на False для продакшн
                "show_error": True,
                "quiet": False,
                "inbrowser": True
            }
            
            # Обновление параметров из kwargs
            launch_params.update(kwargs)

            # Gradio сам найдет свободный порт, если указанный занят или None
            if launch_params["server_port"] is None:
                logger.info("Автоматический поиск свободного порта для GUI...")
            else:
                logger.info(f"Попытка запустить GUI на порту: {launch_params['server_port']}")
            
            logger.debug(f"Параметры запуска GUI: {launch_params}")
            
            # Запуск gradio_interface
            gradio_interface = self.components['gradio_interface']
            
            self.running = True
            self.session_stats["start_time"] = datetime.now()
            
            # Сохранение снимка конфигурации
            config_snapshot_path = save_configuration_snapshot()
            logger.info(f"Снимок конфигурации сохранен: {config_snapshot_path}")
            
            logger.info(f"\n{Colors.BOLD}{Colors.PURPLE}✨ GUI запускается. Пожалуйста, подождите...{Colors.RESET}\n")
            
            # Запуск интерфейса
            gradio_interface.launch(**launch_params)
            
        except Exception as e:
            logger.critical(f"{Colors.RED}КРИТИЧЕСКАЯ ОШИБКА при запуске GUI: {e}{Colors.RESET}")
            raise
        finally:
            self.running = False
            self._cleanup_system()

    def run_cli_mode(self, image_paths: List[str], output_format: str = "json") -> Dict[str, Any]:
        """Запуск в режиме командной строки"""
        if not self.initialized:
            logger.critical(f"{Colors.RED}ОШИБКА: Система не инициализирована. Запуск CLI невозможен.{Colors.RESET}")
            raise RuntimeError("Система не инициализирована")
        
        try:
            logger.info(f"\n{Colors.CYAN}### Запуск системы в режиме КОМАНДНОЙ СТРОКИ (CLI) для {len(image_paths)} файлов...{Colors.RESET}")
            
            self.running = True
            results = {
                "system_info": {
                    "version": SYSTEM_VERSION,
                    "analysis_date": datetime.now().isoformat(),
                    "total_files": len(image_paths)
                },
                "analysis_results": [],
                "summary": {}
            }
            
            # Валидация путей к файлам
            valid_paths = []
            invalid_paths_count = 0
            for path in image_paths:
                if Path(path).exists():
                    valid_paths.append(path)
                else:
                    logger.warning(f"{Colors.YELLOW}ПРЕДУПРЕЖДЕНИЕ: Файл не найден и будет пропущен: {path}{Colors.RESET}")
                    invalid_paths_count += 1
            
            if not valid_paths:
                logger.error(f"{Colors.RED}ОШИБКА: Не найдено ни одного валидного файла для анализа. Отмена операции.{Colors.RESET}")
                raise ValueError("Не найдено валидных файлов для анализа")
            
            if invalid_paths_count > 0:
                logger.warning(f"{Colors.YELLOW}Пропущено {invalid_paths_count} ненайденных файлов.{Colors.RESET}")
            
            logger.info(f"Начинаем обработку {len(valid_paths)} файлов...")
            
            # Создание хронологического индекса через data_manager
            data_manager = self.components['data_manager']
            logger.info("Создание временного индекса для файлов...")
            chronological_index = data_manager.create_master_chronological_index(valid_paths)
            logger.info(f"{Colors.GREEN}✔ Временной индекс создан.{Colors.RESET}")
            
            # Асинхронная обработка файлов
            logger.info("Запуск асинхронной обработки файлов. Это может занять некоторое время...")
            results_list = asyncio.run(self._process_files_async(chronological_index))
            
            # Агрегация результатов через results_aggregator
            results_aggregator = self.components['results_aggregator']
            logger.info(f"Агрегация результатов {len(results_list)} успешно обработанных файлов...")
            # --- ИСПРАВЛЕНИЕ: Преобразуем словари в AnalysisResult ---
            analysis_results_objs = []
            for r in results_list:
                if isinstance(r, dict):
                    # Если есть ошибка, заполняем минимально
                    if 'error' in r:
                        analysis_results_objs.append(AnalysisResult(
                            filepath=r.get('path', r.get('filepath', '')),
                            timestamp=str(datetime.now()),
                            authenticity_score=0.0,
                            anomalies={"processing_error": r.get('error', '')},
                            metrics=r.get('metrics', {}),
                            metadata={k: v for k, v in r.items() if k not in ['path', 'filepath', 'metrics', 'error']}
                        ))
                    else:
                        analysis_results_objs.append(AnalysisResult(
                            filepath=r.get('path', r.get('filepath', '')),
                            timestamp=str(datetime.now()),
                            authenticity_score=r.get('authenticity_score', 0.0),
                            anomalies={},
                            metrics=r.get('metrics', {}),
                            metadata={k: v for k, v in r.items() if k not in ['path', 'filepath', 'metrics', 'authenticity_score']}
                        ))
                else:
                    analysis_results_objs.append(r)
            results_aggregator.add_results(analysis_results_objs)
            
            # Создание сводки
            results["summary"] = results_aggregator.get_statistics()
            results["analysis_results"] = [
                {
                    "filepath": r["path"],
                    "authenticity_score": r.get("authenticity_score", 0.0),
                    "pose": r.get("pose", ""),
                    "metrics": r.get("metrics", {}),
                    "age": r.get("age", 0.0),
                    "date": str(r.get("date", ""))
                }
                for r in results_list
            ]
            
            # Сохранение результатов
            output_path = self._save_cli_results(results, output_format)
            results["output_file"] = str(output_path)
            
            logger.info(f"\n{Colors.BOLD}{Colors.GREEN}✔ CLI анализ завершен!{Colors.RESET}")
            logger.info(f"Результаты сохранены в: {Colors.CYAN}{output_path}{Colors.RESET}")
            return results
            
        except Exception as e:
            logger.critical(f"{Colors.RED}КРИТИЧЕСКАЯ ОШИБКА в CLI режиме: {e}{Colors.RESET}")
            logger.debug(f"Трассировка ошибки: {traceback.format_exc()}")
            raise
        finally:
            self.running = False

    async def _process_files_async(self, chronological_index: List[Dict[str, Any]]) -> List[Any]:
        """Асинхронная обработка файлов по хронологическому индексу"""
        try:
            face_analyzer = self.components['face_3d_analyzer']
            data_manager = self.components['data_manager']
            
            all_results = []
            
            total_items = sum(len(item["images"]) for item in chronological_index)
            processed_count = 0
            
            for item in chronological_index:
                item_date = item["date"]
                item_age = item["age_on_date"]
                
                for image_path in item["images"]:
                    # Передаем image_path, item_date, item_age
                    result = self._analyze_single_image_for_pipeline(image_path, item_date, item_age)
                    all_results.append(result)
                    processed_count += 1
                    
                    # Прогресс (можно использовать progress_callback, если он будет интегрирован на уровне _analyze_single_image_for_pipeline)
                    if processed_count % 10 == 0:
                        logger.info(f"Прогресс: {processed_count}/{total_items} файлов обработано.")

            self.session_stats["files_processed"] = total_items
            self.session_stats["analyses_completed"] = len(all_results)
            
            logger.info(f"{Colors.GREEN}✔ Асинхронная обработка {len(all_results)} файлов завершена.{Colors.RESET}")
            return all_results
            
        except Exception as e:
            logger.error(f"{Colors.RED}ОШИБКА: Не удалось асинхронно обработать файлы: {e}{Colors.RESET}")
            return []

    def _analyze_single_image_for_pipeline(self, image_path: str, item_date: datetime.date, item_age: float) -> Dict[str, Any]:
        """
        Анализ одного изображения в рамках пайплайна bootstrap_pipeline.
        Возвращает структурированный словарь с результатами анализа.
        """
        import cv2
        
        face_analyzer = self.components['face_3d_analyzer']
        data_manager = self.components['data_manager']
        
        logger.info(f"[DEBUG] === НАЧАЛО анализа изображения: {image_path}")
        print(f"[DEBUG] === НАЧАЛО анализа изображения: {image_path}")
        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"{Colors.YELLOW}ПРЕДУПРЕЖДЕНИЕ: Не удалось загрузить изображение {image_path}. Пропускаем.{Colors.RESET}")
                print(f"[DEBUG] Не удалось загрузить изображение: {image_path}")
                return {"path": image_path, "error": "Не удалось загрузить изображение (файл повреждён или не поддерживается)"}
            logger.info(f"[DEBUG] Изображение загружено: shape={img.shape}, dtype={img.dtype}")
            print(f"[DEBUG] Изображение загружено: shape={img.shape}, dtype={img.dtype}")

            # 1. Валидация качества изображения
            logger.info(f"[DEBUG] Валидация качества изображения...")
            quality_results = data_manager.validate_image_quality_for_analysis(img)
            logger.info(f"[DEBUG] Результат валидации качества: {quality_results}")
            print(f"[DEBUG] Результат валидации качества: {quality_results}")
            if quality_results is None or not isinstance(quality_results, dict):
                print(f"[DEBUG] Ошибка валидации качества изображения")
                return {"path": image_path, "error": "Ошибка валидации качества изображения"}
            if quality_results.get('quality_score', 0) < CRITICAL_THRESHOLDS["min_quality_score"]:
                logger.warning(f"{Colors.YELLOW}ПРЕДУПРЕЖДЕНИЕ: Низкое качество изображения {image_path} (score: {quality_results.get('quality_score', 0):.2f}).{Colors.RESET}")
                print(f"[DEBUG] Слишком низкое качество изображения для анализа")
                return {"path": image_path, "quality_issues": quality_results.get("issues", []), "quality_score": quality_results.get("quality_score", 0), "error": "Слишком низкое качество изображения для анализа"}

            # 2. Извлечение 68 ландмарок
            logger.info(f"[DEBUG] Извлечение 68 ландмарок...")
            print(f"[DEBUG] Извлечение 68 ландмарок...")
            landmarks_3d, confidence_array, param = face_analyzer.extract_68_landmarks_with_confidence(img, models={'tddfa': face_analyzer.tddfa, 'face_boxes': face_analyzer.faceboxes})
            logger.info(f"[DEBUG] landmarks_3d: {type(landmarks_3d)}, shape: {getattr(landmarks_3d, 'shape', None)}, size: {getattr(landmarks_3d, 'size', None)}")
            logger.info(f"[DEBUG] confidence_array: {type(confidence_array)}, shape: {getattr(confidence_array, 'shape', None)}, size: {getattr(confidence_array, 'size', None)}")
            logger.info(f"[DEBUG] param: {type(param)}, shape: {getattr(param, 'shape', None)}, size: {getattr(param, 'size', None)}")
            print(f"[DEBUG] landmarks_3d: {type(landmarks_3d)}, shape: {getattr(landmarks_3d, 'shape', None)}, size: {getattr(landmarks_3d, 'size', None)}")
            print(f"[DEBUG] confidence_array: {type(confidence_array)}, shape: {getattr(confidence_array, 'shape', None)}, size: {getattr(confidence_array, 'size', None)}")
            print(f"[DEBUG] param: {type(param)}, shape: {getattr(param, 'shape', None)}, size: {getattr(param, 'size', None)}")
            if landmarks_3d is None or not hasattr(landmarks_3d, 'size') or landmarks_3d.size == 0:
                logger.warning(f"{Colors.YELLOW}ПРЕДУПРЕЖДЕНИЕ: Лицо не обнаружено на изображении {image_path}. Пропускаем.{Colors.RESET}")
                print(f"[DEBUG] Лицо не обнаружено на изображении")
                return {"path": image_path, "error": "Лицо не обнаружено на изображении. Попробуйте другое фото с чётким лицом."}
            if confidence_array is None or not hasattr(confidence_array, 'size') or confidence_array.size == 0:
                logger.warning(f"{Colors.YELLOW}ПРЕДУПРЕЖДЕНИЕ: Confidence array пустой для {image_path}. Пропускаем.{Colors.RESET}")
                print(f"[DEBUG] Confidence array пустой")
                return {"path": image_path, "error": "Ошибка извлечения ландмарок лица (confidence пустой)"}
            if param is None or not hasattr(param, 'size') or param.size == 0:
                logger.warning(f"{Colors.YELLOW}ПРЕДУПРЕЖДЕНИЕ: Параметры модели пустые для {image_path}. Пропускаем.{Colors.RESET}")
                print(f"[DEBUG] Параметры модели пустые")
                return {"path": image_path, "error": "Ошибка извлечения параметров модели лица"}

            # 3. Определение позы
            logger.info(f"[DEBUG] Определение позы лица...")
            pose_analysis = face_analyzer.determine_precise_face_pose(landmarks_3d, param, confidence_array)
            logger.info(f"[DEBUG] pose_analysis: {pose_analysis}")
            print(f"[DEBUG] pose_analysis: {pose_analysis}")
            if pose_analysis is None or not isinstance(pose_analysis, dict):
                print(f"[DEBUG] Ошибка определения позы лица")
                return {"path": image_path, "error": "Ошибка определения позы лица"}
            pose_category = pose_analysis.get("pose_category", "Unknown")
            if pose_category == "Unknown" or pose_category == "Error":
                logger.warning(f"{Colors.YELLOW}ПРЕДУПРЕЖДЕНИЕ: Не удалось определить позу для {image_path}. Пропускаем.{Colors.RESET}")
                print(f"[DEBUG] Не удалось определить позу лица")
                return {"path": image_path, "error": "Не удалось определить позу лица"}

            # 4. Нормализация ландмарок
            logger.info(f"[DEBUG] Нормализация ландмарок...")
            vis = landmarks_3d[:,2] > MIN_VISIBILITY_Z
            norm_landmarks = face_analyzer.normalize_landmarks_by_pose_category(landmarks_3d, pose_category, vis)
            logger.info(f"[DEBUG] norm_landmarks: {type(norm_landmarks)}, shape: {getattr(norm_landmarks, 'shape', None)}, size: {getattr(norm_landmarks, 'size', None)}")
            print(f"[DEBUG] norm_landmarks: {type(norm_landmarks)}, shape: {getattr(norm_landmarks, 'shape', None)}, size: {getattr(norm_landmarks, 'size', None)}")
            if norm_landmarks is None or not hasattr(norm_landmarks, 'size') or norm_landmarks.size == 0:
                logger.warning(f"{Colors.YELLOW}ПРЕДУПРЕЖДЕНИЕ: Не удалось нормализовать ландмарки для {image_path}. Пропускаем.{Colors.RESET}")
                print(f"[DEBUG] Не удалось нормализовать ландмарки лица")
                return {"path": image_path, "error": "Не удалось нормализовать ландмарки лица"}

            # 5. Расчет метрик идентичности
            logger.info(f"[DEBUG] Расчет метрик идентичности...")
            metrics = face_analyzer.calculate_identity_signature_metrics(norm_landmarks, pose_category)
            logger.info(f"[DEBUG] metrics: {metrics}")
            print(f"[DEBUG] metrics: {metrics}")
            if metrics is None or not isinstance(metrics, dict) or len(metrics) == 0:
                logger.warning(f"{Colors.YELLOW}ПРЕДУПРЕЖДЕНИЕ: Не удалось рассчитать метрики для {image_path}. Пропускаем.{Colors.RESET}")
                print(f"[DEBUG] Не удалось рассчитать метрики идентичности")
                return {"path": image_path, "error": "Не удалось рассчитать метрики идентичности"}

            # 6. Проверка shape error (если используется)
            if hasattr(face_analyzer, 'calculate_comprehensive_shape_error'):
                try:
                    logger.info(f"[DEBUG] Расчет shape error...")
                    shape_error = face_analyzer.calculate_comprehensive_shape_error(norm_landmarks)
                    logger.info(f"[DEBUG] shape_error: {shape_error}")
                    print(f"[DEBUG] shape_error: {shape_error}")
                    if shape_error is None or not isinstance(shape_error, dict):
                        print(f"[DEBUG] Ошибка расчёта shape error")
                        return {"path": image_path, "error": "Ошибка расчёта shape error"}
                except Exception as e:
                    print(f"[DEBUG] Ошибка shape error: {str(e)}")
                    return {"path": image_path, "error": f"Ошибка shape error: {str(e)}"}

            # 7. Проверка асимметрии (если используется)
            if hasattr(face_analyzer, 'analyze_facial_asymmetry'):
                try:
                    logger.info(f"[DEBUG] Анализ асимметрии...")
                    print(f"[DEBUG] Анализ асимметрии...")
                    left_indices  = [0, 1, 2, 3, 4, 5, 6, 7, 17, 18, 19, 20, 36, 37, 38, 39, 40, 41, 48, 49, 50, 51, 52, 53, 54]
                    right_indices = [16,15,14,13,12,11,10,9,26,25,24,23,45,44,43,42,47,46,54,53,52,51,50,49,48]
                    asymmetry = face_analyzer.analyze_facial_asymmetry(landmarks_3d, confidence_array)
                    logger.info(f"[DEBUG] asymmetry: {asymmetry}")
                    print(f"[DEBUG] asymmetry: {asymmetry}")
                    if not isinstance(asymmetry, dict) or not asymmetry:
                        logger.warning(f"{Colors.YELLOW}ПРЕДУПРЕЖДЕНИЕ: Не удалось проанализировать асимметрию для {image_path}. Пропускаем.{Colors.RESET}")
                        print(f"[DEBUG] Не удалось проанализировать асимметрию лица")
                        return {"path": image_path, "error": "Не удалось проанализировать асимметрию лица"}
                except Exception as e:
                    logger.error(f"[DEBUG] Ошибка при анализе асимметрии: {e}")
                    print(f"[DEBUG] Ошибка анализа асимметрии: {str(e)}")
                    return {"path": image_path, "error": f"Ошибка анализа асимметрии: {str(e)}"}

            # 8. Извлечение эмбеддинга лица (512D) и анализ эмбеддингов
            logger.info(f"[DEBUG] Извлечение эмбеддинга лица...")
            embedding_analyzer = self.components['embedding_analyzer']
            emb, emb_conf = embedding_analyzer.extract_512d_face_embedding(img, embedding_analyzer.face_app)
            logger.info(f"[DEBUG] emb: {type(emb)}, shape: {getattr(emb, 'shape', None)}, size: {getattr(emb, 'size', None)}; emb_conf: {emb_conf}")
            print(f"[DEBUG] emb: {type(emb)}, shape: {getattr(emb, 'shape', None)}, size: {getattr(emb, 'size', None)}; emb_conf: {emb_conf}")
            if emb is None or emb_conf is None or not hasattr(emb, 'size') or emb.size == 0:
                logger.warning(f"{Colors.YELLOW}ПРЕДУПРЕЖДЕНИЕ: Не удалось извлечь эмбеддинг для {image_path}. Пропускаем.{Colors.RESET}")
                print(f"[DEBUG] Не удалось извлечь эмбеддинг лица")
                return {"path": image_path, "error": "Не удалось извлечь эмбеддинг лица"}
            if not isinstance(emb_conf, (float, int, np.floating)):
                logger.warning(f"{Colors.YELLOW}ПРЕДУПРЕЖДЕНИЕ: Некорректный тип emb_conf для {image_path}: {emb_conf} ({type(emb_conf)}). Пропускаем.{Colors.RESET}")
                print(f"[DEBUG] Некорректный тип emb_conf: {emb_conf}")
                return {"path": image_path, "error": "Некорректный тип уверенности эмбеддинга"}
            if emb_conf < 0.3:
                logger.warning(f"{Colors.YELLOW}ПРЕДУПРЕЖДЕНИЕ: Низкая уверенность эмбеддинга для {image_path} (confidence={emb_conf:.3f} < 0.3). Пропускаем.{Colors.RESET}")
                print(f"[DEBUG] Слишком низкая уверенность эмбеддинга: {emb_conf}")
                return {"path": image_path, "error": f"Слишком низкая уверенность эмбеддинга: {emb_conf:.3f}"}
            emb_result = {
                'embedding': emb.tolist(),
                'extraction_confidence': float(emb_conf)
            }
            fusion = embedding_analyzer.ensemble_embedding_analysis(
                np.expand_dims(emb, axis=0), strategies=("mean-cosine", "median")
            )
            emb_result['ensemble_fusion'] = fusion

            # 9. Анализ текстуры кожи (LBP, Gabor, FFT, entropy, Level)
            logger.info(f"[DEBUG] Анализ текстуры кожи...")
            texture_analyzer = self.components['texture_analyzer']
            texture_zones = texture_analyzer.analyze_skin_texture_by_zones(img, landmarks_3d)
            logger.info(f"[DEBUG] texture_zones: {texture_zones}")
            print(f"[DEBUG] texture_zones: {texture_zones}")
            if texture_zones is None or not isinstance(texture_zones, dict) or len(texture_zones) == 0:
                logger.warning(f"{Colors.YELLOW}ПРЕДУПРЕЖДЕНИЕ: Не удалось проанализировать текстуру для {image_path}. Пропускаем.{Colors.RESET}")
                print(f"[DEBUG] Не удалось проанализировать текстуру кожи")
                return {"path": image_path, "error": "Не удалось проанализировать текстуру кожи"}
            material_score = texture_analyzer.calculate_material_authenticity_score(texture_zones)
            mask_level = texture_analyzer.classify_mask_technology_level(texture_zones, photo_date=item_date)

            # --- ДОБАВЛЯЮ РАСЧЁТ authenticity_score ---
            anomaly_detector = self.components['anomaly_detector']
            geometry_score = metrics.get('geometry_score', 1.0) if 'geometry_score' in metrics else 1.0
            embedding_score = float(emb_conf)
            texture_score = float(material_score)
            temporal_score = 1.0  # если нет временного анализа, ставим 1.0
            print(f"DEBUG: geometry_score={geometry_score}, embedding_score={embedding_score}, texture_score={texture_score}, temporal_score={temporal_score}")
            logger.info(f"[DEBUG] authenticity_score inputs: geometry={geometry_score}, embedding={embedding_score}, texture={texture_score}, temporal={temporal_score}")
            # Проверка на nan/None
            for name, val in zip(["geometry_score", "embedding_score", "texture_score", "temporal_score"], [geometry_score, embedding_score, texture_score, temporal_score]):
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    print(f"ОШИБКА: {name} = {val} (None или nan)")
                    logger.error(f"ОШИБКА: {name} = {val} (None или nan)")
            authenticity_score = anomaly_detector.calculate_identity_authenticity_score(
                geometry=geometry_score,
                embedding=embedding_score,
                texture=texture_score,
                temporal_consistency=temporal_score
            )

            logger.info(f"[DEBUG] === КОНЕЦ анализа изображения: {image_path}")
            print(f"[DEBUG] === КОНЕЦ анализа изображения: {image_path}")
            # Возвращаем расширенный результат с authenticity_score
            return dict(
                path=image_path,
                pose=pose_category,
                metrics=metrics,
                age=item_age,
                date=item_date,
                quality_score=quality_results["quality_score"],
                quality_issues=quality_results["issues"],
                landmarks_confidence=confidence_array.mean(),
                embedding_analysis=emb_result,
                texture_zones=texture_zones,
                material_score=material_score,
                mask_level=mask_level,
                authenticity_score=authenticity_score
            )
        except Exception as e:
            logger.error(f"{Colors.RED}ОШИБКА анализа изображения {image_path}: {e}{Colors.RESET}")
            print(f"[DEBUG] ОШИБКА анализа изображения: {str(e)}")
            return {"path": image_path, "error": f"Внутренняя ошибка анализа: {str(e)}"}

    def _save_cli_results(self, results: Dict[str, Any], format: str) -> Path:
        """Сохранение результатов CLI"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format.lower() == "json":
                output_path = RESULTS_DIR / f"cli_results_{timestamp}.json"
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
                logger.info(f"Результаты сохранены в JSON: {output_path.name}")
            
            elif format.lower() == "csv":
                import pandas as pd
                output_path = RESULTS_DIR / f"cli_results_{timestamp}.csv"
                
                # Преобразование в DataFrame
                df_data = []
                for result in results["analysis_results"]:
                    df_data.append({
                        "filepath": result.get("filepath", ""),
                        "authenticity_score": result.get("authenticity_score", 0),
                        "error": result.get("error", ""),
                        "timestamp": result.get("timestamp", "")
                    })
                
                df = pd.DataFrame(df_data)
                df.to_csv(output_path, index=False, encoding='utf-8')
                logger.info(f"Результаты сохранены в CSV: {output_path.name}")
            
            else:
                logger.error(f"{Colors.RED}ОШИБКА: Неподдерживаемый формат вывода: {format}{Colors.RESET}")
                raise ValueError(f"Неподдерживаемый формат: {format}")
            
            return output_path
            
        except Exception as e:
            logger.critical(f"{Colors.RED}КРИТИЧЕСКАЯ ОШИБКА сохранения результатов: {e}{Colors.RESET}")
            raise

    def _cleanup_system(self) -> None:
        """Очистка системы при завершении"""
        try:
            logger.info(f"\n{Colors.CYAN}### Завершение работы системы: Очистка и сохранение кэшей...{Colors.RESET}")
            
            # ИСПРАВЛЕНО: Сохранение кэшей с оригинальными названиями
            cache_methods = [
                ('data_manager', 'save_cache'),
                ('embedding_analyzer', 'save_cache'),
                ('texture_analyzer', 'save_analysis_cache'),
                ('temporal_analyzer', 'save_analysis_cache'),
                ('anomaly_detector', 'save_analysis_cache'),
                ('medical_validator', 'save_validation_cache'),
                ('visualization_engine', 'save_plot_cache')
            ]
            
            for component_name, method_name in cache_methods:
                if component_name in self.components:
                    component = self.components[component_name]
                    if hasattr(component, method_name):
                        try:
                            getattr(component, method_name)()
                            logger.debug(f"Кэш сохранен для {component_name}")
                        except Exception as e:
                            logger.warning(f"{Colors.YELLOW}ПРЕДУПРЕЖДЕНИЕ: Ошибка сохранения кэша для '{component_name}': {e}{Colors.RESET}")
            
            # Статистика сессии
            uptime = (datetime.now() - self.session_stats["start_time"]).total_seconds()
            self.session_stats["uptime_seconds"] = uptime
            
            logger.info(f"{Colors.GREEN}✔ Очистка системы завершена. Время работы: {uptime:.2f} сек.{Colors.RESET}")
            logger.debug(f"Статистика сессии: {self.session_stats}")
            
        except Exception as e:
            logger.critical(f"{Colors.RED}КРИТИЧЕСКАЯ ОШИБКА при очистке системы: {e}{Colors.RESET}")

    def get_system_status(self) -> Dict[str, Any]:
        """Получение статуса системы"""
        logger.info(f"\n{Colors.CYAN}### Получение статуса системы...{Colors.RESET}")
        status = {
            "initialized": self.initialized,
            "running": self.running,
            "components_count": len(self.components),
            "session_stats": self.session_stats.copy(),
            "system_version": SYSTEM_VERSION
        }
        logger.info(f"{Colors.GREEN}✔ Статус системы получен.{Colors.RESET}")
        return status

# ==================== ФУНКЦИИ КОМАНДНОЙ СТРОКИ ====================

def parse_arguments() -> argparse.Namespace:
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description=f"{SYSTEM_NAME} v{SYSTEM_VERSION} - Система анализа подлинности 3D лиц",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "mode",
        choices=list(OPERATION_MODES.keys()),
        help="Режим работы системы: " + ", ".join(f"{k} ({v})" for k, v in OPERATION_MODES.items())
    )
    
    parser.add_argument(
        "--images",
        nargs='+',
        help="Пути к изображениям для анализа (обязательно для режима 'cli')"
    )
    
    parser.add_argument(
        "--output-format",
        choices=["json", "csv"],
        default="json",
        help="Формат вывода результатов (для режима 'cli', по умолчанию json)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Порт для запуска GUI режима (по умолчанию 7860, если занят - будет найден следующий свободный)"
    )
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Хост для GUI режима (по умолчанию 0.0.0.0)"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Создать публичную ссылку Gradio (для GUI режима)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Включить отладочный режим (много подробных сообщений)"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"{SYSTEM_NAME} v{SYSTEM_VERSION} (Сборка: {BUILD_DATE})"
    )
    
    return parser.parse_args()

def setup_signal_handlers(system: FaceAuthenticitySystem) -> None:
    """Настройка обработчиков сигналов"""
    def signal_handler(signum, frame):
        logger.info(f"{Colors.YELLOW}Получен сигнал завершения ({signum}). Завершаем работу...{Colors.RESET}")
        system.shutdown_requested = True
        system._cleanup_system()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def main():
    """Главная функция запуска"""
    try:
        print("=== main.py стартовал, логгер должен быть настроен ===")
        print(f"\n{'='*70}")
        print(f"Система анализа подлинности 3D лиц v{SYSTEM_VERSION}")
        print(f"Автор: {AUTHOR}")
        print(f"Дата сборки: {BUILD_DATE}")
        print(f"{'='*70}\n")
        
        # Парсинг аргументов
        logger.info(f"{Colors.CYAN}### Обработка аргументов запуска...{Colors.RESET}")
        args = parse_arguments()
        logger.info(f"Выбранный режим работы: {Colors.BOLD}{OPERATION_MODES[args.mode]}{Colors.RESET}")
        
        # Настройка уровня логирования
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.info(f"{Colors.YELLOW}Включен ОТЛАДОЧНЫЙ режим. Вывод будет очень подробным.{Colors.RESET}")
        else:
            logging.getLogger().setLevel(logging.INFO)
            logger.info("Отладочный режим выключен. Вывод будет кратким.")
        
        # ИНИЦИАЛИЗАЦИЯ СИСТЕМЫ ОДИНАКОВАЯ ДЛЯ ВСЕХ РЕЖИМОВ
        system = FaceAuthenticitySystem(run_self_tests=(args.mode != "gui"))
        print("DEBUG: Система инициализирована. Компоненты:", list(system.components.keys()))
        
        # Настройка обработчиков сигналов
        setup_signal_handlers(system)
        
        # Запуск в выбранном режиме
        if args.mode == "gui":
            system.run_gui_mode(
                server_name=args.host,
                server_port=None if args.port == 7860 else args.port,
                share=args.share,
                debug=args.debug
            )
        
        elif args.mode == "cli":
            if not args.images:
                logger.critical(f"{Colors.RED}ОШИБКА: Для режима CLI необходимо указать пути к изображениям с помощью флага --images.{Colors.RESET}")
                print(f"ОШИБКА: Для режима CLI необходимо указать пути к изображениям с помощью флага --images.")
                sys.exit(1)
            try:
                results = system.run_cli_mode(args.images, args.output_format)
                print(f"\n{Colors.BOLD}{Colors.GREEN}Итоговый отчет CLI анализа:{Colors.RESET}")
                print(f"  {Colors.WHITE}Всего файлов обработано:{Colors.RESET} {results['summary'].get('total_files', 0)}")
                print(f"  {Colors.GREEN}Успешно проанализировано:{Colors.RESET} {results['summary'].get('successful_analyses', 0)}")
                print(f"  {Colors.RED}Ошибок при анализе:{Colors.RESET} {results['summary'].get('failed_analyses', 0)}")
                print(f"  {Colors.CYAN}Средний балл аутентичности:{Colors.RESET} {results['summary'].get('average_authenticity', 0.0):.3f}")
                print(f"  {Colors.BLUE}Файл с результатами:{Colors.RESET} {results.get('output_file', '')}")
                logger.info(f"CLI анализ завершён. Итоговый файл: {results.get('output_file', '')}")
            except Exception as e:
                logger.critical(f"{Colors.RED}КРИТИЧЕСКАЯ ОШИБКА в CLI режиме: {e}{Colors.RESET}")
                print(f"КРИТИЧЕСКАЯ ОШИБКА в CLI режиме: {e}")
                sys.exit(1)
        
        elif args.mode == "test":
            logger.info(f"\n{Colors.CYAN}### Запуск системы в ТЕСТОВОМ режиме...{Colors.RESET}")
            status = system.get_system_status()
            print(f"\n{Colors.BOLD}{Colors.PURPLE}Текущий статус системы:{Colors.RESET}")
            print(f"  {Colors.WHITE}Инициализирована:{Colors.RESET} {status['initialized']}")
            print(f"  {Colors.WHITE}Работает:{Colors.RESET} {status['running']}")
            print(f"  {Colors.WHITE}Количество компонентов:{Colors.RESET} {status['components_count']}")
            print(f"  {Colors.WHITE}Версия системы:{Colors.RESET} {status['system_version']}")
            print(f"  {Colors.WHITE}Время непрерывной работы:{Colors.RESET} {status['session_stats']['uptime_seconds']:.2f} сек.")
            logger.info(f"Тестовый режим: статус системы выведен.")
        else:
            logger.critical(f"{Colors.RED}ОШИБКА: Неподдерживаемый режим работы: {args.mode}{Colors.RESET}")
            print(f"ОШИБКА: Неподдерживаемый режим работы: {args.mode}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info(f"{Colors.YELLOW}Выполнение остановлено пользователем (Ctrl+C). Завершение работы.{Colors.RESET}")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"{Colors.RED}КРИТИЧЕСКАЯ ОШИБКА, приложение будет закрыто: {e}{Colors.RESET}")
        logger.debug(f"Трассировка ошибки: {traceback.format_exc()}")
        print("CRITICAL ERROR:", e)
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()

class DataMissingWarning(UserWarning):
    pass

warnings.filterwarnings('once')

def log_and_print(level, msg):
    print(f"[LOG][{level}] {msg}")
    if level == 'info':
        logger.info(msg)
    elif level == 'error':
        logger.error(msg)
    elif level == 'warning':
        logger.warning(msg)
    else:
        logger.debug(msg)
