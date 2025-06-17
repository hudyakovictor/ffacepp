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
log_file_handler = logging.FileHandler('logs/main.log')
log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'))

console_handler = logging.StreamHandler()
console_handler.setFormatter(ColoredFormatter())

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[
        log_file_handler,
        console_handler
    ]
)
logger = logging.getLogger(__name__)

# ИСПРАВЛЕНО: Импорт всех модулей с ОРИГИНАЛЬНЫМИ названиями
try:
    from core_config import (
        validate_configuration, save_configuration_snapshot,
        PUTIN_BIRTH_DATE, START_ANALYSIS_DATE, END_ANALYSIS_DATE,
        DATA_DIR, RESULTS_DIR, CACHE_DIR, LOGS_DIR, CONFIG_DIR,
        MAX_FILE_UPLOAD_COUNT, ERROR_CODES, SYSTEM_VERSION,
        AUTHENTICITY_WEIGHTS, CRITICAL_THRESHOLDS, MIN_VISIBILITY_Z
    )
    from face_3d_analyzer import Face3DAnalyzer, initialize_3ddfa_components
    from embedding_analyzer import EmbeddingAnalyzer
    from texture_analyzer import TextureAnalyzer
    from temporal_analyzer import TemporalAnalyzer
    from anomaly_detector import AnomalyDetector
    from medical_validator import MedicalValidator
    from data_manager import DataManager
    from metrics_calculator import MetricsCalculator
    from data_processing import DataProcessor, ResultsAggregator
    from gradio_interface import GradioInterface, create_modular_interface
    from visualization_engine import VisualizationEngine
    from report_generator import ReportGenerator
    logger.info("Все модули системы успешно импортированы с оригинальными названиями")
except ImportError as e:
    logger.error(f"Критическая ошибка импорта модулей: {e}")
    print(f"ОШИБКА: Не удалось импортировать модули системы: {e}")
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
            self.components['gradio_interface'] = create_modular_interface(
                all_system_components=self.components
            )
            
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
            results_aggregator.add_results(results_list)
            
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
        
        logger.info(f"Анализ изображения: {image_path}")
        
        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"{Colors.YELLOW}ПРЕДУПРЕЖДЕНИЕ: Не удалось загрузить изображение {image_path}. Пропускаем.{Colors.RESET}")
                return {"path": image_path, "error": "Не удалось загрузить изображение"}

            # 1. Валидация качества изображения
            quality_results = data_manager.validate_image_quality_for_analysis(img)
            if quality_results['quality_score'] < CRITICAL_THRESHOLDS["min_quality_score"]:
                logger.warning(f"{Colors.YELLOW}ПРЕДУПРЕЖДЕНИЕ: Низкое качество изображения {image_path} (score: {quality_results['quality_score']:.2f}).{Colors.RESET}")
                return {"path": image_path, "quality_issues": quality_results["issues"], "quality_score": quality_results["quality_score"], "error": "Низкое качество изображения"}

            # 2. Извлечение 68 ландмарок
            landmarks_3d, confidence_array, param = face_analyzer.extract_68_landmarks_with_confidence(img, models={'tddfa': face_analyzer.tddfa, 'face_boxes': face_analyzer.faceboxes}) # Pass models from analyzer
            if landmarks_3d.size == 0:
                logger.warning(f"{Colors.YELLOW}ПРЕДУПРЕЖДЕНИЕ: Лицо не обнаружено на изображении {image_path}. Пропускаем.{Colors.RESET}")
                return {"path": image_path, "error": "Лицо не обнаружено"}

            # 3. Определение позы
            pose_analysis = face_analyzer.determine_precise_face_pose(landmarks_3d, param, confidence_array)
            pose_category = pose_analysis["pose_category"]

            # 4. Нормализация ландмарок
            vis = landmarks_3d[:,2] > MIN_VISIBILITY_Z
            norm_landmarks = face_analyzer.normalize_landmarks_by_pose_category(landmarks_3d, pose_category, vis)

            # 5. Расчет метрик идентичности
            metrics = face_analyzer.calculate_identity_signature_metrics(norm_landmarks, pose_category)

            return dict(
                path=image_path,
                pose=pose_category,
                metrics=metrics,
                age=item_age,
                date=item_date,
                quality_score=quality_results["quality_score"],
                quality_issues=quality_results["issues"],
                landmarks_confidence=confidence_array.mean() # Средняя уверенность по ландмаркам
            )
        except Exception as e:
            logger.error(f"{Colors.RED}ОШИБКА анализа изображения {image_path}: {e}{Colors.RESET}")
            return {"path": image_path, "error": str(e)}

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
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}{SYSTEM_NAME} v{SYSTEM_VERSION}{Colors.RESET}")
        print(f"{Colors.BLUE}Автор: {AUTHOR}{Colors.RESET}")
        print(f"{Colors.BLUE}Дата сборки: {BUILD_DATE}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")
        
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
        
        # Создание системы
        logger.info(f"\n{Colors.CYAN}### Инициализация основной системы...{Colors.RESET}")
        # Передаем run_self_tests=False для GUI режима
        if args.mode == "gui":
            system = FaceAuthenticitySystem(run_self_tests=False)
        else:
            system = FaceAuthenticitySystem(run_self_tests=True) # Для других режимов тесты запускаем
        
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
                sys.exit(1)
            
            results = system.run_cli_mode(args.images, args.output_format)
            
            print(f"\n{Colors.BOLD}{Colors.GREEN}Итоговый отчет CLI анализа:{Colors.RESET}")
            print(f"  {Colors.WHITE}Всего файлов обработано:{Colors.RESET} {results['summary']['total_files']}")
            print(f"  {Colors.GREEN}Успешно проанализировано:{Colors.RESET} {results['summary']['successful_analyses']}")
            print(f"  {Colors.RED}Ошибок при анализе:{Colors.RESET} {results['summary']['failed_analyses']}")
            print(f"  {Colors.CYAN}Средний балл аутентичности:{Colors.RESET} {results['summary']['average_authenticity']:.3f}")
            print(f"  {Colors.BLUE}Файл с результатами:{Colors.RESET} {results['output_file']}")
            
        elif args.mode == "test":
            logger.info(f"\n{Colors.CYAN}### Запуск системы в ТЕСТОВОМ режиме...{Colors.RESET}")
            status = system.get_system_status()
            print(f"\n{Colors.BOLD}{Colors.PURPLE}Текущий статус системы:{Colors.RESET}")
            print(f"  {Colors.WHITE}Инициализирована:{Colors.RESET} {status['initialized']}")
            print(f"  {Colors.WHITE}Работает:{Colors.RESET} {status['running']}")
            print(f"  {Colors.WHITE}Количество компонентов:{Colors.RESET} {status['components_count']}")
            print(f"  {Colors.WHITE}Версия системы:{Colors.RESET} {status['system_version']}")
            print(f"  {Colors.WHITE}Время непрерывной работы:{Colors.RESET} {status['session_stats']['uptime_seconds']:.2f} сек.")
            
        else:
            logger.critical(f"{Colors.RED}ОШИБКА: Неподдерживаемый режим работы: {args.mode}{Colors.RESET}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info(f"{Colors.YELLOW}Выполнение остановлено пользователем (Ctrl+C). Завершение работы.{Colors.RESET}")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"{Colors.RED}КРИТИЧЕСКАЯ ОШИБКА, приложение будет закрыто: {e}{Colors.RESET}")
        logger.debug(f"Трассировка ошибки: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
