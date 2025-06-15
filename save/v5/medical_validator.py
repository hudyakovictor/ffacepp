import numpy as np
from scipy import stats
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

from core_config import PUTIN_BIRTH_DATE, AgingModel, ANOMALY_DETECTION_THRESHOLDS, MEDICAL_VALIDATOR_THRESHOLDS, get_age_adjusted_thresholds
from temporal_analyzer import TemporalAnalyzer
from data_manager import DataManager

logger = logging.getLogger(__name__)

# =====================
# Проверка соответствия изменений естественному старению
# =====================
def validate_aging_consistency_for_identity(identity_metrics_timeline: Dict, age_progression: List[float], data_manager: DataManager) -> Dict:
    """
    Валидирует консистентность изменений метрик идентичности с естественными моделями старения.
    
    Сравнивает фактические изменения лицевых метрик личности с предсказанными моделями старения,
    выявляет аномальные изменения и рассчитывает общий показатель консистентности.

    Args:
        identity_metrics_timeline (Dict): Словарь, где ключи - это даты (datetime), а значения - словари метрик
                                        (например, {'2000-01-01': {'skull_width_ratio': 0.8, ...}}).
        age_progression (List[float]): Список возрастов, соответствующий порядку дат в identity_metrics_timeline.
                                      Предполагается, что это возраст на каждую дату снимка.
        data_manager (DataManager): Экземпляр DataManager для доступа к данным.

    Returns:
        Dict: Словарь с результатами валидации, включая общий балл консистентности и детали аномалий.
    """
    validation_results = {
        'overall_aging_consistency_score': 1.0,
        'anomalies_detected': False,
        'anomalies_details': [],
        'metric_deviations': {}
    }

    if not identity_metrics_timeline or len(identity_metrics_timeline) < 2:
        logger.warning("Недостаточно данных во временной линии метрик для валидации старения.")
        validation_results['overall_aging_consistency_score'] = 0.0
        validation_results['anomalies_details'].append("Недостаточно данных для анализа.")
        return validation_results

    # Сортируем данные по дате, чтобы гарантировать хронологический порядок
    sorted_dates = sorted(identity_metrics_timeline.keys())
    
    # Инициализируем TemporalAnalyzer с data_manager
    try:
        temporal_analyzer = TemporalAnalyzer(data_manager)
        temporal_analyzer.build_medical_aging_model() # Убедимся, что модель старения построена
    except Exception as e:
        logger.error(f"Ошибка инициализации TemporalAnalyzer: {e}")
        validation_results['overall_aging_consistency_score'] = 0.0
        validation_results['anomalies_details'].append(f"Ошибка инициализации анализатора: {e}")
        return validation_results


    # Определяем базовые метрики на основе первой доступной точки
    # Используем метрики с самой ранней даты как baseline
    first_date = sorted_dates[0]
    baseline_metrics = identity_metrics_timeline[first_date]

    if not baseline_metrics:
        logger.warning("Базовые метрики не найдены для первой даты.")
        validation_results['overall_aging_consistency_score'] = 0.0
        validation_results['anomalies_details'].append("Базовые метрики отсутствуют.")
        return validation_results

    all_deviations = []
    
    # Итерируемся по каждой точке во времени, начиная со второй
    for i in range(1, len(sorted_dates)):
        current_date = sorted_dates[i]
        current_metrics = identity_metrics_timeline[current_date]
        current_age = age_progression[i]

        if not current_metrics:
            logger.warning(f"Метрики отсутствуют для даты {current_date}.")
            continue

        # Прогнозируем ожидаемые метрики для текущего возраста, используя baseline
        predicted_metrics = temporal_analyzer.predict_expected_metrics_for_age(current_age, baseline_metrics)

        # Сравниваем фактические метрики с предсказанными
        deviations_for_period = {}
        period_deviations_sum = 0.0
        relevant_metrics_count = 0

        for metric_name, actual_value in current_metrics.items():
            if metric_name in predicted_metrics:
                predicted_value = predicted_metrics[metric_name]
                
                # Расчет отклонения (абсолютное или относительное)
                deviation = abs(actual_value - predicted_value)
                
                # Нормализация отклонения для агрегации
                normalized_deviation = deviation / (baseline_metrics.get(metric_name, 1.0) + 1e-6) # Защита от деления на ноль и отсутствующих метрик
                
                deviations_for_period[metric_name] = normalized_deviation
                all_deviations.append(normalized_deviation)

                period_deviations_sum += normalized_deviation
                relevant_metrics_count += 1
                
                # Детекция аномалий: Резкое изменение или "обратное старение"
                
                if metric_name in ['forehead_height_ratio', 'nose_width_ratio', 'mouth_width_ratio']: # Метрики мягких тканей
                    # Проверка на "обратное старение" (значительное снижение, когда ожидается рост или стабильность)
                    # Здесь мы смотрим на фактическое изменение (actual - predicted) и сравниваем с порогом отрицательного изменения
                    if (actual_value - predicted_value) < ANOMALY_DETECTION_THRESHOLDS['reverse_aging']:
                        validation_results['anomalies_detected'] = True
                        validation_results['anomalies_details'].append(
                            f"На дату {current_date.strftime('%Y-%m-%d')}: Обнаружено аномальное 'омоложение' по метрике {metric_name} (отклонение: {actual_value - predicted_value:.4f})."
                        )
                
                if metric_name in ['skull_width_ratio', 'temporal_bone_angle', 'zygomatic_arch_width', 'inter_pupillary_distance_ratio']: # Костные метрики, которые должны быть стабильны после 25
                    # Проверка на резкое изменение (если метрика должна быть стабильной)
                    if abs(actual_value - predicted_value) > ANOMALY_DETECTION_THRESHOLDS['rapid_change']:
                        validation_results['anomalies_detected'] = True
                        validation_results['anomalies_details'].append(
                            f"На дату {current_date.strftime('%Y-%m-%d')}: Резкое изменение костной метрики {metric_name} (отклонение: {actual_value - predicted_value:.4f})."
                        )
                
        if relevant_metrics_count > 0:
            avg_deviation_for_period = period_deviations_sum / relevant_metrics_count
            validation_results['metric_deviations'][current_date.strftime('%Y-%m-%d')] = deviations_for_period

    if all_deviations:
        mean_normalized_deviation = np.mean(all_deviations)
        validation_results['overall_aging_consistency_score'] = max(0.0, 1.0 - mean_normalized_deviation * MEDICAL_VALIDATOR_THRESHOLDS['AGING_CONSISTENCY_SCALE_FACTOR']) # Используем константу
        
        if validation_results['overall_aging_consistency_score'] < 0.6: # Порог для детектирования общей некорректности
            validation_results['anomalies_detected'] = True
            validation_results['anomalies_details'].append(
                f"Общая консистентность старения низкая: {validation_results['overall_aging_consistency_score']:.2f}"
            )
            
    return validation_results

# =====================
# Проверка неизменности костной структуры черепа после 25 лет
# =====================
def check_bone_structure_immutability(cranial_metrics_timeline, ages):
    """
    Проверяет неизменность костной структуры черепа после 25 лет
    любые изменения указывают на другого человека
    cranial_metrics_timeline: dict {metric_name: [values]}
    ages: список возрастов
    Возвращает: dict с флагами стабильности
    """
    results = {}
    for metric, values in cranial_metrics_timeline.items():
        values = np.array(values)
        adult_mask = np.array(ages) >= 25
        if np.sum(adult_mask) < 2:
            results[metric] = {'stable': True, 'reason': 'not_enough_adult_data'}
            continue
        adult_values = values[adult_mask]
        std = np.std(adult_values)
        if std > MEDICAL_VALIDATOR_THRESHOLDS['BONE_METRIC_STABILITY_THRESHOLD']:  # Используем константу
            results[metric] = {'stable': False, 'std': std}
        else:
            results[metric] = {'stable': True, 'std': std}
    return results

# =====================
# Анализ старения мягких тканей
# =====================
def analyze_soft_tissue_aging_patterns(soft_tissue_metrics, age_progression):
    """
    Анализ естественного старения мягких тканей:
    предсказуемое опущение, потеря эластичности, изменения объема
    soft_tissue_metrics: dict {metric_name: [values]}
    age_progression: список возрастов
    Возвращает: dict с трендами и аномалиями
    """
    results = {}
    for metric, values in soft_tissue_metrics.items():
        values = np.array(values)
        slope, intercept, r_value, p_value, std_err = stats.linregress(age_progression, values)
        # Ожидается отрицательный наклон (опущение, уменьшение)
        results[metric] = {'slope': slope, 'expected_negative': slope < MEDICAL_VALIDATOR_THRESHOLDS['SOFT_TISSUE_AGING_SLOPE_THRESHOLD'], 'p_value': p_value}
    return results

# =====================
# Исключение хирургических гипотез по временным интервалам
# =====================
def exclude_surgical_hypotheses_by_timeline(metrics_changes, time_intervals, medical_events):
    """
    Исключает хирургические гипотезы при недостаточных временных интервалах
    минимум 6 месяцев для серьезных операций без видимых следов
    metrics_changes: dict {period: change_value}
    time_intervals: dict {period: days}
    medical_events: dict {date: event_info}
    Возвращает: dict с флагами хирургии
    """
    results = {}
    for period, change in metrics_changes.items():
        interval = time_intervals.get(period, 0)
        if interval < MEDICAL_VALIDATOR_THRESHOLDS['SURGERY_MIN_INTERVAL_DAYS']:
            results[period] = {'surgery_possible': False, 'reason': 'interval_too_short'}
        else:
            # Проверка на наличие медицинских событий рядом
            event_found = False
            for date, event in medical_events.items():
                if abs((date - period[0]).days) < MEDICAL_VALIDATOR_THRESHOLDS['MEDICAL_EVENT_CORRELATION_DAYS'] or \
                   abs((date - period[1]).days) < MEDICAL_VALIDATOR_THRESHOLDS['MEDICAL_EVENT_CORRELATION_DAYS']:
                    event_found = True
                    break
            results[period] = {'surgery_possible': not event_found, 'event_found': event_found}
    return results

# =====================
# Проверка физиологических ограничений скорости изменений
# =====================
def validate_physiological_change_limits(metric_changes, change_velocities):
    """
    Проверяет соответствие физиологическим ограничениям скорости изменений
    выявляет изменения, превышающие биологические возможности
    metric_changes: dict {metric_name: [changes]}
    change_velocities: dict {metric_name: [velocities]}
    Возвращает: dict с флагами превышения
    """
    results = {}
    for metric, velocities in change_velocities.items():
        velocities = np.array(velocities)
        # Порог скорости изменения (примерно 0.02 в год)
        if np.any(np.abs(velocities) > MEDICAL_VALIDATOR_THRESHOLDS['PHYSIOLOGICAL_CHANGE_LIMIT']):
            results[metric] = {'exceeds_limit': True, 'max_velocity': np.max(np.abs(velocities))}
        else:
            results[metric] = {'exceeds_limit': False, 'max_velocity': np.max(np.abs(velocities))}
    return results

# =====================
# Корреляция аномалий с медицинскими событиями
# =====================
def correlate_anomalies_with_documented_health_events(anomaly_periods, health_events):
    """
    Сопоставляет аномалии с документированными проблемами здоровья
    исключает медицинские объяснения для необъяснимых изменений
    anomaly_periods: список дат или периодов
    health_events: dict {date: event_info}
    Возвращает: dict с корреляциями
    """
    results = {}
    for period in anomaly_periods:
        found = False
        for date, event in health_events.items():
            if abs((period - date).days) < MEDICAL_VALIDATOR_THRESHOLDS['MEDICAL_EVENT_CORRELATION_DAYS']:
                found = True
                results[period] = {'explained_by_event': True, 'event': event}
                break
        if not found:
            results[period] = {'explained_by_event': False}
    return results

# =====================
# Автоматическое генерирование медицинского отчета
# =====================
def auto_generate_medical_report(validation_results: Dict) -> str:
    """
    Автоматически генерирует медицинский отчет на основе результатов валидации.
    
    Args:
        validation_results (Dict): Результаты валидации, полученные от `validate_aging_consistency_for_identity`
                                   и, возможно, других медицинских валидаторов.

    Returns:
        str: Полный медицинский отчет в удобочитаемом формате.
    """
    report_parts = []

    report_parts.append("### Медицинский Отчет об Анализе Идентичности\n")
    report_parts.append(f"**Дата генерации отчета:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_parts.append("\n---\n")

    # 1. Общая оценка консистентности старения
    overall_score = validation_results.get('overall_aging_consistency_score', 0.0)
    if overall_score >= 0.8:
        report_parts.append(f"**Общая консистентность старения:** Высокая ({overall_score:.2f}). Изменения метрик соответствуют естественным процессам старения.\n")
    elif overall_score >= 0.6:
        report_parts.append(f"**Общая консистентность старения:** Средняя ({overall_score:.2f}). Большинство изменений согласуются с естественным старением, но есть незначительные отклонения.\n")
    else:
        report_parts.append(f"**Общая консистентность старения:** Низкая ({overall_score:.2f}). Обнаружены значительные расхождения между фактическими и ожидаемыми возрастными изменениями. Это требует дальнейшего изучения.\n")
    report_parts.append("\n")

    # 2. Детали обнаруженных аномалий
    anomalies_detected = validation_results.get('anomalies_detected', False)
    anomalies_details = validation_results.get('anomalies_details', [])

    if anomalies_detected and anomalies_details:
        report_parts.append("**Обнаруженные Аномалии:\n**")
        for detail in anomalies_details:
            report_parts.append(f"- {detail}\n")
    else:
        report_parts.append("**Обнаруженные Аномалии:** Аномальных изменений, не соответствующих естественному старению, не выявлено.\n")
    report_parts.append("\n")

    # 3. Детали отклонений метрик
    metric_deviations = validation_results.get('metric_deviations', {})
    if metric_deviations:
        report_parts.append("**Детали Отклонений Метрик по Датам:\n**")
        for date_str, deviations in metric_deviations.items():
            report_parts.append(f"**На дату {date_str}:**\n")
            for metric, deviation_value in deviations.items():
                report_parts.append(f"  - Метрика '{metric}': Отклонение {deviation_value:.4f}\n")
        report_parts.append("\n")

    # 4. Дополнительные секции (если есть)
    # Пример: если есть результаты check_bone_structure_immutability
    bone_structure_results = validation_results.get('bone_structure_immutability', None)
    if bone_structure_results:
        report_parts.append("**Анализ Костной Структуры Черепа (после 25 лет):\n**")
        for metric, data in bone_structure_results.items():
            status = "Стабильна" if data['stable'] else "Нестабильна"
            report_parts.append(f"- Метрика '{metric}': {status} (Стандартное отклонение: {data['std']:.4f}).\n")
            if not data['stable']:
                report_parts.append(f"  *Примечание: Нестабильность костной метрики может указывать на изменения, не связанные с естественным старением.*\n")
        report_parts.append("\n")

    # Пример: если есть результаты analyze_soft_tissue_aging_patterns
    soft_tissue_aging_results = validation_results.get('soft_tissue_aging_patterns', None)
    if soft_tissue_aging_results:
        report_parts.append("**Анализ Старения Мягких Тканей:\n**")
        for metric, data in soft_tissue_aging_results.items():
            trend_status = "Соответствует естественному опущению" if data['expected_negative'] else "Не соответствует ожидаемому опущению"
            report_parts.append(f"- Метрика '{metric}': Наклон тренда {data['slope']:.4f}. Статус: {trend_status}. (P-value: {data['p_value']:.4f})\n")
            if not data['expected_negative']:
                report_parts.append(f"  *Примечание: Отклонение от ожидаемого тренда опущения может быть признаком неестественных изменений.*\n")
        report_parts.append("\n")

    report_parts.append("---\n\n")

    return "".join(report_parts)

# =====================
# Балл биологической правдоподобности изменений
# =====================
def calculate_biological_plausibility_score(changes_data: Dict) -> float:
    """
    Вычисляет балл биологической правдоподобности изменений
    changes_data: dict {metric: [changes]}
    Возвращает: float (0-1)
    """
    plaus_scores = []
    for metric, changes in changes_data.items():
        changes = np.array(changes)
        # Чем меньше изменений, тем выше правдоподобие
        plaus = 1.0 / (1.0 + np.mean(np.abs(changes)))
        plaus_scores.append(plaus)
    if plaus_scores:
        return float(np.mean(plaus_scores))
    return 0.0

def validate_soft_tissue_aging_rate(age_progression: List[float],
                                   soft_tissue_metrics: Dict) -> Dict:
    """Валидирует скорость старения мягких тканей"""
    results = {}
    for metric_name, values in soft_tissue_metrics.items():
        if len(age_progression) < 2 or len(values) < 2:
            results[metric_name] = {'anomaly_detected': False, 'reason': 'Недостаточно данных'}
            continue

        # Ожидаемое изменение в год для эластичности (как пример)
        expected_change_per_year = AgingModel.ELASTICITY_LOSS_PER_YEAR

        # Расчет фактического наклона (скорости изменения) метрики
        try:
            # Использование linregress из scipy.stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(age_progression, values)
            actual_slope = slope
        except ValueError:
            results[metric_name] = {'anomaly_detected': False, 'reason': 'Ошибка при расчете тренда'}
            continue

        # Проверка на превышение ожидаемого изменения в N раз (например, в 2 раза)
        # Здесь важно, что metric_name могут быть разными и иметь разную логику для 'изменений'.
        # Для примера, если метрика должна уменьшаться (эластичность), то смотрим на абсолютное значение.
        # Для других метрик может быть другая логика.
        if metric_name == 'elasticity_score': # Предположим, у нас есть такая метрика
            if abs(actual_slope) > expected_change_per_year * 2:  # Превышение в 2 раза
                results[metric_name] = {
                    'anomaly_detected': True,
                    'metric': metric_name,
                    'expected_rate': expected_change_per_year,
                    'actual_rate': actual_slope,
                    'reason': 'Скорость изменения эластичности превышает ожидаемую'
                }
            else:
                results[metric_name] = {'anomaly_detected': False, 'reason': 'Скорость изменения эластичности в норме'}
        elif metric_name == 'tissue_sagging': # Предположим, у нас есть такая метрика
            expected_sagging_per_year = AgingModel.TISSUE_SAGGING_PER_YEAR
            if actual_slope > expected_sagging_per_year * 2: # Если провисание (увеличение значения)
                 results[metric_name] = {
                    'anomaly_detected': True,
                    'metric': metric_name,
                    'expected_rate': expected_sagging_per_year,
                    'actual_rate': actual_slope,
                    'reason': 'Скорость провисания мягких тканей превышает ожидаемую'
                }
            else:
                results[metric_name] = {'anomaly_detected': False, 'reason': 'Скорость провисания мягких тканей в норме'}
        else:
            results[metric_name] = {'anomaly_detected': False, 'reason': 'Метрика не настроена для проверки темпов старения'}

    return results
