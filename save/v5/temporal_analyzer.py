# temporal_analyzer.py
# Хронологический анализ и модель старения

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Tuple, Optional
import logging
from statsmodels.tsa.seasonal import seasonal_decompose
import ruptures as rpt
from collections import Counter

from core_config import PUTIN_BIRTH_DATE, AgingModel, ANOMALY_DETECTION_THRESHOLDS, AGING_COEFFICIENTS
from data_manager import DataManager

logger = logging.getLogger(__name__)

class TemporalAnalyzer:
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.aging_model = None
        self.medical_events = self._load_medical_events()
        
    def build_medical_aging_model(self) -> Dict:
        """Строит функции для моделирования старения на основе AgingModel."""
        
        # Функции для потери эластичности кожи
        def elasticity_coefficient(age):
            if age < AgingModel.BONE_STABILITY_THRESHOLD: # До 25 лет эластичность высокая
                return 1.0
            # После 40 лет идет активная потеря
            if age > 40:
                return max(0.5, 1.0 - AgingModel.ELASTICITY_LOSS_PER_YEAR * (age - 40))
            return 1.0 # Между 25 и 40 - медленные изменения

        # Функции для опущения мягких тканей
        def sagging_offset(age):
            if age < AgingModel.BONE_STABILITY_THRESHOLD:
                return 0.0
            if age > 40:
                return AgingModel.TISSUE_SAGGING_PER_YEAR * (age - 40)
            return 0.0 # Между 25 и 40 - медленные изменения

        # Функции для проверки стабильности костной структуры
        def bone_stability_check(age):
            return age >= AgingModel.BONE_STABILITY_THRESHOLD

        # Функции для межзрачкового расстояния (остается относительно стабильным)
        def ipd_variation_check(ipd_values, ages):
            # IPD не должно изменяться более чем на 2% после 25 лет
            if ages and any(a >= AgingModel.BONE_STABILITY_THRESHOLD for a in ages):
                if ipd_values and np.mean(ipd_values) > 0 and (max(ipd_values) - min(ipd_values)) / np.mean(ipd_values) > AgingModel.IPD_VARIATION_MAX:
                    return False # Превышен допустимый лимит изменения IPD
            return True

        self.aging_model = {
            'elasticity_loss_per_year': elasticity_coefficient,
            'tissue_sagging_per_year': sagging_offset,
            'bone_stability_check': bone_stability_check,
            'ipd_variation_check': ipd_variation_check
        }
        logging.info("Медицинская модель старения построена.")
        return self.aging_model
    
    def calculate_age_on_each_date(self, photo_dates: List[datetime], birth_date: datetime) -> List[float]:
        """Рассчитывает возраст человека на каждую дату фотографии."""
        ages = []
        for photo_date in photo_dates:
            # Разница в годах с учетом дробной части
            age = (photo_date - birth_date).days / 365.25
            ages.append(age)
        return ages

    def predict_expected_metrics_for_age(self, age: float, baseline_metrics: Dict) -> Dict:
        """Прогнозирует ожидаемые значения всех метрик для данного возраста"""
        
        if self.aging_model is None:
            self.build_medical_aging_model() # Убедимся, что модель построена
        
        predicted_metrics = {}
        
        for metric_name, baseline_value in baseline_metrics.items():
            predicted_value = self._predict_single_metric(metric_name, baseline_value, age)
            predicted_metrics[metric_name] = predicted_value
        
        return predicted_metrics
    
    def _predict_single_metric(self, metric_name: str, baseline_value: float, age: float) -> float:
        """
        Прогнозирует изменение отдельной метрики с учетом возраста и физиологии.
        Учитывает стабильность костной структуры, потерю эластичности и опущение мягких тканей.
        """
        if self.aging_model is None:
            self.build_medical_aging_model() # Убедимся, что модель построена

        predicted_value = baseline_value

        # Общие физиологические функции из aging_model
        elasticity_factor_func = self.aging_model['elasticity_loss_per_year']
        sagging_offset_func = self.aging_model['tissue_sagging_per_year']
        bone_stable_func = self.aging_model['bone_stability_check']

        # **1. Костные метрики (стабилизируются после BONE_STABILITY_THRESHOLD)**
        # Эти метрики в основном зависят от костной структуры и незначительно меняются после определенного возраста.
        bone_metrics = [
            'skull_width_ratio', 'temporal_bone_angle', 'zygomatic_arch_width',
            'orbital_depth', 'cephalic_index', 'inter_pupillary_distance_ratio',
            'gonial_angle_asymmetry', 'zygomatic_angle', 'jaw_angle_ratio',
            'mandibular_symphysis_angle'
        ]

        if metric_name in bone_metrics:
            if bone_stable_func(age):
                predicted_value = baseline_value # Костная структура стабильна
            else:
                # До BONE_STABILITY_THRESHOLD, незначительный рост
                # Применяем простую модель роста до стабилизации костей
                growth_factor = 1.0 + (AgingModel.BONE_STABILITY_THRESHOLD - age) * 0.001 if age < AgingModel.BONE_STABILITY_THRESHOLD else 1.0
                predicted_value = baseline_value * growth_factor

        # **2. Мягкотканные метрики (изменяются с возрастом: потеря эластичности, опущение)**
        # Эти метрики подвержены изменениям из-за гравитации, потери коллагена и жира.
        soft_tissue_metrics = [
            'occipital_curvature', 'golden_ratio_deviation', 'nasolabial_angle',
            'orbital_index', 'forehead_height_ratio', 'chin_projection_ratio'
        ]

        if metric_name in soft_tissue_metrics:
            # Применение коэффициентов старения (эластичность и опущение)
            elasticity_loss = elasticity_factor_func(age) # Например, 1.0 для 0% потери, 0.8 для 20% потери
            sagging_change = sagging_offset_func(age) # Положительное значение для опущения

            # Базовое изменение
            predicted_value = baseline_value * elasticity_loss - sagging_change * 0.001 # 0.001 для масштабирования

            # Применение зональных коэффициентов AGING_COEFFICIENTS
            aging_coefficient = 1.0 # По умолчанию
            if metric_name == 'forehead_height_ratio':
                aging_coefficient = AGING_COEFFICIENTS.get('forehead', 1.0)
            elif metric_name == 'nasolabial_angle' or metric_name == 'golden_ratio_deviation':
                # Эти метрики могут быть связаны с зоной рта/носа, где мягкие ткани играют роль
                aging_coefficient = AGING_COEFFICIENTS.get('mouth', 1.0) 
            elif metric_name == 'orbital_index':
                # Орбитальный индекс может быть связан с мягкими тканями вокруг глаз
                aging_coefficient = AGING_COEFFICIENTS.get('eye_area', 1.0)
            elif metric_name == 'chin_projection_ratio':
                aging_coefficient = AGING_COEFFICIENTS.get('chin_jaw', 1.0)
            elif metric_name == 'occipital_curvature':
                # Кривизна затылка: хотя это часть черепа, мягкие ткани могут влиять на восприятие
                aging_coefficient = AGING_COEFFICIENTS.get('chin_jaw', 1.0) # Привязываем к общей зоне лица

            predicted_value *= aging_coefficient # Применяем зональный коэффициент

        # Защита от нереалистичных значений (метрики должны быть положительными и в разумных пределах)
        return max(0.01, predicted_value)

    def detect_temporal_anomalies_in_metrics(self, metrics_timeline: Dict, predicted_timeline: Dict) -> Dict:
        """Выявляет аномалии во временной последовательности"""
        
        anomalies = {}
        
        for metric_name in metrics_timeline.keys():
            if metric_name not in predicted_timeline:
                continue
            
            actual_values = np.array(metrics_timeline[metric_name])
            predicted_values = np.array(predicted_timeline[metric_name])
            dates = list(range(len(actual_values)))  # Временные индексы
            
            # Z-score анализ отклонений
            deviations = actual_values - predicted_values
            
            # Добавляем проверку на нулевое стандартное отклонение для zscore
            if np.std(deviations) == 0:
                z_scores = np.zeros_like(deviations) # Все отклонения равны, z-score = 0
            else:
                z_scores = stats.zscore(deviations)
            
            # Детекция аномалий (|z-score| > ANOMALY_DETECTION_THRESHOLDS['Z_SCORE_ANOMALY_THRESHOLD'])
            anomaly_indices = np.where(np.abs(z_scores) > ANOMALY_DETECTION_THRESHOLDS['Z_SCORE_ANOMALY_THRESHOLD'])[0]
            
            # Анализ скорости изменений
            if len(actual_values) > 1:
                change_rates = np.diff(actual_values)
                change_rate_std = np.std(change_rates)
                if change_rate_std == 0: # Избегаем деления на ноль
                    rapid_changes = []
                else:
                    rapid_changes = np.where(np.abs(change_rates) > ANOMALY_DETECTION_THRESHOLDS['RAPID_CHANGE_STD_MULTIPLIER'] * change_rate_std)[0]
            else:
                rapid_changes = []
            
            if len(anomaly_indices) > 0 or len(rapid_changes) > 0:
                explanation = self._generate_temporal_anomaly_explanation(
                    metric_name, deviations, z_scores, anomaly_indices, rapid_changes
                )
                anomalies[metric_name] = {
                    'anomaly_indices': anomaly_indices.tolist(),
                    'z_scores': z_scores[anomaly_indices].tolist(),
                    'rapid_change_indices': rapid_changes.tolist(),
                    'deviation_magnitude': np.mean(np.abs(deviations[anomaly_indices])) if len(anomaly_indices) > 0 else 0,
                    'significance_level': self._calculate_significance_level(z_scores),
                    'explanation': explanation # Добавляем объяснение
                }
        
        return anomalies

    def _generate_temporal_anomaly_explanation(self, metric_name: str, deviations: np.ndarray, 
                                                z_scores: np.ndarray, anomaly_indices: np.ndarray, 
                                                rapid_change_indices: np.ndarray) -> str:
        """
        Генерирует текстовое объяснение для обнаруженных временных аномалий,
        используя готовые шаблоны.
        """
        explanations = []

        # Шаблоны для различных типов аномалий
        templates = {
            'significant_deviation': "Обнаружено значительное отклонение метрики '{metric_name}' от ожидаемого значения. Это может указывать на необычные изменения или ошибки в данных.",
            'rapid_change': "Зафиксировано резкое изменение метрики '{metric_name}'. Такое быстрое изменение может быть связано с внешними воздействиями или внезапными физиологическими процессами.",
            'consistent_positive_deviation': "Метрика '{metric_name}' показывает постоянное завышенное значение по сравнению с прогнозом. Это может указывать на долгосрочный тренд или систематическую разницу.",
            'consistent_negative_deviation': "Метрика '{metric_name}' демонстрирует постоянное заниженное значение относительно прогноза. Возможно, это свидетельствует о долгосрочной динамике или неточности базовых данных.",
            'fluctuation': "Наблюдаются выраженные колебания метрики '{metric_name}'. Повышенная нестабильность может быть признаком проблем с данными или непредсказуемых изменений."
        }

        # Анализ отклонений
        if len(anomaly_indices) > 0:
            # Односторонние отклонения
            positive_deviations = deviations[anomaly_indices][z_scores[anomaly_indices] > 0]
            negative_deviations = deviations[anomaly_indices][z_scores[anomaly_indices] < 0]

            if len(positive_deviations) > len(anomaly_indices) * 0.7: # Более 70% аномалий положительные
                explanations.append(templates['consistent_positive_deviation'].format(metric_name=metric_name))
            elif len(negative_deviations) > len(anomaly_indices) * 0.7:
                explanations.append(templates['consistent_negative_deviation'].format(metric_name=metric_name))
            else:
                explanations.append(templates['significant_deviation'].format(metric_name=metric_name))

        # Анализ резких изменений
        if len(rapid_change_indices) > 0:
            explanations.append(templates['rapid_change'].format(metric_name=metric_name))

        # Дополнительная проверка на флуктуации (если нет явных аномалий/резких изменений)
        if not explanations and np.std(deviations) > np.mean(np.abs(deviations)) * 1.5: # Высокое стандартное отклонение относительно среднего абсолютного отклонения
            explanations.append(templates['fluctuation'].format(metric_name=metric_name))

        if not explanations:
            return f"Аномалии метрики '{metric_name}' обнаружены, но их тип не классифицирован по шаблонам."

        return " ".join(explanations)

    def build_identity_appearance_timeline(self, identity_clusters: Dict, dates: List[datetime]) -> Dict:
        """Строит детальную временную линию появления каждой личности."""
        
        appearance_timeline = {}
        
        for cluster_id, cluster_info in identity_clusters.items():
            if cluster_id == -1:  # Пропускаем outliers
                continue
            
            # Извлечение дат для кластера
            cluster_dates = cluster_info.get('appearance_dates', [])
            
            if len(cluster_dates) == 0:
                continue
            
            # Анализ периодов активности
            activity_periods = self._analyze_activity_periods(cluster_dates)
            
            # Анализ интервалов отсутствия
            absence_intervals = self._analyze_absence_intervals(cluster_dates)
            
            appearance_timeline[cluster_id] = {
                'first_appearance': min(cluster_dates),
                'last_appearance': max(cluster_dates),
                'total_duration_days': (max(cluster_dates) - min(cluster_dates)).days,
                'appearance_count': len(cluster_dates),
                'activity_periods': activity_periods,
                'absence_intervals': absence_intervals,
                'activity_density': self._calculate_activity_density(cluster_dates),
                'regularity_score': self._calculate_regularity_score(cluster_dates)
            }
        
        return appearance_timeline

    def analyze_identity_switching_patterns(self, appearance_timeline: Dict) -> Dict:
        """Анализирует паттерны смены личностей:
        выявляет систематические интервалы замены (каждые 3-4 месяца)
        корреляция с историческими событиями
        """
        switching_patterns = {}
        for identity_id, data in appearance_timeline.items():
            appearance_dates = sorted(data['appearance_dates'])
            if len(appearance_dates) < 2:
                switching_patterns[identity_id] = {'pattern_detected': False, 'details': 'Недостаточно данных для анализа.'}
                continue

            switch_intervals = []
            for i in range(len(appearance_dates) - 1):
                interval = (appearance_dates[i+1] - appearance_dates[i]).days
                switch_intervals.append(interval)

            # Расширенный анализ: поиск систематических интервалов и более сложных паттернов
            systematic_pattern_detected = False
            if len(switch_intervals) > 0:
                # Пример: Поиск часто встречающихся интервалов
                interval_counts = Counter(switch_intervals)
                most_common_interval = interval_counts.most_common(1)

                if most_common_interval and most_common_interval[0][1] >= 2: # Если интервал повторяется хотя бы 2 раза
                    systematic_pattern_detected = True
                    # Можно добавить проверку на специфические интервалы (90-120 дней)
                    if any(90 <= iv <= 120 for iv in switch_intervals):
                        systematic_pattern_detected = True # Усиление флага

            # Корреляция с историческими событиями - используем DataManager
            correlated_with_events = False
            if appearance_dates and self.data_manager:
                # Получаем релевантные исторические события
                relevant_historical_events = self.data_manager.correlate_with_historical_events(appearance_dates)
                if relevant_historical_events:
                    correlated_with_events = True

            switching_patterns[identity_id] = {
                'pattern_detected': systematic_pattern_detected,
                'switch_intervals_days': switch_intervals,
                'correlated_with_events': correlated_with_events,
                'details': 'Анализ паттернов смены личностей выполнен с расширенной логикой.'
            }
        return switching_patterns

    def correlate_anomalies_with_medical_events(self, anomaly_dates: List[datetime], medical_events: Dict) -> Dict:
        """Сопоставляет аномалии с медицинскими событиями"""
        
        correlations = {}
        
        for anomaly_date in anomaly_dates:
            matched_events = []
            for event_date_str, event_info in medical_events.items():
                event_date = datetime.strptime(event_date_str, '%Y-%m-%d')
                time_diff_days = abs((anomaly_date - event_date).days)
                
                # Если событие произошло в пределах 30 дней от аномалии
                if time_diff_days <= 30:
                    matched_events.append({
                        'event': event_info.get('type', event_info.get('event', 'Unknown')), # Используем 'type' или 'event'
                        'description': event_info.get('description', event_info.get('event', 'Unknown')),
                        'days_diff': time_diff_days,
                        'event_date': event_date # Добавляем event_date для корректного отображения
                    })
            if matched_events:
                correlations[anomaly_date.strftime('%Y-%m-%d')] = matched_events
        
        return correlations

    def _load_medical_events(self) -> Dict:
        """Загружает медицинские события из хранилища данных."""
        # Эта функция должна загружать реальные медицинские события, 
        # например, из базы данных или файла. Для демонстрации 
        # возвращаем пустой словарь.
        return {}

    def _analyze_activity_periods(self, dates: List[datetime]) -> List[Dict]:
        """Анализирует периоды активности на временной шкале."""
        if not dates:
            return []

        dates = sorted(dates)
        periods = []
        current_start = dates[0]
        current_end = dates[0]

        for i in range(1, len(dates)):
            if (dates[i] - current_end).days <= ANOMALY_DETECTION_THRESHOLDS['ACTIVITY_PERIOD_GAP_DAYS']:
                current_end = dates[i]
            else:
                periods.append({
                    'start_date': current_start,
                    'end_date': current_end,
                    'duration_days': (current_end - current_start).days
                })
                current_start = dates[i]
                current_end = dates[i]

        periods.append({
            'start_date': current_start,
            'end_date': current_end,
            'duration_days': (current_end - current_start).days
        })
        return periods

    def _analyze_absence_intervals(self, dates: List[datetime]) -> List[Dict]:
        """Анализирует периоды отсутствия на временной шкале."""
        if len(dates) < 2:
            return []

        dates = sorted(dates)
        absence_intervals = []

        for i in range(len(dates) - 1):
            interval_days = (dates[i+1] - dates[i]).days
            if interval_days > ANOMALY_DETECTION_THRESHOLDS['ABSENCE_INTERVAL_THRESHOLD_DAYS']:
                absence_intervals.append({
                    'start_of_absence': dates[i],
                    'end_of_absence': dates[i+1],
                    'duration_days': interval_days
                })
        return absence_intervals

    def _calculate_activity_density(self, dates: List[datetime]) -> float:
        """Рассчитывает плотность активности (количество появлений в месяц/год)."""
        if not dates: return 0.0
        min_date = min(dates)
        max_date = max(dates)
        total_days = (max_date - min_date).days + 1
        if total_days == 0: return 0.0
        return len(dates) / (total_days / 30.0) # Среднее количество появлений в месяц

    def _calculate_regularity_score(self, dates: List[datetime]) -> float:
        """Оценивает регулярность появлений личности."""
        if len(dates) < 2: return 1.0
        dates_in_seconds = np.array([d.timestamp() for d in dates])
        intervals = np.diff(dates_in_seconds)
        if intervals.size == 0 or np.mean(intervals) == 0: return 1.0
        # Чем меньше стандартное отклонение интервалов по сравнению со средним, тем регулярнее
        cv = np.std(intervals) / np.mean(intervals)
        return max(0.0, 1.0 - cv) # Score от 0 до 1

    def _calculate_significance_level(self, z_scores: np.ndarray) -> float:
        """Рассчитывает уровень значимости (p-value) для обнаруженных аномалий.
        Использует нормальное распределение для z-scores.
        """
        if z_scores.size == 0: return 1.0
        # Для двустороннего теста
        p_value = np.mean([2 * (1 - stats.norm.cdf(np.abs(z))) for z in z_scores])
        return p_value

    def seasonal_decomposition_analysis(self, metrics_timeline: Dict) -> Dict:
        """Выполняет сезонную декомпозицию временных рядов метрик.
        Возвращает тренд, сезонность и остаток.
        """
        decomposition_results = {}
        for metric_name, values in metrics_timeline.items():
            if len(values) < 2 * 12: # Нужен хотя бы 2 полных сезона для надежной декомпозиции (если сезонность = 12)
                logging.warning(f"Недостаточно данных для сезонной декомпозиции метрики {metric_name}.")
                decomposition_results[metric_name] = {'trend': [], 'seasonal': [], 'residual': []}
                continue
            
            # Для временных рядов без явной сезонности, можно использовать additiv_decomposition
            # или просто оценить тренд.
            # Здесь для простоты предполагаем аддитивную модель без строгой сезонности
            # Можно использовать statsmodels.tsa.seasonal.seasonal_decompose
            try:
                # Если данные не являются pd.Series, преобразуем
                series = pd.Series(values)
                # Определяем период сезонности. Если это ежедневные данные, и мы ищем годовую сезонность, то 365.
                # Для произвольных временных рядов, без явной сезонности, это может быть сложнее.
                # Здесь мы можем предположить, что данных достаточно для обнаружения тренда.
                # Для более надежного определения периода, можно использовать ACF/PACF.
                # Пока установим period=1 для оценки общего тренда.
                result = seasonal_decompose(series, model='additive', period=1)
                decomposition_results[metric_name] = {
                    'trend': result.trend.tolist(),
                    'seasonal': result.seasonal.tolist(),
                    'residual': result.resid.tolist()
                }
            except Exception as e:
                logging.error(f"Ошибка при сезонной декомпозиции метрики {metric_name}: {e}")
                decomposition_results[metric_name] = {'trend': [], 'seasonal': [], 'residual': []}

        return decomposition_results

    def detect_change_points(self, time_series_data: Dict) -> Dict:
        """Обнаруживает точки изменения в метриках (например, с помощью CUSUM или Pelt)."""
        change_points_results = {}
        for metric_name, values in time_series_data.items():
            if len(values) < 5: # Нужно достаточно точек для анализа
                change_points_results[metric_name] = {'detected': False, 'points': []}
                continue
            
            # Пример использования CUSUM (Cumulative Sum) для обнаружения изменения среднего
            # Можно использовать более продвинутые библиотеки, такие как ruptures
            # Здесь простая реализация CUSUM для демонстрации
            data = np.array(values)
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            if std_val == 0: # Если нет вариации, нет и точек изменения
                change_points_results[metric_name] = {'detected': False, 'points': []}
                continue

            cusum_pos = np.zeros_like(data, dtype=float)
            cusum_neg = np.zeros_like(data, dtype=float)
            
            for i in range(1, len(data)):
                cusum_pos[i] = max(0, cusum_pos[i-1] + (data[i] - mean_val) / std_val)
                cusum_neg[i] = min(0, cusum_neg[i-1] + (data[i] - mean_val) / std_val)
            
            # Пороги для обнаружения точек изменения (можно настроить)
            threshold = 3.0 * np.sqrt(len(data)) # Примерный порог
            
            detected_points = []
            if np.max(cusum_pos) > threshold or np.min(cusum_neg) < -threshold:
                # Найдем индексы, где превышен порог
                pos_breaches = np.where(cusum_pos > threshold)[0]
                neg_breaches = np.where(cusum_neg < -threshold)[0]
                
                all_breaches = np.unique(np.concatenate((pos_breaches, neg_breaches))).tolist()
                detected_points = sorted(all_breaches)

            change_points_results[metric_name] = {
                'detected': len(detected_points) > 0,
                'points': detected_points
            }
        return change_points_results

    def calculate_temporal_stability_index(self, metrics_history: Dict) -> Dict:
        """Рассчитывает индекс временной стабильности для каждой метрики.
        Основан на средней абсолютной разнице между последовательными значениями.
        """
        stability_indices = {}
        for metric_name, values in metrics_history.items():
            if len(values) < 2:
                stability_indices[metric_name] = 1.0 # Полностью стабилен, если нет изменений
                continue

            # Среднее абсолютное изменение между последовательными точками
            diffs = np.abs(np.diff(values))
            mean_diff = np.mean(diffs)
            
            # Нормализация: чем меньше mean_diff, тем выше стабильность. 
            # Используем инверсию, чтобы 1.0 было для высокой стабильности.
            # Максимально допустимый дрейф должен быть определен эмпирически. 
            # Например, если 0.1 - это критический дрейф, то: max(0, 1 - (mean_diff / 0.1))
            max_allowed_drift = 0.05 # Примерный порог для допустимого дрейфа метрики
            stability_score = max(0.0, 1.0 - (mean_diff / max_allowed_drift)) # Ограничиваем от 0 до 1
            
            stability_indices[metric_name] = stability_score
            
        return stability_indices
