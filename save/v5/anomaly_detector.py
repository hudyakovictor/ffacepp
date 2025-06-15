# anomaly_detector.py

# Детектор аномалий и каскадная верификация

import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_distances
from core_config import (
    AUTHENTICITY_WEIGHTS, 
    MASK_DETECTION_THRESHOLDS, 
    CASCADE_VERIFICATION_THRESHOLDS, 
    ANOMALY_DETECTION_ADVANCED_THRESHOLDS,
    EMBEDDING_ANALYSIS_THRESHOLDS,
    ANOMALY_DETECTION_THRESHOLDS,
    TECHNOLOGY_BREAKTHROUGH_YEARS
)

class AnomalyDetector:
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.bayesian_priors = {'same_person': 0.5, 'different_person': 0.5}

    def apply_bayesian_identity_analysis(self, evidence_dict: Dict, prior_probabilities: Dict) -> Dict:
        """Байесовский анализ для каждой идентифицированной личности"""
        bayesian_results = {}
        
        for identity_id, evidence in evidence_dict.items():
            # Извлечение данных
            geometry_evidence = evidence.get('geometry_score', 0.5)
            embedding_evidence = evidence.get('embedding_score', 0.5)
            texture_evidence = evidence.get('texture_score', 0.5)
            temporal_evidence = evidence.get('temporal_consistency', 0.5)
            
            # Расчет likelihood для одного человека
            likelihood_same = self._calculate_likelihood_same_person(
                geometry_evidence, embedding_evidence, texture_evidence, temporal_evidence
            )
            
            # Расчет likelihood для разных людей
            likelihood_different = self._calculate_likelihood_different_person(
                geometry_evidence, embedding_evidence, texture_evidence, temporal_evidence
            )
            
            # Применение формулы Байеса
            prior_same = prior_probabilities.get('same_person', 0.5)
            prior_different = prior_probabilities.get('different_person', 0.5)
            
            # Posterior вероятности
            evidence_total = (likelihood_same * prior_same + 
                            likelihood_different * prior_different)
            
            posterior_same = (likelihood_same * prior_same) / evidence_total
            posterior_different = (likelihood_different * prior_different) / evidence_total
            
            bayesian_results[identity_id] = {
                'posterior_same_person': posterior_same,
                'posterior_different_person': posterior_different,
                'likelihood_ratio': likelihood_same / likelihood_different if likelihood_different > 0 else float('inf'),
                'confidence': max(posterior_same, posterior_different),
                'decision': 'same_person' if posterior_same > posterior_different else 'different_person'
            }
        
        return bayesian_results

    def perform_cascade_verification(self, geometry_score: float, embedding_score: float,
                                   texture_score: float, temporal_score: float) -> Dict:
        """Каскадная верификация трех независимых систем"""
        
        # Первый уровень: геометрическая верификация
        geometry_passed = geometry_score >= CASCADE_VERIFICATION_THRESHOLDS['geometry']
        
        # Второй уровень: эмбеддинг верификация
        embedding_passed = embedding_score >= CASCADE_VERIFICATION_THRESHOLDS['embedding']
        
        # Третий уровень: текстурная верификация
        texture_passed = texture_score >= CASCADE_VERIFICATION_THRESHOLDS['texture']
        
        # Четвертый уровень: временная консистентность
        temporal_passed = temporal_score >= CASCADE_VERIFICATION_THRESHOLDS['temporal']
        
        # Подсчет пройденных уровней
        levels_passed = sum([geometry_passed, embedding_passed, texture_passed, temporal_passed])
        
        # Итоговая аутентичность по формуле
        final_authenticity = self.calculate_identity_authenticity_score(
            geometry_score, embedding_score, texture_score, temporal_score
        )
        
        return {
            'geometry_passed': geometry_passed,
            'embedding_passed': embedding_passed,
            'texture_passed': texture_passed,
            'temporal_passed': temporal_passed,
            'levels_passed': levels_passed,
            'final_authenticity_score': final_authenticity,
            'verification_result': self._interpret_authenticity_score(final_authenticity),
            'cascade_confidence': levels_passed / 4.0
        }

    def calculate_identity_authenticity_score(self, geometry: float, embedding: float,
                                            texture: float, temporal_consistency: float) -> float:
        """Формула аутентичности с весами 0.3×Геометрия + 0.3×Эмбеддинг + 0.2×Текстура + 0.2×Временная"""
        
        authenticity = (
            AUTHENTICITY_WEIGHTS['geometry'] * geometry +
            AUTHENTICITY_WEIGHTS['embedding'] * embedding +
            AUTHENTICITY_WEIGHTS['texture'] * texture +
            AUTHENTICITY_WEIGHTS['temporal_consistency'] * temporal_consistency
        )
        
        return np.clip(authenticity, 0.0, 1.0)

    def _interpret_authenticity_score(self, score: float) -> str:
        """Интерпретация балла аутентичности"""
        if score < ANOMALY_DETECTION_ADVANCED_THRESHOLDS['MASK_OR_DOUBLE_THRESHOLD']:
            return "mask_or_double"  # Маска/двойник
        elif score < ANOMALY_DETECTION_ADVANCED_THRESHOLDS['REQUIRES_ANALYSIS_THRESHOLD']:
            return "requires_analysis"  # Требует анализа
        else:
            return "authentic_face"  # Подлинное лицо

    def detect_surgical_intervention_evidence(self, metrics_sequence: List[Dict],
                                            time_intervals: List[int]) -> Dict:
        """Выявляет признаки хирургического вмешательства"""
        surgical_evidence = {}
        
        for i in range(1, len(metrics_sequence)):
            current_metrics = metrics_sequence[i]
            previous_metrics = metrics_sequence[i-1]
            interval_days = time_intervals[i-1]
            
            # Анализ резких изменений
            changes = self._calculate_metric_changes(current_metrics, previous_metrics)
            
            # Признаки хирургического вмешательства
            swelling_indicators = self._detect_swelling_patterns(changes)
            asymmetry_changes = self._detect_asymmetry_changes(changes)
            healing_dynamics = self._analyze_healing_dynamics(changes, interval_days)
            
            # Исключение хирургии при коротких интервалах
            surgery_possible = interval_days >= ANOMALY_DETECTION_ADVANCED_THRESHOLDS['SURGERY_MIN_INTERVAL_DAYS_ANOMALY']
            
            if any([swelling_indicators, asymmetry_changes]) and surgery_possible:
                surgical_evidence[f'period_{i}'] = {
                    'swelling_detected': swelling_indicators,
                    'asymmetry_changes': asymmetry_changes,
                    'healing_dynamics': healing_dynamics,
                    'interval_sufficient': surgery_possible,
                    'surgery_likelihood': self._calculate_surgery_likelihood(
                        swelling_indicators, asymmetry_changes, healing_dynamics, interval_days
                    )
                }
        
        return surgical_evidence

    def analyze_mask_technology_evolution(self, texture_scores_timeline: List[float],
                                        dates: List[datetime]) -> Dict:
        """Анализирует эволюцию качества масок во времени"""
        evolution_analysis = {}
        
        # Группировка по годам
        yearly_scores = {}
        for score, date in zip(texture_scores_timeline, dates):
            year = date.year
            if year not in yearly_scores:
                yearly_scores[year] = []
            yearly_scores[year].append(score)
        
        # Анализ трендов
        years = sorted(yearly_scores.keys())
        mean_scores = [np.mean(yearly_scores[year]) for year in years]
        
        # Детекция скачков качества
        quality_jumps = []
        for i in range(1, len(mean_scores)):
            score_change = mean_scores[i] - mean_scores[i-1]
            if score_change > ANOMALY_DETECTION_ADVANCED_THRESHOLDS['MASK_QUALITY_JUMP_THRESHOLD']:
                quality_jumps.append({
                    'year': years[i],
                    'improvement': score_change,
                    'new_technology_suspected': True
                })
        
        # Корреляция с технологическими скачками
        correlated_jumps = []
        for jump in quality_jumps:
            if jump['year'] in TECHNOLOGY_BREAKTHROUGH_YEARS:
                correlated_jumps.append(jump)
        
        evolution_analysis = {
            'yearly_quality_scores': yearly_scores,
            'quality_trend': self._calculate_trend(years, mean_scores),
            'detected_quality_jumps': quality_jumps,
            'correlated_with_breakthrough_years': correlated_jumps,
            'evolution_rate': (mean_scores[-1] - mean_scores[0]) / len(mean_scores) if len(mean_scores) > 1 else 0
        }
        
        return evolution_analysis

    def perform_cross_source_verification(self, same_date_images: Dict) -> Dict:
        """Кросс-анализ фотографий одного дня из разных источников"""
        cross_verification = {}
        
        # Определяем метрики, которые будут использоваться для кросс-валидации
        key_metrics_for_cross_validation = [
            'forehead_height_ratio', 'nose_width_ratio', 'mouth_width_ratio', 
            'chin_width_ratio', 'facial_symmetry_index',
            'skull_width_ratio', 'temporal_bone_angle', 'zygomatic_arch_width',
            'orbital_depth', 'inter_pupillary_distance_ratio'
        ]

        for date, sources_data in same_date_images.items():
            if len(sources_data) < 2:
                continue
            
            source_embeddings = []
            source_names = []
            source_metrics_list = [] # Для сбора метрик из разных источников
            
            for source_name, data in sources_data.items():
                embedding = data.get('embedding')
                if embedding is not None:
                    source_embeddings.append(embedding)
                    source_names.append(source_name)
                
                # Собираем метрики, если они есть
                current_metrics = {k: data.get(k) for k in key_metrics_for_cross_validation if k in data}
                if current_metrics:
                    source_metrics_list.append(current_metrics)
            
            if len(source_embeddings) < 2:
                # Если не хватает эмбеддингов, но есть метрики, можем попробовать только метрики
                if len(source_metrics_list) < 2:
                    continue # Недостаточно данных для кросс-валидации
            
            # --- Анализ консистентности эмбеддингов ---
            embedding_consistency_score = 1.0
            embedding_max_distance = 0.0
            embedding_mean_distance = 0.0
            embedding_critical_anomaly = False

            if len(source_embeddings) >= 2:
                distance_matrix = cosine_distances(source_embeddings)
                embedding_max_distance = np.max(distance_matrix)
                embedding_mean_distance = np.mean(distance_matrix[distance_matrix > 0]) # Исключаем 0, если сравнивается с самим собой
                embedding_critical_anomaly = embedding_max_distance > ANOMALY_DETECTION_ADVANCED_THRESHOLDS['CROSS_SOURCE_CRITICAL_DISTANCE_THRESHOLD']
                embedding_consistency_score = 1.0 - embedding_mean_distance

            # --- Анализ консистентности метрик ---
            metrics_consistency_score, metrics_inconsistency_details = self._calculate_metrics_consistency(source_metrics_list, key_metrics_for_cross_validation)

            # --- Комбинированная оценка и объяснение ---
            overall_consistency_score = (embedding_consistency_score + metrics_consistency_score) / 2.0
            anomaly_detected = embedding_critical_anomaly or (metrics_consistency_score < 0.7) # Порог для метрик

            explanation = self._generate_cross_source_explanation(
                embedding_critical_anomaly, embedding_max_distance, metrics_inconsistency_details
            )
            
            cross_verification[date] = {
                'sources_analyzed': source_names,
                'embedding_max_distance': embedding_max_distance,
                'embedding_mean_distance': embedding_mean_distance,
                'embedding_consistency_score': embedding_consistency_score,
                'metrics_consistency_score': metrics_consistency_score,
                'metrics_inconsistency_details': metrics_inconsistency_details,
                'critical_anomaly_detected': anomaly_detected,
                'overall_consistency_score': overall_consistency_score,
                'explanation': explanation
            }
        
        return cross_verification

    def _calculate_metrics_consistency(self, source_metrics_list: List[Dict], key_metrics: List[str]) -> Tuple[float, Dict]:
        """
        Рассчитывает согласованность ключевых метрик лица между разными источниками.
        Возвращает общий балл согласованности и детали расхождений по метрикам.
        """
        if not source_metrics_list or len(source_metrics_list) < 2: 
            return 1.0, {} # Полная согласованность, если нет данных или только один источник

        inconsistency_details = {}
        consistency_scores_per_metric = []

        for metric_name in key_metrics:
            values_for_metric = [metrics.get(metric_name) for metrics in source_metrics_list if metrics.get(metric_name) is not None]
            
            if len(values_for_metric) < 2: # Недостаточно значений для этой метрики
                continue
            
            values_array = np.array(values_for_metric)
            
            # Используем коэффициент вариации (CV) как меру несогласованности.
            # Чем выше CV, тем ниже согласованность.
            mean_val = np.mean(values_array)
            std_val = np.std(values_array)

            if mean_val == 0:
                # Если среднее 0, и есть вариация, это может быть проблемой, или если все 0 - консистентно
                cv = 0.0 if std_val == 0 else float('inf')
            else:
                cv = std_val / mean_val
            
            # Преобразуем CV в балл согласованности (от 0 до 1), где 1 - идеальная согласованность
            # Порог 0.2 для CV можно считать началом расхождений
            consistency_score = max(0.0, 1.0 - (cv / 0.2)) # Масштабирование: если CV 0.2, score 0
            consistency_scores_per_metric.append(consistency_score)

            if consistency_score < 0.7: # Если согласованность ниже порога, добавляем в детали
                inconsistency_details[metric_name] = {
                    'values': values_array.tolist(),
                    'mean': float(f'{mean_val:.4f}'),
                    'std': float(f'{std_val:.4f}'),
                    'cv': float(f'{cv:.4f}'),
                    'consistency_score': float(f'{consistency_score:.4f}'),
                    'status': 'Несогласовано'
                }
            else:
                inconsistency_details[metric_name] = {
                    'mean': float(f'{mean_val:.4f}'),
                    'std': float(f'{std_val:.4f}'),
                    'cv': float(f'{cv:.4f}'),
                    'consistency_score': float(f'{consistency_score:.4f}'),
                    'status': 'Согласовано'
                }

        overall_metrics_consistency = np.mean(consistency_scores_per_metric) if consistency_scores_per_metric else 1.0

        return overall_metrics_consistency, inconsistency_details

    def _generate_cross_source_explanation(self, embedding_critical_anomaly: bool, embedding_max_distance: float,
                                           metrics_inconsistency_details: Dict) -> str:
        """
        Генерирует текстовое объяснение для результатов кросс-источниковой валидации.
        """
        explanations = []

        if embedding_critical_anomaly:
            explanations.append(f"Обнаружено критическое расхождение в 'отпечатках' лица (embeddings) между источниками (максимальное расстояние: {embedding_max_distance:.2f}). Это может указывать на попытку подмены или серьезную ошибку данных.")
        elif embedding_max_distance > 0.4: # Если есть значительное, но не критическое расхождение
            explanations.append(f"Наблюдается значительное расхождение в 'отпечатках' лица (embeddings) между источниками (максимальное расстояние: {embedding_max_distance:.2f}), что требует внимания.")
        else:
            explanations.append(f"'Отпечатки' лица (embeddings) между источниками согласованы (максимальное расстояние: {embedding_max_distance:.2f}).")
        
        inconsistent_metrics = [metric for metric, details in metrics_inconsistency_details.items() if details['status'] == 'Несогласовано']
        if inconsistent_metrics:
            explanations.append(f"Обнаружены несоответствия в следующих метриках лица между источниками: {', '.join(inconsistent_metrics)}. Это может свидетельствовать о различиях в условиях съемки или манипуляциях с изображением.")
        else:
            explanations.append("Ключевые метрики лица между источниками согласованы.")

        return " ".join(explanations)

    def validate_identity_consistency(self, identity_metrics_over_time: Dict) -> Dict:
        """Проверяет консистентность метрик идентичности во времени"""
        consistency_validation = {}
        
        for identity_id, metrics_timeline in identity_metrics_over_time.items():
            # Извлечение временных рядов для каждой метрики
            metric_series = {}
            dates_series = {}
            
            for timestamp_str, metrics in metrics_timeline.items():
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                for metric_name, value in metrics.items():
                    if metric_name not in metric_series:
                        metric_series[metric_name] = []
                        dates_series[metric_name] = []
                    metric_series[metric_name].append(value)
                    dates_series[metric_name].append(timestamp)
            
            # Анализ консистентности для каждой метрики
            metric_consistency = {}
            
            for metric_name, values in metric_series.items():
                if len(values) < 2:
                    metric_consistency[metric_name] = {
                        'coefficient_of_variation': 0.0,
                        'trend_slope': 0.0,
                        'outlier_count': 0,
                        'rapid_changes_detected': False,
                        'stability_score': 1.0,
                        'reason': 'Недостаточно данных для анализа'
                    }
                    continue
                
                # Коэффициент вариации (мера относительного разброса)
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / mean_val if mean_val != 0 else (0.0 if std_val == 0 else float('inf'))
                
                # Тренд анализ (насколько метрика увеличивается/уменьшается со временем)
                trend_slope = self._calculate_trend_slope(values)
                
                # Детекция выбросов (точек, сильно отличающихся от основной массы)
                z_scores = np.abs(stats.zscore(values))
                outliers = np.sum(z_scores > EMBEDDING_ANALYSIS_THRESHOLDS['EMBEDDING_ANOMALY_Z_SCORE_THRESHOLD'])
                
                # Детекция резких изменений между последовательными точками
                rapid_changes_detected = False
                if len(values) > 1:
                    diffs = np.abs(np.diff(values))
                    if len(diffs) > 0 and np.std(diffs) > 0:
                        if np.any(diffs > ANOMALY_DETECTION_THRESHOLDS['RAPID_CHANGE_STD_MULTIPLIER'] * np.std(diffs)):
                            rapid_changes_detected = True
                
                # Расчет общего балла стабильности для метрики
                stability_score = max(0.0, 1.0 -
                                    (cv * 0.5) -
                                    (abs(trend_slope) * 0.2) -
                                    (outliers * 0.1) -
                                    (0.2 if rapid_changes_detected else 0))
                
                reason = "Стабильно" if stability_score >= ANOMALY_DETECTION_ADVANCED_THRESHOLDS['METRIC_STABILITY_THRESHOLD'] else "Нестабильно"
                if cv > 0.5: reason += ", высокий CV"
                if abs(trend_slope) > 0.1: reason += ", сильный тренд"
                if outliers > 0: reason += f", {outliers} выбросов"
                if rapid_changes_detected: reason += ", резкие изменения"
                
                metric_consistency[metric_name] = {
                    'coefficient_of_variation': f'{cv:.4f}',
                    'trend_slope': f'{trend_slope:.4f}',
                    'outlier_count': int(outliers),
                    'rapid_changes_detected': rapid_changes_detected,
                    'stability_score': f'{stability_score:.4f}',
                    'reason': reason
                }
            
            # Общая оценка консистентности личности
            stability_scores = [float(mc['stability_score']) for mc in metric_consistency.values() if 'stability_score' in mc]
            overall_consistency = np.mean(stability_scores) if stability_scores else 0.0
            
            consistency_validation[identity_id] = {
                'metric_consistency': metric_consistency,
                'overall_consistency_score': f'{overall_consistency:.4f}',
                'identity_stable': overall_consistency > ANOMALY_DETECTION_ADVANCED_THRESHOLDS['IDENTITY_STABILITY_THRESHOLD'],
                'anomalous_metrics': [
                    {'name': name, 'score': float(mc['stability_score']), 'reason': mc['reason']} 
                    for name, mc in metric_consistency.items() 
                    if float(mc['stability_score']) < ANOMALY_DETECTION_ADVANCED_THRESHOLDS['METRIC_STABILITY_THRESHOLD']
                ]
            }
        
        return consistency_validation

    def _calculate_likelihood_same_person(self, geometry: float, embedding: float,
                                        texture: float, temporal: float) -> float:
        """Рассчитывает вероятность наблюдения evidence при условии, что это один и тот же человек."""
        
        combined_score = (
            AUTHENTICITY_WEIGHTS['geometry'] * geometry +
            AUTHENTICITY_WEIGHTS['embedding'] * embedding +
            AUTHENTICITY_WEIGHTS['texture'] * texture +
            AUTHENTICITY_WEIGHTS['temporal_consistency'] * temporal
        )
        
        # Сигмоидная функция для преобразования скора в вероятность
        likelihood = 1 / (1 + np.exp(-5 * (combined_score - 0.7)))
        
        return likelihood

    def _calculate_likelihood_different_person(self, geometry: float, embedding: float,
                                             texture: float, temporal: float) -> float:
        """Рассчитывает вероятность наблюдения evidence при условии, что это разные люди."""
        
        combined_score = (
            AUTHENTICITY_WEIGHTS['geometry'] * geometry +
            AUTHENTICITY_WEIGHTS['embedding'] * embedding +
            AUTHENTICITY_WEIGHTS['texture'] * texture +
            AUTHENTICITY_WEIGHTS['temporal_consistency'] * temporal
        )
        
        likelihood = 1 / (1 + np.exp(-5 * (0.7 - combined_score)))
        
        return likelihood

    def _get_likelihood_ratio(self, metric: str, value: float) -> float:
        """
        Рассчитывает отношение правдоподобия для данного значения метрики.
        L(value | authentic) / L(value | non-authentic)
        Параметры для каждой метрики (среднее для подлинных/неподлинных, чувствительность)
        должны быть откалиброваны на репрезентативных данных.
        """
        # Примерные параметры для различных метрик. Эти значения должны быть откалиброваны.
        # Более высокие значения 'authentic_mean' и 'sensitivity' указывают на то, что метрика
        # хорошо разделяет подлинные и неподлинные лица, и что высокие значения соответствуют подлинности.
        metric_bayesian_props = {
            'geometry_score': {'authentic_mean': 0.85, 'non_authentic_mean': 0.25, 'sensitivity': 7.0},
            'embedding_score': {'authentic_mean': 0.90, 'non_authentic_mean': 0.10, 'sensitivity': 9.0},
            'texture_score': {'authentic_mean': 0.75, 'non_authentic_mean': 0.30, 'sensitivity': 6.0},
            'temporal_consistency': {'authentic_mean': 0.80, 'non_authentic_mean': 0.20, 'sensitivity': 7.0}
            # Добавьте другие метрики, которые будут использоваться как свидетельства
        }

        props = metric_bayesian_props.get(metric)
        if not props:
            logging.warning(f"Метрика '{metric}' не определена для байесовского анализа. Возвращено нейтральное отношение (1.0).")
            return 1.0

        authentic_mean = props['authentic_mean']
        non_authentic_mean = props['non_authentic_mean']
        sensitivity = props['sensitivity']

        # Расчет P(value | authentic) с помощью сигмоидной функции (выше значение -> выше вероятность)
        likelihood_authentic = 1 / (1 + np.exp(-sensitivity * (value - authentic_mean)))
        
        # Расчет P(value | non-authentic) с помощью инвертированной сигмоиды (выше значение -> ниже вероятность)
        likelihood_non_authentic = 1 / (1 + np.exp(-sensitivity * (non_authentic_mean - value)))

        # Добавляем небольшое смещение, чтобы избежать деления на ноль
        epsilon = 1e-9
        if likelihood_non_authentic < epsilon:
            return float('inf') # Очень сильное свидетельство в пользу подлинности
        
        return likelihood_authentic / likelihood_non_authentic

    def calculate_bayesian_authenticity(self, evidence: Dict) -> float:
        """
        Байесовский расчет вероятности аутентичности лица на основе различных свидетельств.
        Args:
            evidence (Dict): Словарь свидетельств, где ключ - название метрики (например, 'geometry_score'),
                             а значение - ее числовая оценка (от 0 до 1).
        Returns:
            float: Апостериорная вероятность того, что лицо является подлинным (от 0 до 1).
        """
        prior_authentic = 0.5  # Априорная вероятность того, что лицо подлинное (по умолчанию 50/50)

        # Произведение отношений правдоподобия для всех свидетельств
        product_likelihood_ratios = 1.0
        for metric, value in evidence.items():
            lr = self._get_likelihood_ratio(metric, value)
            product_likelihood_ratios *= lr
        
        # Применение формулы Байеса: P(A|E) = (P(E|A)/P(E|~A)) * P(A) / [ (P(E|A)/P(E|~A)) * P(A) + P(~A) ]
        # где (P(E|A)/P(E|~A)) - это product_likelihood_ratios
        
        numerator = product_likelihood_ratios * prior_authentic
        denominator = numerator + (1 - prior_authentic)

        if denominator == 0:
            # Если знаменатель равен нулю, это означает бесконечную апостериорную вероятность в одном из направлений.
            # В данном случае, если product_likelihood_ratios очень велико и prior_authentic не ноль,
            # то знаменатель не будет нулем. Если product_likelihood_ratios равно нулю,
            # то и числитель, и знаменатель будут нулем или очень малыми. Обрабатываем этот случай.
            if numerator > 0: # Очень малый знаменатель из-за огромного LR
                return 1.0
            else:
                return 0.0 # Нет свидетельств в пользу подлинности

        posterior_authentic = numerator / denominator
        
        # Убедимся, что результат находится в пределах [0, 1]
        return np.clip(posterior_authentic, 0.0, 1.0)

    def _calculate_metric_changes(self, current: Dict, previous: Dict) -> Dict:
        """Вычисляет процентные изменения между двумя наборами метрик."""
        changes = {}
        
        for metric_name in current.keys():
            if metric_name in previous:
                change = current[metric_name] - previous[metric_name]
                relative_change = change / previous[metric_name] if previous[metric_name] != 0 else 0
                changes[metric_name] = {
                    'absolute_change': change,
                    'relative_change': relative_change,
                    'significant': abs(relative_change) > 0.1
                }
        
        return changes

    def _detect_swelling_patterns(self, changes: Dict) -> bool:
        """Детекция паттернов отечности (2-4 недели после операции)"""
        swelling_indicators = []
        
        # Метрики, изменяющиеся при отечности
        swelling_metrics = ['forehead_height_ratio', 'nose_width_ratio', 'mouth_width_ratio']
        
        for metric in swelling_metrics:
            if metric in changes:
                # Увеличение объема тканей
                if changes[metric]['relative_change'] > ANOMALY_DETECTION_ADVANCED_THRESHOLDS['SWELLING_METRIC_CHANGE_THRESHOLD']:
                    swelling_indicators.append(True)
                else:
                    swelling_indicators.append(False)
        
        return sum(swelling_indicators) >= ANOMALY_DETECTION_ADVANCED_THRESHOLDS['MIN_SWELLING_METRICS_DETECTED']

    def _detect_asymmetry_changes(self, changes: Dict) -> bool:
        """Детекция значительных изменений в асимметрии лица."""
        
        if 'golden_ratio_deviation' in changes:
            if changes['golden_ratio_deviation']['absolute_change'] > ANOMALY_DETECTION_ADVANCED_THRESHOLDS['ASYMMETRY_CHANGE_THRESHOLD']:
                return True
        
        return False

    def _analyze_healing_dynamics(self, changes: Dict, interval_days: int) -> Dict:
        """Анализ динамики заживления после изменений."""
        healing_indicators = {}
        
        # Метрики, связанные с отеком, которые должны уменьшаться
        metrics_to_monitor = ['forehead_height_ratio', 'nose_width_ratio', 'mouth_width_ratio']
        
        for metric in metrics_to_monitor:
            if metric in changes:
                # Если было резкое изменение (например, увеличение), а затем наблюдается уменьшение
                if changes[metric]['absolute_change'] < 0 and changes[metric]['significant']:
                    healing_indicators[metric] = True
                else:
                    healing_indicators[metric] = False
        
        # Дополнительная логика: если интервал достаточно большой для заживления
        if interval_days > ANOMALY_DETECTION_ADVANCED_THRESHOLDS['HEALING_DETECTION_INTERVAL_DAYS']:
            if any(healing_indicators.values()):
                return {'healing_detected': True, 'details': healing_indicators}
        
        return {'healing_detected': False, 'details': healing_indicators}

    def _calculate_surgery_likelihood(self, swelling_detected: bool, asymmetry_changes: bool, 
                                    healing_dynamics: Dict, interval_days: int) -> float:
        """Рассчитывает вероятность хирургического вмешательства на основе индикаторов."""
        likelihood = 0.0
        
        if swelling_detected:
            likelihood += ANOMALY_DETECTION_ADVANCED_THRESHOLDS['SURGERY_SWELLING_WEIGHT']
        
        if asymmetry_changes:
            likelihood += ANOMALY_DETECTION_ADVANCED_THRESHOLDS['SURGERY_ASYMMETRY_WEIGHT']
        
        if healing_dynamics.get('healing_detected', False):
            likelihood += ANOMALY_DETECTION_ADVANCED_THRESHOLDS['SURGERY_HEALING_WEIGHT']
        
        # Корректировка по временному интервалу
        if interval_days < ANOMALY_DETECTION_ADVANCED_THRESHOLDS['SURGERY_SHORT_INTERVAL_DAYS']:
            likelihood *= ANOMALY_DETECTION_ADVANCED_THRESHOLDS['SURGERY_SHORT_INTERVAL_MULTIPLIER']
        elif interval_days > ANOMALY_DETECTION_ADVANCED_THRESHOLDS['SURGERY_LONG_INTERVAL_DAYS']:
            likelihood *= ANOMALY_DETECTION_ADVANCED_THRESHOLDS['SURGERY_LONG_INTERVAL_MULTIPLIER']
        
        return min(1.0, likelihood)

    def detect_systematic_replacement_patterns(self, metrics_timeline: Dict) -> Dict:
        """Выявляет систематические паттерны замены (3-4 месячные циклы)"""
        dates = sorted(metrics_timeline.keys())
        intervals = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
        
        # Поиск 90-120 дневных циклов
        systematic_intervals = [i for i in intervals if 90 <= i <= 120]
        
        if len(systematic_intervals) >= 3:
            return {
                'systematic_pattern_detected': True,
                'pattern_type': '3-4_month_replacement_cycle',
                'confidence': len(systematic_intervals) / len(intervals)
            }
        return {'systematic_pattern_detected': False}

    def _calculate_trend(self, x_values: List, y_values: List) -> str:
        """Расчет тренда временного ряда"""
        if len(x_values) < 2:
            return "insufficient_data"
        
        slope, _, r_value, p_value, _ = stats.linregress(x_values, y_values)
        
        if p_value > 0.05:
            return "no_trend"
        elif slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"

    def _calculate_trend_slope(self, values: List[float]) -> float:
        """Расчет наклона тренда"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope, _, _, _, _ = stats.linregress(x, values)
        
        return slope
