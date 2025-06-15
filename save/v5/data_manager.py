# data_manager.py
# Управление данными и хронологический анализ

import os
import re
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

from core_config import (
    PUTIN_BIRTH_DATE, START_ANALYSIS_DATE, END_ANALYSIS_DATE,
    DATA_DIR, RESULTS_DIR
)

class DataManager:
    def __init__(self):
        self.chronological_index = {}
        self.image_quality_cache = {}
        # Загрузка исторических событий при инициализации
        self.historical_events = self.auto_parse_historical_events((START_ANALYSIS_DATE, END_ANALYSIS_DATE))

    def parse_date_from_filename(self, filename: str) -> Tuple[Optional[datetime], int]:
        """Парсит дату из dd_mm_yy.jpg или dd_mm_yy-N.jpg"""

        # Регулярное выражение для парсинга даты
        pattern = r'(\d{2})_(\d{2})_(\d{2})(?:-(\d+))?\.jpg'
        match = re.match(pattern, filename)

        if not match:
            return None, 0

        day, month, year, sequence = match.groups()
        sequence = int(sequence) if sequence else 1

        # Определение года (если 2-значный)
        year = int(year)
        if year <= 25:  # 2000-2025
            year += 2000
        else:  # 1900-1999
            year += 1900

        try:
            parsed_date = datetime(year, int(month), int(day))
            return parsed_date, sequence
        except ValueError:
            return None, 0

    def calculate_putin_age_on_date(self, photo_date: datetime) -> float:
        """Точный возраст Путина на дату фото с учетом високосных годов"""

        age_delta = photo_date - PUTIN_BIRTH_DATE
        age_years = age_delta.days / 365.25  # Учет високосных годов
        return age_years

    def create_master_chronological_index(self, image_paths: List[str]) -> Dict:
        """Создает мастер-индекс всех фотографий с хронологической сортировкой"""

        chronological_data = {}

        for image_path in image_paths:
            filename = os.path.basename(image_path)
            date, sequence = self.parse_date_from_filename(filename)

            if date and START_ANALYSIS_DATE <= date <= END_ANALYSIS_DATE:
                age_on_date = self.calculate_putin_age_on_date(date)

                date_key = date.strftime('%Y-%m-%d')
                if date_key not in chronological_data:
                    chronological_data[date_key] = {
                        'date': date,
                        'age_on_date': age_on_date,
                        'image_paths': []
                    }

                chronological_data[date_key]['image_paths'].append({
                    'path': image_path,
                    'sequence': sequence,
                    'filename': filename
                })

        # Сортировка по дате
        sorted_dates = sorted(chronological_data.keys())
        sorted_data = {date: chronological_data[date] for date in sorted_dates}

        self.chronological_index = sorted_data
        return sorted_data

    def validate_image_quality_for_analysis(self, image_path: str) -> Dict:
        """
        Проверяет пригодность изображения для анализа:
        resolution_check, blur_detection, lighting_quality, face_visibility.
        Возвращает quality_score[0-1] и список проблем.
        """

        if image_path in self.image_quality_cache:
            return self.image_quality_cache[image_path]

        try:
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                logging.warning(f"Не удалось прочитать файл изображения: {image_path}")
                return {'quality_score': 0.0, 'issues': ['File not readable']}

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            h, w = gray.shape

            issues = []
            
            # --- Resolution Check --- 
            # Предполагаемые минимальные требования к разрешению для качественного анализа
            MIN_RESOLUTION_WIDTH = 200
            MIN_RESOLUTION_HEIGHT = 200
            resolution_score = 1.0
            if w < MIN_RESOLUTION_WIDTH or h < MIN_RESOLUTION_HEIGHT:
                issues.append('Low resolution')
                resolution_score = 0.5 # Значительное снижение за низкое разрешение

            # --- Blur Check (Laplacian variance) ---
            # Чем выше variance, тем резче изображение
            BLUR_THRESHOLD = 100 # Порог, ниже которого изображение считается размытым
            blur_score_val = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_quality_score = 1.0
            if blur_score_val < BLUR_THRESHOLD:
                issues.append('Image too blurry')
                # Линейное масштабирование от 0 до 1, где 150 - это хороший показатель.
                blur_quality_score = max(0.0, blur_score_val / 150.0)

            # --- Noise Check (Standard Deviation of pixel intensities) ---
            # Чем выше std, тем больше шума
            MAX_NOISE_LEVEL = 30 # Порог шума, выше которого изображение считается зашумленным
            noise_level = np.std(gray)
            noise_quality_score = 1.0
            if noise_level > MAX_NOISE_LEVEL:
                issues.append('High noise level')
                # Чем больше шум, тем ниже балл. Нормализация так, чтобы 10 было хорошо, 50 - плохо.
                noise_quality_score = max(0.0, 1.0 - (noise_level - 10) / 40.0)

            # --- Lighting Check (Histogram Extremes) ---
            # Проверяем, не слишком ли много пикселей находятся в крайних (очень темных или очень светлых) областях гистограммы
            HIST_EXTREME_RATIO_THRESHOLD = 0.1 # 10% пикселей в крайних 2% гистограммы
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            total_pixels = np.sum(hist)
            lighting_quality_score = 1.0
            if total_pixels > 0 and ((hist[0] / total_pixels > HIST_EXTREME_RATIO_THRESHOLD) or \
               (hist[255] / total_pixels > HIST_EXTREME_RATIO_THRESHOLD)):
                issues.append('Poor lighting (too dark or too bright extremes)')
                lighting_quality_score = 0.5 # Значительное снижение за плохое освещение

            # --- Face Visibility Check (basic) ---
            # Это базовый уровень. Более точная проверка требует детектора лиц.
            # Если лицо слишком маленькое или не найдено, это также влияет на качество.
            # Пока что будем считать, что если есть достаточный размер изображения, лицо *может* быть видимо.
            # Реальная проверка лица будет в Face3DAnalyzer.
            face_visibility_score = 1.0 # По умолчанию считаем видимым, если прошло другие проверки
            # TODO: Интегрировать FaceBoxes для проверки наличия лица и его размера/качества

            # --- Combine scores with weights ---
            # Эти веса можно вынести в core_config.py для настройки
            weights = {
                'resolution': 0.25,
                'blur': 0.25,
                'noise': 0.25,
                'lighting': 0.25,
                'face_visibility': 0.0 # Вес пока 0, т.к. полная проверка не реализована здесь
            }
            # Нормализация весов, если они не суммируются в 1.0 (для гибкости)
            total_weight = sum(weights.values())
            if total_weight == 0: total_weight = 1.0
            
            combined_quality_score = (
                resolution_score * weights['resolution'] +
                blur_quality_score * weights['blur'] +
                noise_quality_score * weights['noise'] +
                lighting_quality_score * weights['lighting'] +
                face_visibility_score * weights['face_visibility']
            ) / total_weight

            quality_result = {
                'quality_score': np.clip(combined_quality_score, 0.0, 1.0),
                'issues': issues,
                'resolution': (w, h),
                'blur_score': blur_score_val,
                'noise_level': noise_level,
                'lighting_hist_extreme_ratio': (hist[0] / total_pixels, hist[255] / total_pixels)
            }

            self.image_quality_cache[image_path] = quality_result
            return quality_result

        except Exception as e:
            logging.error(f"Ошибка при анализе качества изображения {image_path}: {e}")
            return {'quality_score': 0.0, 'issues': [f'Processing error: {str(e)}']}

    def auto_parse_historical_events(self, date_range: Tuple[datetime, datetime]) -> Dict:
        """Автоматически парсит исторические события из внешних источников.
        
        ВНИМАНИЕ: Для полноценной работы потребуется интеграция с реальными внешними API (например, GDELT, Wikipedia API).
        Текущая реализация возвращает пустые данные.
        """
        logger.info(f"Автоматический парсинг исторических событий для диапазона {date_range}")
        
        # Здесь должна быть логика вызова реальных внешних API, например:
        # import requests
        # response = requests.get(f"https://api.example.com/events?start={date_range[0]}&end={date_range[1]}")
        # raw_events = response.json()
        
        # Временно возвращаем пустой список, пока не будет реализована реальная интеграция с API
        mock_events = [] 
        
        # Преобразование списка событий в словарь для удобства доступа по дате
        events_by_date = {}
        for event in mock_events:
            date_str = event['date'].strftime('%Y-%m-%d')
            if date_str not in events_by_date:
                events_by_date[date_str] = []
            events_by_date[date_str].append(event)
        
        return events_by_date

    def correlate_with_historical_events(self, dates_list: List[datetime]) -> Dict:
        """
        Сопоставляет даты анализа с известными историческими событиями, включая госпитализации,
        отпуска и медицинские процедуры, для исключения альтернативных объяснений аномалий.
        Ищет события в пределах ±30 дней от заданной даты.
        Возвращает словарь корреляций, где ключ - дата, значение - список ближайших событий.
        """

        correlations = {}

        for date in dates_list:
            date_str = date.strftime('%Y-%m-%d')
            nearby_events = []

            # Ищем события в пределах ±30 дней
            for event_date_str, event_info in self.historical_events.items():
                event_date = datetime.strptime(event_date_str, '%Y-%m-%d')
                days_diff = abs((date - event_date).days)

                if days_diff <= 30:
                    is_medical_event = False
                    # Простая проверка на медицинские ключевые слова
                    event_description = event_info.get('description', '').lower()
                    if any(keyword in event_description for keyword in ['госпиталь', 'операция', 'медицинск', 'болезнь', 'реабилитация']):
                        is_medical_event = True

                    nearby_events.append({
                        'event_date': event_date,
                        'days_diff': days_diff,
                        'event_info': event_info,
                        'is_medical_event': is_medical_event
                    })

            if nearby_events:
                correlations[date_str] = sorted(nearby_events, key=lambda x: x['days_diff'])

        return correlations

    def detect_temporal_gaps_and_patterns(self) -> Dict:
        """Выявляет временные разрывы и паттерны в появлениях"""

        if not self.chronological_index:
            return {}

        dates = [datetime.strptime(date_str, '%Y-%m-%d') 
                for date_str in self.chronological_index.keys()]
        dates.sort()

        gaps = []
        patterns = {}

        # Анализ разрывов
        for i in range(1, len(dates)):
            gap_days = (dates[i] - dates[i-1]).days
            if gap_days > 30:  # Разрыв больше месяца
                gaps.append({
                    'start_date': dates[i-1],
                    'end_date': dates[i],
                    'gap_days': gap_days
                })

        # Анализ периодичности
        intervals = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]

        # Поиск системных интервалов (3-4 месяца = 90-120 дней)
        # Можно использовать гистограмму интервалов или поиск моды
        systematic_patterns = []

        if intervals:
            # Гистограмма интервалов для поиска частых значений
            bins = np.arange(0, max(intervals) + 30, 30) # Разделение на интервалы по 30 дней
            hist, bin_edges = np.histogram(intervals, bins=bins)
            
            # Находим пики в гистограмме
            for i in range(len(hist)):
                if hist[i] > 2: # Если интервал встречается более 2 раз, потенциальный паттерн
                    lower_bound = bin_edges[i]
                    upper_bound = bin_edges[i+1]
                    count = hist[i]

                    # Проверка на соответствие 3-4 месяцам (90-120 дней)
                    if (lower_bound >= 90 and upper_bound <= 120) or \
                       (lower_bound <= 120 and upper_bound >= 90 and lower_bound < upper_bound): # более гибкая проверка
                        systematic_patterns.append({
                            'interval_range_days': f'{int(lower_bound)}-{int(upper_bound)}',
                            'occurrences': int(count),
                            'type': '3-4 month cycle (suspected replacement)'
                        })
                    elif count >= 5: # Общий частый паттерн
                         systematic_patterns.append({
                            'interval_range_days': f'{int(lower_bound)}-{int(upper_bound)}',
                            'occurrences': int(count),
                            'type': 'Frequent recurring interval'
                        })

        patterns['total_images'] = len(dates)
        patterns['date_range'] = (dates[0], dates[-1])
        patterns['gaps'] = gaps
        patterns['systematic_intervals'] = systematic_patterns
        patterns['avg_interval'] = np.mean(intervals) if intervals else 0
        patterns['std_interval'] = np.std(intervals) if intervals else 0

        return {
            'gaps': gaps,
            'patterns': patterns
        }

    def create_data_quality_report(self, processed_images: List[str]) -> Dict:
        """
        Создает подробный отчет о качестве обработанных данных (изображений).
        Использует функцию validate_image_quality_for_analysis для оценки каждого изображения.
        Возвращает сводный отчет, включая среднее качество, распределение проблем и количество изображений.
        """
        report = {
            'total_images_processed': len(processed_images),
            'average_quality_score': 0.0,
            'quality_distribution': {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0, 'unusable': 0},
            'issues_breakdown': {},
            'images_by_quality': {'excellent': [], 'good': [], 'fair': [], 'poor': [], 'unusable': []}
        }

        if not processed_images:
            logging.warning("Нет изображений для создания отчета о качестве данных.")
            return report

        total_quality_score = 0.0
        for image_path in processed_images:
            quality_results = self.validate_image_quality_for_analysis(image_path)
            score = quality_results['quality_score']
            issues = quality_results['issues']
            
            total_quality_score += score

            # Классификация по качеству
            if score >= 0.9:
                report['quality_distribution']['excellent'] += 1
                report['images_by_quality']['excellent'].append(image_path)
            elif score >= 0.7:
                report['quality_distribution']['good'] += 1
                report['images_by_quality']['good'].append(image_path)
            elif score >= 0.5:
                report['quality_distribution']['fair'] += 1
                report['images_by_quality']['fair'].append(image_path)
            elif score >= 0.2:
                report['quality_distribution']['poor'] += 1
                report['images_by_quality']['poor'].append(image_path)
            else:
                report['quality_distribution']['unusable'] += 1
                report['images_by_quality']['unusable'].append(image_path)
            
            # Сбор статистики по типам проблем
            for issue in issues:
                report['issues_breakdown'][issue] = report['issues_breakdown'].get(issue, 0) + 1

        report['average_quality_score'] = total_quality_score / len(processed_images)
        report['average_quality_score'] = float(f"{report['average_quality_score']:.2f}") # Округление
        
        return report

    def export_chronological_data(self, output_path: str):
        """Экспортирует хронологические данные в CSV"""

        if not self.chronological_index:
            return

        rows = []
        for date_str, data in self.chronological_index.items():
            for img_data in data['image_paths']:
                rows.append({
                    'date': date_str,
                    'age_on_date': data['age_on_date'],
                    'image_path': img_data['path'],
                    'sequence': img_data['sequence'],
                    'filename': img_data['filename']
                })

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Хронологические данные экспортированы в {output_path}")
