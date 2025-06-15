import sys
import argparse
import cv2
import yaml
import json
import os
import numpy as np
import math
import sys
import os
from scipy.spatial.distance import pdist, squareform
from scipy.stats import skew, kurtosis

# Добавляем директорию текущего скрипта в sys.path для корректного импорта модулей
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

# Используем относительные импорты, предполагая, что demo.py является частью пакета 3DDFA2
# Это должно работать при запуске скрипта из родительской директории (nn)
from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
from utils.depth import depth
from utils.pncc import pncc
from utils.uv import uv_tex
from utils.pose import viz_pose
from utils.serialization import ser_to_ply, ser_to_obj
from utils.functions import draw_landmarks, get_suffix
from utils.tddfa_util import str2bool, _parse_param

# NEW IMPORTS FOR INSIGHTFACE
import insightface
from insightface.app import FaceAnalysis

# Импорт модулей анализа ракурсов
from frontal_metrics import FrontalAnalysisModule
from frontal_edge_metrics import FrontalEdgeAnalysisModule
from semi_profile_metrics import SemiProfileAnalysisModule
from profile_metrics import ProfileAnalysisModule

# Импорт функций из marquardt.py
from marquardt import MarquardtMask # Импортируем сам класс

# Функция для затемнения изображения
def darken_image(image, factor=0.3):
    """Затемняет изображение путем смешивания с черным цветом."""
    black_overlay = np.full(image.shape, (0, 0, 0), dtype=np.uint8)
    # Используем addWeighted для смешивания (factor - вес оригинального изображения)
    darkened_image = cv2.addWeighted(image, factor, black_overlay, 1 - factor, 0)
    return darkened_image

class NumpyEncoder(json.JSONEncoder):
    """Кастомный JSON энкодер для обработки NumPy типов"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            # Округляем числа с плавающей точкой до 4 знаков после запятой
            return round(float(obj), 4) # Включено округление
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def main(args):
    import os
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)
    
    # Инициализация анализаторов и сводных данных
    all_analysis_results = {}
    comparison_summary = {}
    # ideal_proportions_summary больше не нужен, т.к. идеальные пропорции получаются через get_ideal_proportions в модулях анализа

    # Init FaceBoxes and TDDFA
    if args.onnx:
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = args.mode == 'gpu'
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        # face_boxes = FaceBoxes() # Removed

    # Initialize insightface
    print("СТАТУС: Инициализация FaceAnalysis модели InsightFace...")
    # Use 'buffalo_l' model which includes detection and recognition.
    # Set providers to CPUExecutionProvider for CPU inference.
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640)) # Prepare model for inference
    print("СТАТУС: FaceAnalysis модель инициализирована.")

    # Инициализация модулей анализа
    frontal_module = FrontalAnalysisModule()
    frontal_edge_module = FrontalEdgeAnalysisModule()
    semi_profile_module = SemiProfileAnalysisModule()
    profile_module = ProfileAnalysisModule()

    # Создаем директорию для результатов
    os.makedirs('examples/results', exist_ok=True)

    # Обработка изображений
    if os.path.isdir(args.img_fp):
        image_files = []
        for f_name in os.listdir(args.img_fp):
            if f_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(args.img_fp, f_name))

        if not image_files:
            print(f'No image files found in {args.img_fp}')
            sys.exit(-1)

        print(f'Found {len(image_files)} image(s) in {args.img_fp}')
        print("=== АНАЛИЗ ДЛЯ ВЫЯВЛЕНИЯ РАЗЛИЧИЙ МЕЖДУ ФОТОГРАФИЯМИ ===")
        
        print("СТАТУС: Начало обработки папки")
        
        for current_img_fp in image_files:
            print(f'Processing {current_img_fp}...')
            print(f"СТАТУС: Обработка файла {current_img_fp}")

            img = cv2.imread(current_img_fp)
            if img is None:
                print(f'Could not load image {current_img_fp}, skipping.')
                print(f"СТАТУС: Пропуск файла {current_img_fp} - не удалось загрузить изображение")
                continue

            # Detect faces using InsightFace
            # boxes = face_boxes(img) # Replaced by insightface
            faces_insight = app.get(img) # Get faces with embeddings

            n = len(faces_insight)
            print(f"СТАТУС: Обнаружено {n} лиц с помощью InsightFace")
            if n == 0:
                print(f'No face detected in {current_img_fp} by InsightFace, skipping.')
                print(f"СТАТУС: Пропуск файла {current_img_fp} - лица не обнаружены InsightFace")
                continue

            # Prepare boxes for TDDFA from InsightFace detections
            boxes_for_tddfa = []
            embeddings_for_json = []
            for face in faces_insight:
                # InsightFace bbox format is [x1, y1, x2, y2]. TDDFA expects [x1, y1, x2, y2, score].
                # Add a dummy score (e.g., 1.0) if InsightFace doesn't provide one directly in this format
                x1, y1, x2, y2 = face.bbox.astype(int)
                boxes_for_tddfa.append([x1, y1, x2, y2, 1.0]) # Add dummy score
                embeddings_for_json.append(face.embedding.tolist()) # Store embedding

            # Pass the boxes detected by insightface to tddfa
            param_lst, roi_box_lst = tddfa(img, boxes_for_tddfa) # Use boxes from InsightFace
            base_name = os.path.splitext(os.path.basename(current_img_fp))[0]
            
            # Получаем 3D landmarks (68 точек)
            print("СТАТУС: Попытка получить 68 разреженных landmarks")
            landmarks_3d_lst = [l.T for l in tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)]
            
            faces_analysis = []
            for face_idx, (params, landmarks_3d) in enumerate(zip(param_lst, landmarks_3d_lst)):
                print(f"СТАТУС: Обработка лица {face_idx}")

                print(f"Face {face_idx}: landmarks shape = {landmarks_3d.shape}")

                # Используем все доступные landmarks для попытки анализа
                landmarks_to_analyze = landmarks_3d
                print(f"СТАТУС: Попытка анализа с {landmarks_to_analyze.shape[0]} landmarks")

                # Извлекаем углы Эйлера для классификации ракурса (если возможно)
                pitch, yaw, roll = None, None, None
                params_array = np.array(params)
                pose_type = "unknown"

                R_matrix_descaled, t_vec, s_scale = frontal_module.marquardt_mask.extract_pose_parameters(params_array)
                # Получаем alpha_shp и alpha_exp из параметров
                # _parse_param ожидает одномерный массив параметров, params_array уже такой
                R_parsed, offset_parsed, alpha_shp_parsed, alpha_exp_parsed = _parse_param(params_array)

                if len(params_array) >= 12:
                    print("СТАТУС: Извлечение параметров позы (вручную)")
                    # R_matrix = np.array(params_array[1:10]).reshape(3, 3) # Исходная строка
                    # TDDFA param format: [s, r11, r12, r13, r21, r22, r23, r31, r32, r33, tx, ty, ...]
                    # R is at indices 1 to 9 (inclusive)
                    # Use R_matrix_descaled, t_vec, s_scale from extract_pose_parameters
                    # offset_vector and scale are not directly used in MarquardtMask.extract_euler_angles, but can be extracted if needed
                    # offset_vector = params_array[10:12] # This is only 2D offset
                    # s_scale = params_array[0]

                    print(f"DEBUG: R_matrix (directory mode - de-scaled):\\n{R_matrix_descaled}")
                    print("СТАТУС: Извлечение углов Эйлера")
                    euler_angles = frontal_module.marquardt_mask.extract_euler_angles(R_matrix_descaled)
                    pitch, yaw, roll = euler_angles
                    print(f"DEBUG: Raw Euler angles (directory mode): pitch={pitch:.2f}°, yaw={yaw:.2f}°, roll={roll:.2f}°")
                    # Заменить вызов pose_classifier.classify_pose
                    pose_type = frontal_module.marquardt_mask.classify_angle(yaw) # Используем классификацию из marquardt.py
                    # pose_type = pose_classifier.classify_pose(yaw, pitch, roll) # Старый вызов
                    euler_angles_dict = {"pitch": float(pitch), "yaw": float(yaw), "roll": float(roll)}
                    print(f"Pose classification: {pose_type} (yaw={yaw:.1f}°, pitch={pitch:.1f}°, roll={roll:.1f}°)")
                else:
                    print("ВНИМАНИЕ: Недостаточно параметров 3DMM для извлечения углов позы.")
                    pose_type = "insufficient_3dmm_params"

                # Выполняем анализ в зависимости от ракурса
                print(f"СТАТУС: Начало анализа метрик для ракурса {pose_type}")
                analysis_results = {}

                # Получаем плотные вершины для текущего лица
                current_ver_dense = tddfa.recon_vers([params], [roi_box_lst[face_idx]], dense_flag=True)[0]
                # Транспонируем ver_dense, чтобы его форма была (N, 3)
                if current_ver_dense.shape[0] == 3 and current_ver_dense.shape[1] > 3:
                    current_ver_dense = current_ver_dense.T
                    print(f"СТАТУС: Транспонирован current_ver_dense. Новая форма: {current_ver_dense.shape}")

                current_depth_map_image = depth(img.copy(), [current_ver_dense], tddfa.tri, with_bg_flag=True) # Генерируем карту глубины
                current_pncc_map_image = pncc(img.copy(), [current_ver_dense], tddfa.tri, with_bg_flag=True) # Генерируем PNCC-карту

                if pose_type == 'frontal':
                    analysis_results = frontal_module.analyze(landmarks_to_analyze, yaw, roll, alpha_shp_parsed, alpha_exp_parsed, R_matrix_descaled, t_vec, tddfa.bfm.u_base, current_depth_map_image, tddfa.tri, current_ver_dense, current_pncc_map_image)
                elif pose_type == 'frontal_edge':
                    analysis_results = frontal_edge_module.analyze(landmarks_to_analyze, yaw, roll)
                elif pose_type == 'semi_profile':
                    analysis_results = semi_profile_module.analyze(landmarks_to_analyze, yaw, roll)
                elif pose_type == 'profile':
                    analysis_results = profile_module.analyze(landmarks_to_analyze, yaw, roll)
                else: # unknown, extreme_pose, insufficient_3dmm_params
                     print(f"СТАТУС: Ракурс {pose_type}, пропускаем детальный анализ метрик.")
                     # Для неизвестных/экстремальных ракурсов, анализ не выполняется модулем.
                     # В этом случае результат анализа - пустой словарь.
                     analysis_results = {} # Явно устанавливаем пустой словарь, если ракурс не поддерживается

                # Определяем, был ли анализ успешно завершен.
                # Считаем анализ завершенным, если словарь с результатами не пустой.
                analysis_completed = bool(analysis_results)

                if analysis_completed:
                    print(f"СТАТУС: Анализ для ракурса {pose_type} успешно завершен.")
                else:
                    print(f"СТАТУС: Анализ для ракурса {pose_type} не завершен успешно (результат: {analysis_results}).")

                # Убедимся, что в biometric_marquardt_analysis не попадает статус ошибки из-за недостатка landmarks
                # Если analysis_results пустой (модуль вернул {}), то так и оставляем.
                # Если analysis_results не пустой, используем его как есть.
                final_biometric_analysis_section = analysis_results if analysis_completed else {}

                # Get the corresponding InsightFace embedding
                insightface_embedding = []
                if face_idx < len(embeddings_for_json):
                    insightface_embedding = embeddings_for_json[face_idx]
                    print(f"СТАТУС: Эмбеддинг InsightFace для лица {face_idx} добавлен.")
                else:
                    print(f"ВНИМАНИЕ: Нет соответствующего эмбеддинга InsightFace для лица {face_idx}.")

                # Формируем запись для этого лица с результатами анализа
                face_analysis = {
                    "face_id": int(face_idx),
                    "pose_estimation": {
                        "type": pose_type,
                        "angles_degrees": euler_angles_dict
                    },
                    "biometric_analysis": {
                        "analysis_type": final_biometric_analysis_section.get('analysis_type', 'unknown'),
                        "overall_marquardt_similarity_score": float(final_biometric_analysis_section.get('overall_similarity_score', 0.0)),
                        "grouped_proportions": {}, # Новая структура для группировки пропорций
                        "additional_analysis_metrics": {} # Новая структура для дополнительных метрик
                    }
                }

                # Заполняем grouped_proportions и additional_analysis_metrics, если анализ был успешен
                if analysis_completed:
                    measured_props = final_biometric_analysis_section.get('measured_proportions', {})
                    ideal_props = final_biometric_analysis_section.get('ideal_proportions', {})
                    deviations = final_biometric_analysis_section.get('deviations_from_ideal', {})
                    additional_metrics_data = final_biometric_analysis_section.get('additional_metrics', {})

                    # Группировка основных пропорций
                    proportion_categories = {
                        "FaceDimensions": ["face_width_height_ratio", "face_golden_ratio", "face_golden_ratio_height_width"],
                        "EyeMetrics": ["eye_width_face_ratio", "interocular_distance_ratio", "right_eye_aspect_ratio", "left_eye_aspect_ratio"],
                        "NoseMetrics": ["nose_width_eye_distance", "nose_length_face_ratio", "nose_height_width_ratio", "columella_to_nose_length_ratio"],
                        "MouthMetrics": ["mouth_width_nose_ratio", "upper_lip_lower_ratio", "mouth_height_width_ratio"],
                        "JawChinMetrics": ["jaw_width_face_ratio", "chin_width_mouth_ratio", "chin_projection_ratio", "zygomatic_to_bigonial_ratio"],
                        "EyebrowMetrics": ["eyebrow_eye_distance_ratio", "eyebrow_length_eye_ratio"],
                        "SymmetryMetrics": ["bilateral_symmetry_tolerance", "vertical_symmetry_ratio", "palpebral_symmetry_x", "alar_width_ratio", "midface_symmetry_index"],
                        "PositionalMetrics": ["philtrum_length_ratio", "canthal_tilt_normalized", "ocular_to_nasal_angle_degrees", "nasolabial_angle_cos", "upper_third_width_ratio"],
                        "Distances": ["a_face_height", "b_forehead_to_eyes", "c_eyes_to_nose", "d_eyes_to_lips", "e_nose_width", "f_eye_span", "g_face_width", "i_nose_to_chin", "j_lips_to_chin", "k_mouth_width", "l_nose_to_lips", "ipd_interpupillary", "avg_eye_width"]
                    }

                    grouped_proportions_data = {
                        "Measured": {},
                        "Ideal": {},
                        "Deviation": {}
                    }

                    for category, metrics_list in proportion_categories.items():
                        grouped_proportions_data["Measured"][category] = {}
                        grouped_proportions_data["Ideal"][category] = {}
                        grouped_proportions_data["Deviation"][category] = {}
                        for metric_name in metrics_list:
                            measured_val = final_biometric_analysis_section.get('measured_proportions', {}).get(metric_name)
                            ideal_val = final_biometric_analysis_section.get('ideal_proportions', {}).get(metric_name)
                            deviation_val = final_biometric_analysis_section.get('deviations_from_ideal', {}).get(metric_name)

                            # Если метрика находится в additional_metrics, берем ее оттуда
                            if measured_val is None:
                                measured_val = additional_metrics_data.get('stable_distances', {}).get(metric_name)
                            if measured_val is None:
                                measured_val = additional_metrics_data.get('marquardt_ratios', {}).get(metric_name)
                            if measured_val is None:
                                measured_val = additional_metrics_data.get('new_additional_metrics', {}).get(metric_name)
                            if measured_val is None:
                                measured_val = additional_metrics_data.get('stable_biometric_metrics', {}).get(metric_name)

                            if measured_val is not None:
                                grouped_proportions_data["Measured"][category][metric_name] = float(measured_val)
                                # Если есть идеальное значение, используем его, иначе 0.0
                                if ideal_val is not None:
                                    grouped_proportions_data["Ideal"][category][metric_name] = float(ideal_val)
                                    
                                # Если есть отклонение (и идеальное значение, по которому его можно рассчитать), используем его
                                if deviation_val is not None and ideal_val is not None:
                                    grouped_proportions_data["Deviation"][category][metric_name] = float(deviation_val)

                    face_analysis["biometric_analysis"]["grouped_proportions"] = grouped_proportions_data

                    # Добавление новых метрик в отдельную категорию (если они не попали в grouped_proportions)
                    face_analysis["biometric_analysis"]["additional_analysis_metrics"] = {
                        "insightface_embedding": insightface_embedding,
                        "shape_and_expression_errors": additional_metrics_data.get("shape_and_expression_errors", {})
                    }

                print(f"СТАТУС: Добавлен анализ для лица {face_idx} в структуру JSON.")

                faces_analysis.append(face_analysis)

            # Генерация визуализаций если требуется
            print("СТАТУС: Начало генерации визуализаций (если опция позволяет)")
            if args.opt == 'all':
                print("СТАТУС: Опция 'all' активна, генерируем все визуализации.")
                ver_lst_dense = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
                
                generated_files = {}
                
                # Сохраняем оригинал
                original_wfp = f'examples/results/{base_name}_original.jpg'
                cv2.imwrite(original_wfp, img)
                generated_files["original_image"] = original_wfp
                
                # Генерируем различные визуализации
                visualizations = ['2d_sparse', '2d_dense', '3d', 'depth', 'pncc', 'obj']
                
                for viz_type in visualizations:
                    wfp = f'examples/results/{base_name}_{viz_type}.jpg'
                    
                    # Затемняем оригинальное изображение для фона (для всех визуализаций, кроме obj)
                    if viz_type != 'obj':
                        img_darkened = darken_image(img.copy(), factor=0.3)
                    
                    if viz_type == '2d_sparse':
                        # Рисуем sparse landmarks на затемненном фоне для каждого лица с 68+ точками
                        green_color = (0, 255, 0)
                        white_color = (255, 255, 255)
                        point_thickness_sparse = 5
                        line_thickness_sparse = 3
                        landmark_groups_indices = {
                            'jaw': list(range(0, 17)), 'right_eyebrow': list(range(17, 22)),
                            'left_eyebrow': list(range(22, 27)), 'nose_bridge': list(range(27, 31)),
                            'nose_lower': list(range(31, 36)), 'right_eye': list(range(36, 42)),
                            'left_eye': list(range(42, 48)), 'outer_lips': list(range(48, 60)),
                            'inner_lips': list(range(60, 68))
                        }

                        img_sparse_viz = img_darkened.copy()
                        faces_drawn_sparse = 0
                        if landmarks_3d_lst:
                            for face_landmarks in landmarks_3d_lst:
                                if face_landmarks.shape[0] >= 68:
                                    faces_drawn_sparse += 1
                                    # Рисуем линии на затемненном изображении
                                    for group_name, indices in landmark_groups_indices.items():
                                        # Убедимся, что индексы не выходят за пределы массива face_landmarks
                                        valid_indices = [idx for idx in indices if idx < face_landmarks.shape[0]]
                                        if len(valid_indices) > 1:
                                             points = face_landmarks[valid_indices][:, :2].astype(np.int32)
                                             is_closed = group_name in ['right_eye', 'left_eye', 'outer_lips', 'inner_lips'] and len(indices) == len(valid_indices)
                                             cv2.polylines(img_sparse_viz, [points], is_closed, green_color, line_thickness_sparse, cv2.LINE_AA)

                                    # Рисуем точки и подписи на затемненном изображении
                                    for i, (x, y) in enumerate(face_landmarks[:, :2]):
                                        if i < face_landmarks.shape[0]:
                                            cv2.circle(img_sparse_viz, (int(x), int(y)), point_thickness_sparse, white_color, -1)

                            if faces_drawn_sparse > 0:
                                cv2.imwrite(wfp, img_sparse_viz)
                                print(f'2D sparse landmarks visualization saved to {wfp}')
                                generated_files["2d_sparse_landmarks_image"] = wfp
                            else:
                                print(f"СТАТУС: Пропуск визуализации 2d_sparse для {base_name} - ни одно лицо не имеет достаточно landmarks (>=68).")
                        else:
                            print(f"СТАТУС: Пропуск визуализации 2d_sparse для {base_name} - landmarks_3d_lst пуст.")
                    elif viz_type == '2d_dense':
                        # Рисуем dense landmarks на затемненном фоне
                        red_color = (0, 0, 255)
                        point_thickness_dense = 4

                        if ver_lst_dense:
                            img_dense_viz = img_darkened.copy()
                            faces_drawn_dense = 0
                            for face_vertices in ver_lst_dense:
                                if face_vertices.shape[0] > 68:
                                    faces_drawn_dense += 1
                                    points_2d = face_vertices[:, :2].astype(np.int32)
                                    h, w = img_dense_viz.shape[:2]
                                    points_2d_filtered = points_2d[(points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) & (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)]

                                    for (x, y) in points_2d_filtered:
                                        cv2.circle(img_dense_viz, (x, y), point_thickness_dense, red_color, -1)

                        if faces_drawn_dense > 0:
                            cv2.imwrite(wfp, img_dense_viz)
                            print(f'2D dense landmarks visualization saved to {wfp}')
                            generated_files["2d_dense_landmarks_image"] = wfp
                        else:
                            print(f"СТАТУС: Пропуск визуализации 2d_dense для {base_name} - ни одно лицо не имеет достаточно плотных landmarks (>68).")
                    elif viz_type == '3d':
                        render(img_darkened, ver_lst_dense, tddfa.tri, alpha=1.0, show_flag=args.show_flag, wfp=wfp)
                        print(f'3D render saved to {wfp}')
                        generated_files["3d_render_image"] = wfp
                    elif viz_type == 'depth':
                        depth(img_darkened, ver_lst_dense, tddfa.tri, show_flag=args.show_flag, wfp=wfp, with_bg_flag=True)
                        print(f'Depth map saved to {wfp}')
                        generated_files["depth_map_image"] = wfp
                    elif viz_type == 'pncc':
                        pncc(img_darkened, ver_lst_dense, tddfa.tri, show_flag=args.show_flag, wfp=wfp, with_bg_flag=True)
                        print(f'PNCC map saved to {wfp}')
                        generated_files["pncc_map_image"] = wfp
                    elif viz_type == 'obj':
                        wfp_obj = f'examples/results/{base_name}_{viz_type}.obj'
                        ser_to_obj(img, ver_lst_dense, tddfa.tri, height=img.shape[0], wfp=wfp_obj)
                        print(f'OBJ model saved to {wfp_obj}')
                        generated_files["obj_model"] = wfp_obj
                    
                    if viz_type != 'obj':
                        generated_files[viz_type] = wfp

                # Итоговые данные анализа (для режима папки)
                final_analysis = {
                    "image_info": {
                        "path": str(current_img_fp),
                        "name": str(base_name),
                        "faces_detected_count": int(len(faces_analysis)),
                        "processing_timestamp": str(np.datetime64('now')),
                        "analyzer_version": "3DDFA_V2_Complete_Biometric_Analysis"
                    },
                    "faces_data": faces_analysis,
                    "output_files": generated_files
                }

                # Сохраняем JSON с правильным энкодером
                analysis_wfp = f'examples/results/{base_name}_complete_biometric_analysis.json'
                with open(analysis_wfp, 'w', encoding='utf-8') as f:
                    json.dump(final_analysis, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)

                print(f'Updated biometric analysis saved to {analysis_wfp}')
                all_analysis_results[base_name] = final_analysis

            # Сводный анализ (обновляем структуру сводного файла)
            summary_wfp = 'examples/results/complete_biometric_summary.json'
            # simplified_all_analysis_results = {}
            # for img_name, img_data in all_analysis_results.items():
            #     simplified_all_analysis_results[img_name] = {
            #         "image_info": img_data["image_info"],
            #         "faces_data_summary": [
            #             {
            #                 "face_id": face['face_id'],
            #                 "pose_estimation": face['pose_estimation'],
            #                 "biometric_analysis": face['biometric_analysis']
            #             } for face in img_data["faces_data"]
            #         ]
            #     }

            final_summary_with_comparisons = {
                # "analysis_summary_by_image": simplified_all_analysis_results, # Удалено
                "comparison_summary": comparison_summary,
                "all_analysis_results": all_analysis_results # Включаем все результаты напрямую
            }

            with open(summary_wfp, 'w', encoding='utf-8') as f:
                json.dump(final_summary_with_comparisons, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)

            print(f'Updated biometric summary saved to {summary_wfp}')
            print("СТАТУС: Завершено сохранение сводного анализа с новой структурой.")
            print("=== АНАЛИЗ ЗАВЕРШЕН ===")

    elif os.path.isfile(args.img_fp):
        print("СТАТУС: Режим обработки одиночного файла")
        print(f'Processing single file {args.img_fp}...')
        print(f"СТАТУС: Обработка файла {args.img_fp}")
        
        img = cv2.imread(args.img_fp)
        if img is None:
            print(f'Could not load image {args.img_fp}')
            print(f"СТАТУС: Пропуск файла {args.img_fp} - не удалось загрузить изображение")
            sys.exit(-1)

        # Detect faces using InsightFace for single file
        faces_insight = app.get(img)

        n = len(faces_insight)
        print(f"СТАТУС: Обнаружено {n} лиц с помощью InsightFace")
        if n == 0:
            print(f'No face detected in {args.img_fp} by InsightFace, exit')
            print(f"СТАТУС: Пропуск файла {args.img_fp} - лица не обнаружены InsightFace")
            sys.exit(-1)
        print(f'Detect {n} faces')

        # Prepare boxes for TDDFA from InsightFace detections
        boxes_for_tddfa = []
        embeddings_for_json = []
        for face in faces_insight:
            x1, y1, x2, y2 = face.bbox.astype(int)
            boxes_for_tddfa.append([x1, y1, x2, y2, 1.0])
            embeddings_for_json.append(face.embedding.tolist())

        param_lst, roi_box_lst = tddfa(img, boxes_for_tddfa)

        # === Анализ и визуализация для одного файла ===
        base_name = os.path.splitext(os.path.basename(args.img_fp))[0]
        
        # Get 3D landmarks (68 points) for analysis and drawing
        print("СТАТУС: Попытка получить 68 разреженных landmarks")
        landmarks_3d_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)

        # Проверяем, что получили корректные landmarks для анализа
        face_landmarks_68 = None
        if landmarks_3d_lst and len(landmarks_3d_lst) > 0 and landmarks_3d_lst[0].shape[0] >= 68:
             face_landmarks_68 = landmarks_3d_lst[0][:68]
             print("СТАТУС: Успешно получены 68 landmarks для анализа.")
        elif landmarks_3d_lst and len(landmarks_3d_lst) > 0:
             print(f"ВНИМАНИЕ: Получено только {landmarks_3d_lst[0].shape[0]} разреженных landmarks вместо 68.")
             print("СТАТУС: Анализ метрик будет пропущен.")
        
        # === Выполнение анализа метрик (если получены 68 landmarks) ===
        analysis_metrics_result = None
        biometric_analysis_summary_for_json_single = {}

        insightface_embedding_single = []
        if len(embeddings_for_json) > 0:
            insightface_embedding_single = embeddings_for_json[0]
            print(f"СТАТУС: Эмбеддинг InsightFace для одиночного файла добавлен.")
        else:
            print(f"ВНИМАНИЕ: Нет соответствующего эмбеддинга InsightFace для одиночного файла.")

        if face_landmarks_68 is not None:
             print("СТАТУС: Извлечение параметров позы для анализа")
             params_single = param_lst[0] if param_lst else None
             if params_single:
                  print("СТАТУС: Извлечение параметров позы для анализа")
                  R_matrix_descaled, t_vec, s_scale = frontal_module.marquardt_mask.extract_pose_parameters(params_single)
                  
                  # Получаем alpha_shp и alpha_exp для одиночного файла
                  R_parsed_single, offset_parsed_single, alpha_shp_parsed_single, alpha_exp_parsed_single = _parse_param(params_single)

                  print("СТАТУС: Извлечение углов Эйлера для анализа")
                  euler_angles = frontal_module.marquardt_mask.extract_euler_angles(R_matrix_descaled)
                  pitch, yaw, roll = euler_angles
                 
                  print("СТАТУС: Классификация ракурса для анализа")
                  pose_type = frontal_module.marquardt_mask.classify_angle(yaw) # Использование корректной функции
                  print(f"Pose classification for analysis: {pose_type} (yaw={yaw:.1f}°, pitch={pitch:.1f}°, roll={roll:.1f}°)")
                 
                  print(f"СТАТУС: Начало комплексного анализа для ракурса {pose_type} (для одиночного файла)")
                  analysis_metrics = {}

                  # Получаем плотные вершины для текущего лица для карты глубины
                  current_ver_dense_single = tddfa.recon_vers([params_single], [roi_box_lst[0]], dense_flag=True)[0]
                  # Транспонируем ver_dense_single, чтобы его форма была (N, 3)
                  if current_ver_dense_single.shape[0] == 3 and current_ver_dense_single.shape[1] > 3:
                      current_ver_dense_single = current_ver_dense_single.T
                      print(f"СТАТУС: Транспонирован current_ver_dense_single. Новая форма: {current_ver_dense_single.shape}")

                  current_depth_map_image_single = depth(img.copy(), [current_ver_dense_single], tddfa.tri, with_bg_flag=True)
                  current_pncc_map_image_single = pncc(img.copy(), [current_ver_dense_single], tddfa.tri, with_bg_flag=True)

                  if pose_type == 'frontal':
                       analysis_metrics = frontal_module.analyze(face_landmarks_68, yaw, roll, alpha_shp_parsed_single, alpha_exp_parsed_single, R_matrix_descaled, t_vec, tddfa.bfm.u_base, current_depth_map_image_single, tddfa.tri, current_ver_dense_single, current_pncc_map_image_single)
                       print(f"СТАТУС: Завершен фронтальный анализ для одиночного файла. Получено метрик: {len(analysis_metrics)}")
                  elif pose_type == 'frontal_edge':
                       analysis_metrics = frontal_edge_module.analyze(face_landmarks_68, yaw, roll)
                       print(f"СТАТУС: Завершен frontal_edge анализ для одиночного файла. Получено метрик: {len(analysis_metrics)}")
                  elif pose_type == 'semi_profile':
                       analysis_metrics = semi_profile_module.analyze(face_landmarks_68, yaw, roll)
                       print(f"СТАТУС: Завершен semi_profile анализ для одиночного файла. Получено метрик: {len(analysis_metrics)}")
                  elif pose_type == 'profile':
                       analysis_metrics = profile_module.analyze(face_landmarks_68, yaw, roll)
                       print(f"СТАТУС: Завершен profile анализ для одиночного файла. Получено метрик: {len(analysis_metrics)}")
                  else:
                       print("СТАТУС: Экстремальный ракурс для одиночного файла, пропускаем детальный анализ метрик")
                       analysis_metrics = {
                            'analysis_type': pose_type,
                            'status': 'Skipped due to extreme pose'
                       }
                 
                  biometric_marquardt_analysis_output = analysis_metrics if analysis_metrics and analysis_metrics.get('status') != 'Skipped due to extreme pose' else {}

                  if biometric_marquardt_analysis_output:
                      # Новая структура для группировки пропорций
                      grouped_proportions_data_single = {
                          "Measured": {},
                          "Ideal": {},
                          "Deviation": {}
                      }

                      # Распределение метрик по категориям
                      proportion_categories = {
                          "FaceDimensions": ["face_width_height_ratio", "face_golden_ratio", "face_golden_ratio_height_width"],
                          "EyeMetrics": ["eye_width_face_ratio", "interocular_distance_ratio", "right_eye_aspect_ratio", "left_eye_aspect_ratio"],
                          "NoseMetrics": ["nose_width_eye_distance", "nose_length_face_ratio", "nose_height_width_ratio", "columella_to_nose_length_ratio"],
                          "MouthMetrics": ["mouth_width_nose_ratio", "upper_lip_lower_ratio", "mouth_height_width_ratio"],
                          "JawChinMetrics": ["jaw_width_face_ratio", "chin_width_mouth_ratio", "chin_projection_ratio", "zygomatic_to_bigonial_ratio"],
                          "EyebrowMetrics": ["eyebrow_eye_distance_ratio", "eyebrow_length_eye_ratio"],
                          "SymmetryMetrics": ["bilateral_symmetry_tolerance", "vertical_symmetry_ratio", "palpebral_symmetry_x", "alar_width_ratio", "midface_symmetry_index"],
                          "PositionalMetrics": ["philtrum_length_ratio", "canthal_tilt_normalized", "ocular_to_nasal_angle_degrees", "nasolabial_angle_cos", "upper_third_width_ratio"],
                          "Distances": ["a_face_height", "b_forehead_to_eyes", "c_eyes_to_nose", "d_eyes_to_lips", "e_nose_width", "f_eye_span", "g_face_width", "i_nose_to_chin", "j_lips_to_chin", "k_mouth_width", "l_nose_to_lips", "ipd_interpupillary", "avg_eye_width"]
                      }

                      measured_props_single = biometric_marquardt_analysis_output.get('measured_proportions', {})
                      ideal_props_single = biometric_marquardt_analysis_output.get('ideal_proportions', {})
                      deviations_single = biometric_marquardt_analysis_output.get('deviations_from_ideal', {})
                      additional_metrics_data_single_raw = biometric_marquardt_analysis_output.get('additional_metrics', {})

                      for category, metrics_list in proportion_categories.items():
                          grouped_proportions_data_single["Measured"][category] = {}
                          grouped_proportions_data_single["Ideal"][category] = {}
                          grouped_proportions_data_single["Deviation"][category] = {}
                          for metric_name in metrics_list:
                              measured_val_single = biometric_marquardt_analysis_output.get('measured_proportions', {}).get(metric_name)
                              ideal_val_single = biometric_marquardt_analysis_output.get('ideal_proportions', {}).get(metric_name)
                              deviation_val_single = biometric_marquardt_analysis_output.get('deviations_from_ideal', {}).get(metric_name)

                              # Если метрика находится в additional_metrics, берем ее оттуда
                              if measured_val_single is None:
                                  measured_val_single = additional_metrics_data_single_raw.get('stable_distances', {}).get(metric_name)
                              if measured_val_single is None:
                                  measured_val_single = additional_metrics_data_single_raw.get('marquardt_ratios', {}).get(metric_name)
                              if measured_val_single is None:
                                  measured_val_single = additional_metrics_data_single_raw.get('new_additional_metrics', {}).get(metric_name)
                              if measured_val_single is None:
                                  measured_val_single = additional_metrics_data_single_raw.get('stable_biometric_metrics', {}).get(metric_name)

                              if measured_val_single is not None:
                                  grouped_proportions_data_single["Measured"][category][metric_name] = float(measured_val_single)
                                  # Если есть идеальное значение, используем его, иначе 0.0
                                  if ideal_val_single is not None:
                                      grouped_proportions_data_single["Ideal"][category][metric_name] = float(ideal_val_single)
                                  
                                  # Если есть отклонение (и идеальное значение, по которому его можно рассчитать), используем его
                                  if deviation_val_single is not None and ideal_val_single is not None:
                                      grouped_proportions_data_single["Deviation"][category][metric_name] = float(deviation_val_single)

                      biometric_analysis_summary_for_json_single = {
                          "analysis_type": biometric_marquardt_analysis_output.get('analysis_type', 'unknown'),
                          "overall_marquardt_similarity_score": float(biometric_marquardt_analysis_output.get('overall_similarity_score', 0.0)),
                          "grouped_proportions": grouped_proportions_data_single,
                          "additional_analysis_metrics": {
                              "insightface_embedding": insightface_embedding_single,
                              "shape_and_expression_errors": additional_metrics_data_single_raw.get("shape_and_expression_errors", {})
                          }
                      }
                  else: # Если анализ не был успешно завершен (например, из-за экстремального ракурса)
                      biometric_analysis_summary_for_json_single = {
                          "analysis_type": pose_type, # Используем тип ракурса из классификации
                          "overall_marquardt_similarity_score": 0.0, # Очки сходства 0, если анализ не завершен
                          "grouped_proportions": {}, # Пустые пропорции
                          "additional_analysis_metrics": {
                              "insightface_embedding": insightface_embedding_single
                          } # Добавляем эмбеддинг, даже если анализ не выполнен
                      }
                 
                  analysis_metrics_result = {
                       "face_id": 0,
                       "pose_estimation": {
                            "type": pose_type,
                            "angles_degrees": {
                                 "pitch": float(pitch),
                                 "yaw": float(yaw),
                                 "roll": float(roll)
                            }
                       },
                       "biometric_analysis": biometric_analysis_summary_for_json_single
                  }
                  print("СТАТУС: Результаты анализа метрик для одиночного файла сформированы.")
             else:
                  print("ВНИМАНИЕ: Не удалось получить параметры 3DMM для анализа метрик.")
                  print("СТАТУС: Анализ метрик для одиночного файла пропущен.")

        # Сохранение результатов анализа в JSON для одиночного файла
        analysis_data_single = {
             "image_info": {
                  "path": str(args.img_fp),
                  "name": str(base_name),
                  "faces_detected_count": n,
                  "processing_timestamp": str(np.datetime64('now')),
                  "analyzer_version": "3DDFA_V2_Complete_Biometric_Analysis"
             },
             "faces_data": [analysis_metrics_result] if analysis_metrics_result is not None else [],
             "output_files": {},
        }
        
        analysis_wfp_single = f'examples/results/{base_name}_complete_biometric_analysis.json'
        print(f"СТАТУС: Сохранение результатов анализа в {analysis_wfp_single}")
        with open(analysis_wfp_single, 'w', encoding='utf-8') as f:
             json.dump(analysis_data_single, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
        print(f'Complete biometric analysis saved to {analysis_wfp_single}')

        # === Генерация визуализаций для одиночного файла ===
        print("СТАТУС: Начало генерации визуализаций для одиночного файла (если опция позволяет)")
        if args.opt == 'original':
             wfp = f'examples/results/{base_name}_original.jpg'
             cv2.imwrite(wfp, img)
             print(f'Original image saved to {wfp}')
             analysis_data_single['output_files']['original_image'] = wfp
             print("СТАТУС: Оригинальное изображение сохранено.")
        elif args.opt in ('2d_sparse', '2d_dense', '3d', 'depth', 'pncc', 'ply', 'obj', 'uv_tex', 'pose'):
             img_darkened = darken_image(img.copy(), factor=0.3)

             ver_lst_dense = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)

             new_suffix = f'.{args.opt}' if args.opt in ('ply', 'obj') else '.jpg'
             wfp = f'examples/results/{base_name}_{args.opt}' + new_suffix
             
             if args.opt == '2d_sparse':
                 for face_landmarks in landmarks_3d_lst:
                     landmark_groups_indices = {
                         'jaw': list(range(0, 17)), 'right_eyebrow': list(range(17, 22)),
                         'left_eyebrow': list(range(22, 27)), 'nose_bridge': list(range(27, 31)),
                         'nose_lower': list(range(31, 36)), 'right_eye': list(range(36, 42)),
                         'left_eye': list(range(42, 48)), 'outer_lips': list(range(48, 60)),
                         'inner_lips': list(range(60, 68))
                     }
                     green_color = (0, 255, 0)
                     white_color = (255, 255, 255)
                     point_thickness_sparse = 5
                     line_thickness_sparse = 3

                     for group_name, indices in landmark_groups_indices.items():
                         points = face_landmarks[indices][:, :2].astype(np.int32)
                         if len(points) > 1:
                             is_closed = group_name in ['right_eye', 'left_eye', 'outer_lips', 'inner_lips']
                             cv2.polylines(img_darkened, [points], is_closed, green_color, line_thickness_sparse, cv2.LINE_AA)

                     for i, (x, y) in enumerate(face_landmarks[:, :2]):
                         cv2.circle(img_darkened, (int(x), int(y)), point_thickness_sparse, white_color, -1)

                 cv2.imwrite(wfp, img_darkened)
                 print(f'2D sparse landmarks visualization saved to {wfp}')
                 analysis_data_single['output_files']['2d_sparse_landmarks_image'] = wfp

             elif args.opt == '2d_dense':
                 red_color = (0, 0, 255)
                 point_thickness_dense = 4

                 for face_vertices in ver_lst_dense:
                     points_2d = face_vertices[:, :2].astype(np.int32)
                     for (x, y) in points_2d:
                         cv2.circle(img_darkened, (x, y), point_thickness_dense, red_color, -1)

                 cv2.imwrite(wfp, img_darkened)
                 print(f'2D dense landmarks visualization saved to {wfp}')
                 analysis_data_single['output_files']['2d_dense_landmarks_image'] = wfp

             elif args.opt == '3d':
                 render(img_darkened, ver_lst_dense, tddfa.tri, alpha=1.0, show_flag=args.show_flag, wfp=wfp)
                 print(f'3D render saved to {wfp}')
                 analysis_data_single['output_files']['3d_render_image'] = wfp

             elif args.opt == 'depth':
                 depth(img_darkened, ver_lst_dense, tddfa.tri, show_flag=args.show_flag, wfp=wfp, with_bg_flag=True)
                 print(f'Depth map saved to {wfp}')
                 analysis_data_single['output_files']['depth_map_image'] = wfp

             elif args.opt == 'pncc':
                 pncc(img_darkened, ver_lst_dense, tddfa.tri, show_flag=args.show_flag, wfp=wfp, with_bg_flag=True)
                 print(f'PNCC map saved to {wfp}')
                 analysis_data_single['output_files']['pncc_map_image'] = wfp

             elif args.opt == 'uv_tex':
                 uv_tex(img_darkened, ver_lst_dense, tddfa.tri, show_flag=args.show_flag, wfp=wfp)
                 print(f'UV texture map saved to {wfp}')
                 analysis_data_single['output_files']['uv_texture_image'] = wfp

             elif args.opt == 'pose':
                 viz_pose(img_darkened, param_lst, ver_lst_dense, show_flag=args.show_flag, wfp=wfp)
                 print(f'Pose visualization saved to {wfp}')
                 analysis_data_single['output_files']['pose_visualization_image'] = wfp

             elif args.opt == 'ply':
                 wfp_ply = f'examples/results/{base_name}_{args.opt}' + new_suffix
                 ser_to_ply(ver_lst_dense, tddfa.tri, height=img.shape[0], wfp=wfp_ply)
                 print(f'PLY model saved to {wfp_ply}')
                 analysis_data_single['output_files']['ply_model'] = wfp_ply

             elif args.opt == 'obj':
                 wfp_obj = f'examples/results/{base_name}_{args.opt}' + new_suffix
                 ser_to_obj(img, ver_lst_dense, tddfa.tri, height=img.shape[0], wfp=wfp_obj)
                 print(f'OBJ model saved to {wfp_obj}')
                 analysis_data_single['output_files']['obj_model'] = wfp_obj

        elif args.opt != 'all':
             print(f"ОШИБКА: Неизвестная опция визуализации: {args.opt}")
             print("СТАТУС: Генерация визуализаций пропущена.")

        print("СТАТУС: Обновление JSON файла с информацией о сгенерированных файлах.")
        with open(analysis_wfp_single, 'w', encoding='utf-8') as f:
             json.dump(analysis_data_single, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
        print(f'Complete biometric analysis saved to {analysis_wfp_single}')

        print("СТАТУС: Обработка одиночного файла завершена.")

    else:
        print(f'Error: Invalid path {args.img_fp}')
        print(f"СТАТУС: Ошибка - неверный путь {args.img_fp}")
        sys.exit(-1)

if __name__ == '__main__':
    print("СТАТУС: Скрипт запущен")
    parser = argparse.ArgumentParser(description='3DDFA_V2 Complete Biometric Analysis for Identity Verification')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-f', '--img_fp', type=str, default='examples/inputs/trump_hillary.jpg')
    parser.add_argument('-m', '--mode', type=str, default='cpu', help='gpu or cpu mode')
    parser.add_argument('-o', '--opt', type=str, default='all',
                        choices=['2d_sparse', '2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'pose', 'ply', 'obj', 'all', 'original'])
    parser.add_argument('--show_flag', type=str2bool, default='false')
    parser.add_argument('--onnx', action='store_true', default=False)

    args = parser.parse_args()
    main(args)
