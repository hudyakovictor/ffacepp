# before import, make sure FaceBoxes and Sim3DR are built successfully, e.g.,
import sys
from subprocess import call
import os
import torch

torch.hub.download_url_to_file('https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Solvay_conference_1927.jpg/1400px-Solvay_conference_1927.jpg', 'solvay.jpg')

def run_cmd(command):
    try:
        print(command)
        call(command, shell=True)
    except Exception as e:
        print(f"Errorrrrr: {e}!")
        
print(os.getcwd())
os.chdir("/Users/victorkhudyakov/nn/3DDFA2/FaceBoxes/utils")
print(os.getcwd())
run_cmd("python3 build.py build_ext --inplace")
os.chdir("/Users/victorkhudyakov/nn/3DDFA2/Sim3DR")
print(os.getcwd())
run_cmd("python3 setup.py build_ext --inplace")
print(os.getcwd())
os.chdir("/Users/victorkhudyakov/nn/3DDFA2/utils/asset")
print(os.getcwd())
run_cmd("gcc -shared -Wall -O3 render.c -o render.so -fPIC")
os.chdir("/Users/victorkhudyakov/nn/3DDFA2")
print(os.getcwd())


import cv2
import yaml

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
from utils.depth import depth
from utils.pncc import pncc
from utils.uv import uv_tex
from utils.pose import viz_pose
from utils.serialization import ser_to_ply, ser_to_obj
from utils.functions import draw_landmarks, get_suffix

import matplotlib.pyplot as plt
from skimage import io
import gradio as gr
import tempfile
import zipfile
import json

# load config
cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)

# Init FaceBoxes and TDDFA, recommend using onnx flag
onnx_flag = True  # or True to use ONNX to speed up
if onnx_flag:    
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OMP_NUM_THREADS'] = '4'
    from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
    from TDDFA_ONNX import TDDFA_ONNX

    face_boxes = FaceBoxes_ONNX()
    tddfa = TDDFA_ONNX(**cfg)
else:
    face_boxes = FaceBoxes()
    tddfa = TDDFA(gpu_mode=False, **cfg)
    


def inference (img):
    # face detection
    boxes = face_boxes(img)
    # regress 3DMM params
    param_lst, roi_box_lst = tddfa(img, boxes)
    # reconstruct vertices and render
    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
    return render(img, ver_lst, tddfa.tri, alpha=0.6, show_flag=False);    


title = "3DDFA V2"
description = "demo for 3DDFA V2. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2009.09960'>Towards Fast, Accurate and Stable 3D Dense Face Alignment</a> | <a href='https://github.com/cleardusk/3DDFA_V2'>Github Repo</a></p>"
examples = [
    ['solvay.jpg']
]
# --- СТАРЫЙ ИНТЕРФЕЙС ЗАКОММЕНТИРОВАН ---
# gr.Interface(
#     fn=inference,
#     inputs=gr.Image(type="numpy", label="Input"),
#     outputs=gr.Image(type="numpy", label="Output"),
#     title=title,
#     description=description,
#     article=article,
#     examples=examples
# ).launch()

import glob
import math
import pandas as pd

# --- НОВЫЕ ИМПОРТЫ ДЛЯ РАСШИРЕННОГО ИНТЕРФЕЙСА ---
import io
import base64
from PIL import Image, ImageDraw, ImageFont

# --- РАСШИРЕННОЕ ОПИСАНИЕ И FAQ ---
extended_description = """
<b>Что делает этот сервис?</b><br>
Этот сервис автоматически определяет наклон головы (углы yaw, pitch, roll) на ваших фотографиях с помощью нейросети 3DDFA V2.<br><br>
<b>Что означают углы?</b><br>
<ul>
<li><b>Yaw (рысканье)</b> — поворот головы влево/вправо (ось Y).<br>Отрицательное значение — влево, положительное — вправо.</li>
<li><b>Pitch (тангаж)</b> — наклон головы вверх/вниз (ось X).<br>Отрицательное значение — вниз, положительное — вверх.</li>
<li><b>Roll (крен)</b> — наклон головы влево/вправо (ось Z).<br>Отрицательное значение — влево, положительное — вправо.</li>
</ul>
<b>Как пользоваться?</b><br>
1. <b>Пакетная обработка папки:</b> укажите путь к папке с изображениями (jpg, png, jpeg).<br>
2. <b>Пакетная обработка файлов:</b> загрузите несколько изображений вручную.<br>
3. Дождитесь завершения обработки — появится таблица с результатами.<br>
4. Можно скачать результаты в формате CSV.<br><br>
<b>Ошибки:</b><br>
- <i>no face</i> — лицо не найдено на фото.<br>
- <i>not image</i> — файл не является изображением.<br>
- <i>прочие</i> — технические ошибки, попробуйте другое фото.<br><br>
<b>FAQ:</b><br>
- <b>Какие форматы поддерживаются?</b> — jpg, png, jpeg.<br>
- <b>Можно ли обрабатывать сразу много фото?</b> — Да, ограничение только по памяти.<br>
- <b>Как интерпретировать углы?</b> — См. выше.<br>
- <b>Где посмотреть исходный код?</b> — <a href='https://github.com/cleardusk/3DDFA_V2'>GitHub 3DDFA_V2</a><br>
"""

# --- ФУНКЦИЯ ДЛЯ ВИЗУАЛИЗАЦИИ РЕЗУЛЬТАТОВ НА ИЗОБРАЖЕНИИ ---
def draw_results_on_image(img, boxes, angles):
    img_draw = img.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        yaw, pitch, roll = angles[i]
        text = f"Yaw: {yaw:.1f}\nPitch: {pitch:.1f}\nRoll: {roll:.1f}"
        y0 = y1 - 40 if y1 - 40 > 0 else y1 + 5
        for j, line in enumerate(text.split('\n')):
            cv2.putText(img_draw, line, (x1, y0 + 20 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return img_draw

def image_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    img_bytes = buffer.tobytes()
    return "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode()

# --- ДОРАБОТАННАЯ process_folder ---
def process_folder(folder_path):
    results = []
    previews = []
    logs = []
    image_paths = glob.glob(os.path.join(folder_path, '*.jpg')) + \
                  glob.glob(os.path.join(folder_path, '*.png')) + \
                  glob.glob(os.path.join(folder_path, '*.jpeg'))
    for img_path in image_paths:
        print(f"Обрабатываю файл: {img_path}")
        img = cv2.imread(img_path)
        if img is None:
            error_msg = f"{os.path.basename(img_path)}: не удалось открыть как изображение."
            print(error_msg)
            logs.append(error_msg)
            results.append({'file': os.path.basename(img_path), 'yaw': None, 'pitch': None, 'roll': None, 'error': 'not image'})
            previews.append(None)
            continue
        boxes = face_boxes(img)
        print(f"Найдено лиц: {len(boxes)}")
        if len(boxes) == 0:
            error_msg = f"{os.path.basename(img_path)}: лицо не найдено."
            print(error_msg)
            logs.append(error_msg)
            results.append({'file': os.path.basename(img_path), 'yaw': None, 'pitch': None, 'roll': None, 'error': 'no face'})
            previews.append(img)
            continue
        param_lst, roi_box_lst = tddfa(img, boxes)
        angles = []
        for i, param in enumerate(param_lst):
            try:
                from utils.pose import P2sRt, matrix2angle
                P = param[:12].reshape(3, 4)
                s, R, t = P2sRt(P)
                yaw_rad, pitch_rad, roll_rad = matrix2angle(R)
                yaw = math.degrees(yaw_rad)
                pitch = math.degrees(pitch_rad)
                roll = math.degrees(roll_rad)
                results.append({
                    'file': os.path.basename(img_path),
                    'yaw': yaw,
                    'pitch': pitch,
                    'roll': roll,
                    'error': ''
                })
                angles.append((yaw, pitch, roll))
                log_msg = f"{os.path.basename(img_path)}: лицо найдено, yaw={yaw:.1f}, pitch={pitch:.1f}, roll={roll:.1f}"
                print(log_msg)
                logs.append(log_msg)
            except Exception as e:
                error_msg = f"{os.path.basename(img_path)}: ошибка {str(e)}"
                print(error_msg)
                logs.append(error_msg)
                results.append({'file': os.path.basename(img_path), 'yaw': None, 'pitch': None, 'roll': None, 'error': str(e)})
                angles.append((None, None, None))
        # Визуализация
        if len(boxes) > 0 and len(angles) > 0:
            img_vis = draw_results_on_image(img, boxes, angles)
            previews.append(img_vis)
        else:
            previews.append(img)
    if not results:
        return pd.DataFrame([{'file': 'Нет изображений', 'yaw': '', 'pitch': '', 'roll': '', 'error': ''}]), [], "Нет изображений для обработки.", None
    # Сохраняем логи в файл
    with open("debug_log.txt", "a", encoding="utf-8") as f:
        for log in logs:
            f.write(log + "\n")
    return pd.DataFrame(results), previews, "\n".join(logs), results

def save_json(results):
    if not results:
        return None
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        return f.name

def gradio_process_folder(folder_path):
    df, previews, logs, results = process_folder(folder_path)
    json_file = save_json(results)
    return df, previews, logs, json_file

folder_selector = gr.Textbox(label="Путь к папке с фото (например, /Users/victorkhudyakov/nn/3DDFA2/input)", info="Укажите абсолютный путь к папке. Поддерживаются jpg, png, jpeg.")

folder_iface = gr.Interface(
    fn=gradio_process_folder,
    inputs=folder_selector,
    outputs=[
        gr.Dataframe(headers=["file", "yaw", "pitch", "roll", "error"], label="Результаты (можно скачать как CSV)", type="pandas"),
        gr.Gallery(label="Визуализация результатов", show_label=True, elem_id="gallery"),
        gr.Textbox(label="Лог обработки", lines=8, interactive=False),
        gr.File(label="Скачать JSON с результатами")
    ],
    title="3DDFA V2: пакетная обработка папки с фото",
    description=extended_description,
    allow_flagging="never",
    examples=[['input/']],
    article="<b>Ссылки:</b> <a href='https://arxiv.org/abs/2009.09960'>Статья</a> | <a href='https://github.com/cleardusk/3DDFA_V2'>GitHub</a>"
)

# --- ДОРАБОТАННАЯ process_files ---
def process_files(files):
    import pandas as pd
    import math
    results = []
    previews = []
    logs = []
    for img_path in files:
        print(f"Обрабатываю файл: {img_path}")
        img = cv2.imread(img_path)
        if img is None:
            error_msg = f"{os.path.basename(img_path)}: не удалось открыть как изображение."
            print(error_msg)
            logs.append(error_msg)
            results.append({'file': os.path.basename(img_path), 'yaw': None, 'pitch': None, 'roll': None, 'error': 'not image'})
            previews.append(None)
            continue
        boxes = face_boxes(img)
        print(f"Найдено лиц: {len(boxes)}")
        if len(boxes) == 0:
            error_msg = f"{os.path.basename(img_path)}: лицо не найдено."
            print(error_msg)
            logs.append(error_msg)
            results.append({'file': os.path.basename(img_path), 'yaw': None, 'pitch': None, 'roll': None, 'error': 'no face'})
            previews.append(img)
            continue
        param_lst, roi_box_lst = tddfa(img, boxes)
        angles = []
        for i, param in enumerate(param_lst):
            try:
                from utils.pose import P2sRt, matrix2angle
                P = param[:12].reshape(3, 4)
                s, R, t = P2sRt(P)
                yaw_rad, pitch_rad, roll_rad = matrix2angle(R)
                yaw = math.degrees(yaw_rad)
                pitch = math.degrees(pitch_rad)
                roll = math.degrees(roll_rad)
                results.append({
                    'file': os.path.basename(img_path),
                    'yaw': yaw,
                    'pitch': pitch,
                    'roll': roll,
                    'error': ''
                })
                angles.append((yaw, pitch, roll))
                log_msg = f"{os.path.basename(img_path)}: лицо найдено, yaw={yaw:.1f}, pitch={pitch:.1f}, roll={roll:.1f}"
                print(log_msg)
                logs.append(log_msg)
            except Exception as e:
                error_msg = f"{os.path.basename(img_path)}: ошибка {str(e)}"
                print(error_msg)
                logs.append(error_msg)
                results.append({'file': os.path.basename(img_path), 'yaw': None, 'pitch': None, 'roll': None, 'error': str(e)})
                angles.append((None, None, None))
        # Визуализация
        if len(boxes) > 0 and len(angles) > 0:
            img_vis = draw_results_on_image(img, boxes, angles)
            previews.append(img_vis)
        else:
            previews.append(img)
    if not results:
        return pd.DataFrame([{'file': 'Нет изображений', 'yaw': '', 'pitch': '', 'roll': '', 'error': ''}]), [], "Нет изображений для обработки.", None
    # Сохраняем логи в файл
    with open("debug_log.txt", "a", encoding="utf-8") as f:
        for log in logs:
            f.write(log + "\n")
    return pd.DataFrame(results), previews, "\n".join(logs), results

def gradio_process_files(files):
    df, previews, logs, results = process_files(files)
    json_file = save_json(results)
    return df, previews, logs, json_file

files_iface = gr.Interface(
    fn=gradio_process_files,
    inputs=gr.File(label="Загрузите изображения (jpg, png)", file_count="multiple", type="filepath"),
    outputs=[
        gr.Dataframe(headers=["file", "yaw", "pitch", "roll", "error"], label="Результаты (можно скачать как CSV)", type="pandas"),
        gr.Gallery(label="Визуализация результатов", show_label=True, elem_id="gallery2"),
        gr.Textbox(label="Лог обработки", lines=8, interactive=False),
        gr.File(label="Скачать JSON с результатами")
    ],
    title="3DDFA V2: пакетная обработка изображений",
    description=extended_description,
    allow_flagging="never",
    article="<b>Ссылки:</b> <a href='https://arxiv.org/abs/2009.09960'>Статья</a> | <a href='https://github.com/cleardusk/3DDFA_V2'>GitHub</a>"
)

# --- ГЛАВНЫЙ ТАББЕД-ИНТЕРФЕЙС ---
demo = gr.TabbedInterface(
    [folder_iface, files_iface],
    ["Папка с фото", "Загрузка файлов"],
    title="3DDFA V2: расширенный анализ наклона головы"
)

demo.launch()
