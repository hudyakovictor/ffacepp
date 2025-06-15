# coding: utf-8

__author__ = 'cleardusk'

import sys

sys.path.append('..')

import cv2
import numpy as np

from Sim3DR import RenderPipeline
from utils.functions import plot_image
from .tddfa_util import _to_ctype

cfg = {
    'intensity_ambient': 0.5,  # Почти нет фонового освещения
    'color_ambient': (0.0, 0.0, 0.0),  # Чёрный фон

    'intensity_directional': 1.0,  # Яркий направленный свет
    'color_directional': (0.0, 0.0, 1.0),  # Красный цвет света

    'intensity_specular': 0.3,  # Немного блеска
    'specular_exp': 90,  # Очень острые блики

    'light_pos': (0, 10, 5),  # Можно немного сместить свет для интересных теней
    'view_pos': (0, 0, 5)
}

render_app = RenderPipeline(**cfg)


def render(img, ver_lst, tri, alpha=0.6, show_flag=False, wfp=None, with_bg_flag=True):
    if with_bg_flag:
        overlap = img.copy()
    else:
        overlap = np.zeros_like(img)

    for ver_ in ver_lst:
        ver = _to_ctype(ver_.T)  # transpose
        overlap = render_app(ver, tri, overlap)

    if with_bg_flag:
        res = cv2.addWeighted(img, 1 - alpha, overlap, alpha, 0)
    else:
        res = overlap

    if wfp is not None:
        cv2.imwrite(wfp, res)
        print(f'Save visualization result to {wfp}')

    if show_flag:
        plot_image(res)

    return res
