"""
几何计算工具模块
提供边界框、点、速度等几何计算功能
"""

import numpy as np


def bbox_center(bbox):
    """
    计算边界框中心点

    Args:
        bbox: 边界框 [x1, y1, x2, y2]

    Returns:
        tuple: (center_x, center_y)
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def point_in_bbox(point, bbox, expansion=1.0):
    """
    判断点是否在边界框内（支持扩展）

    Args:
        point: 点坐标 (x, y)
        bbox: 边界框 [x1, y1, x2, y2]
        expansion: 扩展比例，1.0表示不扩展，1.2表示扩展20%

    Returns:
        bool: 点是否在边界框内
    """
    x, y = point
    x1, y1, x2, y2 = bbox

    # 扩展边界框
    if expansion != 1.0:
        w, h = x2 - x1, y2 - y1
        expand_w = w * (expansion - 1) / 2
        expand_h = h * (expansion - 1) / 2
        x1 -= expand_w
        x2 += expand_w
        y1 -= expand_h
        y2 += expand_h

    return x1 <= x <= x2 and y1 <= y <= y2


def calculate_velocity(pos1, pos2, time_delta=1):
    """
    计算速度向量

    Args:
        pos1: 起始位置 (x, y)
        pos2: 结束位置 (x, y)
        time_delta: 时间间隔（帧数）

    Returns:
        tuple: (vx, vy, speed) - x方向速度、y方向速度、总速度
    """
    vx = (pos2[0] - pos1[0]) / time_delta
    vy = (pos2[1] - pos1[1]) / time_delta
    speed = np.sqrt(vx**2 + vy**2)
    return vx, vy, speed


def bbox_iou(bbox1, bbox2):
    """
    计算两个边界框的IoU (Intersection over Union)

    Args:
        bbox1: 边界框1 [x1, y1, x2, y2]
        bbox2: 边界框2 [x1, y1, x2, y2]

    Returns:
        float: IoU值 (0-1)
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # 计算交集区域
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # 计算并集区域
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0
