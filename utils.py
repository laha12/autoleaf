import cv2
import numpy as np
import pandas as pd
import random
from ultralytics import YOLO
from pathlib import Path
from skimage import io

# 全局加载一次 YOLO 模型，避免重复加载
_yolo_model = None

def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        try:
            _yolo_model = YOLO('yolov8n-seg.pt')
        except Exception as e:
            print(f"[WARNING] 加载 YOLO 模型失败: {e}")
    return _yolo_model

def hybrid_roi_extraction(image_rgb, image_path=None):
    """
    混合 ROI 提取逻辑：
    """
    h, w = image_rgb.shape[:2]
    
    # 尝试 YOLO
    model = get_yolo_model()
    if model is not None:
        try:
            # 如果提供了路径，直接用路径推理（更快）；否则用内存图像
            source = image_path if image_path else image_rgb
            results = model(source, conf=0.25, verbose=False)
            
            if results[0].masks is not None and len(results[0].masks) > 0:
                mask = results[0].masks.data[0].cpu().numpy()
                mask = cv2.resize(mask, (w, h))
                mask_binary = (mask > 0.5).astype(np.uint8) * 255
                
                # 抠图（背景变黑）
                cutout = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_binary)
                
                # 获取边界框并裁剪
                box = results[0].boxes[0].xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)
                pad = 20
                roi = cutout[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)]
                return roi
        except Exception as e:
            print(f"[DEBUG] YOLO 提取失败: {e}，将使用兜底方案")

    # 2. 兜底方案：1/6-5/6 比例裁剪
    y1, x1 = int(h / 6), int(w / 6)
    y2, x2 = int(5 * h / 6), int(5 * w / 6)
    return image_rgb[y1:y2, x1:x2]

def rotate(image, angle):
    (h, w) = image.shape[:2]
    # 计算旋转中心
    (cX, cY) = (w // 2, h // 2)
    # -表示顺时针旋转
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    # 旋转矩阵中的余弦和正弦值
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # 调整旋转矩阵的偏移量
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH))


def load_image_and_preprocess(path, segmented_path):
    """
    加载图像并进行预处理。
    """
    path = str(path).strip()
    # 读取原始图像 RGB格式
    try:
        image = io.imread(path)
    except Exception as e:
        print(f"[ERROR] 无法读取原图 {path}: {e}")
        return None

    # 判断是否有分割图
    use_hybrid = True
    if segmented_path and not pd.isna(segmented_path):
        seg_path = Path(str(segmented_path).strip())
        if seg_path.exists():
            segmented_image = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
            if segmented_image is not None:
                use_hybrid = False
                img = segmented_image
                img_h, img_w = img.shape[:2]
                ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                rects = [cv2.boundingRect(cnt) for cnt in contours]
                if rects == []:
                    top_y, bottom_y, left_x, right_x = 0, img_h, 0, img_w
                else:
                    top_y = max(0, min([y for (x, y, w, h) in rects]) - 40)
                    bottom_y = min(img_h, max([y + h for (x, y, w, h) in rects]) + 80)
                    left_x = max(0, min([x for (x, y, w, h) in rects]) - 40)
                    right_x = min(img_w, max([x + w for (x, y, w, h) in rects]) + 80)
                
                # 防止裁剪图片退化
                if top_y == bottom_y: 
                    bottom_y = min(img_h, bottom_y + 1)
                    top_y = max(0, top_y - 1)
                if left_x == right_x: 
                    right_x = min(img_w, right_x + 1)
                    left_x = max(0, left_x - 1)
                
                # 使用矩形裁剪原始图像
                img_cropped = image[top_y:bottom_y, left_x:right_x]
            else:
                print(f"[WARNING] 无法加载分割图 {seg_path}，将使用混合方案。")
        else:
            # 分割图文件不存在
            use_hybrid = True
    else:
        # 没有提供分割图路径
        use_hybrid = True

    if use_hybrid:
        # 调用混合方案：YOLO 精准提取 + 传统 1/6-5/6 兜底
        img_cropped = hybrid_roi_extraction(image, image_path=path)


    if img_cropped is not None and img_cropped.size > 0:
        img_final = cv2.resize(img_cropped, (224, 224))
        return img_final
    else:
        return None

