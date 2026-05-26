import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import uvicorn
from io import BytesIO
import time
import pandas as pd
from ultralytics import YOLO
from models.convnext import convnext_tiny

# ===================== 全局配置 =====================
INPUT_SIZE = 224

# 模型权重路径配置（按场景和模型分类）
MODEL_PATHS = {
    "simple_resnet50": "experiments/exp1_resnet50_single/20260422_075325/20260422_075325/checkpoints/resnet50_best_20260422_075329.pth",
    "simple_convnext": "experiments/exp1_convnext_single/20260422_020821/20260422_020821/checkpoints/convnext_tiny_best_20260422_020829.pth",
    "complex_resnet50": "experiments/exp2_resnet50_complex/20260427_160710/20260427_160710/checkpoints/resnet50_best_20260427_160720.pth",
    "complex_convnext": "experiments/exp2_convnext_complex/20260427_160907/20260427_160907/checkpoints/convnext_tiny_best_20260427_160917.pth",
}

# 类别配置
CLASS_CONFIG = {
    "simple": {
        "num_classes": 185,
        "train_dir": Path("dataset/images/field"),
        "csv_path": Path("dataset/csv/leafsnap-dataset-test-images.csv"),
    },
    "complex": {
        "num_classes": 9,
        "train_dir": Path("dataset/complex_bg/center"),
        "csv_path": Path("dataset/csv/complex_bg_raw_test.csv"),
    }
}

# YOLOv8模型路径（用于ROI提取）
YOLO_MODEL_PATH = "results/yolov8n_leaf_roi_best.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取类别列表
def get_classes(scene_type: str):
    config = CLASS_CONFIG[scene_type]
    
    # 优先从CSV文件读取类别名
    csv_path = config.get("csv_path")
    if csv_path and csv_path.exists():
        df = pd.read_csv(csv_path)
        if "species" in df.columns:
            return sorted(df["species"].unique().tolist())
    
    # 其次从训练目录读取
    train_dir = config.get("train_dir")
    if train_dir and train_dir.exists():
        return sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    
    # 最后使用默认类别名
    num_classes = config["num_classes"]
    return [f"Class_{i}" for i in range(num_classes)]

# 图像预处理
TRANSFORM = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ===================== ROI提取模块 =====================
class ROIExtractor:
    """ROI提取器，支持center和yolo两种方式"""
    
    def __init__(self):
        self.yolo_model = None
    
    def load_yolo(self):
        """加载YOLOv8模型"""
        if self.yolo_model is None and os.path.exists(YOLO_MODEL_PATH):
            self.yolo_model = YOLO(YOLO_MODEL_PATH)
            print(f"YOLOv8模型加载成功: {YOLO_MODEL_PATH}")
        elif self.yolo_model is None:
            print(f"警告: YOLOv8模型文件不存在: {YOLO_MODEL_PATH}，将使用center作为兜底")
    
    def extract_center(self, image: np.ndarray) -> np.ndarray:
        """中心裁剪ROI提取"""
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2
        size = min(w, h)
        x1 = center_x - size // 2
        y1 = center_y - size // 2
        x2 = x1 + size
        y2 = y1 + size
        roi = image[y1:y2, x1:x2]
        return roi
    
    def extract_yolo(self, image: np.ndarray) -> np.ndarray:
        """YOLOv8检测ROI提取"""
        self.load_yolo()
        
        if self.yolo_model is None:
            return self.extract_center(image)
        
        try:
            results = self.yolo_model(image, verbose=False)
            boxes = results[0].boxes
            
            if boxes is None or len(boxes) == 0:
                return self.extract_center(image)
            
            # 获取置信度最高的检测框
            best_box = boxes[boxes.conf.argmax()]
            x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
            
            # 确保边界在图像范围内
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            roi = image[y1:y2, x1:x2]
            
            if roi.size == 0:
                return self.extract_center(image)
            
            return roi
        except Exception as e:
            print(f"YOLO提取失败: {e}，使用center兜底")
            return self.extract_center(image)
    
    def extract(self, image: np.ndarray, method: str = "center") -> np.ndarray:
        """根据指定方法提取ROI"""
        if method == "yolo":
            return self.extract_yolo(image)
        else:
            return self.extract_center(image)

# ===================== 模型管理器 =====================
class ModelManager:
    def __init__(self):
        self.current_model = None
        self.current_model_name = None
        self.current_scene_type = None
        self.model_target_layers = {
            "resnet50": lambda model: [model.layer4[-1]],
            "convnext": lambda model: [model.stages[-1][-1]]
        }

    def load_model(self, model_key: str, scene_type: str):
        """加载指定模型的权重
        
        Args:
            model_key: 模型名称 (resnet50/convnext)
            scene_type: 场景类型 (simple/complex)
        """
        full_key = f"{scene_type}_{model_key}"
        
        if full_key == self.current_model_name and self.current_model is not None:
            return self.current_model

        # 初始化对应模型
        num_classes = CLASS_CONFIG[scene_type]["num_classes"]
        
        if model_key == "resnet50":
            model = models.resnet50(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        elif model_key == "convnext":
            model = convnext_tiny(num_classes=num_classes)
        else:
            raise ValueError(f"不支持的模型：{model_key}")

        # 加载权重
        weight_path = MODEL_PATHS.get(full_key)
        if weight_path and os.path.exists(weight_path):
            checkpoint = torch.load(weight_path, map_location=DEVICE)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            new_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict, strict=False)
            print(f"成功加载权重: {weight_path}")
        else:
            print(f"警告: 权重文件不存在 {weight_path}，使用随机初始化权重")

        # 模型配置
        model.to(DEVICE)
        model.eval()
        
        # 更新当前模型
        self.current_model = model
        self.current_model_name = full_key
        self.current_scene_type = scene_type
        print(f"成功加载模型：{full_key}")
        return model

    def predict(self, model_key: str, scene_type: str, input_img: np.ndarray, roi_method: str = "center"):
        """核心预测逻辑（含GradCAM和ROI提取）"""
        perf_metrics = {}
        
        # 载模型
        start_time = time.time()
        model = self.load_model(model_key, scene_type)
        perf_metrics['model_load_time'] = time.time() - start_time
        
        # ROI提取
        roi_extractor = ROIExtractor()
        start_time = time.time()
        roi_img = roi_extractor.extract(input_img, roi_method)
        perf_metrics['roi_extract_time'] = time.time() - start_time
        
        # 保存原图尺寸（用于热力图适配）
        original_h, original_w = roi_img.shape[:2]
        
        # 图像预处理
        img_pil = Image.fromarray(roi_img.astype('uint8'), 'RGB')
        input_tensor = TRANSFORM(img_pil).unsqueeze(0).to(DEVICE)
        
        # 推理获取Top5结果
        start_time = time.time()
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)[0]
        top5_prob, top5_catid = torch.topk(probs, 5)
        perf_metrics['inference_time'] = time.time() - start_time
        
        classes = get_classes(scene_type)
        results = {classes[top5_catid[i]]: float(top5_prob[i]) for i in range(5)}
        
        # 生成GradCAM热力图
        start_time = time.time()
        target_layers = self.model_target_layers[model_key](model)
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(top5_catid[0].item())]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        perf_metrics['gradcam_time'] = time.time() - start_time
        
        # 热力图叠加到ROI图
        img_float = np.float32(img_pil) / 255
        grayscale_cam_resized = cv2.resize(grayscale_cam, (original_w, original_h))
        cam_image = show_cam_on_image(img_float, grayscale_cam_resized, use_rgb=True)
        
        perf_metrics['total_time'] = sum(perf_metrics.values())
        
        return results, cam_image, roi_img, perf_metrics

# 初始化模型管理器
model_manager = ModelManager()

# ===================== FastAPI 接口实现 =====================
app = FastAPI(title="叶片识别API", description="支持多场景多模型的叶片分类接口")

@app.post("/predict", summary="叶片分类预测接口")
async def api_predict(
    file: UploadFile = File(description="上传叶片图像文件（jpg/png格式）"),
    model_name: str = "resnet50",
    scene_type: str = "complex",
    roi_method: str = "center"
):
    """
    FastAPI预测接口：接收图片文件、模型名称、场景类型和ROI方法，返回Top5预测结果
    调用示例（curl）：
    curl -X POST "http://localhost:8000/predict?model_name=resnet50&scene_type=complex&roi_method=center" -F "file=@test.jpg"
    """
    # 校验参数
    valid_models = ["resnet50", "convnext"]
    valid_scenes = ["simple", "complex"]
    valid_roi_methods = ["center", "yolo"]
    
    if model_name not in valid_models:
        raise HTTPException(status_code=400, detail=f"不支持的模型：{model_name}，可选值：{valid_models}")
    if scene_type not in valid_scenes:
        raise HTTPException(status_code=400, detail=f"不支持的场景类型：{scene_type}，可选值：{valid_scenes}")
    if roi_method not in valid_roi_methods:
        raise HTTPException(status_code=400, detail=f"不支持的ROI方法：{roi_method}，可选值：{valid_roi_methods}")
    
    # 读取图片
    try:
        contents = await file.read()
        img_pil = Image.open(BytesIO(contents)).convert("RGB")
        input_img = np.array(img_pil)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图片解析失败：{str(e)}")
    
    # 预测
    try:
        results, _, roi_img, perf_metrics = model_manager.predict(model_name, scene_type, input_img, roi_method)
        return JSONResponse(content={
            "code": 200,
            "msg": "预测成功",
            "data": {
                "model_name": model_name,
                "scene_type": scene_type,
                "roi_method": roi_method,
                "top5_results": results,
                "performance_metrics": {
                    "model_load_time_ms": round(perf_metrics['model_load_time'] * 1000, 2),
                    "roi_extract_time_ms": round(perf_metrics['roi_extract_time'] * 1000, 2),
                    "inference_time_ms": round(perf_metrics['inference_time'] * 1000, 2),
                    "gradcam_time_ms": round(perf_metrics['gradcam_time'] * 1000, 2),
                    "total_time_ms": round(perf_metrics['total_time'] * 1000, 2)
                }
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败：{str(e)}")

# ===================== Gradio 界面设计 =====================
def gradio_predict(scene_type: str, model_name: str, roi_method: str, input_img: np.ndarray):
    if input_img is None:
        return {}, None, None, "未加载模型"
    
    try:
        results, cam_image, roi_img, perf_metrics = model_manager.predict(model_name, scene_type, input_img, roi_method)
        
        perf_info = (
            f"{model_name}推理完成！（ROI方法：{roi_method}）\n"
            f"性能指标：模型加载 {perf_metrics['model_load_time']*1000:.1f}ms | "
            f"ROI提取 {perf_metrics['roi_extract_time']*1000:.1f}ms | "
            f"推理 {perf_metrics['inference_time']*1000:.1f}ms | "
            f"GradCAM {perf_metrics['gradcam_time']*1000:.1f}ms | "
            f"总计 {perf_metrics['total_time']*1000:.1f}ms"
        )
        
        return results, cam_image, roi_img, perf_info
        
    except Exception as e:
        return {}, None, None, f"预测失败：{str(e)}"

with gr.Blocks(title="叶片识别系统") as demo:
    gr.Markdown("""
    # 🌿 叶片识别系统
    基于深度学习的叶片物种分类工具，支持多场景（单一背景/复杂背景）、多模型（ResNet50/LeafConvNeXt）、多ROI提取方式（Center/YOLO）。
    """)
    
    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=350):
            gr.Markdown("### 📥 输入配置")
            input_img = gr.Image(
                label="上传叶片图像",
                type="numpy",
                height=350,
                sources=["upload", "webcam"],
                interactive=True,
                elem_id="input-image"
            )
            
            scene_selector = gr.Dropdown(
                label="选择场景类型",
                choices=[
                    ("复杂背景（9类）", "complex"),
                    ("单一背景（185类）", "simple")
                ],
                value="complex",
                interactive=True,
                info="选择训练数据集的场景类型"
            )
            
            model_selector = gr.Dropdown(
                label="选择推理模型",
                choices=[
                    ("ResNet50", "resnet50"),
                    ("LeafConvNeXt", "convnext")
                ],
                value="resnet50",
                interactive=True,
                info="不同模型精度/速度不同，可按需选择"
            )
            
            roi_selector = gr.Dropdown(
                label="选择ROI提取方法",
                choices=[
                    ("中心裁剪（默认）", "center"),
                    ("YOLOv8检测", "yolo")
                ],
                value="center",
                interactive=True,
                info="复杂背景场景推荐使用YOLOv8提取ROI"
            )
            
            predict_btn = gr.Button(
                "开始识别",
                variant="primary",
                size="lg"
            )
            
            status_text = gr.Textbox(
                label="模型状态",
                value="未加载模型",
                interactive=False,
                placeholder="模型加载中...",
                lines=2,
                elem_id="status-box"
            )
        
        with gr.Column(scale=2, min_width=600):
            gr.Markdown("### 📊 预测结果")
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    label_output = gr.Label(
                        num_top_classes=5,
                        label="Top5 物种预测"
                    )
                with gr.Column(scale=1):
                    cam_output = gr.Image(
                        label="Grad-CAM 热力图",
                        type="numpy",
                        height=350,
                        interactive=False
                    )
            
            gr.Markdown("### 🔍 ROI提取结果")
            roi_output = gr.Image(
                label="提取的ROI区域",
                type="numpy",
                height=250,
                interactive=False
            )
    
    gr.Markdown("""
    > ⚠️ 注意：
    > 1. 上传图像建议为清晰的叶片特写，分辨率不低于224x224；
    > 2. 首次选择模型会加载权重，耗时稍长（约10-30秒）；
    > 3. YOLOv8 ROI提取需要预训练的检测模型，若不存在则自动降级为中心裁剪；
    > 4. API接口地址：`http://localhost:8000/predict`（支持curl/postman调用）。
    """)
    
    predict_btn.click(
        fn=gradio_predict,
        inputs=[scene_selector, model_selector, roi_selector, input_img],
        outputs=[label_output, cam_output, roi_output, status_text]
    )

# ===================== 挂载Gradio到FastAPI + 启动服务 =====================
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    # # 方式1 使用Gradio内置share功能(生成公网链接,有效期72小时)
    # demo.launch(
    #     server_name="0.0.0.0",
    #     server_port=8000,
    #     share=True,  # 开启公网分享
    #     show_error=True
    # )
    
    # 方式2 使用uvicorn(仅本地访问)
    uvicorn.run(
        "web_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
