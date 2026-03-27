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
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from io import BytesIO
import time

# ===================== 1. 全局配置（用户需根据实际情况修改） =====================
NUM_CLASSES = 185  # 所有模型保持相同类别数
INPUT_SIZE = 224
# 各模型权重路径（用户需替换为实际路径）
MODEL_PATHS = {
    "resnet50": "results/resnet50_no_cbam_stage2_best_acc_91.pth",
    "swin_transformer": "results/swin_transformer_best.pth",
    "convnext": "results/convnext_best.pth"
}
TRAIN_DIR = Path("dataset/images/field")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取类别列表
if TRAIN_DIR.exists():
    CLASSES = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
else:
    CLASSES = [f"Species {i}" for i in range(NUM_CLASSES)]

# 图像预处理（适配所有模型的通用预处理）
TRANSFORM = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ===================== 2. 模型管理器（支持多模型加载/切换） =====================
class ModelManager:
    def __init__(self):
        self.current_model = None
        self.current_model_name = None
        # 各模型的GradCAM目标层配置
        self.model_target_layers = {
            "resnet50": lambda model: [model.layer4[-1]],
            "swin_transformer": lambda model: [model.features[-1]],  # Swin Transformer默认目标层
            "convnext": lambda model: [model.features[-1]]           # ConvNeXt默认目标层
        }

    def load_model(self, model_name: str):
        """加载指定模型的权重"""
        if model_name == self.current_model_name and self.current_model is not None:
            return self.current_model  # 避免重复加载

        # 1. 初始化对应模型
        if model_name == "resnet50":
            model = models.resnet50(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
        elif model_name == "swin_transformer":
            model = models.swin_transformer.swin_t(weights=None)  # 可替换为swin_s/m/l
            model.head = torch.nn.Linear(model.head.in_features, NUM_CLASSES)
        elif model_name == "convnext":
            model = models.convnext_tiny(weights=None)  # 可替换为small/large
            model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
        else:
            raise ValueError(f"不支持的模型：{model_name}")

        # 2. 加载权重
        weight_path = MODEL_PATHS[model_name]
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"模型权重文件不存在：{weight_path}")
        
        checkpoint = torch.load(weight_path, map_location=DEVICE)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        # 移除多卡训练的module.前缀
        new_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        
        # 3. 模型配置
        model.to(DEVICE)
        model.eval()
        
        # 4. 更新当前模型
        self.current_model = model
        self.current_model_name = model_name
        print(f"✅ 成功加载模型：{model_name} (权重路径：{weight_path})")
        return model

    def predict(self, model_name: str, input_img: np.ndarray):
        """核心预测逻辑（含GradCAM，优化热力图尺寸适配）"""
        # 1. 加载模型
        model = self.load_model(model_name)
        
        # 2. 保存原图尺寸（用于热力图适配）
        original_h, original_w = input_img.shape[:2]
        
        # 3. 图像预处理
        img_pil = Image.fromarray(input_img.astype('uint8'), 'RGB')
        input_tensor = TRANSFORM(img_pil).unsqueeze(0).to(DEVICE)
        
        # 4. 推理获取Top5结果
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)[0]
        top5_prob, top5_catid = torch.topk(probs, 5)
        results = {CLASSES[top5_catid[i]]: float(top5_prob[i]) for i in range(5)}
        
        # 5. 生成GradCAM热力图（优化尺寸适配）
        target_layers = self.model_target_layers[model_name](model)
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(top5_catid[0].item())]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        
        # 6. 热力图叠加到原图（关键：适配原图尺寸，而非固定224x224）
        img_float = np.float32(img_pil) / 255  # 不缩放原图，保持原始尺寸
        # 调整热力图尺寸到原图尺寸
        grayscale_cam_resized = cv2.resize(grayscale_cam, (original_w, original_h))
        cam_image = show_cam_on_image(img_float, grayscale_cam_resized, use_rgb=True)
        
        return results, cam_image

# 初始化模型管理器
model_manager = ModelManager()

# ===================== 3. FastAPI 接口实现 =====================
app = FastAPI(title="叶片识别API", description="支持ResNet50/Swin Transformer/ConvNeXt的叶片分类接口")

@app.post("/predict", summary="叶片分类预测接口")
async def api_predict(
    file: UploadFile = File(description="上传叶片图像文件（jpg/png格式）"),
    model_name: str = "resnet50"  # 默认使用resnet50
):
    """
    FastAPI预测接口：接收图片文件和模型名称，返回Top5预测结果
    调用示例（curl）：
    curl -X POST "http://localhost:8000/predict?model_name=resnet50" -F "file=@test.jpg"
    """
    # 1. 校验参数
    if model_name not in MODEL_PATHS.keys():
        raise HTTPException(status_code=400, detail=f"不支持的模型：{model_name}，可选值：{list(MODEL_PATHS.keys())}")
    
    # 2. 读取图片
    try:
        contents = await file.read()
        img_pil = Image.open(BytesIO(contents)).convert("RGB")
        input_img = np.array(img_pil)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图片解析失败：{str(e)}")
    
    # 3. 预测
    try:
        results, _ = model_manager.predict(model_name, input_img)
        return JSONResponse(content={
            "code": 200,
            "msg": "预测成功",
            "data": {
                "model_name": model_name,
                "top5_results": results
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败：{str(e)}")

# ===================== 4. Gradio 界面（修复状态更新 + 热力图适配） =====================
def update_status(text):
    """辅助函数：更新状态文本"""
    return gr.update(value=text)

def gradio_predict(model_name: str, input_img: np.ndarray, status_box):
    """Gradio专用预测函数（修复状态更新逻辑）"""
    if input_img is None:
        status_box = update_status("未加载模型")
        return gr.update(), gr.update(), status_box
    
    try:
        # 第一步：更新状态为加载模型
        status_box = update_status(f"正在加载 {model_name} 模型...")
        yield gr.update(), gr.update(), status_box
        
        # 第二步：加载模型并更新状态
        model = model_manager.load_model(model_name)
        status_box = update_status(f"{model_name} 模型加载完成，正在推理...")
        yield gr.update(), gr.update(), status_box
        
        # 第三步：执行预测（含优化后的热力图）
        results, cam_image = model_manager.predict(model_name, input_img)
        status_box = update_status(f"{model_name} 推理完成！")
        yield results, cam_image, status_box
        
    except Exception as e:
        error_msg = f"预测失败：{str(e)}"
        gr.Warning(error_msg)
        status_box = update_status(error_msg)
        yield {}, None, status_box

# 构建优化后的Gradio界面
with gr.Blocks(
    title="叶片识别系统",
    theme=gr.themes.Soft(
        primary_hue="green",  # 适配植物主题
        secondary_hue="blue",
        neutral_hue="gray"
    )
) as demo:
    # 顶部标题区
    gr.Markdown("""
    # 🌿 叶片识别系统
    基于深度学习的叶片物种分类工具，支持ResNet50/Swin Transformer/ConvNeXt三种模型，
    可可视化模型关注区域（Grad-CAM热力图）。
    """)
    
    # 核心交互区（分栏布局）
    with gr.Row(equal_height=True):
        # 左侧：输入与配置区
        with gr.Column(scale=1, min_width=350):
            gr.Markdown("### 📥 输入配置")
            input_img = gr.Image(
                label="上传叶片图像",
                type="numpy",
                height=350,  # 统一输入图高度
                image_mode="RGB",
                sources=["upload", "webcam"],  # 支持上传/摄像头
                interactive=True,
                elem_id="input-image"
            )
            model_selector = gr.Dropdown(
                label="选择推理模型",
                choices=list(MODEL_PATHS.keys()),
                value="resnet50",
                interactive=True,
                info="不同模型精度/速度不同，可按需选择"
            )
            predict_btn = gr.Button(
                "开始识别",
                variant="primary",
                size="lg",
                icon="✅"
            )
            # 模型加载状态提示（修复更新问题）
            status_text = gr.Textbox(
                label="模型状态",
                value="未加载模型",
                interactive=False,
                placeholder="模型加载中...",
                lines=2,  # 增加行数，避免文字截断
                elem_id="status-box"
            )
        
        # 右侧：输出展示区
        with gr.Column(scale=2, min_width=600):
            gr.Markdown("### 📊 预测结果")
            with gr.Row(equal_height=True):
                # 左子列：Top5结果
                with gr.Column(scale=1):
                    label_output = gr.Label(
                        num_top_classes=5,
                        label="Top5 物种预测",
                        height=350,  # 与输入图高度一致
                        elem_id="label-result"
                    )
                # 右子列：GradCAM热力图（优化显示）
                with gr.Column(scale=1):
                    cam_output = gr.Image(
                        label="Grad-CAM 热力图（模型关注区域）",
                        type="numpy",
                        height=350,  # 与输入图高度一致
                        interactive=False,
                        elem_id="cam-image",
                        # 关键：让热力图自适应容器，保持比例
                        image_mode="RGB",
                        show_download_button=True,
                        container=True
                    )
    
    # 底部说明区
    gr.Markdown("""
    > ⚠️ 注意：
    > 1. 上传图像建议为清晰的叶片特写，分辨率不低于224x224；
    > 2. 首次选择模型会加载权重，耗时稍长（约10-30秒）；
    > 3. API接口地址：`http://localhost:8000/predict`（支持curl/postman调用）。
    """)
    
    # 事件绑定（修复状态更新逻辑）
    # 模型选择切换时更新状态
    @model_selector.change
    def update_model_status(model_name):
        return f"已选择模型：{model_name}（点击“开始识别”加载）"
    
    # 点击预测按钮执行推理（使用yield实现分步更新）
    predict_btn.click(
        fn=gradio_predict,
        inputs=[model_selector, input_img, status_text],
        outputs=[label_output, cam_output, status_text],
        # 启用流式输出，实现状态分步更新
        stream=True
    )

# ===================== 5. 挂载Gradio到FastAPI + 启动服务 =====================
# 将Gradio应用挂载到FastAPI的根路径
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    # 启动FastAPI服务（同时包含Gradio界面）
    uvicorn.run(
        "web_app:app",
        host="0.0.0.0",  # 允许局域网访问
        port=8000,
        reload=True,  # 开发模式热重载（生产环境关闭）
        log_level="info"
    )