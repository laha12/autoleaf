import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from models.custom_cnn import custom_cnn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import os
from pathlib import Path

# --- 1. 配置与模型加载 ---
NUM_CLASSES = 185
INPUT_SIZE = 32
MODEL_PATH = "leaf_model_softmax.pth"
TRAIN_DIR = Path("dataset/train")

# 获取类别列表 (确保顺序与训练时一致)
if TRAIN_DIR.exists():
    classes = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
else:
    classes = [f"Species {i}" for i in range(NUM_CLASSES)]

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = custom_cnn(num_classes=NUM_CLASSES)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"成功加载模型: {MODEL_PATH}")
else:
    print(f"警告: 找不到模型文件 {MODEL_PATH}，将使用随机初始化权重。")
model.to(device)
model.eval()

# 预处理转换
transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 2. 预测函数 ---
def predict(input_img):
    if input_img is None:
        return None, None
    
    # 转换图像为 PIL 格式以便 transform
    img_pil = Image.fromarray(input_img.astype('uint8'), 'RGB')
    input_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        outputs, features = model(input_tensor)
        probs = F.softmax(outputs, dim=1)[0]
    
    # 获取前 5 个结果
    top5_prob, top5_catid = torch.topk(probs, 5)
    results = {classes[top5_catid[i]]: float(top5_prob[i]) for i in range(5)}
    
    # --- Grad-CAM 可视化 ---
    # 定义目标层 (通常是最后一个卷积层)
    target_layers = [model.conv2]
    
    # GradCAM 包装器需要 forward 返回 logits
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            out, _ = self.model(x)
            return out
            
    wrapped_model = ModelWrapper(model)
    cam = GradCAM(model=wrapped_model, target_layers=target_layers)
    
    # 生成热力图
    targets = [ClassifierOutputTarget(top5_catid[0].item())]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # 将热力图叠加到原图上
    # 需要先将输入图缩放到模型输入大小 32x32 或保持原图比例
    img_float = np.float32(img_pil.resize((224, 224))) / 255 # 为了可视化效果，我们放大到 224
    grayscale_cam_resized = cv2.resize(grayscale_cam, (224, 224))
    cam_image = show_cam_on_image(img_float, grayscale_cam_resized, use_rgb=True)
    
    return results, cam_image

# --- 3. Gradio 界面 ---
with gr.Blocks(title="叶片识别系统", theme=gr.themes.Soft()) as iface:
    gr.Markdown("# 🌿 叶片识别系统")
    gr.Markdown("上传叶片图像以分类其物种。该模型使用复现论文的 8 层 CNN 架构，输入分辨率为 32x32。")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="上传叶片图像", type="numpy")
            predict_btn = gr.Button("开始识别", variant="primary")
        
        with gr.Column():
            label_output = gr.Label(num_top_classes=5, label="预测结果")
            cam_output = gr.Image(label="Grad-CAM 热力图 (模型关注区域)")
    
    # 事件绑定
    predict_btn.click(
        fn=predict,
        inputs=input_img,
        outputs=[label_output, cam_output]
    )

if __name__ == "__main__":
    iface.launch(share=True)
