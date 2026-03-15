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

# --- 1. 配置与模型加载 ---
NUM_CLASSES = 185  # 保持和你训练ResNet50时的类别数一致
INPUT_SIZE = 224   # ResNet50默认输入尺寸
MODEL_PATH = "results/resnet50_no_cbam_stage2_best_acc_91.pth"  # 你的ResNet50权重文件路径
TRAIN_DIR = Path("dataset/images/field")

# 获取类别列表 (确保顺序与训练时一致)
if TRAIN_DIR.exists():
    classes = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
else:
    classes = [f"Species {i}" for i in range(NUM_CLASSES)]

# 加载ResNet50模型并替换最后一层
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=None)  # 不加载预训练权重
# 修改全连接层以匹配你的类别数
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

# 加载训练好的权重
if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    # 兼容两种权重格式：带/不带module.前缀（多卡训练）
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    # 移除module.前缀（如果有）
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict,strict = False)
    print(f"成功加载ResNet50模型: {MODEL_PATH}")
else:
    print(f"警告: 找不到模型文件 {MODEL_PATH}，将使用随机初始化权重。")
model.to(device)
model.eval()

# 预处理转换 (适配ResNet50的标准预处理)
transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet均值
        std=[0.229, 0.224, 0.225]    # ImageNet标准差
    )
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
        outputs = model(input_tensor)  # ResNet50只有输出，无单独features
        probs = F.softmax(outputs, dim=1)[0]
    
    # 获取前 5 个结果
    top5_prob, top5_catid = torch.topk(probs, 5)
    results = {classes[top5_catid[i]]: float(top5_prob[i]) for i in range(5)}
    
    # --- Grad-CAM 可视化 (适配ResNet50的目标层) ---
    # ResNet50的最后一个卷积层是layer4[-1]
    target_layers = [model.layer4[-1]]
    
    # GradCAM 包装器 (ResNet50 forward直接返回logits)
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # 生成热力图
    targets = [ClassifierOutputTarget(top5_catid[0].item())]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # 将热力图叠加到原图上 (保持224x224可视化尺寸)
    img_float = np.float32(img_pil.resize((224, 224))) / 255
    grayscale_cam_resized = cv2.resize(grayscale_cam, (224, 224))
    cam_image = show_cam_on_image(img_float, grayscale_cam_resized, use_rgb=True)
    
    return results, cam_image

# --- 3. Gradio 界面 ---
with gr.Blocks(title="叶片识别系统", theme=gr.themes.Soft()) as iface:
    gr.Markdown("# 🌿 叶片识别系统")
    gr.Markdown("上传叶片图像以分类其物种。该模型使用 ResNet50 架构，输入分辨率为 224x224。")
    
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