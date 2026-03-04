
import torch
import numpy as np
import cv2
from models.custom_cnn import custom_cnn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Mock model wrapper
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
    
    def forward(self, x):
        out, _ = self.model(x)
        return out

def test_gradcam():
    print("Initializing model...")
    model = custom_cnn(num_classes=185)
    model.eval()
    
    target_layers = [model.conv2]
    model_wrapper = ModelWrapper(model)
    
    print("Initializing GradCAM...")
    try:
        cam = GradCAM(model=model_wrapper, target_layers=target_layers)
    except Exception as e:
        print(f"Failed to initialize GradCAM: {e}")
        return

    # Create dummy input (1, 3, 224, 224)
    input_tensor = torch.rand(1, 3, 224, 224)
    
    # Dummy image for visualization (224, 224, 3)
    input_image = np.random.rand(224, 224, 3).astype(np.float32)
    
    print("Running GradCAM...")
    try:
        targets = [ClassifierOutputTarget(0)] # Target class 0
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        visualization = show_cam_on_image(input_image, grayscale_cam, use_rgb=True)
        print("GradCAM successful. Output shape:", visualization.shape)
    except Exception as e:
        print(f"GradCAM failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gradcam()
