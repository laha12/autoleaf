import torch
from PIL import Image
import torchvision.transforms as transforms
from utils.get_cuda import get_cuda
from models.resnet50 import build_resnet50


_,DEVICE = get_cuda()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])


def predict(image_path):

    model = build_resnet50(185,False)

    model.load_state_dict(
        torch.load("results/resnet50_best.pth")
    )

    model = model.to(DEVICE)

    model.eval()

    img = Image.open(image_path).convert("RGB")

    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        output = model(img)

        _,pred = torch.max(output,1)

    return pred.item()


if __name__=="__main__":

    print(predict("test.jpg"))