import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tv_models
from torchvision import datasets, transforms

from averagemeter import AverageMeter  # 确保你的 averagemeter.py 里有 AverageMeter 类

# -------------------------
# 常量
# -------------------------
INPUT_SIZE = 224
NUM_CLASSES = 185
USE_CUDA = torch.cuda.is_available()

best_prec1 = 0.0
classes = []

# -------------------------
# ARGS Parser
# -------------------------
parser = argparse.ArgumentParser(description='PyTorch LeafSnap Training (torch 1.12 + modelid)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--modelid', default=3, type=int, metavar='MODEL_ID',
                    help='1(resnet18), 2(resnet18 variant), 3(VGG16 in your model.py), '
                         '4(resnet50), 5(densenet121), 6(logisticRegression)')
args = parser.parse_args()
MODEL_ID = args.modelid



def select_model(model_id: int):
    """
    对齐你给的 model.py 逻辑：
    return model, modelName, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, ALPHA
    """
    if model_id == 1:
        BATCH_SIZE = 128
        NUM_EPOCHS = 100
        LEARNING_RATE = 1e-1
        ALPHA = 6
        model = tv_models.resnet18(pretrained=False)
        model.fc = nn.Linear(512, NUM_CLASSES)
        modelName = "resnet18_augment"

    elif model_id == 2:
        BATCH_SIZE = 128
        NUM_EPOCHS = 72
        LEARNING_RATE = 1e-1
        ALPHA = 6
        model = tv_models.resnet18(pretrained=False)
        model.fc = nn.Linear(512, NUM_CLASSES)
        modelName = "resnet18_decay_adam"

    elif model_id == 3:
        # 你给的 model.py 里写的是 models.VGG('VGG16')（自定义 VGG）
        # 这里为了“无需依赖你自定义 models 包也能跑”，用 torchvision 的 vgg16 代替。
        # 如果你确实有自己的 models.VGG('VGG16')，把下面两行替换成你的实现即可。
        BATCH_SIZE = 128
        NUM_EPOCHS = 50
        LEARNING_RATE = 1e-1
        ALPHA = 6
        model = tv_models.vgg16(pretrained=False)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
        modelName = "VGG16"

    elif model_id == 4:
        BATCH_SIZE = 64
        NUM_EPOCHS = 100
        LEARNING_RATE = 1e0
        ALPHA = 10
        model = tv_models.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, NUM_CLASSES)
        modelName = "resnet50"

    elif model_id == 5:
        BATCH_SIZE = 8
        NUM_EPOCHS = 100
        LEARNING_RATE = 1e-1
        ALPHA = 6
        model = tv_models.densenet121(pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
        modelName = "densenet121"

    elif model_id == 6:
        BATCH_SIZE = 1024
        NUM_EPOCHS = 100
        LEARNING_RATE = 1e-1
        ALPHA = 6
        model = nn.Sequential()
        model.add_module("linear", nn.Linear(INPUT_SIZE * INPUT_SIZE * 3, NUM_CLASSES, bias=False))
        modelName = "logisticRegression"

    else:
        print('Model ID must be an integer between 1 and 6')

    return model, modelName, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, ALPHA


# -------------------------
# 工具函数：计算 TopK 准确率
# -------------------------
@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# -------------------------
# 训练 1 个 epoch（仅新增：MODEL_ID==6 时 flatten）
# -------------------------
def train_one_epoch(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    for i, (inputs, targets) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if USE_CUDA:
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        # 对齐 model.py：logisticRegression 需要 flatten
        if MODEL_ID == 6:
            inputs = inputs.view(inputs.size(0), -1)  # [B, 224*224*3]

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            print(
                f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                f'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
            )


# -------------------------
# 验证（仅新增：MODEL_ID==6 时 flatten）
# -------------------------
@torch.no_grad()
def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    end = time.time()
    for i, (inputs, targets) in enumerate(val_loader):
        if USE_CUDA:
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        if MODEL_ID == 6:
            inputs = inputs.view(inputs.size(0), -1)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print(
                f'Test: [{i}/{len(val_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                f'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
            )

    print(f' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}')
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        print('\n[INFO] Saved Best Model to model_best.pth.tar')
        shutil.copyfile(filename, 'model_best.pth.tar')


# -------------------------
# 学习率调整 用 ALPHA 控制衰减间隔
# -------------------------
def adjust_learning_rate(optimizer, epoch):
    lr = LEARNING_RATE * (0.1 ** (epoch // ALPHA))
    lr = max(lr, 1e-4)
    print(f'\n[Learning Rate] {lr:.6f}')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# -------------------------
# main：创建模型（改为 modelid 选择），其余结构不变
# -------------------------
print('\n[INFO] Creating Model (by modelid)')
model, modelName, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, ALPHA = select_model(MODEL_ID)
print(f'[INFO] Selected MODEL_ID={MODEL_ID} -> {modelName} | '
      f'BATCH_SIZE={BATCH_SIZE} NUM_EPOCHS={NUM_EPOCHS} LR={LEARNING_RATE} ALPHA={ALPHA}')

print(f'\n[INFO] Model Architecture:\n{model}')

criterion = nn.CrossEntropyLoss()

if USE_CUDA:
    model = torch.nn.DataParallel(model).cuda()
    criterion = criterion.cuda()

optimizer = optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True
)

start_epoch = 1

# resume
if args.resume:
    if os.path.isfile(args.resume):
        print(f"=> loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location='cuda' if USE_CUDA else 'cpu')

        start_epoch = checkpoint.get('epoch', 1)
        best_prec1 = checkpoint.get('best_prec1', 0.0)

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        print(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch}) best_prec1={best_prec1:.3f}")
    else:
        print(f"=> no checkpoint found at '{args.resume}'")

print('\n[INFO] Reading Training and Testing Dataset')
traindir = os.path.join('dataset', 'train')
testdir = os.path.join('dataset', 'test')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

data_train = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
)

data_test = datasets.ImageFolder(
    testdir,
    transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
)

classes = data_train.classes
print(f'[INFO] Num train samples: {len(data_train)} | Num test samples: {len(data_test)}')
print(f'[INFO] Num classes: {len(classes)}')

train_loader = torch.utils.data.DataLoader(
    data_train,
    batch_size=BATCH_SIZE,          
    shuffle=True,
    num_workers=2,
    pin_memory=USE_CUDA
)

val_loader = torch.utils.data.DataLoader(
    data_test,
    batch_size=BATCH_SIZE,          
    shuffle=False,
    num_workers=2,
    pin_memory=USE_CUDA
)

print('\n[INFO] Training Started')
for epoch in range(start_epoch, NUM_EPOCHS + 1):
    adjust_learning_rate(optimizer, epoch)

    train_one_epoch(train_loader, model, criterion, optimizer, epoch)
    prec1 = validate(val_loader, model, criterion)

    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)

    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
    }, is_best)

    torch.save(model.state_dict(), f'leafsnap_{modelName}_state_dict.pth')
    print(f'[INFO] Saved leafsnap_{modelName}_state_dict.pth')

print('\n[DONE]')
