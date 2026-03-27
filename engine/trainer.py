import torch
from tqdm import tqdm
from torch.cuda.amp import autocast 
from engine.evaluator import evaluate
import time
from torch.nn import utils


def train_stage(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        scheduler,
        scheduler_mode,
        scaler,
        cfg,
        epochs,
        stage_name,
        device
):
    best_acc = 0.0
    best_state = model.state_dict()  # 初始化best_state为模型初始状态，避免None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        train_loss = 0.0
        epoch_start = time.time()

        # 训练循环（tqdm显示进度）
        for images, labels in tqdm(train_loader, desc=f"{stage_name} Epoch {epoch+1}/{epochs}"):
            images = images.to(device, non_blocking=True)  # non_blocking加速GPU传输
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()  
            # 混合精度前向传播
            # scaler 为 None 则不混合精度
            with autocast(enabled=scaler is not None):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # 反向传播 + 梯度处理
            if scaler is not None:
                scaler.scale(loss).backward()
                # 混合精度下：先unscale梯度，再裁剪
                scaler.unscale_(optimizer)
                utils.clip_grad_norm_(model.parameters(), max_norm=cfg.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                utils.clip_grad_norm_(model.parameters(), max_norm=cfg.train.grad_clip)
                optimizer.step()  # 补充非混合精度的step

            # 基于step的学习率调度（预热+余弦退火必须放这里）
            if scheduler_mode == "cosine":  # 修正拼写
                scheduler.step()

            # 统计训练指标
            _, pred = torch.max(outputs, 1)
            train_loss += loss.item() * images.size(0)  # 按样本数加权，避免batch_size不均影响
            total += labels.size(0)
            correct += (pred == labels).sum().item()

        # 计算epoch级训练指标
        train_acc = 100 * correct / total
        train_loss /= total  # 按总样本数平均，更准确
        # 验证阶段
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start

        # 打印日志（修正格式）
        print(
            "[INFO] "
            f"Time {epoch_time:.2f}s | "
            f"TrainLoss {train_loss:.4f} | "
            f"TrainAcc {train_acc:.2f}% | "
            f"ValLoss {val_loss:.4f} | "
            f"ValAcc {val_acc:.2f}%"
        )

        # 基于val_acc的调度（如ReduceLROnPlateau）放epoch后
        if scheduler_mode != "cosine":
            scheduler.step(val_acc)

        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f'[INFO] Current {stage_name} learning rate: {current_lr:.6f}')

        # 早停逻辑（优化写法）
        if val_acc > best_acc + cfg.early_stop.min_delta:
            patience_counter = 0
            best_acc = val_acc
            best_state = model.state_dict()  # 保存最佳权重
            print(f"[INFO] Update best ValAcc: {best_acc:.2f}%")
        else:
            patience_counter += 1
            print(f'[INFO] Patience counter: {patience_counter}/{cfg.early_stop.patience}')
            if patience_counter >= cfg.early_stop.patience:
                print(f'[INFO] Early stopping at epoch {epoch+1}')
                break

    # 加载最佳权重
    model.load_state_dict(best_state)
    print(f"[INFO] {stage_name} training done. Best ValAcc: {best_acc:.2f}%")
    return model