import torch


def evaluate(model, loader, criterion, device, k=5):
    """
    评估模型性能
    
    Args:
        model: 待评估模型
        loader: 数据加载器
        criterion: 损失函数
        device: 计算设备
        k: Top-K准确率的K值
    
    Returns:
        包含loss、top1_acc、topk_acc的字典
    """
    model.eval()

    loss_sum = 0
    correct_top1 = 0
    correct_topk = 0
    total = 0
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss_sum += loss.item() * labels.size(0)

            # Top-1准确率
            _, pred_top1 = torch.max(outputs, 1)
            correct_top1 += (pred_top1 == labels).sum().item()
            
            # Top-K准确率
            _, pred_topk = torch.topk(outputs, k, dim=1)
            correct_topk += pred_topk.eq(labels.view(-1, 1).expand_as(pred_topk)).sum().item()

            total += labels.size(0)
            
            all_preds.extend(pred_top1.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    top1_acc = 100 * correct_top1 / total
    topk_acc = 100 * correct_topk / total

    return {
        "loss": loss_sum / total,
        "top1_acc": top1_acc,
        f"top{k}_acc": topk_acc,
        "all_preds": all_preds,
        "all_labels": all_labels
    }