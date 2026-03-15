import torch


def evaluate(model,loader,criterion,device):

    model.eval()

    loss_sum = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images,labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs,labels)

            loss_sum += loss.item()*labels.size(0)

            _,pred = torch.max(outputs,1)

            total += labels.size(0)

            correct += (pred==labels).sum().item()

    acc = 100*correct/total

    return loss_sum/total,acc