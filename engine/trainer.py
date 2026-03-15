import torch
from tqdm import tqdm
from engine.evaluator import evaluate
import time
import torch.optim as optim
from torch.nn import utils


def train_stage(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        cfg,
        epochs,
        stage_name,
        device
        ):

    best_acc = 0
    best_state = None
    patience_counter = 0

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode=cfg.scheduler.mode,
        factor=cfg.scheduler.factor,
        patience=cfg.scheduler.patience,
        min_lr=cfg.scheduler.min_lr
    )

    for epoch in range(epochs):

        model.train()

        correct = 0
        total = 0
        train_loss = 0
        epoch_start = time.time()

        for images,labels in tqdm(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs,labels)

            loss.backward()

            utils.clip_grad_norm_(model.parameters(), max_norm=cfg.train.grad_clip)
            optimizer.step()

            _,pred = torch.max(outputs,1)

            train_loss += loss.item()
            total += labels.size(0)

            correct += (pred==labels).sum().item()

        train_acc = 100*correct/total
        train_loss /= len(train_loader)
        
        val_loss,val_acc = evaluate(
            model,val_loader,criterion,device
        )

        epoch_time = time.time()-epoch_start

        print(
            "[INFO] "
            f"Time {epoch_time:.2f} "
            f"TrainLoss {train_loss:.2f} "
            f"TrainAcc {train_acc:.2f} "
            f"ValLoss{val_loss:.2f} "
            f"ValAcc {val_acc:.2f} "
        )

        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'[INFO] Current {stage_name} learning rate: {current_lr:.6f}')

        if val_acc>best_acc+cfg.early_stop.min_delta:
            patience_counter = 0
            best_acc = val_acc
            best_state = model.state_dict()
        else:
            patience_counter += 1
            print(f'[INFO] Patience counter: {patience_counter}/{cfg.early_stop.patience}')
            if patience_counter>=cfg.early_stop.patience:
                print(f'[INFO] Early stopping at epoch {epoch+1}')
                break
    model.load_state_dict(best_state)

    return model