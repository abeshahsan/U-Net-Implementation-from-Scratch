from tqdm import tqdm
import torch
from Training.scores import get_scores
from Training.history import History


def train_fn(epoch, loader, model, optimizer, loss_fn, scaler, device, history: History = None):

    loop = tqdm(loader, desc=f"Epoch {epoch}: ", unit="batch", dynamic_ncols=100)

    avg_loss = 0
    avg_acc = 0
    avg_mIoU = 0

    for batch_idx, (image, mask) in enumerate(loop):

        image = image.to(device=device)
        mask = mask.long().to(device=device)

        # forward
        optimizer.zero_grad()

        predictions = model(image)
        loss = loss_fn(predictions, mask)
        score = get_scores(predictions, mask)
        acc = score['pixel_acc']
        iou = score['mIoU']

        # backward
        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix({
            "Loss": loss.item(),
            "Pixel Accuracy": acc.item(),
            "mIoU": iou.item()}
            )

        avg_loss += loss.item()
        avg_acc += acc.item()
        avg_mIoU += iou.item()
        torch.cuda.empty_cache()
    return predictions, mask, avg_loss / len(loop), {"accuracy": avg_acc / len(loop), "mIoU": avg_mIoU / len(loop)}
