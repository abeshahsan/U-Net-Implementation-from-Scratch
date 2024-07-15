from tqdm import tqdm
import torch
from Training.scores import get_scores


def train_fn(epoch, loader, model, optimizer, loss_fn, scaler, device):
    
    loop = tqdm(loader, desc=f"Epoch {epoch}: ", unit="batch")

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

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()

        
        # update tqdm loop
        loop.set_postfix({
            "Loss": loss.item(), 
            "Pixel Accuracy": acc.item(), 
            "mIoU": iou.item()}
            )

        torch.cuda.empty_cache()
    return predictions, mask, loss 