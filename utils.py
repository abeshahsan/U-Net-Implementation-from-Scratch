import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2

def save_checkpoint(state, filename):
    print("=> Saving checkpoint")
    torch.save(state, filename) # model.state_dict() + optimizer.state_dict() + epoch
    
def load_checkpoint(checkpoint, model, optimizer=None):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["epoch"]


def get_transforms(image_height, image_width):
    return A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

def get_loaders(
        train_ds,
        val_ds,
        batch_size,
        num_workers=4,
        pin_memory=True,
):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader


def pixel_accuracy(preds, target):
    correct = (preds == target).float()
    acc = correct.sum() / (correct.numel() + 1e-6)
    return acc

def mIoU(preds, target, num_classes):
    iou = 0.0
    for cls in range(num_classes):
        intersection = (preds == cls) & (target == cls)
        union = (preds == cls) | (target == cls)
        iou += intersection.sum() / (union.sum() + 1e-6)
        m_iou = iou / float(num_classes)
    return m_iou

def get_scores(predictions, targets):
    num_classes = predictions.shape[1]
    predictions = torch.nn.Softmax(dim=1)(predictions)
    predictions = torch.argmax(predictions, dim=1)
    scores = {
        "pixel_acc": pixel_accuracy(predictions, targets),
        "mIoU": mIoU(predictions, targets, num_classes)
    }
    return scores




# def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
#     model.eval()
#     for idx, (x, y) in enumerate(loader):
#         x = x.to(device=device)
#         with torch.no_grad():
#             preds = torch.sigmoid(model(x))
#             preds = (preds > 0.5).float()
#         torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
#         torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

