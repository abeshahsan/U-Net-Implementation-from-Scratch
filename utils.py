import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
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

def mIoU(preds, target, num_classes=22):
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

# def get_scores(loader, model, device="cuda"):
#     model.eval()
#     pixel_accs = []
#     mious = []
#     for x, y in loader:
#         x = x.to(device)
#         y = y.to(device).squeeze(1)
#         with torch.no_grad():
#             preds = model(x)
#             softmax = torch.nn.Softmax(dim=1)
#             preds = torch.argmax(softmax(model(x)), dim=1)
#         pixel_accs.append(pixel_accuracy(preds, y))
#         mious.append(mIoU(preds, y))
#     model.train()
#     scores = {
#         "pixel_acc": torch.mean(pixel_accs),
#         "mIoU": torch.mean(mious)
#     }
#     return scores


# def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
#     model.eval()
#     for idx, (x, y) in enumerate(loader):
#         x = x.to(device=device)
#         with torch.no_grad():
#             preds = torch.sigmoid(model(x))
#             preds = (preds > 0.5).float()
#         torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
#         torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

#     model.train()


# Mapping of ignore categories and valid ones (numbered from 1-19)
cityscapes_labels_map = {
    0: 0,    # Unlabeled
    1: 0,    # Ego vehicle
    2: 0,    # Rectification border
    3: 0,    # Out of ROI
    4: 0,    # Static
    5: 0,    # Dynamic
    6: 0,    # Ground
    7: 1,    # Road
    8: 2,    # Sidewalk
    9: 0,    # Parking
    10: 0,   # Rail track
    11: 3,   # Building
    12: 4,   # Wall
    13: 5,   # Fence
    14: 0,   # Guard rail
    15: 0,   # Bridge
    16: 0,   # Tunnel
    17: 6,   # Pole
    18: 0,   # Polegroup
    19: 7,   # Traffic light
    20: 8,   # Traffic sign
    21: 9,   # Vegetation
    22: 10,  # Terrain
    23: 11,  # Sky
    24: 12,  # Person
    25: 13,  # Rider
    26: 14,  # Car
    27: 15,  # Truck
    28: 16,  # Bus
    29: 17,  # Caravan
    30: 18,  # Trailer
    31: 19,  # Train
    32: 20,  # Motorcycle
    33: 21   # Bicycle
    # Add any additional mappings if necessary
}


def convert_cityscapes_30_to_20(image):
    mapped_image = np.zeros_like(image, dtype=np.uint8)
    for k, v in cityscapes_labels_map.items():
        mapped_image[image == k] = v
    return mapped_image