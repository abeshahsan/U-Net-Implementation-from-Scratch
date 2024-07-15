import torch


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