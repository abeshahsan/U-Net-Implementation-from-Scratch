from torch.utils.data import DataLoader

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