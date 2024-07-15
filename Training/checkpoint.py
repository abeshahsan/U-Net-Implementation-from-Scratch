import torch

def save_checkpoint(state, filename):
    print("=> Saving checkpoint")
    torch.save(state, filename) # model.state_dict() + optimizer.state_dict() + epoch
    
def load_checkpoint(ckpt_file, model, optimizer=None):
    checkpoint = torch.load(ckpt_file)
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["epoch"]
