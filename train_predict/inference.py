import torch
from tqdm import tqdm
import numpy as np


def inference_fn(test_loader, model, device):

    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            y_preds = model(inputs)
        if y_preds.size(dim=-1) == 5:
            y_preds = torch.argmax(y_preds, dim=-1)
        preds.append(y_preds.sigmoid().to('cpu').numpy())
    predictions = np.concatenate(preds)

    return predictions