import wandb
import numpy as np
import gc
import torch
import torch.nn as nn
import time
import os

from torch.utils.data import DataLoader
from gen_data.dataset import TrainDataset
from model.model import CustomModel
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from utils import get_score
from utils import AverageMeter, timeSince
from tqdm import tqdm


def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler,
             device, cfg):

    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.apex)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0

    for step, (inputs, labels) in tqdm(enumerate(train_loader)):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=cfg.apex):
            y_preds = model(inputs)
        loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
        if cfg.gradient_accumulation_steps > 1:
            loss = loss / cfg.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if cfg.batch_scheduler:
                scheduler.step()
        end = time.time()
        if step % cfg.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.8f}  '
                  .format(epoch+1, step, len(train_loader), 
                          remain=timeSince(start, float(step+1)/len(train_loader)),
                          loss=losses,
                          grad_norm=grad_norm,
                          lr=scheduler.get_lr()[0]))
        if cfg.with_wandb:
            wandb.log({f"[fold{fold}] loss": losses.val,
                       f"[fold{fold}] lr": scheduler.get_lr()[0]})
    return losses.avg


def valid_fn(valid_loader, model, criterion, device, cfg):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, (inputs, labels) in enumerate(valid_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds = model(inputs)
        loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
        if cfg.gradient_accumulation_steps > 1:
            loss = loss / cfg.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.sigmoid().to('cpu').numpy())
        end = time.time()
        if step % cfg.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=timeSince(start, float(step+1)/len(valid_loader))))
    predictions = np.concatenate(preds)
    predictions = np.concatenate(predictions)
    return losses.avg, predictions


def train_loop(folds, fold,
               cfg,
               ):

    cfg.logger.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
    valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)
    valid_labels = valid_folds['score'].values

    train_dataset = TrainDataset(cfg, train_folds)
    valid_dataset = TrainDataset(cfg, valid_folds)

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel(cfg, config_path=None, pretrained=True)

    torch.save(model.config, os.path.join(cfg.dir_output, 'model.config'))
    model.to(cfg.device)

    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "model" not in n],
             'lr': decoder_lr, 'weight_decay': 0.0}
        ]
        return optimizer_parameters

    optimizer_parameters = get_optimizer_params(model,
                                                encoder_lr=cfg.encoder_lr,
                                                decoder_lr=cfg.decoder_lr,
                                                weight_decay=cfg.weight_decay)
    optimizer = AdamW(optimizer_parameters, lr=cfg.encoder_lr, eps=cfg.eps, betas=cfg.betas)

    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
            )
        elif cfg.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps,
                num_cycles=cfg.num_cycles
            )
        return scheduler

    num_train_steps = int(len(train_folds) / cfg.batch_size * cfg.epochs)
    scheduler = get_scheduler(cfg, optimizer, num_train_steps)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    best_score = 0.

    dir_model = os.path.join(cfg.dir_output, 'model',)
    if not os.path.exists(dir_model):
        os.makedirs(dir_model)
    path_model = os.path.join(dir_model, f"fold_{fold}_best.model")

    for epoch in range(cfg.epochs):

        start_time = time.time()

        # train_predict
        avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, cfg.device, cfg)

        # eval
        avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, cfg.device, cfg)

        # scoring
        score = get_score(valid_labels, predictions)

        elapsed = time.time() - start_time

        cfg.logger.info(
            f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        cfg.logger.info(f'Epoch {epoch + 1} - Score: {score:.4f}')
        if cfg.with_wandb:
            wandb.log({f"[fold{fold}] epoch": epoch + 1,
                       f"[fold{fold}] avg_train_loss": avg_loss,
                       f"[fold{fold}] avg_val_loss": avg_val_loss,
                       f"[fold{fold}] score": score})
        if (best_score < score) or (epoch == 0):
            best_score = score
            cfg.logger.info(f'Epoch {epoch + 1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'predictions': predictions},
                       path_model)

    valid_folds['pred'] = predictions

    torch.cuda.empty_cache()
    gc.collect()

    return valid_folds
 
def criterion_corrcoef(predicts, labels):
    """
        Use the Pearson correlation coefficient as the loss.
        predicts : 1D list of numerics
        labels : 1D list of numerics
        (the function of torch.corrcoef() needs to be newer version of pytorch)
    """
    x = torch.tensor([predicts, labels])
    corrcoef = torch.corrcoef(x)[0, 1]  # the [0,1] element is the correlation coefficeint in the matrix
    return 1 - corrcoef

 def criterion_CCC(predicts, labels):
  """
      Use the concordance correlation coefficient as the loss.
      (reference: 
        1. Evaluation of Error and Correlation-Based Loss Functions For Multitask Learning Dimensional \
           Speech Emotion Recognition
        2. https://discuss.pytorch.org/t/use-pearson-correlation-coefficient-as-cost-function/8739/6 \
        3. https://discuss.pytorch.org/t/lins-concordance-correlation-coefficient-as-loss-function/24334
      )
      predicts : 1D list of numerics
      labels : 1D list of numerics
      (the function of torch.corrcoef() needs to be newer version of pytorch)
  """ 
    predicts_mean = torch.mean(predicts)
    labels_mean = torch.mean(labels)
    predicts_var = torch.var(predicts)
    labels_var = torch.var(labels)
    predicts_std = predicts_var.pow(0.5)
    labels_std = labels_var.pow(0.5)

    vx = y_true - torch.mean(y_true)
    vy = y_hat - torch.mean(y_hat)
    pcc = torch.corrcoef(x)[0, 1] 

    ccc = (2 * pcc * predicts_std * labels_std) / \
          (predicts_var + labels_var + (predicts_mean - labels_mean) ** 2)
    return 1 - ccc
  
  
