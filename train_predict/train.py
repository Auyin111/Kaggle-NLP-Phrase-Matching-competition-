import wandb
import numpy as np
import pandas as pd
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import shutil
from multiprocessing import Process  # For displaying learning rate plot without blocking

from gen_data.dataset import TrainDataset
from model.model import CustomModel
from train_predict.loss_functions import PCCLoss, CCCLoss1, CCCLoss2

from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, AdamW
from torch.optim.swa_utils import AveragedModel, SWALR  ## New ##
from torch.optim.lr_scheduler import CosineAnnealingLR  ## New ##

from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, DataCollatorWithPadding
from utils import get_score, get_distribution, plot_lr_fn
from utils import AverageMeter, timeSince, highlight_string
from tqdm import tqdm


def train_fn(fold, train_loader, model, swa_model, criterion, optimizer, epoch, scheduler, swa_scheduler, device, cfg):  # Modified to support dynamic padding, batch sampler, swa
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.apex)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    lrs = []

    for step, inputs in enumerate(train_loader):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = inputs["labels"]
        inputs.pop("labels")
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=cfg.apex):
            y_preds = model(inputs)
        if cfg.loss_fn == "MSE":
            labels = labels.to(torch.half)
        loss = criterion(y_preds.view(-1, cfg.target_size), labels.view(-1, cfg.target_size))
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

            ### New ###

            if cfg.batch_scheduler:
                if epoch + 1 >= cfg.swa_start and cfg.use_swa:
                    if swa_model == None:
                        swa_model = AveragedModel(model)
                    else:
                        swa_model.update_parameters(model)
                        swa_scheduler.step()

                else:
                    scheduler.step()

                # if cfg.use_swa and epoch + 1 == cfg.epochs and step == len(train_loader) - 1:
                #    torch.optim.swa_utils.update_bn(train_loader, swa_model)
                #    print("Batch norm for swa updated")

            ### New ###

        end = time.time()

        if step % cfg.print_freq == 0 or step == (len(train_loader) - 1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss(value, avg): {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.8f}  '
                  .format(epoch + 1, step, len(train_loader),
                          remain=timeSince(start, float(step + 1) / len(train_loader)),
                          loss=losses,
                          grad_norm=grad_norm,
                          lr=scheduler.get_lr()[0]))

        if cfg.plot_lr:
            lrs.append(optimizer.param_groups[0]["lr"])

        if cfg.with_wandb:
            wandb.log({f"[fold{fold}] lr": scheduler.get_lr()[0],
                       f"[fold{fold}] grad_norm": grad_norm,
                       f'[fold{fold}] current_loss': losses.val,
                       f'[fold{fold}] current_avg_loss': losses.avg
                       })

    return lrs


def valid_fn(data_loader, model, swa_model, criterion, epoch, device, cfg, dataset):  # Modified to support swa
    model.eval()
    losses = AverageMeter()
    preds = []
    start = end = time.time()

    for step, inputs in enumerate(data_loader):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = inputs["labels"]
        inputs.pop("labels")
        batch_size = labels.size(0)
        with torch.no_grad():
            if epoch + 1 >= cfg.swa_start and cfg.use_swa and swa_model != None:
                y_preds = swa_model(inputs)
            else:
                y_preds = model(inputs)
        loss = criterion(y_preds.view(-1, cfg.target_size), labels.view(-1, cfg.target_size))
        if cfg.gradient_accumulation_steps > 1:
            loss = loss / cfg.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        if cfg.loss_fn == "BCE" or cfg.loss_fn == "BCEWithLogits":
            y_preds = y_preds.sigmoid()
        elif cfg.loss_fn == "CE":
            y_preds = torch.argmax(y_preds, dim=-1)
        preds.append(y_preds.to('cpu').numpy().tolist())
        end = time.time()
        if step % cfg.print_freq == 0 or step == (len(data_loader) - 1):
            print(f'EVAL Dataset: {dataset} [{step}/{len(data_loader)}],'
                  f' Elapsed: {timeSince(start, float(step + 1) / len(data_loader))}')

    predictions = np.concatenate(preds)
    if predictions.ndim != 1:
        predictions = np.concatenate(predictions)

    return losses.val, losses.avg, predictions


def train_loop(folds, fold, cfg):

    cfg.logger.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader (Modified to support dynamic padding, batch sampler, swa
    # ====================================================

    train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
    valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)
    train_labels = train_folds['score']
    valid_labels = valid_folds['score']
    if cfg.target_size == 5:
        # Decode one-hot labels to index (python int dtype)
        train_labels = train_labels.apply(lambda x: torch.argmax(x).item())
        valid_labels = valid_labels.apply(lambda x: torch.argmax(x).item())
    train_labels = train_labels.values
    valid_labels = valid_labels.values

    train_dataset = TrainDataset(cfg, train_folds)
    valid_dataset = TrainDataset(cfg, valid_folds)

    collator = DataCollatorWithPadding(cfg.tokenizer) if cfg.dynamic_padding else None

    if cfg.batch_distribution == "label" or cfg.batch_distribution == "context":
        distribution, weights = get_distribution(train_folds, cfg)
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=False)  # generator=cfg.generator)
    else:
        sampler = None

    cfg.logger.info(f"Batch sampler: {cfg.batch_distribution}")
    if cfg.batch_distribution == "label" or cfg.batch_distribution == "context":
        cfg.logger.info(f"Distribution: {distribution}")

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=False,
                              sampler=sampler,
                              num_workers=cfg.num_workers,
                              collate_fn=collator,
                              pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=False,
                              num_workers=cfg.num_workers,
                              collate_fn=collator,
                              pin_memory=True, drop_last=False)

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel(cfg, config_path=None, pretrained=True)
    model.to(cfg.device)

    if fold == 0:  # Copy the essential files for model loading in prediction
        torch.save(model.config, os.path.join(cfg.dir_output, 'model.config'))
        shutil.copyfile('cfg.py', os.path.join(cfg.dir_output, 'cfg.py'))
        if os.path.isfile('model/model.py'):
            shutil.copyfile('model/model.py', os.path.join(cfg.dir_output, 'model.py'))
        else:
            shutil.copyfile('model\model.py', os.path.join(cfg.dir_output, 'model.py'))

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
    # scheduler (includes CosineAnnealingLR)
    # ====================================================

    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
            )
        elif cfg.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=cfg.num_cycles
            )
        elif cfg.scheduler == "cosine_annealing":  ### New ###
            scheduler = CosineAnnealingLR(optimizer, T_max=num_train_steps)
        return scheduler

    num_train_steps = int(len(train_folds) / cfg.batch_size * cfg.epochs)
    scheduler = get_scheduler(cfg, optimizer, num_train_steps)

    # ====================================================
    # swa initialization (new)
    # ====================================================

    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, anneal_epochs=cfg.anneal_steps, swa_lr=cfg.swa_lr) if cfg.use_swa else None

    # ====================================================
    # loop
    # ====================================================

    # Support multiple loss function in cfg

    if cfg.loss_fn == "MSE":
        criterion = nn.MSELoss(reduction="mean")  # nn.BCEWithLogitsLoss(reduction="mean"), nn.MSELoss(reduction="mean")
    elif cfg.loss_fn == "BCE":
        criterion = nn.BCELoss(reduction="mean")
    elif cfg.loss_fn == "BCEWithLogits":
        criterion = nn.BCEWithLogitsLoss(reduction="mean")
    elif cfg.loss_fn == "CCC1":
        criterion = CCCLoss1()
    elif cfg.loss_fn == "CCC2":
        criterion = CCCLoss2()
    elif cfg.loss_fn == "PCC":
        criterion = PCCLoss()
    elif cfg.loss_fn == "CE":
        criterion = nn.CrossEntropyLoss(reduction="mean")
    else:
        raise Exception("Unknown loss function, please check your spelling.")

    best_score = - 100
    lrs_list = []  # For plotting learning rates

    dir_model = os.path.join(cfg.dir_output, 'model', )
    if not os.path.exists(dir_model):
        os.makedirs(dir_model)
    path_model = os.path.join(dir_model, f"fold_{fold}_best.model")

    es_patience_count = 0
    for epoch in range(cfg.epochs):

        start_time = time.time()

        # train_predict
        lrs = train_fn(fold, train_loader, model, swa_model, criterion, optimizer, epoch, scheduler, swa_scheduler, cfg.device, cfg)

        # eval training and validation set
        train_loss, avg_train_loss, train_predictions = valid_fn(train_loader, model, swa_model, criterion, epoch,
                                                                 cfg.device, cfg, 'train')
        val_loss, avg_val_loss, valid_predictions = valid_fn(valid_loader, model, swa_model, criterion, epoch,
                                                               cfg.device, cfg, 'valid')

        train_score = get_score(train_labels, train_predictions)
        valid_score = get_score(valid_labels, valid_predictions)
        lrs_list += lrs

        elapsed = time.time() - start_time

        cfg.logger.info(
            f'Epoch {epoch + 1}, time: {elapsed:.0f}s '
            f'train_loss: {train_loss:.4f}, avg_train_loss: {avg_train_loss:.4f}, train_score: {train_score:.4f}')
        cfg.logger.info(f'val_loss: {val_loss:.4f}, avg_val_loss: {avg_val_loss:.4f}, valid_score: {valid_score:.4f}')

        if cfg.with_wandb:
            wandb.log({f"[fold{fold}] epoch": epoch + 1,
                       f"[fold{fold}] train_loss": train_loss,
                       f"[fold{fold}] avg_train_loss": avg_train_loss,
                       f"[fold{fold}] train_score": train_score,
                       f"[fold{fold}] val_loss": val_loss,
                       f"[fold{fold}] avg_val_loss": avg_val_loss,
                       f"[fold{fold}] valid_score": valid_score})

        # Handle model saving behavior with swa and early stopping

        if cfg.use_swa:
            if epoch == cfg.epochs:
                torch.save({'model': swa_model.state_dict(),
                            'predictions': valid_predictions},
                           path_model)
                cfg.logger.info("SWA model saved")
            else:
                cfg.logger.info("SWA enabled, model would only be saved on the last epoch")
        else:
            if cfg.early_stopping:
                if (best_score < valid_score) or (epoch == 0):

                    cfg.logger.info(highlight_string(
                        f'in epoch {epoch}, the validation score was improved from {best_score} to {valid_score}'))

                    best_score = valid_score
                    cfg.logger.info(f'Epoch {epoch + 1} - Save Best Score: {best_score:.4f} Model')
                    torch.save({'model': model.state_dict(),
                                'predictions': valid_predictions},
                               path_model)
                    # reset es_patience_count
                    es_patience_count = 0

                else:
                    es_patience_count += 1
                    cfg.logger.info(highlight_string(f'in epoch {epoch}, no any improvement '
                                    f'(es_patience_count: {es_patience_count}/{cfg.es_patience})'))

                    if es_patience_count == cfg.es_patience:
                        cfg.logger.info(highlight_string(
                            'the training process is stopped by early stopping setup', '!'))
                        print()
                        break

            else:
                torch.save({'model': model.state_dict(),
                            'predictions': valid_predictions},
                           path_model)
                cfg.logger.info(f'Model updated (Early stopping disabled)')

    # Plot learning rate in parallel process

    if cfg.plot_lr:
        parallel_plot = Process(target=plot_lr_fn, args=(lrs_list,))
        parallel_plot.start()

    valid_folds['pred'] = valid_predictions

    torch.cuda.empty_cache()
    gc.collect()

    return valid_folds
