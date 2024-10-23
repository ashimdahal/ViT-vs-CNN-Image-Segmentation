#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings("ignore")
 
import numpy as np
import os, random, math
import matplotlib.pyplot as plt
from dataclasses import dataclass
import json
from tqdm.auto import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torchsummary import summary

import albumentations as A
from albumentations.pytorch import ToTensorV2

from data_utils import Validate, CONSTS, UnNormalize, IsaidDataset, load_metadata, to_device
from models import unet_model 
from loss import CombinedLoss
# https://www.kaggle.com/code/vikram12301/multiclass-semantic-segmentation-pytorch/notebook


@dataclass
class TrainingConfig:
    num_epochs = 40 
    LEARNING_RATE = 1e-3
    scaler = torch.cuda.amp.GradScaler()
    accumulation_steps = 4
    training_batch_size = 32

config = TrainingConfig()


def plot_random_selections(training_dataset, unorm):
    random_images = random.sample(range(1,len(training_dataset)), 12)

    fig = plt.figure(constrained_layout=True, figsize=(10,15))
    subfigs = fig.subfigures(6,2)

    for i, subfig in zip(random_images, subfigs.flat):
        axs = subfig.subplots(1,2)
        image, mask = training_dataset[i].transformed_image, training_dataset[i].transformed_segmentation_map
        axs[0].imshow(unorm(image).permute(1,2,0))
        axs[1].imshow(unorm(image).permute(1,2,0))
        axs[1].imshow(mask, cmap="nipy_spectral", alpha=0.8)
        axs[0].set_title("Input Image")
        axs[1].set_title("Segmentation Map ")
        
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[1].set_xticks([])
        axs[1].set_yticks([])

def train(num_epochs, model, optimizer, loss_fn, train_dataloader, validation_dataloader, scaler, accumulation_steps):
    hist = []
    train_loss = []
    for epoch in range(num_epochs):
        model.train()
        # global progress_bar
        progress_bar = tqdm(
                total=len(train_dataloader),
                # position=0,
                # leave=True
            )
        progress_bar.set_description(f"E {epoch}")
        batch_loss = []
        for batch_idx, batch in enumerate(train_dataloader):
            # data , targets = batch
            data = to_device(batch['pixel_values'], CONSTS.DEVICE)
            targets = to_device(batch['pixel_mask'], CONSTS.DEVICE)
            targets = targets.type(torch.long)
    
            # forward
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                predictions = model(data)
                loss = loss_fn(predictions, targets)
                loss = loss / accumulation_steps
                
            scaler.scale(loss).backward()
                    
            # backward
            if  (batch_idx+1) % accumulation_steps == 0:
                # in order to clip grad normalization one has to also unscale the gradients
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 3)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
    
            with torch.no_grad():
                progress_bar.update(1)
                loss = accumulation_steps * loss.item()
                batch_loss.append(loss)
                logs = {"loss": loss}
                progress_bar.set_postfix(**logs)

        train_loss.append(sum(batch_loss) / len(batch_loss))
        
        progress_bar.set_postfix({"t loss":train_loss[epoch],
                                  "validating":"..."
                                 })
        torch.cuda.empty_cache() 
        valid_res = Validate.validate_cnn(validation_dataloader, model, loss_fn)
       
        logs = {"train loss": train_loss[epoch], **valid_res}
        hist.append(logs)
        progress_bar.set_postfix(logs)
        
    return hist

def collate_fn(batch) -> dict:
    original_images = [sample.original_image for sample in batch]
    transformed_images = torch.stack([sample.transformed_image for sample in batch])
    transformed_segmentation_maps = torch.stack([
        sample.transformed_segmentation_map for sample in batch
    ])

    preprocessed_batch = {
        "pixel_values": transformed_images,
        "pixel_mask" : transformed_segmentation_maps,
        "original_images" :original_images,

    }
    return preprocessed_batch

def load_model() -> list:
    model = unet_model().to(CONSTS.DEVICE)
    print(summary(model, ( 3, 512, 512), device=str(CONSTS.DEVICE)))
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)


    return model,optimizer

def load_datasets():

    training_metadata = load_metadata(f"{CONSTS.DS_DIR}/Train/Annotations/iSAID_train.json")
    validation_metadata = load_metadata(f"{CONSTS.DS_DIR}/Validation/Annotations/iSAID_val.json")

    training_dataset = IsaidDataset(training_metadata, f"{CONSTS.DS_DIR}/Train", transforms=CONSTS.transforms)
    validation_dataset = IsaidDataset(validation_metadata, f"{CONSTS.DS_DIR}/Validation", transforms=CONSTS.transforms)

    return training_dataset, validation_dataset

def prepare_data(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_fn
    )

def main():
    # unorm = UnNormalize(
    #     mean = (0.485, 0.456, 0.406),
    #     std = (0.229, 0.224, 0.225)
    # )

    training_dataset, validation_dataset = load_datasets()

    train_dataloader=prepare_data(training_dataset, config.training_batch_size)
    validation_dataloader = prepare_data(validation_dataset, config.training_batch_size *2)

    loss_fn = CombinedLoss()

    model, optimizer = load_model()

    hist = train(
        config.num_epochs, 
        model,
        optimizer,
        loss_fn,
        train_dataloader,
        validation_dataloader,
        config.scaler,
        config.accumulation_steps
    )

    torch.save({"hist":hist, "state_dict": model.state_dict()}, "model_cnn_v2.pt")

if __name__ == "__main__":
    main()
