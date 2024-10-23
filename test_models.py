import os, random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    MaskFormerImageProcessor,
    AutoImageProcessor,
    MaskFormerForInstanceSegmentation,
)

import matplotlib.pyplot as plt

from data_utils import IsaidDataset, UnNormalize, CONSTS, Validate, load_metadata, to_device
from models import unet_model, encoding_block


def load_cnn():
    model = unet_model().to(CONSTS.DEVICE)
    return to_device(model, CONSTS.DEVICE)

def load_transformer():
    model = MaskFormerForInstanceSegmentation.from_pretrained(
        "facebook/maskformer-swin-base-ade", id2label=CONSTS.id2label, ignore_mismatched_sizes=True
    )
    processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-coco")
    return to_device(model, CONSTS.DEVICE), processor

def load_dataset():
    # both are essentially same but iSAIDTransformer has better compute;
    metadata = load_metadata(f"./{CONSTS.DS_DIR}/Validation/Annotations/iSAID_val.json")
    dataset = IsaidDataset(metadata, f"{CONSTS.DS_DIR}/Validation/", transforms=CONSTS.transforms)
    return dataset

@torch.inference_mode()
@torch.no_grad()
def make_predictions(model, batch, name, processor=None):

    if name=="UNet CNN":
        input_image = batch['pixel_values'].to(CONSTS.DEVICE)
        prediction_logits = model(input_image)

        max_probs = F.softmax(prediction_logits, dim=1)
        max_probs, predicted_classes = torch.max(max_probs, dim=1)

        predicted_classes[max_probs < 0.4 ] = -1
        predicted_classes[predicted_classes == 15] = -1
        return predicted_classes.cpu().numpy()

    elif name=="MaskFormer ViT":
        assert processor is not None
        data = batch
        outputs = model(
                    pixel_values=data["pixel_values"].to(CONSTS.DEVICE),
                    mask_labels=to_device(data["mask_labels"], CONSTS.DEVICE),
                    class_labels=to_device(data["class_labels"], CONSTS.DEVICE),
                )
        output_sizes = [(512, 512)] * outputs['masks_queries_logits'].size(0)
        predicted_semantic_map = processor.post_process_panoptic_segmentation(
            outputs, target_sizes=output_sizes
        )
        target = to_device(batch["pixel_mask"], CONSTS.DEVICE)
        
        semantic_maps = torch.stack([segmentation['segmentation'].to(CONSTS.DEVICE) for segmentation in predicted_semantic_map])
        semantic_maps[semantic_maps==15] = -1
        return semantic_maps.cpu().numpy()


def plot_sampled_images(sample, cnn_segmentation_map, transformer_segmentation_map, num_samples=4):
    # Unpack the sample dictionary
    pixel_values = sample['pixel_values'].to("cpu") # The augmented images (batch of 4).
    pixel_mask = sample['augmented_pixel_mask'].to("cpu") # Original segmentation map (batch of 4)
    original_image = sample['original_images']  # Original images (batch of 4)
    pixel_mask[pixel_mask==15] = -1

    # Initialize a figure with 5 columns (original, augmented, pixel_mask, CNN prediction, Transformer prediction)

    unorm = UnNormalize(
        mean = (0.485, 0.456, 0.406),
        std = (0.229, 0.224, 0.225)
    )

    fig, axs = plt.subplots(num_samples, 5, figsize=(20, num_samples * 4))

    for i in range(num_samples):
        # Plot the original image (column 1)
        axs[i, 0].imshow(original_image[i].permute(1, 2, 0))  # Convert from tensor to numpy array and HWC format
        axs[i, 0].axis('off')
        if i == 0:
            axs[i, 0].set_title("Original Image")
            axs[i, 1].set_title("Augmented Image")
            axs[i, 2].set_title("Ground Truth")
            axs[i, 3].set_title("CNN prediction")
            axs[i, 4].set_title("Transformer Prediction")

        # Plot the augmented image (pixel_values, column 2)
        axs[i, 1].imshow(unorm(pixel_values[i]).permute(1, 2, 0).cpu().numpy())  # Convert from tensor to numpy array and HWC format
        axs[i, 1].axis('off')

        # Plot the original segmentation map (pixel_mask, column 3)
        axs[i, 2].imshow(pixel_mask[i], cmap='nipy_spectral')
        axs[i, 2].axis('off')

        # Plot the CNN model's predicted segmentation map (column 4)
        axs[i, 3].imshow(cnn_segmentation_map[i], cmap='nipy_spectral')
        axs[i, 3].axis('off')

        # Plot the Transformer model's predicted segmentation map (column 5)
        axs[i, 4].imshow(transformer_segmentation_map[i], cmap='nipy_spectral')
        axs[i, 4].axis('off')

    # Adjust layout
    plt.tight_layout()
    plt.show()

def plot_metric_graphs(name, *args):

    epochs = range(len(args[0]))
    available_graphs_order = ["Validation IoU", "Validation Dice", "Validation Accuracy"]
    markers = ["o","x","s"]

    # Apply a specific style
    plt.style.use("fivethirtyeight")  

    # Create figure
    plt.figure(figsize=(10, 6))

    for i, item in enumerate(args):
        # Plot IoU, Dice, and Accuracy with labels and styles
        plt.plot(epochs, item, label=available_graphs_order[i], marker=markers[i])

    # Add title and labels
    plt.title(f'{name}\'s Performance over Epochs', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Metric Values', fontsize=14)

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.6)

    # Set x-axis to show integers (epochs)
    plt.xticks(ticks=epochs, labels=epochs)

    # Add legend
    plt.legend(fontsize=12)

    # Show plot
    plt.tight_layout()
    plt.show()

def prepare_dataset(dataset):

    preprocessor = MaskFormerImageProcessor(
        do_reduce_labels=False,
        do_resize=False,
        do_rescale=False,
        do_normalize=False,
        ignore_index=15
    )

    def collate_fn(batch) -> dict:
        original_images = [sample.original_image for sample in batch]
        transformed_images = torch.stack([sample.transformed_image for sample in batch])
        transformed_segmentation_maps = torch.stack([
            sample.transformed_segmentation_map for sample in batch
        ])

        preprocessed_batch = preprocessor(
            transformed_images,
            segmentation_maps=transformed_segmentation_maps,
            return_tensors="pt",
        )

        preprocessed_batch["original_images"] = original_images
        preprocessed_batch['augmented_pixel_mask'] = transformed_segmentation_maps 
        return preprocessed_batch

    return DataLoader(dataset, batch_size=4, shuffle=True, pin_memory=True, collate_fn=collate_fn)


def test_models(snapshot_unet, snapshot_maskformer, unet, maskformer, processor, name_unet, name_maskformer, dataset):
    snapshot_unet = torch.load(snapshot_unet)
    snapshot_maskformer = torch.load(snapshot_maskformer)

    unet.load_state_dict(snapshot_unet['state_dict'])
    hist_unet = snapshot_unet['hist']
    unet.eval()

    maskformer.load_state_dict(snapshot_maskformer['state_dict'])
    hist_mask = snapshot_maskformer['hist']
    maskformer.eval()

    valid_dataloader = prepare_dataset(dataset)

    # print(Validate.validate_cnn(valid_dataloader, model))

    # Extract history values
    train_loss_cnn, valid_loss_cnn, valid_iou_cnn, valid_dice_cnn, valid_acc_cnn = [], [], [], [], []
    train_loss_tr, valid_loss_tr, valid_iou_tr, valid_dice_tr, valid_acc_tr= [], [], [], [], []
    for value in hist_unet:
        train_loss_cnn.append(value['train loss'])
        valid_loss_cnn.append(value['v loss'])
        valid_iou_cnn.append(value['v IoU'])
        valid_dice_cnn.append(value['v Dice'])
        valid_acc_cnn.append(value['v Acc'])

    for value in hist_mask:
        train_loss_tr.append(value['train loss'])
        valid_loss_tr.append(value['v loss'])
        valid_iou_tr.append(value['v IoU'])
        valid_dice_tr.append(value['v Dice'])
        valid_acc_tr.append(value['v Acc'])

    # plot_metric_graphs(name, valid_iou, valid_dice, valid_acc)
    sample_batch = next(iter(valid_dataloader))

    cnn_predictions = make_predictions(unet, sample_batch, name_unet)
    mask_predictions = make_predictions(maskformer, sample_batch, name_maskformer, processor)

    plot_sampled_images(sample_batch, cnn_predictions, mask_predictions)

def main():
    unet = load_cnn()
    dataset = load_dataset()
    # test_model("mode_cnn_v2.pt", unet, "UNet CNN", dataset)

    maskformer, processor = load_transformer()

    test_models("model_cnn_v2.pt", "model.pt", unet, maskformer, processor, "UNet CNN", "MaskFormer ViT", dataset)

if __name__ == "__main__":
    main()

