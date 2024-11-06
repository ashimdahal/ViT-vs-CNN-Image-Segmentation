import os, random
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

from transformers import (
    MaskFormerImageProcessor,
    AutoImageProcessor,
    MaskFormerForInstanceSegmentation,
)

import matplotlib.pyplot as plt
import seaborn as sns

from data_utils import IsaidDataset, UnNormalize, CONSTS, Validate, load_metadata, to_device
from models import unet_model, encoding_block

from torchsummary import summary
from fvcore.nn import FlopCountAnalysis

torch.manual_seed(42)

def load_cnn():
    model = unet_model().to(CONSTS.DEVICE)
    return to_device(model, CONSTS.DEVICE)

def load_transformer():
    model = MaskFormerForInstanceSegmentation.from_pretrained(
        "./model_transformer/", id2label=CONSTS.id2label, ignore_mismatched_sizes=True
    )
    # processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-coco")

    preprocessor = MaskFormerImageProcessor(
        do_reduce_labels=False,
        do_resize=False,
        do_rescale=False,
        do_normalize=False,
        # ignore_index=CONSTS.ignore_index
    )

    return to_device(model, CONSTS.DEVICE), preprocessor

def load_dataset():
    # both are essentially same but iSAIDTransformer has better compute;
    metadata = load_metadata(f"./{CONSTS.DS_DIR}/Validation/Annotations/iSAID_val.json")
    dataset = IsaidDataset(metadata, f"{CONSTS.DS_DIR}/Validation/", transforms=CONSTS.transforms)
    return dataset

def mask_to_rgb(mask, mapping=CONSTS.mapping):
    """
    Convert a segmentation mask to an RGB image based on the provided mapping.
    
    Parameters:
    - mask: numpy array of shape (512, 512), representing the segmentation mask.
    - mapping: dictionary where keys are class values and values are RGB tuples.
    
    Returns:
    - RGB image as a numpy array of shape (512, 512, 3).
    """
    # Initialize an empty RGB image with the same height and width as the mask
    rgb_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # Iterate through the mapping and set the RGB values based on the class values in the mask
    for rgb_value, class_value in mapping.items():
        # Apply the RGB value to all pixels in the mask that match the class_value
        rgb_image[mask == class_value] = rgb_value

    return rgb_image

@torch.inference_mode()
@torch.no_grad()
def make_predictions(model, batch, name, processor=None):

    start = time.time()
    if name=="UNet CNN":
        input_image = batch['pixel_values'].to(CONSTS.DEVICE)
        prediction_logits = model(input_image)

        max_probs = F.softmax(prediction_logits, dim=1)
        max_probs, predicted_classes = torch.max(max_probs, dim=1)

        predicted_classes[max_probs < 0.65] = 15
        # predicted_classes[predicted_classes == 15] = -1
        end = time.time()
        print("-"*50)
        print(f"time spend for unet: {end-start}")
        print("-"*50)
        return predicted_classes.cpu().numpy()

    elif name=="MaskFormer ViT":
        assert processor is not None
        data = batch
        outputs = model(
                    pixel_values=data["pixel_values"].to(CONSTS.DEVICE),
                )
        output_sizes = [(512, 512)] * outputs['masks_queries_logits'].size(0)
        predicted_semantic_maps = torch.stack(
                    processor.post_process_semantic_segmentation(
                    outputs, target_sizes=output_sizes
                )
        )
        end = time.time()
        print("-"*50)
        print(f"time spend for maskformer: {end-start}")
        print("-"*50)
        return to_device(predicted_semantic_maps, "cpu")


def plot_sampled_images(sample, cnn_segmentation_map, transformer_segmentation_map, num_samples=6):
    # Unpack the sample dictionary
    pixel_values = sample['pixel_values'].to("cpu") # The augmented images (batch of 4).
    pixel_mask = sample['augmented_pixel_mask'].to("cpu") # Original segmentation map (batch of 4)
    original_image = sample['original_images']  # Original images (batch of 4)
    # pixel_mask[pixel_mask==15] = -1

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
            axs[i, 0].set_title("Original Image", fontsize=14)
            axs[i, 1].set_title("Augmented Image", fontsize=14)
            axs[i, 2].set_title("Ground Truth", fontsize=14)
            axs[i, 3].set_title("CNN prediction", fontsize=14)
            axs[i, 4].set_title("Transformer Prediction", fontsize=14)

        # Plot the augmented image (pixel_values, column 2)
        axs[i, 1].imshow(unorm(pixel_values[i]).permute(1, 2, 0).cpu().numpy())  # Convert from tensor to numpy array and HWC format
        axs[i, 1].axis('off')

        # Plot the original segmentation map (pixel_mask, column 3)
        axs[i, 2].imshow(mask_to_rgb(pixel_mask[i]))
        axs[i, 2].axis('off')

        # Plot the CNN model's predicted segmentation map (column 4)
        axs[i, 3].imshow(mask_to_rgb(cnn_segmentation_map[i]))
        axs[i, 3].axis('off')

        # Plot the Transformer model's predicted segmentation map (column 5)
        axs[i, 4].imshow(mask_to_rgb(transformer_segmentation_map[i]))
        axs[i, 4].axis('off')

    # Adjust layout
    plt.tight_layout()
    plt.savefig("./graphs/model_sample.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_classwise_iou_dice(iou_scores_vit, dice_scores_vit, iou_scores_cnn, dice_scores_cnn):
    """
    Enhanced plotting for per-class mIoU and Dice scores comparison with annotations and styling.
    
    Parameters:
    - iou_scores_vit: Dict[str, float] - Per-class IoU scores for ViT model.
    - dice_scores_vit: Dict[str, float] - Per-class Dice scores for ViT model.
    - iou_scores_cnn: Dict[str, float] - Per-class IoU scores for CNN model.
    - dice_scores_cnn: Dict[str, float] - Per-class Dice scores for CNN model.
    """
    # Extract classes
    classes = list(iou_scores_vit.keys())
    num_classes = len(classes)
    n = int(np.ceil(np.sqrt(num_classes)))  # Grid size n x n

    # Set up color schemes and figure properties
    vit_colors = ['#4c72b0', '#55a868']  # IoU and Dice for ViT
    cnn_colors = ['#c44e52', '#8172b2']  # IoU and Dice for CNN
    
    # Create subplots
    fig, axes = plt.subplots(n, n, figsize=(18, 18), facecolor='#f7f7f7')
    fig.suptitle("Per-Class mIoU and Dice Score Comparison for ViT and CNN Models", fontsize=20, weight='bold', color='#333333')
    fig.subplots_adjust(top=0.92)

    for idx, cls in enumerate(classes):
        row, col = divmod(idx, n)
        ax = axes[row, col]
        bar_width = 0.35
        x = np.arange(2)  # One position for IoU, one for Dice
        
        # Plot bars for ViT
        vit_bars = ax.bar(x - bar_width / 2, [iou_scores_vit[cls], dice_scores_vit[cls]], 
                          width=bar_width, color=vit_colors, edgecolor='gray', label='ViT', alpha=0.85)

        # Plot bars for CNN
        cnn_bars = ax.bar(x + bar_width / 2, [iou_scores_cnn[cls], dice_scores_cnn[cls]], 
                          width=bar_width, color=cnn_colors, edgecolor='gray', label='CNN', alpha=0.85)

        # Add bar annotations for exact values
        for bars, model in [(vit_bars, 'ViT'), (cnn_bars, 'CNN')]:
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f"{yval:.2f}",
                        ha='center', va='bottom', fontsize=9, weight='bold', color='#333333')

        # Set titles and axis labels
        ax.set_title(f"Class: {cls}", fontsize=14, color='#333333', weight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['IoU', 'Dice'], fontsize=12, color='#444444')
        ax.set_ylim(0, 1)
        ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.5)

        # Improve subplot aesthetics
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#dddddd')
        ax.spines['bottom'].set_color('#dddddd')

    # Add legend once in the top right subplot
    handles, labels = vit_bars + cnn_bars, ['ViT IoU', 'ViT Dice', 'CNN IoU', 'CNN Dice']
    fig.legend(handles, labels, loc='upper right', fontsize=12, frameon=False, bbox_to_anchor=(0.9, 0.95))

    # Turn off any empty subplots in the grid
    for idx in range(num_classes, n * n):
        fig.delaxes(axes.flatten()[idx])

    plt.show()



def plot_model_comparison(epochs, valid_iou_cnn, valid_dice_cnn, valid_acc_cnn,
                          valid_iou_tr, valid_dice_tr, valid_acc_tr):
    # Use a high-quality style for scientific publication
    plt.style.use('seaborn-colorblind')

    # Create figure and axis with larger dimensions for print quality
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    # IoU Plot with detailed line and marker styles
    ax.plot(epochs, valid_iou_cnn, label='CNN IoU', color='royalblue', 
            linestyle='-', linewidth=2.5, marker='o', markersize=6)
    ax.plot(epochs, valid_iou_tr, label='Transformer IoU', color='royalblue', 
            linestyle='--', linewidth=2.5, marker='s', markersize=6)

    # Dice Plot with similar improvements
    ax.plot(epochs, valid_dice_cnn, label='CNN Dice', color='seagreen', 
            linestyle='-', linewidth=2.5, marker='D', markersize=6)
    ax.plot(epochs, valid_dice_tr, label='Transformer Dice', color='seagreen', 
            linestyle='--', linewidth=2.5, marker='^', markersize=6)

    # # Accuracy Plot for additional comparison
    # ax.plot(epochs, valid_acc_cnn, label='CNN Accuracy', color='darkorange', 
    #         linestyle='-', linewidth=2.5, marker='x', markersize=6)
    # ax.plot(epochs, valid_acc_tr, label='Transformer Accuracy', color='darkorange', 
    #         linestyle='--', linewidth=2.5, marker='*', markersize=6)
    #
    # Enhancing title and labels with larger font sizes and bolding
    ax.set_title('Model Comparison: CNN vs Transformer Metrics', fontsize=18, weight='bold', pad=15)
    ax.set_xlabel('Epochs', fontsize=14, labelpad=10, weight='bold')
    ax.set_ylabel('Metric Score', fontsize=14, labelpad=10, weight='bold')

    # Highlight the peak values for each metric
    # For CNN IoU
    max_iou_cnn = max(valid_iou_cnn)
    max_iou_cnn_epoch = epochs[valid_iou_cnn.index(max_iou_cnn)]
    ax.annotate(f'Peak CNN IoU: {max_iou_cnn:.2f}', 
                xy=(max_iou_cnn_epoch, max_iou_cnn), 
                xytext=(max_iou_cnn_epoch, max_iou_cnn + 0.05),
                arrowprops=dict(facecolor='royalblue', arrowstyle='->', lw=1.5),
                fontsize=10, weight='bold', color='royalblue')

    # For Transformer IoU
    max_iou_tr = max(valid_iou_tr)
    max_iou_tr_epoch = epochs[valid_iou_tr.index(max_iou_tr)]
    ax.annotate(f'Peak Transformer IoU: {max_iou_tr:.2f}', 
                xy=(max_iou_tr_epoch, max_iou_tr), 
                xytext=(max_iou_tr_epoch, max_iou_tr + 0.05),
                arrowprops=dict(facecolor='royalblue', arrowstyle='->', lw=1.5),
                fontsize=10, weight='bold', color='royalblue')

    # For CNN Dice
    max_dice_cnn = max(valid_dice_cnn)
    max_dice_cnn_epoch = epochs[valid_dice_cnn.index(max_dice_cnn)]
    ax.annotate(f'Peak CNN Dice: {max_dice_cnn:.2f}', 
                xy=(max_dice_cnn_epoch, max_dice_cnn), 
                xytext=(max_dice_cnn_epoch, max_dice_cnn + 0.05),
                arrowprops=dict(facecolor='seagreen', arrowstyle='->', lw=1.5),
                fontsize=10, weight='bold', color='seagreen')

    # For Transformer Dice
    max_dice_tr = max(valid_dice_tr)
    max_dice_tr_epoch = epochs[valid_dice_tr.index(max_dice_tr)]
    ax.annotate(f'Peak Transformer Dice: {max_dice_tr:.2f}', 
                xy=(max_dice_tr_epoch, max_dice_tr), 
                xytext=(max_dice_tr_epoch, max_dice_tr + 0.05),
                arrowprops=dict(facecolor='seagreen', arrowstyle='->', lw=1.5),
                fontsize=10, weight='bold', color='seagreen')

    # Improving grid and axes for clarity
    ax.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.6)
    ax.minorticks_on()  # Show minor ticks
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)

    # Enhanced legend for readability
    ax.legend(loc='best', fontsize=12, frameon=True, fancybox=True, framealpha=0.9, borderpad=1)

    # Finalize layout and save the figure with high quality for printing
    fig.tight_layout()
    fig.savefig("./graphs/model_metrics.png", dpi=300, bbox_inches='tight', format='png')
    plt.show()


def plot_classwise_iou_dice(iou_scores_vit, dice_scores_vit, iou_scores_cnn, dice_scores_cnn):
    """
    Enhanced plotting for per-class mIoU and Dice scores comparison with annotations and styling.
    
    Parameters:
    - iou_scores_vit: Dict[str, float] - Per-class IoU scores for ViT model.
    - dice_scores_vit: Dict[str, float] - Per-class Dice scores for ViT model.
    - iou_scores_cnn: Dict[str, float] - Per-class IoU scores for CNN model.
    - dice_scores_cnn: Dict[str, float] - Per-class Dice scores for CNN model.
    """
    # Extract classes
    classes = list(iou_scores_vit.keys())
    num_classes = len(classes)
    n = int(np.ceil(np.sqrt(num_classes)))  # Grid size n x n

    # Set up color schemes and figure properties
    vit_colors = ['#4c72b0', '#55a868']  # IoU and Dice for ViT
    cnn_colors = ['#c44e52', '#8172b2']  # IoU and Dice for CNN
    
    # Create subplots
    fig, axes = plt.subplots(n, n, figsize=(18, 18), facecolor='#f7f7f7')
    fig.suptitle("Per-Class mIoU and Dice Score Comparison for ViT and CNN Models", fontsize=20, weight='bold', color='#333333')
    fig.subplots_adjust(top=0.92)

    for idx, cls in enumerate(classes):
        row, col = divmod(idx, n)
        ax = axes[row, col]
        bar_width = 0.35
        x = np.arange(2)  # One position for IoU, one for Dice
        
        # Plot bars for ViT
        vit_bars = ax.bar(x - bar_width / 2, [iou_scores_vit[cls], dice_scores_vit[cls]], 
                          width=bar_width, color=vit_colors, edgecolor='gray', label='ViT', alpha=0.85)

        # Plot bars for CNN
        cnn_bars = ax.bar(x + bar_width / 2, [iou_scores_cnn[cls], dice_scores_cnn[cls]], 
                          width=bar_width, color=cnn_colors, edgecolor='gray', label='CNN', alpha=0.85)

        # Add bar annotations for exact values
        for bars, model in [(vit_bars, 'ViT'), (cnn_bars, 'CNN')]:
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f"{yval:.2f}",
                        ha='center', va='bottom', fontsize=9, weight='bold', color='#333333')

        # Set titles and axis labels
        ax.set_title(f"Class: {cls}", fontsize=14, color='#333333', weight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['IoU', 'Dice'], fontsize=12, color='#444444')
        ax.set_ylim(0, 1)
        ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.5)

        # Improve subplot aesthetics
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#dddddd')
        ax.spines['bottom'].set_color('#dddddd')

    # Add legend once in the top right subplot
    handles, labels = vit_bars + cnn_bars, ['ViT IoU', 'ViT Dice', 'CNN IoU', 'CNN Dice']
    fig.legend(handles, labels, loc='upper right', fontsize=12, frameon=False, bbox_to_anchor=(0.9, 0.95))

    # Turn off any empty subplots in the grid
    for idx in range(num_classes, n * n):
        fig.delaxes(axes.flatten()[idx])

    fig.savefig("./graphs/classwise_iou_dice.png", dpi=300, bbox_inches='tight', format='png')
    plt.show()

def prepare_dataset(dataset, batch_size=5):

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

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=collate_fn)


def test_models(snapshot_unet, snapshot_maskformer, unet, maskformer, processor, name_unet, name_maskformer, dataset):
    snapshot_unet = torch.load(snapshot_unet)
    snapshot_maskformer = torch.load(snapshot_maskformer)

    unet.load_state_dict(snapshot_unet['state_dict'])
    hist_unet = snapshot_unet['hist']
    print(summary(unet, ( 3, 512, 512), device=str(CONSTS.DEVICE)))
    unet.eval()

    maskformer.load_state_dict(snapshot_maskformer['state_dict'])
    hist_mask = snapshot_maskformer['hist']
    # print(summary(maskformer, ( 3, 512, 512), device=str(CONSTS.DEVICE)))
    maskformer.eval()

    valid_dataloader = prepare_dataset(dataset, batch_size=1)
    sample = to_device(next(iter(valid_dataloader))['pixel_values'], CONSTS.DEVICE)

    flops_unet = FlopCountAnalysis(unet, sample)
    flops_maskformer = FlopCountAnalysis(maskformer, sample)

    print("-"*50)
    print(f"flops for unet: {flops_unet.total()}")
    print("-"*50)
    print(f"flops for maskformer: {flops_maskformer.total()}")
    print("-"*50)

    print("-"*50)
    print(f"flops for unet by operator: {flops_unet.by_operator()}")
    print("-"*50)
    print(f"flops for maskformer by operator: {flops_maskformer.by_operator()}")
    print("-"*50)

    valid_dataloader = prepare_dataset(dataset, batch_size=64)


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
        # valid_loss_tr.append(value['v loss'])
        valid_iou_tr.append(value['v IoU'])
        valid_dice_tr.append(value['v Dice'])
        valid_acc_tr.append(value['v Acc'])

    unet_valid_metrics = Validate.validate_cnn(valid_dataloader, unet, per_class=True)
    print("result for unet per class")
    print(unet_valid_metrics[0])
    maskformer_valid_metrics = Validate.validate_vit(valid_dataloader, maskformer, processor, per_class=True)
    print("\nresult for maskformer")
    print(maskformer_valid_metrics[0])

    valid_dataloader = prepare_dataset(dataset, batch_size=64)
    sample_batch = next(iter(valid_dataloader))

    cnn_predictions = make_predictions(unet, sample_batch, name_unet)
    mask_predictions = make_predictions(maskformer, sample_batch, name_maskformer, processor)

    epochs = range(len(hist_mask))

    plot_model_comparison(epochs, valid_iou_cnn, valid_dice_cnn, valid_acc_cnn, 
                          valid_iou_tr, valid_dice_tr, valid_acc_tr)
    plot_sampled_images(sample_batch, cnn_predictions, mask_predictions)
    plot_classwise_iou_dice(*unet_valid_metrics,*maskformer_valid_metrics)

def main():
    unet = load_cnn()
    dataset = load_dataset()
    maskformer, processor = load_transformer()

    # test_models("model_cnn_v3.pt", "model_transformer_v2.pt", unet, maskformer, processor, "UNet CNN", "MaskFormer ViT", dataset)
    test_models("model_cnn_v3.pt", "model_transformer_v2.pt", unet, maskformer, processor, "UNet CNN", "MaskFormer ViT", dataset)

if __name__ == "__main__":
    main()

