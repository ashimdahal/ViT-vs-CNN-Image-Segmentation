from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

from PIL import Image
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

import os, json

@dataclass
class CONSTS:
    id2label = {
                0: 'ship',
                1: 'storage_tank',
                2: 'baseball_diamond',
                3:  'tennis_court',
                4: 'basketball_court',
                5: 'Ground_Track_Field',
                6:   'Bridge',
                7: 'Large_Vehicle',
                8: 'Small_Vehicle',
                9: 'Helicopter',
                10: 'Swimming_pool',
                11: 'Roundabout',
                12: 'Soccer_ball_field',
                13: 'plane',
                14: 'Harbor',
                15:  'background',
            }
    mapping = {
            (0, 0, 63): 0,# 'ship',
            (0, 63, 63): 1, #'storage_tank',
            (0, 63, 0): 2, #'baseball_diamond',
            (0, 63, 127): 3, # 'tennis_court',
            (0, 63, 191): 4, #'basketball_court',
            (0, 63, 255): 5, #'Ground_Track_Field',
            (0, 127, 63):  6, #'Bridge',
            (0, 127, 127): 7, #'Large_Vehicle',
            (0, 0, 127): 8, #'Small_Vehicle',
            (0, 0, 191): 9, #'Helicopter',
            (0, 0, 255): 10, #'Swimming_pool',
            (0, 191, 127): 11, #'Roundabout',
            (0, 127, 191): 12, #'Soccer_ball_field',
            (0, 127, 255): 13, #'plane',
            (0, 100, 155): 14, #'Harbor'
            (0, 0, 0): 15,# 'unlabeled',
        }
        
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transforms = A.Compose(
        [
            A.RandomResizedCrop(height=512, width=512, scale=(0.06, 0.28), p=1),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.SafeRotate(limit=360, p=1),
            A.RandomBrightnessContrast(p=0.2),
            # A.HueSaturationValue(p=0.2),
            # A.GaussNoise(p=0.1),
            # A.ElasticTransform(p=0.2),
            A.CoarseDropout(p=0.35, max_height=32, max_width=32),
            A.augmentations.transforms.Normalize(
                mean = (0.485, 0.456, 0.406),
                std = (0.229, 0.224, 0.225)
            ),
            ToTensorV2(),
        ]
    )

    DS_DIR = "iSAID"
    ignore_index = 15


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

class Validate:

    @torch.no_grad()
    def calculate_accuracy(prediction, target, ignore_class=CONSTS.ignore_index):
        # Create a mask to ignore the specified class in the target
        valid_mask = (target != ignore_class)
        
        # Compute the total number of valid pixels (not ignored)
        total_valid_pixels = valid_mask.sum().item()
        
        if total_valid_pixels == 0:
            return 0.0  # If there are no valid pixels, return 0 accuracy
        
        # Compute the number of correct predictions where the mask is valid
        correct_predictions = (prediction == target) & valid_mask
        correct_count = correct_predictions.sum().item()
        
        # Calculate accuracy as the ratio of correct valid predictions to total valid pixels
        accuracy = correct_count / total_valid_pixels
        
        return accuracy

    @torch.no_grad()
    def calculate_dice_score(prediction, target, num_classes=15, ignore_index=CONSTS.ignore_index, per_class=False):
        if per_class:
            return Validate.calculate_iou_per_class(prediction, target, num_classes)
        # Ignore predictions where target is ignore_index
        valid_mask = (target != ignore_index)

        # Create binary masks for each class and calculate Dice score
        dice_scores = []
        for cls in range(num_classes):
            pred_mask = (prediction == cls)
            target_mask = (target == cls)
            
            if pred_mask.sum() == 0 and target_mask.sum() == 0:
                continue

            # Apply valid_mask to ignore specific areas
            pred_mask = pred_mask & valid_mask
            target_mask = target_mask & valid_mask

            intersection = (pred_mask & target_mask).sum((1,2))
            union = pred_mask.sum((1,2)) + target_mask.sum((1,2))

            dice_score = (2 * intersection + 1e-8) / (union + 1e-8)
            dice_scores.append(dice_score)

        # Return the mean Dice score
        return torch.mean(torch.stack(dice_scores))

    @torch.no_grad()
    def calculate_iou(prediction, target, num_classes=15, ignore_index=CONSTS.ignore_index, per_class=False):
        if per_class:
            return Validate.calculate_iou_per_class(prediction, target, num_classes)
        # Ignore predictions where target is ignore_index
        valid_mask = (target != ignore_index)    
        # store all intersection over union for all classes
        iou_per_class = []
        for cls in range(num_classes):        
            # Create binary masks for this class
            pred_mask = (prediction == cls)
            target_mask = (target == cls)
            
            # Skip class if there are no predictions or targets for this class
            if pred_mask.sum() == 0 and target_mask.sum() == 0:
                continue  # No need to compute IoU for empty classes

            pred_mask = pred_mask & valid_mask
            target_mask = target_mask & valid_mask
            
            # Calculate intersection and union
            intersection = (pred_mask & target_mask).sum((1,2)).float()
            union = (pred_mask | target_mask).sum((1,2)).float()

                
            iou_score = (intersection+1e-8)/(union + 1e-8)
            iou_per_class.append(iou_score)
            
        return torch.mean(torch.stack(iou_per_class))

    @torch.no_grad()
    def calculate_iou_per_class(prediction, target, num_classes, ignore_index = CONSTS.ignore_index):

        iou_per_class = torch.zeros(num_classes)
        valid_mask = (target!=ignore_index)
        for cls in range(num_classes):        
            # Create binary masks for this class
            pred_mask = (prediction == cls)
            target_mask = (target == cls)

            # Skip class if there are no predictions or targets for this class
            if pred_mask.sum() == 0 and target_mask.sum() == 0:
                continue  # No need to compute IoU for empty classes

            pred_mask = pred_mask & valid_mask
            target_mask = target_mask & valid_mask
            
            # Calculate intersection and union
            intersection = (pred_mask & target_mask).sum((1,2)).float()
            union = (pred_mask | target_mask).sum((1,2)).float()

            iou_score = (intersection+1e-8)/(union + 1e-8)
            iou_per_class[cls] = torch.mean(iou_score)

        return iou_per_class

    @torch.no_grad()
    def calculate_dice_score_per_class(prediction, target, num_classes, ignore_index = CONSTS.ignore_index):
        dice_score_per_class = torch.zeros(num_classes)
        valid_mask = (target!=ignore_index)

        for cls in range(num_classes):
            pred_mask = (prediction ==cls)
            target_mask = (target==cls)

            if pred_mask.sum() == 0 and target_mask.sum() == 0:
                continue

            pred_mask = pred_mask & valid_mask
            target_mask = target_mask & valid_mask

            intersection = (pred_mask & target_mask).sum()
            union = (pred_mask + target_mask).sum()
            dice_score = (2 * intersection + 1e-8) / (union + 1e-8)
            dice_score_per_class[cls] = iou_score 

        return dice_score_per_class

    @torch.no_grad()
    @torch.inference_mode()
    def validate_cnn(valid_dataloader, model, loss_fn = None, per_class=False):
        model.eval()
        valid_losses, valid_IoUs, valid_dice_scores, valid_accs = [],[],[],[]
        
        for batch in valid_dataloader:
            y_preds = model(batch["pixel_values"].to(CONSTS.DEVICE))
            
            target = to_device(batch["augmented_pixel_mask"], CONSTS.DEVICE)

            # Calculate loss
            # Make probability distribution from the logits
            probabilities = torch.argmax(F.softmax(y_preds, dim=1), axis=1)
            
            # Calculate mean IoU
            mean_IoU = Validate.calculate_iou(probabilities, target, y_preds.size(1)-1, per_class=per_class)
            
            # Calculate Dice score
            mean_dice = Validate.calculate_dice_score(probabilities, target, y_preds.size(1) -1, per_class=per_class)

            acc = Validate.calculate_accuracy(probabilities, target, y_preds.size(1) -1)
            # Store metrics
            loss = loss_fn(y_preds, target) if loss_fn is not None else torch.tensor([0.,0])

            valid_accs.append(acc)
            valid_losses.append(loss)
            valid_IoUs.append(mean_IoU)
            valid_dice_scores.append(mean_dice)

        if per_class:
            # 0th element is variance , 1st element is the actual mean
            valid_IoUs = torch.stack(valid_IoUs)
            valid_iou = torch.mean(valid_IoUs,dim = 0).cpu().numpy()
            valid_dice_scores = torch.stack(valid_dice_scores)
            valid_dice_scores = torch.mean(valid_dice_scores,dim = 0).cpu().numpy()
            return dict(zip(list(CONSTS.id2label.values()), valid_iou)), dict(zip(CONSTS.id2label.values(), valid_dice_scores)) 
        # Calculate mean metrics
        valid_loss = torch.mean(torch.stack(valid_losses)).item() 
        valid_iou = torch.mean(torch.stack(valid_IoUs)).item()
        valid_dice = torch.mean(torch.stack(valid_dice_scores)).item()
        valid_acc = torch.mean(torch.tensor(valid_accs)).item()
        
        return {
            "v loss": valid_loss, 
            "v IoU": valid_iou, 
            "v Dice": valid_dice,
            "v Acc" : valid_acc
        }

    @torch.no_grad()
    @torch.inference_mode()
    def validate_vit(valid_dataloader, model, processor,per_class=False):
        model.eval()
        valid_IoUs, valid_dice_scores, valid_accs = [],[],[]
        
        for batch in valid_dataloader:
            outputs = model(
                        pixel_values=batch["pixel_values"].to(CONSTS.DEVICE),
                        mask_labels=to_device(batch["mask_labels"], CONSTS.DEVICE),
                        class_labels=to_device(batch["class_labels"], CONSTS.DEVICE),
                    )
            output_sizes = [(512, 512)] * outputs['masks_queries_logits'].size(0)
            predicted_semantic_maps = (
                    processor.post_process_semantic_segmentation(
                    outputs, target_sizes=output_sizes
                )
            )
            target = to_device(batch["augmented_pixel_mask"], CONSTS.DEVICE)
            
            probabilities = torch.stack(predicted_semantic_maps)
            # Calculate mean IoU
            mean_IoU = Validate.calculate_iou(probabilities, target,  per_class=per_class)
            
            # Calculate Dice score
            mean_dice = Validate.calculate_dice_score(probabilities, target, per_class=per_class)

            acc = Validate.calculate_accuracy(probabilities, target )
            # Store metrics
            valid_accs.append(acc)
            valid_IoUs.append(mean_IoU)
            valid_dice_scores.append(mean_dice)

        if per_class:
            valid_IoUs = torch.stack(valid_IoUs)
            valid_iou = torch.mean(valid_IoUs, dim=0).cpu().numpy()
            valid_dice_scores = torch.stack(valid_dice_scores)
            valid_dice_scores = torch.mean(valid_dice_scores,dim = 0).cpu().numpy()
            return dict(zip(list(CONSTS.id2label.values()), valid_iou)), dict(zip(CONSTS.id2label.values(), valid_dice_scores)) 

        # Calculate mean metrics
        valid_iou = torch.mean(torch.stack(valid_IoUs)).item()
        valid_dice = torch.mean(torch.stack(valid_dice_scores)).item()
        valid_acc = torch.mean(torch.tensor(valid_accs)).item()
        
        return {
            "v IoU": valid_iou, 
            "v Dice": valid_dice,
            "v Acc" : valid_acc
        }


@dataclass
class SegmentationDataInput:
    original_image: np.ndarray
    transformed_image: np.ndarray
    original_segmentation_map: np.ndarray
    transformed_segmentation_map: np.ndarray


class IsaidDataset(Dataset):
    def __init__(self, metadata, isaid_dir, transforms = None):
        super().__init__()
        self.metadata = metadata
        self.mapping = {
            (0, 0, 0): 15,# 'unlabeled',
            (0, 0, 63): 0,# 'ship',
            (0, 63, 63): 1, #'storage_tank',
            (0, 63, 0): 2, #'baseball_diamond',
            (0, 63, 127): 3, # 'tennis_court',
            (0, 63, 191): 4, #'basketball_court',
            (0, 63, 255): 5, #'Ground_Track_Field',
            (0, 127, 63):  6, #'Bridge',
            (0, 127, 127): 7, #'Large_Vehicle',
            (0, 0, 127): 8, #'Small_Vehicle',
            (0, 0, 191): 9, #'Helicopter',
            (0, 0, 255): 10, #'Swimming_pool',
            (0, 191, 127): 11, #'Roundabout',
            (0, 127, 191): 12, #'Soccer_ball_field',
            (0, 127, 255): 13, #'plane',
            (0, 100, 155): 14 #'Harbor'
        }
        
        self.ISAID_DIR = isaid_dir
        self.DOTA_DIR = "dota"
        
        # must be albumenations transformations
        self.transforms = transforms
        self.to_rgb = A.ToRGB()

        self.INCREASE_FACTOR = 1
            
    
    def __len__(self):
        return len(self.metadata['images']) * self.INCREASE_FACTOR

    # def mask_to_class(self, mask):
    #     mask = torch.from_numpy(np.array(mask))
    #     mask = torch.squeeze(mask)

    #     # transform from (C,H,W) to (H,W,C)
    #     class_mask = mask
    #     class_mask = class_mask.permute(2,0,1).contiguous()
    #     H,W = class_mask.shape[1], class_mask.shape[2]

    #     mask_class = torch.zeros(H,W, dtype=torch.long)

    #     for key in self.mapping:
    #         idx = (class_mask == torch.tensor(key, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
    #         validx = (idx.sum(0) == 3)

    #         mask_class[validx] = torch.tensor(self.mapping[key], dtype=torch.long)

    #     return mask_class

    def mask_to_class(self, mask):
        # Convert mask to tensor if not already
        mask = torch.from_numpy(np.array(mask)).permute(2, 0, 1).contiguous()
    
        # Flatten mask to (H*W, C) and apply mapping
        h, w = mask.shape[1:]
        flat_mask = mask.view(3, -1).t()  # Shape: (H*W, C)
    
        # Build a lookup table to map colors to classes
        color_map = torch.tensor(list(self.mapping.keys()), dtype=torch.uint8)
        class_map = torch.tensor(list(self.mapping.values()), dtype=torch.long)
    
        # Find matching colors (expand the color_map to (N_colors, H*W, C) and compare)
        flat_mask = flat_mask.unsqueeze(0).expand(len(self.mapping), -1, -1)  # Shape: (N_colors, H*W, C)
        color_map = color_map.unsqueeze(1).expand(-1, flat_mask.size(1), -1)  # Shape: (N_colors, H*W, C)
    
        # Calculate per-pixel matches across all colors
        matches = (flat_mask == color_map).all(dim=-1)
    
        # Pick the corresponding class for each pixel (first match per pixel)
        mask_class = torch.zeros(h * w, dtype=torch.long) - 1  # Start with -1 (unlabeled)
        for i, match in enumerate(matches):
            mask_class[match] = class_map[i]
    
        # Reshape back to (H, W)
        return mask_class.view(h, w)

    def rgba_to_rgb(self, image):
        img_tmp = Image.fromarray(image)
        if img_tmp.mode == "RGBA" or img_tmp.mode =="CMYK":
            return np.array(img_tmp.convert("RGB"))
        
    def __getitem__(self, idx):
        idx = idx % len(self.metadata['images'])
        
        real_image_file_name = self.metadata['images'][idx]['file_name']
        segment_image_file_name = self.metadata['images'][idx]['seg_file_name']

        real_image_path = os.path.join(
            self.DOTA_DIR, 
            "images", 
            real_image_file_name
        )
        
        segment_image_path = os.path.join(
            self.ISAID_DIR, 
            "Semantic_masks",
            "images", 
            segment_image_file_name
        )
        
        real_image = Image.open(real_image_path)
        segment_image = Image.open(segment_image_path)
        mask = segment_image
        image = real_image

        assert(self.transforms is not None)
        #common bro its compvis
        
        original_image = np.array(image)
        original_segmentation_map = np.array(mask)
        #some images have 4 channels, if not, then having more than 4 elements on last row would mean they are (H,W>4) image 
        # this is with assumption that there are no images less than 4 can to tensorpixel width
        if original_image.shape[-1] > 4: 
            original_image = self.to_rgb(image=original_image)['image']
        elif original_image.shape[-1] == 4:
            original_image = self.rgba_to_rgb(original_image)
            
        augmentations = self.transforms(image=original_image, mask=original_segmentation_map)
        transformed_image = augmentations['image']
        transformed_segmentation_map = augmentations['mask']
        

        assert(transformed_segmentation_map.shape[2] == 3)
        transformed_segmentation_map = self.mask_to_class(transformed_segmentation_map).to(torch.long)
        # original_segmentation_map = self.mask_to_class(original_segmentation_map).to(torch.long)
        
        original_image = torch.from_numpy(original_image).permute(2,0,1)
        
        return SegmentationDataInput(
            original_image=original_image,
            transformed_image=transformed_image,
            original_segmentation_map=original_segmentation_map,
            transformed_segmentation_map=transformed_segmentation_map,
        )



def load_metadata(fn):
    with open(fn, 'r') as f:
        metadata = json.load(f)

    return metadata
    
def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(d, device) for d in data]
    return data.to(device, non_blocking = True)


