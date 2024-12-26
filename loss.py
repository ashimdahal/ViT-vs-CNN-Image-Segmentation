import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils import CONSTS

class CombinedLoss(nn.Module):
    def __init__(self, alpha=1, beta=0.8, gamma = 10, ignore_index=15):
        """
        Args:
            num_classes (int): Number of classes in the segmentation task.
            alpha (float): Weight for the Dice loss component.
            beta (float): Weight for the IoU (Jaccard) loss component.
            ignore_index (int): Label to ignore in the target.
        """
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ignore_index = ignore_index
        weight = [1]*15 #for the first class
        weight.extend([0.15])
        weights = torch.tensor(weight).to(CONSTS.DEVICE)
        # self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=weights)

        self.SMOOTH = 1e-8

    def forward(self, prediction_logits, target):
        """
        Args:
            prediction_logits: Tensor of shape (batch_size, num_classes, height, width) containing the logits of the predicted segmentation masks.
            target: Tensor of shape (batch_size, height, width) containing the ground truth segmentation masks.
        
        Returns:
            Combined loss as a weighted sum of Dice, IoU, and CrossEntropy losses.
        """
        # Compute CrossEntropy loss
        cross_entropy_loss = self.cross_entropy_loss(prediction_logits, target)
        self.num_classes = prediction_logits.size(1)
        # Convert logits to predicted class labels
        # unlike testing even in making prediction we dont use argmax
        # since torch.argmax is not a differentiable function
        prediction = F.softmax(prediction_logits, dim=1)
        valid_mask = (target != self.ignore_index)

        # Dice Loss
        dice_loss = self._dice_loss(prediction, target, valid_mask)
        
        # IoU Loss
        iou_loss = self._iou_loss(prediction, target, valid_mask)

        # Combined loss
        combined_loss = (self.alpha * dice_loss + 
                         self.beta * iou_loss + 
                         (self.gamma) * cross_entropy_loss)

        return combined_loss

    def _dice_loss(self, probabilities, target, valid_mask):
        dice_scores = []
        for cls in range(self.num_classes):
            pred_mask = probabilities[:, cls]
            # to be used for testing; faster but no computation graph
            # pred_mask = (pred_mask == cls)
           
            target_mask = (target == cls).float()

            if pred_mask.sum() == 0 and target_mask.sum() == 0:
                continue
            #  in the loss function we use * instead of bitwise operations like
            # & and | because we need to have a differentiable loss function
            # while testing in valudation though, we don't need such kind of 
            # contuinity of differentiability so we use & and | for faster computation
            pred_mask = pred_mask * valid_mask
            target_mask = target_mask * valid_mask

            intersection = (pred_mask * target_mask).sum())
            union = pred_mask.sum((1,2)) + target_mask.sum())

            dice_score = (2 * intersection + self.SMOOTH) / (union + self.SMOOTH)
            dice_loss = 1 - dice_score  # Dice loss is 1 - Dice score
            dice_scores.append(dice_loss)

        return torch.mean(torch.stack(dice_scores))

    def _iou_loss(self, probabilities, target, valid_mask):
        iou_scores = []
        for cls in range(self.num_classes):
            pred_mask = probabilities[:, cls]
            # to be used for testing; faster but no computation graph
            # pred_mask = (pred_mask == cls)
            
            target_mask = (target == cls).float()

            if pred_mask.sum() == 0 and target_mask.sum() == 0:
                continue

            pred_mask = pred_mask * valid_mask
            target_mask = target_mask * valid_mask

            intersection = (pred_mask * target_mask).sum())
            union = (pred_mask + target_mask - pred_mask * target_mask).sum())

            iou_score = (intersection +self.SMOOTH)/ (union+self.SMOOTH)
            iou_loss = 1 - iou_score  # IoU loss is 1 - IoU score
            iou_scores.append(iou_loss)

        return torch.mean(torch.stack(iou_scores))

