import torch
from torch.nn import functional as F
from typing import Callable, List, Optional, Union
import numpy as np
from math import ceil, floor
from torchvision.ops import masks_to_boxes, generalized_box_iou_loss

def convert_similarity_to_image_size(
    similarity_map: torch.Tensor,
    width: int,
    height: int,
    resize_size: Optional[int],
    crop_size: Optional[int],
    val_img_transform: Optional[Callable] = None,
    interpolation: str = "bilinear",
) -> np.ndarray:
    """
    Convert similarity map from raw patch grid to original image size,
    taking into account whether the image has been resized and/or cropped prior to entering the network.
    """
    n_patches_h, n_patches_w = similarity_map.shape[0], similarity_map.shape[1]
    target_shape = 1, 1, n_patches_h, n_patches_w
    smallest_dimension = min(height, width)

    # TODO:
    # verify_resize_params(val_img_transforms, resize_size, crop_size)

    reshaped_similarity = similarity_map.reshape(target_shape)
    align_corners_modes = "linear", "bilinear", "bicubic", "trilinear"
    align_corners = False if interpolation in align_corners_modes else None

    if crop_size is not None:
        if resize_size is not None:
            cropped_size_orig_space = int(crop_size * smallest_dimension / resize_size)
            target_size = cropped_size_orig_space, cropped_size_orig_space
        else:
            target_size = crop_size, crop_size
        similarity_map = F.interpolate(
            reshaped_similarity,
            size=target_size,
            mode=interpolation,
            align_corners=align_corners,
        )
        margin_w, margin_h = (width - target_size[0]), (height - target_size[1])
        margins_for_pad = (floor(margin_w / 2), ceil(margin_w / 2), floor(margin_h / 2), ceil(margin_h / 2))
        similarity_map = F.pad(similarity_map[0, 0], margins_for_pad, value=float(0))
    else:
        similarity_map = F.interpolate(
            reshaped_similarity,
            size=(height, width),
            mode=interpolation,
            align_corners=align_corners,
        )[0, 0]
    return similarity_map.numpy()


def bbox_to_mask(image_shape, x, y, w, h):
    """
    Convert bounding box coordinates to a binary mask.
    
    Parameters:
    - image_shape: Tuple (height, width) of the image.
    - x, y: Coordinates of the top-left corner of the bounding box.
    - w, h: Width and height of the bounding box.
    
    Returns:
    - A binary mask as a numpy array.
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    mask[y:y+h, x:x+w] = 1
    return mask

def RSNA_bbox_string_to_mask(bbox_string, image_width, image_height):
    # Initialize a blank mask
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    # Split the string into individual bounding boxes
    bbox_list = bbox_string.split('|')
    
    # Process each bounding box
    for bbox in bbox_list:
        x, y, w, h = map(float, bbox.split(';'))
        
        # Convert coordinates and dimensions to integers
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        # Set the mask to 1 for the area covered by the bounding box
        mask[y:y+h, x:x+w] = 1
    
    return mask

def dice_score(mask1, mask2):
    """
    Calculate the Dice score between two binary masks.
    
    Parameters:
    - mask1: First binary mask as a numpy array.
    - mask2: Second binary mask as a numpy array.
    
    Returns:
    - Dice score: A float between 0 and 1.
    """
    assert mask1.shape == mask2.shape
    intersection = np.sum(mask1 * mask2)
    sum_mask1 = np.sum(mask1)
    sum_mask2 = np.sum(mask2)
    
    if sum_mask1 + sum_mask2 == 0:
        return 1.0 if sum_mask1 == sum_mask2 == 0 else 0.0
    
    return 2. * intersection / (sum_mask1 + sum_mask2)


def iou_score(mask1, mask2):
    """
    Calculate the Intersection over Union (IoU) score between two binary masks.
    
    Parameters:
    - mask1: First binary mask as a numpy array.
    - mask2: Second binary mask as a numpy array.
    
    Returns:
    - IoU score: A float between 0 and 1.
    """
    assert mask1.shape == mask2.shape
    intersection = np.sum(mask1 * mask2)
    union = np.sum(mask1) + np.sum(mask2) - intersection
    
    if union == 0:
        return 1.0 if np.sum(mask1) == np.sum(mask2) == 0 else 0.0
    
    return intersection / union

def pointing_game_hit(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    assert pred_mask.shape == gt_mask.shape, f"Predicted mask and ground truth mask must have the same shape. got {pred_mask.shape} and {gt_mask.shape}"
    pred_point = np.unravel_index(np.argmax(pred_mask), pred_mask.shape)
    return gt_mask[pred_point]==1

def compute_cnr(pred, mask):
    # Ensure the prediction and mask have the same shape
    assert pred.shape == mask.shape, "Prediction and mask must have the same shape"

    # Create masks for the interior and exterior regions
    interior_mask = mask > 0  # Interior where mask is greater than 0
    exterior_mask = mask == 0  # Exterior where mask is equal to 0

    # Check if there are any valid interior or exterior regions for each batch element
    valid_interior = np.any(interior_mask, axis=(1, 2))  # Boolean array indicating if there's any interior region
    valid_exterior = np.any(exterior_mask, axis=(1, 2))  # Boolean array indicating if there's any exterior region

    # Set predictions to NaN where the mask condition is not met, for both interior and exterior
    interior_pred = np.where(interior_mask, pred, np.nan)  # Predictions within the interior region
    exterior_pred = np.where(exterior_mask, pred, np.nan)  # Predictions within the exterior region

    # Calculate the mean values for the interior and exterior regions
    mu_A = np.nanmean(interior_pred, axis=(1, 2))  # Mean of interior region, ignoring NaNs
    mu_A = np.where(valid_interior, mu_A, 0.0)  # Set mean to 0.0 if there is no valid interior region

    mu_A_bar = np.nanmean(exterior_pred, axis=(1, 2))  # Mean of exterior region, ignoring NaNs
    mu_A_bar = np.where(valid_exterior, mu_A_bar, 0.0)  # Set mean to 0.0 if there is no valid exterior region

    # Calculate the variance values for the interior and exterior regions
    var_A = np.nanvar(interior_pred, axis=(1, 2))  # Variance of interior region, ignoring NaNs
    var_A = np.where(valid_interior, var_A, 0.0)  # Set variance to 0.0 if there is no valid interior region

    var_A_bar = np.nanvar(exterior_pred, axis=(1, 2))  # Variance of exterior region, ignoring NaNs
    var_A_bar = np.where(valid_exterior, var_A_bar, 0.0)  # Set variance to 0.0 if there is no valid exterior region

    # Calculate the sum of variances
    var_sum = var_A + var_A_bar  # Sum of variances for interior and exterior regions

    # Calculate the Contrast to Noise Ratio (CNR)
    cnr = np.abs(mu_A - mu_A_bar) / np.sqrt(var_sum)  # CNR calculation
    cnr = np.where(var_sum > 0, cnr, 0.0)  # Set CNR to 0.0 if the variance sum is zero to avoid division by zero

    return cnr

def compute_giou(pred: torch.Tensor, mask: torch.Tensor, thresh=0.5) -> torch.Tensor:
    """
    Computes the Generalized Intersection over Union (gIoU) between predicted and ground truth masks.

    Args:
        pred (torch.Tensor): Predicted masks, shape (B, H, W).
        mask (torch.Tensor): Ground truth masks, shape (B, H, W). Values should be binary (0 or 1).

    Returns:
        torch.Tensor: gIoU for each element in the batch, shape (B,)
    """
    pred = (pred > thresh).float()
    mask = (mask > thresh).float()

    # Convert masks to bounding boxes
    # masks_to_boxes expects masks in (B, H, W) and returns (B, 4) with (x1, y1, x2, y2)
    pred_boxes = masks_to_boxes(pred)  # Shape: (B, 4)
    target_boxes = masks_to_boxes(mask)  # Shape: (B, 4)

    # Handle cases where masks might be empty (no object)
    # masks_to_boxes returns an empty tensor if no mask is present; we need to handle it
    # For simplicity, we'll assign zero boxes where masks are empty
    if pred_boxes.shape[0] != pred.shape[0]:
        # Some predictions have no boxes; pad with zeros
        padded_pred_boxes = torch.zeros((pred.shape[0], 4), device=pred.device)
        padded_pred_boxes[:pred_boxes.shape[0]] = pred_boxes
        pred_boxes = padded_pred_boxes

    if target_boxes.shape[0] != mask.shape[0]:
        # Some ground truths have no boxes; pad with zeros
        padded_target_boxes = torch.zeros((mask.shape[0], 4), device=mask.device)
        padded_target_boxes[:target_boxes.shape[0]] = target_boxes
        target_boxes = padded_target_boxes

    # Compute gIoU loss (which is 1 - gIoU)
    giou_loss = generalized_box_iou_loss(pred_boxes, target_boxes)  # Shape: (B,)

    # Convert loss to gIoU
    giou = 1 - giou_loss  # Shape: (B,)

    return giou


class LinearScheduler:
    def __init__(self, max_value, num_steps):
        """
        Initialize the scheduler.
        
        :param max_value: The maximum value to reach.
        :param num_steps: The number of steps to reach the maximum value.
        """
        self.max_value = max_value
        self.num_steps = num_steps
        self.current_step = 0

    def step(self):
        """
        Increment the scheduler and return the value for the current step.
        """
        if self.current_step < self.num_steps:
            # Calculate the value based on current step
            value = (self.max_value / self.num_steps) * self.current_step
        else:
            # If max step reached, return the max value
            value = self.max_value
        
        self.current_step += 1
        return value

    def reset(self):
        """
        Reset the scheduler back to step 0.
        """
        self.current_step = 0
