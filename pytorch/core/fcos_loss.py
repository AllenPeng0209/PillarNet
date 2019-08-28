import torch
from torch import nn
from IPython import embed

#To do: revise to 3d IOU for kitti


class IOULoss(nn.Module):
    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_aera = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_aera = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect

        losses = -torch.log((area_intersect + 1.0) / (area_union + 1.0))

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()


class FCOS_resgression_Smooth_L1Loss(nn.Module):
  """Smooth L1 localization loss function.

  The smooth L1_loss is defined elementwise as .5 x^2 if |x|<1 and |x|-.5
  otherwise, where x is the difference between predictions and target.

  See also Equation (3) in the Fast R-CNN paper by Ross Girshick (ICCV 2015)
  """
  def __init__(self, sigma=3.0, code_weights=None, codewise=False):
    super().__init__()
    self._sigma = sigma
    if code_weights is not None:
      self._code_weights = np.array(code_weights, dtype=np.float32)
      self._code_weights = Variable(torch.from_numpy(self._code_weights).cuda())
    else:
      self._code_weights = None
    self._codewise = codewise
  def forward(self, prediction_tensor, target_tensor, weights=None):
    """Compute loss function.

    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_positive_label_pixal_per_images,
        code_size] representing the (encoded) predicted locations of objects.
      target_tensor: A float tensor of shape [batch_size, num_positive_label_pixal_per_images,
        code_size] representing the regression targets
      weights: a float tensor of shape [batch_size, num_positive_label_pixal_per_images]

    Returns:
      loss: a float tensor of shape [batch_size, num_positive_label_pixal_per_images] tensor
        representing the value of the loss function.
    """
    diff = prediction_tensor - target_tensor
    if self._code_weights is not None:
      code_weights = self._code_weights.type_as(prediction_tensor)
      diff = code_weights.view(1, 1, -1) * diff
    abs_diff = torch.abs(diff)
    abs_diff_lt_1 = torch.le(abs_diff, 1 / (self._sigma**2)).type_as(abs_diff)
    loss = abs_diff_lt_1 * 0.5 * torch.pow(abs_diff * self._sigma, 2) \
      + (abs_diff - 0.5 / (self._sigma**2)) * (1. - abs_diff_lt_1)
    if self._codewise:
      pixal_wise_smooth_l1norm = loss
      if weights is not None:
        pixal_wise_smooth_l1norm *= weights.unsqueeze(-1)
    else:
      pixal_wise_smooth_l1norm = torch.sum(loss)#  * weights
      if weights is not None:
        pixal_wise_smooth_l1norm *= weights
    return pixal_wise_smooth_l1norm


