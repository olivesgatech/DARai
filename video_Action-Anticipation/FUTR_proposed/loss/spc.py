# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from math import log
# class SupervisedContrastiveLoss(nn.Module):
#     def __init__(self, temperature=0.07):
#         """
#         Implementation of the loss described in the paper Supervised Contrastive Learning :
#         https://arxiv.org/abs/2004.11362

#         :param temperature: int
#         """
#         super(SupervisedContrastiveLoss, self).__init__()
#         self.temperature = temperature

#     def forward(self, projections, targets):
#         """

#         :param projections: torch.Tensor, shape [batch_size, projection_dim]
#         :param targets: torch.Tensor, shape [batch_size]
#         :return: torch.Tensor, scalar
#         """
#         projections = F.normalize(projections, p=2, dim=1)
#         device = torch.device("cuda") if projections.is_cuda else torch.device("cpu")

#         dot_product_tempered = torch.mm(projections, projections.T) / self.temperature
#         # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
#         exp_dot_tempered = (
#             torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
#         )

#         mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
#         mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
#         mask_combined = mask_similar_class * mask_anchor_out
#         cardinality_per_samples = torch.sum(mask_combined, dim=1)

#         log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)+ 1e-6))
#         supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples

#         # Filter out NaN values
#         non_nan_mask = ~torch.isnan(supervised_contrastive_loss_per_sample)  # Mask for non-NaN values
#         non_nan_values = supervised_contrastive_loss_per_sample[non_nan_mask]  # Select only non-NaN values

#         # Compute mean of non-NaN values
#         if non_nan_values.numel() > 0:  # Ensure there are non-NaN values
#             print("here are not nan")
#             supervised_contrastive_loss = torch.mean(non_nan_values)
#         else:
#             print("0")
#             supervised_contrastive_loss = torch.tensor(0.0, device=supervised_contrastive_loss_per_sample.device)
#             #supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

#         return supervised_contrastive_loss

"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.ones_like(mask_pos_pairs), mask_pos_pairs)
        #mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
