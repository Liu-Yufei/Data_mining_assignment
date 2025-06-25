import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: [batch_size, feature_dim]
        # labels: [batch_size]
        
        # L2 normalize the features
        features = F.normalize(features, dim=1)
        
        # Compute logits matrix (dot product of features)
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T), # [a,a]
            self.temperature)
        
        # Remove self-contrast
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Create mask to identify positives
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        
        # Compute log-softmax
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(features.size(0)).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask
        
        # Compute log prob
        exp_logits = torch.exp(logits) * logits_mask
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)  # 加小数值防止log(0)

        
        # Compute mean of log-likelihood over positive samples
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        # Loss
        loss = -mean_log_prob_pos
        loss = loss.mean()
        
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

class CombinedLoss(nn.Module):
    def __init__(self, device, temperature=0.07, alpha=1, gamma=2):
        super(CombinedLoss, self).__init__()
        self.supervised_contrastive_loss1 = SupervisedContrastiveLoss(temperature)
        self.supervised_contrastive_loss2 = SupConLoss()
        self.supervised_contrastive_loss3 = ContrastiveCenterLoss(device)

        self.focal_loss = FocalLoss(alpha, gamma)
        self.CE_loss = nn.CrossEntropyLoss()
        self.BCE_loss = nn.BCELoss()

    def forward(self, features, outputs, labels):
        contrastive_loss1 = self.supervised_contrastive_loss1(features, labels)
        contrastive_loss2 = self.supervised_contrastive_loss2(features, outputs, labels)
        contrastive_loss3 = self.supervised_contrastive_loss3(features, labels)
        focal_loss = self.focal_loss(outputs, labels)
        CE_loss = self.CE_loss(outputs, labels)
        # labels = F.one_hot(labels,2).float()
        # BCE_loss = self.BCE_loss(outputs, labels)
        # return contrastive_loss1 + contrastive_loss2 + contrastive_loss3 + focal_loss + CE_loss
        # return 0.8*CE_loss + 0.2*contrastive_loss1
        # return 0.5*focal_loss + 0.5*contrastive_loss1
        # return CE_loss*0.8 + contrastive_loss3*10
        # return CE_loss*0.8 + contrastive_loss3*1.2
        return contrastive_loss3*1+CE_loss*0.08
        # return focal_loss + CE_loss +contrastive_loss1
        # return focal_loss + contrastive_loss3
        # return CE_loss
    

class CELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.xent_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        return self.xent_loss(outputs, targets)

class ContrastiveCenterLoss(nn.Module):
    def __init__(self, device, dim_hidden=32, num_classes=2, lambda_c=1.0, use_cuda=True):
        super(ContrastiveCenterLoss, self).__init__()
        self.device = device
        self.dim_hidden = dim_hidden # dim_hidden=hidden.size()[1]
        self.num_classes = num_classes
        self.lambda_c = lambda_c
        self.centers = nn.Parameter(torch.randn(num_classes, dim_hidden)).to(device)
        self.use_cuda = use_cuda

    # may not work due to flowing gradient. change center calculation to exp moving avg may work.
    def forward(self, hidden , y):
        batch_size = hidden.size()[0]
        expanded_centers = self.centers.expand(batch_size, -1, -1)
        expanded_hidden = hidden.expand(self.num_classes, -1, -1).transpose(1, 0)
        distance_centers = (expanded_hidden - expanded_centers).pow(2).sum(dim=-1)
        distances_same = distance_centers.gather(1, y.unsqueeze(1))
        intra_distances = distances_same.sum()
        inter_distances = distance_centers.sum().sub(intra_distances)
        epsilon = 1e-6
        # loss = (self.lambda_c / 2.0 / batch_size) * intra_distances / \
        #        (inter_distances + epsilon) / 0.1
        loss = (self.lambda_c / 2.0 ) * intra_distances / \
               (inter_distances + epsilon) / 0.1
        return loss

    def cuda(self, device_id=None):
        """Moves all model parameters and buffers to the GPU.

        Arguments:
            device_id (int, optional): if specified, all parameters will be
                copied to that device
        """
        self.use_cuda = True
        return self._apply(lambda t: t.to(self.device))

class SupConLoss(nn.Module):

    def __init__(self, alpha=0.6, temp=0.07):
        super().__init__()
        self.xent_loss = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.temp = temp

    def nt_xent_loss(self, anchor, target, labels):
        with torch.no_grad():
            labels = labels.unsqueeze(-1)
            mask = torch.eq(labels, labels.transpose(0, 1))
            # delete diag elem
            mask = mask ^ torch.diag_embed(torch.diag(mask))
        # compute logits
        anchor_dot_target = torch.einsum('bd,cd->bc', anchor, target) / self.temp
        # delete diag elem
        anchor_dot_target = anchor_dot_target - torch.diag_embed(torch.diag(anchor_dot_target))
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_target, dim=1, keepdim=True)
        logits = anchor_dot_target - logits_max.detach()
        # compute log prob
        exp_logits = torch.exp(logits)
        # mask out positives
        logits = logits * mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        # in case that mask.sum(1) is zero
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        # compute log-likelihood
        pos_logits = (mask * log_prob).sum(dim=1) / mask_sum.detach()
        loss = -1 * pos_logits.mean()
        return loss

    def forward(self, features, outputs, targets):
        # normed_cls_feats = F.normalize(outputs['cls_feats'], dim=-1)
        normed_cls_feats = F.normalize(features, dim=-1)
        # ce_loss = (1 - self.alpha) * self.xent_loss(outputs['predicts'], targets)
        ce_loss = (1 - self.alpha) * self.xent_loss(outputs, targets)
        cl_loss = self.alpha * self.nt_xent_loss(normed_cls_feats, normed_cls_feats, targets)
        return ce_loss + cl_loss


class DualLoss(SupConLoss):

    def __init__(self, alpha, temp):
        super().__init__(alpha, temp)

    def forward(self, outputs, targets):
        normed_cls_feats = F.normalize(outputs['cls_feats'], dim=-1)
        normed_label_feats = F.normalize(outputs['label_feats'], dim=-1)
        normed_pos_label_feats = torch.gather(normed_label_feats, dim=1, index=targets.reshape(-1, 1, 1).expand(-1, 1, normed_label_feats.size(-1))).squeeze(1)
        ce_loss = (1 - self.alpha) * self.xent_loss(outputs['predicts'], targets)
        cl_loss_1 = 0.5 * self.alpha * self.nt_xent_loss(normed_pos_label_feats, normed_cls_feats, targets)
        cl_loss_2 = 0.5 * self.alpha * self.nt_xent_loss(normed_cls_feats, normed_pos_label_feats, targets)
        return ce_loss + cl_loss_1 + cl_loss_2

class Loss_func(nn.Module):
    def __init__(self, args, device):
        super(Loss_func, self).__init__()
        self.loss_func = args.loss_func
        if args.loss_func == 'CombinedLoss':
            self.loss = CombinedLoss(device)
        elif args.loss_func == 'CE':
            self.loss = CELoss()
        elif args.loss_func == 'FocalLoss':
            self.loss = FocalLoss()
        elif args.loss_func == 'SCLoss':
            self.loss = SupConLoss()
        # elif args.loss_func == 'CCLoss':
        #     self.loss_func = ContrastiveCenterLoss()
    
    def forward(self, features, outputs, labels):
        if self.loss_func == 'CombinedLoss':
            loss = self.loss(features, outputs, labels)
        elif self.loss_func == 'SCLoss':
            loss = self.loss(features, outputs, labels)
        else:
            loss =  self.loss(outputs, labels)
        return loss