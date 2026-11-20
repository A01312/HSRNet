import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_reconstruction(reconstructed, visible, mask, edges, tau):
    reconstructed, visible, mask, edges = (item.float() for item in (reconstructed, visible, mask, edges))
    visible_mask = 1 - mask
    queries = F.normalize((reconstructed * mask).sum((2, 3)) / mask.sum((2, 3)).clamp_min(1), dim=1)
    keys = F.normalize(visible.sum((2, 3)) / visible_mask.sum((2, 3)).clamp_min(1), dim=1)
    has_visible = visible_mask.flatten(1).any(dim=1)
    valid = has_visible & mask.flatten(1).any(dim=1)
    region_loss = reconstructed.sum() * 0
    if valid.any() and has_visible.sum() > 1:
        logits = queries[valid] @ keys[has_visible].t() / tau
        targets = (has_visible.long().cumsum(dim=0) - 1)[valid]
        region_loss = F.cross_entropy(logits, targets)
    edge_loss = ((reconstructed - visible) * edges).flatten(1).square().sum(dim=1).mean()
    return region_loss + edge_loss


def cross_modal_triplet(images, texts, pids, margin):
    distances = torch.cdist(F.normalize(images.float(), dim=1), F.normalize(texts.float(), dim=1))
    positive = pids[:, None].eq(pids[None, :])
    valid = positive.any(dim=1) & (~positive).any(dim=1)
    if not valid.any():
        return distances.sum() * 0

    def loss(matrix, matches):
        hardest_positive = matrix.masked_fill(~matches, -torch.inf).amax(1)
        hardest_negative = matrix.masked_fill(matches, torch.inf).amin(1)
        return (margin + hardest_positive[valid] - hardest_negative[valid]).relu().mean()

    return (loss(distances, positive) + loss(distances.t(), positive.t())) / 2

 
def compute_sdm_per(scores, pid, logit_scale, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1)) 
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    t2i_cosine_theta = scores
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.sum(i2t_loss, dim=1) + torch.sum(t2i_loss, dim=1)

    return loss

def compute_TRL_per(scores, pid, margin = 0.2, tau=0.02):       
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float().cuda()
    mask = 1 - labels

    alpha_1 =((scores/tau).exp()* labels / ((scores/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()
    alpha_2 = ((scores.t()/tau).exp()* labels / ((scores.t()/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()

    pos_1 = (alpha_1 * scores).sum(1)
    pos_2 = (alpha_2 * scores.t()).sum(1)

    neg_1 = (mask*scores).max(1)[0]
    neg_2 = (mask*scores.t()).max(1)[0]

    cost_1 = (margin + neg_1 - pos_1).clamp(min=0)
    cost_2 = (margin + neg_2 - pos_2).clamp(min=0)
    return cost_1 + cost_2

 
def compute_InfoNCE_per(scores, logit_scale):
    
    # cosine similarity as logits
    logits_per_image = logit_scale * scores
    logits_per_text = logits_per_image.t()

    p1 = F.softmax(logits_per_image, dim=1)
    p2 = F.softmax(logits_per_text, dim=1)

    loss = (- p1.diag().log() - p2.diag().log())/2    
    return loss

def compute_TAL_per(scores, pid, tau, margin):
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float().cuda()
    mask = 1 - labels

    alpha_i2t =((scores/tau).exp()* labels / ((scores/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()
    alpha_t2i = ((scores.t()/tau).exp()* labels / ((scores.t()/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()

    loss = (-  (alpha_i2t*scores).sum(1) + tau * ((scores / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)  \
        +  (-  (alpha_t2i*scores.t()).sum(1) + tau * ((scores.t() / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)
    
    return loss 

def compute_rbs(i_feats, t_feats, i_tsel_f, t_tsel_f, pid, label_hat=None, tau=0.02, margin=0.1, loss_type='TAL', logit_scale=50):

    loss_vsel, _ = compute_per_loss(i_feats, t_feats, pid, tau, margin, loss_type, logit_scale)
    loss_tsel, _ = compute_per_loss(i_tsel_f, t_tsel_f, pid, tau, margin, loss_type, logit_scale)

    loss_vsel = (label_hat*loss_vsel).sum()
    loss_tsel = (label_hat*loss_tsel).sum()
    
    if loss_type in ['TAL','TRL']:
        return loss_vsel, loss_tsel
    else:
        return loss_vsel/label_hat.sum(), loss_tsel/label_hat.sum() # mean

def compute_per_loss(image_features, text_features, pid, tau=0.02, margin=0.2, loss_type='TAL', logit_scale=50):
    
    # # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    scores = text_norm @ image_norm.t()

    if 'TAL' in loss_type:
        per_loss = compute_TAL_per(scores, pid, tau, margin=margin)
    elif 'TRL' in loss_type:
        per_loss = compute_TRL_per(scores, pid, tau=tau, margin=margin)
    elif 'InfoNCE' in loss_type:
        per_loss = compute_InfoNCE_per(scores, logit_scale)
    elif 'SDM' in loss_type:
        per_loss = compute_sdm_per(scores, pid, logit_scale)
    else:
        exit()

    return per_loss, scores.diag()



