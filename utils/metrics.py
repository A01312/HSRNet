import logging
import time
import torch
import torch.nn.functional as F
from utils.meter import AverageMeter
from prettytable import PrettyTable
import numpy as np

def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(similarity, k=max_rank, dim=1, largest=True, sorted=True)  # q * topk
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100
    return all_cmc, mAP, mINP, indices


def get_metrics(sims, qids, gids, name, get_mAP=True):
    qids = torch.tensor(qids)
    gids = torch.tensor(gids)
    if get_mAP:
        r, mAP, mINP, _ = rank(sims, qids, gids, get_mAP=get_mAP)
        r1, r5, r10 = r[0], r[4], r[9]
        return [name, 
                round(r1.item(), 2), 
                round(r5.item(), 2), 
                round(r10.item(), 2), 
                round(mAP.item(), 2), 
                round(mINP.item(), 2), 
                round((r1 + r5 + r10).item(), 2)]
    else:
        r, _ = rank(sims, qids, gids, get_mAP=get_mAP)
        r1, r5, r10 = r[0], r[4], r[9]
        return [name, 
                round(r1.item(), 2), 
                round(r5.item(), 2), 
                round(r10.item(), 2), 
                0, 
                0, 
                round((r1 + r5 + r10).item(), 2)]


class Evaluator():
    def __init__(self, img_loader, txt_loader):
        self.img_loader = img_loader # gallery
        self.txt_loader = txt_loader # query
        self.logger = logging.getLogger("HSRNet.eval")

    @torch.no_grad()
    def _compute_embeddings(self, model):
        device = next(model.parameters()).device
        qids, gids, qfeats, gfeats, local_q, local_g = [], [], [], [], [], []
        for pid, caption in self.txt_loader:
            inputs = (caption.to(device),) if torch.is_tensor(caption) else tuple(item.to(device) for item in caption)
            text, local = model.encode_text_branches_shared(*inputs)
            qids.extend(pid)
            qfeats.append(text.cpu())
            local_q.append(local.cpu())
        for pid, img in self.img_loader:
            feats, local = model.encode_image_branches_shared(img.to(device))
            gids.extend(pid)
            gfeats.append(feats.cpu())
            local_g.append(local.cpu())
        return tuple(torch.cat(item, dim=0) for item in (qfeats, gfeats, local_q, local_g)) + (qids, gids)

    def eval(self, model, i2t_metric=False):
        model.eval()
        qfeats, gfeats, vq_feats, vg_feats, qids, gids = self._compute_embeddings(model)
        qfeats = F.normalize(qfeats, p=2, dim=1) # text features
        gfeats = F.normalize(gfeats, p=2, dim=1) # image features
        similarity = qfeats @ gfeats.t()

        vq_feats = F.normalize(vq_feats, p=2, dim=1) # text features
        vg_feats = F.normalize(vg_feats, p=2, dim=1) # image features
        sims_tsel = vq_feats@vg_feats.t()


        sims_vsel = similarity  
        sims_tsel_calc = sims_tsel  
        sims_combined = (sims_vsel + sims_tsel) / 2  

        t = PrettyTable(['task', 'R1', 'R5', 'R10', 'mAP', 'mINP', 'rSum'])

        rs = get_metrics(sims_combined, qids, gids, 'VSEL+TSEL-t2i', True)
        r1 = rs[1]
        t.add_row(rs)

        self.logger.info('\n' + str(t))

        if i2t_metric:
            t = PrettyTable(['task', 'R1', 'R5', 'R10', 'mAP', 'mINP', 'rSum'])
            rs = get_metrics(sims_combined.t(), gids, qids, 'VSEL+TSEL-i2t', True)
            t.add_row(rs)
            self.logger.info('\n' + str(t))

        return r1
