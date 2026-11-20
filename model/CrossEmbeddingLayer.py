import math
import torch
import torch.nn as nn
import torch.nn.functional as F
 
def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def maxk_pool1d_var(x, dim, k, lengths):
    """https://github.com/woodfrog/vse_infty, thanks!"""
    results = list()
    lengths = list(lengths.cpu().numpy())
    lengths = [int(x) for x in lengths]
    for idx, length in enumerate(lengths):
        count = min(k, length)
        max_k_i = maxk(x[idx, :length, :], dim - 1, count).mean(dim - 1) if count else x[idx, 0] * 0
        results.append(max_k_i)
    results = torch.stack(results, dim=0)
    return results

def maxk_pool1d(x, dim, k):
    max_k = maxk(x, dim, k)
    return max_k.mean(dim)

def maxk(x, dim, k):
    index = x.topk(k, dim=dim)[1]
    return x.gather(dim, index)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN) from https://github.com/woodfrog/vse_infty, thanks!"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.bns = nn.ModuleList(nn.BatchNorm1d(k) for k in h + [output_dim])

    def forward(self, x):
        B, N, D = x.size()
        x = x.reshape(B * N, D)
        for i, (bn, layer) in enumerate(zip(self.bns, self.layers)):
            x = F.relu(bn(layer(x))) if i < self.num_layers - 1 else layer(x)
        x = x.view(B, N, self.output_dim)
        return x
 
class TexualEmbeddingLayer(nn.Module):
    def __init__(self, input_dim=512, embed_dim=1024,ratio=0.3, structural=False):
        super(TexualEmbeddingLayer, self).__init__()
        self.embed_dim= embed_dim
        self.linear = nn.Linear(input_dim, embed_dim)
        self.mlp = (nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU(), nn.Linear(512, embed_dim))
                    if structural else MLP(input_dim, 512, embed_dim, 2))
        self.ratio = ratio
        self.structural = structural
        if structural:
            if not 0 < ratio < 1:
                raise ValueError('select_ratio must be between zero and one')
            self.gcn_first = nn.Linear(input_dim, 512)
            self.gcn_second = nn.Linear(512, 1)
            self.keep_ratio = nn.Linear(input_dim, 1)
            nn.init.zeros_(self.keep_ratio.weight)
            nn.init.constant_(self.keep_ratio.bias, math.log(ratio / (1 - ratio)))
            self.visual_query = nn.Linear(embed_dim, embed_dim, bias=False)
            self.text_key = nn.Linear(input_dim, embed_dim, bias=False)

    def _structural_pool(self, features, text, atten, words, parents, visual):
        if words is None or parents is None:
            raise ValueError('Dependency inputs are required for DSGR')
        b, length, width = features.shape
        rows = torch.arange(b, device=features.device)
        eos = text.argmax(dim=-1)
        membership = F.one_hot(words.clamp_min(0), length).to(features.dtype) * words.ge(0)[..., None]
        word_count = membership.sum(dim=1)
        word_features = (membership.transpose(1, 2) @ features) / word_count.clamp_min(1)[..., None]
        adj = F.one_hot(parents.clamp_min(0), length).bool() & parents.ge(0)[..., None]
        active = word_count.gt(0)
        adj = (adj | adj.transpose(1, 2) | torch.eye(length, device=features.device, dtype=torch.bool))
        adj = (adj & active[:, :, None] & active[:, None, :]).to(features.dtype)
        norm = adj.sum(dim=-1).clamp_min(1).rsqrt()
        adj = adj * (norm[:, :, None] * norm[:, None, :])
        node_weights = self.gcn_second(adj @ self.gcn_first(adj @ word_features).relu()).sigmoid()
        token_weights = (membership @ node_weights).squeeze(-1)
        valid = words.ge(0)
        fallback = ~valid.any(dim=1)
        valid[rows[fallback], eos[fallback]] = True
        scores = (atten[rows, eos, :] * token_weights).masked_fill(~valid, -torch.inf)
        ranked, indices = scores.sort(dim=1, descending=True)
        counts = valid.sum(dim=1, keepdim=True)
        global_text = features[rows, eos]
        proportion = self.keep_ratio(global_text).sigmoid()
        ranks = torch.arange(length, device=features.device)[None, :]
        selected_mask = ranked.isfinite() & (ranks < (proportion * counts).ceil().clamp_min(1))
        strength = torch.sigmoid(8 * (proportion - ranks / counts))
        selected = F.normalize(features.gather(1, indices[..., None].expand(-1, -1, width)), dim=-1)
        density = (selected * F.normalize(global_text, dim=-1)[:, None, :]).sum(dim=-1)
        logits = density + ranked.clamp_min(1e-6).log() + strength.clamp_min(1e-6).log()
        logits = logits.masked_fill(~selected_mask, -torch.inf)
        pooled = (selected * logits.softmax(dim=1)[..., None]).sum(dim=1)
        if visual is None:
            return self.linear(pooled) + self.mlp(pooled), None
        query = self.visual_query(visual)
        key = self.text_key(selected)
        cross = (query @ key.transpose(1, 2) / math.sqrt(key.size(-1))).masked_fill(
            ~selected_mask[:, None, :], -torch.inf
        ).softmax(dim=-1).mean(dim=1)
        paired_weight = (logits + cross.clamp_min(1e-6).log()).softmax(dim=1)
        paired_pool = (selected * paired_weight[..., None]).sum(dim=1)
        outputs = torch.stack((pooled, paired_pool), dim=1)
        outputs = self.linear(outputs) + self.mlp(outputs)
        return outputs[:, 0], outputs[:, 1]

    def forward(self, features, text, atten, words=None, parents=None, visual=None):
        if self.structural:
            return self._structural_pool(features, text, atten, words, parents, visual)
        atten = atten.clone()

        mask =  ((text != 0) + 0)
        lengths = mask.sum(1).view(-1) - 2 
        k = max(1, int((atten.size(1)-2)*self.ratio))
        bs = features.size(0)
        atten[torch.arange(bs), :, text.argmax(dim=-1)] = -1 
        atten[torch.arange(bs), :, 0] = -1 # first token 
        atten = atten[torch.arange(bs), text.argmax(dim=-1), :] # 64 x 77
        atten = atten * mask
        
        atten_topK = atten.topk(dim=-1,k = k)[1].unsqueeze(-1).expand(bs,k,features.size(2)) # 64 x k x 512
        features = torch.gather(input=features,dim=1,index=atten_topK)  # 64 x k x 512
        features = l2norm(features, dim=-1)

        lengths = torch.Tensor([lengths[i] if lengths[i] < k else k for i in range(bs)]) # Keep at least K
        
  
        cap_emb = self.linear(features)
        features = self.mlp(features) + cap_emb
        features = maxk_pool1d_var(features, 1, 1, lengths.to(cap_emb.device))  # max 
        
        return features.float()

class VisualEmbeddingLayer(nn.Module):
    def __init__(self, input_dim=512, embed_dim=1024,ratio=0.3):
        super(VisualEmbeddingLayer, self).__init__()
        self.embed_dim= embed_dim
        self.linear = nn.Linear(input_dim, embed_dim)
        self.ratio = ratio
        self.fc = nn.Linear(input_dim, embed_dim)
        self.mlp = MLP(input_dim, embed_dim // 2, embed_dim, 2)
    
    def forward(self, base_features, atten, guide=None, return_components=False, visibility=None):
        if return_components:
            count = max(1, int(atten.size(1) * self.ratio))
            scores = atten if guide is None else atten * guide
            values, indices = scores.topk(count, dim=1)
            patches = base_features[:, 1:]
            selected = F.normalize(patches.gather(1, indices[..., None].expand(-1, -1, patches.size(2))), dim=-1)
            visible = 1 if visibility is None else visibility.gather(1, indices)[..., None]
            if self.training and selected.size(0) * selected.size(1) == 1:
                selected = selected.expand(-1, 2, -1)
                values = values.expand(-1, 2)
            enhanced = self.fc(selected) + visible * self.mlp(selected)
            weights = values.softmax(dim=1)[..., None]
            return (selected * weights).sum(dim=1), (enhanced * weights).sum(dim=1), enhanced
        atten = atten.clone()
        k = int((atten.size(1)-1)*self.ratio) # 192
        
        bs = base_features.size(0)
        atten[torch.arange(bs), :, 0] = -1 # CLS token   
        atten_topK = atten[:,0].topk(dim=-1,k = k)[1]
        
        atten_topK = atten_topK.unsqueeze(-1).expand(bs, k, base_features.size(2)) # 64 x k x 512
        base_features = torch.gather(input=base_features,dim=1,index=atten_topK)  # 64 x k x 512
        base_features = l2norm(base_features, dim=-1) 
        
        # Remove .half() to maintain float32 precision
        feat_lengths = torch.zeros(base_features.size(0)).to(base_features.device).float()
        feat_lengths[:] = base_features.size(1)
        
        features = self.fc(base_features)
        features = self.mlp(base_features) + features 
        features = maxk_pool1d_var(features, 1, 1, feat_lengths)   # max 
 
        return features.float()
