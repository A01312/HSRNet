from model import objectives

from .CrossEmbeddingLayer import TexualEmbeddingLayer, VisualEmbeddingLayer
from .clip_model import build_CLIP_from_openai_pretrained, convert_weights
from .human_parsing import load_human_parsing_model, ATR_SETTINGS
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import os

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class HSRNet(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        
        self.current_task = args.loss_names.split('_', 1)[0].upper()
        tasks = set(self.current_task.split('+'))
        self.use_triplet = 'TRI' in tasks
        self.use_id = self.use_triplet and 'ID' in tasks
        self.use_recon = self.use_triplet and 'RE' in tasks
        self.use_dsgr = not getattr(args, 'disable_dsgr', False)
        self.use_hspr = getattr(args, 'enable_parsing', True) and not getattr(args, 'disable_hspr', False)
        if self.use_triplet and tasks not in ({'TRI'}, {'TRI', 'ID'}, {'TRI', 'ID', 'RE'}):
            raise ValueError(args.loss_names)

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 
        self.parsing_fusion_dim = getattr(args, 'parsing_fusion_dim', 256)
        local_dim = self.parsing_fusion_dim if self.use_dsgr else 1024
        self.visul_emb_layer = VisualEmbeddingLayer(self.embed_dim, local_dim, args.select_ratio)
        self.texual_emb_layer = TexualEmbeddingLayer(self.embed_dim, local_dim, args.select_ratio, self.use_dsgr)
        if self.use_dsgr:
            self.layer_weights = nn.Parameter(torch.zeros(base_cfg['vision_layers']))
            self.local_projection = nn.Linear(self.embed_dim, local_dim)
            if self.use_hspr:
                self.part_projection = nn.Linear(256, self.embed_dim)
        
        # Initialize Human Parsing Model
        self.human_parsing_model = None
        self.parsing_enabled = getattr(args, 'enable_parsing', True) and (self.use_hspr or self.use_dsgr)
        self.parsing_model_path = getattr(args, 'parsing_model_path', 'exp-schp-201908301523-atr.pth')
        self.parsing_weight = getattr(args, 'parsing_weight', 0.1)
        
        if self.parsing_enabled:
            self._init_human_parsing()
        if self.use_recon and not self.use_hspr:
            raise ValueError('RE requires SCHP')
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        self.classifier.requires_grad_(self.use_id)
 
        if self.use_triplet:
            loss_type = 'TRI'
        elif 'TAL' in self.current_task:
            loss_type = 'TAL'
        elif 'TRL' in self.current_task:
            loss_type = 'TRL'
        elif 'InfoNCE' in self.current_task:
            loss_type = 'InfoNCE'
        elif 'SDM' in self.current_task:
            loss_type = 'SDM'
        else:
            exit()
        self.loss_type = loss_type
        
    def _init_human_parsing(self):
        """Initialize human parsing model"""
        try:
            # Load human parsing model
            self.human_parsing_model = load_human_parsing_model(
                self.parsing_model_path, 
                num_classes=ATR_SETTINGS['num_classes']
            )
            
            # Ensure model is in float32 precision
            self.human_parsing_model = self.human_parsing_model.float()
            
            # Move to device if available
            if torch.cuda.is_available():
                self.human_parsing_model = self.human_parsing_model.cuda()
            
            # Ensure all parameters and buffers are float32 after moving to device
            for name, param in self.human_parsing_model.named_parameters():
                if param.dtype != torch.float32:
                    param.data = param.data.float()
                    
            for name, buffer in self.human_parsing_model.named_buffers():
                if buffer.dtype != torch.float32:
                    buffer.data = buffer.data.float()
            
            self.human_parsing_model.eval()
            
            # Freeze human parsing model parameters
            for param in self.human_parsing_model.parameters():
                param.requires_grad = False
                
            # Parsing feature fusion layers for different dimensions
            # For CLIP features (512 dim)
            if not self.use_dsgr:
                self.parsing_fusion_clip = nn.Sequential(
                    nn.Conv2d(256, self.parsing_fusion_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(self.parsing_fusion_dim),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(self.parsing_fusion_dim, self.embed_dim)  # CLIP dimension (512)
                )

            # For TSEL features (1024 dim)
            self.parsing_fusion_tsel = nn.Sequential(
                nn.Conv2d(256, self.parsing_fusion_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.parsing_fusion_dim),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(self.parsing_fusion_dim, self.visul_emb_layer.embed_dim)  # TSEL dimension (1024)
            )
            
            if self.use_hspr:
                self.cross_attention = nn.MultiheadAttention(256, 1, batch_first=True)
                self.context_projection = nn.Conv2d(1024, 256, 1)
                self.part_calibration = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
                self.edge_calibration = nn.Conv2d(2, 1, 3, padding=1)
                self.gamma = nn.Sequential(nn.Linear(1, 512), nn.ReLU(), nn.Linear(512, 1))
                self.omega = nn.Sequential(nn.Linear(1, 512), nn.ReLU(), nn.Linear(512, 1))
                self.region_calibration = nn.Conv2d(256, 256, 3, padding=1)
                self.region_completion = nn.Conv2d(256, 256, 3, padding=1)
                self.edge_completion = nn.Conv2d(1, 256, 3, padding=1)

        except Exception as e:
            print(f"Failed to initialize human parsing model: {e}")
            raise RuntimeError('Unable to load SCHP') from e

    def train(self, mode=True):
        super().train(mode)
        if self.human_parsing_model is not None:
            self.human_parsing_model.eval()
        return self

    def get_human_parsing_features(self, images, feature_type='clip', return_all=False):
        """Extract human parsing features from images
        
        Args:
            images: Input images
            feature_type: 'clip' for CLIP features (512 dim) or 'tsel' for TSEL features (1024 dim)
            return_all: Whether to return (features, parsing_logits, edge_logits) tuple
        """
        if not self.parsing_enabled or self.human_parsing_model is None:
            return None
            
        with torch.no_grad():
            # Ensure input tensor is float32
            parsing_input = images.float()
                
            # Preprocess images for human parsing
            # Apply normalization (images are already normalized from dataloader)
            # We need to denormalize first, then apply parsing normalization
            # Standard ImageNet normalization used in HSRNet
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(images.device)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(images.device)
            
            # Denormalize
            parsing_input = parsing_input * std + mean
            
            # Apply parsing normalization
            parsing_mean = torch.tensor([0.406, 0.456, 0.485]).view(1, 3, 1, 1).to(images.device)
            parsing_std = torch.tensor([0.225, 0.224, 0.229]).view(1, 3, 1, 1).to(images.device)
            parsing_input = (parsing_input[:, [2, 1, 0]] - parsing_mean) / parsing_std

            features = self.human_parsing_model(parsing_input, return_features=True)
            if return_all:
                return features
            parsing_fea = features[0]

            # Global Average Pooling
            pooled_fea = F.adaptive_avg_pool2d(parsing_fea, (1, 1)).flatten(1)
            
            # Project to match feature dimension
            if feature_type == 'clip':
                target_dim = 512
            else:
                target_dim = 1024
                
            if pooled_fea.shape[1] != target_dim:
                if pooled_fea.shape[1] < target_dim:
                    padding = torch.zeros(pooled_fea.shape[0], target_dim - pooled_fea.shape[1]).to(images.device)
                    final_fea = torch.cat([pooled_fea, padding], dim=1)
                else:
                    final_fea = pooled_fea[:, :target_dim]
            else:
                final_fea = pooled_fea
            
            return final_fea

    def _semantic_reconstruction(self, region, part_logits, edge_logits, context, compute_loss=False):
        parts = part_logits.softmax(dim=1)
        edge = edge_logits.softmax(dim=1)[:, 1:2]
        mask = ((parts.amax(dim=1, keepdim=True) < getattr(self.args, 'region_conf_thres', 0.5))
                & (edge < getattr(self.args, 'edge_conf_thres', 0.5))).to(region.dtype)
        visible_mask = 1 - mask
        visible = region * visible_mask
        ratio = mask.mean(dim=(2, 3))
        part_scores = parts[:, 1:]
        occluded_edge = edge * mask
        confidence = self.part_calibration(torch.stack(
            (part_scores, occluded_edge.expand_as(part_scores)), dim=-1
        )).squeeze(-1).sigmoid()
        body = confidence.amax(dim=1, keepdim=True) >= getattr(self.args, 'body_conf_thres', 0.5)
        dilated = F.max_pool2d(occluded_edge * body, 3, stride=1, padding=1)
        corrected = dilated.sigmoid() + self.edge_calibration(torch.cat((dilated, visible_mask), dim=1))
        edges = edge * visible_mask + corrected * mask
        calibrated = region + self.gamma(ratio).relu()[..., None, None] * self.region_calibration(region * edges)
        weights = torch.where(visible_mask.any(dim=(2, 3), keepdim=True), part_scores * visible_mask, part_scores)
        counts = weights.sum(dim=(2, 3)).clamp_min(1e-6)
        keys = (weights.flatten(2) @ calibrated.flatten(2).transpose(1, 2)) / counts[..., None]
        bias = ((weights * confidence).sum(dim=(2, 3)) / counts).clamp_min(1e-6).log()
        ctx = self.context_projection(context)
        query = ctx.flatten(2).transpose(1, 2)
        attn, _ = self.cross_attention(query, keys, keys,
                                       attn_mask=bias[:, None, :].expand(-1, query.size(1), -1),
                                       need_weights=False)
        attn = F.interpolate(attn.transpose(1, 2).reshape_as(ctx), size=region.shape[-2:], mode='nearest')
        restored = self.region_completion(attn) * mask
        omega = self.omega(ratio).sigmoid()[..., None, None]
        reconstructed = omega * restored + (1 - omega) * region
        structure = reconstructed + self.edge_completion(edges) * mask
        if not compute_loss:
            return structure, None, mask
        return structure, objectives.compute_reconstruction(reconstructed, visible, mask, edges, self.args.tau), mask

    def _image_embeddings(self, images, tokens, attention, compute_loss=False):
        global_features = tokens[:, 0].float()
        structure = loss = guide = visibility = None
        if self.parsing_enabled:
            parsed = self.get_human_parsing_features(images, return_all=True)
            if self.use_hspr:
                structure, loss, mask = self._semantic_reconstruction(*parsed, compute_loss=compute_loss)
            else:
                structure = parsed[0]
        if not self.use_dsgr:
            local = self.visul_emb_layer(tokens.float(), attention.float())
            if structure is not None:
                global_features = global_features + self.parsing_weight * self.parsing_fusion_clip(structure)
                local = local + self.parsing_weight * self.parsing_fusion_tsel(structure)
            return global_features, local, loss, None
        if self.use_hspr:
            parts = parsed[1].float().softmax(dim=1)[:, 1:].flatten(2)
            part_features = (parts @ structure.flatten(2).transpose(1, 2)) / parts.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            parts = F.normalize(self.part_projection(part_features), dim=-1)
            similarity = F.normalize(tokens[:, 1:].float(), dim=-1) @ parts.transpose(1, 2)
            guide = similarity.softmax(dim=-1).amax(dim=-1)
            grid = (self.base_model.visual.num_y, self.base_model.visual.num_x)
            visibility = F.adaptive_avg_pool2d(1 - mask, grid).flatten(1)
        attention = (attention.float() * self.layer_weights.softmax(dim=0)[None, :, None]).sum(dim=1)
        selected, enhanced, visual = self.visul_emb_layer(
            tokens.float(), attention, guide, return_components=True, visibility=visibility
        )
        fused = self.local_projection(selected) + enhanced
        if structure is not None:
            fused = fused + self.parsing_fusion_tsel(structure)
        return global_features, fused, loss, visual

    def encode_image_branches_shared(self, image):
        return self._image_embeddings(image, *self.base_model.encode_image(image, collect_attention=self.use_dsgr))[:2]

    def encode_image(self, image):
        return self.encode_image_branches_shared(image)[0]

    def encode_text(self, text):
        x, _ = self.base_model.encode_text(text.long())
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def encode_image_tsel(self, image):
        return self.encode_image_branches_shared(image)[1]

    def _text_embeddings(self, text, tokens, attention, words=None, parents=None, visual=None):
        rows = torch.arange(text.size(0), device=text.device)
        global_features = tokens[rows, text.argmax(dim=1)].float()
        local = self.texual_emb_layer(tokens.float(), text, attention.float(), words, parents, visual)
        return (global_features, *local) if self.use_dsgr else (global_features, local, None)

    def encode_text_branches_shared(self, text, words=None, parents=None):
        return self._text_embeddings(text, *self.base_model.encode_text(text.long()), words, parents)[:2]

    def encode_text_tsel(self, text, words=None, parents=None):
        return self.encode_text_branches_shared(text, words, parents)[1]

    def compute_per_loss(self, batch):
        images, captions = batch['images'], batch['caption_ids']
        i_tokens, i_attn, t_tokens, t_attn = self.base_model(images, captions, collect_attention=self.use_dsgr)
        i_feats, i_local, _, _ = self._image_embeddings(images, i_tokens, i_attn)
        t_feats, t_local, _ = self._text_embeddings(captions, t_tokens, t_attn, batch.get('words'), batch.get('parents'))
        kwargs = dict(tau=self.args.tau, margin=self.args.margin, loss_type=self.loss_type, logit_scale=self.logit_scale)
        loss_a, sim_a = objectives.compute_per_loss(i_feats, t_feats, batch['pids'], **kwargs)
        loss_b, sim_b = objectives.compute_per_loss(i_local, t_local, batch['pids'], **kwargs)
        return loss_a.detach().cpu(), loss_b.detach().cpu(), sim_a, sim_b

    def forward(self, batch):
        images, captions, pids = batch['images'], batch['caption_ids'], batch['pids']
        i_tokens, i_attn, t_tokens, t_attn = self.base_model(images, captions, collect_attention=self.use_dsgr)
        i_feats, i_local, reconstruction_loss, visual = self._image_embeddings(
            images, i_tokens, i_attn, compute_loss=self.training and self.use_recon
        )
        t_feats, t_local, paired_text = self._text_embeddings(
            captions, t_tokens, t_attn, batch.get('words'), batch.get('parents'),
            visual if self.training and self.use_triplet else None
        )
        if self.use_triplet:
            loss_a = objectives.cross_modal_triplet(i_feats, t_feats, pids, self.args.margin) / 2
            loss_b = objectives.cross_modal_triplet(i_local, t_local, pids, self.args.margin) / 2
            if paired_text is not None:
                loss_b = (loss_b + objectives.cross_modal_triplet(i_local, paired_text, pids, self.args.margin) / 2) / 2
        else:
            loss_a, loss_b = objectives.compute_rbs(
                i_feats, t_feats, i_local, t_local, pids,
                label_hat=batch['label_hat'].to(i_feats.device), margin=self.args.margin,
                tau=self.args.tau, loss_type=self.loss_type, logit_scale=self.logit_scale
            )
        ret = {'temperature': 1 / self.logit_scale, 'vsel_loss': loss_a, 'tsel_loss': loss_b}
        if self.use_id:
            image_logits, text_logits = self.classifier(i_feats), self.classifier(t_feats)
            ret['id_loss'] = (F.cross_entropy(image_logits, pids) + F.cross_entropy(text_logits, pids)) / 2
            ret['img_acc'] = (image_logits.argmax(1) == pids).float().mean()
            ret['txt_acc'] = (text_logits.argmax(1) == pids).float().mean()
        if reconstruction_loss is not None:
            ret['recon_loss'] = reconstruction_loss
        return ret


def build_model(args, num_classes=11003):
    model = HSRNet(args, num_classes)
    
    if hasattr(model, 'base_model'):
        convert_weights(model.base_model)
    
    if hasattr(model, 'human_parsing_model') and model.human_parsing_model is not None:
        model.human_parsing_model.float()
        
        for param in model.human_parsing_model.parameters():
            param.data = param.data.float()
        
        for buffer in model.human_parsing_model.buffers():
            buffer.data = buffer.data.float()
    
   
    if hasattr(model, 'visul_emb_layer'):
        model.visul_emb_layer.float()
        for param in model.visul_emb_layer.parameters():
            param.data = param.data.float()
    
    if hasattr(model, 'texual_emb_layer'):
        model.texual_emb_layer.float()
        for param in model.texual_emb_layer.parameters():
            param.data = param.data.float()
    
    return model
