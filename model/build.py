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
        
        # Set current task based on loss names
        loss_names = getattr(args, 'loss_names', 'sdm+id+mlm')
        self.current_task = loss_names.upper()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 
        self.visul_emb_layer = VisualEmbeddingLayer(ratio=args.select_ratio)
        self.texual_emb_layer = TexualEmbeddingLayer(ratio=args.select_ratio)
        
        # Initialize Human Parsing Model
        self.human_parsing_model = None
        self.parsing_enabled = getattr(args, 'enable_parsing', True)
        self.parsing_model_path = getattr(args, 'parsing_model_path', 'exp-schp-201908301523-atr.pth')
        self.parsing_fusion_dim = getattr(args, 'parsing_fusion_dim', 256)
        self.parsing_weight = getattr(args, 'parsing_weight', 0.1)
        
        if self.parsing_enabled:
            self._init_human_parsing()
 
        if 'TAL' in self.current_task:
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
            self.parsing_fusion_clip = nn.Sequential(
                nn.Conv2d(ATR_SETTINGS['num_classes'], self.parsing_fusion_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.parsing_fusion_dim),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(self.parsing_fusion_dim, self.embed_dim)  # CLIP dimension (512)
            )
            
            # For TSEL features (1024 dim)
            self.parsing_fusion_tsel = nn.Sequential(
                nn.Conv2d(ATR_SETTINGS['num_classes'], self.parsing_fusion_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.parsing_fusion_dim),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(self.parsing_fusion_dim, self.visul_emb_layer.embed_dim)  # TSEL dimension (1024)
            )
            
            print("Human parsing model initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize human parsing model: {e}")
            self.parsing_enabled = False
            self.human_parsing_model = None

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
            # Resize to parsing model input size
            parsing_input = F.interpolate(parsing_input, size=ATR_SETTINGS['input_size'], mode='bilinear', align_corners=True)
            
            # Apply normalization (images are already normalized from dataloader)
            # We need to denormalize first, then apply parsing normalization
            # Standard ImageNet normalization used in HSRNet
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(images.device)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(images.device)
            
            # Denormalize
            parsing_input = parsing_input * std + mean
            
            # Apply parsing normalization
            parsing_mean = torch.tensor([0.406, 0.456, 0.485]).view(1, 3, 1, 1).to(images.device)
            parsing_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
            parsing_input = (parsing_input - parsing_mean) / parsing_std

            # Forward pass
            # Returns [[parsing_result, parsing_fea], [edge_result]]
            outputs = self.human_parsing_model(parsing_input)
            
            parsing_result = outputs[0][0] # Logits [N, 18, H, W]
            parsing_fea = outputs[0][1]    # Features [N, 256, H/4, W/4]
            edge_result = outputs[1][0]    # Edge Logits [N, 2, H, W]
            
            if return_all:
                return parsing_fea, parsing_result, edge_result
            
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

    def _semantic_reconstruction_parameter_free(self, parsing_fea, parsing_logits, edge_logits):
        """
        Zero-parameter semantic reconstruction module with occlusion gating
        Args:
            parsing_fea: [N, 256, H, W]
            parsing_logits: [N, 18, H, W]
            edge_logits: [N, 2, H, W]
        Returns:
            pooled_reconstructed_fea: [N, TargetDim]
        """
        # 1. Confidence and Mask
        # Region confidence
        region_probs = torch.softmax(parsing_logits, dim=1)
        region_conf, _ = torch.max(region_probs, dim=1) # [N, H, W]
        
        # Edge confidence (class 1 is edge)
        edge_probs = torch.softmax(edge_logits, dim=1)
        edge_conf = edge_probs[:, 1, :, :] # [N, H, W]
        
        # Occlusion Mask
        mask = (region_conf < getattr(self.args, 'region_conf_thres', 0.4))
        mask = mask.float().unsqueeze(1) # [N, 1, H, W]
        
        # 2. Gating
        occ_rates = mask.sum(dim=(1, 2, 3)) / mask[0].numel()
        gate = (occ_rates > getattr(self.args, 'occ_rate_thres', 0.1)).float().view(-1, 1, 1, 1)
        
        # If no reconstruction needed for batch
        if gate.sum() == 0:
            # Just pool and return
            pooled_fea = F.adaptive_avg_pool2d(parsing_fea, (1, 1)).flatten(1)
            
            # Pad to match target dim (512 for CLIP)
            target_dim = 512
            if pooled_fea.shape[1] < target_dim:
                padding = torch.zeros(pooled_fea.shape[0], target_dim - pooled_fea.shape[1]).to(parsing_fea.device)
                pooled_fea = torch.cat([pooled_fea, padding], dim=1)
                
            return pooled_fea, torch.tensor(0.0).to(parsing_fea.device)

        # 3. Reconstruction
        f_noc = parsing_fea * (1 - mask)
        f_occ = parsing_fea * mask
        
        # Global context from non-occluded regions
        noc_area = (1 - mask).sum(dim=(2, 3), keepdim=True) + 1e-6
        f_context_global = f_noc.sum(dim=(2, 3), keepdim=True) / noc_area
        f_ctx_attn = f_context_global.expand_as(parsing_fea)
        
        f_re = f_ctx_attn * mask
        
        # Edge enhancement
        edge_weight = edge_conf.unsqueeze(1)
        f_oe = f_re * edge_weight * 0.1
        
        # Integration
        omega = occ_rates.view(-1, 1, 1, 1)
        f_reconstructed_part = omega * (f_re + f_oe) + (1 - omega) * f_occ
        f_candidate = f_noc + f_reconstructed_part
        
        f_final_spatial = gate * f_candidate + (1 - gate) * parsing_fea
        
        # 4. Loss
        recon_loss = torch.tensor(0.0).to(parsing_fea.device)
        if self.training and gate.sum() > 0:
            l_region = F.mse_loss(f_reconstructed_part, f_context_global.expand_as(f_reconstructed_part), reduction='none')
            l_region = (l_region * mask * gate).sum() / (mask * gate).sum().clamp(min=1)
            
            l_edge = (((f_final_spatial - f_context_global) ** 2) * edge_weight * gate).sum() / (gate.sum() * f_final_spatial.shape[1] * f_final_spatial.shape[2] * f_final_spatial.shape[3])
            
            recon_loss = (l_region + l_edge)
            
        # Pool and pad
        pooled_fea = F.adaptive_avg_pool2d(f_final_spatial, (1, 1)).flatten(1)
        target_dim = 512
        if pooled_fea.shape[1] < target_dim:
            padding = torch.zeros(pooled_fea.shape[0], target_dim - pooled_fea.shape[1]).to(parsing_fea.device)
            pooled_fea = torch.cat([pooled_fea, padding], dim=1)
            
        return pooled_fea, recon_loss

    def encode_image(self, image):
        x, _ = self.base_model.encode_image(image)
        image_features = x[:, 0, :].float()
        
        # Add human parsing features if enabled
        if self.parsing_enabled:
            parsing_features = self.get_human_parsing_features(image, feature_type='clip')
            if parsing_features is not None:
                # Fuse image features with parsing features
                image_features = image_features + self.parsing_weight * parsing_features
        
        return image_features
      
    def encode_text(self, text):
        x, _ = self.base_model.encode_text(text.long())
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def encode_image_tsel(self, image):
        x,atten_i = self.base_model.encode_image(image)
       
        i_tsel_f = self.visul_emb_layer(x.float(), atten_i.float())
        
     
        if self.parsing_enabled:
            parsing_features = self.get_human_parsing_features(image, feature_type='tsel')
            if parsing_features is not None:
          
                i_tsel_f = i_tsel_f + self.parsing_weight * parsing_features
        
        return i_tsel_f.float()
 
    def encode_text_tsel(self, text):
        x,atten_t = self.base_model.encode_text(text.long())
   
        t_tsel_f = self.texual_emb_layer(x.float(), text, atten_t.float())
        return t_tsel_f.float()

    def compute_per_loss(self, batch):
        images = batch['images']
        caption_ids = batch['caption_ids']
        
        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
   
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        
        if self.parsing_enabled:
          
            parsing_output = self.get_human_parsing_features(images, feature_type='clip', return_all=True)
            if parsing_output is not None:
                parsing_features, parsing_logits, edge_logits = parsing_output
                
            
                parsing_features, _ = self._semantic_reconstruction_parameter_free(
                    parsing_features, parsing_logits, edge_logits
                )
                
                i_feats = i_feats + self.parsing_weight * parsing_features


  
        i_tsel_f = self.visul_emb_layer(image_feats.float(), atten_i.float())
        t_tsel_f = self.texual_emb_layer(text_feats.float(), caption_ids, atten_t.float())
        
    
        if self.parsing_enabled:
            parsing_features_tsel = self.get_human_parsing_features(images, feature_type='tsel')
            if parsing_features_tsel is not None:
               
                i_tsel_f = i_tsel_f + self.parsing_weight * parsing_features_tsel

        lossA, simsA = objectives.compute_per_loss(i_feats, t_feats, batch['pids'], \
                                                    tau=self.args.tau, \
                                                    margin=self.args.margin, \
                                                    loss_type=self.loss_type, \
                                                    logit_scale=self.logit_scale)
        lossB, simsB = objectives.compute_per_loss(i_tsel_f, t_tsel_f, batch['pids'],\
                                                    tau=self.args.tau, \
                                                    margin=self.args.margin, \
                                                    loss_type=self.loss_type, \
                                                    logit_scale=self.logit_scale)
        
        return lossA.detach().cpu(), lossB.detach().cpu(), simsA, simsB

    def forward(self, batch):
        ret = dict()
        ret.update({'temperature': 1 / self.logit_scale})

        images = batch['images']
        caption_ids = batch['caption_ids']
        
        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
      
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()


        if self.parsing_enabled:
            parsing_output = self.get_human_parsing_features(images, feature_type='clip', return_all=True)
            if parsing_output is not None:
                parsing_features, parsing_logits, edge_logits = parsing_output
                
      
                parsing_features, recon_loss = self._semantic_reconstruction_parameter_free(
                    parsing_features, parsing_logits, edge_logits
                )
                
                i_feats = i_feats + self.parsing_weight * parsing_features
                
      
                recon_loss = recon_loss * self.args.lambda_reconstruct

                if 'recon_loss' not in ret:
                    ret['recon_loss'] = recon_loss
                else:
                    ret['recon_loss'] += recon_loss


        i_tsel_f = self.visul_emb_layer(image_feats.float(), atten_i.float())
        t_tsel_f = self.texual_emb_layer(text_feats.float(), caption_ids, atten_t.float())
        

        if self.parsing_enabled:
            parsing_features_tsel = self.get_human_parsing_features(images, feature_type='tsel')
            if parsing_features_tsel is not None:
                # Both i_tsel_f and parsing_features_tsel are 2D tensors [batch_size, embed_dim]
                i_tsel_f = i_tsel_f + self.parsing_weight * parsing_features_tsel
            
        label_hat = batch['label_hat'].to(i_feats.device) 
     
        loss1, loss2 = objectives.compute_rbs(i_feats, t_feats, i_tsel_f, t_tsel_f, batch['pids'], \
                                              label_hat=label_hat, margin=self.args.margin,tau=self.args.tau,\
                                                loss_type=self.loss_type,logit_scale=self.logit_scale)
        ret.update({'vsel_loss':loss1})
        ret.update({'tsel_loss':loss2})
        ret.update({'id_loss': torch.tensor(0.0).to(i_feats.device)})  
        ret.update({'img_acc': torch.tensor(0.0).to(i_feats.device)})  
        ret.update({'txt_acc': torch.tensor(0.0).to(i_feats.device)})  
  
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
