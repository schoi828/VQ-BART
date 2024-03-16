# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from .third_party.ive import ive
from torch.distributions import Categorical

def feed_forward(d_model, d_ff, ff_p=0.,layernorm=False):
    layers = [nn.Linear(d_model, d_ff), nn.ReLU()]
    if ff_p > 0:
        layers.append(nn.Dropout(ff_p))
    if layernorm:
        layers.append(nn.LayerNorm(d_model))
    return nn.Sequential(*layers)

def sample_gumbel(shape, eps=1e-10):
    U = torch.rand(shape, device="cuda")
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    g = sample_gumbel(logits.size())
    y = logits + g
    return F.softmax(y / temperature, dim=-1)


def calc_distance(z_continuous, codebook, dim_dict):
    z_continuous_flat = z_continuous.view(-1, dim_dict)
    distances = (torch.sum(z_continuous_flat**2, dim=1, keepdim=True) 
                + torch.sum(codebook**2, dim=1)
                - 2 * torch.matmul(z_continuous_flat, codebook.t()))

    return distances

def build_MLP(config,encoder_embed_dim,final_layernorm=True):
        mlp = []
        for i in range(config.MLPLayers-1):
            mlp.append(feed_forward(encoder_embed_dim,encoder_embed_dim,config.vqdropout,config.vqlayernorm))
        mlp.append(feed_forward(encoder_embed_dim,encoder_embed_dim,0,(final_layernorm and config.vqlayernorm)))

        return nn.Sequential(*mlp)

class VQLayer(nn.Module):

    def __init__(self,config):
        super().__init__()
        enc_dim = config.encoder.embed_dim
        self.datatype = torch.float16 if config.vqfp16 else torch.float32

        self.VQ = VectorQuantizer(config.codebook, config.emb_dim, config.comit_cost)#.to(self.datatype)
        self.encoder = build_MLP(config,enc_dim, not config.no_final_layernorm)#.to(self.datatype)
        self.decoder = build_MLP(config,enc_dim)#.to(self.datatype)
        self.fp16 = config.vqfp16

    def forward(self,x,vq_encode=False):
        
        #x = x.to(self.datatype)
        #print('ori_enc_out',torch.mean(x))
        z = self.encoder(x)
        ori = x.dtype
        #z = z.to(self.datatype)
        loss, quantized, info = self.VQ(z)
        if vq_encode:
            return quantized, loss, info
        #quantizied = quantizied.to(ori)
        out = self.decoder(quantized)
        #out = BaseModelOutput(last_hidden_state=out)
        return out, loss, info
    
    def encode(self,x):
        
        z = self.encoder(x)
        _ , quantized, _ = self.VQ(z)
        return quantized

    def quantizie(self,x):
            
        _, quantized, _ = self.VQ(x)

        return quantized

    def decode(self,x,quantize=False):
    
        #quantizied = quantizied.to(ori)
        if quantize:
            _, x, _ = self.VQ(x)
        out = self.decoder(x)
        #out = BaseModelOutput(last_hidden_state=out)
        return out

    def decode_indices(self,x,shape):
    
        #quantizied = quantizied.to(ori)
        x = self.VQ.get_codebook_entry(x, shape)
        out = self.decoder(x)
        #out = BaseModelOutput(last_hidden_state=out)
        return out

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

        print('######codebook shape############',self._embedding.weight.shape)
        
    def forward(self, inputs):
        bs = inputs.size(0)
        seq_len = inputs.size(1)
        dtype = inputs.dtype
        # convert inputs from BSC -> BSHE 
        inputs = inputs.view(bs,seq_len,-1,self._embedding_dim)
        input_shape = inputs.shape
        # Flatten input
        flat_input = inputs.flatten(0,2) #(b*seq_len*128,8)
        # Calculate distances: (x-y)^2
        #print('mean',torch.mean(flat_input))

        #(x-y)^2   (flatten,512)
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t())) 
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, dtype=dtype,device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        #(f)
        # Quantize and unflatten
        #print('input',encodings.dtype)
        #print('weight',self._embedding.weight.dtype)
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        #(flatten,dim)
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        #inputs <- encoder
        quantized = inputs + (quantized - inputs).detach() 
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BSHE TO BSC
        return loss, quantized.view(bs,seq_len,-1).contiguous(), (perplexity, encodings, encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        # TODO: check for more easy handling with nn.Embedding
        #print(indices.shape[0], self._num_embeddings,indices.device)
        min_encodings = torch.zeros(indices.shape[0], self._num_embeddings,device=indices.device)
        min_encodings.scatter_(1, indices[:], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self._embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape).contiguous()

            # reshape back to match original input shape
            #z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)



class SQVAE(nn.Module):
    def __init__(self, cfgs):
        super(SQVAE, self).__init__()
        enc_dim = cfgs.encoder.embed_dim
        self.datatype = torch.float16 if cfgs.vqfp16 else torch.float32
        self.encoder = build_MLP(cfgs,enc_dim, not cfgs.no_final_layernorm)#.to(self.datatype)
        self.decoder = build_MLP(cfgs,enc_dim)#.to(self.datatype)
        self.fp16 = cfgs.vqfp16

        self.param_var_q = "vmf" if cfgs.arch == "vmfbart_large" else "gaussian_1"
        
        # Codebook
        self.size_dict = cfgs.codebook
        self.dim_dict = cfgs.emb_dim
        self.codebook = nn.Parameter(torch.randn(self.size_dict, self.dim_dict))
        self.log_param_q_scalar = nn.Parameter(torch.tensor(cfgs.log_param_q_init))
        if self.param_var_q == "vmf":
            self.quantizer = VmfVectorQuantizer(
                self.size_dict, self.dim_dict, 1.0)
        else:
            self.quantizer = GaussianVectorQuantizer(
                self.size_dict, self.dim_dict, 1.0, self.param_var_q)
            
    
    def forward(self, x, flg_train=False, flg_quant_det=True):
        device=x.device
        # Encoding
        if self.training:
            flg_train = True
            flg_quant_det = False
        if self.param_var_q == "vmf":
            z_from_encoder = self.encoder(x)
            self.param_q = (self.log_param_q_scalar.exp() + torch.tensor([1.0], device="cuda"))
        else:
            if self.param_var_q == "gaussian_1":
                z_from_encoder = self.encoder(x)
                log_var_q = torch.tensor([0.0], device=device)
            else:
                z_from_encoder, log_var = self.encoder(x)
                if self.param_var_q == "gaussian_2":
                    log_var_q = log_var.mean(dim=(1,2,3), keepdim=True)
                elif self.param_var_q == "gaussian_3":
                    log_var_q = log_var.mean(dim=1, keepdim=True)
                elif self.param_var_q == "gaussian_4":
                    log_var_q = log_var
                else:
                    raise Exception("Undefined param_var_q")
            self.param_q = (log_var_q.exp() + self.log_param_q_scalar.exp())
        
        # Quantization
        z_quantized, loss_latent, perplexity = self.quantizer(
            z_from_encoder, self.param_q, self.codebook, flg_train, flg_quant_det)
        latents = dict(z_from_encoder=z_from_encoder, z_to_decoder=z_quantized)

        # Decoding
        x_reconst = self.decoder(z_quantized)
        
        return x_reconst, loss_latent, (perplexity, latents) 
    
    def _calc_loss(self):
        raise NotImplementedError()
    
    def encode(self,x):
        flg_train = False
        flg_quant_det = True
        z = self.encoder(x)
        z_from_encoder = self.encoder(x)
        log_var_q = torch.tensor([0.0], device=x.device)
        self.param_q = (log_var_q.exp() + self.log_param_q_scalar.exp())
        quantized, _,_ = self.quantizer(
        z_from_encoder, self.param_q, self.codebook, flg_train, flg_quant_det)
        return quantized

    def quantizie(self,x):
            
        _, quantized, _ = self.VQ(x)

        return quantized

    def decode(self,x,quantize=False):
        flg_train = False
        flg_quant_det = True
        if quantize:
            log_var_q = torch.tensor([0.0], device=x.device)
            self.param_q = (log_var_q.exp() + self.log_param_q_scalar.exp())
            x, _,_ = self.quantizer(
            x, self.param_q, self.codebook, flg_train, flg_quant_det)
        out = self.decoder(x)
        #out = BaseModelOutput(last_hidden_state=out)
        return out
        
class BaseQuantizer(nn.Module):
    def __init__(self, size_dict, dim_dict, temperature=0.5):
        super().__init__()
        self.size_dict = size_dict
        self.dim_dict = dim_dict
        self.temperature = temperature
    
    def forward(self, z_from_encoder, param_q, codebook, flg_train, flg_quant_det=False):
        raise NotImplementedError()
    
    def set_temperature(self, value):
        self.temperature = value
    
    def _calc_distance_bw_enc_codes(self):
        raise NotImplementedError()
    
    def _calc_distance_bw_enc_dec(self):
        raise NotImplementedError()

class VmfVectorQuantizer(BaseQuantizer):
    def __init__(self, size_dict, dim_dict, temperature=0.5):
        super(VmfVectorQuantizer, self).__init__(size_dict, dim_dict, temperature)
    
    def forward(self, z_from_encoder, kappa_q, codebook, flg_train=True, flg_quant_det=False):
        bs, seq_len = z_from_encoder.shape[:2]
        dtype = z_from_encoder.dtype
        device = z_from_encoder.device
        z_from_encoder_permuted = z_from_encoder.view(bs,seq_len,-1,self.dim_dict)
        permuted_shape = z_from_encoder_permuted.shape

        z_from_encoder_permuted = F.normalize(z_from_encoder_permuted, p=2.0, dim=3)

        codebook_norm = F.normalize(codebook, p=2.0, dim=1)

        logit = -self._calc_distance_bw_enc_codes(z_from_encoder_permuted, codebook_norm, kappa_q)
        probabilities = torch.softmax(logit, dim=-1)
        log_probabilities = torch.log_softmax(logit, dim=-1)
        # Quantization
        if flg_train:
            encodings = gumbel_softmax_sample(logit, self.temperature)
            z_quantized = torch.mm(encodings, codebook_norm).view(permuted_shape)
            avg_probs = torch.mean(probabilities.detach(), dim=0)
        else:
            if flg_quant_det:
                indices = torch.argmax(logit, dim=1).unsqueeze(1)
                encodings_hard = torch.zeros(indices.shape[0], self.size_dict, device=device)
                encodings_hard.scatter_(1, indices, 1)
                avg_probs = torch.mean(encodings_hard, dim=0)
            else:
                dist = Categorical(probabilities)
                indices = dist.sample().view(bs, seq_len)
                encodings_hard = F.one_hot(indices, num_classes=self.size_dict).type_as(codebook)
                avg_probs = torch.mean(probabilities, dim=0)
            z_quantized = torch.matmul(encodings_hard, codebook_norm).view(permuted_shape)

        z_to_decoder = z_quantized.view(bs,seq_len,-1).contiguous()

        # Latent loss
        kld_discrete = torch.sum(probabilities * log_probabilities, dim=(0,1)) / bs
        kld_continuous = self._calc_distance_bw_enc_dec(z_from_encoder, z_to_decoder, kappa_q).mean()        
        loss = kld_discrete + kld_continuous
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-7)))

        return z_to_decoder, loss, perplexity
 
    def _calc_distance_bw_enc_codes(self, z_from_encoder, codebook, kappa_q):
        z_from_encoder_flat = z_from_encoder.view(-1, self.dim_dict)
        distances = -kappa_q * torch.matmul(z_from_encoder_flat, codebook.t())

        return distances
    
    def _calc_distance_bw_enc_dec(self, x1, x2, weight):
        return torch.sum(x1 * (x1-x2) * weight, dim=(1,2))

class GaussianVectorQuantizer(BaseQuantizer):
    def __init__(self, size_dict, dim_dict, temperature=0.5, param_var_q="gaussian_1"):
        super(GaussianVectorQuantizer, self).__init__(size_dict, dim_dict, temperature)
        self.param_var_q = param_var_q
    
    def forward(self, z_from_encoder, var_q, codebook, flg_train=True, flg_quant_det=False):
        bs, seq_len = z_from_encoder.shape[:2]
        dtype = z_from_encoder.dtype
        device = z_from_encoder.device
        z_from_encoder_permuted = z_from_encoder.view(bs,seq_len,-1,self.dim_dict)
        permuted_shape = z_from_encoder_permuted.shape

        precision_q = 1. / torch.clamp(var_q, min=1e-10)

        logit = -self._calc_distance_bw_enc_codes(z_from_encoder_permuted, codebook, 0.5 * precision_q)
        probabilities = torch.softmax(logit, dim=-1)
        log_probabilities = torch.log_softmax(logit, dim=-1)
        
        # Quantization
        if flg_train:
            encodings = gumbel_softmax_sample(logit, self.temperature)
            z_quantized = torch.mm(encodings, codebook).view(permuted_shape)
            avg_probs = torch.mean(probabilities.detach(), dim=0)
        else:
            if flg_quant_det:
                indices = torch.argmax(logit, dim=1).unsqueeze(1)
                encodings_hard = torch.zeros(indices.shape[0], self.size_dict, device=device)
                encodings_hard.scatter_(1, indices, 1)
                avg_probs = torch.mean(encodings_hard, dim=0)
            else:
                dist = Categorical(probabilities)
                indices = dist.sample().view(bs, seq_len)
                encodings_hard = F.one_hot(indices, num_classes=self.size_dict).type_as(codebook)
                avg_probs = torch.mean(probabilities, dim=0)
            z_quantized = torch.matmul(encodings_hard, codebook).view(permuted_shape)
        z_to_decoder = z_quantized.view(bs,seq_len,-1).contiguous()
        
        # Latent loss
        kld_discrete = torch.sum(probabilities * log_probabilities, dim=(0,1)) / bs
        kld_continuous = self._calc_distance_bw_enc_dec(z_from_encoder, z_to_decoder, 0.5 * precision_q).mean()
        loss = kld_discrete + kld_continuous
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-7)))

        return z_to_decoder, loss, perplexity

    def _calc_distance_bw_enc_codes(self, z_from_encoder, codebook, weight):        
        if self.param_var_q == "gaussian_1":
            distances = weight * calc_distance(z_from_encoder, codebook, self.dim_dict)
        elif self.param_var_q == "gaussian_2":
            weight = weight.tile(1, 1, 8, 8).view(-1,1)
            distances = weight * calc_distance(z_from_encoder, codebook, self.dim_dict)
        elif self.param_var_q == "gaussian_3":
            weight = weight.view(-1,1)
            distances = weight * calc_distance(z_from_encoder, codebook, self.dim_dict)
        elif self.param_var_q == "gaussian_4":
            z_from_encoder_flat = z_from_encoder.view(-1, self.dim_dict).unsqueeze(2)
            codebook = codebook.t().unsqueeze(0)
            weight = weight.permute(0, 2, 3, 1).contiguous().view(-1, self.dim_dict).unsqueeze(2)
            distances = torch.sum(weight * ((z_from_encoder_flat - codebook) ** 2), dim=1)

        return distances
        
    def _calc_distance_bw_enc_dec(self, x1, x2, weight):
        return torch.sum((x1-x2)**2 * weight, dim=(1,2))