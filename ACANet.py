import torch
import torch.nn as nn
import math
from TDNN import TDNNBlock, BatchNorm1d

# This code is partially adapted from
# https://medium.com/@curttigges/the-annotated-perceiver-74752113eefb
# Code from Speechbrain is also used

class AsymmetricCrossAttention(nn.Module):
    """Basic decoder block used both for cross-attention and the latent transformer
    """
    def __init__(self, embed_dim, mlp_dim, n_heads, dropout=0.0):
        super().__init__()

        self.lnorm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads)

        self.lnorm2 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(mlp_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, q):
        # x will be of shape [PIXELS x BATCH_SIZE x EMBED_DIM]
        # q will be of shape [LATENT_DIM x BATCH_SIZE x EMBED_DIM] when this is
        # used for cross-attention; otherwise same as x

        # attention block
        x = self.lnorm1(x)
        out, _ = self.attn(query=q, key=x, value=x)
        # out will be of shape [LATENT_DIM x BATCH_SIZE x EMBED_DIM] after matmul
        # when used for cross-attention; otherwise same as x
        
        # first residual connection
        resid = out + q

        # dense block
        out = self.lnorm2(resid)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)
        out = self.drop(out)

        # second residual connection
        out = out + resid

        return out
    
class LatentTransformer(nn.Module):
    """Latent transformer module with n_layers count of decoders.
    """
    def __init__(self, embed_dim, mlp_dim, n_heads, dropout, n_layers):
        super().__init__()
        self.transformer = nn.ModuleList([
            AsymmetricCrossAttention(
                embed_dim=embed_dim, 
                mlp_dim=mlp_dim, 
                n_heads=n_heads, 
                dropout=dropout) 
            for l in range(n_layers)])
        self.ch_reduction = nn.Conv1d(embed_dim*(n_layers+1),embed_dim,1)

    def forward(self, l):
        
        L = l.clone()
        
        for trnfr in self.transformer:
            l = trnfr(l, l)
            L = torch.cat([L,l],2)

        L = L.permute(0,2,1)
        L = torch.nn.functional.relu(self.ch_reduction(L))
        L = L.permute(0,2,1)
            
        return L
    
class ACABlock(nn.Module):
    """Block consisting of one cross-attention layer and one latent transformer
    """
    def __init__(self, embed_dim, embed_reps, attn_mlp_dim, trnfr_mlp_dim, trnfr_heads, dropout, trnfr_layers):
        super().__init__()
        
        self.embed_reps = embed_reps
        
        self.cross_attention = nn.ModuleList([
            AsymmetricCrossAttention(
            embed_dim, attn_mlp_dim, n_heads=1, dropout=dropout)
            for _ in range(embed_reps)])

        self.latent_transformer = LatentTransformer(
            embed_dim, trnfr_mlp_dim, trnfr_heads, dropout, trnfr_layers)

    def forward(self, x, l):
        for ca in self.cross_attention:
            l = ca(x, l)

        l = self.latent_transformer(l)

        return l

#modified from speechbrain    
class ACANetPositionalEncoding1D(nn.Module):
    """Positional encoder for the pytorch transformer.
    
    This was modified from the original speechbrain implementation
    
    Arguments
    ---------
    d_model : int
        Representation dimensionality.
    max_len : int
        Max sequence length.
    
    Example
    -------
    
    >>> x = torch.randn(5, 512, 999) #Tensor Shape [Batch, Filters, Time]
    >>> enc = ACANetPositionalEncoding1D(512)
    >>> x = enc(x)
    """

    def __init__(self, d_model, max_len):
        super(ACANetPositionalEncoding1D, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Returns the encoded output.
        Arguments
        ---------
        x : torch.Tensor
            Tensor shape [B, N, L],
            where, B = Batchsize,
                   N = number of filters
                   L = time points
                   
        NOTE: self.pe was designed originally to accept Tensor shape [B, L, N]
        However, for speechbrain, we want Tensor shape [B, N, L]. Therefore, here we must permute.
        """
        x = x.permute(0,2,1)
        x = x + self.pe[: x.size(0), :]
        x = x.permute(0,2,1)

        return x

class ACANet(nn.Module):
    """ACANet Classification Network
    """
    def __init__(
        self, ch_in, latent_dim, embed_dim, embed_reps, attn_mlp_dim, trnfr_mlp_dim, trnfr_heads, 
        dropout, trnfr_layers, n_blocks, max_len,final_layer):
        super().__init__()
        
        self.ch_expansion = TDNNBlock(
                                    in_channels = ch_in,
                                    out_channels = embed_dim,
                                    kernel_size=1,
                                    dilation=1
                                    )
        
        # Initialize latent array
        self.latent = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.zeros((latent_dim, 1, embed_dim)), 
                mean=0, 
                std=0.02, 
                a=-2, 
                b=2))

        # Initialize embedding with position encoding
        self.embed = ACANetPositionalEncoding1D(d_model = embed_dim, max_len = max_len)

        # Initialize arbitrary number of blocks
        self.ACA_blocks = nn.ModuleList([
            ACABlock(
                embed_dim=embed_dim, #n_encoder_out
                embed_reps = embed_reps, #number of times to run the embedding cross attention
                attn_mlp_dim=attn_mlp_dim, #typical transformer MLP bottleneck dim, for the encoder
                trnfr_mlp_dim=trnfr_mlp_dim, #for the latent transformer
                trnfr_heads=trnfr_heads, #for the latent transformer
                dropout = dropout, 
                trnfr_layers = trnfr_layers) #number of layers in each block
            for b in range(n_blocks)])

        # Compress embed dimension
        #final_later determines the type. currently implemented is 'fc' and '1dE' and '1dL'
        self.fl = final_layer

        
        if self.fl == '1dE':
            self.ch_compression = nn.Conv1d(embed_dim,1,1)
            self.final_norm = BatchNorm1d(input_size = latent_dim)
        elif self.fl == '1dL':
            self.ch_compression = nn.Conv1d(latent_dim,1,1)
            self.final_norm = BatchNorm1d(input_size = embed_dim)
        elif self.fl == 'fc':
            self.ch_compression = nn.Linear(embed_dim*latent_dim,latent_dim)
            self.final_norm = BatchNorm1d(input_size = latent_dim)
        else:
            raise Exception("invalid final layer configuration")
            
        self.embed_reps = embed_reps
        

    def forward(self, x):
        #x should come in as [batch, time, filters]
        if len(x.shape)!=3:
            raise Exception("Check formatting of input")        
        
        #Expects x to be in BATCH FIRST format [Batch, Filters, Time]
        x = x.permute(0,2,1)
        x = self.ch_expansion(x) #perform channel expansion before anything else
        
        
        # First we expand our latent query matrix to size of batch
        batch_size = x.shape[0]
        input_length = x.shape[2]
        latent = self.latent.expand(-1, batch_size, -1)

        # Next, we pass the image through the embedding module to get flattened input
        x = self.embed(x)
        
        #Next, we permute the input x because for the ACA Blocks, x needs to be [time, batch, filters]
        x = x.permute(2,0,1)
        
        # Next, we iteratively pass the latent matrix and image embedding through
        # ACA blocks
        for pb in self.ACA_blocks:
            latent = pb(x, latent)
        #at this point the latent has dimensions: [Latent, batch, Emb]
        
        #two options for 1dconv: 
        # 1dE has the 1dconv run over the embedding so shape has to be [Batch, Emb, latent]
        # 1dL has the 1dconv run over the Latnets so shape has to be [Batch, Latent, Emb] or       
        if self.fl == '1dE':
            # [Batch, Emb, latent] Emb was originally the channel dimension anyway
            latent = latent.permute(1,2,0)
        elif self.fl == '1dL':
            latent = latent.permute(1,0,2) ##ooops. this does not actually work because the dimensions won't be correct. 
        elif self.fl == 'fc':
            latent = latent.permute(1,2,0) #does not matter as long as batch is put back into the first dimension
            latent = latent.flatten(1,2)
        out = self.ch_compression(latent)
        out = self.final_norm(out.squeeze()).unsqueeze(1)
        # Finally, we project the output to the number of target classes


        return out #reorder inputs back to [Batch, filters, time] format for the rest of speechbrain
    
if __name__ == '__main__':
    per = ACANet(
                ch_in = 80,
                latent_dim=192, 
                embed_dim=256,
                embed_reps=2,
                attn_mlp_dim=256, 
                trnfr_mlp_dim=256, 
                trnfr_heads=8, 
                dropout=0.2, 
                trnfr_layers=3, 
                n_blocks=2, 
                max_len = 10000,
                final_layer = '1dE'
               )
    x = torch.randn(5, 999,80)
    x = per(x)
    print(x.shape)
