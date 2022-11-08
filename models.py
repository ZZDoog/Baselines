import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.builders import RecurrentEncoderBuilder
from fast_transformers.masking import TriangularCausalMask

class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents):
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss  # [B x D x H x W]

class ResidualLayer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                kernel_size=3, padding=1, bias=False),
                                      nn.ReLU(True),
                                      nn.Conv2d(out_channels, out_channels,
                                                kernel_size=1, bias=False))

    def forward(self, input):
        return input + self.resblock(input)


class VQVAE(nn.modules):

    def __init__(self, n_token, codebook_szie, codebook_dim, beta):
        # n_token: compound word 中各个属性的种类数量
        # codebook_size: 离散化后codebook中向量的个数
        # codebook_dim: 离散化后codebook中每个向量的维度
        # beta: 计算离散化loss的一个超参数
        super(VQVAE, self).__init__()

        self.d_model = 512
        self.codebook_size = codebook_szie
        self.codebook_dim = codebook_dim
        self.beta = beta

        # 将CompoundWord中的各个元素词嵌入
        self.emb_sizes = [128, 256, 64, 32, 512, 128, 128]

        self.word_emb_tempo     = nn.Embedding(self.n_token[0], self.emb_sizes[0])
        self.word_emb_chord     = nn.Embedding(self.n_token[1], self.emb_sizes[1])
        self.word_emb_barbeat   = nn.Embedding(self.n_token[2], self.emb_sizes[2])
        self.word_emb_type      = nn.Embedding(self.n_token[3], self.emb_sizes[3])
        self.word_emb_pitch     = nn.Embedding(self.n_token[4], self.emb_sizes[4])
        self.word_emb_duration  = nn.Embedding(self.n_token[5], self.emb_sizes[5])
        self.word_emb_velocity  = nn.Embedding(self.n_token[6], self.emb_sizes[6])

        self.input_linear = nn.Linear(np.sum(self.emb_sizes), self.d_model)

        self.proj_tempo    = nn.Linear(self.d_model, self.n_token[0])        
        self.proj_chord    = nn.Linear(self.d_model, self.n_token[1])
        self.proj_barbeat  = nn.Linear(self.d_model, self.n_token[2])
        self.proj_type     = nn.Linear(self.d_model, self.n_token[3])
        self.proj_pitch    = nn.Linear(self.d_model, self.n_token[4])
        self.proj_duration = nn.Linear(self.d_model, self.n_token[5])
        self.proj_velocity = nn.Linear(self.d_model, self.n_token[6])


        # Build Encoder
        self.encoder = nn.LSTM(input_size=self.d_model, hidden_size=self.d_model, 
                                bath_first=True)
        
        # The VectorQuantizer Layer
        self.vq_layer = VectorQuantizer(self.codebook_size,
                                        self.codebook_dim,
                                        self.beta)

        # Build Decoder
        self.decoder = nn.LSTM(input_size=self.d_model, hidden_size=self.d_model, 
                                bath_first=True)

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        """
        # embeddings
        emb_tempo =    self.word_emb_tempo(input[..., 0])
        emb_chord =    self.word_emb_chord(input[..., 1])
        emb_barbeat =  self.word_emb_barbeat(input[..., 2])
        emb_type =     self.word_emb_type(input[..., 3])
        emb_pitch =    self.word_emb_pitch(input[..., 4])
        emb_duration = self.word_emb_duration(input[..., 5])
        emb_velocity = self.word_emb_velocity(input[..., 6])
        
        embs = torch.cat(
            [
                emb_tempo,
                emb_chord,
                emb_barbeat,
                emb_type,
                emb_pitch,
                emb_duration,
                emb_velocity,
            ], dim=-1)
        
        emb_linear = self.input_linear(embs)
        encoder_output = self.encoder(emb_linear)

        return encoder_output

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        """
        result = self.decoder(z)
        return result

    def forward(self, input, **kwargs):
        encoding = self.encode(input)[0]
        quantized_inputs, vq_loss = self.vq_layer(encoding)
        return [self.decode(quantized_inputs), input, vq_loss]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        vq_loss = args[2]

        recons_loss = F.mse_loss(recons, input)

        loss = recons_loss + vq_loss
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'VQ_Loss':vq_loss}

    def sample(self,
               num_samples: int,
               current_device, **kwargs):
        raise Warning('VQVAE sampler is not implemented.')

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]