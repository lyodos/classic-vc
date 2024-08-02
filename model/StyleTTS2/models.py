"""
MIT License

Copyright (c) 2023 Aaron (Yinghao) Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# Lyodos による改造部分も MIT License で配布

"""
The MIT License (MIT)

Copyright (c) 2024 Lyodos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm


####

# 以下の 2 つのクラスは、直後の ResBlk の定義で 1 回ずつ使う

class LearnedDownSample(nn.Module):
    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type

        if self.layer_type == 'none':
            self.conv = nn.Identity()
        elif self.layer_type == 'timepreserve':
            self.conv = spectral_norm(
                nn.Conv2d(dim_in, dim_in, kernel_size=(3, 1), stride=(2, 1), groups=dim_in, padding=(1, 0))
            )
        elif self.layer_type == 'half':
            self.conv = spectral_norm(
                nn.Conv2d(dim_in, dim_in, kernel_size=(3, 3), stride=(2, 2), groups=dim_in, padding=1)
            )
        else:
            raise RuntimeError('Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)
            
    def forward(self, x):
        return self.conv(x)


# StyleEncoder の定義だと downsaple = 'half' なので、jit.trace すると末尾次元が偶数か奇数かで影響が出る。
# 要するに avg_pool2d() が奇数でも正しく動作するように設定しないといけないのだが、面倒なのでまあいいや

class DownSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == 'half':
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool2d(x, 2)
        else:
            raise RuntimeError('Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


# 残差ブロックの定義。この後の StyleEncoder で 1 回だけ使う。このときは downsample = 'half' 決め打ちである。

class ResBlk(nn.Module):
    def __init__(
        self, 
        dim_in, 
        dim_out, 
        actv = nn.LeakyReLU(0.2),
        normalize = False, 
        downsample = 'none',
    ):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = DownSample(downsample)
        self.downsample_res = LearnedDownSample(downsample, dim_in)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = spectral_norm(nn.Conv2d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(dim_in, dim_out, 3, 1, 1))
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = spectral_norm(nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample_res(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance

####

# ここから StyleEncoder の定義。
# StyleTTS 2 では acoustic style encoder と prosodic style encoder の両方（いずれも入力は mel）が、以下の定義を使う。
# ClassicVC でも同じ基本構造を踏襲するが、HarmoF0 で得た 352 bins の log spectrogram から低音 1 オクターブを削った、
# 304 bins の log spectrogram を入力にする。これでも普通に動く。

# ちなみに ClassicVC には prosodic style encoder に相当する構造がないので、単に (acoustic) style encoder と呼んでいる。

class StyleEncoder(nn.Module):
    def __init__(
        self, 
        dim_in = 64, # StyleTTS 2 では 64 → ClassicVC では 304 に変更して使う
        style_dim = 128, 
        max_conv_dim = 512,
    ):
        super().__init__()
        blocks = []
        blocks += [spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1))]

        repeat_num = 4
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample = 'half')]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, dim_out, 5, 1, 0))]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.Linear(dim_out, style_dim)

    # 入力は (batch, 1, bins, frame) の time last を持つ
    # unsqueeze(1) して入れることに注意
    def forward(
        self, 
        x: torch.Tensor,
    ):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        s = self.unshared(h)
        return s



####

# ここから、ProsodyPredictor を定義する部分。

# AdaLayerNorm は DurationEncoderNoPack で 1 回だけ使う。

class AdaLayerNorm(nn.Module):
    def __init__(
        self, 
        style_dim: int = 128, 
        channels: int = 768, 
        eps = 1e-5,
    ):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.fc = nn.Linear(style_dim, channels*2)
        self.layer_norm = nn.LayerNorm(channels, eps=self.eps)
    
    def forward(
        self, 
        x: torch.Tensor, # (batch, time, d_model)
        s: torch.Tensor, # (batch, 128)
    ):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks = 2, dim = 1)
        gamma, beta = gamma.transpose(2, 1), beta.transpose(2, 1)
        
        # 最初はここで ONNX 化がエラーを返していた。
        # 理由は AdaLayerNorm で F.layer_norm(x, (self.channels,), eps=self.eps) を使っていたため。 
        # F が ONNX や Torchscript 化に対応していないため、書き出しに失敗する。
        x = self.layer_norm(x)
#        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        
        return x


# 以下はもともと StyleTTS 2 で使われている DurationEncoder だが、
# packing を使わないことで構造を単純化する改造を施してある

class DurationEncoderNoPack(nn.Module):
    def __init__(
        self, 
        sty_dim: int = 128, # 128
        d_model: int = 768, # 512
        nlayers: int = 3, # 3
        dropout: float = 0.1, # ただし LibriTTS の実験コードおよび配布重みでは 0.2
    ):
        super().__init__()
        self.sty_dim = sty_dim
        self.d_model = d_model
        self.lstms = nn.ModuleList()
        # ここから [LSTM, AdaLayerNorm, LSTM, AdaLayerNorm, LSTM, AdaLayerNorm]
        # LSTM で (batch, d_model + sty_dim, n_phoneme) -> (batch, d_model, n_phoneme) になり、AdaLayerNorm で戻る
        for _ in range(nlayers):
            self.lstms.append(
                nn.LSTM(
                    d_model + sty_dim, # 640
                    d_model // 2, # d_model // 2 = 256 だが双方向なので出力は d_model = 512 になる
                    num_layers = 1, 
                    batch_first = True, 
                    bidirectional = True, 
                    dropout = dropout,
                ) # 双方向なので、どの層も d_model + sty_dim で入って d_model で抜けることになる
            )
            self.lstms.append(AdaLayerNorm(sty_dim, d_model))  # AdaLayerNorm(x, s)
        self.dropout = dropout
        self.d_model = d_model
        self.sty_dim = sty_dim
        
    def forward(
        self, 
        x: torch.Tensor,       # h_bert で（変数名 d_en）であり、サイズは (batch, d_model = 512 or 768, n_phoneme) の time last
        style: torch.Tensor,   # 実際に合成に使う style (batch, sty_dim = 128)
    ):
        x = x.transpose(2, 1) # channel last に変形
        s = style.unsqueeze(1)  # (batch, 128) を (batch, 1, 128) に変形
        s = s.expand(-1, x.shape[1], -1) # (batch, n_phoneme, style_dim = 128) に変形
        x = torch.cat([x, s], dim = 2)  # (batch, n_phoneme, d_model + sty_dim)

        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x, style) # AdaLayerNorm は (batch, n_phoneme, d_model) の channel last が正式
                # この段階で x = (batch, n_phoneme, d_model), s = (n_phoneme, batch, sty_dim)
                x = torch.cat([x, s], axis = 2) # 最終的に (batch, n_phoneme, d_model + sty_dim) に
            else:
                block.flatten_parameters()
                x, _ = block(x) # 1 層の双方向 LSTM なので channel last にして入れる
                x = F.dropout(x, p = self.dropout, training = self.training) # 推論だけならここはカットしてもいい

        return x.permute(0, 2, 1) # d: torch.Size([batch, d_model + sty_dim, n_phoneme]) の time last


####

# UpSample1d は AdainResBlk1d でのみ 1 回使う。

class UpSample1d(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        else:
            return F.interpolate(x, scale_factor=2, mode='nearest')


# AdaIN1d は AdainResBlk1d でのみ 2 回使う。

class AdaIN1d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


# AdainResBlk1d は F0NPredictorAll で 6 回使う。

class AdainResBlk1d(nn.Module):
    def __init__(
        self, 
        dim_in, 
        dim_out, 
        style_dim = 64,  # 実際使うときは 128 なので注意
        actv = nn.LeakyReLU(0.2),
        upsample = 'none', 
        dropout_p = 0.0,
    ):
        super().__init__()
        self.actv = actv
        self.upsample_type = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)
        self.dropout = nn.Dropout(dropout_p)
        
        if upsample == 'none':
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(
                nn.ConvTranspose1d(
                    dim_in, dim_in, kernel_size = 3, stride = 2, groups = dim_in, padding = 1, output_padding = 1,
                )
            )
    
    def _build_weights(self, dim_in, dim_out, style_dim):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.pool(x)
        x = self.conv1(self.dropout(x))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out

####

# 以下が最終的に、f0 と energy を予測するネットワーク

class F0NPredictorAll(nn.Module):
    def __init__(
        self, 
        style_dim: int = 128, # 128
        d_hid: int = 768, # 512
        nlayers: int = 3, # 3
        dropout: float = 0.1, # ただし LibriTTS の実験コードおよび配布重みでは 0.2
    ):
        super().__init__() 
        self.text_encoder = DurationEncoderNoPack(
            sty_dim = style_dim, 
            d_model = d_hid,
            nlayers = nlayers, 
            dropout = dropout,
        )

        # self.shared は F0 と N の共通の前処理部分
        self.shared = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first = True, bidirectional = True)

        # F0 を予測するネットワーク
        self.F0 = nn.ModuleList()
        self.F0.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p = dropout))
        self.F0.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample = True, dropout_p = dropout))
        self.F0.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p = dropout))
        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0) # self.F0 の直後に通す

        # Energy を予測するネットワーク
        self.N = nn.ModuleList()
        self.N.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p = dropout))
        self.N.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample = True, dropout_p = dropout))
        self.N.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p = dropout))
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0) # self.N の直後に通す

    def forward(
        self, 
        x: torch.Tensor, 
        style: torch.Tensor, 
    ):
        x = self.text_encoder(x, style)
        # DurationEncoderNoPack は time last で入れて time last で返るが、
        # self.shared は LSTM なので channel last で入力するため transpose が必要。
        x, _ = self.shared(x.transpose(2, 1)) # 最初に 1 層の双方向 LSTM に通す処理（F0, N で共通）

        # ここから F0 予測が分岐して self.F0 を通過する
        F0 = x.transpose(2, 1) # 入力が 3D であることを仮定している
        for block in self.F0:
            F0 = block(F0, style)
        F0 = self.F0_proj(F0)

        # N 予測も分岐して self.N を通過する
        N = x.transpose(2, 1)
        for block in self.N:
            N = block(N, style)
        N = self.N_proj(N)

        return F0.squeeze(1), N.squeeze(1)


####

