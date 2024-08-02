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

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, Conv2d
from torch.nn.utils import weight_norm, spectral_norm


LRELU_SLOPE = 0.1


# stft() は  SpecDiscriminator の forward で 1 回使う

def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    return torch.abs(torch.stft(x, fft_size, hop_size, win_length, window, return_complex = True)).transpose(2, 1)


class SpecDiscriminator(nn.Module):
    """docstring for Discriminator."""
    def __init__(
        self, 
        fft_size = 1024, 
        shift_size = 120, 
        win_length = 600, 
        window = "hann_window", 
        use_spectral_norm = False,
    ):
        super(SpecDiscriminator, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        self.discriminators = nn.ModuleList([
            norm_f(nn.Conv2d( 1, 32, kernel_size=(3, 9), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ])
        self.out = norm_f(nn.Conv2d(32, 1, 3, 1, 1))

    def forward(self, y):
        fmap = []
        y = y.squeeze(1)
        y = stft(y, self.fft_size, self.shift_size, self.win_length, self.window.to(y.get_device()))
        y = y.unsqueeze(1)
        for i, d in enumerate(self.discriminators):
            y = d(y)
            y = F.leaky_relu(y, LRELU_SLOPE)
            fmap.append(y)
        y = self.out(y)
        fmap.append(y)
        return torch.flatten(y, 1, -1), fmap

####

# こちらはユーザーが直接読み込む。

class MultiResSpecDiscriminator(torch.nn.Module):
    def __init__(
        self,
        fft_sizes = [1024, 2048, 512],
        hop_sizes = [120, 240, 50],
        win_lengths = [600, 1200, 240],
        window = "hann_window",
    ):
        super(MultiResSpecDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            SpecDiscriminator(fft_sizes[0], hop_sizes[0], win_lengths[0], window),
            SpecDiscriminator(fft_sizes[1], hop_sizes[1], win_lengths[1], window),
            SpecDiscriminator(fft_sizes[2], hop_sizes[2], win_lengths[2], window)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

####

# MultiPeriodDiscriminator は HFG の初期のものと同じ。

# 各 Discriminator が見分ける対象は本物 or 偽物の waveform なので 1D データである。長さは T あるとする (1, T)。

# まずこれを、(height, width) = (T/p, p) の 2D データに直す。ここから Conv2d に掛けていく。
# kernel_size = (5, 1), stride = (3, 1), padding=(int((5*1-1)/2)=2, 0), dilation=1
# なお、Conv2d の出力サイズは floor( (W_in + 2*padding -dilation*(kernel_size -1)-1) / stride + 1) なので、
# 出力 Height = floor( (H_in -1) / 3 + 1), Width = W_in である。
# つまり Height（T/p）だけが約 1/3 に縮小されていくとともに、feature の次元が 1 -> 32 -> 128 -> 512 -> 1024 にアップする。

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period # p の値を属性として保存
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm # 正則化は weight_norm が標準
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        # in_channels =    1, out_channels =   32, kernel_size = (5, 1), stride = (3, 1), padding=(2, 0), dilation = 1
        # in_channels =   32, out_channels =  128, kernel_size = (5, 1), stride = (3, 1), padding=(2, 0), dilation = 1
        # in_channels =  128, out_channels =  512, kernel_size = (5, 1), stride = (3, 1), padding=(2, 0), dilation = 1
        # in_channels =  512, out_channels = 1024, kernel_size = (5, 1), stride = (3, 1), padding=(2, 0), dilation = 1
        # in_channels = 1024, out_channels = 1024, kernel_size = (5, 1), stride =    1  , padding=(2, 0), dilation = 1
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))) # 事後処理で、1024 -> 1 まで特徴量を落とす。

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape # つまり [n_batch, c, n_sample] の次元を持つ。C はチャンネルのことだと思う。
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period) # [n_batch, c, ceil(n_sample/p), p]

        for l in self.convs:
            x = l(x) # それぞれの層について Conv2d して weight_norm してから、
            x = F.leaky_relu(x, LRELU_SLOPE) # Leaky ReLU で活性化する
            fmap.append(x) # 層ごとの feature map も保存しておく。
        x = self.conv_post(x)
        fmap.append(x) # 最終層の feature map も追加
        x = torch.flatten(x, 1, -1)

        return x, fmap # 返り値は特徴量 x（flatten したもの）に加え、各段階の feature map を保存したリスト


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(
        self, 
        periods = [2, 3, 5, 7, 11, 17], # 実際の訓練は [1, 2, 3, 5, 7, 11, 17, 23, 37] で入る
    ):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(i) for i in periods])
    
    # 順伝播を実行する際は y, y_hat すなわち、本来の wav および再変換された wav' のそれぞれの波形データが必要。
    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            # d はインスタンス変数として初期化した 2, 3, 5, 7, 11 各スケールの DiscriminatorP() 識別器。
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            # それぞれ y, y_hat を通して得た結果をリストに追加。
            y_d_rs.append(y_d_r) # 本物の判定結果
            fmap_rs.append(fmap_r) # 本物に対する各段階の特徴量マップ
            y_d_gs.append(y_d_g) # 贋物の判定結果
            fmap_gs.append(fmap_g) # 贋物に対する各段階の特徴量マップ

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs # この結果は後で loss function に通す。


# MPD は disjoint sample しか扱わないため、オーディオシーケンスの大域的な特徴は MSD で取り扱う。これも MelGAN 譲り。

####

class WavLMDiscriminator(nn.Module):
    """docstring for Discriminator."""
    def __init__(
        self, 
        slm_hidden = 768, 
        slm_layers = 13, 
        initial_channel = 64, 
        use_spectral_norm = False,
    ):
        super(WavLMDiscriminator, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.pre = norm_f(Conv1d(slm_hidden * slm_layers, initial_channel, 1, 1, padding=0))
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(initial_channel, initial_channel * 2, kernel_size=5, padding=2)),
            norm_f(nn.Conv1d(initial_channel * 2, initial_channel * 4, kernel_size=5, padding=2)),
            norm_f(nn.Conv1d(initial_channel * 4, initial_channel * 4, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(initial_channel * 4, 1, 3, 1, padding=1))
        
    def forward(self, x):
        x = self.pre(x)
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        x = torch.flatten(x, 1, -1)

        return x


