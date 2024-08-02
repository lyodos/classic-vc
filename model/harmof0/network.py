"""
Adopted from Weixing Wei's https://github.com/WX-Wei/HarmoF0

The MIT License (MIT)

Copyright (c) 2022 Weiweixing

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


# LogSpectrogram は基本的に末尾次元しか演算対象にしない。また対象区間（1024 samples）に決め打ちでハミング窓を掛ける。
# なので、事前に移動窓で 1 区間分の信号を切り出してバッチ化しておく必要がある。

# Multiple Rate Dilated Convolution
# HarmoF0 の最初の block で使われる。畳み込み層そのものが dilated になっているわけではないことに注意。

class MRDConv(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, # 同じ入力 specgram を使い回しているので in と out が異なっていても大丈夫
        dilation_list = [0, 12, 19, 24, 28, 31, 34, 36],
    ):
        super().__init__()
        self.dilation_list = dilation_list
        conv_list = []
        for i in range(len(dilation_list)):
            conv_list.append(nn.Conv2d(in_channels, out_channels, kernel_size = [1, 1]))
        self.conv_list = nn.ModuleList(conv_list)
        
    def forward(
        self, # (batch, channel, time, bins) 
        specgram,
    ):
        dilation = self.dilation_list[0] # 0
        y = self.conv_list[0](specgram) # conv の最初の層に通す
        # 出力の最終次元を zero padding する。pad は y の次元数の半分以下で、今回は最終次元の冒頭 0 末尾 dilation
        y = F.pad(y, pad = [0, dilation]) 
        y = y[:, :, :, dilation:] # 最終次元の冒頭から dilation 分を削除。
        # この 2 行の操作は、つまり左シフトしてから巻き戻された箇所を 0 にしている。
        for i in range(1, len(self.conv_list)):
            dilation = self.dilation_list[i] # 12, 19, 24, 28, 31, 34, 36
            x = self.conv_list[i](specgram) # conv の 2 番目以降の層に通す
            x = x[:, :, :, dilation:] # 最終次元の冒頭から dilation 分を削除。今度は pad がないので短くなる
            bins_short = x.size(3) # 短くなった最終次元の長さ
            y[:, :, :, :bins_short] += x # y の冒頭部分に各回の conv 結果を貼り付けていく
        # つまりだんだん短くなる = 深い層ほど、低域だけに結果が加算される
        return y


def dila_conv_block( 
    in_channel: int, # 1, 32, 64, 128, 128, 64
    out_channel: int, # 32, 64, 128, 128, 64, 1
    bins_per_octave: int, # 48 = 12*4
    n_har: int, # 初回は 12 で以降は 3
    dilation_mode: str,
    dilation_rate: int, # 4 層とも 48
    dil_kernel_size: list, # 4 層とも [1, 3]
    kernel_size: list = [1, 3],
    padding: list = [0, 1],
):
    conv = nn.Conv2d(in_channel, out_channel, kernel_size = kernel_size, padding = padding)

    if dilation_mode == 'log_scale':
        # MRDC-Conv
        a = torch.log(torch.arange(1, n_har + 1)) / math.log(2**(1.0 / bins_per_octave))
        dilation_list = a.round().to(torch.int)
        conv_dil = MRDConv(out_channel, out_channel, dilation_list)
    else:
        # SD-Conv
        conv_dil = nn.Conv2d(
            out_channel, 
            out_channel, 
            kernel_size = dil_kernel_size, 
            padding = [0, dilation_rate], 
            dilation = [1, dilation_rate],
        )

    batch_norm = nn.BatchNorm2d(out_channel)

    return nn.Sequential(
        conv,
        nn.ReLU(),
        conv_dil,
        nn.ReLU(),
        batch_norm,
    )

# おそらく、LogSpectrogram の部分を改造すれば時間次元のあるサンプルを直接処理可能。
# 現時点での実装では、batch 次元が 1 でないと conv1 で止まる。

class HarmoF0(nn.Module):
    def __init__(self, 
            sample_rate = 16000, 
            n_freq = 512, 
            n_har = 12, 
            bins_per_octave = 12 * 4, 
            dilation_modes = ['log_scale', 'fixed', 'fixed', 'fixed'],
            dilation_rates = [48, 48, 48, 48],
            channels = [32, 64, 128, 128],
            fmin = 27.5,
            freq_bins = 88 * 4,
            dil_kernel_sizes: list = [[1, 3], [1, 3], [1, 3], [1, 3]],
        ):
        super().__init__()
        self.n_freq = n_freq # 512
        self.n_fft = n_freq * 2 # 1024
        self.freq_bins = freq_bins # 352
        bins = bins_per_octave

        # [b x 1 x T x 88*8] => [b x 32 x T x 88*4]
        self.block_1 = dila_conv_block(
            1, 
            channels[0], 
            bins, 
            n_har, 
            dilation_mode = dilation_modes[0], 
            dilation_rate = dilation_rates[0], 
            dil_kernel_size = dil_kernel_sizes[0], 
            kernel_size = [3, 3], 
            padding = [1, 1],
        )
        bins = bins // 2
        
        # => [b x 64 x T x 88*4]
        self.block_2 = dila_conv_block(
            channels[0], 
            channels[1], 
            bins, 
            3, 
            dilation_mode = dilation_modes[1], 
            dilation_rate = dilation_rates[1], 
            dil_kernel_size = dil_kernel_sizes[1], 
            kernel_size = [3, 3], 
            padding = [1, 1],
        )
        
        # => [b x 128 x T x 88*4]
        self.block_3 = dila_conv_block(
            channels[1], 
            channels[2], 
            bins, 
            3, 
            dilation_mode = dilation_modes[2], 
            dilation_rate = dilation_rates[2], 
            dil_kernel_size = dil_kernel_sizes[2], 
            kernel_size = [3, 3], 
            padding = [1, 1],
        )
        
        # => [b x 128 x T x 88*4]
        self.block_4 = dila_conv_block(
            channels[2], 
            channels[3], 
            bins, 
            3, 
            dilation_mode = dilation_modes[3], 
            dilation_rate = dilation_rates[3], 
            dil_kernel_size = dil_kernel_sizes[3], 
            kernel_size = [3, 3], 
            padding = [1, 1],
        )

        self.conv_5 = nn.Conv2d(channels[3],    channels[3]//2, kernel_size = [1, 1])
        self.conv_6 = nn.Conv2d(channels[3]//2, 1,              kernel_size = [1, 1])

    def forward(
        self, 
        spectrogram: torch.Tensor, # spectrogram (batch, num_frames, 352) # 元は wav の (batch, num_frames, 1024) だった
    ):
        assert spectrogram.size(0) == 1, "Size of the batch (dim 0) must be 1"
        x = spectrogram[None, :] # (batch = 1, num_frames, n_bins) ->  (1, batch = 1, num_frames, n_bins)

        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        # [b x 128 x T x 352] => [b x 64 x T x 352]
        x = self.conv_5(x)
        x = torch.relu(x)
        x = self.conv_6(x)
        x = torch.sigmoid(x)
        x = x.squeeze(1) # (n_frame, 1, 352) -> (n_frame, 352)
        return x # 元実装は log spectrogram も返していたが、ここで返す必要がない


####