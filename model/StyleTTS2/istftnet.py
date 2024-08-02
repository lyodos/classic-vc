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
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils import remove_weight_norm

import math
import random
import numpy as np 
from scipy.signal import get_window

LRELU_SLOPE = 0.1

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

# 以下は styletts2 utils から

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

####

# Generator の定義中で 1 回使う。定義は hifigan.py にあるものと同じ

class AdaINResBlock1(torch.nn.Module):
    def __init__(
        self, 
        channels, 
        kernel_size = 3, 
        dilation = (1, 3, 5), 
        style_dim = 64,
    ):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)
        
        self.adain1 = nn.ModuleList([
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels),
        ])
        
        self.adain2 = nn.ModuleList([
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels),
        ])
        
        self.alpha1 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs1))])
        self.alpha2 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs2))])

    def forward(self, x, s):
        for c1, c2, n1, n2, a1, a2 in zip(self.convs1, self.convs2, self.adain1, self.adain2, self.alpha1, self.alpha2):
            xt = n1(x, s)
            xt = xt + (1 / a1) * (torch.sin(a1 * xt) ** 2)  # Snake1D
            xt = c1(xt)
            xt = n2(xt, s)
            xt = xt + (1 / a2) * (torch.sin(a2 * xt) ** 2)  # Snake1D
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


####

# Generator の定義中で 1 回使う、torch.stft のラッパー

class TorchSTFT(torch.nn.Module):
    def __init__(
        self, 
        filter_length = 800, 
        hop_length = 200, 
        win_length = 800, 
        window = 'hann',
    ):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.from_numpy(get_window(window, win_length, fftbins = True).astype(np.float32))

    def transform(self, input_data):
        forward_transform = torch.stft(
            input_data,
            self.filter_length, 
            self.hop_length, 
            self.win_length, 
            window = self.window.to(input_data.device),
            return_complex = True,
        )
        return torch.abs(forward_transform), torch.angle(forward_transform)

    def inverse(self, magnitude, phase):
        inverse_transform = torch.istft(
            magnitude * torch.exp(phase * 1j),
            self.filter_length, 
            self.hop_length, 
            self.win_length, 
            window = self.window.to(magnitude.device),
        )
        return inverse_transform.unsqueeze(-2)  # unsqueeze to stay consistent with conv_transpose1d implementation

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction

####

# padDiff は以下の SineGen の中で使うが、実は使用箇所がコメントアウトされている
def padDiff(x):
    return F.pad(F.pad(x, (0,0,-1,1), 'constant', 0) - x, (0,0,0,-1), 'constant', 0)


# Definition of sine generator
# 定義は hifigan.py にあるものと同じ

class SineGen(torch.nn.Module):
    def __init__(
        self, 
        samp_rate: float, # sampling rate in Hz
        upsample_scale, 
        harmonic_num: int = 0, # number of harmonic overtones (default 0)
        sine_amp: float = 0.1, # amplitude of sine-wavefrom (default 0.1)
        noise_std: float = 0.003, # std of Gaussian noise (default 0.003)
        voiced_threshold = 0, # F0 threshold for U/V classification (default 0)
        flag_for_pulse = False, # this SinGen is used inside PulseGen (default False)
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

        self.flag_for_pulse = flag_for_pulse
        self.upsample_scale = upsample_scale
        # Note: when flag_for_pulse is True, the first time step of a voiced segment is always sin(np.pi) or cos(0)

    def _f02uv(self, f0):
        # generate uv signal
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv

    def _f02sine(self, f0_values):
        """ f0_values: (batchsize, length, dim)
            where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The interger part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1

        # initial phase noise (no noise for fundamental component)
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device = f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        if not self.flag_for_pulse:
#             # for normal case

#             # To prevent torch.cumsum numerical overflow,
#             # it is necessary to add -1 whenever \sum_k=1^n rad_value_k > 1.
#             # Buffer tmp_over_one_idx indicates the time step to add -1.
#             # This will not change F0 of sine because (x-1) * 2*pi = x * 2*pi
#             tmp_over_one = torch.cumsum(rad_values, 1) % 1
#             tmp_over_one_idx = (padDiff(tmp_over_one)) < 0
#             cumsum_shift = torch.zeros_like(rad_values)
#             cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

#             phase = torch.cumsum(rad_values, dim=1) * 2 * np.pi
            rad_values = torch.nn.functional.interpolate(rad_values.transpose(1, 2), 
                                                         scale_factor=1/self.upsample_scale, 
                                                         mode="linear").transpose(1, 2)
    
#             tmp_over_one = torch.cumsum(rad_values, 1) % 1
#             tmp_over_one_idx = (padDiff(tmp_over_one)) < 0
#             cumsum_shift = torch.zeros_like(rad_values)
#             cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0
    
            phase = torch.cumsum(rad_values, dim=1) * 2 * np.pi
            phase = torch.nn.functional.interpolate(
                phase.transpose(1, 2) * self.upsample_scale, 
                scale_factor = self.upsample_scale, 
                mode = "linear",
            ).transpose(1, 2)
            sines = torch.sin(phase)
        else:
            # If necessary, make sure that the first time step of every
            # voiced segments is sin(pi) or cos(0)
            # This is used for pulse-train generation

            # identify the last time step in unvoiced segments
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)

            # get the instantanouse phase
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            # different batch needs to be processed differently
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                # stores the accumulation of i.phase within
                # each voiced segments
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum
            # rad_values - tmp_cumsum: remove the accumulation of i.phase within the previous voiced segment.
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)
            sines = torch.cos(i_phase * 2 * np.pi) # get the sines
        return sines


    def forward(self, f0):
        """ sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim,
                             device=f0.device)
        # fundamental component
        fn = torch.multiply(f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device))

        # generate sine waveforms
        sine_waves = self._f02sine(fn) * self.sine_amp

        # generate uv signal
        # uv = torch.ones(f0.shape)
        # uv = uv * (f0 > self.voiced_threshold)
        uv = self._f02uv(f0)

        # noise: for unvoiced should be similar to sine_amp
        #        std = self.sine_amp/3 -> max value ~ self.sine_amp
        # .       for voiced regions is self.noise_std
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)

        # first: set the unvoiced part to 0 by uv
        # then: additive noise
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise

####

# Generator の定義中で 1 回使う
# 定義は hifigan.py にあるものと同じ

class SourceModuleHnNSF(torch.nn.Module):
    def __init__(
        self, 
        sampling_rate: float, # sampling_rate in Hz
        upsample_scale, 
        harmonic_num = 0, # number of harmonic above F0 (default: 0)
        sine_amp = 0.1, # amplitude of sine source signal (default: 0.1)
        add_noise_std = 0.003, #std of additive Gaussian noise (default: 0.003)
        voiced_threshod = 0, # threhold to set U/V given F0 (default: 0)
    ):
        super().__init__()
        
        self.sine_amp = sine_amp # note that amplitude of noise in unvoiced is decided by sine_amp
        self.noise_std = add_noise_std
        
        # to produce sine waveforms
        self.l_sin_gen = SineGen(
            sampling_rate, 
            upsample_scale, 
            harmonic_num,
            sine_amp, 
            add_noise_std, 
            voiced_threshod,
        )
        
        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(
        self, 
        x, # F0_sampled でありサイズは (batchsize, length, 1)
    ):
        # source for harmonic branch
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs)) # (batchsize, length, 1)

        # source for noise branch, in the same shape as uv
        noise = torch.randn_like(uv) * self.sine_amp / 3 # (batchsize, length 1)
        return sine_merge, noise, uv # uv: (batchsize, length, 1)


####

# ここから下の構造は HiFi-GAN 版の同名クラスと異なってくる

# さらに順伝播で upsample するときの処理を、少しだけ StyleTTS 2 時点での実装からも変えてあるので注意。
# upsample の層が深くなる（3 層以上）と、元実装では時間系列長が合わなくなり、残差を足し合わせる処理で止まってしまっていた。
# 下記のコードではその部分を修正してある。ただし実際には istftnet で層を深くしても性能は上がりにくいようだ。

class Generator(torch.nn.Module):
    def __init__(
        self, 
        sampling_rate: int, 
        style_dim, 
        resblock_kernel_sizes, 
        upsample_rates, 
        upsample_initial_channel, 
        resblock_dilation_sizes, 
        upsample_kernel_sizes, 
        gen_istft_n_fft, # HiFi-GAN 側の Generator にない引数
        gen_istft_hop_size, # HiFi-GAN 側の Generator にない引数
        harmonic_num: int = 8, # Sine で予めでっち上げる倍音の数。24k モデルでは 8、もっと上のモデルでは 16
    ):
        super().__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        resblock = AdaINResBlock1

        self.m_source = SourceModuleHnNSF(
            sampling_rate = sampling_rate,
            upsample_scale = np.prod(upsample_rates) * gen_istft_hop_size,
            harmonic_num = harmonic_num, 
            voiced_threshod = 10,
        )
        
        self.f0_upsamp = torch.nn.Upsample(scale_factor = np.prod(upsample_rates) * gen_istft_hop_size)
        self.noise_convs = nn.ModuleList()
        self.noise_res = nn.ModuleList()
        
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i), 
                        upsample_initial_channel // (2**(i + 1)),
                        k, 
                        u, 
                        padding = (k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes,resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d, style_dim))
                
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            
            if i + 1 < len(upsample_rates):  #
                stride_f0 = np.prod(upsample_rates[i + 1:])
                self.noise_convs.append(Conv1d(
                    gen_istft_n_fft + 2, c_cur, kernel_size=stride_f0 * 2, stride=stride_f0, padding=(stride_f0+1) // 2))
                self.noise_res.append(resblock(c_cur, 7, [1,3,5], style_dim))
            else:
                self.noise_convs.append(Conv1d(gen_istft_n_fft + 2, c_cur, kernel_size=1))
                self.noise_res.append(resblock(c_cur, 11, [1,3,5], style_dim))
        
        self.post_n_fft = gen_istft_n_fft
        self.conv_post = weight_norm(Conv1d(ch, self.post_n_fft + 2, 7, 1, padding = 3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))
        self.stft = TorchSTFT(
            filter_length = gen_istft_n_fft, 
            hop_length = gen_istft_hop_size, 
            win_length = gen_istft_n_fft,
        )
    
    
    # 順伝播には音素系列 x の他に s こと長さ情報、ピッチ f0 が必要。
    def forward(
        self, 
        x, 
        s, 
        f0,
    ):
        with torch.no_grad():
            f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t

            har_source, noi_source, uv = self.m_source(f0)
            har_source = har_source.transpose(1, 2).squeeze(1)
            
            har_spec, har_phase = self.stft.transform(har_source)
            har = torch.cat([har_spec, har_phase], dim=1)
        
        # 0 1: torch.Size([1, 512, 676]) torch.Size([1, 256, 6760])
        # 0 1.5: torch.Size([1, 256, 6760])
        # 0 2: torch.Size([1, 256, 6760]) torch.Size([1, 256, 6760])
        # 1 1: torch.Size([1, 256, 6760]) torch.Size([1, 128, 40561])
        # 1 1.5: torch.Size([1, 128, 40560])
        # 1 2: torch.Size([1, 128, 40561]) torch.Size([1, 128, 40561])

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x_source = self.noise_convs[i](har)
            x_source = self.noise_res[i](x_source, s)
#            print(i, "1:", x.shape, x_source.shape) 

            x = self.ups[i](x)
#            print(i, "1.5:", x.shape)

            # 2 回以上回すと、 ups して得た x の最終次元のサイズが 1 つ短くなってしまう（x_source 側が正しい）
            # なので x の末尾を 1 つ足す。元実装は「最終段で」という条件だが、任意の ups 回数だと 2 回に 1 回必要。
#            if i == self.num_upsamples - 1:
            if x.size(2) < x_source.size(2):
                x = self.reflection_pad(x)
            if x.size(2) > x_source.size(2):
                x = x[:, :, :x_source.size(2)]
#            print(i, "2:", x.shape, x_source.shape)
            x = x + x_source
#            x = x[:, :, :x_source.size(2)] + x_source

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x, s)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x, s)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        spec = torch.exp(x[:,:self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :])
        
        return self.stft.inverse(spec, phase)
    
    def fw_phase(self, x, s):
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x, s)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x, s)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.reflection_pad(x)
        x = self.conv_post(x)
        spec = torch.exp(x[:,:self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :])
        
        return spec, phase

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

####

# 下の Decoder 定義で使う
# こいつを使うたびに、style embedding の情報が混ざる。
# なお style_dim のデフォルトが 64 になっているが、実際の値は 128 が入るので注意

class AdainResBlk1d(nn.Module):
    def __init__(
        self, 
        dim_in, 
        dim_out, 
        style_dim = 64, # 実際は 128 なので注意
        actv = nn.LeakyReLU(0.2),
        upsample = 'none', 
        dropout_p = 0.0,
    ):
        super().__init__()
        self.actv = actv
        self.upsample_type = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out # dim_in と dim_out が異なる場合があるので、feature dim の投影を行う。
        self._build_weights(dim_in, dim_out, style_dim)
        self.dropout = nn.Dropout(dropout_p)
        
        if upsample == 'none':
            self.pool = nn.Identity() # Decoder では、最後の 1 回を除きこちら
        else:
            # 最後の 1 回では pooling として、サイズは dim_in のままだが前後の時点と情報が混合する
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
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias = False))

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(
        self, 
        x, 
        s,
    ):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.pool(x)
        x = self.conv1(self.dropout(x))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(
        self, 
        x, # (batch, 1026, n_frame)
        s, # (128,)
    ):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out

####

class UpSample1d(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        else:
            return F.interpolate(x, scale_factor = 2, mode = 'nearest')


####

# なお、sampling_rate は今回追加した。SourceModuleHnNSF() の呼び出しに存在したマジックナンバー 24000 を解消するため。

# ちなみに、Generator に渡す引数として gen_istft_n_fft, gen_istft_hop_size が入る以外は hifigan.py の Decoder と同じ。

class Decoder(nn.Module):
    def __init__(
        self, 
        sampling_rate = 24000,
        dim_in = 512, # args.hidden_dim, 
        style_dim = 64,  # args.style_dim, 実際は 128 なので注意
        resblock_kernel_sizes = [3, 7, 11],# args.decoder.resblock_kernel_sizes,
        resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]], # args.decoder.resblock_dilation_sizes,
        upsample_rates = [10, 6], # args.decoder.upsample_rates,
        upsample_initial_channel = 512, # args.decoder.upsample_initial_channel,
        upsample_kernel_sizes = [20, 12],  # args.decoder.upsample_kernel_sizes,
        gen_istft_n_fft = 20, 
        gen_istft_hop_size = 5, # upsample で水増しした時間に、さらに hop が掛かるので [10, 6], 5 だと 300 倍。
        harmonic_num: int = 8, # Sine で予めでっち上げる倍音の数。24k モデルでは 8、もっと上のモデルでは 16
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.style_dim = style_dim
        self.dim_in = dim_in
        self.encode = AdainResBlk1d(dim_in + 2, 1024, style_dim) # in, out, style_dim = 128

        # これは Generator の手前で適用されるネットワークで、発話内容、f0, energy, style を 4 回練り込む。
        # 手前にもう 1 つ self.encode として同じ構造があるので、計 5 回注入されることになる。
        self.decode = nn.ModuleList()
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 512, style_dim, upsample = True)) # 最後のみ upsample する

        # F0, N は長さが asr の 2 倍あるので、結合前にストライド量 2 の Conv1d に通して長さを半分にする
        self.F0_conv = weight_norm(nn.Conv1d(1, 1, kernel_size = 3, stride = 2, groups = 1, padding = 1))
        self.N_conv = weight_norm(nn.Conv1d(1, 1, kernel_size = 3, stride = 2, groups = 1, padding = 1))
        self.asr_res = nn.Sequential(
            weight_norm(nn.Conv1d(dim_in, 64, kernel_size = 1)),
        )
        self.generator = Generator(
            sampling_rate,
            style_dim, 
            resblock_kernel_sizes, 
            upsample_rates, 
            upsample_initial_channel, 
            resblock_dilation_sizes, 
            upsample_kernel_sizes, 
            gen_istft_n_fft, 
            gen_istft_hop_size,
            harmonic_num,
        )
    
    def forward(
        self, 
        asr: torch.Tensor,
        F0_curve: torch.Tensor, 
        N: torch.Tensor,
        s: torch.Tensor,
    ):
        # 訓練時のみ、F0 curve および energy curve のランダムなダウンサンプルが行われる。
        if self.training:
            F0_downlist = [0, 3, 7]
            F0_down = F0_downlist[random.randint(0, 2)]
            N_downlist = [0, 3, 7, 15]
            N_down = N_downlist[random.randint(0, 3)]
            
            if F0_down:
                F0_curve = nn.functional.conv1d(
                    F0_curve.unsqueeze(1), 
                    torch.ones(1, 1, F0_down).to(F0_curve.device), 
                    padding = F0_down // 2,
                ).squeeze(1) / F0_down
            if N_down:
                N = nn.functional.conv1d(
                    N.unsqueeze(1), 
                    torch.ones(1, 1, N_down).to(N.device), 
                    padding = N_down // 2,
                ).squeeze(1) / N_down
        
        # F0, N は予め Conv 層で長さを半分にしてから、
        F0 = self.F0_conv(F0_curve.unsqueeze(1))
        N = self.N_conv(N.unsqueeze(1))
        # asr と feature 次元で単純結合。
        x = torch.cat([asr, F0, N], axis = 1) # (batch, 512+1+1, 445)

        # その直後 style embedding と一緒に AdainResBlk1d に通して 1024 まで feature dim を上げる。
        x = self.encode(x, s) # (batch, 1024, 445)
        
        # さらに、asr だけを対象に feature size = 512 を 64 に落とす処理が入る。
        # これは以下の block 内で、x に発話内容を再度混ぜるために使用される。
        asr_res = self.asr_res(asr) # torch.Size([1, 64, 445])
        
        # 以下でさらに 4 回、 AdainResBlk1d で発話内容と pitch と energy と style を混合する。
        res = True
        for block in self.decode:
            if res:
                # 現在の特徴量 + 64 に落とした発話内容、pitch, energy を混合
                x = torch.cat([x, asr_res, F0, N], axis = 1) # (batch, 1024 + 64 + 1 + 1, n_frame)
            x = block(x, s) # self.decode の各要素こと AdainResBlk1d を経由すると 1090 → 1024 に戻る（最後のみ → 512）
            if block.upsample_type != "none":
                res = False

        # ここでようやく generator が走る。しかも、スタイルと F0 はここでも注入される。
        x = self.generator(x, s, F0_curve)
        
        return x
    