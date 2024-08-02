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

import os
import glob
from typing import Union
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
import math

# 以下は train.py に元々存在するコードの移植。
# lin_one_cycle() はスケジューラの学習率を制御する関数。
# flatten_cfg() は訓練の全エポックが終了したときに hyper parameters を保存用に整形する。

def lin_one_cycle(startlr, maxlr, endlr, warmup_pct, total_iters, iters):
    """ 
    Linearly warms up from `startlr` to `maxlr` for `warmup_pct` fraction of `total_iters`, 
    and then linearly anneals down to `endlr` until the final iter.
    """
    warmup_iters = int(warmup_pct*total_iters)
    if iters < warmup_iters:
        # Warmup part
        m = (maxlr - startlr)/warmup_iters
        return m*iters + startlr
    else:
        m = (endlr - maxlr)/(total_iters - warmup_iters)
        c = endlr - total_iters*m
        return m*iters + c    


def flatten_cfg(cfg: Union[DictConfig, ListConfig]) -> dict:
    """ 
    Config をフラットな辞書形式になるよう再帰的に加工して、TensorBoard の `add_hparams` 関数に合致させる。
    """
    out_dict = {}
    if type(cfg) == ListConfig:
        cfg = DictConfig({f"[{i}]": v for i, v in enumerate(cfg)})

    for key in cfg:
        if type(getattr(cfg, key)) in (int, str, bool, float):
            out_dict[key] = getattr(cfg, key)
        elif type(getattr(cfg, key)) in [DictConfig, ListConfig]:
            out_dict = out_dict | {f"{key}{'.' if type(getattr(cfg, key)) == DictConfig else ''}{k}": v for k, v in flatten_cfg(getattr(cfg, key)).items()}
        else: raise AssertionError
    return out_dict


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????' + '.pt')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]



def plot_spectrogram(
    spectrogram,
    size = (7, 2.5),
    aspect = None,
    vmin = -2,
    vmax = 4,
    cmap = "inferno",
):
    import matplotlib
    matplotlib.use("Agg") # plt の前に指定
    import matplotlib.pylab as plt
    matplotlib.rcParams["font.family"] = "Noto Sans Mono"

    fig, ax = plt.subplots(figsize = size)
    im = ax.imshow(
        spectrogram, 
        aspect = "auto" if aspect is None else aspect, 
        origin = "lower", 
        interpolation = 'none',
        cmap = matplotlib.colormaps[cmap],
        vmin = vmin,
        vmax = vmax,
    )
#    plt.colorbar(im, ax = ax, orientation = 'horizontal', aspect = 50)
    fig.canvas.draw()
    plt.close()
    return fig

#### こちらは 

import numpy as np

import torch
import torch.nn.functional as F

# こちらは Hz 単位の周波数を、352 bins の one-hot feature map に戻す（ただし使用された形跡がない）
def hz_to_onehot(
    hz: torch.Tensor, 
    fmin = 27.5*2,
    fmax = 4186,
    freq_bins = 352, 
    bins_per_octave = 48,
):
    indices = ( torch.log((hz + 1e-7) / fmin) / np.log(2.0**(1.0 / bins_per_octave)) + 0.5 ).long()
    indices = torch.clip(indices, 0, freq_bins)
    return indices


def plot_spectrogram_harmof0(
    spectrogram,
    f0 = None,
    act = None,
    size = (7, 2.5),
    aspect = None,
    vmin = -50,
    vmax = 40,
    cmap = "inferno",
):
    import matplotlib
    import matplotlib.cm as cm
    matplotlib.use("Agg") # plt の前に指定
    import matplotlib.pylab as plt
    matplotlib.rcParams["font.family"] = "Noto Sans Mono"

    fig, ax = plt.subplots(figsize = size)
    im = ax.imshow(
        spectrogram, 
        aspect = "auto" if aspect is None else aspect, 
        origin = "lower", 
        interpolation = 'none',
        cmap = matplotlib.colormaps[cmap],
        vmin = vmin,
        vmax = vmax,
    )
    # HarmoF0 activation sequence
    if act is not None:
        ln_act = ax.plot(
            np.linspace(start = 0, stop = spectrogram.shape[-1], num = f0.shape[-1]), 
            300*act.numpy().squeeze(), 
            linewidth = 1, 
            linestyle = "dashed",
            color = cm.hsv(0.6),
        )
    
    # HarmoF0 pitch sequence
    if f0 is not None:
        ln_f0 = ax.plot(
            np.linspace(start = 0, stop = spectrogram.shape[-1], num = f0.shape[-1]), 
            hz_to_onehot(f0).numpy().squeeze(), 
            linewidth = 1, 
#            linestyle = "dashed",
            color = cm.hsv(0.4),
        )

#    plt.colorbar(im, ax = ax, orientation = 'horizontal', aspect = 50)
    fig.canvas.draw()
    plt.close()
    return fig


# 2d tensor の長さ次元を、両端を zero padding して揃える処理

import torch.nn.functional as F

def pad_2d_tensors(x, y):
    if x.size(1) > y.size(1):
        pad_size = x.size(1) - y.size(1)
        left = pad_size // 2
        right = pad_size - left
        y_padded = F.pad(y, (left, right), mode = "constant", value = 0.0)
        x_padded = x
    else:
        pad_size = y.size(1) - x.size(1)
        left = pad_size // 2
        right = pad_size - left
        x_padded = F.pad(x, (left, right), mode = "constant", value = 0.0)
        y_padded = y
    return x_padded, y_padded



####
