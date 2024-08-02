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

import typing
from pathlib import Path
import os
import numpy as np

import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from .network import HarmoF0
from .conv_stft import ConvSTFT


####


# 以下が本来の wav2spec モジュール（のマイナーチェンジ）である。
# 概ね HarmoF0 のオリジナル実装と同じだが、torchaudio.stft の仕様に合わせており center = True での計算が可能。

# せっかく作ったものの古典的なアルゴリズムによる stft があると、ONNX 化に失敗する。しゃあないので CNN 版を使うことにした

class LogSpectrogram(nn.Module):
    def __init__(
        self, 
        sample_rate: int = 16000, # 16000
        n_fft: int = 1024, # 1024
        fmin: float = 27.5, # 27.5
        bins_per_octave: int = 48, # Q = 48
        freq_bins: int = 352, # 352
        hop_length: int = 160, # 160
        center: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft # FFT frame length
        self.fmin = fmin
#        assert bins_per_octave % 12 == 0 # 半音の bin 幅が整数になること。
        self.bins_per_octave = bins_per_octave
        self.freq_bins = freq_bins
        self.hop_length = hop_length
        self.center = center
        self.device = device

        self.window = torch.hann_window(self.n_fft).to(self.device)

        # 次に対数周波数の bin を登録する
        fre_resolution = self.sample_rate/self.n_fft # 16000/1024
        idxs = torch.arange(0, self.freq_bins) # 0, 1, 2, ..., 352
        log_idxs = self.fmin * (2**(idxs/self.bins_per_octave)) / fre_resolution # 分子は各 bin に対応する Hz (27.5 ~ 4371.3394)
        log_idxs = log_idxs.to(self.device)
        # Linear interpolation： y_k = y_i * (k-i) + y_{i+1} * ((i+1)-k)
        log_idxs_0 = torch.floor(log_idxs).long().to(self.device)
        log_idxs_0w = (log_idxs - log_idxs_0).reshape([1, 1, self.freq_bins]).to(self.device)
        log_idxs_1 = torch.ceil(log_idxs).long().to(self.device)
        log_idxs_1w = (log_idxs_1 - log_idxs).reshape([1, 1, self.freq_bins]).to(self.device)
        self.register_buffer("log_idxs_0", log_idxs_0, persistent = False)
        self.register_buffer("log_idxs_0w", log_idxs_0w, persistent = False)
        self.register_buffer("log_idxs_1", log_idxs_1, persistent = False)
        self.register_buffer("log_idxs_1w", log_idxs_1w, persistent = False)
        #print(log_idxs_0.shape, log_idxs_0w.shape, log_idxs_1.shape, log_idxs_1w.shape)
        #torch.Size([352]) torch.Size([1, 1, 352]) torch.Size([352]) torch.Size([1, 1, 352])
        # PyTorch の実装上、registered buffer は to(device) 時に inplace ではなくコピーが必要らしい。

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db = 80) # stype = 'power'


    def forward(
        self, 
        x: torch.Tensor, # (1, sample)
    ):
#        assert x.ndim == 2, "Input waveform must be 2D (1, sample)"
        spect_center = torch.stft(
            x, 
            n_fft = self.n_fft, 
            hop_length = self.hop_length, 
            win_length = self.n_fft,
            window = self.window.to(x.device), # 自動で入力データと同じデバイスに window も移動
            center = self.center,
            pad_mode = 'reflect',
            normalized = False, # torch.fft.fft に合わせると、パワーの正規化は行わない
            onesided = True, # 負の周波数を返さない
            return_complex = True, # PyTorch v2 では必ず True で指定すること。
        ).permute(0, 2, 1) # torch.Size([1, 513, n_frame]) -> torch.Size([1, n_frame, 513])
        spec = torch.square(torch.abs(spect_center)) # power に変換
        # 対数周波数の bin に変換
        spec =  spec[..., self.log_idxs_0.to(x.device)] * self.log_idxs_0w.to(x.device) + \
                spec[..., self.log_idxs_1.to(x.device)] * self.log_idxs_1w.to(x.device)
        spec = self.amplitude_to_db(spec).float() # デフォルトの stype = 'power' でインスタンス化されている
        return spec # (1, n_frame, 352)


####

# こちらは元のリポジトリにない、完全な改造版 
# torch.stft を  https://github.com/pseeth/pytorch-stf に置換することで強引に ONNX 化を可能にした

class ConvLogSpectrogram(nn.Module):
    def __init__(
        self, 
        sample_rate: int = 16000, # 16000
        n_fft: int = 1024, # 1024
        fmin: float = 27.5, # 27.5
        bins_per_octave: int = 48, # Q = 48
        freq_bins: int = 352, # 352
        hop_length: int = 160, # 160
        center: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft # FFT frame length
        self.fmin = fmin
#        assert bins_per_octave % 12 == 0 # 半音の bin 幅が整数になること。
        self.bins_per_octave = bins_per_octave
        self.freq_bins = freq_bins
        self.hop_length = hop_length
        self.center = center
        self.device = device

        self.conv_stft = ConvSTFT(
            filter_length = self.n_fft, # n_fft
            hop_length = self.hop_length, 
            win_length = self.n_fft,
            window = 'hann',
        ).to(self.device)

        # 次に対数周波数の bin を登録する
        fre_resolution = self.sample_rate/self.n_fft # 16000/1024
        idxs = torch.arange(0, self.freq_bins) # 0, 1, 2, ..., 352
        log_idxs = self.fmin * (2**(idxs/self.bins_per_octave)) / fre_resolution # 分子は各 bin に対応する Hz (27.5 ~ 4371.3394)
        log_idxs = log_idxs.to(self.device)
        # Linear interpolation： y_k = y_i * (k-i) + y_{i+1} * ((i+1)-k)
        log_idxs_0 = torch.floor(log_idxs).long().to(self.device)
        log_idxs_0w = (log_idxs - log_idxs_0).reshape([1, 1, self.freq_bins]).to(self.device)
        log_idxs_1 = torch.ceil(log_idxs).long().to(self.device)
        log_idxs_1w = (log_idxs_1 - log_idxs).reshape([1, 1, self.freq_bins]).to(self.device)
        self.register_buffer("log_idxs_0", log_idxs_0, persistent = False)
        self.register_buffer("log_idxs_0w", log_idxs_0w, persistent = False)
        self.register_buffer("log_idxs_1", log_idxs_1, persistent = False)
        self.register_buffer("log_idxs_1w", log_idxs_1w, persistent = False)

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db = 80) # stype = 'power'


    def forward(
        self, 
        x: torch.Tensor, # (1, sample)
    ):
        magnitude, _ = self.conv_stft(x)
        spec = torch.square(torch.abs(magnitude.permute(0, 2, 1))) # power に変換
        # 対数周波数の bin に変換
        spec =  spec[..., self.log_idxs_0.to(x.device)] * self.log_idxs_0w.to(x.device) + \
                spec[..., self.log_idxs_1.to(x.device)] * self.log_idxs_1w.to(x.device)
        spec = self.amplitude_to_db(spec).float() # デフォルトの stype = 'power' でインスタンス化されている
        return spec # (1, n_frame, 352)



####

# HarmoF0 の主クラスを少し改造。
# 原実装に存在するが使用されていない、活性化マップの postProcessing 機能を完全にオミットした
# スペクトログラム部分を CNN に置換した

# 現時点での制約：
# Batch 次元が 1 でないと HarmoF0 の conv 層に通らないので、後の BatchedPitchEnergyTracker では
# 強引に for 文で回して擬似的に batch を扱えるようにした。

# また長すぎる音声は一括で処理しようとして VRAM が枯渇する。ただし数十秒であれば問題ない。

class PitchTracker():
    def __init__(
        self, 
        checkpoint_path: typing.Union[str, Path] = None, # 標準では harmof0/checkpoints/mdb-stem-synth.pth
        fmin: float = 27.5, # f0 として想定する最低周波数の Hz で、ピアノの最低音の A に相当する。
        sample_rate: int = 16000,
        hop_length: int = 160, # f0 を推定する間隔。160/16000 = 10 ms
        frame_len: int = 1024, # sliding window を切り出す長さ。n_fft に等しい
        center: bool = True, # wav2spectrogram の center = True を指定するか。変換結果のフレーム数が変わる。
        frames_per_step: int = 1000, # 1 回の forward で何 frame ずつ投入するか（VRAM に依存）ただし現在は未実装
        high_threshold: float = 0.8, 
        low_threshold: float = 0.1, 
        freq_bins_in: int = 88*4,
        bins_per_octave_in: int = 48, # LogSpectrogram の定義に使う
        bins_per_octave_out: int = 48, # onehot_to_hz に使う
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        return_spectrogram: bool = False,
        freeze: bool = True, # HarmoF0 の weight をチェックポイントの読み込み後に凍結し、学習不可にする
    ) -> None:
        self.fmin = fmin
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.frame_len = frame_len
        self.center = center
        self.frames_per_step = frames_per_step
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.freq_bins_in = freq_bins_in
        self.bins_per_octave_in = bins_per_octave_in
        self.bins_per_octave_out = bins_per_octave_out
        self.device = device
        self.return_spectrogram = return_spectrogram
        self.freeze = freeze
        self.fmax = fmin*(2**((self.freq_bins_in - 1)/self.bins_per_octave_in)) # 27.5*2^(351/48) = 4371.3394

        self.w2ls = ConvLogSpectrogram(
            sample_rate = self.sample_rate, # 16000
            n_fft = self.frame_len, # 1024
            fmin = self.fmin, # 27.5
            bins_per_octave = self.bins_per_octave_in, # Q = 48
            freq_bins = self.freq_bins_in, # 352
            hop_length = self.hop_length, # 160
            center = self.center,
            device = self.device,
        )
        self.net = HarmoF0().to(device) # モデル実体は network にある
        
        if checkpoint_path is None:
            package_dir = os.path.dirname(os.path.realpath(__file__))
            weights_name = "mdb-stem-synth.pth"
            checkpoint_path = os.path.join(package_dir, 'checkpoints', weights_name)
        self.net.load_state_dict(torch.load(checkpoint_path, map_location = "cpu"), strict = False)

        if self.freeze == True:
            for param in self.net.parameters():
                param.requires_grad = False # Pitch tracker の net は凍結

        self.net.eval()


    def pred(
        self, 
        waveform: torch.Tensor, # (1, n_sample)
        return_map_only: bool = False,
    ):
        if waveform.ndim == 1:
            waveform = waveform[None, :] # 1D だったら冒頭に batch 次元を足して 2D に
#        assert waveform.size(0) == 1, "Batch size (dim 0) must be 1"

        # スペクトログラムの作成
        spectrogram = self.w2ls(waveform).to(self.device) # (1, num_frames, n_bins)
        # 順伝播
        with torch.no_grad():
            act_map = self.net(spectrogram).squeeze(0) # torch.Size([T, 352])
        
        # ここにあった活性化マップの後処理を廃止した
        
        if return_map_only == True:
            if self.return_spectrogram == True:
                return act_map, spectrogram # torch.Size([T, 352]), torch.Size([T, 352])
            else:
                return act_map # torch.Size([T, 352])
        else:
            # 活性化マップが最大値をとる bin (batch, n_frame) の位置（音高）と、そのときの最大信号強度
            est_freqs, est_activations = self.onehot_to_hz(
                act_map[None, :], # singleton dimension として dim 1 を入れてから代入
                self.bins_per_octave_out, 
                threshold = 0.0,
            )
            pred_freq = est_freqs.flatten()
            pred_activation = est_activations.flatten()
            
            if self.return_spectrogram == True:
                return pred_freq, pred_activation, act_map, spectrogram # [T], [T], [T x 352], [T x 352]
            else:
                return pred_freq, pred_activation, act_map # [T], [T], [T x 352]

    # pred() が __call__() すなわち、インスタンス名をそのまま関数として呼んだ時の実行内容となる。
    def __call__(
        self, 
        waveform: torch.Tensor, # (1, n_sample)
        sr: int,
        return_map_only: bool = False,
    ):
        return self.pred(waveform, sr, return_map_only)
    
    # 活性化マップの周波数の解像度は 1/4 半音で、88 半音（ピアノの鍵盤数）をカバーするので 352 bins である。

    # 活性化マップの各タイムフレームについて、強度が最大となる bin を見つけ、Hz 単位の周波数推定値を計算する
    def onehot_to_hz(
        self, 
        onehot: torch.Tensor, # (batch, n_frame, bins = 352)
        bins_per_octave: int,
        threshold: int = 0.6,
    ):
        # 要素 1 が、活性化マップが最大値をとる bin (batch, n_frame) の位置。要素 0 がそのときの最大信号強度
        max_onehot_values, max_onehot_indices = torch.max(onehot, dim = 2)
        mask = (max_onehot_values > threshold).float() # 強度が threshold を上回るフレームは 1, それ以外は 0
        hz = self.fmin * (2**(max_onehot_indices / bins_per_octave))
        hz = hz * mask # 強度が threshold に満たないフレームの値を 0 にする（つまり無声音＝ゼロ決め打ち）
        return hz, max_onehot_values

    # こちらは Hz 単位の周波数を、352 bins の one-hot feature map に戻す（ただし使用された形跡がない）
    def hz_to_onehot(
        self, 
        hz, 
        freq_bins, 
        bins_per_octave,
    ):
        # input: [b x T]
        # output: [b x T x freq_bins]
        fmin = self.fmin
        indices = ( torch.log((hz+0.0000001)/fmin) / np.log(2.0**(1.0/bins_per_octave)) + 0.5 ).long()
#        assert(torch.max(indices) < freq_bins)
        mask = (indices >= 0).long()
        # => [b x T x 1]
        mask = torch.unsqueeze(mask, dim=2)
        # => [b x T x freq_bins]
        onehot = F.one_hot(torch.clip(indices, 0), freq_bins)
        onehot = onehot * mask # mask the freq below fmin
        return onehot


####

# バッチ処理を擬似的に実装し、さらにピッチに加えてエネルギーも返すクラスを作成。これを訓練に使う。

class BatchedPitchEnergyTracker(nn.Module):
    def __init__(
        self, 
        checkpoint_path,
        fmin = 27.5, # f0 として想定する最低周波数の Hz で、ピアノの最低音の A に相当する。
        sample_rate = 16000,
        hop_length = 160, # f0 を推定する間隔。160/16000 = 10 ms これは 20 ms でも動作はするが、精度が落ちるので非推奨
        frame_len = 1024, # sliding window を切り出す長さ
        frames_per_step = 1000, # 1 回の forward で投入する最大セグメント数。
        high_threshold = 0.8, 
        low_threshold = 0.1, 
        freq_bins_in = 88*4,
        bins_per_octave_in = 48,
        bins_per_octave_out = 48,
        cutoff: float = 0.0, # 活性化マップ（0 以上 1 以下）の値が一定レベル以下の部分をカットして 0.0 にする
        device = "cuda:0",
        compile: bool = False,
        dry_run: int = 10,
    ) -> None:
        super().__init__()
        self.sr = sample_rate
        self.device = device
        self.cutoff_level = cutoff
        # 上で定義したとおり、条件の仔細は PitchTracker インスタンスに変数として保持させる。
        self.single_tracker = PitchTracker(
            checkpoint_path = checkpoint_path,
            fmin = fmin,
            sample_rate = sample_rate,
            hop_length = hop_length,
            frame_len = frame_len, 
            frames_per_step = frames_per_step, 
            high_threshold = high_threshold, 
            low_threshold = low_threshold, 
            freq_bins_in = freq_bins_in,
            bins_per_octave_in = bins_per_octave_in,
            bins_per_octave_out = bins_per_octave_out,
            return_spectrogram = True,
            device = device,
        )
        self.cutoff_filter = nn.Threshold(self.cutoff_level, 0.0, inplace = False)

        if compile == True:
            self.single_tracker.net = torch.compile(self.single_tracker.net)
            with torch.no_grad():
                sample_tensor = torch.rand(dry_run, 16000*1).to(device)
                for i in range(dry_run):
                    _ = self.single_tracker.pred(sample_tensor[i, :], sr = self.sr, return_map_only = True) 

    def forward(
        self,
        x: torch.Tensor, # (batch, sample) 2d が基本だが、さらに batch 次元が付いていても内部で 2D に落とせる。
    ):
#        assert x.ndim == 2 or x.ndim == 3, "input tensor must be 2D (batch, sample) or 3D (batch1, batch2, sample)"
        orig_shape = list(x.shape) # 入力テンソルの次元（基本的に batch, sample である）を単なる int list にして保存
        if x.ndim == 3:
            x = x.view(orig_shape[0]*orig_shape[1], orig_shape[2]) # 内部処理は 2D に落としてから
        orig_device = x.device
        x = x.to(self.device)

        # batch は single_tracker を for 文で回して list に入れる
        freq = []
        act = []
        spec = []
        for i in range(x.size(0)):
            pred_freq, pred_activation, _, source_spec = self.single_tracker.pred(
                x[i, :], 
                return_map_only = False,
            ) # [T], [T], [T x 352], [T x 352] ただし 3 番目の act_map は捨てる
            freq.append(pred_freq.unsqueeze(0)) # [(1, time), ...]
            act.append(pred_activation.unsqueeze(0)) # [(1, time), ...]
            spec.append(source_spec) # [(1, time, 352), ...]
        
        # F0
        freq = torch.cat(freq, dim = 0).contiguous() # torch.Size([batch, time])
        # 活性化マップ（0 以上 1 以下）
        act = torch.cat(act, dim = 0).contiguous() # torch.Size([batch, time])
        # 活性化マップの値が一定レベル self.cutoff_level 以下の部分を 0.0 にする。
        act = self.cutoff_filter(act) 
        # 変換元のスペクトログラム。なお dB スケールに変換されている
        spec = torch.cat(spec, dim = 0).transpose(2, 1).contiguous() # torch.Size([batch, time, 352]) -> time last

        # Energy の計算。
        energy = torch.log(torch.exp(spec * 0.25 + 1).norm(dim = 1)) # ONNX 化のために mean, std を固定
        
        # バッチ次元の復元
        if len(orig_shape) == 2:
            result_shape = orig_shape[:1] + list(freq.shape[1:])
            spec_shape = orig_shape[:1] + list(spec.shape[1:])
        else:
            result_shape = orig_shape[:2] + list(freq.shape[1:])
            spec_shape = orig_shape[:2] + list(spec.shape[1:])

        return freq.to(orig_device).view(result_shape), act.to(orig_device).view(result_shape), energy.to(orig_device).view(result_shape), spec.to(orig_device).view(spec_shape)

