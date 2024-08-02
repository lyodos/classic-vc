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
import torchaudio
from transformers import AutoModel

####

class SpectralConvergengeLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p=1) / torch.norm(y_mag, p=1)

# 以下の MultiResolutionSTFTLoss の定義で使う

class STFTLoss(torch.nn.Module):
    """STFT loss module."""
    def __init__(
        self, 
        sample_rate: int = 24000,
        fft_size = 1024, # n_fft はデフォルト 400
        shift_size = 120, # hop_length デフォルトは window // 2
        win_length = 600, 
        window = torch.hann_window,
    ):
        """Initialize STFT loss module."""
        super().__init__()
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        # f_min, f_max がないのでデフォルトの 0--12000 になる
        # n_mels がないのでデフォルトの 128 bins になる
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate = sample_rate, # 24000
            n_fft = fft_size, # [1024, 2048, 512]
            win_length = win_length, # [600, 1200, 240]
            hop_length = shift_size, # [120, 240, 50]
            window_fn = window,
        )

        self.spectral_convergenge_loss = SpectralConvergengeLoss()

    # x が predicted signal で y が正解。正規化の分母が y 側なので一応方向性がある。
    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = self.to_mel(x)
        mean, std = -4, 4
        x_mag = (torch.log(1e-5 + x_mag) - mean) / std
        
        y_mag = self.to_mel(y)
        mean, std = -4, 4
        y_mag = (torch.log(1e-5 + y_mag) - mean) / std
        
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)    
        return sc_loss

####

# こちらはユーザーが直接読み込む
# 実装を見ると fft_sizes = [1024, 2048, 512] は順番関係なく [512, 1024, 2048] でもいいが、慣例として真ん中が最大
# 実際に mel を描いてみると、1024 が最も鮮明で 2048 は周波数分解能が高いが時間解像度が低い。
# 逆に 512 は周波数分解能は低いが時間解像度が高い。おそらく 4096、256 はあまり意味がない。

class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""
    def __init__(
        self,
        sample_rate: int = 24000,
        fft_sizes = [1024, 2048, 512], # n_fft なので、lin spec の bin 数 n_fft // 2 + 1 = [513, 1025, 257]
        hop_sizes = [120, 240, 50], # 5 ms, 10 ms, 2.8 ms
        win_lengths = [600, 1200, 240], # 25 ms, 50 ms, 10 ms
        window = torch.hann_window,
    ):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
        """
        super().__init__()
        self.sample_rate = sample_rate
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)

        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(self.sample_rate, fs, ss, wl, window)]

    def forward(
        self, 
        x, # Predicted signal (B, T).
        y, # Groundtruth signal (B, T).
    ):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        for f in self.stft_losses:
            sc_l = f(x, y)
            sc_loss += sc_l
        sc_loss /= len(self.stft_losses)

        return sc_loss

####


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses

####

def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


""" https://dl.acm.org/doi/abs/10.1145/3573834.3574506 """
def discriminator_TPRLS_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        tau = 0.04
        m_DG = torch.median((dr-dg))
        L_rel = torch.mean((((dr - dg) - m_DG)**2)[dr < dg + m_DG])
        loss += tau - F.relu(tau - L_rel)
    return loss

def generator_TPRLS_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dg, dr in zip(disc_real_outputs, disc_generated_outputs):
        tau = 0.04
        m_DG = torch.median((dr-dg))
        L_rel = torch.mean((((dr - dg) - m_DG)**2)[dr < dg + m_DG])
        loss += tau - F.relu(tau - L_rel)
    return loss

####

# こちらはユーザーが直接読み込む

class GeneratorLoss(torch.nn.Module):

    def __init__(self, mpd, msd):
        super(GeneratorLoss, self).__init__()
        self.mpd = mpd
        self.msd = msd
        
    def forward(self, y, y_hat):
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, y_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, y_hat)
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)

        loss_rel = generator_TPRLS_loss(y_df_hat_r, y_df_hat_g) + generator_TPRLS_loss(y_ds_hat_r, y_ds_hat_g)
        
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_rel
        
        return loss_gen_all.mean()

####

# こちらはユーザーが直接読み込む

class DiscriminatorLoss(torch.nn.Module):

    def __init__(self, mpd, msd):
        super(DiscriminatorLoss, self).__init__()
        self.mpd = mpd
        self.msd = msd
        
    def forward(self, y, y_hat):
        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y, y_hat)
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)
        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y, y_hat)
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
        
        loss_rel = discriminator_TPRLS_loss(y_df_hat_r, y_df_hat_g) + discriminator_TPRLS_loss(y_ds_hat_r, y_ds_hat_g)


        d_loss = loss_disc_s + loss_disc_f + loss_rel
        
        return d_loss.mean()

####

# こちらはユーザーが直接読み込む。
# StyleTTS 2 では 2nd-stage training で複雑な損失を定義するが、
# 1st-stage および ClassicVC では下記機能の一部しか使用しない。

class WavLMLoss(torch.nn.Module):
    def __init__(
        self, 
        model, 
        wd, # ここに  WavLMDiscriminator(nn.Module) インスタンスを指定。
        model_sr, # 通常は 24000 だが、任意の入力音声を 16000 Hz にするので他のレートでも動く
        slm_sr = 16000,
    ):
        super().__init__()
        # wavlm のエンコーダを用意する。transformers.AutoModel は 378 MB ある
        self.wavlm = AutoModel.from_pretrained(model) # 
        self.wd = wd
        self.resample = torchaudio.transforms.Resample(model_sr, slm_sr) # 任意の入力音声を 16000 Hz にする
    
    def forward(
        self, 
        wav: torch.Tensor, # (batch, n_sample)
        y_rec: torch.Tensor, # (batch, n_sample)
    ):
        if wav.ndim > 2:
            n_trial = 0
            while wav.ndim > 2 and n_trial <= 3:
                wav = wav.squeeze(-2)
                n_trial += 1
        assert wav.ndim == 2

        if y_rec.ndim > 2:
            n_trial = 0
            while y_rec.ndim > 2 and n_trial <= 3:
                y_rec = y_rec.squeeze(-2)
                n_trial += 1
        assert y_rec.ndim == 2

        # まず wav, y_rec それぞれ 16 kHz に揃えてから WavLM に変換する。 
        # output_hidden_states = True で WavLM モデルの中間層も保持する
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(input_values = wav_16, output_hidden_states = True).hidden_states
        
        # reconst 側のみ勾配を記録する。同様に任意の入力音声を 16000 Hz にする
        y_rec_16 = self.resample(y_rec)
        # 以下、元コードは squeeze() しているがおかしくなるっぽい
        y_rec_embeddings = self.wavlm(input_values = y_rec_16, output_hidden_states = True).hidden_states
#        y_rec_embeddings = self.wavlm(input_values = y_rec_16.squeeze(), output_hidden_states = True).hidden_states

        floss = 0
        for er, eg in zip(wav_embeddings, y_rec_embeddings):
            floss += torch.mean(torch.abs(er - eg)) # 中間表現マップの L1 を全層足し合わせる
        return floss.mean()
    
    # 以下のコードは、StyleTTS 2 の 2nd stage の SLMAdversarialLoss のみで使う
    def generator(self, y_rec):
        y_rec_16 = self.resample(y_rec)
        y_rec_embeddings = self.wavlm(input_values=y_rec_16, output_hidden_states=True).hidden_states
        y_rec_embeddings = torch.stack(y_rec_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)
        y_df_hat_g = self.wd(y_rec_embeddings)
        loss_gen = torch.mean((1-y_df_hat_g)**2)
        
        return loss_gen
    
    def discriminator(self, wav, y_rec):
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(input_values=wav_16, output_hidden_states=True).hidden_states
            y_rec_16 = self.resample(y_rec)
            y_rec_embeddings = self.wavlm(input_values=y_rec_16, output_hidden_states=True).hidden_states

            y_embeddings = torch.stack(wav_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)
            y_rec_embeddings = torch.stack(y_rec_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)

        y_d_rs = self.wd(y_embeddings)
        y_d_gs = self.wd(y_rec_embeddings)
        
        y_df_hat_r, y_df_hat_g = y_d_rs, y_d_gs
        
        r_loss = torch.mean((1-y_df_hat_r)**2)
        g_loss = torch.mean((y_df_hat_g)**2)
        
        loss_disc_f = r_loss + g_loss
        
        return loss_disc_f.mean()

    def discriminator_forward(self, wav):
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(input_values=wav_16, output_hidden_states=True).hidden_states
            y_embeddings = torch.stack(wav_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)

        y_d_rs = self.wd(y_embeddings)
        
        return y_d_rs

####

