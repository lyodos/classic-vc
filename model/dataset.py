
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

import random
import itertools

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchaudio
from torchaudio.functional import resample



class VCDataset(Dataset):
    def __init__(
        self,
        x: list, #  全ての「話者＞発話」を一覧できるリスト。直下要素もリストで、各発話情報の dict を束ねる。
        n_utterances: int, # 1 回の呼び出しでサンプルしたい 1 話者分の発話数（データセットに実際含まれる数ではない）。
        sampling_rate: int = 16000, # ネットワークに流すための事前整形で目標とするサンプリング周波数。
        min_sec: float = 2.0, # 教師データの wav が最低限満たすべき有効な秒数
        sec: float = None, # サンプル秒数を一意に定める場合はここに秒数（例： 2.0）を指定。None なら max_sec に合わせる
        max_sec: float = 15.0, # sec = None の場合に最長限度とするサンプル秒数
        valid_mode: bool = False, # モードが train か valid か。
        with_ref: bool = False, # 同じ話者の別の発話を reference として作成する。StyleTTS 2 の拡散モデルを訓練する場合は必要
        debug: bool = False,
    ):
        self.speakers = x
        self.n_utterances = n_utterances # 発話数がこれに満たない話者はデータセットに含めない。
        self.sr = sampling_rate # データセットには音声処理用のサンプリング周波数の情報が必要。
        self.min_sec = min_sec
        self.sec = sec
        self.max_sec = max_sec
        self.valid_mode = valid_mode
        self.with_ref = with_ref
        
        # バッチを作るときのフレーム長
        if self.sec is not None:
            self.batch_frame_len = int(self.sec*self.sr)
        else:
            self.batch_frame_len = int(self.max_sec*self.sr)
        self.debug = debug

        speaker_use = []
        # uttr_info_list は話者を単位とするリストであり、当該話者の発話単位のメタ情報をまとめた辞書を要素に持つ。
        for uttr_info_list in self.speakers:
            # 話者単位
            uttr_use = []
            for uttr_info in uttr_info_list:
                # 発話単位
                if uttr_info["seconds"] > self.min_sec:
                    uttr_use.append(True)
            # 各話者について、欲しい個数の発話が得られた話者のみデータセットに含める
            if len(uttr_use) >= self.n_utterances*2 if with_ref is True else self.n_utterances:
                speaker_use.append(True)
            else:
                speaker_use.append(False)
        self.speakers = [spk_l for spk_l, use in zip(self.speakers, speaker_use) if use == True]


    def __len__(self):
        return len(self.speakers) # データセットの長さ＝「十分個数の発話データを持つ話者数」


    # 「1 話者に相当する」 n_utterances 個の発話の埋め込み情報を取り出す。
    # ただし、wav が nan を含む場合はスキップする。その場合は batch size が n_utterances 個よりも小さくなる
    def __getitem__(self, index):
        uttr_info_list = self.speakers[index]
        wavs = []
        wav_paths = []
        speaker_names = []
        if self.with_ref:
            ref_wavs = []
            ref_wav_paths = []

        if self.valid_mode:
            # index で指定した話者について、ランダマイズせず冒頭から self.n_utterances 個の発話を選ぶ。
            # Validation では、同じ発話が同じ順番で選ばれた方が TensorBoard で評価しやすいため。
            selected_ids = list(range(self.n_utterances))
        else:
            # Train では「非復元」抽出で発話を選ぶ。十分数の発話があることは保証されている。
            if self.with_ref:
                selected_ids_all = random.sample(list(range(len(uttr_info_list))), self.n_utterances*2) 
                # batch size の 2 倍の数の発話を選んでから、前半を通常のクリップ、後半を ref にする
                selected_ids = selected_ids_all[self.n_utterances:]
                selected_ids_ref = selected_ids_all[:self.n_utterances]
            else:
                selected_ids = random.sample(list(range(len(uttr_info_list))), self.n_utterances) 

        # バッチに含めることに決まった id の音声を処理する
        for id in selected_ids:
            uttr_info = uttr_info_list[id] # uttr_info は発話ごとのメタデータを格納した辞書
            wav_path = uttr_info["audio_path"]
            # 音声
            wav, orig_sr = torchaudio.load(wav_path, normalize = True) # torch.Size([n_channel, n_sample])
            wav = wav.mean(dim = 0, keepdim = True) # .mean() メソッドで常にモノラルにミックスダウン

            # 最初に指定したサンプリング周波数に強引に合わせる。つまりデータセットによっては upsample される
            if orig_sr != self.sr:
                wav = resample(wav, orig_freq = orig_sr, new_freq = self.sr, resampling_method = 'sinc_interp_kaiser')
    
            # バッチを束ねた 1 本のテンソルとして取り出すため、時間長を揃える
            if wav.shape[-1] > self.batch_frame_len:
                # 元音声のほうが長い→ランダムに区間抽出
                sample_from = random.randint(0, wav.shape[-1] - self.batch_frame_len)
                wav = wav[:, sample_from:sample_from + self.batch_frame_len]
            elif wav.shape[-1] < self.batch_frame_len:
                # 元音声のほうが短い→末尾をゼロ埋め
                wav = F.pad(wav, (0, self.batch_frame_len - wav.shape[-1]), mode = 'constant', value = 0)

            if self.debug == True:
                assert wav.ndim == 2 and wav.size(1) == self.batch_frame_len, "wav size: {}".format(wav.shape)

            # ごくまれに NaN を含む音声がロードされることがあるので除外
            if torch.isnan(wav).any() == False:
                wavs.append(wav.detach()) # torch.Size([1, self.batch_frame_len])
                wav_paths.append(wav_path)
                speaker_names.append(uttr_info["speaker"])

        batch = torch.cat(wavs, dim = 0) #  [self.n_utterances, self.batch_frame_len] の wav tensor 

        if self.with_ref:
            for id in selected_ids_ref:
                uttr_info = uttr_info_list[id]
                wav_path = uttr_info["audio_path"]
                # 音声
                wav, orig_sr = torchaudio.load(wav_path, normalize = True) 
                wav = wav.mean(dim = 0, keepdim = True) 

                if orig_sr != self.sr:
                    wav = resample(wav, orig_freq = orig_sr, new_freq = self.sr, resampling_method = 'sinc_interp_kaiser')

                if wav.shape[-1] > self.batch_frame_len:
                    sample_from = random.randint(0, wav.shape[-1] - self.batch_frame_len)
                    wav = wav[:, sample_from:sample_from + self.batch_frame_len]
                elif wav.shape[-1] < self.batch_frame_len:
                    wav = F.pad(wav, (0, self.batch_frame_len - wav.shape[-1]), mode = 'constant', value = 0)

                if self.debug == True:
                    assert wav.ndim == 2 and wav.size(1) == self.batch_frame_len, "wav size: {}".format(wav.shape)

                if torch.isnan(wav).any() == False:
                    ref_wavs.append(wav.detach()) # torch.Size([1, self.batch_frame_len])
                    ref_wav_paths.append(wav_path)
            ref_batch = torch.cat(ref_wavs, dim = 0) #  [self.n_utterances, self.batch_frame_len] の wav tensor 

            return (batch, wav_paths, speaker_names, ref_batch, ref_wav_paths)
        else:
            return (batch, wav_paths, speaker_names)


# 上記が返すのは同一話者の n_utterance 個の発話を束ねたテンソルを、batch_size 個つなげたリストである。
# 訓練時はこれをさらに 1 本のテンソルに束ねる必要があるため、 collator と呼ばれる補助関数を使う。

# なお以下の方法だと、第 0 次元が話者ごとに self.n_utterances ずつ、固まって配置されることに注意。
# 今回は VC タスクなので問題ない。Speaker verification の訓練だったらシャッフルしないと使えない。

class WavCollator():
    def __init__(
        self, 
    ) -> None:
        pass

    def create_batch(
        self, 
        x: list,
    ):
        wav_t = torch.cat([record[0] for record in x], dim = 0)
        filenames = list(itertools.chain.from_iterable([record[1] for record in x]))
        speakers = list(itertools.chain.from_iterable([record[2] for record in x]))
        
        if len(x[0]) == 3:
            # with_ref == False
            return wav_t, filenames, speakers
        else:
            # with_ref == True の場合に対応
            ref_wav_t = torch.cat([record[3] for record in x], dim = 0)
            ref_filenames = list(itertools.chain.from_iterable([record[4] for record in x]))
            return wav_t, filenames, speakers, ref_wav_t, ref_filenames

    def __call__(self, xs: list):
        return self.create_batch(xs)


####

# 入力テンソルの末尾次元が指定サイズよりも長ければランダムな区間を抽出し、短ければ末尾をゼロ埋めして返す。
# ただし、以下の関数は適用ごとに乱数を作るため、同じイテレーションで複数回呼び出すと開始位置がずれてしまう。
# 結果的に音声の比較ができなくなり、使い勝手がアレなので、最終的にはお蔵入りになった


def pad_or_extract(
    x: torch.Tensor, 
    target_len: int,
):
    input_length = x.size(-1)
    if input_length < target_len:
        m = torch.nn.ConstantPad1d((0, target_len - input_length), 0.0)
        return m(x)
    else:
        start_id = random.randint(0, input_length - target_len)
        return x[..., start_id:start_id+target_len]


def pad_or_extract_spec(
    x: torch.Tensor, 
    target_len: int,
    dim: int,
):
    ndim = x.ndim
    x = x.transpose(ndim-1, dim) # 転換対象の dim が必ず最後に来るように入れ替え
    input_length = x.size(-1)
    if input_length < target_len:
        # 目標のほうが長い→ center で pad を作る
        pad_left = (target_len - input_length) // 2
        pad_right = target_len - input_length - pad_left
#        print(pad_left, pad_right)
        m = torch.nn.ConstantPad1d((pad_left, pad_right), value = 0.0)
        x = m(x)
    else:
        start_id = random.randint(0, input_length - target_len)
        x = x[..., start_id:start_id+target_len]
    x = x.transpose(ndim-1, dim)
    return x

####
