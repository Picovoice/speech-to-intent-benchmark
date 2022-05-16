import os
import shutil
from typing import *

import numpy as np
import soundfile
from numpy.typing import NDArray


def path(x: str) -> str:
    return os.path.join(os.path.dirname(__file__), f'../{x}')


def _max_frame_energy(pcm: NDArray[float], frame_length: int = 2048) -> float:
    num_frames = pcm.size // frame_length

    pcm_frames = pcm[:(num_frames * frame_length)].reshape((num_frames, frame_length))
    frames_power = (pcm_frames ** 2).sum(axis=1)

    return frames_power.max()


def _noise_scale(speech: NDArray[float], noise: NDArray[float], snr_db: float) -> NDArray[float]:
    assert speech.shape[0] == noise.shape[0]

    return np.sqrt(_max_frame_energy(speech) / (_max_frame_energy(noise) * (10 ** (snr_db / 10.))))


def mix(clean_folder: str, mix_folder: str, noise: str, snr_db: float) -> None:
    noise, sample_rate = soundfile.read(path(f'data/noise/{noise}.wav'))
    assert sample_rate == 16000

    for clean_file in os.listdir(clean_folder):
        if clean_file.endswith('.wav'):
            clean_pcm, sample_rate = soundfile.read(os.path.join(clean_folder, clean_file))
            assert sample_rate == 16000
            assert len(clean_pcm) <= len(noise)

            noise_start_index = np.random.randint(0, len(noise) - len(clean_pcm))
            noise_end_index = noise_start_index + len(clean_pcm)

            noise_scale = _noise_scale(clean_pcm, noise[noise_start_index:noise_end_index], snr_db)

            noisy_pcm = clean_pcm + noise_scale * noise[noise_start_index:noise_end_index]
            noisy_pcm /= 2 * np.max(np.abs(noisy_pcm))

            soundfile.write(os.path.join(mix_folder, clean_file), noisy_pcm, sample_rate)


def run(noise: str, snrs_ds: Sequence[float], overwrite: bool = False) -> None:
    for snr_db in snrs_ds:
        snr_dir = path(f'data/speech/{noise}_{snr_db}db')
        if os.path.isdir(snr_dir):
            if overwrite:
                shutil.rmtree(snr_dir)
            else:
                continue
        os.mkdir(snr_dir)

        mix(path('data/speech/clean'), snr_dir, noise, snr_db)


__all__ = [
    'run',
]
