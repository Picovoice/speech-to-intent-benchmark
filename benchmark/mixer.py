import os
import shutil
import sys

import numpy as np
import soundfile


def _path(x):
    return os.path.join(os.path.dirname(__file__), '../%s' % x)


def _max_frame_energy(pcm, frame_length=2048):
    num_frames = pcm.size // frame_length

    pcm_frames = pcm[:(num_frames * frame_length)].reshape((num_frames, frame_length))
    frames_power = (pcm_frames ** 2).sum(axis=1)

    return frames_power.max()


def _noise_scale(speech, noise, snr_db):
    assert speech.shape[0] == noise.shape[0]

    return np.sqrt(_max_frame_energy(speech) / (_max_frame_energy(noise) * (10 ** (snr_db / 10.))))


def mix(clean_folder, mix_folder, noise_name, snr_db):
    noise, sample_rate = soundfile.read(_path('data/noise/%s.wav' % noise_name))
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


def run(noise):
    for snr_db in [24, 21, 18, 15, 12, 9, 6]:
        snr_dir = _path('data/speech/%s_%ddb' % (noise, snr_db))
        if os.path.isdir(snr_dir):
            shutil.rmtree(snr_dir)
        os.mkdir(snr_dir)

        mix(_path('data/speech/clean'), snr_dir, noise, snr_db)


if __name__ == '__main__':
    run(sys.argv[1])
