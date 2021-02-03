import os
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import librosa
import librosa.display


def init_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)


def random_select_path(path_reg):
    path = tf.io.matching_files(path_reg)
    index = tf.random.categorical(tf.ones_like(path[tf.newaxis], dtype=tf.float32), 1)[0, 0]
    return path[index]


def preprocess_audio(id, url):
    audio_path_reg = tf.strings.join([str(data_root), 'wav', id, url, '*'], '\\')
    audio_path = random_select_path(audio_path_reg)

    audio = tf.io.read_file(audio_path)
    audio, sr = tf.audio.decode_wav(audio)

    return audio[16000:26500, 0]


class LogMelSpectrogram(tf.keras.layers.Layer):
    """Compute log-magnitude mel-scaled spectrograms."""

    def __init__(self, sample_rate=16000, fft_size=2048, hop_size=160, window_len=400, n_mels=64,
                 f_min=0.0, f_max=8000, **kwargs):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.window_len = window_len
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max else sample_rate / 2
        self.mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels,
            num_spectrogram_bins=fft_size // 2 + 1,
            sample_rate=self.sample_rate,
            lower_edge_hertz=self.f_min,
            upper_edge_hertz=self.f_max)

    def build(self, input_shape):
        self.non_trainable_weights.append(self.mel_filterbank)
        super().build(input_shape)

    def call(self, waveforms, **kwargs):
        """Forward pass.
        Parameters
        ----------
        waveforms : tf.Tensor, shape = (None, n_samples)
            A Batch of mono waveforms.
        Returns
        -------
        log_mel_spectrograms : (tf.Tensor), shape = (None, time, freq, ch)
            The corresponding batch of log-mel-spectrograms
        """

        def _tf_log10(x):
            numerator = tf.math.log(x)
            denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator

        def power_to_db(magnitude, amin=1e-16, top_db=80.0):
            """
            https://librosa.github.io/librosa/generated/librosa.core.power_to_db.html
            """
            ref_value = tf.reduce_max(magnitude)
            log_spec = 10.0 * _tf_log10(tf.maximum(amin, magnitude))
            log_spec -= 10.0 * _tf_log10(tf.maximum(amin, ref_value))
            log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)

            return log_spec

        spectrograms = tf.signal.stft(waveforms,
                                      window_fn=tf.signal.hamming_window,
                                      frame_length=self.window_len,
                                      frame_step=self.hop_size,
                                      fft_length=self.fft_size,
                                      pad_end=False)

        magnitude_spectrograms = tf.abs(spectrograms)

        mel_spectrograms = tf.matmul(tf.square(magnitude_spectrograms),
                                     self.mel_filterbank)

        log_mel_spectrograms = power_to_db(mel_spectrograms)

        # add channel dimension
        log_mel_spectrograms = tf.expand_dims(log_mel_spectrograms, 3)
        log_mel_spectrograms = log_mel_spectrograms / 40 + 1

        return log_mel_spectrograms

    def get_config(self):
        config = {
            'fft_size': self.fft_size,
            'hop_size': self.hop_size,
            'n_mels': self.n_mels,
            'sample_rate': self.sample_rate,
            'f_min': self.f_min,
            'f_max': self.f_max,
        }
        config.update(super().get_config())

        return config


def logmel_model():
    model = tf.keras.Sequential([
        tf.keras.Input([10500, ]),
        LogMelSpectrogram()
    ])
    return model


def voice_encoder():
    xs = tf.keras.Input([64, 64, 1])
    noise = xs + tf.random.normal(tf.shape(xs), 0, 0.1)
    model = tf.keras.Sequential([
        tf.keras.Input([64, 64, 1]),
        tf.keras.layers.Conv2D(40, 3, 2, padding='same'),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(60, 3, 2, padding='same'),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(2, 3, 1, padding='same'),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.BatchNormalization(),
    ])
    model = tf.keras.Model(inputs=xs, outputs=model(noise))
    return model


def voice_decoder():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2DTranspose(60, 3, 1, padding='same'),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(40, 3, 2, padding='same'),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(1, 3, 2, padding='same'),
    ])
    return model


def autoencoder_model():
    model = tf.keras.Sequential([
        voice_encoder(),
        voice_decoder()
    ])
    return model


class AudioAutoEncoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.logmel = logmel_model()
        self.encoder = voice_encoder()
        self.decoder = voice_decoder()

    def call(self, inputs, training=None, mask=None):
        lm_spec = self.logmel(inputs)
        embedding = self.encoder(lm_spec)
        rectr = self.decoder(embedding)
        return lm_spec, rectr, embedding

    def encode(self, inputs):
        lm_spec = self.logmel(inputs)
        embedding = self.encoder(lm_spec)
        return embedding

    def decode(self, embedding):
        voice = self.decoder(embedding)
        return voice


@tf.function
def train_step(model, audios):
    with tf.GradientTape() as tape:
        lm_spec, reconstructed, emb = model(audios)
        loss = mse(lm_spec, reconstructed)

    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss


if __name__ == '__main__':
    init_gpu()
    test = False
    data_root = pathlib.Path('<Location of the dataset>')

    df = pd.read_csv(data_root / 'vox1_meta.csv', sep='\t', index_col='VoxCeleb1 ID')
    id_to_names = df['VGGFace1 ID'].str.rstrip('.')

    wav = pathlib.Path(data_root / 'wav')
    ids = os.listdir(wav)
    fp = pathlib.Path(data_root / 'unzippedFaces')

    id_url_pairs = [(str(id), str(u))
                    for id in ids
                    for u in os.listdir(wav / id)]

    id_train, url_train = [list(d) for d in zip(*id_url_pairs)]

    pair_ds = tf.data.Dataset.from_tensor_slices((id_train, url_train))
    pair_ds = pair_ds.map(
        preprocess_audio, num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).shuffle(5000).batch(64).prefetch(2)

    model = AudioAutoEncoder()
    model.build(input_shape=[None, 10500])
    optimizer = tf.keras.optimizers.Adam(1e-5)
    mse = tf.keras.losses.MeanSquaredError()

    for epoch in range(20):
        for step, audios in enumerate(pair_ds):
            loss = train_step(model, audios)

            if step % 40 == 0:
                print(f'step {step}, loss = {loss}')

    if test:
        y, sr = librosa.load(data_root / 'test_wav/id10270/5r0dWxy17C8/00001.wav', sr=16000)

        lm, rc, em = model(y[None, 16000:26500])
        print(f'loss: {mse(lm, rc)}')
        db = (rc.numpy().squeeze() - 1) * 40
        db_true = (lm.numpy().squeeze() - 1) * 40
        pw = librosa.db_to_power(db)
        pw_true = librosa.db_to_power(db_true)
        y_inv = librosa.feature.inverse.mel_to_audio(pw.T, sr, fmax=8000,
                                                     window='hamming', win_length=400, hop_length=160)
        y_true = librosa.feature.inverse.mel_to_audio(pw_true.T, sr, fmax=8000,
                                                      window='hamming', win_length=400, hop_length=160)

        librosa.display.specshow(db.T)
        plt.colorbar()
        plt.show()
        librosa.display.specshow(db_true.T)
        plt.colorbar()

        model.save('autoencoder')
        model.save_weights('autoencoder.h5')
