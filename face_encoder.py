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


def preprocess_image(image):
    image = tf.image.resize(image, (80, 80))
    image = tf.image.random_crop(image, (64, 64, 3))
    image = tf.image.random_flip_left_right(image)
    image = image / 127.5 - 1
    return image


def preprocess(id, name, url):
    audio_path_reg = tf.strings.join([str(data_root), 'wav', id, url, '*'], '\\')
    image_path_reg = tf.strings.join([str(data_root), 'unzippedFaces', name, '1.6', url, '*'], '\\')

    audio_path = random_select_path(audio_path_reg)
    image_path = random_select_path(image_path_reg)

    audio = tf.io.read_file(audio_path)
    audio, sr = tf.audio.decode_wav(audio)
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image)
    image = preprocess_image(image)

    return audio[16000:26500, 0], image


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
        tf.keras.Input([16, 16, 2]),
        tf.keras.layers.Conv2DTranspose(60, 3, 1, padding='same'),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(40, 3, 2, padding='same'),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(1, 3, 2, padding='same'),
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

    def dec1(self, embedding):
        d = self.decoder.layers[0](embedding)
        d = self.decoder.layers[1](d)
        return d


def face_encoder():
    model = tf.keras.Sequential([
        tf.keras.Input([64, 64, 3]),
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
    return model


@tf.function
def train_step(v_model, f_model, audios, images):
    with tf.GradientTape() as tape:
        emb_true = v_model.encode(audios)
        emb_pred = f_model(images)
        spec_true = v_model.decode(emb_true)
        spec_pred = v_model.decode(emb_pred)
        m_loss = mse(emb_true, emb_pred)
        d_loss = mse(spec_true, spec_pred)
        loss = m_loss + d_loss

    grads = tape.gradient(loss, f_model.trainable_weights)
    optimizer.apply_gradients(zip(grads, f_model.trainable_weights))
    return m_loss, d_loss


if __name__ == '__main__':
    init_gpu()
    test = False
    data_root = pathlib.Path(r'E:\ML dataset')

    df = pd.read_csv(data_root / 'vox1_meta.csv', sep='\t', index_col='VoxCeleb1 ID')
    id_to_names = df['VGGFace1 ID'].str.rstrip('.')

    wav = pathlib.Path(data_root / 'wav')
    ids = os.listdir(wav)
    fp = pathlib.Path(data_root / 'unzippedFaces')

    id_url_pairs = [(str(id), str(u))
                    for id in ids
                    for u in os.listdir(wav / id)]

    id_train, url_train = [list(d) for d in zip(*id_url_pairs)]
    name_train = id_to_names[id_train].values

    pair_ds = tf.data.Dataset.from_tensor_slices((id_train, name_train, url_train))
    pair_ds = pair_ds.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).shuffle(4000).batch(64).prefetch(2)

    voice_model = AudioAutoEncoder()
    voice_model.build(input_shape=[None, 10500])
    voice_model.load_weights('autoencoder.h5')

    face_model = face_encoder()
    optimizer = tf.keras.optimizers.Adam(1e-3, beta_1=0.7, epsilon=1e-4)

    mse = tf.keras.losses.MeanSquaredError()
    mae = tf.keras.losses.MeanAbsoluteError()
    kld = tf.keras.losses.KLDivergence()
    cetp = tf.keras.losses.BinaryCrossentropy()

    for epoch in range(10):
        for step, (audios, images) in enumerate(pair_ds):
            m_loss, d_loss = train_step(audios, images)

            if step % 40 == 0:
                print(f'step {step}, loss = {m_loss} + {d_loss}')

    if test:
        y, sr = librosa.load(data_root / 'test_wav/id10270/5r0dWxy17C8/00001.wav', sr=16000)

        img = plt.imread(data_root / r'unzippedFaces\Agyness_Deyn\1.6\1uUxa00zKXE\0004900.jpg')
        plt.imshow(img)
        img = preprocess_image(img)

        embed = face_model(img[None])
        db = voice_model.decode(embed)

        pw = librosa.db_to_power((db.numpy().squeeze() - 1) * 40)
        y_inv = librosa.feature.inverse.mel_to_audio(pw.T, 16000, fmax=8000,
                                                     window='hamming', win_length=400, hop_length=100)

        librosa.display.specshow(((db.numpy().squeeze() - 1)).T)
        plt.colorbar()

        librosa.display.waveplot(y_inv)

        face_model.save('face_encoder')
        face_model.save_weights('face_encoder.h5')
