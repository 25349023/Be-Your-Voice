import os
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import vgg16
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


class NormalizedLogMelSpectrogram(tf.keras.layers.Layer):
    """Compute log-magnitude mel-scaled spectrograms."""

    def __init__(self, sample_rate=16000, fft_size=2048, hop_size=160, window_len=400, n_mels=128,
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
        NormalizedLogMelSpectrogram()
    ])
    return model


def generator_model():
    noise_z = tf.keras.Input([128])
    face_cond = tf.keras.Input([64, 64, 3])

    seq_z = tf.keras.Sequential([
        tf.keras.layers.Dense(4 * 4 * 512, use_bias=False),
        tf.keras.layers.Reshape([4, 4, 512]),
        tf.keras.layers.BatchNormalization()
    ], name='seq_z')

    seq_f = tf.keras.Sequential([
        tf.keras.layers.Conv2D(6, 3, 2, padding='same', use_bias=False),
        tf.keras.layers.Reshape([4, 4, 384]),
        tf.keras.layers.BatchNormalization()
    ], name='seq_f')

    proj_z = seq_z(noise_z)
    proj_f = seq_f(face_cond)
    inputs = tf.keras.layers.Concatenate()([proj_z, proj_f])

    seq_c = tf.keras.Sequential([
        tf.keras.layers.Conv2DTranspose(512, 5, 2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Conv2DTranspose(256, 5, 2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Conv2DTranspose(256, 5, (2, 4), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Conv2DTranspose(1, 5, 2, padding='same')
    ], name='seq_c')

    outputs = seq_c(inputs)
    model = tf.keras.Model(inputs=[noise_z, face_cond], outputs=outputs)
    return model


def discriminator_model():
    xs = tf.keras.Input([64, 128, 1])
    face_cond = tf.keras.Input([64, 64, 3])

    noised_xs = xs + tf.random.normal(tf.shape(xs), mean=0.0, stddev=0.2)

    seq1 = tf.keras.Sequential([
        tf.keras.layers.Conv2D(128, 5, 2, padding='same'),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Conv2D(256, 5, (2, 4), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Conv2D(256, 5, 2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(0.2)
    ], name='seq_1')
    conv_x = seq1(noised_xs)

    ext_f = tf.keras.Sequential([
        tf.keras.layers.Conv2D(1, 5, 4, padding='same', use_bias=False),
        tf.keras.layers.Conv2D(2, 3, 2, padding='same', use_bias=False),
        tf.keras.layers.Reshape([1, 1, 128])
    ], name='ext_f')
    face_y = ext_f(face_cond)

    face_y = face_y * tf.ones([1, *conv_x.shape[1:3], 128])
    conv = tf.keras.layers.Concatenate()([conv_x, face_y])

    seq2 = tf.keras.Sequential([
        tf.keras.layers.Conv2D(512, 5, 2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ], name='seq_2')

    outputs = seq2(conv)
    model = tf.keras.Model(inputs=[xs, face_cond], outputs=outputs)
    return model


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


@tf.function
def train_step(lm, gen, dis, audio, face_cond):
    noise = tf.random.normal([audio.shape[0], 128])
    mel_spec = lm(audio)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_audio = gen((noise, face_cond), training=True)

        real_output = dis((mel_spec, face_cond), training=True)
        fake_output = dis((generated_audio, face_cond), training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, gen.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, dis.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, dis.trainable_variables))
    return gen_loss, disc_loss


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
    name_train = id_to_names[id_train].values

    pair_ds = tf.data.Dataset.from_tensor_slices((id_train, name_train, url_train))
    pair_ds = pair_ds.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).shuffle(3000).batch(64).prefetch(2)

    cross_entropy = tf.keras.losses.BinaryCrossentropy()

    generator_optimizer = tf.keras.optimizers.Adam(1.5e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(0.8e-5)

    log_mel = logmel_model()
    generator = generator_model()
    discriminator = discriminator_model()

    gen_losses = []
    dis_losses = []

    for epoch in range(30):
        print('Epoch', epoch)
        for step, (audios, images) in enumerate(pair_ds):
            g_loss, d_loss = train_step(log_mel, generator, discriminator, audios, images)
            gen_losses.append(g_loss)
            dis_losses.append(d_loss)
            if step % 40 == 0:
                print(f'  step {step}, G loss = {g_loss}, D loss = {d_loss}')

    if test:
        plt.plot(gen_losses[::150], label='Gen. loss')
        plt.plot(dis_losses[::150], label='Dis. loss')
        plt.legend()

        img = plt.imread('test_img/1.png')
        plt.imshow(img)
        pimg = preprocess_image(img)

        noise = tf.random.normal([10, 128])
        s = generator((noise, tf.tile(pimg[None], [10, 1, 1, 1])))

        db = (s.numpy() - 1) * 40
        pw = librosa.db_to_power(db)

        y_invs = []
        for p in pw:
            y_inv = librosa.feature.inverse.mel_to_audio(p.squeeze().T, 16000, fmax=8000,
                                                         window='hamming', win_length=400, hop_length=160)
            y_invs.append(y_inv)

        librosa.display.specshow(db[0].squeeze().T)
        plt.colorbar()

        librosa.display.waveplot(y_invs[0])

        generator.save(f'generator')
        generator.save_weights(f'generator.h5')
        discriminator.save(f'discriminator')
        discriminator.save_weights(f'discriminator.h5')
