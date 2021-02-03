# Be Your Voice
We want to predict people's voice from their face.  
Particularly, we use a 64x64 RGB face image to predict (generate) the mel spectrogram of corresponding voice.  
The mel spectrogram can be transformed into raw audio (around 0.5 second).

## Dataset
using the [VoxCeleb 1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) dataset

## Model Architecture
We came up with two different architectures, which are autoencoder-based and GAN-based, to achieve our goal.  
However, only the GAN-based model works.

For the detailed architecture and training hyperparameters, please refer to the code.

### Autoencoder Based (Failed)
This architecture is adapted from the model in *Speech2Face* \[1\].  
![](https://github.com/25349023/Be-Your-Voice/blob/master/architecture1.png)

### GAN Based
We use deep convolutional GAN, also known as DCGAN, with face images as the condition, to generate the mel spectrogram.  
![](https://github.com/25349023/Be-Your-Voice/blob/master/architecture2.png)

## Experimental Results
In addition to the raw audio, we also attempted to combine it with existing TTS technology - *Real-Time-Voice-Cloning* [2].  
The files in the following folders are the results of our experiments. All single numbered files are the generated raw audios, and double numbered files are the corresponding speech audios output from TTS.  

[male 1](http://bit.ly/beurvoice_lowmale)  
[male 2](http://bit.ly/beurvoice_lightmale)  
[female](http://bit.ly/beurvoice_female)

## Authors
[@25349023](https://github.com/25349023)  
[@jokejay](https://github.com/jokejay)  
[@AC-god](https://github.com/ac-god)

## References
\[1\]  [Speech2Face: Learning the Face Behind a Voice](https://speech2face.github.io/)  
\[2\]  [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning)
