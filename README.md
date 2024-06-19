# Deep-Learning-Applications-in-Image-Processing-Denoising-and-Vehicle-Re-identification
![Git Badge](https://img.shields.io/badge/-Git-blue?style=flat&logo=Git&logoColor=white)
[![Python Badge](https://img.shields.io/badge/-Python-blue?style=flat&logo=Python&logoColor=white)](https://www.python.org)
![TensorFlow Badge](https://img.shields.io/badge/-TensorFlow-blue?style=flat&logo=TensorFlow&logoColor=white)
![Keras Badge](https://img.shields.io/badge/-Keras-blue?style=flat&logo=Keras&logoColor=white)
![NumPy Badge](https://img.shields.io/badge/-NumPy-blue?style=flat&logo=NumPy&logoColor=white)
![sklearn Badge](https://img.shields.io/badge/-sklearn-blue?style=flat&logo=scikitlearn&logoColor=white)
![GCP Badge](https://img.shields.io/badge/-GCP-blue?style=flat&logo=googlecloud&logoColor=white)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eQZ96dTk6Q9DHhtXyC9PQJqW2aZGP7aH?usp=sharing)

## Image Denoising

### Visualisation of the dataset

To better understand our training dataset, we are displaying some sample.

<p align="center">
  <img src="image/original.png" alt="train" width="300"/>
</p>

Now we have added a random noise :
<p align="center">
  <img src="image/adding_noise.png" alt="train" width="300"/>
</p>

### Encoder - Decoder structure

Design a fully convolutional network with an encoder-decoder structure.

The encoder will consist of convolutional and pooling layers, and the decoder will use upsampling layers to reconstruct the denoised images.

More precisely, our autoencoder is a convolutional neural network with : an encoder that compresses 64x64 RGB images into an 8x8x256 representation, and a decoder that reconstructs the images back to their original size. It uses Conv2D, MaxPooling2D, and UpSampling2D layers, with ReLU activations in hidden layers and a sigmoid activation in the output layer also the model is optimized using the Adam optimizer and mean squared error loss.

<p align="center">
  <img src="image/autoencoder.png" alt="train" width="300"/>
</p>


### Training curve

<p align="center">
  <img src="image/training_curve.png" alt="train" width="300"/>
</p>


### Encoder - Decoder structure

The model performance on the noisy test images can be evaluated based on the visual quality of the denoised images and the quantitative test loss (MSE). The denoised images appear cleaner with less visible noise compared to the noisy inputs, and they closely resemble the original clean images. .

As we can the quality is very good and the test loss is very small **1 e-03**

<p align="center">
  <img src="image/result.png" alt="train" width="300"/>
</p>







