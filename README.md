# Leprosy Detection Model

## Overview

This repository contains a deep learning model trained to detect leprosy from images. Leprosy, also known as Hansen's disease, is a chronic infectious disease that primarily affects the skin, peripheral nerves, and mucosal surfaces of the upper respiratory tract and eyes. Early detection and treatment are crucial for preventing disability and further transmission of the disease.

## Dataset

The dataset used for training and evaluation is not provided in this repository due to privacy and licensing restrictions. However, you can use your own dataset or obtain relevant data from medical research organizations or repositories.

## Model Architecture

The CNN architecture used for this model consists of multiple convolutional and pooling layers, followed by fully connected layers. The final layer utilizes a sigmoid activation function to output the probability of the input image belonging to the leprosy-positive class.

## Usage

To use the model for leprosy detection:

1. Clone this repository to your local machine.
2. Ensure you have TensorFlow and other required dependencies installed (`requirements.txt`).
3. Prepare your dataset or obtain relevant images for testing.
4. Load the trained model using TensorFlow's `tf.keras.models.load_model()` function.
5. Use the model to predict leprosy from input images.

```python

