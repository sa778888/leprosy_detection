# leprosy_detection
This repository contains a deep learning model trained to detect leprosy from images. Leprosy, also known as Hansen's disease, is a chronic infectious disease that primarily affects the skin, peripheral nerves, and mucosal surfaces of the upper respiratory tract and eyes. Early detection and treatment are crucial for preventing disability and further transmission of the disease.

Overview
The model is built using TensorFlow and Keras, leveraging convolutional neural networks (CNNs) to learn patterns and features from input images. It is trained on a dataset consisting of images of skin lesions, categorized as either leprosy-positive or leprosy-negative.

Dataset
The dataset used for training and evaluation is not provided in this repository due to privacy and licensing restrictions. However, you can use your own dataset or obtain relevant data from medical research organizations or repositories.

Model Architecture
The CNN architecture used for this model consists of multiple convolutional and pooling layers, followed by fully connected layers. The final layer utilizes a sigmoid activation function to output the probability of the input image belonging to the leprosy-positive class.

Usage
To use the model for leprosy detection:

Clone this repository to your local machine.
Ensure you have TensorFlow and other required dependencies installed (requirements.txt).
Prepare your dataset or obtain relevant images for testing.
Load the trained model using TensorFlow's tf.keras.models.load_model() function.
Use the model to predict leprosy from input images.
python
Copy code
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the trained model
loaded_model = tf.keras.models.load_model('trained_model.h5')

# Load and preprocess test image
test_image = image.load_img('test_image.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = tf.expand_dims(test_image, axis=0)

# Predict leprosy from the test image
result = loaded_model.predict(test_image)

if result[0][0] == 1:
    prediction = 'Leprosy Detected'
else:
    prediction = 'No Leprosy Detected'

print(prediction)
Contributing
Contributions to improve the model's performance, add features, or fix bugs are welcome. Please open an issue or pull request to discuss changes or propose enhancements.
