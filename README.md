📌 Project Overview

Loads the MNIST dataset (60,000 training and 10,000 testing images of digits 0–9).

Preprocesses images by reshaping them into (28,28,1) and normalizing pixel values.

Builds a CNN model with:

Conv2D layers for feature extraction

MaxPooling2D layers for down-sampling

Flatten layer to convert features into a 1D vector

Dense layers for classification

Softmax output layer for predicting digits 0–9

Compiles the model using Adam optimizer and categorical crossentropy loss.

Trains and evaluates the CNN on MNIST dataset.

Predicts digits from custom external images after preprocessing.

⚙️ Tech Stack

Python

TensorFlow / Keras

NumPy

Matplotlib

PIL (Pillow)

scikit-learn

🚀 How It Works

Training Phase

CNN is trained on MNIST dataset.

Uses Conv2D + MaxPooling layers for feature learning.

Dense layers classify the digits.

Evaluation

Model is tested on 10,000 unseen MNIST images.

Accuracy score and confusion matrix are generated.

Custom Image Prediction

External handwritten digit images are preprocessed (grayscale, resized to 28×28, normalized).

Model predicts digit and displays result with Matplotlib.

📊 Results

Achieves higher accuracy compared to ANN.

Confusion matrix shows strong classification performance.

Correctly predicts most custom handwritten images.
