# Handwritten Digit Classification using Convolutional Neural Networks (CNN)

This project focuses on classifying handwritten digits (0–9) from the MNIST dataset using Convolutional Neural Networks (CNNs). It leverages TensorFlow and Keras for model building and training. The project includes different CNN architectures to compare performance and accuracy.

---

## Installation

### Prerequisites
- Python 3.6+
- TensorFlow
- Keras
- Scikit-learn
- Matplotlib
- Seaborn
- NumPy

---

## Setup

1. **Clone this repository or download the code.**

2. **Install the required dependencies:**
   ```bash
   pip install tensorflow keras scikit-learn matplotlib seaborn numpy
Download the dataset:

The MNIST dataset is available directly via tensorflow.keras.datasets.

No manual download is required.

Usage

Clone the repository and open it in VS Code.
Make sure you have Python and required packages installed (see Requirements section).

Dataset Loading

The MNIST dataset can be loaded directly in Python:

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

Training

Run the notebook or Python script

Open the .ipynb notebook or .py script in VS Code.

Run the cells or execute the script as needed.

Steps included:

Loading and visualizing the dataset

Preprocessing the images (reshaping, normalization, one-hot encoding)

Building and training the CNN model

Evaluating the trained model

Visualizing results with confusion matrices

3.Models
Basic CNN Model

A simple convolutional network designed as an introduction to image classification tasks.
Architecture example:

Conv2D → MaxPooling → Flatten → Dense → Output

Advanced CNN Model (Optional)

An enhanced CNN with additional convolutional layers, dropout, and batch normalization to improve generalization and accuracy.

Dataset

The dataset used in this project is the MNIST dataset consisting of 70,000 grayscale images of handwritten digits (0–9):

Training set: 60,000 images

Test set: 10,000 images

Each image is 28x28 pixels.

Final Prediction

At the end of the code, a random digit image is uploaded (or selected from the test set) to test the trained CNN model.
The model, saved as an .h5 file, is loaded and used for prediction:

from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = load_model("digit_cnn_model.h5")

# Example: testing a random image from the test set
index = 1234
plt.imshow(x_test[index].reshape(28,28), cmap="gray")
plt.show()

pred = model.predict(x_test[index].reshape(1,28,28,1))
print("Predicted Digit:", np.argmax(pred))


✅ The model correctly predicts the digit class (0–9).

Acknowledgements

The MNIST dataset is provided by Yann LeCun and widely used as a benchmark dataset in computer vision.

This project was inspired by the need to build a fundamental understanding of CNNs for image classification tasks.


Now it’s a **single continuous `README.md` file** — copy-paste friendly for GitHub.  

Do you also want me to add **Accuracy/Loss plot and Confusion Matrix visualization instructions** inside this same README (like a professional repo), or keep it simple
'''
