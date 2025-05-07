🩺 Lung Cancer Detection Using CNN
This project implements a Convolutional Neural Network (CNN) to classify lung CT scan images into three categories: benign, malignant, and normal. It utilizes TensorFlow (Keras API) for model building and training, and was developed in a Google Colab environment.

📌 Overview
Lung cancer is among the most prevalent and deadly forms of cancer. Accurate and early classification of lung nodules can significantly improve prognosis. This project aims to assist in automatic diagnosis using a deep learning approach applied to grayscale medical images.

📁 Repository Contents
lung_cancer_detection.ipynb: Google Colab notebook with all implementation steps.

README.md: Project description and usage instructions (this file).

(Optional) Supporting files like sample images, model weights, or environment setup can be added later.

🛠️ Technologies Used
Python

OpenCV

TensorFlow / Keras

Scikit-learn

imbalanced-learn (for oversampling)

NumPy

Matplotlib

🧠 Model Architecture
The CNN is designed as follows:

Input: Grayscale CT scan images resized to 128x128

Conv2D + MaxPooling2D layers for feature extraction

Dropout layers to reduce overfitting

Flatten + Dense layers for classification

Softmax activation at the output for multi-class classification (3 classes)

🚀 How to Run
Open lung_cancer_detection.ipynb in Google Colab.

Mount your Google Drive where the dataset is stored.

Run all cells sequentially to:

Load and preprocess the image data

Balance the dataset using RandomOverSampler

Build and train the CNN model

Evaluate the model and visualize metrics

📁 Dataset: The project uses the IQ-OTH/NCCD Lung Cancer Dataset. It must be manually uploaded to Google Drive due to its size.

📊 Results
The model was trained on balanced data using oversampling.

Metrics such as accuracy, classification report, and confusion matrix were used for evaluation.

Visualization includes:

Training & validation loss/accuracy curves

Confusion matrix plot

You can customize the dataset path and parameters to fine-tune the performance.
