# ChestX: Lung X-ray Classification

This repository contains a deep learning model for classifying chest X-ray images into three categories: **COVID-19**, **Normal**, and **Pneumonia**. The project involves data preprocessing, dimensionality reduction techniques (PCA, t-SNE, LLE, MDS), clustering (K-Means, GMM), and a convolutional neural network (CNN) for image classification. The goal is to assist in the rapid diagnosis of lung-related diseases, particularly COVID-19, using chest X-ray images.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Dimensionality Reduction](#dimensionality-reduction)
4. [Clustering](#clustering)
5. [CNN Model](#cnn-model)
6. [Results](#results)

## Introduction

The COVID-19 pandemic has highlighted the need for rapid and accurate diagnostic tools. Chest X-rays are a common diagnostic tool for lung-related diseases, and deep learning models can be used to automate the classification of these images. This project aims to classify chest X-ray images into three categories: **COVID-19**, **Normal**, and **Pneumonia**.

## Dataset

The dataset used in this project is a combination of publicly available chest X-ray images. The dataset is divided into three classes:

- **COVID-19**: 417 images
- **Normal**: 406 images
- **Pneumonia**: 404 images

### Dataset Split
- **Training Set**: 903 images
- **Validation Set**: 324 images

## Dimensionality Reduction

### PCA (Principal Component Analysis)
- **Number of Components for 90% Variance**: 75 components are required to preserve 90% of the variance.
- **Cumulative Variance Plot**: The cumulative variance plot shows the number of components needed to capture the variance in the data.

### t-SNE (t-Distributed Stochastic Neighbor Embedding)
- **Clustering Observations**: t-SNE provides a more meaningful clustering, preserving local structures and creating visually distinguishable groups.

### LLE (Locally Linear Embedding)
- **Clustering Observations**: LLE preserves local distances but is ineffective in distinguishing between clusters. Data points are packed tightly along a narrow band.

### MDS (Multidimensional Scaling)
- **Clustering Observations**: MDS provides a more global view of data distribution with better separation but still lacks strong cluster distinction.

## Clustering

### K-Means Clustering
- **Optimal Number of Clusters**: Determined using the Elbow Method and Silhouette Analysis.
- **Accuracy**: 53.82%
- **Cluster Visualization**: The clusters are visualized in a 2D space using PCA.

### Gaussian Mixture Model (GMM)
- **Optimal Number of Clusters**: Determined using BIC and AIC.
- **Accuracy**: 55.48%
- **Cluster Visualization**: The clusters are visualized in a 2D space using PCA.

## CNN Model

### Model Architecture
- **Input Shape**: (224, 224, 3)
- **Convolutional Layers**: 3 layers with 32, 64, and 128 filters respectively.
- **Pooling Layers**: MaxPooling after each convolutional layer.
- **Dense Layers**: 2 dense layers with 128 and 8 units respectively.
- **Output Layer**: Dense layer with 3 units (one for each class) and softmax activation.

### Training
- **Optimizer**: Adam with a learning rate of 0.0001
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 20

### Results
- **Training Accuracy**: 99.65%
- **Validation Accuracy**: 93.52%
- **Overfitting**: Signs of overfitting appear after around the 10th epoch.

## Results

### Model Performance
- **Training Loss**: Decreases steadily, reaching very low values.
- **Validation Loss**: Decreases up to a point but then fluctuates and increases slightly toward the later epochs.
- **Training Accuracy**: Increases steadily and reaches very high levels (~99%).
- **Validation Accuracy**: Follows a similar trend initially, peaking around 95%, but fluctuates after certain epochs.

### Overfitting
- **Observation**: The divergence between training and validation losses and the fluctuations in validation accuracy suggest potential overfitting.

## Usage

### Prerequisites
- Python 3.7 or higher
- TensorFlow 2.0 or higher
- Keras
- NumPy
- Matplotlib
- Scikit-learn

## Acknowledgments

- Thanks to the creators of the COVID-19 Radiography Database and the Chest X-Ray Images (Pneumonia) dataset for making their data publicly available.
- Special thanks to the TensorFlow and Keras communities for their excellent documentation and tutorials.
- Thanks to Kratik Rathi for providing this excellent and easy to understand solution.

---

For any questions or issues, please open an issue on the GitHub repository or contact the maintainer directly.

**Happy Coding!** 🚀
"# ChestX-Lung-X-ray-Images-Classification" 
