# Handwritten Chinese Character Recognition using Deep Learning

This repository contains the practical implementation for the bachelor thesis:

**Development of a Software Module for Recognizing Handwritten Chinese Characters Using Deep Learning**

The project focuses on offline recognition of isolated handwritten Chinese characters using deep learning models. The work includes dataset preparation, preprocessing, model training, model comparison, result analysis, and implementation of a Streamlit-based software module.

---

## Project Overview

Handwritten Chinese character recognition is a challenging computer vision task because Chinese characters have complex structures, many visually similar forms, and high variability in handwriting styles.

The main goal of this project is to develop a software module that can recognize uploaded images of isolated handwritten Chinese characters and compare several deep learning models under the same training and evaluation setup.

---

## Dataset

The experiments are based on the prepared **CASIA_246** dataset.

The dataset is a reduced subset prepared from CASIA handwritten Chinese character data.

### Dataset statistics

| Property | Value |
|---|---:|
| Number of classes | 246 |
| Training images | 147,265 |
| Test images | 35,108 |
| Total images | 182,373 |
| Image type | Grayscale handwritten Chinese character images |
| Input size | 64×64 pixels |

The dataset was reduced to 246 classes to make the task computationally feasible while preserving the main challenges of handwritten Chinese character recognition, such as visually similar characters, stroke-level variation, and handwriting style differences.

Dataset link:  
**Kaggle:** https://www.kaggle.com/datasets/dina25020/casia-246 

Original CASIA handwriting database website:  
https://nlpr.ia.ac.cn/databases/handwriting/home.html

---

## Preprocessing and Augmentation

The following preprocessing steps were applied:

- convert images to grayscale;
- resize images to `64×64` pixels;
- convert images to tensor format;
- normalize images with mean `0.5` and standard deviation `0.5`.

For training, light data augmentation was used:

- random rotation up to ±5 degrees;
- small translation;
- scale variation from 0.95 to 1.05;
- small shear transformation.

---

## Implemented Models

Five deep learning models were implemented and compared:

| Model | Type | Purpose |
|---|---|---|
| Simple CNN | CNN | Baseline convolutional model |
| Improved CNN | CNN | Final best-performing model |
| ResNet18 | Residual CNN | Deeper residual architecture |
| LW-ViT | Vision Transformer | Lightweight transformer-based model |
| PF-ViT | Hybrid CNN + ViT | CNN stem with transformer encoder |

---

## Results

The models were evaluated using:

- test accuracy;
- test macro F1-score;
- loss;
- number of trainable parameters;
- model size.

### Final model comparison

| Model | Test Accuracy | Test Macro F1-score | Parameters | Model Size | Conclusion |
|---|---:|---:|---:|---:|---|
| Improved CNN | 0.9660 | 0.9660 | 3,779,222 | 14.43 MB | Best overall model |
| Simple CNN | 0.9619 | 0.9619 | 1,431,382 | 5.47 MB | Strong baseline |
| ResNet18 | 0.9601 | 0.9604 | 11,293,878 | 43.12 MB | Large, no accuracy gain |
| PF-ViT | 0.9319 | 0.9320 | 784,982 | 3.00 MB | Compact, lower accuracy |
| LW-ViT | 0.8736 | 0.8737 | 643,190 | 2.45 MB | Smallest model, lowest accuracy |

The best overall result was achieved by **Improved CNN**, with:

- **Test accuracy:** 96.60%
- **Test macro F1-score:** 96.60%

This model was selected as the final recognition model for the software module.

---

## Software Module

The software module was implemented as a **Streamlit web application**.

The application allows the user to:

- upload an image of a handwritten Chinese character;
- select a trained recognition model;
- receive the predicted Chinese character;
- view the confidence score;
- view the top-5 predictions.

Demo video:  
`PASTE_YOUR_VIDEO_DEMO_LINK_HERE`

---

## Repository Structure

```text
.
├── app/
│   ├── app.py
│   ├── inference.py
│   └── model_architecture.py
│
├── notebooks/
│   ├── 01_dataset_preparation.ipynb
│   ├── 02_simple_cnn_baseline.ipynb
│   ├── 03_improved_cnn_model.ipynb
│   ├── 04_resnet18_model.ipynb
│   ├── 05_lwvit_model.ipynb
│   ├── 06_pfvit_model.ipynb
│   └── 07_model_comparison.ipynb
│
├── results/
│   ├── figures/
│   ├── model_comparison_summary.csv
│   ├── model_comparison_final_table.csv
│   ├── model_comparison_conclusion.txt
│   └── model summary and cross-validation CSV files
│
├── requirements.txt
└── README.md
