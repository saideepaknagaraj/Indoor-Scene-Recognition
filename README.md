
# 🏠 Indoor Scene Recognition (R Implementation)

## 📘 Overview

Indoor Scene Recognition is a complex computer vision task focused on classifying indoor environments — such as bedrooms, kitchens, and bathrooms — from images. This project explores the implementation of multiple deep learning models in **R** to solve this challenge. Our objective is to compare their performance and determine the best model for classifying indoor scenes accurately.

---

## 🎯 Objectives

- 🧠 **Model Development**: Implement four deep learning models in R using different configurations and learning strategies.
- ⚖️ **Model Comparison**: Evaluate the models using training/validation/test sets and analyze their accuracy and generalization.
- 🧪 **Performance Testing**: Measure real-world performance on unseen test data.

---

## 🗂️ Dataset Description

The dataset includes categorized images of various indoor scenes:
- 📷 **Categories**: Bathroom, Bedroom, Children’s Room, Closet, Corridor, Dining Room, Garage, Kitchen, Living Room, Stairs
- 🔹 **Training Set**: 52–367 images per category used for learning patterns.
- 🔹 **Validation Set**: Roughly half the size of the training set, used to tune hyperparameters and prevent overfitting.
- 🔹 **Test Set**: Used to assess final model performance on unseen examples.

Organize the data as:

```
dataset/
├── train/
│   ├── Bedroom/
│   ├── Kitchen/
│   └── ...
├── validation/
│   ├── Bedroom/
│   ├── Kitchen/
│   └── ...
└── test/
    ├── Bedroom/
    ├── Kitchen/
    └── ...
```

---

## 🧰 Getting Started

### 🛠 Prerequisites

- **R 4.x**
- **RStudio** (recommended)
- Required packages:
```R
install.packages(c("keras", "tensorflow", "tidyverse"))
```

### 🚀 Usage

1. Download and organize your dataset as described above.
2. Update file paths in the R scripts (`model1.R`, `model2.R`, etc.).
3. Run scripts to train and evaluate each model.
4. Compare the results.

---

## 🤖 Models Overview

### 🔹 Model 1 — Deep Neural Network (DNN)
- Architecture: 3 hidden layers (256, 128, 64 units)
- Activation: ReLU (hidden), Softmax (output)
- Techniques: L2 regularization, Adam optimizer
- 🧪 Results: 
  - Training Accuracy: **23.6%**
  - Validation Accuracy: **22%**

---

### 🔸 Model 2 — Convolutional Neural Network (CNN)
- Architecture: 4 convolutional layers (32–128 filters)
- Techniques: Max Pooling, Dropout, L2 regularization
- 🧪 Results: 
  - Training Accuracy: **87%**
  - Validation Accuracy: **37%**

---

### 🟠 Model 3 — CNN with Data Augmentation
- Techniques: Data augmentation, batch normalization
- 🧪 Results: 
  - Validation Accuracy: **43%**

---

### 🟣 Model 4 — Aggregation + Data Augmentation
- Techniques: Aggregated CNN layers, augmentation, batch normalization
- 🧪 Results:
  - Training Accuracy: **57%**
  - Validation Accuracy: **43%**

---

## 📈 Summary Table

| Model | Architecture                   | Techniques                         | Training Acc | Validation Acc |
|-------|--------------------------------|------------------------------------|--------------|----------------|
| DNN   | 3 Dense Layers                 | L2, Adam                           | 23.6%        | 22%            |
| CNN   | 4 Conv Layers                  | Dropout, MaxPool, L2               | 87%          | 37%            |
| Aug   | CNN + Augmentation             | Data Augment, BatchNorm            | -            | 43%            |
| Final | Aggregated CNN + Augmentation | Augment, BatchNorm, Aggregation    | 57%          | **43%**        |

---

## ✅ Conclusion

Model 4, combining convolutional aggregation and data augmentation, showed the most promising performance in indoor scene classification using R. While improvements are still needed, this project demonstrates the potential of deep learning in structured visual recognition tasks in home environments.

---

## 👨‍💻 Author

*Developed as part of an academic deep learning project in R.*

---

## 📎 License

MIT License
