# Indoor Scene Recognition (R Implementation)

## Abstract

Indoor Scene Recognition is a challenging task in computer vision, requiring models to classify images into various indoor scenes commonly found in homes. This project explores the development and comparison of deep learning models for accurate scene classification in the R programming language. The goal is to identify the best-performing model and evaluate its performance on test data.

## Dataset

The dataset used in this project contains images of indoor scenes categorized into different room types, including bathrooms, bedrooms, kitchens, living rooms, and more. It is divided into training, validation, and test sets, providing the necessary data for model training and evaluation.

**Training Set:** This subset contains images used to train the deep learning models. It encompasses a range of indoor scenes, with the number of images for each scene category varying from 52 to 367. The training set is crucial for teaching the models to recognize different indoor scenes effectively.

**Validation Set:** The validation set is used to fine-tune the models during training and to prevent overfitting. It includes images from the same scene categories as the training set but is roughly half its size, providing an independent evaluation of model performance.

**Test Set:** The test set serves as the final evaluation stage for the trained models. It contains images representing indoor scenes, and the models' predictions are assessed based on their ability to classify these scenes accurately.

The dataset categorizes scenes into the following room types:
- Bathroom
- Bedroom
- Children's Room
- Closet
- Corridor
- Dining Room
- Garage
- Kitchen
- Living Room
- Stairs

## Project Objectives

1. **Model Development:** We create and experiment with four distinct deep learning systems in R, each with unique configurations, hyperparameters, and training settings. These systems encompass different architectures, hidden unit sizes, regularization techniques, and optimization strategies.

2. **Model Comparison:** We conduct a thorough comparison of the developed deep learning models in R. This comparison includes evaluating and discussing their performance, training characteristics, and predictive capabilities. The goal is to identify the most effective model for indoor scene recognition.

3. **Testing Performance:** To assess real-world performance, we use the test dataset to evaluate the chosen model's ability to correctly classify diverse indoor scenes using R.

## Getting Started

### Prerequisites

- R 4.x
- RStudio (optional but recommended)
- Necessary R packages (e.g., TensorFlow, Keras, tidyverse)

 ###  Usage
- Organize the dataset into the appropriate folders: train, validation, and test, with subfolders for each room category.
- Modify the data paths in the R scripts to match your dataset's location.
- Run the R scripts to train and evaluate the deep learning models.

## Models

### Model 1 - Deep Neural Network (DNN)

- Architecture: Three hidden layers with 256, 128, and 64 neuron units.
- Activation Function: ReLU for input layers and Softmax for output.
- Techniques: L2 Regularization, Adaptive Moment Estimation (Adam).
- Results: Training Accuracy (23.6%), Validation Accuracy (22%).

  ### Model 2 - Convolutional Neural Network (CNN)

- Architecture: Four convolutional layers with 32, 64, 128 filters.
- Techniques: Max pooling, dropout, and L2 regularization.
- Results: Training Accuracy (87%), Validation Accuracy (37%).

  ### Model 3 - Data Augmentation

- Architecture: Four convolutional layers.
- Techniques: Data augmentation, batch normalization.
- Results: Validation Accuracy (43%).

  ### Model 4 - Aggregation and Data Augmentation

- Architecture: Four convolutional layers.
- Techniques: Data augmentation, batch normalization.
- Results: Validation Accuracy (43%), Training Accuracy (57%).

  ## Conclusion

This project explored various deep learning models for indoor scene recognition. While each model exhibited strengths and weaknesses, Model 4, combining aggregation and data augmentation, demonstrated the highest accuracy. Further enhancements are necessary to improve predictive performance.


