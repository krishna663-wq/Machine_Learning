
# Crop Production Application

You can view and run the Crop Production Application notebook on Google Colab using the following link:
[1Crop_Production_applicationofMl.ipynb](https://colab.research.google.com/github/krishna663-wq/Machine_Learning/blob/main/1Crop_Production_applicationofMl.ipynb)

### Overview
This notebook focuses on analyzing and predicting crop production based on historical data.

### Methodology
1. **Data Import and Exploration**: Load and explore the dataset `crop_production.csv`.
2. **Data Cleaning**: Handle missing values appropriately.
3. **Feature Selection and Splitting**: Select features `Area` and `Production`, and split the dataset into training and testing sets.
4. **Model Training**: Train a Linear Regression model using the training data.
5. **Prediction and Evaluation**: Use the model to predict crop production on the test data and calculate the Mean Squared Error (MSE).

### Results
The model's performance is evaluated using the Mean Squared Error (MSE).





# Sentiment Analysis with Machine Learning

## Sentiment Analysis Notebook

You can view and run the Sentiment Analysis notebook on Google Colab using the following link:
[Sentiment_Analysis.ipynb](https://colab.research.google.com/github/krishna663-wq/Machine_Learning/blob/main/Sentiment_Analysis.ipynb)

This project demonstrates a basic sentiment analysis using a Naive Bayes classifier. The notebook is implemented in Google Colab and uses Python libraries such as pandas, scikit-learn, and NLTK.

## Overview

In this notebook, we perform sentiment analysis on a dataset of tweets. The goal is to classify tweets as positive or negative based on their content.

## Methodology

1. **Data Loading**: We use a CSV file containing pre-processed tweets.
2. **Data Preprocessing**: 
   - Replace sentiment labels.
   - Clean the text by removing URLs, mentions, hashtags, punctuations, and converting text to lowercase.
   - Remove stopwords using NLTK.
3. **Feature Extraction**: Convert text data into TF-IDF features.
4. **Model Training**: Split the data into training and testing sets. Train a Multinomial Naive Bayes model on the training set.
5. **Evaluation**: Evaluate the model's performance using accuracy score.

## Results

The model achieved an accuracy of **76.90%** on the test set.

## How to Run

1. **Dependencies**: 
   - pandas
   - scikit-learn
   - NLTK

2. **Steps**:
   - Open the notebook in Google Colab: [Sentiment_Analysis.ipynb](https://colab.research.google.com/github/krishna663-wq/Machine_Learning/blob/main/Sentiment_Analysis.ipynb)
   - Run all the cells to execute the code.

## Conclusion

This notebook provides a simple yet effective approach to sentiment analysis using machine learning techniques. The accuracy can be further improved by tuning the model and preprocessing steps.
