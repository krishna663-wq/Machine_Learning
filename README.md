# Machine Learning Projects

## Crop Production Application

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

## Sentiment Analysis with Machine Learning

### Sentiment Analysis Notebook

You can view and run the Sentiment Analysis notebook on Google Colab using the following link:
[Sentiment_Analysis.ipynb](https://colab.research.google.com/github/krishna663-wq/Machine_Learning/blob/main/Sentiment_Analysis.ipynb)

This project demonstrates a basic sentiment analysis using a Naive Bayes classifier. The notebook is implemented in Google Colab and uses Python libraries such as pandas, scikit-learn, and NLTK.

### Overview

In this notebook, we perform sentiment analysis on a dataset of tweets. The goal is to classify tweets as positive or negative based on their content.

### Methodology

1. **Data Loading**: We use a CSV file containing pre-processed tweets.
2. **Data Preprocessing**: 
   - Replace sentiment labels.
   - Clean the text by removing URLs, mentions, hashtags, punctuations, and converting text to lowercase.
   - Remove stopwords using NLTK.
3. **Feature Extraction**: Convert text data into TF-IDF features.
4. **Model Training**: Split the data into training and testing sets. Train a Multinomial Naive Bayes model on the training set.
5. **Evaluation**: Evaluate the model's performance using accuracy score.

### Results

The model achieved an accuracy of **76.90%** on the test set.

### How to Run

1. **Dependencies**: 
   - pandas
   - scikit-learn
   - NLTK

2. **Steps**:
   - Open the notebook in Google Colab: [Sentiment_Analysis.ipynb](https://colab.research.google.com/github/krishna663-wq/Machine_Learning/blob/main/Sentiment_Analysis.ipynb)
   - Run all the cells to execute the code.

### Conclusion

This notebook provides a simple yet effective approach to sentiment analysis using machine learning techniques. The accuracy can be further improved by tuning the model and preprocessing steps.

## Bank Customer Segmentation

You can view and run the Bank Customer Segmentation notebook on Google Colab using the following link:
[Bank_Customer_Segmentation.ipynb](https://colab.research.google.com/github/krishna663-wq/Machine_Learning/blob/main/Bank_Customer_Segmentation.ipynb)

### Overview
This notebook focuses on segmenting bank customers based on their transaction data.

### Methodology
1. **Data Import and Exploration**: Load and explore the dataset `bank_transactions.csv`.
2. **Data Cleaning**: Handle missing values appropriately.
3. **Feature Encoding**: Encode categorical features using Label Encoding.
4. **Dimensionality Reduction**: Apply PCA for reducing the dimensionality of data.
5. **Clustering**: Use KMeans clustering to segment the customers.

### Results
The customers are segmented into different clusters based on their transaction patterns.

## Credit Card Fraud Detection

You can view and run the Credit Card Fraud Detection notebook on Google Colab using the following link:
[Credit_card_fraudTrain_Detection.ipynb](https://colab.research.google.com/github/krishna663-wq/Machine_Learning/blob/main/Credit_card_fraudTrain_Detection.ipynb)

### Overview
This notebook focuses on detecting fraudulent credit card transactions.

### Methodology
1. **Data Import and Exploration**: Load and explore the dataset `fraudTrain.csv`.
2. **Data Cleaning**: Handle missing values and encode categorical variables.
3. **Data Visualization**: Visualize the distribution of fraudulent transactions and correlations.
4. **Model Training**: Train a machine learning model to detect fraud.

### Results
The model's performance is evaluated using accuracy, confusion matrix, and other metrics.

## Diabetes Prediction

You can view and run the Diabetes Prediction notebook on Google Colab using the following link:
[Diabetes_price_Prediction_LAb2.ipynb](https://colab.research.google.com/github/krishna663-wq/Machine_Learning/blob/main/Diabetes_price_Prediction_LAb2.ipynb)

### Overview
This notebook focuses on predicting the likelihood of diabetes based on patient data.

### Methodology
1. **Data Import and Exploration**: Load and explore the dataset `diabetes_prediction_dataset.csv`.
2. **Data Cleaning**: Handle missing values and encode categorical variables.
3. **Feature Engineering**: Create new features and transform existing ones.
4. **Model Training**: Train a machine learning model to predict diabetes.

### Results
The model's performance is evaluated using accuracy and other metrics.

## Boston House Price Prediction

You can view and run the Boston House Price Prediction notebook on Google Colab using the following link:
[boston_HousePrediction_Lab2.ipynb](https://colab.research.google.com/github/krishna663-wq/Machine_Learning/blob/main/boston_HousePrediction_Lab2.ipynb)

### Overview
This notebook focuses on predicting house prices in Boston based on various features.

### Methodology
1. **Data Import and Exploration**: Load and explore the dataset `boston.csv`.
2. **Data Cleaning**: Handle missing values and outliers.
3. **Feature Engineering**: Create new features and transform existing ones.
4. **Model Training**: Train a Linear Regression model to predict house prices.
5. **Evaluation**: Evaluate the model's performance using metrics like R-squared and Mean Squared Error.

### Results
The model's performance is evaluated using R-squared and Mean Squared Error metrics.





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
