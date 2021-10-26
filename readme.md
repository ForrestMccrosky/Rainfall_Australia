# Predicting Australian Rainfall: The Next Day

## Project Description
 - This purpose of this project is to create a classification model that predicts whether it will rain in Australia tomorrow after each observation in the dataset
 - Project created using the data science pipeline (Acquisition, Preparation, Exploration, Analysis & Statistical Testing, and finally Modeling and Evaluation)

## Project Goals
 - Create a Final Jupyter Notebook that reads like a report and follows the data science pipeline
 - In the Jupyter Notebook Create a classification model that our performs the baseline
 - Create Function Files to help peers execute project reproduction
 - Create a baseline Machine Learning Model that creates a baseline for easy prediction of weather for each future day in Australia 

  ## Deliverables
 - Final Jupyter Notebook
 - Function Files for reproduction
 - Detailed Readme

## Executive Summary
- The purpose of this notebook is to acquire, prep, explore a csv file downloaded from kaggle.com that contains Australian rainfall data and use it to predict whether it will rain tomorrow (the day after each observations recorded date)
 - Target variable: raintom
 - After visual exploration and statistical testing the features that were inputted into our models were

    - Humidity3pm
    - raintod
    - Rainfall
    - Humidity9am
    - Pressure3pm
    - Pressure9am

Our most successful model that was used on our Out-of-sample (test) dataframe was the K Neareast Neighbor model which performed with the folling metrics:

 - Accuracy: 86.61%
 - True Positive Rate: 55.21%
 - True Negative Rate: 95.54%
 - False Positive Rate: 4.45%
 - False Negative Rate: 44.78%

Overall the modeling and project was a success and you can follow the steps below to reproduce it!

Link to dataset: https://www.kaggle.com/jsphyg/weather-dataset-rattle-package?select=weatherAUS.csv



