import warnings
warnings.filterwarnings('ignore')

from math import sqrt
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

import numpy as np
import pandas as pd
import seaborn as sns


############################# Function File for Preparing Data ############################

def drop_duplcites_and_nulls(df):
    '''
    This function is designed to to take in the dataframe and remove certain columns that had more
    than 50 percent null values and then remove the rows with less than 10 percent duplicates
    '''
    
    df = df.drop_duplicates() ## dropping duplicates
    
    ## dropping the columns with about 50 percent or more null values
    df = df.drop(columns = ['Cloud3pm', 'Cloud9am', 'Sunshine', 'Evaporation'])
    
    df = df.dropna() ## dropping the other null because they are now 10 percent less than the 
    ## amount of observations 
    
    return df

def strip_strings(df):
    '''
    Stripping string valued columns to avoid complications in exploration and modeling
    phases of the pipeline
    '''
    
    ## stripping trailing and leading whitespace from string valued columns just in case

    df.model = df.Location.str.strip()
    df.fuelType = df.WindGustDir.str.strip()
    df.transmission = df.WindDir9am.str.strip()
    df.transmission = df.WindDir3pm.str.strip()
    df.transmission = df.RainToday.str.strip()
    df.transmission = df.RainTomorrow.str.strip()
    
    return df

def split_data(df):
    '''
    This function is designed to split out data for modeling into train, validate, and test 
    dataframes.
    
    It will also perform quality assurance checks on each dataframe to make sure the target 
    variable was correctcly stratified into each dataframe.
    '''
    
    ## splitting the data stratifying for out target variable is_fraud
    train_validate, test = train_test_split(df, test_size=.2, 
                                            random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                       random_state=123)
    
    print('Making Sure Our Shapes Look Good')
    print(f'Train: {train.shape}, Validate: {validate.shape}, Test: {test.shape}')
    
    return train, validate, test