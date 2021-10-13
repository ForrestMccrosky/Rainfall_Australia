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


def strip_strings(df):
    '''
    This function is designed to take in a dataframe and strip the trailing and leading whitespace
    from the columns with string values in case they are there becuase this will cause issues in
    exploration and modeling.
    '''
    
    ## stripping trailing and leading whitespace from string valued columns just in case

    df.model = df.model.str.strip()
    df.fuelType = df.fuelType.str.strip()
    df.transmission = df.transmission.str.strip()
    
    return df

def drop_dups_create_features(df):
    '''
    This function is designed to completely prepare the data before the train, validate, test 
    split for modeling and exploration.
    
    It will drop duplicate rows of information and create one hot encoded features from some
    of the categorical columns in the original dataframe.
    '''
    
    ## dropping the duplicates 
    df = df.drop_duplicates()
    
    ## using one hot encoding to get dummy categorical columns for diesel, electric, and hybrid
    ## fuel types
    df['is_diesel'] = np.where(df.fuelType == 'Diesel', 1, 0)
    df['is_electric'] = np.where(df.fuelType == 'Electric', 1, 0)
    df['is_hybrid'] = np.where(df.fuelType == 'Hybrid', 1, 0)
    
    ## using one hot encoding to get dummy categorical columns for manual, automatic, 
    ## and semi automatic transmission types
    df['is_manual'] = np.where(df.transmission == 'Manual', 1, 0)
    df['is_semiauto'] = np.where(df.transmission == 'Semi-Auto', 1, 0)
    df['is_automatic'] = np.where(df.transmission == 'Automatic', 1, 0)
    
    return df

def double_checker(df):
    '''
    This function is designed to outpute categorical value counts for the original columns 
    in the dataframe and then output the value counts for our one hot encoded columns to make
    sure they were done correctly.
    '''
    
    ## looking at certain object value counts    
    ## Looking at the fuel type values so I can one hot encode different columns for each fuel
    ## type

    print(df.fuelType.value_counts())
    print('--------------------\n')
    list = ['is_diesel', 'is_electric', 'is_hybrid']
    
    for x in list:
        print(f'Value Counts for {x}:\n')
        print(df[x].value_counts())
        print('--------------------')
        
    
    print(df.transmission.value_counts())
    print('--------------------\n')

    list = ['is_manual', 'is_semiauto', 'is_automatic']

    for x in list:
        print(f'Value Counts for {x}:\n')
        print(df[x].value_counts())
        print('--------------------')

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