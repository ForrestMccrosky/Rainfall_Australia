## import statements for the regression pipeline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns


# feature selection imports
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import RFE

# import scaling methods
from sklearn.preprocessing import RobustScaler, StandardScaler
from scipy import stats
from sklearn.model_selection import train_test_split

# import modeling methods
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import explained_variance_score
from scipy import stats

# import to remove warnings
import warnings
warnings.filterwarnings("ignore")



####################### Acquire Zillow Database Function ##########################

def get_bmw_data():
    '''
    This function is designed to pull our used bmw carmdata from the csv file into 
    a pandas dataframe then return the dataframe.
    
    It will also print out the shape of the dataframe after removing an unneccasry column
    '''
    df = pd.read_csv('bmw.csv', index_col=0) ## reading our csv into a pandas dataframe
    
    df = df.reset_index() ## we also are going to reset the index so that the model information
    ## is a useable column

    return df 

    print("Shape of Dataframe (rows, columns):\n")
    print(df.shape)  ## look at shape
    

def acquire_stats(df):
    '''
    This function is designed to take in our dataframe and display the numerical statistics and 
    goes the extra step to add a range column for the numerical columns
    '''
    
    ## taking one step further and adding a range column

    stats_df = df.describe().T

    stats_df['range'] = stats_df['max'] - stats_df['min'] 

    return stats_df

def summarize_df(df):
    '''
    this function is designed to look at our dataframe and print out a short summary of the
    dataframe.
    
    This will include things like:
    info on the columns and data types
    value counts of categorical columns
    '''
    
    print('Info on Columns and Datatypes:\n')
    print(df.info()) ## <-- looking at our columns and datatypes
    print('------------------------------------------------\n')
    
    ## creating a list of columns I want value counts for
    list = ['fuelType', 'transmission']
    
    ## using list comprehension to look through our custom list of columns and print
    ## out their value counts
    for x in list:
        print(f'Value Counts for {x}:\n')
        print(df[x].value_counts())
        print('--------------------')
        
    
def univariate_distributions(df):
    '''
    This function is designed to take in the player stats dataframe and 
    look at univariate distributions using .hist()
    '''
    
    ## looking at our continuous variable distributions
    stats = ['year', 'price', 'mileage', 'tax', 'mpg', 'engineSize']
    
    ## using list comprehension to look through our custom list of columns and print
    ## out their histograms viewing their distributions
    for x in stats:
        print(f'Distribution of {x}\n')
        df[x].hist()
        plt.show()
        print('--------------------')   