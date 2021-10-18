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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from scipy import stats


############################# Function File For Modeling Pipeline Phase ################################

def select_rfe(X, y, k, return_rankings=False, model=LinearRegression()):
    '''
    Function to utilize recursive feature elimination and return a list of the best features 
    to be used for regression modeling
    '''

    # Use the passed model, LinearRegression by default
    rfe = RFE(model, n_features_to_select=k)
    rfe.fit(X, y)
    features = X.columns[rfe.support_].tolist()
    if return_rankings:
        rankings = pd.Series(dict(zip(X.columns, rfe.ranking_)))
        return features, rankings
    else:
        return features


def get_metrics(df, model_name,rmse_validate,r2_validate):
    '''
    Function designed to create a dataframe of model metrics for easy comparison of RMSE and R-squared
    values when using regression machine learning
    '''
    df = df.append({
        'model': model_name,
        'rmse_outofsample':rmse_validate, 
        'r^2_outofsample':r2_validate}, ignore_index=True)
    return df