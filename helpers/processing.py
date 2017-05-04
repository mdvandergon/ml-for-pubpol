'''
Processing Functions
'''
import os
import numpy as np
import pandas as pd

def bin_values(array, N):
    # takes a 1D series of vals and returns a list of series
    bins = np.linspace(0, 1, N, endpoint=False)
    rv = []

    for i, bin in enumerate(bins):
        vals = array[np.digitize(array,bins)==i]
        vals.name = str(bin)
        rv.append(vals)

    return rv

def tag_bin(array, N):
    # takes a 1D series of vals and returns what bin they should be in
    bins = np.linspace(0, 1, N, endpoint=False)
    col = array.name
    df = pd.concat([array, pd.Series([0],name="bin")], axis=1)

    for i, bin in enumerate(bins):
        df.bin = np.digitize(df[col],bins)

    return df

def mean_inputation(df):
    '''
    Replace the null vals with the column mean
    '''
    null_columns = df.columns[pd.isnull(df).sum() > 0].tolist()
    for column in null_columns:
        # input the mean of the data if they are numeric
        data = df[column]
        if data.dtype in [int, float]:
            inputed_value = data.mean()
            df.loc[:,(column)] = data.fillna(inputed_value)

    return df

def categorical_dummies(df, columns):
    '''
    Pandas function wrapper to inplace combine a new set of dummy variable columns
    '''
    for column in columns:
        dummies = pd.get_dummies(df[column], prefix=column+"_is", prefix_sep='_', dummy_na=True)
        df = pd.concat([df, dummies], axis=1)

    return df


# def scaling():

# def leakage():

# def correlations():

# def transformations():

# def interactions():

# def seasonality():

# def spatio_temporal():
