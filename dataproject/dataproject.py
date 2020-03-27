# Importing packages
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import pandas as pd
#import pydst
#dst = pydst.Dst(lang='en')

#setting up work directory
import os 
assert os.path.isdir('dataproject/')
assert os.path.isfile('dataproject/RAS200.xlsx')
assert os.path.isfile('dataproject/INDKP101.xlsx')
os.listdir('dataproject/')

#cleaning data
empl_data = 'dataproject/RAS200.xlsx'
inc_data = 'dataproject/INDKP101.xlsx'

for var in empl, inc:

    var = pd.read_excel(var +"_data"), skiprows=2)
    drop_unnamed = ['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3']
    var.drop(drop_unnamed, axis=1, inplace=True)
    var.rename(columns = {'Unnamed: 4':'municipality'}, inplace=True)
    var.rename(columns = {str(i): f'var{i}' for i in range(2004,2018)}, inplace=True)
    var.rename(columns = myDict, inplace=True)
    var+"_long" = pd.wide_to_long(var, stubnames= var, i='municipality', j='year')

next





def only_keep_municipalities(df):
    """ delete all non-municipalities

    Args:
        df (pd.DataFrame): pandas dataframe with the column "municipality" as a string

    Returns:
        df (pd.DataFrame): pandas dataframe

    """ 
    
    for val in ['Region', 'Province', 'All Denmark']:
        
        I = df.municipality.str.contains(val)
        df = df.loc[I == False] # keep everything else
    
    return df