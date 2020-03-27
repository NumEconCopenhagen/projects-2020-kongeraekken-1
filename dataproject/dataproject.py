# Importing packages
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import pandas as pd
import pandas_datareader
import datetime
import pydst 

#Import Nasdaq and Nikkei from FRED API
start = datetime.datetime(2000,1,1)
end = datetime.datetime(2020,3,26)

nasdaq = pandas_datareader.data.DataReader('NASDAQCOM', 'fred', start, end)
nikkei = pandas_datareader.data.DataReader('NIKKEI225', 'fred', start, end)

#Import OMXC20 data from Statistics Denmark API
Dst = pydst.Dst(lang='en') # setup data loader with the language 'english'
omx = Dst.get_data(table_id = 'MPK13', variables={})
omx.head(5)

#plotting
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

nasdaq.plot(ax=ax)













#cleaning data
empl_data = 'dataproject/RAS200.xlsx'
inc_data = 'dataproject/INDKP101.xlsx'

for var in empl, inc:

    var = pd.read_excel(var +"_data"), skiprows=2)
    drop_unnamed = ['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3']
    var.drop(drop_unnamed, axis=1, inplace=True)
    var.rename(columns = {'Unnamed: 4t':'municipality'}, inplace=True)
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