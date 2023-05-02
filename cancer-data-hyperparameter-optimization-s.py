# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

dataframe = pd.read_csv("/kaggle/input/cancer-data/Cancer_Data.csv") #reading data

#dataframe.shape # checking shape of data
#dataframe.columns#columns present
#dataframe.isnull().sum() #checking nulls

#dropping column because contains all values as nulls
dataframe.drop("Unnamed: 32",axis=1, inplace=True)


#replacing M with 2 and B with 1
dataframe["diagnosis"] = dataframe["diagnosis"].replace("M",2) 
dataframe["diagnosis"] = dataframe["diagnosis"].replace("B",1)


if __name__ == "__main__":
    x = dataframe.drop(["id","diagnosis"],axis = 1).values
    y = dataframe["diagnosis"].values
    
    classifier = ensemble.RandomForestClassifier(n_jobs =-1)
    param_grid = {
        "n_estimators":[100,200,300,400],
        "max_depth":[1,3,5,7],
        "criterion":["gini","entropy"],
    }
    
    model = model_selection.GridSearchCV(
       estimator = classifier,
        param_grid = param_grid,
        scoring= "accuracy",
        verbose = 10,
        n_jobs = 1,
        cv = 5, #it will be 5 if not specified
    )
    
    #fit
    model.fit(x,y)
    
#model.best_score_
#0.9666200900481293 

#for best parameters
# model.best_estimator_.get_params()
# {'bootstrap': True,
#  'ccp_alpha': 0.0,
#  'class_weight': None,
#  'criterion': 'entropy',
#  'max_depth': 7,
#  'max_features': 'sqrt',
#  'max_leaf_nodes': None,
#  'max_samples': None,
#  'min_impurity_decrease': 0.0,
#  'min_samples_leaf': 1,
#  'min_samples_split': 2,
#  'min_weight_fraction_leaf': 0.0,
#  'n_estimators': 200,
#  'n_jobs': -1,
#  'oob_score': False,
#  'random_state': None,
#  'verbose': 0,
# 'warm_start': False}
