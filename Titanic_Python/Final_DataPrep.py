# -*- coding: utf-8 -*-
import warnings
from collections import OrderedDict
import numpy as np
import pandas as pd

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
import missingno as msno
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

sns.set(color_codes=True)
pal = sns.color_palette("Set2", 10)
sns.set_palette(pal)

titanic_df = pd.read_excel('C:\\dev\Demos\\BAS2018\\Titanic_Python\\titanic.xls', 'RAW', index_col = None, na_values=['NA'])

titanic_df2 = titanic_df.drop(['boat', 'home.dest', 'body','ticket'],axis=1)
titanic_df2['died'] = 1 - titanic_df2['survived']

titanic_df3 = get_titles(titanic_df2)
titanic_df3.head(5)
titanic_df3.drop('name', axis=1, inplace=True)
titanic_df3[titanic_df3['title'].isnull()]        

# Dummy and clean title
titanic_df4 = dummy_title(titanic_df3)
titanic_df4.drop('title', axis=1, inplace=True)

titanic_df5 = handle_embarked(titanic_df4)
titanic_df5.drop('embarked', axis=1, inplace=True)

titanic_df6 = dummy_cabin(titanic_df5)
titanic_df6.drop('cabin', axis=1, inplace=True)

titanic_df7 = map_sex(titanic_df6)

titanic_df8 = impute_age_mean(titanic_df7)

titanic_df9 = titanic_df8.dropna()

titanic_df10 = scale_fares(titanic_df9)

titanic_df10.to_csv('C:\\dev\Demos\\BAS2018\\Titanic_Python\\final_titanic_clean.csv', index = False)
