# -*- coding: utf-8 -*-
Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}

def get_titles(source_df):
    # we extract the title from each name
    source_df['title'] = source_df['name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    
    # a map of more aggregated title
    # we map each title
    source_df['title'] = source_df['title'].map(Title_Dictionary)
    return source_df

def impute_age_row(source_df, row):
    condition = (
        (source_df['sex'] == row['sex']) & 
        (source_df['title'] == row['title']) & 
        (source_df['pclass'] == row['pclass'])
    ) 
    return source_df[condition]['age'].values[0]

def impute_age(source_df):
    # a function that fills the missing values of the Age variable
    source_df['age'] = source_df.apply(lambda row: impute_age_row(source_df, row) if np.isnan(row['age']) else row['age'], axis=1)
    return source_df

def impute_age_mean(source_df):
    MedianAge = source_df['age'].median()
    source_df['age'] = source_df['age'].fillna(value=MedianAge)
    return source_df
    
def dummy_title(source_df):
    titles_dummies = pd.get_dummies(source_df['title'], prefix='title')
    source_df = pd.concat([source_df, titles_dummies], axis=1)
    return source_df

def impute_fares(source_df):
    # there's one missing fare value - replacing it with the mean.
    source_df['fare'].fillna(source_df['fare'].mean(), inplace=True)
    return source_df

def scale_fares(source_df):
    scale = StandardScaler().fit(source_df[['fare']])
    source_df[['fare']] = scale.transform(source_df[['fare']])    
    return source_df

def dummy_embarked(source_df):
    # two missing embarked values - filling them with the most frequent one in the train  set(S)
    source_df['embarked'].fillna('S', inplace=True)
    # dummy encoding 
    embarked_dummies = pd.get_dummies(source_df['embarked'], prefix='embarked')
    source_df = pd.concat([source_df, embarked_dummies], axis=1)
    return source_df

def handle_embarked(source_df):
    ModeEmbarked = source_df['embarked'].mode()[0]
    source_df['embarked'] = source_df['embarked'].fillna(value=ModeEmbarked)
    embarked_dummies = pd.get_dummies(source_df['embarked'], prefix='embarked')
    source_df = pd.concat([source_df, embarked_dummies], axis=1)
    return source_df

def dummy_cabin(source_df):
    # replacing missing cabins with U (for Uknown)
    source_df['cabin'].fillna('U', inplace=True)
    
    # mapping each Cabin value with the cabin letter
    source_df['cabin'] = source_df['cabin'].map(lambda c: c[0])
    
    # dummy encoding ...
    cabin_dummies = pd.get_dummies(source_df['cabin'], prefix='cabin')    
    source_df = pd.concat([source_df, cabin_dummies], axis=1)

    return source_df

def map_sex(source_df):
    # mapping string values to numerical one 
    source_df['sex'] = source_df['sex'].map({'male':1, 'female':0})
    return source_df

def dummy_pclass(source_df):
    # encoding into 3 categories:
    pclass_dummies = pd.get_dummies(source_df['pclass'], prefix="pclass")
    
    # adding dummy variable
    source_df = pd.concat([source_df, pclass_dummies],axis=1)
        
    return source_df

