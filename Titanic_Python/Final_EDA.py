# -*- coding: utf-8 -*-
titanic_df.columns
titanic_df.shape
titanic_df.describe()
titanic_df.count()
titanic_df.info()
titanic_df.head(5)

msno.matrix(titanic_df)
sns.heatmap(titanic_df.corr(), annot=True)

view_survivalanalysis(titanic_df2)

def view_survivalanalysis(source_df):
    source_df.groupby('sex').agg('sum')[['survived', 'died']].plot(kind='bar', figsize=(5, 7), stacked=True, color=['g', 'r']);
    source_df.groupby('sex').agg('mean')[['survived', 'died']].plot(kind='bar', stacked=True, color=['g', 'r']);
    
    fig = plt.figure()
    sns.violinplot(x='sex', y='age', 
                   hue='survived', data=source_df, 
                   split=True,
                   palette={0: "r", 1: "g"}
                  );
    
    figure = plt.figure()
    plt.hist([source_df[source_df['survived'] == 1]['fare'], source_df[source_df['survived'] == 0]['fare']], 
             stacked=True, color = ['g','r'],
             bins = 50, label = ['survived','dead'])
    plt.xlabel('Fare')
    plt.ylabel('Number of passengers')
    plt.legend();


