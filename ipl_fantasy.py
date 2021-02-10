# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 23:12:41 2021

@author: HOME
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
sns.heatmap(data.corr(), annot=True)

data = pd.read_csv('Training.csv')
matches_pre20 = pd.read_csv('Matches IPL 2008-2019.csv')
matches_20 = pd.read_csv('Matches IPL 2020.csv')
squads_20 = pd.read_csv('IPL 2020 Squads.csv', encoding='latin1')
submission = pd.read_csv('sample_submission.csv')

extra_col = data['Id'].str.split('_',expand=True)
data['player_id'] = extra_col[0]
data['player_name'] = extra_col[1]
data.head()

#computing player with highest total points
Best_player = data.nlargest(20, ['Total Points'])

plt.figure(figsize=(10,6))
plt.bar(Best_player['player_name'], Best_player['Total Points'])
plt.xlabel('Player names')  
plt.ylabel('Points')
plt.xticks(rotation=90)
plt.show()

# computing top players who scored most number of bowling points
Bowling_Points= data.groupby('player_name').Bowling_Points.sum().nlargest(n=20)
Bowling_Points.plot(kind = 'bar',figsize = (10,5))
plt.xlabel('Player Name')
plt.ylabel('Total Bowling_points')
plt.title('Top players who scored most number of bowling points')
plt.show()

# computing top players who scored most number of batting points
Batting_Points= data.groupby('player_name').Batting_Points.sum().nlargest(n=20)
Batting_Points.plot(kind = 'bar',figsize = (10,5))
plt.xlabel('Player Name')
plt.ylabel('Total Batting_points')
plt.title('Top players who scored most number of batting points')
plt.show()

#computing player with highest played matches
plt.figure(figsize = (10,6))
sns.countplot(x=data['player_name'], data= data, order=data['player_name'].value_counts().iloc[:10].index)
plt.xticks(rotation=90)
plt.show()


matches_pre20.isnull().sum()

#computing season with most number of matches 

plt.figure(figsize = (10,6))
sns.countplot(x= matches_pre20['season'],data= matches_pre20)
plt.show()

#computing highest number of wins by a team

plt.figure(figsize=(10,6))
plt.xticks(rotation=90)
sns.countplot(matches_pre20['winner'])

#computing highest number of matches played at a particular venue

plt.figure(figsize = (10,6))
sns.countplot(x=matches_pre20["venue"], data= matches_pre20, order=matches_pre20.venue.value_counts().iloc[:20].index)
plt.xticks(rotation=90)
plt.show()

#computing players with highest number of man of the match awards

plt.figure(figsize = (10,6))
sns.countplot(x=matches_pre20["player_of_match"], data=matches_pre20, order=matches_pre20.player_of_match.value_counts().iloc[:20].index)
plt.xticks(rotation=90)
plt.show()

#data modelling
df = pd.DataFrame()
df['match_number'] = data['player_id']
df['player_name'] = data['player_name']
df['total_score'] = data['Total Points']
df['Id']= data['Id']
df.head()

df.groupby(['player_name','match_number']).sum()

df = df.drop(['player_name'],axis=1)
df = df.iloc[:1283]
df.shape

df.match_number=  df.match_number.astype(int)

cat_col=[col for col in df.columns if df[col].dtype=='O']
cat_col
categorical = df[cat_col]
categorical.head()

Id=pd.get_dummies(categorical['Id'],drop_first=True)
Id.head()

cont_col=[col for col in df.columns if df[col].dtype!='O']
cont_col

df= pd.concat([Id,df[cont_col]],axis = 1)
df.head()

X=df.drop(['total_score'],axis=1)
y=df['total_score']


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_train.shape
X_test = scaler.transform(X_test)
X_test.shape

# importing linear regression

from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X_train,y_train)
from sklearn import metrics

y_pred = reg.predict(X_test)
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

from sklearn.ensemble import RandomForestRegressor

reg2 = RandomForestRegressor()
reg2.fit(X_train,y_train)

y_pred2 = reg2.predict(X_test)
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred2)))

y_pred = np.round(y_pred)
y_pred_new = [0 if i < 0 else i for i in y_pred] 
   #removing the neagtive values
submission['id'] = submission['Id'].str.split('_').str[0]  
#splitting the names and id's
submission['Player'] = submission['Id'].str.split('_').str[1]
submission.head()

xs = submission.drop(['Total Points','Player'],axis=1)
xs.id = xs.id.astype(int)
xs.id.dtypes

#now separating out the categorical columns and numerical columns
#printing all the categorical columns
cat_col = [col for col in xs.columns if xs[col].dtype=='O']
cat_col

cont_col=[col for col in xs.columns if xs[col].dtype!='O']
cont_col

categorical=xs[cat_col]
categorical.head()

Id=pd.get_dummies(categorical['Id'],drop_first=True)
Id.head()

xs= pd.concat([Id,xs[cont_col]],axis = 1)
xs.head()

scaler.fit(xs)
xs = scaler.transform(xs)
pred_y = reg.predict(xs)
pred_y

pred_y = np.round(pred_y)
len(pred_y)

pred_y_new = [0 if i < 0 else i for i in pred_y]
neg = []
for i in pred_y:
    if i <0:
        neg.append(i)
submission = submission.drop(['Player','id'],axis=1)


submission.to_csv('ipl_predictions.csv',index=False)