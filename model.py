import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('E:\AQI-Project-master\AQI-Data\Real Data\Real_Combine.csv')
df.head()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df=df.dropna()
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
X.isnull()
y.isnull()
sns.pairplot(df)
df.corr()
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap='RdYlGn')
corrmat.index
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
model = ExtraTreesRegressor()
model.fit(X,y)
X.head()
print(model.feature_importances_)
feat_importance = pd.Series(model.feature_importances_,index=X.columns)
feat_importance.nlargest(5).plot(kind='barh')
plt.show()
sns.distplot(y)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()
regressor.fit(X_train,y_train)
print("Coefficient of determination R^2 <-- train set: {}".format(regressor.score(X_train,y_train)))
print("Coefficient of determination R^2 <-- train set: {}".format(regressor.score(X_test,y_test)))
from sklearn.model_selection import cross_val_score
score=cross_val_score(regressor,X,y,cv=5)
score.mean()
prediction=regressor.predict(X_test)
sns.distplot(y_test-prediction)
plt.scatter(y_test, prediction)
# Hyperparameter tuning
RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
print(n_estimators)
n_estimators =[int(x) for x in np.linspace(start=100, stop=1200, num=12)]
max_features = ["auto","sqrt"]
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]
random_grid = {"n_estimators": n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf':min_samples_leaf}
               
print(random_grid)
rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions=random_grid, scoring='neg_mean_squared_error', n_iter = 100, cv = 5, verbose = 2, random_state = 42, n_jobs= 1)
rf_random.fit(X_train,y_train)
rf_random.best_params_
rf_random.best_score_
predictions = rf_random.predict(X_test)
sns.distplot(y_test-predictions)
plt.scatter(y_test,predictions)
from sklearn import metrics
print('MAE:',metrics.mean_absolute_error(y_test,prediction))
print('MSE:',metrics.mean_squared_error(y_test,prediction))
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test,prediction)))
import pickle
file = open('random_forest_regression_model.pkl','wb')
pickle.dump(rf_random, file)


