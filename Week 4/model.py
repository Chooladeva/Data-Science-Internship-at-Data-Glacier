import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV

import pickle

import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv('melb_data.csv',index_col=0)

data.isna().sum()
data=data.dropna(axis=0)

y=data["Price"]

data_features=['Rooms','Bedroom2','Bathroom','Car','Landsize','BuildingArea','Lattitude','Longtitude']
X=data[data_features]


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)


rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)


# Evaluating the Model
#print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
#print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#score= rf_model.score(X_test,y_test) #R2 Score
#print(score)


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


ran_model_1 = RandomizedSearchCV(estimator = rf_model, param_distributions = random_grid,
                               n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
ran_model_1.fit(X_train,y_train)

ran_model_1.best_params_


predictions=ran_model_1.best_estimator_.predict(X_test)


# Evaluating the Algorithm

#print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))   
#print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

#score= ran_model_1.best_estimator_.score(X_test,y_test) #R2 Score
#print(score)


pickle.dump(ran_model_1,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))

