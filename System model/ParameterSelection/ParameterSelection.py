import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
import time

start_time = time.time()
#######################################
#Data loading
x=np.load('Data.npy')
y=np.load('Lable.npy')
#######################################
#Grid Search Method: Define a logarithmic grid, using a basis of 2, a finer 
#tuning can be achieved but at a much higher cost.
C_range = np.logspace(-3, 3, 7)
gamma_range = np.logspace(-3, 3, 7)
param_grid = dict(gamma=gamma_range, C=C_range)
#######################################
#Split data for 5 fold cross validation
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=3)
#######################################
#Classifier training
grid = GridSearchCV(SVC(kernel='rbf'), param_grid=param_grid, cv=cv)
grid.fit(x, y)

print(
    "The best parameters are %s with a score of %0.2f"
    % (grid.best_params_, grid.best_score_)
)
print("--- %s seconds ---" % (time.time() - start_time))