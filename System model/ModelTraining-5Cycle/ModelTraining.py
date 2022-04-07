import numpy as np
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

x=np.load('5Cycle-Data.npy')
y=np.load('5Cycle-Label.npy')
x = StandardScaler().fit_transform(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3) 

clf = OneVsRestClassifier(SVC(kernel='rbf', decision_function_shape='ovo',gamma=0.01,probability=True)) .fit(x_train,y_train)

#####Save model
filename = 'Model-5Cycle.sav'
joblib.dump(clf, filename) 
#### load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.score(x_test, y_test)
print(result)

y_pred=clf.predict(x_test)
acc=accuracy_score(y_test,y_pred)
print('Accuracy is {:.4f}%'.format(acc*100))
