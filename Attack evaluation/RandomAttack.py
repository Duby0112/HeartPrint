import numpy as np
from sklearn.metrics import confusion_matrix
import joblib
from sklearn.preprocessing import StandardScaler

filename = 'Attack-Detection.sav'
clf = joblib.load(filename)

x1=StandardScaler().fit_transform(np.load('Attack-User1-Data.npy'))
y1=np.load('Attack-User1-Label.npy')
y_pred1=clf.predict(x1)
TN1, FP1, FN1, TP1=confusion_matrix(y1, y_pred1).ravel()
FNRate1 = FN1/(TP1+FN1)
print('FN Rate1=',FNRate1)

x2=StandardScaler().fit_transform(np.load('Attack-User2-Data.npy'))
y2=np.load('Attack-User2-Label.npy')
y_pred2=clf.predict(x2)
TN2, FP2, FN2, TP2=confusion_matrix(y2, y_pred2).ravel()
FNRate2 = FN2/(TP2+FN2)
print('FN Rate2=',FNRate2)


x3=StandardScaler().fit_transform(np.load('Attack-User3-Data.npy'))
y3=np.load('Attack-User3-Label.npy')
y_pred3=clf.predict(x3)
TN3, FP3, FN3, TP3=confusion_matrix(y3, y_pred3).ravel()
FNRate3 = FN3/(TP3+FN3)
print('FN Rate3=',FNRate3)

x4=StandardScaler().fit_transform(np.load('Attack-User4-Data.npy'))
y4=np.load('Attack-User4-Label.npy')
y_pred4=clf.predict(x4)
TN4, FP4, FN4, TP4=confusion_matrix(y4, y_pred4).ravel()
FNRate4 = FN4/(TP4+FN4)
print('FN Rate4=',FNRate4)

x5=StandardScaler().fit_transform(np.load('Attack-User5-Data.npy'))
y5=np.load('Attack-User5-Label.npy')
y_pred5=clf.predict(x5)
TN5, FP5, FN5, TP5=confusion_matrix(y5, y_pred5).ravel()
FNRate5 = FN5/(TP5+FN5)
print('FN Rate5=',FNRate5)

x6=StandardScaler().fit_transform(np.load('Attack-User6-Data.npy'))
y6=np.load('Attack-User6-Label.npy')
y_pred6=clf.predict(x6)
TN6, FP6, FN6, TP6=confusion_matrix(y6, y_pred6).ravel()
FNRate6= FN6/(TP6+FN6)
print('FN Rate6=',FNRate6)

x7=StandardScaler().fit_transform(np.load('Attack-User7-Data.npy'))
y7=np.load('Attack-User7-Label.npy')
y_pred7=clf.predict(x7)
TN7, FP7, FN7, TP7=confusion_matrix(y7, y_pred7).ravel()
FNRate7= FN7/(TP7+FN7)
print('FN Rate7=',FNRate7)

x8=StandardScaler().fit_transform(np.load('Attack-User8-Data.npy'))
y8=np.load('Attack-User8-Label.npy')
y_pred8=clf.predict(x8)
TN8, FP8, FN8, TP8=confusion_matrix(y8, y_pred8).ravel()
FNRate8= FN8/(TP8+FN8)
print('FN Rate8=',FNRate8)

x9=StandardScaler().fit_transform(np.load('Attack-User9-Data.npy'))
y9=np.load('Attack-User9-Label.npy')
y_pred9=clf.predict(x9)
TN9, FP9, FN9, TP9=confusion_matrix(y9, y_pred9).ravel()
FNRate9= FN9/(TP9+FN9)
print('FN Rate9=',FNRate9)

x10=StandardScaler().fit_transform(np.load('Attack-User10-Data.npy'))
y10=np.load('Attack-User10-Label.npy')
y_pred10=clf.predict(x10)
TN10, FP10, FN10, TP10=confusion_matrix(y10, y_pred10).ravel()
FNRate10= FN10/(TP10+FN10)
print('FN Rate10=',FNRate10)

Random_FNR=[FNRate1,FNRate2,FNRate3,FNRate4,FNRate5,FNRate6,FNRate7,FNRate8,FNRate9,FNRate10]
print('Mean:',np.mean(Random_FNR))
print('Std:',np.std(Random_FNR))