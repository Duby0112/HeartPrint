import numpy as np
from sklearn.metrics import confusion_matrix
import joblib

x_test1=np.load('1Cycle-TestData.npy')
y_test1=np.load('1Cycle-TestLabel.npy')

x_test2=np.load('2Cycle-TestData.npy')
y_test2=np.load('2Cycle-TestLabel.npy')

x_test3=np.load('3Cycle-TestData.npy')
y_test3=np.load('3Cycle-TestLabel.npy')

x_test4=np.load('4Cycle-TestData.npy')
y_test4=np.load('4Cycle-TestLabel.npy')

x_test5=np.load('5Cycle-TestData.npy')
y_test5=np.load('5Cycle-TestLabel.npy')

x_test6=np.load('6Cycle-TestData.npy')
y_test6=np.load('6Cycle-TestLabel.npy')

#### load the model from disk
filename1 = 'Model-1Cycle.sav'
loaded_model1 = joblib.load(filename1)
result1 = loaded_model1.score(x_test1, y_test1)
print('accuracy1:',result1)

filename2 = 'Model-2Cycle.sav'
loaded_model2 = joblib.load(filename2)
result2 = loaded_model2.score(x_test2, y_test2)
print('accuracy2:',result2)

filename3 = 'Model-3Cycle.sav'
loaded_model3 = joblib.load(filename3)
result3 = loaded_model3.score(x_test3, y_test3)
print('accuracy3:',result3)

filename4 = 'Model-4Cycle.sav'
loaded_model4 = joblib.load(filename4)
result4 = loaded_model4.score(x_test4, y_test4)
print('accuracy4:',result4)

filename5 = 'Model-5Cycle.sav'
loaded_model5 = joblib.load(filename5)
result5 = loaded_model5.score(x_test5, y_test5)
print('accuracy5:',result5)


filename6 = 'Model-6Cycle.sav'
loaded_model6 = joblib.load(filename6)
result6 = loaded_model6.score(x_test6, y_test6)
print('accuracy6:',result6)


###########################calculate confusion matrix
y_pred1=loaded_model1.predict(x_test1)
confusion1 = confusion_matrix(y_test1, y_pred1)
print('Confusion Matrix1:\n')
print(confusion1)

y_pred2=loaded_model2.predict(x_test2)
confusion2 = confusion_matrix(y_test2, y_pred2)
print('Confusion Matrix2:\n')
print(confusion2)

y_pred3=loaded_model3.predict(x_test3)
confusion3 = confusion_matrix(y_test3, y_pred3)
print('Confusion Matrix3:\n')
print(confusion3)

y_pred4=loaded_model4.predict(x_test4)
confusion4 = confusion_matrix(y_test4, y_pred4)
print('Confusion Matrix4:\n')
print(confusion4)

y_pred5=loaded_model5.predict(x_test5)
confusion5 = confusion_matrix(y_test5, y_pred5)
print('Confusion Matrix5:\n')
print(confusion5)

y_pred6=loaded_model6.predict(x_test6)
confusion6 = confusion_matrix(y_test6, y_pred6)
print('Confusion Matrix6:\n')
print(confusion6)






