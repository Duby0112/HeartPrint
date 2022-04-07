import numpy as np
from sklearn.metrics import confusion_matrix
import joblib

x_test3=np.load('3Cycle-TestData.npy')
y_test3=np.load('3Cycle-TestLabel.npy')


filename3 = 'Model-3Cycle.sav'
loaded_model3 = joblib.load(filename3)
result3 = loaded_model3.score(x_test3, y_test3)
print('accuracy3:',result3)

y_pred3=loaded_model3.predict(x_test3)
confusion3 = confusion_matrix(y_test3, y_pred3,normalize='true')


import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df_cm = pd.DataFrame(confusion3, index = [i for i in range(54)],
                  columns = [i for i in range(54)])
plt.figure(figsize=(2.7,1.8),dpi=400)
sn.heatmap(df_cm,annot=False,cbar=True)


plt.tight_layout()
plt.show()