import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
import joblib
from sklearn.preprocessing import StandardScaler

filename = 'Attack-Detection.sav'
clf = joblib.load(filename)

x_random=StandardScaler().fit_transform(np.load('RandomAttack-Data.npy'))
y_random=np.load('RandomAttack-Label.npy')
y_score_random= clf.decision_function(x_random)
fpr_random,tpr_random,_=roc_curve(y_random,y_score_random)
roc_random=auc(fpr_random,tpr_random)

x_replay=StandardScaler().fit_transform(np.load('ReplayAttack-Data.npy'))
y_replay=np.load('ReplayAttack-Label.npy')
y_score_replay= clf.decision_function(x_replay)
fpr_replay,tpr_replay,_=roc_curve(y_replay,y_score_replay)
roc_replay=auc(fpr_replay,tpr_replay)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(fpr_random, tpr_random,label='ROC curve (area = %f)' % roc_random)
plt.plot(fpr_replay, tpr_replay,label='ROC curve (area = %f)' % roc_replay)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot([0, 1], [1, 0], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()