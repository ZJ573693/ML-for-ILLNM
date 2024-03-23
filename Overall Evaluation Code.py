## 6.1 Plotting Overall ROC Curve
### 6.1.1 Method 1: Pay attention to parameter settings! Especially for neural networks and random forests, use the code with the parameter settings learned earlier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
data = pd.read_csv('/Users/zj/Desktop/4.机器学习/数据分析/PYTHON数据/矫正后的数据删除一些不必要的变量的数据.csv')
data_feature = data[['Age','Tumor.border', 'Aspect.ratio', 
'Internal.echo.homogeneous','Calcification', 
'Tumor.internal.vascularization', 'Size','Location',
'Side.of.position',
'ICLNM','Classification.of.ICLNMR',
'pretracheal.LNM','Classification.of.pretracheal.LNMR',
'paratracheal.LNM','Classification.of.paratracheal.LNMR',
'RLNLNM','prelaryngeal.NLNM', 'pretracheal.NLNM','paratracheal.NLNM','RLNNLNM' ]]
data_target=data['ILLNM']
data_target.unique()#二分类
data_featureCata=data[['Age','Tumor.border', 'Aspect.ratio', 'Internal.echo.homogeneous','Calcification', 'Tumor.internal.vascularization', 'Size',
'Side.of.position','Location',
'ICLNM','Classification.of.ICLNMR',
'pretracheal.LNM','Classification.of.pretracheal.LNMR',
'paratracheal.LNM','Classification.of.paratracheal.LNMR',
'RLNLNM',
]]
data_featureNum=data[['ICNLNM', 'prelaryngeal.NLNM', 'pretracheal.NLNM','paratracheal.NLNM','RLNNLNM' ]]
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
data_featureNum=scaler.fit_transform(data_featureNum)
data_featureCata=np.array(data_featureCata)
import numpy as np
data_feature = np.hstack((data_featureCata, data_featureNum))
data_target = data['ILLNM']
class_x_tra, class_x_val, class_y_tra, class_y_val = train_test_split(data_feature, data_target, test_size=0.3, random_state=0)
models = [
LogisticRegression(),
RidgeClassifier(alpha=1, fit_intercept=True),
DecisionTreeClassifier(min_samples_split=3, max_depth=5, min_samples_leaf=10, max_features=20),
RandomForestClassifier(n_estimators=25000, max_depth=5, min_samples_leaf=5, min_samples_split=5),
GradientBoostingClassifier(n_estimators=25000, max_depth=7, learning_rate=0.01),
SVC(kernel='linear', gamma=0.2, probability=True),
KNeighborsClassifier(n_neighbors=11, leaf_size=40, n_jobs=6),
GaussianNB(),
MLPClassifier()
]
model_names = [
'Logistic Regression',
'Ridge Regression',
'Decision Tree',
'Random Forest',
'Gradient Boosting',
'Support Vector Machine',
'K-Nearest Neighbors',
'Gaussian Naive Bayes',
'Neural Network'
]
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'cyan']
nn_param_grid = {
'hidden_layer_sizes': [(10,), (20,), (30,)],
'activation': ['relu', 'tanh'],
'solver': ['adam', 'sgd'],
'alpha': [0.0001, 0.001, 0.01],
'learning_rate': ['constant', 'adaptive'],
'learning_rate_init': [0.001, 0.01, 0.1]
}
nn_grid_search = GridSearchCV(estimator=MLPClassifier(), 
param_grid=nn_param_grid, cv=5)
plt.figure(figsize=(8, 6))
for model, name, color in zip(models, model_names, colors):
if name == 'Neural Network':
nn_grid_search.fit(class_x_tra, class_y_tra)
best_nn_model = nn_grid_search.best_estimator_
model = best_nn_model
else:
model.fit(class_x_tra, class_y_tra)
if hasattr(model, 'predict_proba'):
y_pred_prob = model.predict_proba(class_x_val)[:, 1]
else:
y_pred_prob = model.decision_function(class_x_val)
auc = roc_auc_score(class_y_val, y_pred_prob)
fpr, tpr, _ = roc_curve(class_y_val, y_pred_prob)
plt.plot(fpr, tpr, color=color, label='%s (AUC = %0.3f)' % (name, auc))

plt.plot([0, 1], [0, 1], 'k--') 
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('IV.level.LNM Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()
