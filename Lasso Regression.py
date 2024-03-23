## 5.1.1 Loading and Correcting Data
from sklearn import feature_selection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv('/Users/zj/Desktop/4.机器学习/数据分析/PYTHON数据/矫正后的数据-纯数值变量_ILLNM.csv')
data_featureCata=data[['Sex','Aspect.ratio','Calcification','Size','Location','Mulifocality',
'ICLNM','prelaryngeal.LNM',
'pretracheal.LNM','paratracheal.LNM','RLNLNM',
]]
data_featureNum=data[['ICLNMR','ICNLNM','prelaryngeal.LNMR','prelaryngeal.NLNM',
'pretracheal.LNMR','pretracheal.NLNM','paratracheal.LNMR',
'paratracheal.NLNM','RLNLNMR','RLNNLNM']]
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
data_featureNum=scaler.fit_transform(data_featureNum)
data_featureNum.shape
data_featureCata=np.array(data_featureCata)
data_featureCata
import numpy as np
data_feature = np.hstack((data_featureCata, data_featureNum))
data_feature.shape
## 5.1.2 Splitting into Training and Validation Sets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
data_targetclass = data['ILLNM']
data_targetNum=data['ILLNM']

class_x_tra, class_x_val, class_y_tra, class_y_val = train_test_split(data_feature, data_targetclass, test_size=0.3, random_state=0)
reg_x_tra, reg_x_val, reg_y_tra, reg_y_val = train_test_split(data_feature, data_targetNum, test_size=0.3, random_state=0)
print(class_x_tra.shape,class_y_tra.shape)
## 5.1.3 Model Fitting
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
clf=RidgeClassifier(alpha=1,fit_intercept=True)
clf.fit(class_x_tra,class_y_tra)
clf.score(class_x_val,class_y_val)
accuracy = clf.score(class_x_val, class_y_val)
print("Validation Accuracy:", accuracy)
clf=RidgeClassifierCV(alphas=[1e-3,1e-2,1e-1,1],cv=10)
clf.fit(class_x_tra,class_y_tra)
clf.score(class_x_val,class_y_val)
clf.predict(class_x_val)
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
import numpy as np
n_alphas =200
alphas = np.logspace(-10 -2, n_alphas)
print(alphas)
coefs=[]
for a in alphas:
ridge=Ridge(alpha=a,fit_intercept=False)
ridge.fit(class_x_tra,class_y_tra)
coef=ridge.coef_
coefs.append(ridge.coef_) 
coefs len(coefs)
# Ridge Trace Plot
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
coefs = []
for a in alphas:
ridge = Ridge(alpha=a, fit_intercept=False)
ridge.fit(class_x_tra, class_y_tra)
coef = ridge.coef_
coefs.append(coef)
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1]) 
plt.xlabel('ILLNM alpha')
plt.ylabel('coefficient')
plt.title('Ridge Coefficient Plot')
plt.axis('tight')
plt.show()
array=([1,101,201,301,401,501,601,701,801,901])
Ridge_=RidgeCV(alphas=np.arange(1,1001,100),store_cv_values=True)
Ridge_.fit(reg_x_tra,reg_y_tra )
Ridge_.score(reg_x_val,reg_y_val)
# Cross-Validation Results
Ridge_.score(reg_x_val,reg_y_val)
Ridge_.cv_values_.shape
Ridge_.cv_values_.mean(axis=0)
Ridge_.alpha_
## 5.1.4 Evaluation Metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
y_pred = clf.predict(class_x_val)
y_pred_proba = clf.decision_function(class_x_val)
accuracy = accuracy_score(class_y_val, y_pred)
auc = roc_auc_score(class_y_val, y_pred_proba)
conf_matrix = confusion_matrix(class_y_val, y_pred)
precision = precision_score(class_y_val, y_pred)
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
sensitivity = recall_score(class_y_val, y_pred)
npv = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0])
ppv = precision
recall = sensitivity
f1_score = f1_score(class_y_val, y_pred)
false_positive_rate = conf_matrix[0, 1] / (conf_matrix[0, 1] + conf_matrix[0, 0])
print("Accuracy:", accuracy)
print("AUC:", auc)
print("DCA:", net_benefit) 
print("Precision:", precision)
print("Specificity:", specificity)
print("Sensitivity:", sensitivity)
print("Negative Predictive Value:", npv)
print("Positive Predictive Value:", ppv)
print("Recall:", recall)
print("F1 Score:", f1_score)
print("False Positive Rate:", false_positive_rate)
labels = ['Accuracy', 'AUC', 'DCA', 'Precision', 'Specificity', 'Sensitivity', 'NPV', 'PPV', 'Recall', 'F1 Score', 'FPR']
values = [accuracy, auc, net_benefit, precision, specificity, sensitivity, npv, ppv, recall, f1_score, false_positive_rate]
plt.figure(figsize=(8, 6))
plt.bar(labels, values)
plt.xlabel('Evaluation Metric')
plt.ylabel('Value')
plt.title('Model Evaluation Metrics')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
y_pred_proba = clf.decision_function(class_x_val)

# Calculating ROC Curve
fpr, tpr, thresholds = roc_curve(class_y_val, y_pred_proba)
auc = roc_auc_score(class_y_val, y_pred_proba)
# Plotting ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (AUC = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ILLNM Ridge regression ROC curve')
plt.legend(loc="lower right")
plt.show()
####LASSO
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV

clf = Lasso(alpha=0.1, fit_intercept=True)
clf.fit(class_x_tra, class_y_tra)

accuracy = clf.score(class_x_val, class_y_val)
print("Validation Accuracy:", accuracy)
clf=LassoCV(alphas=[1e-3,1e-2,1e-1,1],cv=5,max_iter=10000)

clf.fit(class_x_tra,class_y_tra)
clf.score(class_x_val,class_y_val)
