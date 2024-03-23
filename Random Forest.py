## 5.3.1 Loading Data
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn import feature_selection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import feature_selection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('/Users/zj/Desktop/4.机器学习/数据分析/PYTHON数据/矫正后的数据-纯数值变量.csv')
data_feature = data[['Tumor.border','Aspect.ratio','Calcification',
'Tumor.internal.vascularization','ETE','Size','Location',
'Mulifocality','ICLNM','prelaryngeal.LNM',
'pretracheal.LNM','paratracheal.LNM','RLNLNM',

'ICLNMR','ICNLNM','prelaryngeal.LNMR','prelaryngeal.NLNM',
'pretracheal.LNMR','pretracheal.NLNM','paratracheal.LNMR',
'paratracheal.NLNM','RLNLNMR','RLNNLNM'
]]
data_target=data['ILLNM']
data_target.unique()
data_featureCata=data[['Tumor.border','Aspect.ratio','Calcification',
'Tumor.internal.vascularization','ETE','Size','Location',
'Mulifocality','ICLNM','prelaryngeal.LNM',
'pretracheal.LNM','paratracheal.LNM','RLNLNM',
]]


data_featureNum=data[['ICLNMR','ICNLNM','prelaryngeal.LNMR','prelaryngeal.NLNM',
'pretracheal.LNMR','pretracheal.NLNM','paratracheal.LNMR',
'paratracheal.NLNM','RLNLNMR','RLNNLNM']]
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
data_featureNum=scaler.fit_transform(data_featureNum)
data_featureCata=np.array(data_featureCata)
import numpy as np
data_feature = np.hstack((data_featureCata, data_featureNum))
### 5.3.2 Categorical Data
data_feature.shape
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

class_x_tra, class_x_val, class_y_tra, class_y_val = train_test_split(data_feature, data_target, test_size=0.3, random_state=0)
print(class_x_tra.shape,class_x_val.shape)
### 5.3.3 Model Code
clf=RandomForestClassifier(n_estimators=25000,max_depth=5,min_samples_leaf=5,min_samples_split=5,)
clf.fit(class_x_tra,class_y_tra)
clf.score(class_x_val,class_y_val)
clf.feature_importances_
feature_name=['Tumor.border','Aspect.ratio','Calcification',
'Tumor.internal.vascularization','ETE','Size','Location',
'Mulifocality','ICLNM','prelaryngeal.LNM',
'pretracheal.LNM','paratracheal.LNM','RLNLNM','ICLNMR','ICNLNM','prelaryngeal.LNMR','prelaryngeal.NLNM',
'pretracheal.LNMR','pretracheal.NLNM','paratracheal.LNMR','paratracheal.NLNM','RLNLNMR','RLNNLNM']
[*zip(feature_name,clf.feature_importances_)]
### 5.3.4 Feature Importance Selection
import matplotlib.pyplot as plt
import numpy as np
importances = clf.feature_importances_
positive_importances = [imp for imp in importances if imp > 0]
negative_importances = [imp for imp in importances if imp < 0]
positive_indices = np.argsort(positive_importances)[::-1]
negative_indices = np.argsort(negative_importances)
positive_features = [feature_name[i] for i in positive_indices]
positive_importances = [positive_importances[i] for i in positive_indices]
negative_features = [feature_name[i] for i in negative_indices]
negative_importances = [abs(negative_importances[i]) for i in negative_indices]
plt.figure(figsize=(10, 6))
plt.barh(range(len(positive_features)), positive_importances, tick_label=positive_features, color='blue', label='Positive Importance')
plt.barh(range(len(negative_features)), negative_importances, tick_label=negative_features, color='red', label='Negative Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importances')
for i in range(len(positive_features)):
plt.text(positive_importances[i], i, f'{positive_importances[i]:.2f}', ha='left', va='center')
for i in range(len(negative_features)):
plt.text(negative_importances[i], i + len(positive_features), f'{negative_importances[i]:.2f}', ha='left', va='center')
plt.gca().invert_yaxis()
plt.legend()
plt.show()
import matplotlib.pyplot as plt
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = [feature_name[i] for i in indices]
importances = importances[indices]
plt.figure(figsize=(10, 6))
plt.barh(range(len(features)), importances, tick_label=features, color='palegoldenrod') 
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('ILLNM Feature Importances')
for i in range(len(features)):
plt.text(importances[i], i, f'{importances[i]:.2f}', ha='left', va='center')
plt.gca().invert_yaxis() plt.show()
import matplotlib.pyplot as plt
colors = ['palegreen', 'hotpink', 'palegoldenrod', 'skyblue']
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
top_features = [feature_name[i] for i in indices[:10]]
top_importances = importances[indices[:10]]
plt.figure(figsize=(10, 6))
plt.barh(range(len(top_features)), top_importances, tick_label=top_features, color='palegoldenrod') plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('ILLNM Top 10 Feature Importances')
for i in range(len(top_features)):
plt.text(top_importances[i], i, f'{top_importances[i]:.2f}', ha='left', va='center')
plt.gca().invert_yaxis() plt.show()
import matplotlib.pyplot as plt
### 5.3.4 Evaluation Metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
y_pred = clf.predict(class_x_val)
y_pred_proba = clf.predict_proba(class_x_val)[:, 1]
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
print("DCA:", net_benefit) print("Precision:", precision)
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
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(class_y_val, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plotting ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ILLNM Random forest ROC curve')
plt.legend(loc="lower right")
plt.show()
