data=pd.read_csv('/Users/zj/Desktop/4.机器学习/数据分析/PYTHON数据/矫正后的数据删除一些不必要的变量的数据.csv')
data_feature = data[['Tumor.border', 'Aspect.ratio', 'Calcification', 'Tumor.internal.vascularization', 'Tumor.Peripheral.blood.flow',
'Size','Location','Hashimoto','ETE',
'ICLNM','Classification.of.ICNLNM',
'Classification.of.prelaryngeal.LNMR','Classification.of.pretracheal.LNMR',
'Classification.of.paratracheal.LNMR','ICLNMR','prelaryngeal.LNMR','pretracheal.LNMR',
'RLNLNMR',
]]
data_target=data['ILLNM']
data_target.unique()
data_featureCata=data[['Tumor.border', 'Aspect.ratio', 'Calcification', 'Tumor.internal.vascularization', 'Tumor.Peripheral.blood.flow',
'Size','Location','Hashimoto','ETE',
'ICLNM','Classification.of.ICNLNM',
'Classification.of.prelaryngeal.LNMR','Classification.of.pretracheal.LNMR','Classification.of.paratracheal.LNMR',
]]
data_featureNum=data[['ICLNMR','prelaryngeal.LNMR','pretracheal.LNMR',
'RLNLNMR',]]
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
data_featureNum=scaler.fit_transform(data_featureNum)
data_featureCata=np.array(data_featureCata)
import numpy as np
data_feature = np.hstack((data_featureCata, data_featureNum))from sklearn import feature_selection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import feature_selection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('/Users/zj/Desktop/4.机器学习/数据分析/PYTHON数据/矫正后的数据删除一些不必要的变量的数据.csv')
from sklearn.model_selection import train_test_split
data_feature=data[['Age','Sex','Tumor.border','Aspect.ratio','Internal.echo.pattern','Internal.echo.homogeneous','Calcification','Tumor.Peripheral.blood.flow',
'Size','Location','Mulifocality','ETE','Side.of.position',
'ICLNM','Classification.of.ICLNMR','Classification.of.ICNLNM',
'prelaryngeal.LNM','Classification.of.prelaryngeal.LNMR','Classification.of.prelaryngeal.NLNM',
'pretracheal.LNM','Classification.of.pretracheal.LNMR','Classification.of.pretracheal.NLNM',
'paratracheal.LNM','Classification.of.paratracheal.NLNM','Classification.of.paratracheal.LNMR',
'RLNLNM','Classification.of.RLNLNMR','Classification.of.RLNNLNM']]
### 5.5.3 Model Fitting
from sklearn.svm import SVR
from sklearn.svm import SVC
clf=SVC(kernel='linear',gamma=0.2)
clf.fit(class_x_tra,class_y_tra)
clf.score(class_x_val,class_y_val)
### 5.5.4 Evaluation Metrics
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
from sklearn.metrics import roc_curve
y_pred_proba = clf.decision_function(class_x_val)
fpr, tpr, thresholds = roc_curve(class_y_val, y_pred_proba)
auc = roc_auc_score(class_y_val, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='SVC (AUC = {:.2f})'.format(auc))
plt.plot([0, 1], [0, 1], 'k--') # 绘制对角线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve of SVM')
plt.legend(loc='lower right')
plt.show()
