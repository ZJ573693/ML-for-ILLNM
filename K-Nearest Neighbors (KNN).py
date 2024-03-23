### 5.6.1 Loading Data
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
data_feature = np.hstack((data_featureCata, data_featureNum))
### 5.6.2 Data Classification
data_feature.shape
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
class_x_tra, class_x_val, class_y_tra, class_y_val = train_test_split(data_feature, data_target, test_size=0.3, random_state=0)
print(class_x_tra.shape,class_x_val.shape)

### 5.6.3 Model Fitting
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
best_score = 0
best_k = None
for i in range(1, 51):
clf = KNeighborsClassifier(n_neighbors=i, leaf_size=40, n_jobs=6)
clf.fit(class_x_tra, class_y_tra)
score = clf.score(class_x_val, class_y_val)
if score > best_score:
best_score = score
best_k = i
print("Best score:", best_score)
print("Best k:", best_k)
clf=KNeighborsClassifier(n_neighbors=11,leaf_size=40,n_jobs=6)
clf.fit(class_x_tra,class_y_tra)
clf.score(class_x_val,class_y_val)
5.6.4 Evaluation indicators
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
print("DCA:", net_benefit) # 假设已经计算了DCA的net_benefit
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
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
y_pred_proba = clf.predict_proba(class_x_val)[:, 1]
fpr, tpr, thresholds = roc_curve(class_y_val, y_pred_proba)
auc = roc_auc_score(class_y_val, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='KNN (AUC = {:.2f})'.format(auc))
plt.plot([0, 1], [0, 1], 'k--') # 绘制对角线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ILLNM KNN ROC Curve ')
plt.legend(loc='lower right')
plt.show()
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
y_pred_proba = clf.predict_proba(class_x_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(class_y_val, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('ILLNM KNN Precision-Recall Curve')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid(True)
plt.show()
