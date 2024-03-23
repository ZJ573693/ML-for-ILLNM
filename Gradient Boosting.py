### 5.4.4 Model Code - Similar to Random Forest and Decision Tree
clf=GradientBoostingClassifier(n_estimators=25000,max_depth=7,learning_rate=0.01,)
clf.fit(class_x_tra,class_y_tra)
clf.score(class_x_val,class_y_val)
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
from sklearn.metrics import roc_curve
y_pred_proba = clf.predict_proba(class_x_val)[:, 1]
fpr, tpr, thresholds = roc_curve(class_y_val, y_pred_proba)
auc = roc_auc_score(class_y_val, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='Gradient Boosting (AUC = {:.2f})'.format(auc))
plt.plot([0, 1], [0, 1], 'k--') 
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ILLNM Gradient boosting ROC curve')
plt.legend(loc='lower right')
plt.show()
