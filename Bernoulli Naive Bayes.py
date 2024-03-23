### 5.7.3 Model Fitting (Similar to previous code, no need to repeat)
from sklearn.naive_bayes import BernoulliNB #二分类的
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB
param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0], 'fit_prior': [True, False]}
clf = BernoulliNB()
grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(class_x_tra, class_y_tra)

best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("Best Parameters: ", best_params)
print("Best Score: ", best_score)
best_clf = BernoulliNB(alpha=best_params['alpha'], fit_prior=best_params['fit_prior'])
best_clf.fit(class_x_tra, class_y_tra)
accuracy = best_clf.score(class_x_val, class_y_val)
print("Validation Accuracy: ", accuracy)
5.7.4 Evaluation indicators
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
y_pred = best_clf.predict(class_x_val)
y_pred_proba = best_clf.predict_proba(class_x_val)[:, 1]
accuracy = accuracy_score(class_y_val, y_pred)
auc = roc_auc_score(class_y_val, y_pred_proba)
dca = 2 * (auc - 0.5)
precision = precision_score(class_y_val, y_pred)
recall = recall_score(class_y_val, y_pred)
f1 = f1_score(class_y_val, y_pred)
tn, fp, fn, tp = confusion_matrix(class_y_val, y_pred).ravel()
specificity = tn / (tn + fp)
npv = tn / (tn + fn)
ppv = tp / (tp + fp)
fpr = fp / (fp + tn)
print("Accuracy: ", accuracy)
print("AUC: ", auc)
print("DCA: ", dca)
print("Precision: ", precision)
print("Specificity: ", specificity)
print("Sensitivity/Recall: ", recall)
print("Negative Predictive Value: ", npv)
print("Positive Predictive Value: ", ppv)
print("F1 Score: ", f1)
print("False Positive Rate: ", fpr)
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(class_y_val, y_pred_proba)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(' ROC CurvILLNM Bayesian-modele')
plt.show()
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(class_y_val, y_pred_proba)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('ILLNM Bayesian-model Precision-Recall Curve')
plt.show()
from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()
clf.fit(class_x_tra,class_y_tra)
clf.score(class_x_val,class_y_val)
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
y_pred = clf.predict(class_x_val)
y_pred_proba = clf.predict_proba(class_x_val)[:, 1]
accuracy = accuracy_score(class_y_val, y_pred)
auc = roc_auc_score(class_y_val, y_pred_proba)
dca = 2 * (auc - 0.5)
precision = precision_score(class_y_val, y_pred)
recall = recall_score(class_y_val, y_pred)
f1 = f1_score(class_y_val, y_pred)
tn, fp, fn, tp = confusion_matrix(class_y_val, y_pred).ravel()
specificity = tn / (tn + fp)
npv = tn / (tn + fn)
ppv = tp / (tp + fp)
fpr = fp / (fp + tn)

print("Accuracy: ", accuracy)
print("AUC: ", auc)
print("DCA: ", dca)
print("Precision: ", precision)
print("Specificity: ", specificity)
print("Sensitivity/Recall: ", recall)
print("Negative Predictive Value: ", npv)
print("Positive Predictive Value: ", ppv)
print("F1 Score: ", f1)
print("False Positive Rate: ", fpr)
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thresholds = roc_curve(class_y_val, y_pred_proba)
auc = roc_auc_score(class_y_val, y_pred_proba)
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ILLNM GNB ROC Curve')
plt.legend(loc="lower right")
plt.text(0.6, 0.2, 'AUC = %0.2f' % auc)
plt.show()
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(class_y_val, y_pred_proba)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('ILLNM GMN Precision-Recall Curve')
plt.show()
