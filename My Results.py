## 1.1.1 ROC Curve for Training Set
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
import numpy as np
data = pd.read_csv('/Users/zj/Desktop/4.机器学习/数据分析/PYTHON数据/矫正后的数据-纯数值变量.csv')
data_feature = data[['Sex','Tumor.border','Aspect.ratio',
'Calcification','Tumor.internal.vascularization',
'ETE','Size','Hashimoto',
'prelaryngeal.LNM','paratracheal.LNM','RLNLNM','ICLNM',
'ICLNMR','pretracheal.NLNM',
'paratracheal.LNMR',
'RLNLNMR',
]]
data_target=data['IV.level.LNM']
data_target.unique()
data_featureCata=data[['Sex','Tumor.border','Aspect.ratio',
'Calcification','Tumor.internal.vascularization',
'ETE','Size','Hashimoto',
'prelaryngeal.LNM','paratracheal.LNM','RLNLNM','ICLNM',]]


data_featureNum=data[['ICLNMR','pretracheal.NLNM',
'paratracheal.LNMR',
'RLNLNMR',]]

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
data_featureNum=scaler.fit_transform(data_featureNum)
data_featureCata=np.array(data_featureCata)
import numpy as np

data_feature = np.hstack((data_featureCata, data_featureNum))
data_target = data['IV.level.LNM']
class_x_tra, class_x_val, class_y_tra, class_y_val = train_test_split(data_feature, data_target, test_size=0.3, random_state=0)

models = [
LogisticRegression(),
DecisionTreeClassifier(min_samples_split=3, max_depth=5, min_samples_leaf=10, max_features=20),
RandomForestClassifier(n_estimators=25000, max_depth=5, min_samples_leaf=5, min_samples_split=20),
GradientBoostingClassifier(n_estimators=5, max_depth=4, learning_rate=0.1),
SVC(kernel='linear', gamma=0.2, probability=True),
KNeighborsClassifier(n_neighbors=13),
GaussianNB(),
MLPClassifier()
]
model_names = [
'Logistic Regression',
'Decision Tree',
'Random Forest',
'Gradient Boosting',
'Support Vector Machine',
'K-Nearest Neighbors',
'Gaussian Naive Bayes',
'Neural Network'
]
colors = ['blue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'cyan']
nn_param_grid = {
'hidden_layer_sizes': [(10,), (20,), (30,)],
'activation': ['relu', 'tanh'],
'solver': ['adam', 'sgd'],
'alpha': [0.0001, 0.001, 0.01],
'learning_rate': ['constant', 'adaptive'],
'learning_rate_init': [0.001, 0.01, 0.1]
}
nn_grid_search = GridSearchCV(estimator=MLPClassifier(), 
param_grid=nn_param_grid, cv=10)
plt.figure(figsize=(8, 6))
for model, name, color in zip(models, model_names, colors):
if name == 'Neural Network':
nn_grid_search.fit(class_x_tra, class_y_tra)
best_nn_model = nn_grid_search.best_estimator_
model = best_nn_model
else:
model.fit(class_x_tra, class_y_tra)
if hasattr(model, 'predict_proba'):
y_train_pred_prob = model.predict_proba(class_x_tra)[:, 1]
else:
y_train_pred_prob = model.decision_function(class_x_tra)
auc = roc_auc_score(class_y_tra, y_train_pred_prob)
fpr, tpr, _ = roc_curve(class_y_tra, y_train_pred_prob)
plt.plot(fpr, tpr, color=color, label='%s (AUC = %0.3f)' % (name, auc))
plt.grid(True)
plt.plot([0, 1], [0, 1], 'k--') 
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('IV.level.LNM ROC (Training set)')
plt.legend(loc='lower right')
plt.show()
##1.1.2 Evaluation index of training set
train_accuracy_scores = []
train_auc_scores = []
train_precision_scores = []
train_specificity_scores = []
train_sensitivity_scores = []
train_npv_scores = []
train_ppv_scores = []
train_recall_scores = []
train_f1_scores = []
train_fpr_scores = []
for model, name in zip(models, model_names):
if name == 'Neural Network':
nn_grid_search.fit(class_x_tra, class_y_tra)
best_nn_model = nn_grid_search.best_estimator_
model = best_nn_model
else:
model.fit(class_x_tra, class_y_tra)
train_y_pred = model.predict(class_x_tra)
if hasattr(model, 'predict_proba'):
train_y_pred_prob = model.predict_proba(class_x_tra)[:, 1]
else:
train_y_pred_prob = model.decision_function(class_x_tra)
train_accuracy = accuracy_score(class_y_tra, train_y_pred)
train_auc = roc_auc_score(class_y_tra, train_y_pred_prob)
train_precision = precision_score(class_y_tra, train_y_pred)
train_cm = confusion_matrix(class_y_tra, train_y_pred)
train_tn, train_fp, train_fn, train_tp = train_cm.ravel()
train_specificity = train_tn / (train_tn + train_fp)
train_sensitivity = recall_score(class_y_tra, train_y_pred)
train_npv = train_tn / (train_tn + train_fn)
train_ppv = train_tp / (train_tp + train_fp)
train_recall = train_sensitivity
train_f1 = f1_score(class_y_tra, train_y_pred)
train_fpr = train_fp / (train_fp + train_tn)
train_accuracy_scores.append(train_accuracy)
train_auc_scores.append(train_auc)
train_precision_scores.append(train_precision)
train_specificity_scores.append(train_specificity)
train_sensitivity_scores.append(train_sensitivity)
train_npv_scores.append(train_npv)
train_ppv_scores.append(train_ppv)
train_recall_scores.append(train_recall)
train_f1_scores.append(train_f1)
train_fpr_scores.append(train_fpr)
train_metrics_df = pd.DataFrame({
'Model': model_names,
'Accuracy': train_accuracy_scores,
'AUC': train_auc_scores,
'Precision': train_precision_scores,
'Specificity': train_specificity_scores,
'Sensitivity': train_sensitivity_scores,
'Negative Predictive Value': train_npv_scores,
'Positive Predictive Value': train_ppv_scores,
'Recall': train_recall_scores,
'F1 Score': train_f1_scores,
'False Positive Rate': train_fpr_scores
})
print(train_metrics_df)
train_metrics_df.to_csv('/Users/zj/Desktop/4.机器学习/结果/4.评价指标表格/4.1 IV.level.LNM训练集的评价指标.csv', index=False)
##1.1.3 DCA curve of training set
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
thresholds = np.linspace(0, 1, 100)
train_net_benefit = []
for model, model_name, color in zip(models, model_names, colors):
model.fit(class_x_tra, class_y_tra)
train_model_predictions = model.predict_proba(class_x_tra)[:, 1]
train_model_net_benefit = []
for threshold in thresholds:
train_predictions = (train_model_predictions >= threshold).astype(int)
train_net_benefit_value = (precision_score(class_y_tra, train_predictions) - threshold * (1 - precision_score(class_y_tra, train_predictions))) / (threshold + 1e-10)
train_model_net_benefit.append(train_net_benefit_value)
train_net_benefit.append(train_model_net_benefit)
train_net_benefit = np.array(train_net_benefit)
train_all_predictions = np.ones_like(class_y_tra) train_all_net_benefit = (precision_score(class_y_tra, train_all_predictions) - thresholds * (1 - precision_score(class_y_tra, train_all_predictions))) / (thresholds + 1e-10)
for i in range(train_net_benefit.shape[0]):
plt.plot(thresholds, train_net_benefit[i], color=colors[i], label=model_names[i])

plt.plot(thresholds, np.zeros_like(thresholds), color='black', linestyle='-', label='None')
plt.plot(thresholds, train_all_net_benefit, color='gray', linestyle='--', label='All')
plt.xlim(0, 0.8)
plt.ylim(-0.5,6)
plt.xlabel('Threshold')
plt.ylabel('Net Benefit')
plt.title('IV.level.LNM Decision Curve Analysis (Training set)')
plt.legend(loc='upper right')
plt.show()
1.1.4 Calibration curve of the training set
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from scipy.stats import ttest_ind
train_calibration_curves = []
train_brier_scores = []
for model, model_name, color in zip(models, model_names, colors):
model.fit(class_x_tra, class_y_tra)
y_proba = model.predict_proba(class_x_tra)[:, 1]
train_fraction_of_positives, train_mean_predicted_value = calibration_curve(class_y_tra, y_proba, n_bins=10)
train_calibration_curves.append((train_fraction_of_positives, train_mean_predicted_value, model_name, color))
train_brier_score = brier_score_loss(class_y_tra, y_proba)
train_brier_scores.append((model_name, train_brier_score))
print(f'{model_name} - Training Brier Score: {train_brier_score:.4f}')
fig, ax1 = plt.subplots(figsize=(10, 6))

for curve in train_calibration_curves:
train_fraction_of_positives, train_mean_predicted_value, model_name, color = curve
train_brier_score = next((score for model, score in train_brier_scores if model == model_name), None)
if train_brier_score is not None:
model_name += f' (Training Brier Score: {train_brier_score:.4f})'
ax1.plot(train_mean_predicted_value, train_fraction_of_positives, "s-", label=model_name, color=color)
ax1.plot([0, 1], [0, 1], "k:",label="Perfectly calibrated")
ax1.set_ylabel("Fraction of positives")
ax1.set_xlabel("Mean predicted value")
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")

plt.title("IV.level.LNM Calibration Curves (Training set)")
plt.tight_layout()
plt.show()
##1.1.5 Exact recall curve for training set
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
def plot_pr_curve(y_true, y_prob, model_name, color):
precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
aupr = auc(recall, precision)

plt.plot(recall, precision, color=color, label=model_name + f' (AUPR = {aupr:.3f})')
train_y_true_list = [class_y_tra] * len(models)
train_y_prob_list = [model.fit(class_x_tra, class_y_tra).predict_proba(class_x_tra)[:, 1] for model in models]
plt.figure(figsize=(10, 8))
for i, (train_y_true, train_y_prob, model_name, color) in enumerate(zip(train_y_true_list, train_y_prob_list, model_names, colors)):
plot_pr_curve(train_y_true, train_y_prob, model_name, color)
plt.plot([0, 1], [class_y_tra.mean(), class_y_tra.mean()], linestyle='--', color='black', label='Random Guessing')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('IV.level.LNM Precision-Recall Curve (Training set)')
plt.legend()
plt.show()
