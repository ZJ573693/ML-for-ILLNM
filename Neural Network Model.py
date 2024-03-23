### 5.8.3 Model Fitting
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
clf=MLPClassifier(hidden_layer_sizes=(10,50),activation='relu',solver='adam',
alpha=0.0001,batch_size='auto',learning_rate='constant',
learning_rate_init=0.001)
clf.fit(class_x_tra,class_y_tra)
clf.score(class_x_val,class_y_val)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
param_grid = {
'hidden_layer_sizes': [(10,), (20,), (30,)],
'activation': ['relu', 'tanh'],
'solver': ['adam', 'sgd'],
'alpha': [0.0001, 0.001, 0.01],
'learning_rate': ['constant', 'adaptive'],
'learning_rate_init': [0.001, 0.01, 0.1]
}
clf = MLPClassifier()
grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(class_x_tra, class_y_tra)
print("Best Parameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)
best_clf = grid_search.best_estimator_
best_clf.fit(class_x_tra, class_y_tra)
score = best_clf.score(class_x_val, class_y_val)
print("Validation Score: ", score)
### 5.8.4 Feature Variable Histogram
#### 5.8.4.1 Without Importance Values
import matplotlib.pyplot as plt
best_clf.fit(class_x_tra, class_y_tra)
weights = best_clf.coefs_
feature_importance = np.sum(np.abs(weights[0]), axis=1)
all_indices = range(len(feature_importance))
all_importance = feature_importance[all_indices]
all_features = [feature_name[i] for i in all_indices]
sorted_indices = sorted(all_indices, key=lambda i: all_importance[i], reverse=True)
sorted_importance = [all_importance[i] for i in sorted_indices]
sorted_features = [all_features[i] for i in sorted_indices]
for feature, importance in zip(sorted_features, sorted_importance):
print(f"Variable: {feature}, Importance: {importance}")
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_features)), sorted_importance, tick_label=sorted_features, color='pink')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importances - Neural Network')
plt.gca().invert_yaxis()
plt.show()
#### 5.8.4.2 With Importance Values at the End
import matplotlib.pyplot as plt
best_clf.fit(class_x_tra, class_y_tra)
weights = best_clf.coefs_
feature_importance = np.sum(np.abs(weights[0]), axis=1)
all_indices = range(len(feature_importance))
all_importance = feature_importance[all_indices]
all_features = [feature_name[i] for i in all_indices]
sorted_indices = sorted(all_indices, key=lambda i: all_importance[i], reverse=True)
sorted_importance = [all_importance[i] for i in sorted_indices]
sorted_features = [all_features[i] for i in sorted_indices]
for feature, importance in zip(sorted_features, sorted_importance):
print(f"Variable: {feature}, Importance: {importance}")
plt.figure(figsize=(10, 6))
bars = plt.barh(range(len(sorted_features)), sorted_importance, tick_label=sorted_features, color='pink')
for i, bar in enumerate(bars):
importance = sorted_importance[i]
plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{importance:.2f}', va='center')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importances - Neural Network')
plt.gca().invert_yaxis()
plt.show()
#### 5.8.4.3 Top Ten Ranked
import matplotlib.pyplot as plt
best_clf.fit(class_x_tra, class_y_tra)
weights = best_clf.coefs_
feature_importance = np.sum(np.abs(weights[0]), axis=1)
top_indices = np.argsort(feature_importance)[::-1][:10]
top_importance = feature_importance[top_indices]
top_features = [feature_name[i] for i in top_indices]
for feature, importance in zip(top_features, top_importance):
print(f"Variable: {feature}, Importance: {importance}")
plt.figure(figsize=(10, 6))
bars = plt.barh(range(len(top_features)), top_importance, tick_label=top_features, color='pink')
for i, bar in enumerate(bars):
importance = top_importance[i]
plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{importance:.2f}', va='center')

plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Top 10 Feature Importances - Neural Network')
plt.gca().invert_yaxis()
plt.show()
####5.8.4.4 Importance Ratio rankings
import matplotlib.pyplot as plt
best_clf.fit(class_x_tra, class_y_tra)
weights = best_clf.coefs_
feature_importance = np.sum(np.abs(weights[0]), axis=1)
total_importance = np.sum(feature_importance)
importance_ratio = feature_importance / total_importance
all_indices = range(len(importance_ratio))
all_ratio = importance_ratio[all_indices]
all_features = [feature_name[i] for i in all_indices]
sorted_indices = sorted(all_indices, key=lambda i: all_ratio[i], reverse=True)
sorted_ratio = [all_ratio[i] for i in sorted_indices]
sorted_features = [all_features[i] for i in sorted_indices]

for feature, ratio in zip(sorted_features, sorted_ratio):
print(f"Variable: {feature}, Ratio: {ratio}")
plt.figure(figsize=(10, 6))
bars = plt.barh(range(len(sorted_features)), sorted_ratio, tick_label=sorted_features, color='pink')
for i, bar in enumerate(bars):
ratio = sorted_ratio[i]
plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{ratio:.2}', va='center')

plt.xlabel('Importance Ratio')
plt.ylabel('Features')
plt.title('Feature Importance Ratios - Neural Network')
plt.gca().invert_yaxis()
plt.show()
###5.8.4 Evaluation indicators
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
y_pred = best_clf.predict(class_x_val)
accuracy = accuracy_score(class_y_val, y_pred)
auc = roc_auc_score(class_y_val, best_clf.predict_proba(class_x_val)[:, 1])
tn, fp, fn, tp = confusion_matrix(class_y_val, y_pred).ravel()
precision = tp / (tp + fp)
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)
npv = tn / (tn + fn)
ppv = tp / (tp + fp)
recall = sensitivity
f1_score = 2 * (precision * recall) / (precision + recall)
fpr = fp / (fp + tn)
print("Accuracy: ", accuracy)
print("AUC: ", auc)
print("Precision: ", precision)
print("Specificity: ", specificity)
print("Sensitivity: ", sensitivity)
print("Negative Predictive Value: ", npv)
print("Positive Predictive Value: ", ppv)
print("Recall: ", recall)
print("F1 Score: ", f1_score)
print("False Positive Rate: ", fpr)
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(class_y_val, best_clf.predict_proba(class_x_val)[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.fill_between(fpr, tpr, alpha=0.3)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ILLNM MLP ROC Curve')
plt.legend(loc='lower right')
plt.show()
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve
precision, recall, _ = precision_recall_curve(class_y_val, best_clf.predict_proba(class_x_val)[:, 1])
smooth_precision = np.linspace(0, 1, 100)
smooth_recall = np.interp(smooth_precision, precision, recall)
plt.figure(figsize=(8, 6))
plt.plot(smooth_recall, smooth_precision, color='b', lw=2, label='Precision-Recall Curve')
plt.fill_between(smooth_recall, smooth_precision, alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('ILLNM MLP Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()
import numpy as np

weights = best_clf.coefs_
feature_contributions = np.abs(weights).mean(axis=0)
feature_contributions_dict = dict(zip(data.columns, feature_contributions))
sorted_contributions = sorted(feature_contributions_dict.items(), key=lambda x: x[1], reverse=True)
for feature, contribution in sorted_contributions:
print(f"{feature}: {contribution}")
import matplotlib.pyplot as plt
features = [feature for feature, _ in sorted_contributions]
contributions = [contribution for _, contribution in sorted_contributions]
plt.figure(figsize=(10, 6))
plt.bar(features, contributions)
plt.xticks(rotation=90)
plt.xlabel('Feature Variables')
plt.ylabel('Contribution')
plt.title('Feature Variable Contributions')
plt.tight_layout()
plt.show()
