## 7.1 Feature Ranking for Each Model
import matplotlib.pyplot as plt
fig, axs = plt.subplots(len(models), 1, figsize=(10, 6 * len(models)))
for i, (model, model_name, color) in enumerate(zip(models, model_names, colors)):
model.fit(class_x_tra, class_y_tra)
if hasattr(model, 'coef_'):
feature_importance = np.abs(model.coef_[0])
else:
feature_importance = np.zeros(len(data.columns))
sorted_idx = np.argsort(feature_importance)
axs[i].barh(range(len(sorted_idx)), feature_importance[sorted_idx], color=color)
axs[i].set_yticks(range(len(sorted_idx)))
axs[i].set_yticklabels(np.array(data.columns)[sorted_idx])
axs[i].set_xlabel('Relative Importance')
axs[i].set_title(f'{model_name} - Feature Importances')
plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
for i, (model, model_name, color) in enumerate(zip(models, model_names, colors)):
fig, ax = plt.subplots(figsize=(10, 6))
model.fit(class_x_tra, class_y_tra)
if hasattr(model, 'coef_'):
feature_importance = np.abs(model.coef_[0])
else:
feature_importance = np.zeros(len(data.columns))
sorted_idx = np.argsort(feature_importance)
ax.barh(range(len(sorted_idx)), feature_importance[sorted_idx], color=color)
ax.set_yticks(range(len(sorted_idx)))
ax.set_yticklabels(np.array(data.columns)[sorted_idx])
ax.set_xlabel('Relative Importance')
ax.set_title(f'{model_name} - Feature Importances')
plt.show()
import matplotlib.pyplot as plt
fig, axs = plt.subplots(len(models), 1, figsize=(10, 6 * len(models)))
for i, (model, model_name, color) in enumerate(zip(models, model_names, colors)):
model.fit(class_x_tra, class_y_tra)
if hasattr(model, 'coef_'):
feature_importance = np.abs(model.coef_[0])
else:
feature_importance = np.zeros(len(data.columns))
sorted_idx = np.argsort(feature_importance)
axs[i].barh(range(len(sorted_idx)), feature_importance[sorted_idx], color=color)
axs[i].set_yticks(range(len(sorted_idx)))
axs[i].set_yticklabels(np.array(data.columns)[sorted_idx])
axs[i].set_xlabel('Relative Importance')
axs[i].set_title(f'{model_name} - Feature Importances')
axs[i].set_xlim(0, 2) # 设置x轴的取值范围为0-4
plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt

# Set subplot layout
fig, axs = plt.subplots(len(models), 1, figsize=(10, 6 * len(models)))

# Iterate over each model
for i, (model, model_name, color) in enumerate(zip(models, model_names, colors)):
# Train the model
model.fit(class_x_tra, class_y_tra)
# Calculate feature importance
if hasattr(model, 'coef_'):
feature_importance = np.abs(model.coef_[0])
else:
feature_importance = np.zeros(len(data.columns))
# Sort feature importance
sorted_idx = np.argsort(feature_importance)
# Plot bar chart
axs[i].barh(range(len(sorted_idx)), feature_importance[sorted_idx], color=color)
axs[i].set_yticks(range(len(sorted_idx)))
axs[i].set_yticklabels(np.array(data.columns)[sorted_idx])
axs[i].set_xlabel('Relative Importance')
axs[i].set_title(f'{model_name} - Feature Importances')

# Adjust subplot layout
plt.tight_layout()
plt.show()
