from flask import Flask, request, render_template
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
top_features = ['ICLNMR', 'ETE', 'Tumor.border', 'ICNLNM', 'Tumor.internal.vascularization',
'Calcification', 'ICLNM', 'paratracheal.LNMR', 'Size', 'paratracheal.NLNM']
top_importances = [0.14, 0.12, 0.12, 0.1, 0.08, 0.08, 0.07, 0.06, 0.05, 0.02]

@app.route('/')
def index():
return render_template('calculator.html', features=top_features, importances=top_importances)

@app.route('/calculate', methods=['POST'])
def calculate_result():
feature_values = [request.form[feature] for feature in top_features]
data = pd.read_csv('/Users/zj/Desktop/4.机器学习/数据分析/PYTHON数据/矫正后的数据-纯数值变量.csv')
data_feature = data[top_features]
data_target = data['ILLNM']
model = RandomForestClassifier(n_estimators=25000, max_depth=5, min_samples_leaf=5, min_samples_split=5)
model.fit(data_feature, data_target)
result = model.predict([feature_values])
return result
if __name__ == '__main__':
app.run()
