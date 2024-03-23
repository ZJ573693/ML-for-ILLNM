Install skleran
!pip install -U scikit-learn
!pip install pandas

Import data
import pandas as pd
data2=pd.read_csv('/Users/zj/Desktop/4.机器学习/数据分析/PYTHON数据/测试集赋值的数据.csv')
data.to_csv('/Users/zj/Desktop/4.机器学习/数据分析/PYTHON数据/导入到爬虫的数据.csv')
data2.head
1.Encoding Categorical Variables
data2.head
data_category = data2.select_dtypes(include=['object'])
data_category
data_Number=data2.select_dtypes(exclude=['object'])
data_Number
data_Number.columns.values
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
encoder.fit(data_category)
data_category_enc = pd.DataFrame(encoder.transform(data_category), columns=data_category.columns)
data_category_enc
data_category_enc['Age'].value_counts()
data_category['Age'].value_counts()
data_enc=pd.concat([data_category_enc,data_Number],axis=1)
data_enc
data_enc.to_csv('/Users/zj/Desktop/4.机器学习/数据分析/PYTHON数据/测试集编码后的数据.csv')

2、Missing Value Imputation
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
data_encImpute = pd.DataFrame(imp.fit_transform(data_enc))
data_encImpute.columns = data_enc.columns
data_encImpute
data_encImpute['II.level.LNM'].value_counts()
data_encImpute.to_csv('/Users/zj/Desktop/4.机器学习/数据分析/PYTHON数据/2.测试集插补缺失值后的数据.csv')
3、Data Correction and Normalization for Numerical Data
data_scale=data_encImpute
target=data_encImpute['ILLNM'].astype(int)
target.value_counts()
from sklearn import preprocessing
scaler=preprocessing.StandardScaler()
data_scaled=pd.DataFrame(scaler.fit_transform(data_scale))
data_scaled.columns=data_scale.columns
data_scaled
data_encImpute.to_csv('/Users/zj/Desktop/4.机器学习/数据分析/PYTHON数据/矫正后的数据.csv')
4、Dimensionality Reduction (Reducing the issue of multicollinearity among factors)
data=pd.read_csv('/Users/zj/Desktop/4.机器学习/数据分析/PYTHON数据/矫正后的数据-纯数值变量_ILLNM.csv')
data.iloc[:,[0,2,4]]
data.shape
data.info()
#4.1Removing Low Variance Features
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8))) 
data_sel = sel.fit_transform(data)
data_sel
a=sel.get_support(indices=True)
a
data.iloc[:,a]
data_sel=data.iloc[:,a]
data_sel.info()
#4.2Univariate Feature Selection
from sklearn.feature_selection import SelectKBest, chi2
data=pd.read_csv('/Users/zj/Desktop/4.机器学习/数据分析/PYTHON数据/矫正后的数据-纯数值变量_ILLNM.csv')
data_feature = data[['Age','Sex','Classification.of.BMI','Tumor.border','Aspect.ratio','Ingredients','Internal.echo.pattern','Internal.echo.homogeneous','Calcification','Tumor.Peripheral.blood.flow','Tumor.internal.vascularization','ETE','Size','Location','Mulifocality','Hashimoto','T.staging','Side.of.position',
'ICLNM','prelaryngeal.LNM','pretracheal.LNM','paratracheal.LNM','RLNLNM',
'age','size',
'ICLNMR','ICNLNM','prelaryngeal.LNMR','prelaryngeal.NLNM',
'pretracheal.LNMR','pretracheal.NLNM',
'paratracheal.LNMR','paratracheal.NLNM',
'RLNLNMR','RLNNLNM']]
data_feature.shape
data_target=data['ILLNM']
data_target.unique()
set_kit=SelectKBest(chi2,k=20)
data_sel=set_kit.fit_transform(data_feature,data_target)
data_sel.shape
a=set_kit.get_support(indices=True)
a
data_sel=data_feature.iloc[:,a]
data_sel
data_sel.info()
#4.3Recursive Feature Elimination (RFE) - Linear Models
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR 
from sklearn.model_selection import cross_val_score 
data=pd.read_csv('/Users/zj/Desktop/4.机器学习/数据分析/PYTHON数据/矫正后的数据-纯数值变量_ILLNM.csv')
data_feature = data[['Age','Sex','Classification.of.BMI','Tumor.border','Aspect.ratio','Ingredients','Internal.echo.pattern','Internal.echo.homogeneous','Calcification','Tumor.Peripheral.blood.flow','Tumor.internal.vascularization','ETE','Size','Location','Mulifocality','Hashimoto','T.staging','Side.of.position',
'ICLNM','prelaryngeal.LNM','pretracheal.LNM','paratracheal.LNM','RLNLNM',
'age','size',
'ICLNMR','ICNLNM','prelaryngeal.LNMR','prelaryngeal.NLNM',
'pretracheal.LNMR','pretracheal.NLNM',
'paratracheal.LNMR','paratracheal.NLNM',
'RLNLNMR','RLNNLNM']]
data_feature.shape
estimator=SVR(kernel='linear')
sel=RFE(estimator,n_features_to_select=25,step=1) 
data_target=data['ILLNM']
data_target.unique()
sel.fit(data_feature,data_target)
a=sel.get_support(indices=True)
a
data_sel=data_feature.iloc[:,a]
data_sel=data_feature.iloc[:,a]
data_sel
data_sel.info()
#4.4RFECV (Recursive Feature Elimination with Cross-Validation)
RFC_ = RandomForestClassifier() 
RFC_.fit(data_sel, data_target)
c = RFC_.feature_importances_
print('重要性：')
print(c)
selector = RFECV(RFC_, step=1, cv=10,min_features_to_select=10) 
selector.fit(data_sel, data_target)
X_wrapper = selector.transform(data_sel)
score = cross_val_score(RFC_, X_wrapper, data_target, cv=10).mean() 
print(score)
print('最佳数量和排序')
print(selector.support_)
print(selector.n_features_)
print(selector.ranking_)
a = selector.get_support(indices=True)
data_sel.iloc[:,a]
data_sel=data_feature.iloc[:,a]
data_sel.info()
import matplotlib.pyplot as plt
score = []
best_score = 0
best_features = 0
for i in range(1, 8):
X_wrapper = RFE(RFC_, n_features_to_select=i, step=1).fit_transform(data_sel, data_target) # 最优特征
once = cross_val_score(RFC_, X_wrapper, data_target, cv=10).mean()
score.append(once)
if once > best_score:
best_score = once
best_features = i
print("当前最高得分:", best_score)
print("最佳特征数量:", best_features)
print("得分列表:", score)
plt.figure(figsize=[20, 5])
plt.plot(range(1, 8), score)
plt.show()
from sklearn.model_selection import StratifiedKFold
rfecv=RFECV(estimator=RFC_,step=1,cv=StratifiedKFold(2),scoring='accuracy')
rfecv.fit(data,data_target)
print("最优特征数量：%d" % rfecv.n_features_)
print("选择的特征：", rfecv.support_)
print("特征排名：", rfecv.ranking_)
print("Optimal number of features: %d" % selector.n_features_)
# plot number of features vs. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross-validation score (number of correct classifications)")
plt.plot(range(1, len(selector.cv_results_['mean_test_score']) + 1), selector.cv_results_['mean_test_score'])
plt.show()
print("Optimal number of features: %d" % rfecv.n_features_)
# plot number of features vs. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross-validation score (number of correct classifications)")
plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
plt.show()
rfecv.get_support(indices=True)
a
data.iloc[:,a]
data_sel=data.iloc[:,a]
data_sel.info()
#4.5 L1 Regularization-based Feature Selection - SelectFromModel with Linear Regression and LinearSVC for Classification
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold
clf = LogisticRegression()
clf.fit(data_feature, data_target)

model = SelectFromModel(clf, prefit=True)
data_new = model.transform(data_feature)
model.get_support(indices=True)
a=model.get_support(indices=True)
data_features=pd.DataFrame(data_feature)
data_features.columns=data_feature.columns
data_featurenew=data_features.iloc[:,a]
data_featurenew
data_featurenew.info()
#4.6Tree-based Feature Selection
data=pd.read_csv('/Users/zj/Desktop/4.机器学习/数据分析/PYTHON数据/矫正后的数据-纯数值变量_ILLNM.csv')
data_feature = data[['Age','Sex','Classification.of.BMI','Tumor.border','Aspect.ratio','Ingredients','Internal.echo.pattern','Internal.echo.homogeneous','Calcification','Tumor.Peripheral.blood.flow','Tumor.internal.vascularization','ETE','Size','Location','Mulifocality','Hashimoto','T.staging','Side.of.position',
'prelaryngeal.LNM','pretracheal.LNM','paratracheal.LNM','RLNLNM','ICLNM',
'age','size',
'ICLNMR','ICNLNM','prelaryngeal.LNMR','prelaryngeal.NLNM',
'pretracheal.LNMR','pretracheal.NLNM',
'paratracheal.LNMR','paratracheal.NLNM',
'RLNLNMR','RLNNLNM']]
data_target=data['ILLNM']
data_target.unique()
clf = ExtraTreesClassifier()
clf.fit(data_feature, data_target)
clf.feature_importances_
model=SelectFromModel(clf,prefit=True)
x_new=model.transform(data_feature)
x_new
model.get_support(indices=True)
a=model.get_support(indices=True)
data_features=pd.DataFrame(data_feature)
data_features.columns=data_feature.columns
data_featurenew=data_features.iloc[:,a]
data_featurenew
data_featurenew.info()
