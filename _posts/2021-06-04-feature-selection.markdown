---
layout: post
title:  "特征选择Feature Selection"
date:   2021-06-04 16:15:36 +0800
categories: feature selection, sklearn
---
本文介绍scikit-learn中常见的几种[特征选择方法](https://scikit-learn.org/stable/modules/feature_selection.html)，并提供相应的代码。
## 特征选择的优点
特征选择主要有三个优点，分别是：
- 降低过拟合：剔除与结果不相关的特征让减少数据噪音对于结果的影响；
- 提高准确性：减少不相关的特征可以提高结果的准确度；
- 提高训练速度：更少的数据意味着训练时间更少了。

## 特征选择的分类
按照特征选择的形式可以分为三类：
- Filter: 过滤法，即在训练之前对数据进行特征选择，设定阈值或者个数选择特征；
- Wrapper: 包装法，将目标学习器的性能作为评价标准（一般是效果评分），每次选择/排除若干特征
- Embedded: 嵌入法，学习器学习过程中进行特征选择。

## Filter
### 单变量选择 Univariate feature selection
单独计算每个变量的统计指标，选出与目标变量关系最强的特征变量。
对于目标变量是离散的分类问题，可以使用：
卡方验证chi2、f_classif（ANOVA F-value for classification tasks），mutual_info_classif等
```python
>>> from sklearn.datasets import load_iris
>>> from sklearn.feature_selection import SelectKBest
>>> from sklearn.feature_selection import f_classif
>>> X, y = load_iris(return_X_y=True)
>>> X_new = SelectKBest(f_classif, k=2).fit_transform(X, y)
```
对于目标变量是连续的回归问题，可以使用：
皮尔森相关系数（Pearson correlation），f_regression（F-value for regression tasks），mutual_info_regression

## Wrapper
### 递归特征消除 Recursive Feature Elimination-RFE
RFE是一种wrapper类型的特征选择方法，即给出不同的机器学习算法，它们被包装(wrapped)在RFE中来做特征选择。RFE递归地移除每轮训练中嘴不重要的特征，直到剩余的特征数量达到设定的特征数量 -- 具体来说是通过REF每轮返回的*coef_*或者*feature_importances_*属性来判断每个特征的重要程度。
```python
>>> from sklearn.datasets import make_friedman1
>>> from sklearn.feature_selection import RFE
>>> from sklearn.svm import SVR
>>> X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
>>> model = SVR(kernel="linear")
>>> selector = RFE(model, n_features_to_select=5, step=1)
>>> selector = selector.fit(X, y)
```
## Embedded
### SelectFromModel
sklearn的feature selection中内置了一个SelectFromModel函数，它可以结合模型本身的指标对特征进行选择，只需要模型有*coef_*或者*feature_importances_*属性。如果属性值*coef_*或者*feature_importances_*低于预设阈值*threshold*，这些特征将被剔除掉。除了设定阈值，也可以通过调用启发式算法来设定阈值，包括mean, median和它们与浮点数的乘积。
在模型中有两种选择方式：
- 基于L1的特征选择：使用L1正则化的线性模型可以得到系数解，常用的稀疏模型有（回归）linear_model.Lasso，（分类）linear_model.LogisticRegression和svm.LinearSVC.
```python
>>> from sklearn.svm import LinearSVC
>>> from sklearn.datasets import load_iris
>>> from sklearn.feature_selection import SelectFromModel
>>> iris = load_iris()
>>> X, y = iris.data, iris.target
>>> lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
>>> model = SelectFromModel(lsvc, prefit=True)
>>> X_new = model.transform(X)
```
- 基于树的特征选择：树类的算法包括决策树sklearn.tree，森林sklearn.ensemble等，能够计算特征的重要程度，从而去除不相关的特征。
```python
>>> from sklearn.ensemble import ExtraTreesClassifier
>>> from sklearn.datasets import load_iris
>>> from sklearn.feature_selection import SelectFromModel
>>> X, y = load_iris(return_X_y=True)
>>> clf = ExtraTreesClassifier(n_estimators=50)
>>> clf = clf.fit(X, y)
>>> model = SelectFromModel(clf, prefit=True)
>>> X_new = model.transform(X)
```
### Pipeline
将特征选择融入到Pipeline中，作为学习之前的预处理。在sklearn中推荐使用sklearn.pipeline.Pipeline.
```python
clf = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
  ('classification', RandomForestClassifier())
])
clf.fit(X, y)
```
参考：
https://www.cnblogs.com/stevenlk/p/6543628.html
