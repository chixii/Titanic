# data analysis and wrangling
import warnings

import inline
import pandas as pd
import numpy as np
import random as rnd
import os
import re
import warnings

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv('HousePrice/Data/train.csv')
test_df = pd.read_csv('HousePrice/Data/test.csv')

def FindandPrediction():
    # Logistic Regression
    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    Y_pred = logreg.predict(X_test)
    acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
    print("LOG score: " + '{:2.2f}'.format(acc_log))

    # Print Correlation
    coeff_df = pd.DataFrame(train_df.columns.delete(0))
    coeff_df.columns = ['Feature']
    coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

    print(coeff_df.sort_values(by='Correlation', ascending=False))

    # Support Vector Machines
    svc = SVC()
    svc.fit(X_train, Y_train)
    Y_pred = svc.predict(X_test)
    acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

    # KNN or k-Nearest Neighbors
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

    # Gaussian Naive Bayes
    gaussian = GaussianNB()
    gaussian.fit(X_train, Y_train)
    Y_pred = gaussian.predict(X_test)
    acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

    # Perceptron
    perceptron = Perceptron()
    perceptron.fit(X_train, Y_train)
    Y_pred = perceptron.predict(X_test)
    acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

    # Linear SVC
    linear_svc = LinearSVC()
    linear_svc.fit(X_train, Y_train)
    Y_pred = linear_svc.predict(X_test)
    acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

    # Stochastic Gradient Descent
    sgd = SGDClassifier()
    sgd.fit(X_train, Y_train)
    Y_pred = sgd.predict(X_test)
    acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

    # Decision Tree
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, Y_train)
    Y_pred = decision_tree.predict(X_test)
    acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

    # Random Forest
    random_forest = RandomForestClassifier(n_estimators=100, n_jobs=1)
    random_forest.fit(X_train, Y_train)
    Y_pred = random_forest.predict(X_test)
    random_forest.score(X_train, Y_train)
    acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

    models = pd.DataFrame({
        'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
                  'Random Forest', 'Naive Bayes', 'Perceptron',
                  'Stochastic Gradient Decent', 'Linear SVC',
                  'Decision Tree'],
        'Score': [acc_svc, acc_knn, acc_log,
                  acc_random_forest, acc_gaussian, acc_perceptron,
                  acc_sgd, acc_linear_svc, acc_decision_tree]})
    print(models.sort_values(by='Score', ascending=False))

    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
    submission.to_csv('submission.csv', index=False)

def main():
    combine = [train_df, test_df]
    print(train_df.columns.values)
    # train_df.head()
    df_train = pd.DataFrame(train_df['MSZoning'])
    df_test = pd.DataFrame(train_df['MSZoning'])
    print(list(df_train['MSZoning'].value_counts().index))
    print(list(df_test['MSZoning'].value_counts().index))
    for dataset in combine:
        dataset['MSZoning'] = dataset['MSZoning'].map({'RL': 1, 'RM': 2, 'FV': 3, 'RH': 4, 'C (all)': 5}).astype(int)

    # g = sns.FacetGrid(train_df, col='Survived')
    # g.map(plt.hist, 'Age', bins=20, density=True)
    # grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=2.2, aspect=1.6)
    # grid.map(plt.hist, 'Age', alpha=.5, bins=20)
    # grid.add_legend();





main()
X_train = train_df
Y_train = train_df["SalePrice"]
X_test = test_df.drop("Id", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape

