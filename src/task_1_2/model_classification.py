import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold, LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from utils import *
import argparse
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA, SparsePCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

param_gridsearchCV = {
    'LogisticRegression': {
        'penalty': ['l1', 'l2'],
        'C': [0.001, 0.01, 0.1, 1, 10 ],
        'solver': ['newton-cg',  'liblinear', 'sag', 'saga'],
        'fit_intercept': ['True', 'False'],
        'tol': [0.0001, 0.001, 0.01, 0.1, 1]
        # 'max_iter': [100, 200, 300, 400, 500]
        }, 
    'LDA': {
        'solver': ['svd', 'lsqr', 'eigen'],
        'shrinkage': [None, 'auto'],
        'n_components': [None, 1, 2, 3],
        'tol': [1.0e-4, 1.0e-3, 1.0e-2],
        'store_covariance': [True, False],
        # 'priors': [None, [0.1, 0.2, 0.7], [0.3, 0.3, 0.4]],
        }, 
    'GaussianNB': {'var_smoothing': [1e-9, 1e-8, 1e-7]},
        'SVM': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'degree': [2, 3, 4],  # applicable only for 'poly' kernel
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10, 100],
        'coef0': [0.0, 0.1, 0.5],
        'shrinking': [True, False],
        'probability': [True, False],
        'tol': [1e-3, 1e-4, 1e-5],
        'cache_size': [100, 200, 300],
        'class_weight': [None, 'balanced'],
        'verbose': [True, False],
        'decision_function_shape': ['ovo', 'ovr'],
        'break_ties': [True, False],
        },
    'DecisionTree': {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [5, 10, 15, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2', None]
    }, 
    'RandomForest':  {
        'n_estimators': [50, 100, 200],
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 10, 15, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2', None],
        'bootstrap': [True, False]
    }

}

def evaluate_classification_models(data_dir, test_size, stratify = True, type_pca="pca", n_components=50, is_scaler=True, is_show_metrics=True, model="SVM", validation=True, valid_method='kfold', n_splits=5, tuning=True):
    X, y = load_dataset(data_dir)
    X = X.reshape((X.shape[0],X.shape[1]*X.shape[2]))
    predictions = {"pred":[], "true":[]}
    if stratify:
        X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=test_size, stratify=y, random_state=1511)
    else:
        X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=test_size, random_state=1511)

    if is_scaler:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

  
    if type_pca == "pca":
        pca = PCA(n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
    elif type_pca == "kernel":
        pca = KernelPCA(n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
    elif type_pca == "incremental":
        pca = IncrementalPCA(n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
    elif type_pca == "sparse":
        pca = SparsePCA(n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
    else: 
      raise Exception("Name of PCA should be in the list ['pca', 'kernel', 'incremental', 'sparse']")

    if model == 'LogisticRegression':
        model = LogisticRegression()
    elif model == 'LDA':
        model = LinearDiscriminantAnalysis()
    elif model == 'GaussianNB':
        model = GaussianNB()
    elif model == 'SVM':
        model = SVC()
    elif model == 'DecisionTree':
        model = DecisionTreeClassifier()
    elif model == 'RandomForest':
        model = RandomForestClassifier()
    else: 
        raise Exception("Name of model should be in the list ['LogisticRegression', 'LDA', 'GaussianNB', 'SVM', 'DecisionTree', 'RandomForest']")

    if validation: 
        if valid_method == 'kfold':
            cross_val=KFold(n_splits=n_splits, shuffle=True, random_state=0)
        elif valid_method == 'stratifiedkfold':
            cross_val=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        elif valid_method == 'leaveoneout':
            cross_val=LeaveOneOut()
        else:
            raise Exception("Name of validation method should be in the list ['kfold', 'stratifiedkfold', 'leaveoneout']")
        if isinstance(model, SVC):
            model.kernel = 'linear' 
        scores =cross_val_score(model, X_train_pca, y_train, cv=cross_val)
        model.fit(X_train_pca, y_train)
        y_pred = model.predict(X_test_pca)
        
    else:
        if tuning:
            kfold = KFold(n_splits=5, shuffle=True, random_state=0)
            grid_result = GridSearchCV(model, param_grid=param_grid[model], cv=kfold, scoring='accuracy')
            grid_result.fit(X_train_pca, y_train)
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            model = grid_result.best_estimator_
            y_pred = model.predict(X_test_pca)
        else: 
            model.fit(X_train_pca, y_train)
            y_pred = model.predict(X_test_pca)
    if is_show_metrics:
        performance_report(y_test, y_pred, model)
        print(classification_report(y_test, y_pred))
        plot_feature_importance(pca, 20, n_components, model)
    return performance_report(y_test, y_pred, model)



def plot_feature_importance(pca, n_features, n_components, model):
    if isinstance(pca, KernelPCA):
        raise Exception("KernelPCA does not have components_ attribute")
    else:
        eigen_faces = pca.components_.reshape((len(pca.components_), 112, 92))[:n_components]

    coefficients = model.coef_[0]
    feature_names = [i for i in range(n_components)]
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': np.abs(coefficients)})
    feature_importance = feature_importance.sort_values('Importance', ascending=True)
    feature_importance = feature_importance[:n_features]
    nrows = int(n_features/10) if n_features%10 == 0 else int(n_features/10) + 1
    fig, axarr=plt.subplots(nrows=nrows, ncols=10, figsize=(15,15))
    axarr=axarr.flatten()
    for i in range(nrows * 10):
        index = feature_importance.iloc[i, feature_importance.columns.get_loc('Feature')]
        axarr[i].imshow(eigen_faces[index],cmap="gray")
        axarr[i].set_xticks([])
        axarr[i].set_yticks([])
        axarr[i].set_title("eigen id:{}".format(i))
    plt.suptitle("Feature Importance".format(10*"=", 10*"="))
    plt.show()



if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Evaluate models with various parameters.')
    parser.add_argument('--data_dir', type=str, default='G:\My Drive\Data Mining\Project 1\ATT images', help='Path to the dataset')
    parser.add_argument('--test_size', type=float, default=0.4, help='Test size for train_test_split')
    parser.add_argument('--stratify', type=bool, default=True, help='Whether to stratify split or not')
    parser.add_argument('--type_pca', type=str, default='kernel', choices=['pca', 'kernel', 'incremental', 'sparse'], help='Type of PCA')
    parser.add_argument('--n_components', type=int, default=50, help='Number of components for PCA')
    parser.add_argument('--is_scaler', type=bool, default=False, help='Whether to scale the data or not')
    parser.add_argument('--is_show_metrics', type=bool, default=True, help='Whether to show performance metrics or not')
    parser.add_argument('--model', type=str, default='LogisticRegression', choices=['LogisticRegression', 'LDA', 'GaussianNB', 'SVM', 'DecisionTree', 'RandomForest'], help='Model type')
    parser.add_argument('--validation', type=bool, default=True, help='Perform validation or not')
    parser.add_argument('--valid_method', type=str, default='kfold', choices=['kfold', 'stratifiedkfold', 'leaveoneout'], help='Validation method')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of splits for cross-validation')
    parser.add_argument('--tuning', type=bool, default=False, help='Perform hyperparameter tuning or not')

    args = parser.parse_args()
    data_dir = args.data_dir
    test_size = args.test_size
    stratify = args.stratify
    type_pca = args.type_pca
    n_components = args.n_components
    is_scaler = args.is_scaler
    is_show_metrics = args.is_show_metrics
    model = args.model
    validation = args.validation
    valid_method = args.valid_method
    n_splits = args.n_splits
    tuning = args.tuning

    evaluate_classification_models(data_dir, test_size, stratify, type_pca, n_components, is_scaler, is_show_metrics, model, validation, valid_method, n_splits, tuning)
















