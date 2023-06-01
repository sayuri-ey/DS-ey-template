#!/usr/bin/env python
# coding: utf-8

# custom modules
from configs.custom_config import get_timezone
from pipeline.train import get_sets
from pipeline.train import plot_confusion_matrix
from pipeline.train import plot_precision_recall_curve
from pipeline.train import plot_roc_curve
# external modules
import datetime
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
# from sklearn.linear_model import LogisticRegression


def random_search(X_train, y_train):
    # Define the hyperparameter space
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=3)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num=3)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    param_dist = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap
    }

    rf = RandomForestClassifier()

    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=1,
        cv=3,
        random_state=42,
        n_jobs=-1
    )

    # Fit the randomized search cross-validation object to the data
    random_search.fit(X_train, y_train)
    print(f"Best hyperparameters: {random_search.best_params_}")

    return random_search.best_estimator_

def main():
    print(f'Started Evaluate step at {datetime.now().strftime("%H:%M:%S")}')

    # Load datasets
    print(f'{datetime.now().strftime("%H:%M:%S")}: Read train, test and valid sets')
    X_train, y_train = get_sets('train')
    X_test, y_test = get_sets('test')
    X_valid, y_valid = get_sets('valid')

    # Load model
    print(f'{datetime.now().strftime("%H:%M:%S")}: Load model')
    model = joblib.load('model/model.pkl')

    # # Define the hyperparameters to tune Logistic Regression
    # param_grid = {
    #     'penalty': ['l1', 'l2'],
    #     'C': [0.01, 0.1, 1.0, 10.0],
    #     'max_iter': [100, 200, 300]
    # }
    
    # # Define the hyperparameters to tune Random Forest
    # param_grid = {
    #     'n_estimators': [100, 200, 300],
    #     'max_depth': [5, 10, 15],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4],
    #     'max_features': ['sqrt', 'log2']
    # }
    
    # Perform a grid search to find the best hyperparameters
    print(f'{datetime.now().strftime("%H:%M:%S")}: Tune hyperparameters')
    # lr = LogisticRegression()
    # rf = RandomForestClassifier()
    # estimator = rf
    # best_estim = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5)
    # best_estim.fit(X_train, y_train)
    best_estim = random_search(X_train, y_train)

    # Evaluate the model on the test set
    print(f'{datetime.now().strftime("%H:%M:%S")}: Evaluate on test set')
    test_score = best_estim.score(X_test, y_test)
    print('Test score:', test_score)

    # Evaluate the model on the validation set
    print(f'{datetime.now().strftime("%H:%M:%S")}: Evaluate on valid set')
    valid_score = best_estim.score(X_valid, y_valid)
    print('Validation score:', valid_score)
    
    # Metrics for chosen model
    print(f'{datetime.now().strftime("%H:%M:%S")}: Plotting metrics from test set for the model')
    y_pred = best_estim.predict(X_test)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    print(f'Metrics for random_forest: precision: {precision}, recall: {recall}, f1_score: {f1_score}')
    plot_precision_recall_curve(y_test, y_pred)
    plot_roc_curve(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred)    

    # Save the model
    print(f'{datetime.now().strftime("%H:%M:%S")}: Save tuned model')
    joblib.dump(best_estim, 'model/model.pkl')

    print(f'Finished Evaluate at {datetime.now().strftime("%H:%M:%S")}')


if __name__ == "__main__":

    main()

