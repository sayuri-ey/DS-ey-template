#!/usr/bin/env python
# coding: utf-8

# custom modules
from configs.custom_config import get_timezone, get_target_column, get_feature_columns, get_categorical_columns, get_numerical_columns
# external modules
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from sklearn import linear_model, tree, ensemble, neighbors
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


# set environment and define functions

# set timezone
get_timezone()

def get_sets(set):
    X = pd.read_csv(f'data/{set}.csv')
    y = pd.read_csv(f'data/y_{set}.csv')
    return X, y

def train_model(X_train, y_train, models):
    trained_models = {}
    for model_key, model in models.items():
        # Train the model on the training data
        model.fit(X_train, y_train)
        # plot_learning_curve(model, X_train, y_train)
        
        # Add the trained model object to the dictionary of trained models
        trained_models[model_key] = model
        
    return trained_models

def plot_precision_recall_curve(y_test, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    labels = ['Not Fraud', 'Fraud']
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    cbar = ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           xlabel='Predicted label',
           ylabel='True label')
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
             rotation_mode="anchor")
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f'{cm[i, j]} ({cm[i, j]/np.sum(cm)*100:.1f}%)'
            ax.text(j, i, text,
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_title("Confusion Matrix")
    plt.show()
    
    return cm

def plot_roc_curve(y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) curve')
    plt.legend(loc="lower right")
    plt.show()

def evaluate_models(trained_models, X_test, y_test):
    results = {}
    y_preds = {}
    y_test = y_test['fraude'].to_numpy()
    for model_key, model in trained_models.items():
        print(f'Evaluating {model_key} model')
        print(model)
        # Make predictions on the test data
        y_pred = model.predict(X_test)
        
        # Calculate precision, recall, and F1-score for the model
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        plot_precision_recall_curve(y_test, y_pred)
        plot_roc_curve(y_test, y_pred)
        plot_confusion_matrix(y_test, y_pred)
        
        # Add the evaluation results to the dictionary
        results[model_key] = {'precision': precision, 'recall': recall, 'f1_score': f1_score}
        y_preds[f'{model_key}_y_pred'] = y_pred
        print(f'Metrics for {model_key}: {results[model_key]}')
        
    results_df = pd.DataFrame(results).T
    results_df.index.name = 'model'
    print(results_df)
    
    return results_df, y_preds

def plot_bar_chart(metrics_df):
    plt.figure(figsize=(10, 8))
    ax = metrics_df.plot(kind='bar')
    ax.set_xticklabels(metrics_df.index, rotation=0)
    plt.xlabel('Model')
    plt.ylabel('Metric Score')
    plt.title('Comparison of Model Performance')
    plt.legend(loc="lower right")
    plt.show()

def plot_box_plot(metrics_df):
    plt.figure(figsize=(10, 8))
    ax = metrics_df.plot(kind='box')
    ax.set_xticklabels(metrics_df.columns, rotation=0)
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title('Distribution of Model Performance')
    plt.show()
    
def plot_model_metrics(models, X_test, y_test):
    # create a grid of subplots
    fig, axes = plt.subplots(nrows=3, ncols=len(models), figsize=(15, 10))

    # plot the performance metrics for each model
    for i, model in enumerate(models):
        # plot the precision-recall curve
        plot_precision_recall_curve(model, X_test, y_test)
        axes[0, i].set_title(f"Model {i+1} PR Curve")

        # plot the ROC curve
        plot_roc_curve(model, X_test, y_test)
        axes[1, i].set_title(f"Model {i+1} ROC Curve")

        # plot the confusion matrix
        cm = confusion_matrix(y_test, model.predict(X_test))
        axes[2, i].imshow(cm, cmap=plt.cm.Blues)
        axes[2, i].set_xticks([0, 1])
        axes[2, i].set_yticks([0, 1])
        axes[2, i].set_xticklabels(["Negative", "Positive"])
        axes[2, i].set_yticklabels(["Negative", "Positive"])
        axes[2, i].set_title(f"Model {i+1} Confusion Matrix")
        
        plt.tight_layout()
        plt.show()

def main():

    print(f'Started Train step at {datetime.now().strftime("%H:%M:%S")}')

    # Read train set
    print(f'{datetime.now().strftime("%H:%M:%S")}: Read train set')
    X_train, y_train = get_sets('train')
    
    #Train models
    print(f'{datetime.now().strftime("%H:%M:%S")}: Train dataset')
    
    # Models to test

    models = {
        "logistic_regression": linear_model.LogisticRegression(),
        "decision_trees": tree.DecisionTreeClassifier(),
        "random_forest": ensemble.RandomForestClassifier(),
        "knn": neighbors.KNeighborsClassifier()
    }
    
    trained_models = train_model(X_train, y_train, models)

    # Get metrics, compare and choose
    X_test, y_test = get_sets('test')
    
    print(f'{datetime.now().strftime("%H:%M:%S")}: Getting metrics')
    metrics_df, y_preds = evaluate_models(trained_models, X_test, y_test)    
    plot_bar_chart(metrics_df)
    plot_box_plot(metrics_df)
    # plot_model_metrics(models, X_test, y_test)
    
    print(y_preds)
    
    # Plots for chosen model
    chosen_model = 'random_forest'
    y_pred = y_preds[f'{chosen_model}_y_pred']
    plot_precision_recall_curve(y_test, y_pred)
    plot_roc_curve(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred)    
    # Save model
    print(f'{datetime.now().strftime("%H:%M:%S")}: Saving model')
    best_model = trained_models[chosen_model]
    joblib.dump(best_model, 'model/model.pkl')

    print(f'Finished Train step at {datetime.now().strftime("%H:%M:%S")}')


if __name__ == "__main__":

    main()

