#######################################
# PIPELINE
#######################################

from helpers.modelconfig import *
from helpers.processing import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

def get_models_and_params(models_to_run):
    models = [CLFS[model] for model in models_to_run]
    params = [PARAMS[model] for model in models_to_run]

    return models, params


def gridsearch_report(results, n_top=3):
    for clf in results:
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(clf['rank_test_score'] == i)
            for candidate in candidates:
                print(str(clf) + "Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                      clf['mean_test_score'][candidate],
                      clf['std_test_score'][candidate]))
                print("Parameters: {0}".format(clf['params'][candidate]))
                print("")


def save_clf_report(report):
    data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        data.append(row)
    df = pd.DataFrame.from_dict(data)
    df.to_csv('clf_report.csv', index = False)


def plot_precision_recall_population(clf, y_test, y_pred):
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_pred)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_pred)
    for value in pr_thresholds:
        num_above_thresh = len(y_pred[y_pred>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)

    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])

    plt.title(clf)
    #plt.savefig(name)
    plt.show()

def plot_precision_recall(clf, y_test, y_pred):
    color = 'blue'
    precision = dict()
    recall = dict()
    average_precision = dict()
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
        y_pred.ravel())
    average_precision["micro"] = average_precision_score(y_test, y_pred,
                                                         average="micro")

    plt.clf()
    plt.plot(recall["micro"], precision["micro"], color=color,
             label='micro-average Precision-recall curve (area = {0:0.2f})'
                   ''.format(average_precision["micro"]))
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.tight_layout()
    plt.title("Best Precision Recall")
    plt.show()

def p_r_pdf(clf, precision, recall):
    colors = cycle(['navy', 'darkorange', 'teal', 'red'])

    with PdfPages('precision_recall_'+ str(clf) +'.pdf') as pdf:
        fig, subplot = plt.subplots(nrows=2, ncols=1)
        fig.set_size_inches(8, 10)
        axes = subplot.flatten()

        # Plot Precision-Recall curve for each class
        plt.clf()
        plt.plot(recall["micro"], precision["micro"], color='blue', lw=lw,
                 label='micro-average Precision-recall curve (area = {0:0.2f})'
                       ''.format(average_precision["micro"]))

        for i, color in zip(range(N_CLASSES), colors):
            plt.plot(recall[i], precision[i], color=color, lw=lw,
                     label='Precision-recall curve of class {0} (area = {1:0.2f})'
                           ''.format(i, average_precision[i]))
            plt.xlim([0.0, 1.05])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
        # plt.tight_layout()
        pdf.savefig()
        plt.close()


def run_search(X, y, models_to_run):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)

    models, parameters = get_models_and_params(models_to_run)

    model_results = []
    r_columns = ['clf', 'target', 'accuracy', 'precision', 'recall', 'time', 'params']
    for score in SCORES:
        for i, model in enumerate(models):
            print()
            print("1.) Tuning hyperparameters for",'\n', model, "on", score)
            print()
            print(parameters[i])
            # returns the best estimator and fits it
            clf = GridSearchCV(model, parameters[i], scoring=score, n_jobs=8, pre_dispatch='2*n_jobs')
            clf.fit(X_train, y_train)

            print("2.) Classification report")
            print()
            y_test, y_pred = y_test, clf.predict(X_test)
            report = classification_report(y_test, y_pred)
            print(report)

            # precision_recall
            print("3.) Precision Recall")
            print()
            g = plot_precision_recall_population(clf, y_test, y_pred)

            # Save metrics to results
            a = accuracy_score(y_test, y_pred)
            p  = precision_score(y_test, y_pred)
            roc = roc_auc_score(y_test, y_pred)
            mean = max(clf.cv_results_['mean_fit_time'])
            std = max(clf.cv_results_['std_fit_time'])
            model_results.append([model, score, a, p, roc, mean, clf.get_params()])

            # if you want a full report of a model
            # gridsearch_report(clf.cv_results_)

    df = pd.DataFrame(model_results, columns=r_columns)
    df.to_csv("gridsearch_results_"+ str(datetime.now().date())+".csv")

    return df




