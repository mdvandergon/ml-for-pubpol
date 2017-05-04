'''#######################################
CONFIG FILE FOR PIPELINE
Models supported:
    SVC
    DecisionTreeClassifier
    KNeighborsClassifier
    GaussianNB, MultinomialNB, BernoulliNB
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
    LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
    NearestCentroid

'''#######################################

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid

TEST_SIZE = 0.4
N_CLASSES = 1
SCORES = ['accuracy', 'precision']

CLFS = {
    'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
    'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
    'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
    'LR': LogisticRegression(penalty='l1', C=1e5),
    'SVM': SVC(kernel='linear', probability=True, random_state=0, cache_size=7000),
    'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
    'NB': GaussianNB(),
    'DT': DecisionTreeClassifier(),
    'SGD': SGDClassifier(loss="hinge", penalty="l2"),
    'KNN': KNeighborsClassifier(n_neighbors=3)
    }

PARAMS = {
    'RF':{'n_estimators': [1,10], 'max_depth': [1,5,10,20], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'ET': { 'n_estimators': [1,10], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.0001,0.01,0.1,1]},
    'SVM' :{'C' :[0.0001,0.01,0.1,1],'kernel':['linear']},
    'GB': {'n_estimators': [1,10], 'learning_rate' : [0.001,0.01,0.05,0.1],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'KNN' :{'n_neighbors': [1,5,10,25,50],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
    }

TEST_PARAMS = {
    'RF':{'n_estimators': [1,10], 'max_depth': [1,5], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5]},
    'ET': { 'n_estimators': [1,10], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001]},
    'SVM' :{'C' :[0.00001,0.0001],'kernel':['linear']},
    'GB': {'n_estimators': [1,10], 'learning_rate' : [0.001,0.01],'subsample' : [0.1,0.5], 'max_depth': [1,3]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5]},
    'SGD': { 'loss': ['hinge'], 'penalty': ['l2']},
    'KNN' :{'n_neighbors': [1,5],'weights': ['uniform'],'algorithm': ['auto']}
    }
