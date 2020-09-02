# -*- coding: utf-8 -*-
"""
@author: Allen Annom
"""
import pandas as pd
from time import time
import numpy as np
import itertools
import statistics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


#import CSV Dataset
df = pd.read_csv('cmc.csv')
print(df['CM'].value_counts())
print(df.head)
print(df.describe())
#Data visualisation and preprocessing
#bar diagrams to understand the data

ax = df['CM'].value_counts().plot(kind = 'bar', title = "Contraceptive Method Used")
ax.set_xlabel("Contraceptive Type Used")
ax.set_ylabel("Instances")
ax.xaxis.set(ticklabels=["No Use","Short Term","Long Term"])
plt.show()

ax = df['SoL'].value_counts().plot(kind = 'bar', title = "Standard of Living")
ax.set_xlabel("Levels of Living")
ax.set_ylabel("Instances")
ax.xaxis.set(ticklabels=["Low","Meduim","High","Very High"])
plt.show()

ax = df['WR'].value_counts().plot(kind = 'bar', title = "Wife's Religion")
ax.set_xlabel("Religion")
ax.set_ylabel("Instances")
ax.xaxis.set(ticklabels=["Non Islam","Islam"])
plt.show()



#Algorithm method number 1
#split into x and y
X, y = df.iloc[:, :9].values, df.iloc[:, 9].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)
# defining an algorithm based on which to build a classification model
clf = RandomForestClassifier(n_estimators=20)
#cross validation mean score, results are printed on line 103
scores = cross_val_score(clf, X, y, cv=10)
mean_forest_score = scores.mean()


#define parameters for GridSearch CV
param_grid = {"max_depth": [6, None],
              "n_estimators":[16,64],
              "max_features": ["auto"],
              "min_samples_split": [2, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
#GridSearch
#Algorithm method number 2 - Grid Search CV implementation 
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=10)
#Starting time it takes
start = time()

#Fits data into GridSearch to find the best score and saves it in a variable to be printed on line 104
grid_search.fit(X_train, y_train)
y_pred=grid_search.predict(X_test)
grid_search_results = grid_search.cv_results_['mean_test_score']
means = grid_search_results.mean()

CV_most = max(scores)
gridsearch_most = max(grid_search_results)

cv_deviation = statistics.stdev(scores)
gridsearch_deviation = statistics.stdev(grid_search_results)

#printing Grid Search results 
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

#time function stopped once the GridSearch results have been printed
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))

report(grid_search.cv_results_)





#printing gridsearch mean score in comparison to cross validation method
print("\nMean Cross Validation Accuracy Score: %0.2f" % (scores.mean()))
print("Mean Grid Search Accuracy Score: %0.2f" % means)

print("\nHighest Cross Validation Accuracy Score: %0.2f" % CV_most)
print("Highest Grid Search Accuracy Score: %0.2f" % gridsearch_most)

print("\nStandard Deviation for Cross Validation : %0.4f " % cv_deviation)
print("Standard Deviation for Grid Search : %0.4f" % gridsearch_deviation)

#Graph of Cross Validation method and Grid Search method scores
objects = ('Cross Validation', 'Grid Search')
y_pos = np.arange(len(objects))
methods_to_plot = [mean_forest_score,means]

plt.bar(y_pos, methods_to_plot, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Scores')
plt.title('Best Algorithm Method for Random Forest Algorithm')
 
plt.show()

#Graph of Cross Validation method and Grid Search method scores
objects = ('Cross Validation', 'Grid Search')
y_pos = np.arange(len(objects))
methods_to_plot = [CV_most,gridsearch_most]

plt.bar(y_pos, methods_to_plot, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Scores')
plt.title('Highest score of Cross-Validation compared to GridSearch')
 
plt.show()

#Graph of Cross Validation method and Grid Search method scores
objects = ('Cross Validation', 'Grid Search')
y_pos = np.arange(len(objects))
methods_to_plot = [cv_deviation,gridsearch_deviation]

plt.bar(y_pos, methods_to_plot, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Scores')
plt.title('Standard deviation of Grid Search and Cross-Validation')
plt.show()


#class name for confusion matrix
class_names = ['No Use', 'Long Term',"Short Term"]


#function for confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

