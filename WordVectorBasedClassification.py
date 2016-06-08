
# coding: utf-8

# This takes a file which contains a word and cluster number. The cluster number is based on word embedding. We read the input sentence and create features with cluster id.

# In[18]:

import pickle
import numpy as np


# # Load training and test data from Pickle file

# In[7]:

#load word_centroid_map
pkl_file = open("E:\CompLing575\ProcessedData\data\word2vec.common.crawl.42B.300d.cluster_38349.pkl", 'rb')
word_centroid_map = pickle.load(pkl_file)
pkl_file.close()
#load the switchboard data
TrainFile = open('E:\CompLing575\ProcessedData\swda\TrainingData.pkl', 'rb')
TrainingData = pickle.load(TrainFile)
TrainFile.close()
#load the switchboard data
TestFile = open('E:\CompLing575\ProcessedData\swda\TestData.pkl', 'rb')
TestData = pickle.load(TestFile)
TestFile.close()


# # Create Features 
# based on word vector cluster id

# In[9]:

def Create_Features (Data):
    FeaturesList =[]
    for utt in Data:
        tag = utt[0]
        unigrams = utt[1][0]
        Feature = []
        for word in unigrams:
            if(word in word_centroid_map ):
                Feature.append(str(word_centroid_map[word]))
            else:
                Feature.append(word)
        data = [tag,Feature,unigrams]
        FeaturesList.append(data)
    return FeaturesList


# In[10]:

TrainFeatures = Create_Features(TrainingData)
TestFeatures = Create_Features(TestData)
train_labels = list(x[0] for x in TrainFeatures)
train_data  = list(' '.join(x[1]) for x in TrainFeatures)
test_labels = list(x[0] for x in TestFeatures)
test_data  = list(' '.join(x[1]) for x in TestFeatures)


# # Classification with scikit learn

# In[11]:

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from time import time
from sklearn.utils.extmath import density
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


# In[12]:

vectorizer = CountVectorizer(min_df=1)
X_train = vectorizer.fit_transform(train_data)
X_test = vectorizer.transform(test_data)


# In[13]:

# Benchmark classifiers
def benchmark(clf,X_train,y_train,X_test,y_test):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))
        print()

    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


# In[21]:

results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
        #(KNeighborsClassifier(n_neighbors=10), "kNN"),
        (RandomForestClassifier(n_estimators=100), "Random forest")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf,X_train, train_labels,X_test,test_labels))


# In[22]:

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
                                            dual=False, tol=1e-3),X_train, train_labels,X_test,test_labels))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty=penalty),X_train, train_labels,X_test,test_labels))


# In[23]:

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                       penalty="elasticnet"),X_train, train_labels,X_test,test_labels))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid(),X_train, train_labels,X_test,test_labels))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01),X_train, train_labels,X_test,test_labels))
results.append(benchmark(BernoulliNB(alpha=.01),X_train, train_labels,X_test,test_labels))


# In[24]:

# make some plots
get_ipython().magic('matplotlib inline')
indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='r')
plt.barh(indices + .3, training_time, .2, label="training time", color='g')
plt.barh(indices + .6, test_time, .2, label="test time", color='b')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()


# In[ ]:




# In[ ]:



