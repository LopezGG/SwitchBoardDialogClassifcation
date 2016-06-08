
# coding: utf-8

# In[24]:

import pickle
import numpy as np


# Load test ,  training data and Vocabulary from previous script i.e. ProcessData.py

# In[11]:

infile = open('E:\CompLing575\ProcessedData\swda\TrainingData.pkl', 'rb')
training = pickle.load(infile)
infile.close()
infile = open('E:\CompLing575\ProcessedData\swda\TestData.pkl', 'rb')
test = pickle.load(infile)
infile.close()
infile = open('E:\CompLing575\ProcessedData\swda\Vocab.pkl', 'rb')
Vocab = pickle.load(infile)
infile.close()
infile = open('E:\CompLing575\ProcessedData\swda\Vocab_pos.pkl', 'rb')
Vocab_pos = pickle.load(infile)
infile.close()


# # Ngrams Generation (i.e Feature Generation)
# - Here the dataset is split into Two
#     - We have a data set with Ngrams alone
#     - Second data set includes the pos Tags of words along with the ngrams
# 

# In[14]:

#this creates features for the data(here we look at only words)
TrainFeatures =[]
for utt in training:
    tag = utt[0]
    unigrams = utt[1][0]
    bigrams = utt[1][1]
    trigrams = utt [1][2]
    Features =[]
    rareFlag = False
    for uni in unigrams:
        if(Vocab.get(uni, 0) >= 2):
            Features.append(uni)
        else:
            rareFlag = True
    for bi in bigrams:
        if(Vocab.get(bi, 0) >= 2):
            Features.append(bi)
    for tri in trigrams:
        if(Vocab.get(tri, 0) >= 2):
            Features.append(tri)
    if (rareFlag):
        Features.append("rareWord")
    #For Pos
    unigrams_pos = utt[1][3]
    bigrams_pos = utt[1][4]
    trigrams_pos = utt [1][5]
    Features_pos =[]
    rareFlag_pos = False
    for uni in unigrams_pos:
        if(Vocab_pos.get(uni, 0) >= 2):
            Features_pos.append(uni)
        else:
            rareFlag_pos = True
    for bi in bigrams_pos:
        if(Vocab_pos.get(bi, 0) >= 2):
            Features_pos.append(bi)
    for tri in trigrams_pos:
        if(Vocab_pos.get(tri, 0) >= 2):
            Features_pos.append(tri)
    if (rareFlag_pos):
        Features_pos.append("rareWord")
    data = [tag,Features,Features_pos]
    TrainFeatures.append(data)


# In[15]:

TestFeatures =[]
for utt in test:
    tag = utt[0]
    unigrams = utt[1][0]
    bigrams = utt[1][1]
    trigrams = utt [1][2]
    Features =[]
    rareFlag = False
    for uni in unigrams:
        if(Vocab.get(uni, 0) >= 2):
            Features.append(uni)
        else:
            rareFlag = True
    for bi in bigrams:
        if(Vocab.get(bi, 0) >= 2):
            Features.append(bi)
    for tri in trigrams:
        if(Vocab.get(tri, 0) >= 2):
            Features.append(tri)
    if (rareFlag):
        Features.append("rareWord")
    #For Pos
    unigrams_pos = utt[1][3]
    bigrams_pos = utt[1][4]
    trigrams_pos = utt [1][5]
    Features_pos =[]
    rareFlag_pos = False
    for uni in unigrams_pos:
        if(Vocab_pos.get(uni, 0) >= 2):
            Features_pos.append(uni)
        else:
            rareFlag_pos = True
    for bi in bigrams_pos:
        if(Vocab_pos.get(bi, 0) >= 2):
            Features_pos.append(bi)
    for tri in trigrams_pos:
        if(Vocab_pos.get(tri, 0) >= 2):
            Features_pos.append(tri)
    if (rareFlag_pos):
        Features_pos.append("rareWord")
    data = [tag,Features,Features_pos]
    TestFeatures.append(data)
        


# Next we create labels and call on all the scikit learn classifiers

# In[16]:

train_labels = list(x[0] for x in TrainFeatures)
train_data  = list(' '.join(x[1]) for x in TrainFeatures)
train_data_pos  = list(' '.join(x[2]) for x in TrainFeatures)
test_labels = list(x[0] for x in TestFeatures)
test_data  = list(' '.join(x[1]) for x in TestFeatures)
test_data_pos  = list(' '.join(x[2]) for x in TestFeatures)


# In[17]:

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


# In[18]:

vectorizer = CountVectorizer(min_df=1)
X_train = vectorizer.fit_transform(train_data)
X_test = vectorizer.transform(test_data)
X_train_pos = vectorizer.fit_transform(train_data_pos)
X_test_pos = vectorizer.transform(test_data_pos)


# In[19]:

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


# In[32]:

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


# In[33]:

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
                                            dual=False, tol=1e-3),X_train, train_labels,X_test,test_labels))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty=penalty),X_train, train_labels,X_test,test_labels))


# In[34]:

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


# In[35]:

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


# In[36]:

results_pos = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
        #(KNeighborsClassifier(n_neighbors=10), "kNN"),
        (RandomForestClassifier(n_estimators=100), "Random forest")):
    print('=' * 80)
    print(name)
    results_pos.append(benchmark(clf,X_train_pos, train_labels,X_test_pos,test_labels))


# In[37]:

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results_pos.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
                                            dual=False, tol=1e-3),X_train_pos, train_labels,X_test_pos,test_labels))

    # Train SGD model
    results_pos.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty=penalty),X_train_pos, train_labels,X_test_pos,test_labels))


# In[38]:

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results_pos.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                       penalty="elasticnet"),X_train_pos, train_labels,X_test_pos,test_labels))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results_pos.append(benchmark(NearestCentroid(),X_train_pos, train_labels,X_test_pos,test_labels))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results_pos.append(benchmark(MultinomialNB(alpha=.01),X_train_pos, train_labels,X_test_pos,test_labels))
results_pos.append(benchmark(BernoulliNB(alpha=.01),X_train_pos, train_labels,X_test_pos,test_labels))


# In[39]:

# make some plots
get_ipython().magic('matplotlib inline')
indices = np.arange(len(results_pos))

results_pos = [[x[i] for x in results_pos] for i in range(4)]

clf_names, score, training_time, test_time = results_pos
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score with POS labels")
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



