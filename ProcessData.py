
# coding: utf-8

# # Obtain Data and Save
# Data was obtained from http://compprag.christopherpotts.net/swda.html which is the switchboard data. It is processed and has pos tags. 

# In[6]:

import os
import numpy as np
from builtins  import any as b_any
from nltk.util import ngrams
import math
import pickle
import random 
PATH = "E:\CompLing575\ProcessedData\swda\swda"
FileList = [os.path.join(dp, f) for dp, dn, filenames in os.walk(PATH) for f in filenames if os.path.splitext(f)[1] == '.csv']


# Data  comes with a script to load it. I ported the script imported in the next line to python 3.4. 
# I am loading the data using the given function Transcript and I also create a set of n grams to go with it. 
# I will save it to a pickle file which can be loaded later on.

# In[7]:



from swda import Transcript
Transcript_list = []

for f in FileList:
    trans = Transcript(f, "E:\CompLing575\ProcessedData\swda\swda\swda-metadata")
    utts=trans.utterances
    
    for utt in utts:
        pos_text = utt.pos.replace("["," ")
        pos_text = pos_text.replace("]", " ")
        pos_list = list(filter(None,pos_text.split(' ')))
        pos_words = list(word[0:word.index('/')] for word in pos_list)
        bigrams=list(ngrams(pos_words,2))
        trigrams = list (ngrams(pos_words,3))
        pos_bigrams=list(ngrams(pos_list,2))
        pos_trigrams = list (ngrams(pos_list,3))

        data = [utt.act_tag,pos_words,bigrams,trigrams,pos_list,pos_bigrams,pos_trigrams ]
        Transcript_list.append(data)

outputFile = open('E:\CompLing575\ProcessedData\swda\Transcript_list.pkl', 'wb')
pickle.dump(Transcript_list, outputFile)
outputFile.close()


# In[8]:

#Separate into 2 classes and create test and training set. Also create vocab from training set
StatementTag = ['sd','sv']
Statement = []
Rest = []
st_count = 0
r_count = 0
for utt in Transcript_list:
    if b_any(i in utt[0] for i in StatementTag):
        tag = "1"
        data = [st_count,tag,utt[1:]]
        st_count +=1
        Statement.append(data)
    else:
        tag = "0"
        data = [r_count,tag,utt[1:]]
        r_count+=1
        Rest.append(data)

StateTestId = random.sample(range(0,len(Statement)),  int(math.floor(len(Statement)*0.3)))
RestTestId = random.sample(range(0,len(Rest)),  int(math.floor(len(Rest)*0.3)))


# In[9]:

#generate training set with vocab and create test set
Vocab = dict()
Vocab_pos = dict()
training = []
test = []
for st in Statement:
    if (st[0] not in StateTestId):
        training.append(st[1:])
        for word in st[2][0]:
            Vocab[word] = Vocab.get(word, 0) + 1
        for bi in st[2][1]:
            word = '_'.join(bi)
            Vocab[word] = Vocab.get(word, 0) + 1
        for tri in st[2][2]:
            word = '_'.join(tri)
            Vocab[word] = Vocab.get(word, 0) + 1
        for word in st[2][3]:
            Vocab_pos[word] = Vocab_pos.get(word, 0) + 1
        for bi in st[2][4]:
            word = '_'.join(bi)
            Vocab_pos[word] = Vocab_pos.get(word, 0) + 1
        for tri in st[2][5]:
            word = '_'.join(tri)
            Vocab_pos[word] = Vocab_pos.get(word, 0) + 1
    else:
        test.append(st[1:])
for rt in Rest:
    if (rt[0] not in RestTestId):
        training.append(rt[1:])
        for word in rt[2][0]:
            Vocab[word] = Vocab.get(word, 0) + 1
        for bi in rt[2][1]:
            word = '_'.join(bi)
            Vocab[word] = Vocab.get(word, 0) + 1
        for tri in rt[2][2]:
            word = '_'.join(tri)
            Vocab[word] = Vocab.get(word, 0) + 1
        for word in rt[2][3]:
            Vocab_pos[word] = Vocab_pos.get(word, 0) + 1
        for bi in rt[2][4]:
            word = '_'.join(bi)
            Vocab_pos[word] = Vocab_pos.get(word, 0) + 1
        for tri in rt[2][5]:
            word = '_'.join(tri)
            Vocab_pos[word] = Vocab_pos.get(word, 0) + 1
    else:
        test.append(rt[1:])       


# In[10]:

print ("DATASET STATISTICS: ")
print("RestTestid Count:"+str(len(RestTestId)))
print("Rest data Count:"+str(len(Rest)))
print("StatementTestid Count:"+str(len(StateTestId)))
print("Statement data Count:"+str(len(Statement)))
print("Total utterance Count:"+str(len(Statement)+len(Rest)))
print("Percentage statement Count:"+str((len(Statement)*100)/ (len(Statement)+len(Rest))))


# We are going to save all the files for future processing

# In[12]:

outputFile = open('E:\CompLing575\ProcessedData\swda\TrainingData.pkl', 'wb')
pickle.dump(training, outputFile)
outputFile.close()
outputFile = open('E:\CompLing575\ProcessedData\swda\TestData.pkl', 'wb')
pickle.dump(test, outputFile)
outputFile.close()
outputFile = open('E:\CompLing575\ProcessedData\swda\Vocab.pkl', 'wb')
pickle.dump(Vocab, outputFile)
outputFile.close()
outputFile = open('E:\CompLing575\ProcessedData\swda\Vocab_pos.pkl', 'wb')
pickle.dump(Vocab_pos, outputFile)
outputFile.close()


# In[ ]:



