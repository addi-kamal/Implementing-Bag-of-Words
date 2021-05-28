#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Implementing Bag of Words from scratch


# In[2]:


import string
import pprint
import pandas as pd
from collections import Counter
import nltk
nltk.download('punkt')


# In[3]:


documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']


# In[4]:


# convert into lower case.
lower_case_documents = []
for i in documents:
    lower_case_documents.append(i.lower())
print(lower_case_documents)


# In[5]:


# remove the punctuation.
without_punctuation_documents = []
for i in lower_case_documents:
    without_punctuation_documents.append(''.join(c for c in i if c not in string.punctuation))
    
print(without_punctuation_documents)


# In[6]:


preprocessed_documents = []
for sentence in without_punctuation_documents:
    preprocessed_documents.append(nltk.word_tokenize(sentence))
print(preprocessed_documents)


# In[7]:


frequency_list = []
for i in preprocessed_documents:
    frequency_list.append(Counter(i))
    
pprint.pprint(frequency_list)


# In[8]:


# Implementing Bag of Words in scikit-learn


# In[9]:


documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']


# In[10]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
 
CountVec = CountVectorizer()
#transform
Count_data = CountVec.fit_transform(documents)


# In[11]:


#create dataframe
frequency_matrix = pd.DataFrame(Count_data.toarray(),index=documents,
                                columns=CountVec.get_feature_names())
frequency_matrix

