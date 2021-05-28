# Implementing Bag-of-Words :

The Bag of Words Model is a very simple way of representing text data for a machine learning algorithm to understand. It has proven to be very effective in NLP problem domains like document classification.

## 1. Understanding the Bag of Words Model :

To understand the bag of words Model, let's first start with the help of an example.

Consider the following text which we wish to represent in the form of vector using BOW model :

* Hello, how are you!
* Win money, win from home.
* Call me now.
* Hello, Call hello you tomorrow?

Now if we have to perform text classification, or any other task, on the above data using statistical techniques, we can not do so since statistical techniques work only with numbers. Therefore we need to convert these sentences into numbers.

Let's create a set of all the words in the given text :

```python
set = {'are', 'call', 'from', 'hello', 'home', 'how', 'me',
       'money', 'now', 'tomorrow', 'win', 'you'
      }
```

We have 12 different words in our text corpus. This will be the length of our vector.

Now we just have to count the frequency of words appearing in each document and the result we get is a *Bag of Words* representation of the sentences.



| **Document** | **are** | **call**  | **from**  | **hello**  | **home**  | **how**  | **me**  | **money**  | **now**  | **tomorrow**  | **win**  | **you**  |
| ------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| Hello, how are you! | 1 |	0 |	0 | 1 | 0 |	1 |	0 |	0 |	0 |	0 |	0 | 1 |
| Win money, win from home. | 0 |	0 |	1 |	0 |	1 |	0 |	0 |	1 |	0 |	0 | 2 |	0 |
| Call me now. | 0 |	1 |	0 |	0 |	0 |	0 |	1 |	0 |	1 |	0 |	0 |	0 |
| Hello, Call hello you tomorrow? | 0 |	1 |	0 |	2 |	0 |	0 |	0 |	0 |	0 |	1 |	0 |	1 |
<p align="center">
    Bag Of Worsd Model
</p>

In the above figure, it is shown that we just keep count of the number of times each word is occurring in a sentence.

## 2. Implementing Bag-of-Words Model With Python :

First, let's import the required libraries :

```Python
#Importing the required modules
import string
import pprint
import pandas as pd
from collections import Counter 
 ```
 
### Data Preprocessing :

```Python
# Sample text corpus
documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']
```

Our text contains punctuations. We don't want punctuations to be the part of our word frequency dictionary. 
In the following code, we first convert our text into lower case and then will remove the punctuation from our text.

 ```Python
# convert into lower case.
lower_case_documents = []
for i in documents:
    lower_case_documents.append(i.lower())
print(lower_case_documents)
```
```Python
['hello, how are you!', 'win money, win from home.', 'call me now.', 'hello, call hello you tomorrow?']
```
```Python
# remove the punctuation.
without_punctuation_documents = []
for i in lower_case_documents:
    without_punctuation_documents.append(''.join(c for c in i if c not in string.punctuation))
    
print(without_punctuation_documents)
```
```Python
['hello how are you', 'win money win from home', 'call me now', 'hello call hello you tomorrow']
```
You can see that the text doesn't contain any special character.

The next step is to tokenize the sentences in our documents and create a frequency_list that contains words and their corresponding frequencies. 

Look at the following code :

```Python
preprocessed_documents = []
for sentence in without_punctuation_documents:
    preprocessed_documents.append(nltk.word_tokenize(sentence))
print(preprocessed_documents)
```
```Python
[['hello', 'how', 'are', 'you'], ['win', 'money', 'win', 'from', 'home'], 
['call', 'me', 'now'], ['hello', 'call', 'hello', 'you', 'tomorrow']]
```
```Python
frequency_list = []
for i in preprocessed_documents:
    frequency_list.append(Counter(i))
    
pprint.pprint(frequency_list)
```
```
[Counter({'hello': 1, 'how': 1, 'are': 1, 'you': 1}),
 Counter({'win': 2, 'money': 1, 'from': 1, 'home': 1}),
 Counter({'call': 1, 'me': 1, 'now': 1}),
 Counter({'hello': 2, 'call': 1, 'you': 1, 'tomorrow': 1})]
```

## 3. Implementing Bag of Words in scikit-learn

Sklearn provide us with an excellent function called CountVectorizer.

```Python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
 
CountVec = CountVectorizer()
#transform
Count_data = CountVec.fit_transform(documents)

#create dataframe
frequency_matrix = pd.DataFrame(Count_data.toarray(),index=documents,
                                columns=CountVec.get_feature_names())
frequency_matrix
```

In the above code the CountVectorizer's fit transform method will create a matrix, the rows in the matrix are the sentence and the columns are the feature words, each cell in the matrix will denote the number of times a word appeared in the sentence. Such as in the below table.


| **Document** | **are** | **call**  | **from**  | **hello**  | **home**  | **how**  | **me**  | **money**  | **now**  | **tomorrow**  | **win**  | **you**  |
| ------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| Hello, how are you! | 1 |	0 |	0 | 1 | 0 |	1 |	0 |	0 |	0 |	0 |	0 | 1 |
| Win money, win from home. | 0 |	0 |	1 |	0 |	1 |	0 |	0 |	1 |	0 |	0 | 2 |	0 |
| Call me now. | 0 |	1 |	0 |	0 |	0 |	0 |	1 |	0 |	1 |	0 |	0 |	0 |
| Hello, Call hello you tomorrow? | 0 |	1 |	0 |	2 |	0 |	0 |	0 |	0 |	0 |	1 |	0 |	1 |

## 4. Conclusion :

In this article, we saw how to implement the Bag of Words approach from scratch in Python and with Sklearn. The theory of the approach has been explained along with the hands-on code to implement the approach.
