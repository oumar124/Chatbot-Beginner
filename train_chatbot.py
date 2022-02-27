#!/usr/bin/env python
# coding: utf-8

# In[9]:


import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random


# In[10]:


lemmatizer = WordNetLemmatizer()


# In[12]:


words = []
classes = []
documents = []
ignore_words=['?','!']
data_file = open('Intents.json').read()
intents=json.loads(data_file)


# In[18]:


#Preprocessing the Data


# In[22]:


nltk.download('wordnet')


# In[20]:


for intent in intents['intents']:
    for pattern in intent['patterns']:
        
        #tokenizing each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        
        #adding documents in the corpus
        documents.append((w,intent['tag']))
        
        #adding to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


# In[23]:


# lemmatize, lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

#sort classes
classes = sorted(list(set(classes)))

#documents = combination between patterns and intents
print(len(documents),"documents")

#classes = intents
print(len(classes),"classes",classes)

#words = all words, vocabulary
print(len(words),"unique lemmatized words",words)

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))


# In[24]:


#Creating training and testing data


# In[30]:


#Creating the training data
training = []

#Empty array for the output
output_empty = [0] * len(classes)

#training set, bag of words for each sentence
for doc in documents:
    
    #initializing the bag of words
    bag=[]
    
    #list of tokenized words for the pattern
    pattern_words = doc[0]
    
    #lemmatizing each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    #creating a bog of words array with 1, if word match found in current patter
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
        
    #Ouptut is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])
    
#shuffling the features and turn it into an array
random.shuffle(training)
training = np.array(training)

#creating train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training Data Created")


# In[33]:


#Creating model with 3 layers
#First layer : 128 neurons
#Second layer : 64 neurons
#Thirds layer : contains number of neurons equal to number of intents to predict output intent with softmax

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))

#compiling the model
sgd= SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model
hist = model.fit(np.array(train_x),np.array(train_y),epochs=200,batch_size=5, verbose=1)
model.save('chatbott_model.h5',hist)

print("model created")


# In[ ]:




