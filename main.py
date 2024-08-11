import json
import string
import random

import numpy as np
import pandas as pd
import nltk

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Dropout

nltk.download('wordnet')
nltk.download('punkt')
lem = nltk.WordNetLemmatizer()

# Reading JSON file
with open("intents.json") as file:
    f=json.load(file)

words=[]
classes=[]
x_label=[]
y_label=[]

# collecting all words, classes (tags), input variable x -> Patterns, Output Variable y -> Responses

for intent in f["intents"]:
    for pattern in intent["patterns"]:
        tokens=nltk.word_tokenize(pattern)
        words.extend(tokens)
        x_label.append(pattern)
        y_label.append(intent["tag"])
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

# Lemmatizing every word
words=[lem.lemmatize(word.lower()) for word in words if word not in string.punctuation]
words=sorted(set(words))
classes=sorted(set(classes))

# TRAINING MODEL
training=[]
out_empty=[0]*len(classes)


for idx, doc in enumerate(x_label):
    BOW =[]           # creating a bag of words
    text=lem.lemmatize(doc.lower())
    for word in words:
        BOW.append(1) if word in text else BOW.append(0)
    output_row=list(out_empty)
    output_row[classes.index(y_label[idx])]=1

    training.append([BOW , output_row])

random.shuffle(training)

training=np.array(training,dtype=object)

train_X=np.array(list(training[:,0]))
train_y=np.array(list(training[:,1]))

input_shape=(len(train_X[0]),)
output_shape=len(train_y[0])
epochs=500

# Adding hidden layers to model

model=Sequential()
model.add(Dense(128, input_shape=input_shape,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(output_shape,activation='softmax'))
adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.01, decay=1e-6)

model.compile(loss='categorical_crossentropy',
             optimizer="adam",
             metrics="accuracy")

print(model.summary())

# Making model fit with algorithm
model.fit(x=train_X, y=train_y, epochs=500, verbose=1)

# Cleaning User Entered Text
def clean_text(text):
    tokens=nltk.word_tokenize(text)
    tokens=[lem.lemmatize(word) for word in tokens]
    return tokens

# Making word bag of User Input
def bag_of_words(text,vocab):
    tokens=clean_text(text)
    BOW =[0]*len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word==w:
                BOW [idx]=1
    return np.array(BOW )

# Predicting the class of input
def pred_class(text, vocab,labels):
    BOW =bag_of_words(text, vocab)
    result=model.predict(np.array([BOW]))[0]
    thresh=0.2
    y_pred=[[idx,res] for idx, res in enumerate(result) if res>thresh]

    y_pred.sort(key=lambda x:x[1], reverse=True)
    return_list=[]
    for r in y_pred:
        return_list.append(labels[r[0]])
    return return_list

# According to predicted class, choosing a random response
def get_response(intents_list, intents_json):
    tag=intents_list[0]
    list_of_intents=intents_json["intents"]
    for i in list_of_intents:
        if i["tag"]==tag:
            result=random.choice(i["responses"])
            break
        else:

            result = random.choice(f["default greetings"])
    return result



# Running the trained ChatBot
while True:
    message=input("User: ")
    intents=pred_class(message, words, classes)
    result=get_response(intents,f)
    print("Bot: "+result)
    if result in [
          "Goodbye! Feel free to come back anytime.",
          "See you later! Have a wonderful day.",
          "Talk to you soon! Take care.",
          "It was nice chatting with you too. Have a great day!",\
        "It was nice speaking to you","See you later","Speak Soon"]:
      break;