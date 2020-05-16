import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np


#  read dataset

paths = {
    'amazon': 'D:\\LEVEL 4\\Semester 2\\NLP\\sentiment labelled sentences\\amazon_cells_labelled.txt',
    'imdb': 'D:\\LEVEL 4\\Semester 2\\NLP\\sentiment labelled sentences\\imdb_labelled.txt',
    'yelp': 'D:\\LEVEL 4\\Semester 2\\NLP\\sentiment labelled sentences\\yelp_labelled.txt'
}


data_fame = []
for source, path in paths.items():
    df = pd.read_csv(path, names=['sentence', 'label'], sep='\t')
    df['source'] = source  # Add another column filled with the source name amazon,  imdb, yelp
    data_fame.append(df)

print('\n\n\nData Frame')
print(data_fame)


df = pd.concat(data_fame)


X, y = [], []
for i in range(len(df)):
    X.append(df.iloc[i][0])
    y.append(df.iloc[i][1])

'''print(X)
print(y)'''

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


vectorizer = CountVectorizer()  # Convert a collection of text documents to a matrix of token counts
vectorizer.fit(x_train)         # to get vocabulary
#  it counts the number of times a token shows up in the document and uses this value as its weight.


x_train = vectorizer.transform(x_train)  # convert the data into bag of words
x_test = vectorizer.transform(x_test)


classifier = LogisticRegression()
classifier.fit(x_train, y_train)
score = classifier.score(x_test, y_test)  # 0 for negative and 1 for positive
print("Accuracy:", score)


# test classifier
for i in range(10):
    print("predicted: ", classifier.predict(x_test[i]))
    print("real", y_test[i], "\n")


# my own sentences
test = ['so good movie', 'bad one', 'nice :)']
test = vectorizer.transform(test)

for i in range(3):
    print("new predicted value : ", classifier.predict(test[i]))



