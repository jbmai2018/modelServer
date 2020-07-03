import nltk
import random
import string
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
import sys

##############Function to find feature dictionary of each document####################
def findFeatures(document,features):
    featureDict = {}
    for f in features:
        featureDict[f] = 0

    for w in document:
        if w in features:
            featureDict[w] += 1

    return featureDict
######################################################################################

def classifyComments(inputData):
    # print(inputData)
    test = inputData
    test_text = []

    #Adding punctuations to stop words
    wnl = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    punctuations = set(string.punctuation)
    for w in punctuations:
        stop_words.add(w)

    #tokenizing, removing stop words and lemmatizing to get the testing data the way we want
    for word in test:
        tokenized = word_tokenize(word)
        lemmatized = []
        for w in tokenized:
            if w.lower() not in stop_words:
                root = wnl.lemmatize(w)
                lemmatized.append(root.lower())

        test_text.append(lemmatized)
    #################################################################

    #Shuffling data for good model
    random.shuffle(test_text)

    classifier_f = open("/home/hercules/aiProject/modelServer/nltkServer/naivebayes.pickle","rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()

    feature_f = open("/home/hercules/aiProject/modelServer/nltkServer/features.pickle","rb")
    features = pickle.load(feature_f)
    feature_f.close()

    #Making testing data as required in the nltk NaiveBayesClassifier i.e list of tuples of
    #(dictionary of feature count of document i,category of document i)
    classes = []
    new_test_text = []
    for word in test_text:
        featureDict = findFeatures(word,features)
        new_test_text.append(featureDict)
        classes.append(classifier.classify(featureDict))

    classes = FreqDist(classes)
    classes = classes.most_common()
    return(dict(classes))
