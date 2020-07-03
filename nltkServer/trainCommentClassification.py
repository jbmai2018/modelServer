import nltk
import random
import string
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist

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

#Adding punctuations to stop words
wnl = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
punctuations = set(string.punctuation)
for w in punctuations:
    stop_words.add(w)

#training data
given = [
("plant closed",'Plant Issue'),
("lunch time",'Lunch'),
("shift change",'Shift Change'),
("dinner time",'Dinner'),
("plant close",'Plant Issue'),
("tea time",'Tea Time Extension'),
("dinner time",'Dinner'),
("lunch time changed",'Lunch'),
("plant closed",'Plant Issue'),
("tea time",'Tea Time Extension'),
("lunch time",'Lunch'),
("tea time",'Tea Time Extension'),
("We faced some issue in tea break.",'Tea Time Extension'),
("plant closed",'Plant Issue'),
("DISCUSSION ABOUT PARTS PROBLEM AMOUNG TEAM",'Meeting Called'),
("MASKING PROBLEM",'Masking Issue'),
("PLAN NOT UPDATE FOR 28.05.2019",'Others'),
("plant closed",'Plant Issue'),
("dinner time",'Dinner'),
("Machine stopped",'machine'),
("plant closed",'Plant Issue'),
("plant closed",'Plant Issue'),
("lunch time",'Lunch'),
("dinner time",'Dinner'),
("tea time",'Tea Time Extension'),
("plant closed",'Plant Issue'),
("lunch time",'Lunch'),
("plant closed",'Plant Issue'),
("tea time",'Tea Time Extension'),
("Machine not working",'Machine Issue'),
("dinner time",'Dinner'),
("plant closed",'Plant Issue'),
("dinner time",'Dinner'),
("PLAN NOT UPDATE FOR 28.05.2019",'Others'),
("manpower shifted to another line.",'Others'),
("tool problem",'Machine Issue'),
("plant closed",'Plant Issue'),
("dinner time",'Dinner'),
("plant closed",'Plant Issue'),
("tea time",'Tea Time Extension'),
("plant closed",'Plant Issue'),
("lunch time",'Lunch'),
("tea time",'Tea Time Extension'),
("dinner time",'Dinner'),
("plant closed",'Plant Issue'),
("lunch time",'Lunch'),
("dinner time",'Dinner'),
("plant closed",'Plant Issue'),
("lunch time",'Lunch'),
("dinner time",'Dinner'),
("plant closed",'Plant Issue'),
("shift change",'Shift Change'),
("PLAN NOT UPDATE FOR 28.05.2019",'Planned Manpower Problem')]

train_text = []
vocabulary = []

#tokenizing, removing stop words and lemmatizing to get the training data the way we want
for word,category in given:
    tokenized = word_tokenize(word)
    lemmatized = []
    for w in tokenized:
        if w.lower() not in stop_words:
            root = wnl.lemmatize(w)
            lemmatized.append(root.lower())
            vocabulary.append(root.lower())

    train_text.append((lemmatized,category))
#################################################################

#Shuffling data for good model
random.shuffle(train_text)

#Finding features which are top frequency words
vocab = FreqDist(vocabulary)
vocab = vocab.most_common(8)
features = []
for w,i in vocab:
    features.append(w)

save_feature = open("features.pickle","wb")
pickle.dump(features,save_feature)
save_feature.close()

print("\nFeatures used:\n",features)

#Making training data as required in the nltk NaiveBayesClassifier i.e list of tuples of
#(dictionary of feature count of document i,category of document i)
new_train_text = []
for word,category in train_text:
    featureDict = findFeatures(word,features)
    new_train_text.append((featureDict,category))

#Training data with NB classifier
classifier = nltk.NaiveBayesClassifier.train(new_train_text)
classifier.show_most_informative_features()

save_classifier = open("naivebayes.pickle","wb")
pickle.dump(classifier,save_classifier)
save_classifier.close()
