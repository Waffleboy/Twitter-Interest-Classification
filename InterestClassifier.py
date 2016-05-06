# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 19:29:13 2015

Twitter Account Classifier

This program takes in a twitterhandle, trains a classifier based on given training
data, and then classifies the given twitterhandle into one of the 10 preset
categories. 

This program hosts 2 types of classifiers: Multinomial Naive Bayes and Linear SVM.
There are various settings for the preprocessing and classifiers which are outlined
in their instructions.

TO USE:
1)Modify the SETTINGS FOR CLASSIFIER section appropriately.

[Application can run with default settings once 1) is done. For customization, read on.]

2)Scroll to bottom, if you want to use naive bayes, uncomment naive bayes classifier command.
likewise for SVM. Important: Only uncomment one at a time!

3)Additional settings (ngram,stopwords,tf or tfidf) can be changed by modifying the input
to the preProcess function. If no additional arguments given, preProcess runs with default 
settings. Look at function for more details.

4)Additional settings for the classifiers can be modified by changing the inputs to the
classifier functions (optional arguments). If these are not given, the classifier runs with
default settings. Look at the classifiers for the optional arguments available.

5)**optional, there is a function testClassifier that can test the accuracy of the classifier if
you so wish. It prints out the accuracy % of the classifier, and returns the confusion
matrix in a nice graphical format.


@author: Thiru
"""
import csv,tweepy,random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB #Naive Bayes
from sklearn.linear_model import SGDClassifier #SVM
from sklearn import metrics
from collections import Counter
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize          
#######


"""
SETTINGS FOR CLASSIFIER: Modify accordingly!
"""
##Twitter login details of format ClientID/Client Secret/ Access token / Access Secret
accesstokenlist=[]
accesstokenlist.append(['<Insert client id>','<Include Client secret>','<Include access token>','<Include access secret>'])
#Interest Names/Categories as per CSV File!
names = ['News','Sports','Games','Religious','Celebrity','Food','Music','Finance','Political','Technology']
##Directory for CSV's. ASSUME THAT SAME DIRECTORY.
trainingcsv='BTAssignmentTraining.csv'
#If you want to test classifier accuracy
testcsv='BTAssignmentValidation.csv'
"""end settings """


"""
Variables to store tweets,interest, and the tokeniser and TFIDF functions, DO-NOT-TOUCH.
"""
textlst=[]
interestlst=[]
vectorizer=None
tfidfTransformer=None
""" end variables"""

"""
Description: Function to openCSV. Takes in a csv, outputs a list.
Sample command: openCSV('test.csv')
"""
def openCSV(name):
    with open(name,'r') as f:
        reader = csv.reader(f)
        lst=list(reader)
    return lst



"""
preProcess
Description: This function takes in a CSV of form [Tweets,tag],

OPTINAL ARGUMENTS: 
ngram_range = (1,1) for uni, (2,2) for bigram etc
stop_words = 'english' or None.
tfidf = True for tfidf, False for TF

It then converts the csv and splits into one list each for tweets and interest tag.
Then, it tokenizes the tweetslst, uses TF/TFIDF to create the term document matrix,
and returns the matrix.


Post Cond: returns term document matrix
Sample commands: preProcess(test.csv), preProcess(test.csv,True,ngram_range=(1,2),stop_words='None')
"""
def preProcess(csvname,tfidf=True,ngram_range=(1,1),stop_words='english',stem=False):
    print('Beginning preprocessing with TFIDF = '+str(tfidf) + 
    ', ngrams selected as '+ str(ngram_range) + ' and stop words selected as '+str(stop_words) +
    ', stemming set to '+str(stem))
    
    #Subsidary function to tokenize words according to settings
    def tokenizer(lst,ngram_range=(1,1),stop_words='english',stem=False):
        global vectorizer
        if stem == False:
            count_vect = CountVectorizer(ngram_range=ngram_range,stop_words=stop_words)
            vectorizer=count_vect
            X_train_counts = count_vect.fit_transform(textlst)
            return X_train_counts
        else:
            stemmer = SnowballStemmer()
            def stem_tokens(tokens, stemmer):
                stemmed = []
                for item in tokens:
                    stemmed.append(stemmer.stem(item))
                return stemmed
            
            def tokenizeSnowball(text):
                tokens = word_tokenize(text)
                stems = stem_tokens(tokens, stemmer)
                return stems
            vectorizer = Count_Vectorizer(tokenizer = tokenizeSnowball,
                                          ngram_range=ngram_range,stop_words=stop_words)
            X_train_counts = vectorizer.fit_transform(textlst)
            return X_train_counts
        
    #Subsidary function to convert tokens to term document matrix
    def TFIDF(tokens,tfidf):
        global tfidfTransformer
        tf_transformer = TfidfTransformer(use_idf=True).fit(tokens)
        tfidfTransformer=tf_transformer
        X_train_tf = tf_transformer.transform(tokens)
        return X_train_tf
    
    #open CSV, split the columns into one list for tweets, one list for interest
    data=openCSV(csvname)
    for i in range(1,len(data)):
        textlst.append(data[i][1])
        interestlst.append(data[i][2])

    tokens=tokenizer(textlst,ngram_range,stop_words)
    tfidfdoc=TFIDF(tokens,tfidf)
    print('Preprocessing done!')
    return tfidfdoc
        

""" Mutinomial Naive Bayes classifier
Options: Alpha = x, where x >=0"""
def makeClassifierBayes(tfidf,result,alpha=1.0):
    clf = MultinomialNB(alpha=alpha).fit(tfidf, result)
    return clf
""" Descripton: SVM Classifier. Takes in tfidf, and optional arguments are:
loss, penalty,alpha,n_iter,random_state. for more details, see scikit learn library"""
def makeClassifierSVM(tfidf,result,loss='hinge', penalty='l2',alpha=1e-3, n_iter=9, random_state=42):
    clf = SGDClassifier(loss=loss, penalty=penalty,alpha=alpha, n_iter=n_iter, random_state=random_state)
    clf=clf.fit(tfidf,result)
    return clf

def predictTweetClassifier(classifier,tweettext):
    newcounts = vectorizer.transform(tweettext)
    newtfidf=tfidfTransformer.transform(newcounts)
    predicted = classifier.predict(newtfidf)
    predictions=[]
    for i in range(len(tweettext)):
        predictions.append([tweettext[i],predicted[i]])
    return predictions
    
def testClassifier(classifier,testingcsv,sample_size):
    data=openCSV(testingcsv)
    data=data[1:] #remove headers
    lst=[]
    for i in range(sample_size):
        lst.append(data[random.randint(0,sample_size)])
    
    newtext=[]
    newinterestlst=[]
    for i in range(len(lst)):
        newtext.append(lst[i][1])
        newinterestlst.append(lst[i][2])
    
    newcounts = vectorizer.transform(newtext)
    newtfidf=tfidfTransformer.transform(newcounts)
    
    predicted = classifier.predict(newtfidf)
    print('Accuracy of classifications = '+ str((np.mean(predicted == newinterestlst))*100)+'%')
    print('Classification report:')
    print(metrics.classification_report(newinterestlst,predicted,target_names=names))
    return plot_confusion_matrix(metrics.confusion_matrix(newinterestlst, predicted))

#Sets the settings for confusion matrix. Does not need modification
def confusion_matrix_settings(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#Plots the confusion matrix. Does not need modification
def plot_confusion_matrix(cm):
    np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization:')
    plt.figure()
    confusion_matrix_settings(cm)
    plt.show()
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    confusion_matrix_settings(cm_normalized, title='Normalized confusion matrix')
    plt.show()

"""
Description: This function mines the tweets of the specified user,
then uses the above functions to classify the user and determine their primary
interest.
"""
def classifyUser(user):
    print('Processing user: '+user)
    global accesstokenlist
    
    currKeyID=0
    currentKey=accesstokenlist[currKeyID]
    auth = tweepy.auth.OAuthHandler(currentKey[0], currentKey[1])
    auth.set_access_token(currentKey[2], currentKey[3])
    api = tweepy.API(auth)
    #Remove links and formatting of the tweets mined.
    def removeLinksandFormatting(lst):
        for i in range(0,len(lst)):
            text = lst[i]
            x = text.find('http')
            while x != -1:
                text = lst[i][:x] + lst[i][x+22:]
                lst[i] = text
                x = text.find('http')
            lst[i]=lst[i][:-1]
            lst[i]=lst[i][2:]
        return lst
        
    tweetlst = []
    counter=4
    print('Mining tweets..')
    new_tweets = api.user_timeline(screen_name = user,count=200)
    tweetlst.extend(new_tweets)
    oldest = tweetlst[-1].id - 1
    ##Mine 1k tweets
    while len(new_tweets) > 0 and counter>0:
        counter-=1
        new_tweets = api.user_timeline(screen_name = user,count=200,max_id=oldest)
        tweetlst.extend(new_tweets)
        oldest = tweetlst[-1].id - 1
    
    tweetstxt=[]
    ##process tweets to list
    for i in range(len(tweetlst)):
        tweetstxt.append(str(tweetlst[i].text.encode('utf-8')))
    print('Formatting tweets..')
    tweetstxt=removeLinksandFormatting(tweetstxt)
    #Predict the interest of each tweet
    taggedtweets=predictTweetClassifier(classifier,tweetstxt)
    interests=[]
    #find the most occuring interest in the tweets. this is the interest of user
    for i in range(len(taggedtweets)):
        interests.append(taggedtweets[i][1])
    words_to_count = (word for word in interests if word[:1].isupper())
    c = Counter(words_to_count)
    return (c.most_common(1)[0][0])
    

##Actual commands
tfis = preProcess(trainingcsv) #For options, check the function above.
classifier=makeClassifierSVM(tfis,interestlst) 
#lassifier=makeClassifierBayes(tfis,interestlst)

#testClassifier(classifier,testcsv,10000) #uncomment if you want to test the classifier
while True:
    interest = classifyUser(input('Enter a Twitter User: '))
    print('Users interest is '+interest)

