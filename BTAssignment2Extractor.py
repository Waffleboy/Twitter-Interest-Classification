# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 12:19:25 2015

Description:

This script enables one with multiple twitter keys to cycle through them, and make repeated API
requests to extract the tweets of different users into a nice CSV file.

Instructions:

***If first time running***:

1) Edit dicOfAccounts() with Key = Category, Values = List of twitter accounts to mine
Examples are given.
2)Follow instructions to add your twitter API test keys below! uncomment out accesstokenlist and add
all the keys you have
3) (Optional) Run verifyTwitterAccounts to verify the twitter accounts given.

Press F5 and just run. Youll get a BTAssignment.csv in your python folder

**If >1 time running*
If you want to recreate the CSV, treat it as first time running. else, you have to comment out
make csv in the last line (just run again luh honestly)

@author: Thiru
"""

import time,tweepy,csv,random
#from accesstokenTwitter import *

#Uncomment above if you have a config file with a lst
#of accesstokens,
#else:
#add manually below in the form of
# ClientID/ Client Secret/ Access Token / Access Secret

#Eg,

#accesstokenlist=[]
#accesstokenlist.append(['clientid','clientsecret','accesstoken','accesssecret'])

def dicOfAccounts():
    dic={}
    #eg,
    dic['News'] = ['cnn','bbc','nytimes']
    dic['Politics']=['barackobama']
    return dic

##Run one time only.
def makeCSV():
    with open('BTAssignment.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(["Twitter Account","text","Interest"])

def verifyTwitterAccounts(dic):
    print('Beginning verification of twitter accounts')
    currKeyID=0
    currentKey=accesstokenlist[currKeyID]
    auth = tweepy.auth.OAuthHandler(currentKey[0], currentKey[1])
    auth.set_access_token(currentKey[2], currentKey[3])
    api = tweepy.API(auth)
    error=False
    errorlist=[]
    for key,value in dic.items():
        try:
            for i in range(len(value)):
                api.get_user(value[i])
        except:
            errorlist.append(value[i])
            error=True
    if error == False:
        print('Error: The following users are private/do not exist :' + str(errorlist))
    else:
        return "No errors"

    
"""
PreCond: Takes in a dic of category:[twitternames]

Description: Writes CSV of the each accounts latest 3.2k tweets in utf-8 format.

"""
def extractTweets(dic):
    currKeyID=0
    currentKey=accesstokenlist[currKeyID]
    numtoken=len(accesstokenlist)     # Total number of access keys
    auth = tweepy.auth.OAuthHandler(currentKey[0], currentKey[1])
    auth.set_access_token(currentKey[2], currentKey[3])
    api = tweepy.API(auth)
    rateID=0
    timeStart=time.time()
    
    def changekey():
        nonlocal currKeyID
        nonlocal currentKey
        nonlocal numtoken
        nonlocal api,auth
        currKeyID = (currKeyID+1)%numtoken
        currentKey=accesstokenlist[currKeyID]
        auth = tweepy.auth.OAuthHandler(currentKey[0], currentKey[1])
        auth.set_access_token(currentKey[2], currentKey[3])
        api = tweepy.API(auth)
        
    def updateAPIRate():
        nonlocal rateID
        x=api.rate_limit_status()
        rateID=x['resources']['statuses']['/statuses/user_timeline']['remaining']
        
    def checkRateID():
        nonlocal rateID
        nonlocal timeStart
        if rateID<=1:
            changekey()
            updateAPIRate()
            if rateID<=1:
                timeDifference = time.time() - timeStart
                if timeDifference > 0:
                    print('RateID Exhausted, sleeping for rate reset. Key: '+str(currKeyID)) 
                    time.sleep(905 - timeDifference)
                    timeStart = time.time()
                    
    def removeLinksAndLastCharacter(lst):
        lst=[lst[0]]+lst[2:]
        for i in range(1,len(lst)):
            text = lst[i][1]
            text=text.lower()
            x = text.find('http')
            while x != -1:
                text = lst[i][1][:x] + lst[i][1][x+22:]
                lst[i][1] = text
                x = text.find('http')
            lst[i][1]=lst[i][1][:-1]
            lst[i][1]=lst[i][1][2:]
        return lst
    
    def shuffleList(lst):
        header=[lst[0]]
        lst=lst[1:]
        random.shuffle(lst)
        random.shuffle(lst)
        random.shuffle(lst)
        random.shuffle(lst)
        random.shuffle(lst)
        finallst = header + lst
        return finallst
    
    for key,value in dic.items():
        try:
            print('Currently processing topic: '+str(key))
            for i in range(len(value)):
                print('Currently processing user :' + value[i])
                tweetlst = []
                new_tweets = api.user_timeline(screen_name = value[i],count=200)
                tweetlst.extend(new_tweets)
                updateAPIRate()
                checkRateID()
                oldest = tweetlst[-1].id - 1
                
                while len(new_tweets) > 0:
                    checkRateID()
                    rateID-=1
                    new_tweets = api.user_timeline(screen_name = value[i],count=200,max_id=oldest)
                    tweetlst.extend(new_tweets)
                    oldest = tweetlst[-1].id - 1
                    
                outtweets = [[value[i],tweet.text.encode('utf8'),str(key)] for tweet in tweetlst]
                		
                with open('BTAssignment.csv', 'a',newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(outtweets)
            
        except Exception as e:
            print(e, 'Error occured while processing '+key + '' + str(value) + '' + 'Skipping!')
            print('Currently using key: '+str(currKeyID))
    print('Done with extraction, now removing links and last hyphen from CSV.')
    
    with open('BTAssignment.csv','r') as f:
        reader = csv.reader(f)
        lst=list(reader)
  
    
    ##Remove links, first b' and last '
    lst = removeLinksAndLastCharacter(lst)
    #Shuffle the CSV
    print('Shuffling CSV')
    lst = shuffleList(lst)
    print('Shuffling completed! Writing to CSV.')
    with open('BTAssignmentProcessed.csv','w',newline='') as f:
        writer = csv.writer(f)
        writer.writerows(lst)
    print('BTAssignmentProcessed.csv is now ready for use')

makeCSV()
dic = dicOfAccounts()
extractTweets(dic)
