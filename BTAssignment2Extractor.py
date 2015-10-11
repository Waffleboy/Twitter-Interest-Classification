# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 12:19:25 2015

Description:

This script enables one with multiple twitter keys to cycle through them, and make repeated API
requests to extract the tweets of different users into CSV files.

@author: Thiru
"""

import time,tweepy,csv
from accesstokenTwitter import *

#Uncomment above if you have a config file with a lst
#of accesstokens, else, add manually below in the form of
# ClientID/ Client Secret/ Access Token / Access Secret

#Eg,

#accesstokenlist=[]
#accesstokenlist.append('clientid','clientsecret','accesstoken','accesssecret')
def dicOfAccounts():
    dic={}
    ##Add your dic stuff here. eg,
    dic['News'] = ['cnn','bbc']
    return dic

##Run one time only.
def makeCSV():
    with open('BTAssignment.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(["Twitter Account","text","Interest"])
        
"""
PreCond: Takes in a lst of twitter screennames.

Description: Writes CSV's of the user's latest 3.2k tweets in utf-8 format.
            one CSV per user.

Instructions: If you want to start from a specific user, change startFrom from
              0 to the index of the user. Else, just run.
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
    
    def verifyTwitterAccounts(dic):
        print('Beginning verification of twitter accounts')
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
                
                counter=3 #get 600 tweets
                while len(new_tweets) > 0 and counter >0:
                    checkRateID()
                    counter-=1
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

makeCSV()
dic = dicOfAccounts()
extractTweets(dic)
