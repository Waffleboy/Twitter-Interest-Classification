# Dataset Generator for Classifying Twitter Interests

This script generates a dataset for classification of twitter interests. It has the ability to use multiple twitter keys to speed up the process. 

Output: 

1)A CSV file of tweets / interest
2)A processed CSV file of the above.

# Instructions:

1) Edit the dicOfAccounts() function with inputs where Key = interest, Values = List of twitter accounts to mine

eg, dic['News'] = ['cnn','bbc','nytimes']. 'cnn','bbc','nytimes' will be mined and their tweets will be tagged as 'news'.

2)Add your twitter API test keys. uncomment out accesstokenlist and add all the keys you have.

3) (Optional) Run verifyTwitterAccounts to verify the twitter accounts given.

Upon running, Youll get a twitterInterests.csv in your python folder

Important:
If you want to recreate the CSV, treat it as first time running. else, COMMENT OUT makeCSV() in the last few lines.
