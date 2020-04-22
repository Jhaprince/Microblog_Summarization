#This program count the #hastag words in actual tweets and gold summary

import HTMLParser
import sys
import re
import preprocessor as p
reload(sys)
import numpy as np
import pandas as pd
import re

#https://datascienceplus.com/twitter-analysis-with-python/


tweets = pd.read_csv('/home/saini/PycharmProjects/Microblog_Summ_three_objectives/preprocessing/clean_tweets/hangupit/clean2_hangupit_replace@amp.csv', sep="\t", header=None) #error_bad_lines=False)

tweets['#tagactual']=''
for i in range(len(tweets[0])):
    print(tweets[0][i])
    counter=0
    #tweets['#tagactual'][i]=0
    split_tokens= tweets[0][i].split()
    print("tokens: ", split_tokens)
    for k in range(len(split_tokens)):
        if '#' in split_tokens[k]:
            counter=counter+1
    print("counter :", counter)
    tweets['#tagactual'][i] = counter
    #print(tweets['#tagactual'][i])
    print("======================================")
print("count of # tage in actual tweets : \n\n", tweets['#tagactual'])

"""=================================================================================================="""

tweet_gold1 = pd.read_csv('/home/saini/PycharmProjects/Microblog_Summ_three_objectives/data_set/ensemble-summarization-intellisys-2018-dataset/gold-standard-summaries/annotator1/annotator1-hagupit-summary.csv',sep="\t", header=None)
tweets['#tag_Gold1']=''
for i in range(len(tweet_gold1[0])):
    counter1=0
    split=tweet_gold1[0][i].split()
    for j in range(len(split)):
        if '#' in split[j]:
            counter1+=1
    tweets['#tag_Gold1'][i]=counter1

print("count in gold1 tweets:", tweets['#tag_Gold1'][0:len(tweet_gold1[0])])

print("====================================================================")

tweet_gold2 = pd.read_csv('/home/saini/PycharmProjects/Microblog_Summ_three_objectives/data_set/ensemble-summarization-intellisys-2018-dataset/gold-standard-summaries/annotator2/annotator2-hagupit-summary.csv',sep="\t", header=None)
tweets['#tag_Gold2']=''
for i in range(len(tweet_gold2[0])):
    counter2=0
    split=tweet_gold2[0][i].split()
    for j in range(len(split)):
        if '#' in split[j]:
            counter2+=1
    tweets['#tag_Gold2'][i]=counter2

print("count in gold2 tweets:", tweets['#tag_Gold2'][0:len(tweet_gold2[0])])

print("====================================================================")

tweet_gold3 = pd.read_csv('/home/saini/PycharmProjects/Microblog_Summ_three_objectives/data_set/ensemble-summarization-intellisys-2018-dataset/gold-standard-summaries/annotator3/annotator3-hagupit-summary.csv',sep="\t", header=None)
tweets['#tag_Gold3']=''
for i in range(len(tweet_gold3[0])):
    counter3=0
    split=tweet_gold3[0][i].split()
    for j in range(len(split)):
        if '#' in split[j]:
            counter3+=1
    tweets['#tag_Gold3'][i]=counter3

print("count in gold3 tweets:", tweets['#tag_Gold3'][0:len(tweet_gold3[0])])
fname_summ = '/home/saini/PycharmProjects/Microblog_Summ_three_objectives/preprocessing/count_#tag/hangupit/'+'coung#tag_hangupit.csv'
df = pd.DataFrame({'Actual_tweets': tweets['#tagactual'],'Annotator1_summary': tweets['#tag_Gold1'], 'Annotator2_summary': tweets['#tag_Gold2'], 'Annotator3_summary': tweets['#tag_Gold3']})
df.to_csv(fname_summ)


