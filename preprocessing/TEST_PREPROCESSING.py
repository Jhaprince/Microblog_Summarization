import HTMLParser
import sys
import re
import preprocessor as p
reload(sys)
import numpy as np
import pandas as pd
import re
import warnings
import bs4
#https://datascienceplus.com/twitter-analysis-with-python/


#tweets = pd.read_csv('/home/saini/PycharmProjects/Microblog_Summ_three_objectives/data_set/ensemble-summarization-intellisys-2018-dataset/input_datasets/uflood_input_data.txt',  sep='\n', header=None) #error_bad_lines=False)

actual_tweets=[]
filepath1 = '/home/saini/PycharmProjects/Microblog_Summ_three_objectives/data_set/ensemble-summarization-intellisys-2018-dataset/input_datasets/uflood_input_data.txt'
with open(filepath1) as fp:
   for cnt1, line1 in enumerate(fp):
       actual_tweets.append(line1)#.decode('utf-8', 'ignore'))  ##load the actual tweets from file
print("first actual tweet : ", actual_tweets[0])
print("total number of actual tweet : ", len(actual_tweets))

tweets = pd.DataFrame({0:actual_tweets})


print "No. of fetch tweets :", len(tweets[0])
# remove URLs, RTs, and twitter handles
for i in range(len(tweets[0])):
    tweets[0][i] = " ".join([word for word in tweets[0][i].split() if 'RT' not in word and 'http' not in word and '@' not in word and '<' not in word])
    s1 = re.sub("[\"()|,.;!:?']", " ", tweets[0][i])
    s1=s1.lower()
    tweets[0][i]=s1
print("tweets after removing @username, RT, url   : \n", tweets[0])
print "second tweet :", tweets[0][2]
print("14th tweet :", tweets[0][14])
print "no. of unique tweets :", len(np.unique(tweets[0]))

import pandas as pd
file1='clean_tweets/ukflood/clean1_ukflood_removeRT@URLSplchar_lowercase.csv'
df1 = pd.DataFrame(tweets[0])    #,'Solution no':avg_sol_no,'rouge_1_p': avg_rouge_1_p,'rouge_1_r': avg_rouge_1_r,'rouge_1_f': avg_rouge_1_f,'rouge_2_p': avg_rouge_2_p,'rouge_2_r': avg_rouge_2_r,'rouge_2_f': avg_rouge_2_f,'rouge_l_p': avg_rouge_l_p,'rouge_l_r': avg_rouge_l_r,'rouge_l_f': avg_rouge_l_f})
df1.to_csv(file1, index=False, index_label=False, header=False)

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++==")
tweets['new'] = ''

for i in range(len(tweets[0])):
    tweets['new'][i] = " ".join([word.replace('&amp;', '&') if '&amp;' in word else word  for word in tweets[0][i].split()  ]) #bs4.BeautifulSoup(tweets[0][i], "html.parser")
print tweets['new'][2]
print(tweets['new'][14])
print "len unique tweets :", len(np.unique(tweets['new']))

import pandas as pd
file2='clean_tweets/ukflood/clean2_ukflood_replace@amp.csv'
df2 = pd.DataFrame(tweets['new'])    #,'Solution no':avg_sol_no,'rouge_1_p': avg_rouge_1_p,'rouge_1_r': avg_rouge_1_r,'rouge_1_f': avg_rouge_1_f,'rouge_2_p': avg_rouge_2_p,'rouge_2_r': avg_rouge_2_r,'rouge_2_f': avg_rouge_2_f,'rouge_l_p': avg_rouge_l_p,'rouge_l_r': avg_rouge_l_r,'rouge_l_f': avg_rouge_l_f})
df2.to_csv(file2, index=False, index_label=False, header=False)

seen = set()
pos = []
uniq_tweets=[]
for x in range(len(tweets['new'])):
    if tweets['new'][x] not in seen and not seen.add(tweets['new'][x]):
        pos.append(x)
        uniq_tweets.append(tweets['new'][x])
    else:
        print "{0}th tweet is duplicate ".format(x)

file2='clean_tweets/ukflood/clean3_ukflood_unique_tweets.csv'
df2 = pd.DataFrame({"unique_tweets" : uniq_tweets})    #,'Solution no':avg_sol_no,'rouge_1_p': avg_rouge_1_p,'rouge_1_r': avg_rouge_1_r,'rouge_1_f': avg_rouge_1_f,'rouge_2_p': avg_rouge_2_p,'rouge_2_r': avg_rouge_2_r,'rouge_2_f': avg_rouge_2_f,'rouge_l_p': avg_rouge_l_p,'rouge_l_r': avg_rouge_l_r,'rouge_l_f': avg_rouge_l_f})
df2.to_csv(file2, index=False, index_label=False, header=False)



file3='clean_tweets/ukflood/clean3_ukflood_unique_tweets_positions.csv'
df3 = pd.DataFrame({"tweet_position " : pos})    #,'Solution no':avg_sol_no,'rouge_1_p': avg_rouge_1_p,'rouge_1_r': avg_rouge_1_r,'rouge_1_f': avg_rouge_1_f,'rouge_2_p': avg_rouge_2_p,'rouge_2_r': avg_rouge_2_r,'rouge_2_f': avg_rouge_2_f,'rouge_l_p': avg_rouge_l_p,'rouge_l_r': avg_rouge_l_r,'rouge_l_f': avg_rouge_l_f})
df3.to_csv(file3, index=False, index_label=False, header=False)

print("no. of unique tweets :", len(uniq_tweets))
#
# #Preprocessing delete  RT
# tweets['tweetos1'] = ''
#
# #add tweetos first part
# for i in range(len(tweets[0])):
#     try:
#         tweets['tweetos1'][i] = tweets[0].str.split(' ')[i][0]
#     except AttributeError:
#         tweets['tweetos1'][i] = 'other'
# #print(tweets['tweetos1'])
#
# #Preprocessing tweetos. select tweetos contains 'RT @'
# for i in range(len(tweets[0])):
#     if tweets['tweetos1'].str.contains('RT')[i]  == False:# or tweets['tweetos'].str.contains('RT')[i]  == False:
#         tweets['tweetos1'][i] = 'other'
#
# #print(tweets['tweetos1'])
# print("==============================")
#
# for i in range(len(tweets[0])):
#     tweets[0][i] = " ".join([word for word in tweets[0][i].split()
#                                 if 'RT' not in word])
# print("tweets after removing @username, url : \n ", tweets[0])