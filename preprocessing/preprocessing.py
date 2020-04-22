# encoding=utf8
from bs4 import BeautifulSoup
import HTMLParser
import sys
import re
import preprocessor as p
reload(sys)
sys.setdefaultencoding('utf8')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

"""
Below function remove the special characters, punctuation symbols, convert tweet into lower case
"""
def remove_punctuation_specialchar_htmltags_uppercase(pattern, phrase):
    """
    :param pattern: special characters to remove
    :param phrase: text tweet
    :return: cleaned tweet
    """
    print("hello :", phrase)
    phrase=HTMLParser.HTMLParser().unescape(phrase)   #remove HTML tags
    phrase=phrase.replace('(','')                    #Replace () by space
    phrase = phrase.replace(')', '')
    phrase=phrase.replace('RT', '')
    phrase=phrase.replace('-', ' ')
    phrase=phrase.lower()
    phrase=phrase.replace('#','')
    phrase = phrase.replace('\'s', '')
    phrase = phrase.replace('\'', '')
    #phrase=phrase.strip()
    #phrase = re.sub('[^A-Za-z]', '', phrase)
    #lower case conversion of tweets
    tt=None
    for pat in pattern:
        tt= re.findall(pat, phrase)                   #remove special symbol given in pattern list
    return tt
""" END of function """



text_data=[]
filepath = '/home/saini/PycharmProjects/Microblog_summarization_updated/data_set/ensemble-summarization-intellisys-2018-dataset/input_datasets/hagupit_input_data.txt'
with open(filepath) as fp:
   for cnt, line in enumerate(fp):
       text_data.append(line)  ##load the actual tweets from file



from wordcloud import WordCloud, STOPWORDS
pattern = ['[^&!./?;:,]+']
file1=open('/home/saini/PycharmProjects/Microblog_summarization_updated/data_set/ensemble-summarization-intellisys-2018-dataset/input_datasets/clean_tweets/clean_hangupit_tweets.txt', 'w')
for i in range(len(text_data)):
    p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.SMILEY,p.OPT.EMOJI )
    cleaned_tweet = p.clean(text_data[i])
    a = "".join(remove_punctuation_specialchar_htmltags_uppercase(pattern, cleaned_tweet))
    print "cleand tweet #{0} :".format(i), a
    tokens=a.split(' ')
    print("tokenize tweet #{0}".format(i), a.split(' '))
    #tokens = [c for c in tokens if (c.isalpha() and not c.isdigit()) ]
    filtered_sentence = [w.strip() for w in tokens if not w in stop_words]
    print("token after removing alphanumeric : ", tokens)
    tt=" ".join(w for w in filtered_sentence)
    print("after removing stop words :", tt)
    file1.write(str(tt)+'\n')
    print "------------------------------------------"
file1.close()


# """
# Visualization of tweets
# """
# import matplotlib.pyplot as plt
# import matplotlib
# #import seaborn as sns
# from IPython.display import display
# #from mpl_toolkits.basemap import Basemap
# def wordcloud(tweets):
#     stopwords = set(STOPWORDS)
#     wordcloud = WordCloud(background_color="white",stopwords=stopwords,random_state = 2016).generate(" ".join([i for i in tweets]))
#     plt.figure( figsize=(20,10), facecolor='k')
#     plt.imshow(wordcloud)
#     plt.axis("off")
#     plt.title("Good Morning Datascience+")
#     plt.show()
#     plt.savefig("/home/saini/PycharmProjects/Microblog_Text_summ/data_set/ensemble-summarization-intellisys-2018-dataset/input_datasets/clean_tweets/word_cloud/hangupit_word_cloud.jpeg")
# wordcloud(updated_text_data)
#
# """ END of visualization code"""

















#
# phrase = 'Welcome to Quora! We are happy to help you out. Need any help?'
#
#
# print("".join(remove_punctuation(pattern, phrase)))
#
# import re
#
#
# def remove_punctuation(pattern, phrase):
#     for pat in pattern:
#         return (re.findall(pat, phrase))
#         return ('\n')
#
#
# phrase = 'Welcome to Quora! We are happy to help you out. Need any help?'
# pattern = ['[^!.?]+']
#
# print("".join(remove_punctuation(pattern, phrase)))

