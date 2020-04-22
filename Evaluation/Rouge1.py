
import re
from nltk.tokenize import sent_tokenize, word_tokenize

with open("/home/lenovo/Text-Summarization/TextRank/do3a/tr11sum") as f1, open( "/home/lenovo/Text-Summarization/Actual Summary/do3a/ac11") as f2:
    f11 = f1.read()
    f22 = f2.read()


word1 = re.findall(r'\w+', f11)
word2 = re.findall(r'\w+', f22)
arr = []
count = 0
for w1 in word1:
    for w2 in word2:
        if w1 == w2:
            c = 0
            for dt in arr:
                if w1 == dt:
                    c = 1
                    break
            if c == 0:
                c1 = word1.count(w1)
                c2 = word2.count(w2)
                if c1 < c2:
                    count += c1
                else:
                    count += c2
                arr.append(w1)
print ('No. of words in predicted summary : ', len(word1))
print ('No. of words in Actual summary : ', len(word2))
print ('No of overlapping words :  ', count)
p = float(count) / len(word1)
r = float(count) / len(word2)
print ('Precision  : ', p)
print ('Recall  : ', r)
print ('F-Measure  : ', (2.0 * p * r) / (p + r))

