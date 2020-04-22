

import gensim


# Load Google's pre-trained Word2Vec model.
model =  gensim.models.KeyedVectors.load_word2vec_format('./crisis_word2vec/crisisNLP_word_vector.bin', binary=True)  



text_data=[]
filepath = '/home/saini/PycharmProjects/Microblog_summarization_updated/data_set/ensemble-summarization-intellisys-2018-dataset/input_datasets/clean_tweets/clean_hangupit_tweets.txt'

f5=open('Hangup_T2T_WMD_matrix.txt','w')


"""
with open(filepath) as fp:
   for cnt, line in enumerate(fp):
       text_data.append(line)  ##load the actual tweets from file


for i in range(len(text_data)):
	t=i+1	
	for k in range(t,len(text_data)):
		if i==k:
			pass
		elif k>i:
			sentence1 = text_data[i]
			sentence2 = text_data[k+1]
			print "sentence1 {0} :".format(i), sentence1
			print "sentence2  {1} :".format(k), sentence2
			sentence1 = sentence1.lower().split()
			sentence2 = sentence2.lower().split()
			distance = model.wmdistance(sentence1, sentence2)
			print 'distance = %.4f' % distance
			print "============================"
	break
"""
with open(filepath) as fp:
	for cnt, line in enumerate(fp):
		f5.write(str(cnt))
		with open(filepath) as fp1: 
			for cnt1, line1 in enumerate(fp1):
				#start  =  time()		
				sentence_obama = line.lower().split()
				sentence_president = line1.lower().split()
				sentence_obama = [w for w in sentence_obama]
				sentence_president = [w for w in sentence_president]
				#model.init_sims(replace=True)  # Normalizes the vectors in the word2vec class.
				distance = model.wmdistance(sentence_obama, sentence_president)
				f5.write('\t'+ str(distance))
				print("Line {}: {}".format(cnt, line))
				print("Line {}: {}".format(cnt1, line1))
				print ('distance b/w sentence cnt and cnt1  = %.3f' % distance )
				#print('Cell took %.2f seconds to run.' % (time() - start))      
				print("-----------------------------------------") 		
			f5.write('\n')
		

		
