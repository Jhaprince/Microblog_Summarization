#from metrics.problems.zdt import ZDT3Metrics ##comment it
from SMEA.evolution import Evolution
from SMEA.problems.zdt import ZDT
from SMEA.problems.zdt.zdt3_definitions import ZDT3Definitions
from plotter import Plotter
import numpy as np
import pandas as pd
import time


def print_generation(population, generation_num):
    print("Generation: {}".format(generation_num))

dataset=raw_input("Enter dataset name : ")
from numpy import genfromtxt, asarray, unique


"""---------------Fetch Tweet to Tweet Similarity (WMD Distance)  Matrix --------------"""
EMD_matrix=genfromtxt('../preprocessing/T2T_WMD_matrices/hblast2_T2T_WMD_matrix.txt', skip_header=0) #load EMD matrix for sentences
EMD_matrix=EMD_matrix[ :, 1:]
#print(EMD_matrix[0])
print("MATRIX SHAPE : ", EMD_matrix.shape)
"""----------------------------------END-----------------------------------------------------"""



"""---------------Fetch tf-idf value of each tweet --------------"""
MAX_TFIDF_matrix = genfromtxt('../preprocessing/MAX_tfidf_SCORE/hblast2_max_tfidf_score.txt', skip_header=0) #load EMD matrix for sentences
#print((MAX_TFIDF_matrix))
MAX_TFIDF_matrix= MAX_TFIDF_matrix[ :, 1:]
print("first tweet tf-idf score :", MAX_TFIDF_matrix[0])
print("MAX TFIDF MATRIX SHAPE : ", MAX_TFIDF_matrix.shape)
"""----------------------------------END-----------------------------------------------------"""



"""---------------Fetch tweet length of each tweet ------------------------------------------"""
MAX_TWEET_length_matrix=genfromtxt('../preprocessing/MAX_TWEET_LENGTH/hblast2_max_tweet_length.txt', skip_header=0) #load EMD matrix for sentences
#print((MAX_TWEET_length_matrix))
MAX_TWEET_length_matrix= MAX_TWEET_length_matrix[ :, 1:]
print("first tweet length :", MAX_TWEET_length_matrix[0])
print("MAX Tweet Length MATRIX SHAPE : ", MAX_TWEET_length_matrix.shape)
"""----------------------------------END-----------------------------------------------------"""


"""----------------------------Fetch cleaned Tweets ----------------------------------------- """
clean_text_data=[]
filepath = '../preprocessing/clean_tweets/hblast/clean2_hblast_replace@amp.csv'
with open(filepath) as fp:
   for cnt, line in enumerate(fp):
       clean_text_data.append(line.decode('utf-8', 'ignore'))  ##load the actual tweets from file
print("Clean fist tweet :", clean_text_data[0])
"""----------------------------------END-----------------------------------------------------"""


"""----------------------------Fetch actual Tweets ----------------------------------------- """
actual_text_data=[]
filepath1 = '../data_set/ensemble-summarization-intellisys-2018-dataset/input_datasets/hblast_input_data.txt'
with open(filepath1) as fp:
   for cnt1, line1 in enumerate(fp):
       actual_text_data.append(line1.decode('utf-8', 'ignore').lower())  ##load the actual tweets from file
print("first actual tweet : ", actual_text_data[0])
print("total number of tweet : ", len(actual_text_data))
"""----------------------------------END-----------------------------------------------------"""


"""----------------------------Fetch actual summary1 ----------------------------------------- """
actual_summary1=''
count_summary_line=0
filepath2 = '../data_set/ensemble-summarization-intellisys-2018-dataset/gold-standard-summaries/annotator1/annotator1-hblast-summary.txt'
with open(filepath2) as fp:
   for cnt2, line2 in enumerate(fp):
       actual_summary1+=''+line2.decode('utf-8', 'ignore').lower()
       count_summary_line+=1
print("actual summary1 :", actual_summary1)
"""----------------------------------END-----------------------------------------------------"""



"""----------------------------Fetch actual summary2 ----------------------------------------- """
actual_summary2=''
filepath3 = '../data_set/ensemble-summarization-intellisys-2018-dataset/gold-standard-summaries/annotator2/annotator2-hblast-summary.txt'
with open(filepath3) as fp:
   for cnt3, line3 in enumerate(fp):
       actual_summary2+=''+line3.decode('utf-8', 'ignore').lower()
       #actual_summary1.append(line2)  ##load the actual tweets from file
#print("actual summary : ", actual_text_data[0])
#print("total number of tweet : ", len(actual_text_data))
print("actual summary2 :", actual_summary2)
"""----------------------------------END-----------------------------------------------------"""



"""----------------------------Fetch actual summary3 ----------------------------------------- """
actual_summary3=''

filepath4 = '../data_set/ensemble-summarization-intellisys-2018-dataset/gold-standard-summaries/annotator3/annotator3-hblast-summary.txt'
with open(filepath4) as fp:
   for cnt4, line4 in enumerate(fp):
       actual_summary3+=''+line4.decode('utf-8', 'ignore').lower()
       
       #actual_summary1.append(line2)  ##load the actual tweets from file
print("actual summary3 : ", actual_summary3)
#print("total number of tweet : ", len(actual_text_data))
"""----------------------------------END-----------------------------------------------------"""


max_len_solution = len(clean_text_data)
print ("maximum length of solution : ", max_len_solution)
print("no. of sentence in the article : ", len(clean_text_data))
print ("********no. of line in summary :******", count_summary_line)
pop_size=input("Enter size of population : ")
H=input("Enter mating pool size : ")
print ("no. of line in summary :", count_summary_line)
smin = int(input("Enter the minimum number of tweets in the summary: "))
smax = int(input("Enter the maximum number of tweets in the summary: "))
T=input("Enter maximum no. of generation : ")

start = time.time()
print( "starting time :", start)



SMEA_clustering = ZDT3Definitions(max_len_solution, clean_text_data,  EMD_matrix, MAX_TWEET_length_matrix, MAX_TFIDF_matrix)        #n=30   ===>is for no. of features
problem = ZDT(SMEA_clustering, max_len_solution)
evolution = Evolution(problem, T, pop_size, H)
evolution.register_on_new_generation(print_generation)
final_population,K = evolution.evolve(EMD_matrix, MAX_TFIDF_matrix , MAX_TWEET_length_matrix, smin, smax, max_len_solution, dataset, actual_text_data, actual_summary1, actual_summary2, actual_summary3)
print("length final population:",  len(final_population))

end = time.time()
print  "ending time : ", end
print "Total execution time :", end-start
total_time=end-start


fname1 = '../Output/' + str(dataset) + '/running_time'
text_file1 = open(fname1, "w")
text_file1.write("Starting time : "+ str(start)+'\n')
text_file1.write("Starting time : "+ str(end)+'\n')
text_file1.write("Total execution time : "+ str(total_time)+'\n')
text_file1.close()

fname2 = '../Output/' + str(dataset) + '/Min_max_sentence'
text_file2 = open(fname2, "w")
text_file2.write("Minium number of sentence taken : "+ str(smin)+'\n')
text_file2.write("Maximum number of sentence taken : "+ str(smax)+'\n')
text_file2.write("Number of sentence in the summary : "+ str(count_summary_line)+'\n')
text_file2.close()



import os
All_summary=[]
 #record solution no. (same solution number for different annotators)




solution_no=0

annotator1_sol_no = []
ann1_datasetname=[]
ann1_rouge_1_p, ann1_rouge_1_r, ann1_rouge_1_f = [], [], []
ann1_rouge_2_p, ann1_rouge_2_r, ann1_rouge_2_f = [], [], []
ann1_rouge_l_p, ann1_rouge_l_r, ann1_rouge_l_f = [], [], []

annotator2_sol_no = []
ann2_datasetname=[]
ann2_rouge_1_p, ann2_rouge_1_r, ann2_rouge_1_f = [], [], []
ann2_rouge_2_p, ann2_rouge_2_r, ann2_rouge_2_f = [], [], []
ann2_rouge_l_p, ann2_rouge_l_r, ann2_rouge_l_f = [], [], []


annotator3_sol_no = []
ann3_datasetname=[]
ann3_rouge_1_p, ann3_rouge_1_r, ann3_rouge_1_f = [], [], []
ann3_rouge_2_p, ann3_rouge_2_r, ann3_rouge_2_f = [], [], []
ann3_rouge_l_p, ann3_rouge_l_r, ann3_rouge_l_f = [], [], []


avg_sol_no=[]            #record solution number
avg_dataset_name=[]
avg_rouge_l_p, avg_rouge_l_r, avg_rouge_l_f = [], [], []
avg_rouge_2_p, avg_rouge_2_r, avg_rouge_2_f = [], [], []
avg_rouge_1_p, avg_rouge_1_r, avg_rouge_1_f = [], [], []

for individual in final_population:
    Summary = ''
    features=individual.features

    #print "Summary {0}  : \n".format(solution_no)
    for j in range(len(features)):
        if features[j]==1:
            #position=Clean_tweet_positions[j]
            #print(type(position))
            #ones_position.append(j)
            #print " Sentence number {0} :".format(j), actual_text_data[j]
            Summary+= " "+ actual_text_data[j] #.decode('utf-8', 'ignore')
    #print("Summary {0} :  ".format(solution_no), Summary)
    All_summary.append(Summary)
    if not os.path.isdir('../Output/' + str(dataset) + '/'+ 'Predicted_summary'):
        os.makedirs('../Output/' + str(dataset) + '/Predicted_summary')

    fname = '../Output/' + str(dataset) + '/Predicted_summary/' + "Summary-{0}".format(solution_no)

    text_file = open(fname, "w")
    text_file.write(Summary.encode('utf-8'))

    from rouge import Rouge
    rouge = Rouge()
    actual1_scores = rouge.get_scores(Summary, actual_summary1)
    print("summary1 score :", actual1_scores)
    text_file.write('\n\nRouge score with annotator1 : \n'+str(actual1_scores)+'\n')

    """Record annotator1 score of all solutions to store in .csv"""
    annotator1_sol_no.append(solution_no)
    #dataset_name.append(dataset)
    #Annotator_no.append(1)
    ann1_datasetname.append(dataset)
    ann1_rouge_l_p.append(actual1_scores[0]['rouge-l']['p'])
    ann1_rouge_l_r.append(actual1_scores[0]['rouge-l']['r'])
    ann1_rouge_l_f.append(actual1_scores[0]['rouge-l']['f'])

    ann1_rouge_2_p.append(actual1_scores[0]['rouge-2']['p'])
    ann1_rouge_2_r.append(actual1_scores[0]['rouge-2']['r'])
    ann1_rouge_2_f.append(actual1_scores[0]['rouge-2']['f'])

    ann1_rouge_1_p.append(actual1_scores[0]['rouge-1']['p'])
    ann1_rouge_1_r.append(actual1_scores[0]['rouge-1']['r'])
    ann1_rouge_1_f.append(actual1_scores[0]['rouge-1']['f'])

    """Calculate and Record annotator2 score of all solutions to store in .csv"""
    actual2_scores = rouge.get_scores(Summary, actual_summary2)
    print("summary2 score :", actual2_scores)
    text_file.write('\n\nRouge score with annotator2 : \n' + str(actual2_scores) + '\n')

    annotator2_sol_no.append(solution_no)
    ann2_datasetname.append(dataset)
    #dataset_name.append(dataset)
    #Annotator_no.append(2)
    ann2_rouge_l_p.append(actual2_scores[0]['rouge-l']['p'])
    ann2_rouge_l_r.append(actual2_scores[0]['rouge-l']['r'])
    ann2_rouge_l_f.append(actual2_scores[0]['rouge-l']['f'])

    ann2_rouge_2_p.append(actual2_scores[0]['rouge-2']['p'])
    ann2_rouge_2_r.append(actual2_scores[0]['rouge-2']['r'])
    ann2_rouge_2_f.append(actual2_scores[0]['rouge-2']['f'])

    ann2_rouge_1_p.append(actual2_scores[0]['rouge-1']['p'])
    ann2_rouge_1_r.append(actual2_scores[0]['rouge-1']['r'])
    ann2_rouge_1_f.append(actual2_scores[0]['rouge-1']['f'])

    """Calculate and Record annotator3 score of all solutions to store in .csv"""
    actual3_scores = rouge.get_scores(Summary, actual_summary3)
    print("summary3 score :", actual3_scores)
    text_file.write('\n\nRouge score with annotator3 : \n' + str(actual3_scores) + '\n')
    text_file.close()

    annotator3_sol_no.append(solution_no)
    ann3_datasetname.append(dataset)
    ann3_rouge_l_p.append(actual3_scores[0]['rouge-l']['p'])
    ann3_rouge_l_r.append(actual3_scores[0]['rouge-l']['r'])
    ann3_rouge_l_f.append(actual3_scores[0]['rouge-l']['f'])

    ann3_rouge_2_p.append(actual3_scores[0]['rouge-2']['p'])
    ann3_rouge_2_r.append(actual3_scores[0]['rouge-2']['r'])
    ann3_rouge_2_f.append(actual3_scores[0]['rouge-2']['f'])

    ann3_rouge_1_p.append(actual3_scores[0]['rouge-1']['p'])
    ann3_rouge_1_r.append(actual3_scores[0]['rouge-1']['r'])
    ann3_rouge_1_f.append(actual3_scores[0]['rouge-1']['f'])
    """ End of storing annotator wise score """



    """Now calculating average score of annotators for current solution number to store into .csv file"""
    #print(type(actual1_scores))
    #print(type(actual1_scores[0]))

    rouge_avg_score_score={'rouge-l': {},'rouge-2': {}, 'rouge-1':{}   }
    for k in range(len(actual1_scores[0].keys())):
        key = actual1_scores[0].keys()[k]
        #print "key :", key
        m_key_val = actual1_scores[0][key]
        n_key_val = actual2_scores[0][key]
        p_key_val = actual3_scores[0][key]
        score = {k: (m_key_val.get(k, 0) + n_key_val.get(k, 0) + p_key_val.get(k, 0)) / float(3) for k in
                 set(m_key_val) & set(n_key_val) & set(p_key_val)}
        #print score
        rouge_avg_score_score[key] = score

    avg_sol_no.append(solution_no)
    avg_dataset_name.append(dataset)
    avg_rouge_l_p.append(rouge_avg_score_score['rouge-l']['p'])
    avg_rouge_l_r.append(rouge_avg_score_score['rouge-l']['r'])
    avg_rouge_l_f.append(rouge_avg_score_score['rouge-l']['f'])

    avg_rouge_2_p.append(rouge_avg_score_score['rouge-2']['p'])
    avg_rouge_2_r.append(rouge_avg_score_score['rouge-2']['r'])
    avg_rouge_2_f.append(rouge_avg_score_score['rouge-2']['f'])

    avg_rouge_1_p.append(rouge_avg_score_score['rouge-1']['p'])
    avg_rouge_1_r.append(rouge_avg_score_score['rouge-1']['r'])
    avg_rouge_1_f.append(rouge_avg_score_score['rouge-1']['f'])
    """End of storing average rouge score for current solution number"""

    print("total rouge score of solution-{0} : ".format(solution_no), rouge_avg_score_score)
    print("==============================================================")
    solution_no+=1





#ann1_max_R2_recall_index=ann1_rouge_2_r.index(max(ann1_rouge_2_r))
#ann1_max_RL_recall_index=ann1_rouge_l_r.index(max(ann1_rouge_l_r))
#ann1_datasetname.append('Max_R2_recal({0})'.format(annotator1_sol_no[ann1_max_R2_recall_index]))
f1name_summ = '../Output/' + str(dataset) + '/'+'Annotator1_solutionwise_summary_score_overview.csv'
df1 = pd.DataFrame({'dataset': ann1_datasetname,'Solution no':annotator1_sol_no, 'rouge_1_p': ann1_rouge_1_p,'rouge_1_r': ann1_rouge_1_r,'rouge_1_f': ann1_rouge_1_f,'rouge_2_p': ann1_rouge_2_p,'rouge_2_r': ann1_rouge_2_r,'rouge_2_f': ann1_rouge_2_f,'rouge_l_p': ann1_rouge_l_p,'rouge_l_r': ann1_rouge_l_r,'rouge_l_f': ann1_rouge_l_f})
df1.to_csv(f1name_summ)

f2name_summ = '../Output/' + str(dataset) + '/'+'Annotator2_solutionwise_summary_score_overview.csv'
df2 = pd.DataFrame({'dataset': ann2_datasetname,'Solution no':annotator2_sol_no, 'rouge_1_p': ann2_rouge_1_p,'rouge_1_r': ann2_rouge_1_r,'rouge_1_f': ann2_rouge_1_f,'rouge_2_p': ann2_rouge_2_p,'rouge_2_r': ann2_rouge_2_r,'rouge_2_f': ann2_rouge_2_f,'rouge_l_p': ann2_rouge_l_p,'rouge_l_r': ann2_rouge_l_r,'rouge_l_f': ann2_rouge_l_f})
df2.to_csv(f2name_summ)


f3name_summ = '../Output/' + str(dataset) + '/'+'Annotator3_solutionwise_summary_score_overview.csv'
df3 = pd.DataFrame({'dataset': ann3_datasetname,'Solution no':annotator3_sol_no, 'rouge_1_p': ann3_rouge_1_p,'rouge_1_r': ann3_rouge_1_r,'rouge_1_f': ann3_rouge_1_f,'rouge_2_p': ann3_rouge_2_p,'rouge_2_r': ann3_rouge_2_r,'rouge_2_f': ann3_rouge_2_f,'rouge_l_p': ann3_rouge_l_p,'rouge_l_r': ann3_rouge_l_r,'rouge_l_f': ann3_rouge_l_f})
df3.to_csv(f3name_summ)


f4name_summ = '../Output/' + str(dataset) + '/'+'Average_summary_score_overview.csv'
df4 = pd.DataFrame({'dataset': avg_dataset_name,'Solution no':avg_sol_no,'rouge_1_p': avg_rouge_1_p,'rouge_1_r': avg_rouge_1_r,'rouge_1_f': avg_rouge_1_f,'rouge_2_p': avg_rouge_2_p,'rouge_2_r': avg_rouge_2_r,'rouge_2_f': avg_rouge_2_f,'rouge_l_p': avg_rouge_l_p,'rouge_l_r': avg_rouge_l_r,'rouge_l_f': avg_rouge_l_f})
df4.to_csv(f4name_summ)

results = ''

#print('article No : {0} Fig No : {1}  Pop-size : {2}'.format(articleno, fig_number, pop_size))
results+= "Best Solution as per Avg. Max Rouge_1_precision: Solution no={0}, Rouge-1 precision score={1}, No. of tweet={2} \n".format(avg_rouge_1_p.index(max(avg_rouge_1_p)), avg_rouge_1_p[avg_rouge_1_p.index(max(avg_rouge_1_p))],K[avg_rouge_1_p.index(max(avg_rouge_1_p))])
results+= "Best Solution as per Avg. Max Rouge_1_recall: Solution no={0}, Rouge-1 recall score={1}, No. of tweet={2}\n".format(avg_rouge_1_r.index(max(avg_rouge_1_r)),avg_rouge_1_r[avg_rouge_1_r.index(max(avg_rouge_1_r))],K[avg_rouge_1_r.index(max(avg_rouge_1_r))])
results+= "Best Solution as per Avg. Max Rouge_1_F1: Solution no={0}, Rouge-1 F1 score={1}, No. of tweet={2}\n".format(avg_rouge_1_f.index(max(avg_rouge_1_f)),avg_rouge_1_f[avg_rouge_1_f.index(max(avg_rouge_1_f))],K[avg_rouge_1_f.index(max(avg_rouge_1_f))])

results+= "Best Solution as per Avg. Max Rouge_2_precision: Solution no={0}, Rouge-2 precision score={1}, No. of tweet={2}\n".format(avg_rouge_2_p.index(max(avg_rouge_2_p)),avg_rouge_2_p[avg_rouge_2_p.index(max(avg_rouge_2_p))],K[avg_rouge_2_p.index(max(avg_rouge_2_p))])
results+= "Best Solution as per Avg. Max Rouge_2_recall: Solution no={0}, Rouge-2 recall score={1}, No. of tweets={2}\n".format(avg_rouge_2_r.index(max(avg_rouge_2_r)),avg_rouge_2_r[avg_rouge_2_r.index(max(avg_rouge_2_r))],K[avg_rouge_2_r.index(max(avg_rouge_2_r))])
results+= "Best Solution as per Avg. Max Rouge_2_F1: Solution no={0}, Rouge-2 F1 score={1}, No. of tweet={2}\n".format(avg_rouge_2_f.index(max(avg_rouge_2_f)),avg_rouge_2_f[avg_rouge_2_f.index(max(avg_rouge_2_f))],K[avg_rouge_2_f.index(max(avg_rouge_2_f))])

results+= "Best Solution as per Avg. Max Rouge_L_precision: Solution no={0}, Rouge-L precision score={1}, No. of tweet={2}\n".format(avg_rouge_l_p.index(max(avg_rouge_l_p)),avg_rouge_l_p[avg_rouge_l_p.index(max(avg_rouge_l_p))],K[avg_rouge_l_p.index(max(avg_rouge_l_p))])
results+= "Best Solution as per Avg. Max Rouge_L_recall: Solution no={0}, Rouge-L recall score={1}, No. of tweets={2}\n".format(avg_rouge_l_r.index(max(avg_rouge_l_r)),avg_rouge_l_r[avg_rouge_l_r.index(max(avg_rouge_l_r))],K[avg_rouge_l_r.index(max(avg_rouge_l_r))])
results+= "Best Solution as per Avg. Max Rouge_L_F1: Solution no={0}, Rouge-L F1 score={1}, No. of tweet={2}\n".format(avg_rouge_l_f.index(max(avg_rouge_l_f)),avg_rouge_l_f[avg_rouge_l_f.index(max(avg_rouge_l_f))],K[avg_rouge_l_f.index(max(avg_rouge_l_f))])

results+= str(K)

print(results)
fname = '../Output/' + str(dataset) + '/Best_resulting_Solutions.txt'

text_file = open(fname, "w")
text_file.write(results.encode('utf-8'))
text_file.close()
