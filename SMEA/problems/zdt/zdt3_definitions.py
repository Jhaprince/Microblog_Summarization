import math
from SMEA import seq
from SMEA.problems.problem_definitions import ProblemDefinitions

from cluster_validity_indices.anti_redundancy import Sent_to_sent
# from cluster_validity_indices.sent_to_caption import Sent_to_caption
# from cluster_validity_indices.sent_ref_fig import Sent_ref_fig
class ZDT3Definitions(ProblemDefinitions):

    def __init__(self,solution_length, text_data, SS_EMD_matrix, Tweet_length_Matrix, tfidf_Matrix ):
        self.n = solution_length
        self.Tweet_cleaned_data=text_data
        self.SS_WMD_matrix =  SS_EMD_matrix
        self.Tweet_length_Matrix=Tweet_length_Matrix
        self.Tweet_tfidf_Matrix=tfidf_Matrix


    def f1(self, individual):
        obj1=Sent_to_sent(self.SS_WMD_matrix, individual.features)
        return obj1
        # obj2=Sent_to_caption(self.SC_WMD_matrix, individual.features, self.Fig_number)
        # return obj2
        # return individual.features[0]
        # obj1 = Sent_ref_fig(self.text_data, individual.features, self.Fig_number)
        # # print('obj1',obj1)
        # return obj1
    """
    def f2(self, individual):       #return sum of tweet length in the solution
        average_length = 0
        counter=0
        chromosome= individual.features
        for i in range(len(chromosome)):
            if chromosome[i] == 1:
                average_length+=self.Tweet_length_Matrix[i]
                counter+=1
        average_length=average_length/float(counter)
        return average_length


	"""
    def f2(self, individual):  #return sum of tf-idf value of each tweet in the solution
        tweet_tfidf_value=0
        counter=0
        chromosome = individual.features
        for i in range(len(chromosome)):
            if chromosome[i] == 1:
                tweet_tfidf_value+=self.Tweet_tfidf_Matrix[i]
                counter+=1
        tweet_tfidf_value=float(tweet_tfidf_value)/counter
        return tweet_tfidf_value

