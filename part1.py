# @author: Sanjeev Kumar Singh(UIN: 922002623)
# Created as part of Assignment-3 of Information Retrieval course(CSCE-670)
# filename: part1.py

from pybing import Bing
import httplib
import urllib, urllib2
import json
import re
import math
import numpy as np
from pylab import *
import random

class DocCluster():
    
    username = 'ffdcc815-e721-48cd-b32a-692afcf2d343'
    accountKey='Qnw3YkoQdoRjdOPbXPA4oph8C160JGM5QKQjqQYYO6E='
    
    '''splits the text and returns the tokens'''  
    def split_text(self, text):
        text = text.encode('ascii', 'ignore')
        text = re.compile("[^\w]|_").sub(" ", text)
        word_list = re.findall("\w+", text.lower())    
        return word_list
    
    '''makes a bing search query for given terms with the skip parameter'''
    def search_Bing(self, termsList, skip):
        queryBingFor = "'" + ' '.join(termsList) + "'"
        quoted_query = urllib.quote(queryBingFor)
    
        rootURL = "https://api.datamarket.azure.com/Data.ashx/Bing/Search/v1/"
        searchURL = rootURL + "News?Query=" + quoted_query + "&$format=json" + "&$skip=%d" %(skip)
    
        password_mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()
        password_mgr.add_password(None, searchURL,self.username,self.accountKey)
    
        handler = urllib2.HTTPBasicAuthHandler(password_mgr)
        opener = urllib2.build_opener(handler)
        urllib2.install_opener(opener)
        readURL = urllib2.urlopen(searchURL).read()
        
        data = json.loads(readURL)
        documentList = data['d']['results']
        
        return documentList
    
    ''' dumps the description + tile of each of the searched result into a file''' 
    def create_repository(self):
        f = open("repository_part1.txt", "wb")
        f_titles = open("titles_part1.txt", "wb")
        query_list = [['texas', 'aggies'], ['texas', 'longhorns'], ['duke', 'blue', 'devils'],\
                      ['dallas', 'cowboys'], ['dallas', 'mavericks']]
        
        for query in query_list:
            query_results_1 = self.search_Bing(query, 0)
            query_results_2 = self.search_Bing(query, 15)
            
            for result in query_results_1:
                description = result['Description']
                title = result['Title']
                f.write(description + title + "\n")
                f_titles.write(title + "\n")
              
            for result in query_results_2:
                description = result['Description']
                title = result['Title']
                f.write(description + title + "\n") 
                f_titles.write(title + "\n") 
        
        f.close()        
        return
    
    ''' tokenizes each of the documents and stores in a list '''
    def tokenize_documents(self):
        tokenized_doc_list = []
        f = open("repository_part1.txt", "rb")
        documents = f.readlines()
        
        for document in documents:
            tokenized_doc_list.append(self.split_text(document))
            
        f.close()
        return tokenized_doc_list
          
    '''creates the sorted vector with all unique terms existing in the corpus'''
    def create_unique_token_vector(self, tokenized_doc_list):
        unique_token_vector = []
        for doc_list in tokenized_doc_list:
            for term in doc_list:
                if term not in unique_token_vector:
                    unique_token_vector.append(term)
                    
        unique_token_vector = sorted(unique_token_vector)
        return unique_token_vector
    
    ''' creates the global dictionary for the whole corpus with tf values of different terms'''
    def create_dictionary(self, tokenized_doc_list):
        dict = {}    
        for index in range(len(tokenized_doc_list)):
            set_tokens = set(tokenized_doc_list[index])
            
            for term in set_tokens:
                count = 0
                for word in tokenized_doc_list[index]:
                    if term == word:
                        count += 1 
                
                temp_dict = {}
                if term in dict:
                    temp_dict = dict[term]
                    
                temp_dict[index] = 1.0 + math.log(count, 2)
                dict[term] = temp_dict
                              
        return dict
    
    '''updates the global dictionary with idf values'''
    def updateDictWithIDF(self, dictionary, tokenized_doc_list):
        terms = dictionary.keys()
        corpus_length = len(tokenized_doc_list)
    
        for term in terms:
            temp_dict = dictionary[term]
            docIDList = temp_dict.keys()
            calculated_idf = math.log(float(corpus_length)/float(len(docIDList)), 2)
            
            for docID in docIDList:
                temp_dict[docID] *= calculated_idf
                
            dictionary[term] = temp_dict
            
        return dictionary
    
    '''calculates the normalization factor for each doc to be used for normalizing the tf-idf'''
    def calculateNormalizationFactorForEachDoc(self, dict):
        terms = dict.keys()
        dictionary = {}
        
        for term in terms:
            temp_dict = dict[term]
            docIDList = temp_dict.keys()
            
            for docID in docIDList:
                if docID in dictionary:
                    dictionary[docID] += temp_dict[docID] * temp_dict[docID]
                else:
                    dictionary[docID] = temp_dict[docID] * temp_dict[docID]
                    
        for key, value in dictionary.items():
            dictionary[key] = math.sqrt(value)
            
        return dictionary 
    
    '''updates the global dictionary with the calculated normalization factor'''
    def updateDictWithDocNormFactor(self, dict, normalized_dict):
        terms = dict.keys()
        
        for term in terms:
            temp_dict = dict[term]
            docIDList = temp_dict.keys()
            
            for docID in docIDList:
                if normalized_dict[docID] != 0.0:
                    temp_dict[docID] = float(temp_dict[docID])/float(normalized_dict[docID])
                else:
                    temp_dict[docID] = 0.0
                
            dict[term] = temp_dict
            
        return dict
     
    '''creates dictionary with DocID and terms with their normalized tf-idfs'''
    def createDictWithDocIDandTermsWithTheirTFIDFs(self, updatedDict):
        dict = {}
        for key, value in updatedDict.items():
            for k, v in value.items():
                temp_dict = {}
                if k in dict:
                    temp_dict = dict[k]
                    temp_dict[key] = v
                else:
                    temp_dict[key] = v
                    
                dict[k] = temp_dict
                
        return dict
    
    '''creates dictionary with docID and corresponding vector representation of the doc'''
    def createDictDocIDandVector(self, finalDict, unique_token_vector):
        dict = {}
        for key, value in finalDict.items():
            temp_list = []
            for token in unique_token_vector:
                if token in value:
                    temp_list.append(value[token])
                else:
                    temp_list.append(0.0)
            dict[key] = temp_list
            
        return dict
                    
    ''' for calculating the Euclidean distance betwen two documents '''
    def calculate_euclidean_distance(self, tokenized_doc_list, docID1, docID2):
        doc1 = tokenized_doc_list[docID1]
        doc2 = tokenized_doc_list[docID2]
        dictionary = self.updateDictWithIDF()
        
        set_doc1 = set(tokenized_doc_list[docID1])
        set_doc2 = set(tokenized_doc_list[docID2])
        
        terms = set_doc1.union(set_doc2)
        total_euclidean_distance = 0.0
        
        for term in terms:
            temp_dict = dictionary[term]
            if doc1 and doc2 in temp_dict:
                total_euclidean_distance += abs(temp_dict[doc1]**2 - temp_dict[doc2]**2)
            elif doc1 in temp_dict:
                total_euclidean_distance += temp_dict[doc1]**2
            else:
                total_euclidean_distance += temp_dict[doc2]**2
            
        return total_euclidean_distance
    
    ''' compares two vectors '''
    def compare_vectors(self, vector1, vector2):
        flag = True
        if len(vector1) == len(vector2):
            for index in range(len(vector1)):
                if vector1[index] != vector2[index]:
                    flag = False
        else:
            flag = False
            
        return flag
        
    ''' implementation of k-means algorithm '''
    def implement_kMeans(self, dictWithDocIDandVector, K):
        dict_clusters = {}
        dict_centroid = {}
        initial_dict_clusters = {}
        steady_state_count = 0
        
        random_list = []
        i = 0
        while i < K:
            rand = random.randint(0,len(dictWithDocIDandVector.keys())-1)
            if rand not in random_list:
                str = 'cluster_%d' %(i)
                dict_centroid[str] = dictWithDocIDandVector[rand]
                i += 1
        
        while True:
            for j in range(K):
                str = 'cluster_%d' %(j)
                dict_clusters[str] = []
                
            for key, value in dictWithDocIDandVector.items():
                distance_dict = {}
                for k, v in dict_centroid.items():
                    distance = 0.0
                    for index in range(len(value)):
                        distance +=  (value[index] - v[index])**2
                        
                    distance_dict[k] = math.sqrt(distance)
                clusterWithMinDistance = ''
                clusterWithMinDistance = min(distance_dict.iterkeys(), key=lambda k: distance_dict[k])
                dict_clusters[clusterWithMinDistance].append(key)
                
            '''compute new centroids'''
            for key, value in dict_clusters.items():
                doc_count = len(value)
                
                new_vector = [0.0 for x in range(len(dictWithDocIDandVector[0]))]
                for elem in value:
                    vector_dash = dictWithDocIDandVector[elem]
                    new_vector = [new_vector[i] + vector_dash[i] for i in range(len(new_vector))]
                   
                if doc_count != 0: 
                    new_vector = [x/float(doc_count) for x in new_vector]
                dict_centroid[key] = new_vector
            if initial_dict_clusters == dict_clusters:
                steady_state_count += 1
                if steady_state_count == 5:
                    break  
            else:
                steady_state_count = 0
        
            for k, v in dict_clusters.items():
                    initial_dict_clusters[k] = v  
                          
        return dict_clusters, dict_centroid
    
    ''' implementation of k-means algorithm  for cosine similarity'''
    def implement_kMeansWithCosineSimilarity(self, dictWithDocIDandVector, K):
        dict_clusters = {}
        dict_centroid = {}
        initial_dict_clusters = {}
        steady_state_count = 0
        
        random_list = []
        i = 0
        while i < K:
            rand = random.randint(0,len(dictWithDocIDandVector.keys())-1)
            if rand not in random_list:
                str = 'cluster_%d' %(i)
                dict_centroid[str] = dictWithDocIDandVector[rand]
                i += 1
        
        while True:
            for j in range(K):
                str = 'cluster_%d' %(j)
                dict_clusters[str] = []
                
            for key, value in dictWithDocIDandVector.items():
                distance_dict = {}
                for k, v in dict_centroid.items():
                    distance = 0.0
                    for index in range(len(value)):
                        distance +=  (value[index] - v[index])**2
                        
                    distance_dict[k] = math.sqrt(distance)
                clusterWithMinDistance = ''
                clusterWithMinDistance = min(distance_dict.iterkeys(), key=lambda k: distance_dict[k])
                dict_clusters[clusterWithMinDistance].append(key)
                
            '''compute new centroids'''
            for key, value in dict_clusters.items():
                doc_count = len(value)
                
                new_vector = [0.0 for x in range(len(dictWithDocIDandVector[0]))]
                for elem in value:
                    vector_dash = dictWithDocIDandVector[elem]
                    new_vector = [new_vector[i] + vector_dash[i] for i in range(len(new_vector))]
                if doc_count != 0: 
                    new_vector = [x/float(doc_count) for x in new_vector]
                dict_centroid[key] = new_vector
            if initial_dict_clusters == dict_clusters:
                steady_state_count += 1
                if steady_state_count == 5:
                    break  
            else:
                steady_state_count = 0
        
            for k, v in dict_clusters.items():
                    initial_dict_clusters[k] = v  
                          
        return dict_clusters, dict_centroid
    
    '''calculates the RSS value'''
    def calculate_RSS(self, dict_clusters, dictWithDocIDandVector, dict_centroid):
        rss = 0.0
        for key, value in dict_clusters.items():
            for elem in value:
                vector = dictWithDocIDandVector[elem]
                centroid = dict_centroid[key]
                for index in range(len(vector)):
                    rss += (vector[index] - centroid[index])**2
          
        return rss
    
    '''calculates the purity'''
    def calculate_purity(self, dict_clusters, totalNoOfDocs):
        purity = 1.0/float(totalNoOfDocs)
        sum = 0
        for key, value in dict_clusters.items():        
            classElementCountList = [0,0,0,0,0]
            for elem in value:
                doc_class = elem/30
                classElementCountList[doc_class] += 1
            sum += max(classElementCountList)   
        purity *= float(sum)
        return purity
    
    '''calculates the rand-index'''
    def calculate_rand_index(self, dict_clusters, totalNoOfDocs):
        rand_index = 0.0
        total_doc_pairs = (float(totalNoOfDocs) * float(totalNoOfDocs-1))/2.0
        print total_doc_pairs
        list_classElementCountList = []
        true_positives = 0.0
        for key, value in dict_clusters.items():
            classElementCountList = [0,0,0,0,0]
            for elem in value:
                doc_class = elem/30
                classElementCountList[doc_class] += 1
            list_classElementCountList.append(classElementCountList)
            for index in range(len(classElementCountList)):
                number = classElementCountList[index]
                if number > 1:
                    true_positives += float(number * (number-1))/2.0
        
        true_negatives = 0
        for i in range(len(list_classElementCountList)-1):
            classElemCountList = list_classElementCountList[i]
            j = i
            while j < (len(list_classElementCountList)-1):
                j += 1
                classElemCountList_dash = list_classElementCountList[j]
                for k in range(len(classElemCountList)):
                    for m in range(len(classElemCountList)):
                        if k == m:
                            continue
                        else:
                            true_negatives += classElemCountList[k] * classElemCountList_dash[m]
        print true_positives, true_negatives
                    
        rand_index = float(true_positives + true_negatives)/float(total_doc_pairs)
        return rand_index
    
    '''maps the docID with the corresponding title of the doc'''
    def map_docID_title(self, dict_clusters):
        dict_clusters_titles = {}
        query_list = ['texas aggies','texas longhorns','duke blue devils','dallas cowboys','dallas mavericks']
        file_titles = open("titles_part1.txt", "rb")
        documents = file_titles.readlines()
        for key, value in dict_clusters.items():
            dict_clusters_titles[key] = []
            for elem in value:
                str = query_list[elem/30] + ': ' + documents[elem]
                dict_clusters_titles[key].append(str)
        return dict_clusters_titles
        
                      
if __name__ == '__main__':
    docCluster = DocCluster()
    
    '''for creating the repository please uncomment the following line'''
    #docCluster.create_repository()
    
    tokenized_doc_list = docCluster.tokenize_documents()
    unique_token_vector = docCluster.create_unique_token_vector(tokenized_doc_list)
    dict = docCluster.create_dictionary(tokenized_doc_list)
    tfidfDict = docCluster.updateDictWithIDF(dict, tokenized_doc_list)
    normalizedDict = docCluster.calculateNormalizationFactorForEachDoc(tfidfDict)
    updatedDict = docCluster.updateDictWithDocNormFactor(tfidfDict, normalizedDict)
    finalDict = docCluster.createDictWithDocIDandTermsWithTheirTFIDFs(updatedDict)
    dictWithDocIDandVector = docCluster.createDictDocIDandVector(finalDict, unique_token_vector)
    totalNoOfDocs = len(dictWithDocIDandVector)
    dict_clusters = docCluster.implement_kMeans(dictWithDocIDandVector, 5)
    dict_clusters_titles = docCluster.map_docID_title(dict_clusters[0])
    sorted_clusters_titles_list = sorted(dict_clusters_titles)
    
    for elem in sorted_clusters_titles_list:
        print '###################',elem,'####################'
        for item in dict_clusters_titles[elem]:
            print item

    
    ############## rand_value calculation for different values of K ################
    """
    k = 2
    while k < 20:   
        dict_clusters = docCluster.implement_kMeans(dictWithDocIDandVector, k)
        rand_index_value = docCluster.calculate_rand_index(dict_clusters[0], totalNoOfDocs)
        
        print k, '----->', rand_index_value
        k += 1
    """
    
    ############## RSS_value calculation for different values of K ################
    """
    k = 2
    while k < 20:   
        dict_clusters = docCluster.implement_kMeans(dictWithDocIDandVector, k)
        rss_value = docCluster.calculate_RSS(dict_clusters[0], dictWithDocIDandVector, dict_clusters[1])        
        print k, '----->', rss_value
        k += 1
    """
    
    ############## purity calculation for different values of K ################
    """
    k = 2
    while k < 20:   
        dict_clusters = docCluster.implement_kMeans(dictWithDocIDandVector, k)
        rss_value = docCluster.calculate_purity(dict_clusters[0], totalNoOfDocs)        
        print k, '----->', rss_value
        k += 1
    """
    
    ####################### RSS Vs K Plot #################################################
    """
    x_axis_points = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    y_axis_points = [142.436786376,141.087381226,139.05463849,136.204594576,132.845761212,\
                     133.117348932,129.913731323,127.718385384,127.409610678,125.760606933,\
                     120.775788347,123.963804416,119.288753177,123.218667074,116.908000654,
                     114.033019169,114.725453336,112.407931076]
    
    figure()
    plot(x_axis_points, y_axis_points, "K")
    ylabel('RSS-values')
    xlabel('K-values')
    title('RSS vs K Plot')
    grid(True)
    show()
    """
    
    ###################### Purity Vs K Plot #################################################
    """
    x_axis_points = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    y_axis_points = [0.34,0.333333333333,0.553333333333,0.506666666667,\
                     0.786666666667,0.54,0.706666666667,0.646666666667,0.54,0.76,0.646666666667,\
                     0.806666666667,0.706666666667,0.733333333333,0.8,0.78,0.64,0.84]
    figure()
    plot(x_axis_points, y_axis_points, "K")
    ylabel('Purity-values')
    xlabel('K-values')
    title('Purity vs K Plot')
    grid(True)
    show()
    """
    
    ###################### Rand Index Vs K Plot #############################################
    """
    x_axis_points = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    y_axis_points = [0.515973154362,0.580581655481,0.663624161074,0.714630872483,0.771901565996,\
                     0.773422818792,0.78711409396,0.777807606264,0.804026845638,0.798747203579,\
                     0.807516778523,0.789530201342,0.815928411633,0.803400447427,0.805458612975,\
                     0.803400447427,0.814675615213,0.823713646532]
    figure()
    plot(x_axis_points, y_axis_points, "K")
    ylabel('Rand Index-values')
    xlabel('K-values')
    title('Rand Index vs K Plot')
    grid(True)
    show()
    """
    

    
    
    
  
    
    
    
    
    
    
    
    