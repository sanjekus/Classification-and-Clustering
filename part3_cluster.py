# @author: Sanjeev Kumar Singh(UIN: 922002623)
# Created as part of Assignment-3 of Information Retrieval course(CSCE-670)
# filename: part3_cluster.py

from pybing import Bing
import httplib
import urllib, urllib2
import json
import re
import math
import numpy as np
from pylab import *
import time
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
        f = open("repository_part3_clustering.txt", "wb")
        f_titles = open("part3_clustering_titles.txt", "wb")
        query_list = [['texas', 'aggies'], ['texas', 'longhorns'], ['duke', 'blue', 'devils'],\
                      ['dallas', 'cowboys'], ['dallas', 'mavericks']]
        
        for query in query_list:
            query_results_1 = self.search_Bing(query, 0)
            query_results_2 = self.search_Bing(query, 15)
            query_results_3 = self.search_Bing(query, 30)
            
            url_list = []
            for result in query_results_1:
                if result['Url'] not in url_list:
                    url_list.append(result['Url'])
                    description = result['Description']
                    title = result['Title']
                    f.write(description + title + "\n")
                    f_titles.write(title + "\n")
        
            for result in query_results_2:
                if result['Url'] not in url_list:
                    url_list.append(result['Url'])
                    description = result['Description']
                    title = result['Title']
                    f.write(description + title + "\n") 
                    f_titles.write(title + "\n")
            
            for result in query_results_3:
                if len(url_list) < 30:
                    if result['Url'] not in url_list:
                        url_list.append(result['Url'])
                        description = result['Description']
                        title = result['Title']
                        f.write(description + title + "\n") 
                        f_titles.write(title + "\n")
                else:
                    break
        f.close() 
        f_titles.close()       
        return
  
    ''' tokenizes each of the documents and stores in a list '''
    def tokenize_documents(self):
        tokenized_doc_list = []
        f = open("repository_part3_clustering.txt", "rb")
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
    
    def dict_withoutStopWords(self, dict, stopWords):
        dict_withoutStopWords = {}
        for key, value in dict.items():
            if key not in stopWords:
                dict_withoutStopWords[key] = value
                
        return dict_withoutStopWords
    
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
    
    ''' lists stop words '''
    def list_stop_words(self):
        stopWords = []
        f = open("stop_words.txt", "rb")
        tokens = f.readlines()
        for token in tokens:
            stopWords.extend(self.split_text(token))
        return stopWords
    
    '''tuple with docID and distance from center for each vector for finding initial seeds'''
    def initial_seed_selection(self,dictWithDocIDandVector):
        dict_distanceFromCenter = {}
        for key, value in dictWithDocIDandVector.items():
            distance = 0.0
            for elem in value:
                distance += (elem)**2
            dict_distanceFromCenter[key] = distance
            
        tuple_docID_distanceFromCenter = sorted(dict_distanceFromCenter.items(), key=lambda x: x[1])
        return tuple_docID_distanceFromCenter
    
    ''' implementation of k-means algorithm '''
    def implement_kMeans(self, dictWithDocIDandVector, tuple_docID_distanceFromCenter, K):
        dict_clusters = {}
        dict_centroid = {}
        initial_dict_clusters = {}
        steady_state_count = 0
        
        total_docs = len(tuple_docID_distanceFromCenter)
        initial_seed_list = []
        initial_seed_list.append(tuple_docID_distanceFromCenter[0][0])
        initial_seed_list.append(tuple_docID_distanceFromCenter[total_docs-1][0])
        
        increment_factor = total_docs/(K-1)
        if K > 2:
            for i in range(K-2):
                initial_seed_list.append((i+1)*increment_factor)
        
        index = 0        
        for seed in initial_seed_list:
            str = 'cluster_%d' %(index)
            dict_centroid[str] = dictWithDocIDandVector[seed]
            index += 1
        
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
    
    
    def calculate_RSS(self, dict_clusters, dictWithDocIDandVector, dict_centroid):
        rss = 0.0
        for key, value in dict_clusters.items():
            for elem in value:
                vector = dictWithDocIDandVector[elem]
                centroid = dict_centroid[key]
                for index in range(len(vector)):
                    rss += (vector[index] - centroid[index])**2
          
        return rss

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
    
    def calculate_rand_index(self, dict_clusters, totalNoOfDocs):
        rand_index = 0.0
        total_doc_pairs = (float(totalNoOfDocs) * float(totalNoOfDocs-1))/2.0
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
                    
        rand_index = float(true_positives + true_negatives)/float(total_doc_pairs)
        return rand_index
    
    ''' maps the docID with the title of the doc'''
    def map_docID_title(self, dict_clusters):
        dict_clusters_titles = {}
        query_list = ['texas aggies','texas longhorns','duke blue devils','dallas cowboys','dallas mavericks']
        file_titles = open("part3_clustering_titles.txt", "rb")
        documents = file_titles.readlines()
        for key, value in dict_clusters.items():
            dict_clusters_titles[key] = []
            for elem in value:
                str = query_list[elem/30] + ': ' + documents[elem]
                dict_clusters_titles[key].append(str)
        return dict_clusters_titles
        
                          
if __name__ == '__main__':
    docCluster = DocCluster()
    ''' uncomment the following line in order to create repository'''
    #docCluster.create_repository()
    tokenized_doc_list = docCluster.tokenize_documents()
    unique_token_vector = docCluster.create_unique_token_vector(tokenized_doc_list)
    stopWords = docCluster.list_stop_words()
    dict = docCluster.create_dictionary(tokenized_doc_list)
    dict_withoutStopWords = docCluster.dict_withoutStopWords(dict, stopWords)
    tfidfDict = docCluster.updateDictWithIDF(dict_withoutStopWords, tokenized_doc_list)
    stopWords = docCluster.list_stop_words()
    normalizedDict = docCluster.calculateNormalizationFactorForEachDoc(tfidfDict)
    updatedDict = docCluster.updateDictWithDocNormFactor(tfidfDict, normalizedDict)
    finalDict = docCluster.createDictWithDocIDandTermsWithTheirTFIDFs(updatedDict)
    dictWithDocIDandVector = docCluster.createDictDocIDandVector(finalDict, unique_token_vector)
    totalNoOfDocs = len(dictWithDocIDandVector)
    tuple_docID_distanceFromCenter = docCluster.initial_seed_selection(dictWithDocIDandVector)
    dict_clusters = docCluster.implement_kMeans(dictWithDocIDandVector, tuple_docID_distanceFromCenter, 5)
    dict_clusters_titles = docCluster.map_docID_title(dict_clusters[0])
    sorted_clusters_titles_list = sorted(dict_clusters_titles)
    
    for elem in sorted_clusters_titles_list:
        print '###################',elem,'####################'
        for item in dict_clusters_titles[elem]:
            print item
    
     
    ############## RSS_value calculation for different values of K ################
    """
    k = 2
    while k < 20:   
        dict_clusters = docCluster.implement_kMeans(dictWithDocIDandVector,tuple_docID_distanceFromCenter,k)
        rss_value = docCluster.calculate_RSS(dict_clusters[0], dictWithDocIDandVector, dict_clusters[1])        
        print k, '----->', rss_value
        k += 1
    """
    ############################# Calculate Rand Index for different K values ##########################
    """
    k = 2
    while k < 20:   
        dict_clusters = docCluster.implement_kMeans(dictWithDocIDandVector, tuple_docID_distanceFromCenter, k)
        rand_index_value = docCluster.calculate_rand_index(dict_clusters[0], totalNoOfDocs)
        print k, '----->', rand_index_value
        k += 1
    """
    ############################# Calculate Purity for different K values ##########################
    """
    k = 2
    while k < 20:   
        dict_clusters = docCluster.implement_kMeans(dictWithDocIDandVector, tuple_docID_distanceFromCenter, k)
        purity_value = docCluster.calculate_purity(dict_clusters[0], totalNoOfDocs)
        
        print k, '----->', purity_value
        k += 1
    """
    ############################# Plot Rand Index Vs K Graph ###################################
    """
    x_axis_points = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    y_axis_points = [0.525279642058,0.699418344519,0.695659955257,0.745592841163,0.808859060403,\
                     0.798031319911,0.802416107383,0.819597315436,0.805369127517,0.814675615213,\
                     0.798120805369,0.803937360179,0.823624161074,0.809843400447,0.812975391499,\
                     0.808769574944,0.806979865772,0.806711409396]
            
    figure()
    plot(x_axis_points, y_axis_points, "K")
    ylabel('Rand-Index-values')
    xlabel('K-values')
    title('Rand Index vs K Plot(Modified Version)')
    grid(True)
    show()
    """
    
    ############################# Plot Purity Vs K Graph ###################################
    """
    x_axis_points = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    y_axis_points = [0.353333333333,0.526666666667,0.453333333333,0.566666666667,\
                     0.68,0.666666666667,0.68,0.686666666667,0.673333333333,0.713333333333,0.66,\
                     0.666666666667,0.746666666667,0.726666666667,0.74,0.733333333333,0.706666666667,0.713333333333]
    figure()
    plot(x_axis_points, y_axis_points, "K")
    ylabel('Purity-values')
    xlabel('K-values')
    title('Purity vs K Plot(Modified Version)')
    grid(True)
    show()
    """
    ####################### Plot RSS Vs K Graph #################################################
    """
    x_axis_points = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    y_axis_points = [144.723490148,142.090434323,141.504174587,139.826533325,137.183189416,\
                     135.805544854,135.385646872,133.976840381,132.319657781,130.383344683,\
                     130.392203202,128.882388544,126.406264938,126.804735614,125.547087124,\
                     122.127311934,121.300002195,120.281894166]
    
    figure()
    plot(x_axis_points, y_axis_points, "K")
    ylabel('RSS-values')
    xlabel('K-values')
    title('RSS vs K Plot(Modified Version)')
    grid(True)
    show()
    """
    
    
  
    
    
    
    
    
    
    
    