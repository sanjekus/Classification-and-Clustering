# @author: Sanjeev Kumar Singh(UIN: 922002623)
# Created as part of Assignment-3 of Information Retrieval course(CSCE-670)
# filename: part2.py

from pybing import Bing
import httplib
import urllib, urllib2
import json
import re
import math
import random
import numpy as np

class DocClassifier:
    username = 'ffdcc815-e721-48cd-b32a-692afcf2d343'
    accountKey='Qnw3YkoQdoRjdOPbXPA4oph8C160JGM5QKQjqQYYO6E='
    
    '''splits the text and returns the tokens'''  
    def split_text(self, text):
        text = re.compile("[^\w]|_").sub(" ", text)
        word_list = re.findall("\w+", text.lower())    
        return word_list
    
    '''makes a bing search query for given terms with the skip and category parameter'''
    def search_Bing(self, termsList, skip, category_parameter):
        queryBingFor = "'" + ' '.join(termsList) + "'"
        quoted_query = urllib.quote(queryBingFor)
    
        rootURL = "https://api.datamarket.azure.com/Data.ashx/Bing/Search/v1/"
        searchURL = rootURL + "News?Query=" + quoted_query + "&$format=json" + category_parameter + "&$skip=%d" %(skip)
    
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
        query_list = ['bing','amazon','twitter','yahoo','google',\
                      'beyonce','bieber','television','movies','music',\
                      'obama','america','congress','senate','lawmakers']
        
        category_list = ['&NewsCategory=%27rt_Entertainment%27',\
                         '&NewsCategory=%27rt_Business%27',\
                         '&NewsCategory=%27rt_Politics%27']
      
        category_count = 0
        for category in category_list:
            if category_count == 0:
                f = open("entertainment_part2.txt", "wb")
            elif category_count == 1:
                f = open("business_part2.txt", "wb")
            else:
                f = open("politics_part2.txt", "wb")
                
            for query in query_list:
                query_results_1 = self.search_Bing(query, 0, category) 
                query_results_2 = self.search_Bing(query, 15, category)
                
                for result in query_results_1:
                    description = result['Description'].encode('ascii', 'ignore')
                    title = result['Title'].encode('ascii', 'ignore')
                    f.write(description + title + "\n")
                
                for result in query_results_2:
                    description = result['Description'].encode('ascii', 'ignore')
                    title = result['Title'].encode('ascii', 'ignore')
                    f.write(description + title + "\n")  
                    
            f.close() 
            category_count += 1       
        return 
    
    ''' creates global dictionary'''
    def create_dictionary(self):
        dict = {}
        category_list = ['entertainment', 'business', 'politics']
        
        category_count = 0
        for category in category_list:
            if category_count == 0:
                f = open("entertainment_part2.txt", "rb")
            elif category_count == 1:
                f = open("business_part2.txt", "rb")
            else:
                f = open("politics_part2.txt", "rb")
   
            documents = f.readlines()
            tokens_list = []
            for document in documents:
                tokens_list.extend(self.split_text(document))
            
            dict[category] = tokens_list
            category_count += 1
            f.close()
            
        resulting_dict = {}
        for key, value in dict.items():
            temp_dict = {}
            for elem in value:
                if elem in temp_dict:
                    temp_dict[elem] += 1
                else:
                    temp_dict[elem] = 1
            resulting_dict[key] = temp_dict
        
        return resulting_dict
    

    ''' creates dictionary with key as the class and value as the number of tokens in that class'''
    def dict_noOfTokens(self, corpus_dict):
        dict = {}
        dict['entertainment'] = 0
        dict['business'] = 0
        dict['politics'] = 0
        
        for key, value in corpus_dict.items():
            for k, v in value.items():
                dict[key] += v    
        return dict
    
    ''' computes number of unique tokens/terms existing in the entire corpus'''
    def compute_noOfUniqueTokens(self, corpus_dict):
        totalNoOFUniqueTokens = 0
        for k, v in corpus_dict.items():
            totalNoOFUniqueTokens += len(v.keys())
            
        return totalNoOFUniqueTokens
    
    ''' populates the test set which contains documents that needs to be classfied'''
    def populate_testSet(self):
        query_list = ['apple', 'facebook', 'westeros', 'gonzaga', 'banana']
        category_list = ['&NewsCategory=%27rt_Entertainment%27',\
                         '&NewsCategory=%27rt_Business%27',\
                         '&NewsCategory=%27rt_Politics%27']
    
        category_count = 0
        for category in category_list:
            if category_count == 0:
                f = open("entertainment_testSet.txt", "wb")
                f_titles = open("entertainment_titles.txt", "wb")
            elif category_count == 1:
                f = open("business_testSet.txt", "wb")
                f_titles = open("business_titles.txt", "wb")
            else:
                f = open("politics_testSet.txt", "wb") 
                f_titles = open("politics_titles.txt", "wb")
                
            for query in query_list:
                query_results_1 = self.search_Bing(query, 0, category) 
                query_results_2 = self.search_Bing(query, 15, category)
                
                for result in query_results_1:
                    description = result['Description'].encode('ascii', 'ignore')
                    title = result['Title'].encode('ascii', 'ignore')
                    f.write(description + title + "\n")
                    f_titles.write(title + "\n")
                
                for result in query_results_2:
                    description = result['Description'].encode('ascii', 'ignore')
                    title = result['Title'].encode('ascii', 'ignore')
                    f.write(description + title + "\n")  
                    f_titles.write(title + "\n")
                
            f.close() 
            f_titles.close()
            category_count += 1  
                 
        return   
    
    def dict_trainingSetDocs(self):
        dict = {}
        category_list = ['entertainment_testSet', 'business_testSet', 'politics_testSet']
        
        '''initialize dictionary'''
        for category in category_list:
            dict[category] = []
        
        category_count = 0
        while category_count < 3:
            if category_count == 0:
                f = open("entertainment_testSet.txt", "rb")
            elif category_count == 1:
                f = open("business_testSet.txt", "rb")
            else:
                f = open("politics_testSet.txt", "rb")
              
            documents = f.readlines() 
            for document in documents:
                tokens = self.split_text(document)
                dict[category_list[category_count]].append(tokens)
                
            category_count += 1
            f.close() 
                 
        return dict
    
    '''maps the docID with the corresponding title of the doc'''
    def map_docID_title(self):
        dict_titles = {}
        file_titles = open("entertainment_titles.txt", "rb")
        documents = file_titles.readlines()
        file_titles.close()
        temp_dict = {}
        for index in range(len(documents)):
            temp_dict[index] = documents[index]
        dict_titles['entertainment_testSet'] = temp_dict
        
        file_titles = open("business_titles.txt", "rb")
        documents = file_titles.readlines()
        file_titles.close()
        temp_dict = {}
        for index in range(len(documents)):
            temp_dict[index] = documents[index]
        dict_titles['business_testSet'] = temp_dict
        
        file_titles = open("politics_titles.txt", "rb")
        documents = file_titles.readlines()
        file_titles.close()
        temp_dict = {}
        for index in range(len(documents)):
            temp_dict[index] = documents[index]
        dict_titles['politics_testSet'] = temp_dict
        
        return dict_titles
        
    
    '''classifies the documents'''
    def classify_documents(self, noOfUniqueTokens, corpus_dict, dict_noOfTokens, dict_trainingSetDocs):
        classifiedDocs_dict = {}
        classes = ['entertainment', 'business', 'politics']
        trainingSets = ['entertainment_testSet', 'business_testSet', 'politics_testSet']
        for trainingSet in trainingSets:
            classifiedDocs_dict[trainingSet] = {}
            for category in classes:
                classifiedDocs_dict[trainingSet][category] = []
                
        for key, value in dict_trainingSetDocs.items(): 
            docID = 0 
            for elem in value:     
                dict_classes = {}
                for ctgry in classes:
                    dict_classes[ctgry] = math.log(0.33,2)
                for category in classes:
                    for token in elem:
                        weight = 0.0
                        if token in corpus_dict[category]:
                            weight = float((float(corpus_dict[category][token]) + 1.0)/(float(noOfUniqueTokens) + float(dict_noOfTokens[category])))
                
                        if weight > 0.0:
                            weight = math.log(weight, 2)
                    
                        dict_classes[category] += weight
                
                keyWithMaxValue = min(dict_classes.iterkeys(), key=lambda k: dict_classes[k])
                classifiedDocs_dict[key][keyWithMaxValue].append(docID)
                docID += 1
    
        return classifiedDocs_dict
    
    ''' computes the confusion matrix and microaveraged F'''
    def calculate_microaveraging_parameters(self, classifiedDocs_dict):
        dict_microaveraged_parameters = {}
        dict_microaveraged_parameters['TP'] = 0
        dict_microaveraged_parameters['TN'] = 0
        dict_microaveraged_parameters['FP'] = 0
        dict_microaveraged_parameters['FN'] = 0
        
        for key, value in classifiedDocs_dict.items():
            for k, v in value.items():
                if key.find(k) == 0:
                    dict_microaveraged_parameters['TP'] += len(v)
                else:
                    dict_microaveraged_parameters['FP'] += len(v)
            
        dict_microaveraged_parameters['FN'] = dict_microaveraged_parameters['FP']
        dict_microaveraged_parameters['TN'] = 2 * dict_microaveraged_parameters['TP'] + dict_microaveraged_parameters['FP']
        
        precision = float(dict_microaveraged_parameters['TP'])/float(dict_microaveraged_parameters['TP'] + dict_microaveraged_parameters['FP'])
        recall = float(dict_microaveraged_parameters['TP'])/float(dict_microaveraged_parameters['TP'] + dict_microaveraged_parameters['FN'])
                
        microaveraged_F = float(2.0 * precision * recall)/float(precision + recall)
        return dict_microaveraged_parameters, microaveraged_F    

                
if __name__ == '__main__':
    docClassifier = DocClassifier()
    
    ''' populate training set '''
    #docClassifier.create_repository()
    ''' populate test set '''
    #docClassifier.populate_testSet()

    corpus_dict = docClassifier.create_dictionary()
    dict_titles = docClassifier.map_docID_title()
    dict_noOfTokens = docClassifier.dict_noOfTokens(corpus_dict)
    noOfUniqueTokens = docClassifier.compute_noOfUniqueTokens(corpus_dict)
    dict_trainingSetDocs = docClassifier.dict_trainingSetDocs()
    classifiedDocs_dict = docClassifier.classify_documents(noOfUniqueTokens, corpus_dict, dict_noOfTokens, dict_trainingSetDocs)
    
    for key, value in classifiedDocs_dict.items():
        print key, ":"
        print
        for k, v in value.items():
            for elem in v:
                print k, ": ", dict_titles[key][elem]
                
    print
    print "################### The final/aggregate Confusion Matrix ###########################"
    microaveraged_F = docClassifier.calculate_microaveraging_parameters(classifiedDocs_dict)
    for key, value in microaveraged_F[0].items():
        print key, value
    print
    print "Microaveraged F1: ", microaveraged_F[1]



    
        

    
    
    
    
    
        

    

    
        