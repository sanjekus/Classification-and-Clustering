# @author: Sanjeev Kumar Singh(UIN: 922002623)
# Created as part of Assignment-3 of Information Retrieval course(CSCE-670)
# filename: part3_classifier.py

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
                f = open("entertainment.txt", "wb")
            elif category_count == 1:
                f = open("business.txt", "wb")
            else:
                f = open("politics.txt", "wb")
                
            for query in query_list:
                query_results_1 = self.search_Bing(query, 0, category) 
                query_results_2 = self.search_Bing(query, 15, category)
                query_results_3 = self.search_Bing(query, 30, category)
                
                url_list = []
                for result in query_results_1:
                    if result['Url'] not in url_list:
                        url_list.append(result['Url'])
                        description = result['Description'].encode('ascii', 'ignore')
                        title = result['Title'].encode('ascii', 'ignore')
                        f.write(description + title + "\n")
                
                for result in query_results_2:
                    if result['Url'] not in url_list:
                        url_list.append(result['Url'])
                        description = result['Description'].encode('ascii', 'ignore')
                        title = result['Title'].encode('ascii', 'ignore')
                        f.write(description + title + "\n")
                        
                for result in query_results_3:
                    if len(url_list) < 30:
                        if result['Url'] not in url_list:
                            url_list.append(result['Url'])
                            description = result['Description'].encode('ascii', 'ignore')
                            title = result['Title'].encode('ascii', 'ignore')
                            f.write(description + title + "\n")  
                    
            f.close() 
            category_count += 1       
        return 
    
    '''lists the stop words'''
    def list_stop_words(self):
        stopWords = []
        f = open("stop_words.txt", "rb")
        tokens = f.readlines()
        for token in tokens:
            stopWords.extend(self.split_text(token))
        return stopWords
    
    ''' creates global dictionary'''
    def create_dictionary(self):
        dict = {}
        category_list = ['entertainment', 'business', 'politics']
        
        category_count = 0
        for category in category_list:
            if category_count == 0:
                f = open("entertainment.txt", "rb")
            elif category_count == 1:
                f = open("business.txt", "rb")
            else:
                f = open("politics.txt", "rb")
   
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
    
    '''dictionary after excluding stop words'''
    def dict_withoutStopWords(self, corpus_dict, stopWords):
        dict_withoutStopWords = {}
        for key, value in corpus_dict.items():
            temp_dict = {}
            for k,v in value.items():  
                if k not in stopWords:
                    temp_dict[k] = v
            dict_withoutStopWords[key] = temp_dict
                
        return dict_withoutStopWords
    

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
    
    def calculate_mutual_information(self, corpus_dict, dict_trainingSetDocs):
        dict_mutual_information = {}
        classes = ['entertainment', 'business', 'politics']
        
        for key, value in corpus_dict.items():
            tokens = []
            for key_dash, value_dash in corpus_dict.items():
                for item in value_dash.keys():
                        tokens.append(item)
                tokens.extend(value_dash.keys())
                
            for token in tokens:
                N11 = 0.0
                N10 = 0.0
                N01 = 0.0
                N00 = 0.0
                    
                for k, v in dict_trainingSetDocs.items():
                    if k.find(key) == 0:
                        for elem in v:
                            if token in elem:
                                N11 += 1.0
                            else:
                                N01 += 1.0
                    else:
                        for elem in v:
                            if token in elem:
                                N10 += 1.0
                            else:
                                N00 += 1.0
                                
                N = N11 + N10 + N01 + N00
                N1 = N11 + N10
                N_dot_1 = N11 + N01
                N0 = N01 + N00
                N_dot_0 = N10 + N00
                
                if N11 != 0.0:
                    term1 = (N11/N) * math.log((N * N11)/(N1 * N_dot_1), 2)  
                else:
                    term1 = 0.0    
                if N01 != 0:
                    term2 = (N01/N) * math.log((N * N01)/(N0 * N_dot_1), 2)
                else:
                    term2 = 0.0
                if N10 != 0.0:
                    term3 = (N10/N) * math.log((N * N10)/(N1 * N_dot_0), 2)
                else:
                    term3 = 0.0
                if N00 != 0.0:
                    term4 = (N00/N) * math.log((N * N00)/(N0 * N_dot_0), 2)
                else:
                    term4 = 0.0
  
                mutual_information = term1 + term2 + term3 + term4
                temp_dict = {}
                if token in dict_mutual_information:
                    temp_dict = dict_mutual_information[token]
                    temp_dict[key] = mutual_information
                else:
                    temp_dict[key] = mutual_information
                    
                dict_mutual_information[token] = temp_dict
                
        return dict_mutual_information
    
    '''selects the appropriate features out of the whole lot of features'''
    def feature_selection(self, dict_withoutStopWords, dict_mutual_information):
        dict = {}
        dict['entertainment'] = {}
        dict['business'] = {}
        dict['politics'] = {}
        
        for key, value in dict_mutual_information.items():
            for k, v in value.items():
                dict[k][key] = v
                
        entertainment_tuples = sorted(dict['entertainment'].items(), key=lambda x: x[1])
        business_tuples = sorted(dict['business'].items(), key=lambda x: x[1])
        politics_tuples = sorted(dict['politics'].items(), key=lambda x: x[1])
            
        dict_terms = {}
        dict_terms['entertainment'] = []
        dict_terms['business'] = []
        dict_terms['politics'] = []
        
        for index in range(500):
            dict_terms['entertainment'].append(entertainment_tuples[index][0])
            
        for index in range(500):
            dict_terms['business'].append(business_tuples[index][0])
            
        for index in range(500):
            dict_terms['politics'].append(politics_tuples[index][0])
            
        updated_dict_dict_withoutStopWords = {}
        for key, value in dict_withoutStopWords.items():
            temp_dict = {}
            for k, v  in value.items():
                if k not in dict_terms[key]:
                    temp_dict[k] = v
                    
            updated_dict_dict_withoutStopWords[key] = temp_dict
            
    
        return updated_dict_dict_withoutStopWords
    
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
    stopWords = docClassifier.list_stop_words()
    dict_withoutStopWords = docClassifier.dict_withoutStopWords(corpus_dict, stopWords)
    dict_titles = docClassifier.map_docID_title()
    dict_trainingSetDocs = docClassifier.dict_trainingSetDocs()
    dict_mutual_information = docClassifier.calculate_mutual_information(dict_withoutStopWords, dict_trainingSetDocs)
    updated_dict_dict_withoutStopWords = docClassifier.feature_selection(dict_withoutStopWords, dict_mutual_information)
    noOfUniqueTokens = docClassifier.compute_noOfUniqueTokens(updated_dict_dict_withoutStopWords)
    dict_noOfTokens = docClassifier.dict_noOfTokens(updated_dict_dict_withoutStopWords)
    classifiedDocs_dict = docClassifier.classify_documents(noOfUniqueTokens, updated_dict_dict_withoutStopWords, dict_noOfTokens, dict_trainingSetDocs)
    
    for key, value in classifiedDocs_dict.items():
        print key, ":"
        print
        for k, v in value.items():
            for elem in v:
                print k, ": ", dict_titles[key][elem]
    
                
    print
    print "################### Final/Aggregate Confusion Matrix ###########################"
    microaveraged_F = docClassifier.calculate_microaveraging_parameters(classifiedDocs_dict)
    for key, value in microaveraged_F[0].items():
        print key, value
    print
    print "Microaveraged F1: ", microaveraged_F[1]
    
    
    



    
        

    
    
    
    
    
        

    

    
        