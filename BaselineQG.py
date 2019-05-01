# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Python Libraries
import string
from string import digits
import keras
import tensorflow as tf
import json
import nltk
import requests
import pandas as pd
import numpy as np
import pycorenlp
from pycorenlp import StanfordCoreNLP
import re
import os
import sys
import pickle
import networkx as nx
import wikipediaapi
import operator
import math


# Self-Made Files
import classes
import relation_standardizer

# import node
# nltk.download('punkt')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Function Defs

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""
Prepare the content of a text passage
 - Separate into sentences
 - Lower case, remove quotes, remove punctuation, remove extra spaces
"""""""""""""""

# For removing punctuation
remove_punc = str.maketrans('', '', string.punctuation)
# For removing digits
remove_digits = str.maketrans('', '', digits)

def prepContent(content):
    
    temp = nltk.sent_tokenize(content)
    
    '''
    Prepping the sentences
    '''
    # Lower-cases
    temp = [x.lower() for x in temp]
    
    # Remove quotes
    temp = [re.sub("'", '', x) for x in temp]
    
    # Remove punctuation / digits
    temp = [x.translate(remove_punc) for x in temp]
    #temp = [x.translate(remove_digits) for x in temp]
    
    # Remove spaces
    temp = [x.strip() for x in temp]
    temp = [re.sub(" +", " ", x) for x in temp] 

    return temp
    

"""""""""""""""
Loads pre-processed Georgetown data
"""""""""""""""

def load_GU_Data():

    # Location of Scipt being run
    abspath = sys.path[0]
    
    # Loading datasets previously constructed
    G_augment =         pickle.load(open(abspath + "/Other-Data/G_augmented.pkl", "rb"))
    cleaned_tuples =    pickle.load(open(abspath + "/Other-Data/cleaned_tuples.pkl", "rb"))
    topic_graph =       pickle.load(open(abspath + "/Other-Data/Topic_Graph.pkl", "rb"))
    encoded_tuples =    pickle.load(open(abspath + "/Other-Data/relation_tuples.pkl", "rb"))
    
    return G_augment, cleaned_tuples, topic_graph, encoded_tuples


# I got this somewhere on Wiki and can't find the page again.  When I can I'll
# expand this so a new topic can automatically be downloaded.
    
def GU_pages():
    
    return [
        'Campuses_of_Georgetown_University',
        'Category:Georgetown_University_schools',
        'Category:Georgetown_University_programs',
        'Category:Georgetown_Hoyas',
        'Category:Georgetown_University_student_organizations',
        'Category:Georgetown_University_buildings',
        'Category:Georgetown_University_publications',
        'Category:Georgetown_University_people',
        'History_of_Georgetown_University',
        'Hoya_Saxa',
        'Housing_at_Georgetown_University',
        'Category:Georgetown_University_templates',
        'Georgetown_University_Library',
        'Georgetown_University_Alma_Mater',
        'President_and_Directors_of_Georgetown_College',
        'Category:Georgetown_University_Medical_Center',
        'Energy:_A_National_Issue',
        'Center_for_Strategic_and_International_Studies',
        'Georgetown_University',
        'Georgetown_University_Police_Department',
        '1838_Georgetown_slave_sale',
        'St._Thomas_Manor',
        'Anne_Marie_Becraft',
        'List_of_Georgetown_University_commencement_speakers',
        'Bishop_John_Carroll_(statue)',
        '2019_college_admissions_bribery_scandal'
        ]



# Recursive method to retrieve all the pages in a category page of wikipedia
# inlcusive of nested category pages.

def get_all_children_pages(categories, wiki_wiki):
    good_list = []
    input_dict = categories.categorymembers
    keys = input_dict.keys()
    
    for key in keys:
        
        if "Category" in key:
            
            # I'm not taking the alumni pages specifically to reduce the #
            # of pages.  This reduced it slightly more than 1/2.
            if "alumni" not in key:
                temp = get_all_children_pages(wiki_wiki.page(key), wiki_wiki)
                good_list += temp
            
        else:
            
            good_list.append(key)
            
    return good_list


def get_pages(Page_List, wiki_wiki):
    individual_pages = []
    Topic_Dict = {}

    # Builds the list
    for item in Page_List:
        if "Category" in item:
            output_list = get_all_children_pages(wiki_wiki.page(item), wiki_wiki)
            individual_pages += output_list
            
        else:
            individual_pages.append(item)
    
    # Puts the pages into a dictionary for easy access.
    for page in individual_pages:
        Topic_Dict[page] = wiki_wiki.page(page)

    return individual_pages, Topic_Dict

G = G_augment

G_augment[topic]

def determine_node_knowledge(current_node, G, sess, critic):
    
    connecting_nodes = list(G.adj[topic])

    # A single name of a node
    # node = connecting_nodes[0]
    node_info = []

    for node in connecting_nodes:
        
        encodes = G.nodes[node]['encodes']
        
        pred_rewards = sess.run(critic.layer3, feed_dict = {critic.observation: np.reshape(encodes,(-1, 768))})
        
        node_info.append([node, np.mean(pred_rewards), np.std(pred_rewards)])
        
    return node_info

def explore_new_page(current_node, bc, nlp, wiki_wiki, G):
    
    topic_page = wiki_wiki.page(current_node)
    sentences = prepContent(topic_page.text)
    
    _relation_tuples = []
    _cleaned_tuples = []
    _encodes = []
    
    print('Currently processing page: ' + current_node)
    
    for sentence in sentences:
        
        try:
    
            results = nlp.annotate(sentence, properties = {'annotators': 'openie', 'outputFormat': 'json'})
            sub_result = results['sentences'][0]['openie']
            
            for i in range(len(sub_result)):
                _relation_tuples.append([sub_result[i]['subject'], sub_result[i]['relation'], sub_result[i]['object']])
                
        except:
            
            print("Something Happened.  Booooooooo!!!!")


    for tup in _relation_tuples:
        relation = rs.standardize(tup[1])
        
        if relation != 'null':
            _cleaned_tuples.append([tup[0], relation, tup[2]])

    # Need to add code to create encodes from BERT
    
    for relation in _cleaned_tuples:

        _encodes.append([relation[1], bc.encode([relation[0] + ' ||| ' + relation[2]])]) 
        
    # Need to add node to G along with summary, relation_tuples, and encodes
    
    G_relations = []
    G_encodes = []
    
    node = G.nodes[current_node]
    
    try:
        sentences = prepContent(node['summary'])
        
        for sentence in sentences:
            try:
                results = nlp.annotate(sentence, properties = {'annotators': 'openie', 'outputFormat': 'json'})
                sub_result = results['sentences'][0]['openie']
                for i in range(len(sub_result)):
                    G_relations.append([sub_result[i]['subject'], sub_result[i]['relation'], sub_result[i]['object']])
            except: 
                print("Problem in Core-NLP.  Skipping")
            
        if len(G_relations) > 10:
            G_relations = random.sample(G_relations, 10)
            
        for relation in G_relations:
            try:
                G_encodes.append(bc.encode([relation[0] + " ||| " + relation[2]]))
            except:
                print("Problem in bc encoding.  Skipping")
    except:
        print("Problem in Summary.  Leaving empty.")
        
            
    node['relations'] = G_relations
    node['encodes'] = G_encodes
        
    return _cleaned_tuples, _encodes
    
    
    
    
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Main Program

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def run(args):

    """""""""""""""""""""""""""""""""
    
    Set up
    
    """""""""""""""""""""""""""""""""
    
    # topic = "Georgetown_University"
    topic =         args.topic
    actor_lr =      args.actor_LR
    critic_lr =     args.critic_LR
    episodes =      args.eps
    
    topic =         "Georgetown_University"
    actor_lr =      0.001
    critic_lr =     0.001
    episodes =      10
    
    buffer = classes.ReplayBuffer()
    wiki_wiki = wikipediaapi.Wikipedia('en')
    rs = relation_standardizer.Relation_Standardizer("Bob")
    
    # TODO: Future add code to let a person choose another topic

    G_augment, cleaned_tuples, topic_graph, encoded_tuples = load_GU_Data()
    page_list = GU_pages()
    buffer.relations = encoded_tuples
    individual_pages, Topic_Dict = get_pages(page_list, wiki_wiki)
        
    n_features =    768     # Default used in BERT
    n_output =      25      # Num possilbe relations.  Currently set to 25
    actor_H1 =      768     # Num of hidden units in first layer of actor
    actor_H2 =      768     # Num of hidden units in second layer of actor
    critic_H1 =     768     # Num of hidden units in first layer of critic
    critic_H2 =     768     # Num of hidden units in second layer of critic
    
    # TensorFlow Setup and Initialization
    tf.reset_default_graph()    
    
    actor = classes.Actor(n_features, n_output, actor_lr, actor_H1, actor_H2)
    critic = classes.Critic(n_features, critic_lr, critic_H1, critic_H2)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # BERT Setup
    print("In a terminal window Python Environment can access, run the following:\n" + "bert-serving-start -model_dir ~/Final_Proj/BERT-Data/ -num_worker=2\n\nPress Enter when done.")
    x = input()
    
    from bert_serving.client import BertClient
    bc = BertClient()
    
    # Core-NLP Setup
    print("In a terminal window run the following:\ncd ~/Final_Proj/Stan-Core-NLP; java -mx6g -cp \"*\" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000\n\nPress Enter when done.")
    x = input()
    
    nlp = StanfordCoreNLP('http://localhost:9000')
    
    current_node = topic
    
    
    """""""""""""""""""""""""""""""""
    
    Running Episodes
    
    """""""""""""""""""""""""""""""""

    for episode in range(len(episodes)):
        
        relations = []
        probs = []
        chosen = []
        rewards = []
        states = []
        pred_rewards = []
        td_error = []
        
        # Run the Training Routine for the Critic
        training_sample = buffer.sample(20)
        
        # sample = training_sample[0]
        
        # Use the Actor to determine the predicted relation for a state
        for sample in training_sample:
            relations.append(rs.relation_to_int(sample[0]))
            states.append(sample[1])
            probs.append(sess.run(actor.layer3, feed_dict = {actor.observation: sample[1]}))

        # Formatting the probabilities to make them easier to use
        for prob in probs:
            prob_list = prob[0].tolist()
            chosen.append(prob_list.index(max(prob_list)))

        # Determine reward from the environment       
        for actual, pred in zip(relations, chosen):
            if actual == pred:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        
        # Training the Critic
        loss, _ = sess.run([critic.loss, critic.train], feed_dict = {critic.observation: np.reshape(states,(-1, 768)), critic.reward: np.reshape(rewards, (-1,1))})
        print("Training loss for critic is: " + str(loss))
        
        ######
        # Exploration code
        ######
        
        # Run the links available for the current node through the critic to get lowest mean
        # The std is also included if we want to include a LCB version later.
        node_predictions = determine_node_knowledge(current_node, G_augment, sess, critic)
        
        # Filter out nan entries & sort
        filtered = [x for x in node_predictions if not math.isnan(x[1])]
        filtered.sort(key = lambda x: x[1])

        # Determine the next node to go to
        for node in filtered:
            if node[0] in individual_pages:
                # print("Already Explored")
            else:
                # print("It's good to go.")
                current_node = node[0]
                individual_pages.append(current_node)
                
                break
        
        # Explore page 
        clean_tuples, encodes = explore_new_page(current_node, bc, nlp, wiki_wiki, G_augment)
        
        # Add encoded tuples to the replay buffer
        buffer.relations += encodes
        
        # Add cleaned_tuples to the that list
        
        
        
        relations = []
        states = []
        chosen = []
        probs = []
        
        # Gather info for training the Actor
        for encode in encodes:
            relations.append(rs.relation_to_int(encode[0]))
            states.append(encode[1])
            probs.append(sess.run(actor.layer3, feed_dict = {actor.observation: encode[1]}))

        
        # MOVE CRITIC PRED HERE and DETERMINE
        pred_rewards = sess.run(critic.layer3, feed_dict = {critic.observation: np.reshape(states,(-1, 768))})
        pred_rewards = [item for sublist in pred_rewards for item in sublist]
        
        '''
        # MOVE CRITIC PRED HERE and DETERMINE
        pred_rewards = sess.run(critic.layer3, feed_dict = {critic.observation: np.reshape(states,(-1, 768))})
        pred_rewards = [item for sublist in pred_rewards for item in sublist]
        '''
        
        '''
        # Determine reward from the environment       
        for actual, pred in zip(relations, chosen):
            if actual == pred:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        '''
        
        # td_error = [(p - r) for p, r in zip(pred_rewards, rewards)] 
        
        s = states[1]
        r = relations[1]
        p = pred_rewards[1]
        
        for s, r, p in zip(states, relations, pred_rewards):
            
            print(str(s) + " " + str(r) + " " + str(p))
            
            actor_prob = sess.run(actor.layer3, feed_dict = {actor.observation: s})
            chosen = actor_prob.index(max(prob_list))
            
            # I need to come back and make sure this is correct
            reward = -p
            if chosen == r:
                reward = 1 + rewards
                
            loss, _ = sess.run([actor.loss, actor.train], feed_dict = {actor.observation: s, actor.td_error: reward, actor.relation: r})

    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required = True)
    parser.add_argument("-topic", type = str, default = "Georgetown_University")
    parser.add_argument("-actor_LR", type = float, default = 0.001)
    parser.add_argument("-critic_LR", type = float, default = 0.001)
    parser.add_argument("-eps", type = int, default = 2000)


    #TODO: add your own parameters

    args = parser.parse_args()
    run(args)

