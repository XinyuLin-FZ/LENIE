import pandas as pd
import random
import numpy as np
import json
from ast import literal_eval
from sentence_transformers import SentenceTransformer
import os

def read_json(json_path):
    with open(json_path, 'r') as f:
        relation2rel_id = json.load(f)
        
    rel_id2relation = {}
    for relation in relation2rel_id:
        rel_id2relation[str(relation2rel_id[relation])] = relation
    #print(rel_id2relation)

    return rel_id2relation

def get_node_id2name_dict(file_path):  
    with open(file_path, 'r') as file:  
        content = file.read()
    lines = content.split('\n')[1:]  

    id2name_dict = {}  
    for line in lines:  
    #for line in lines[:4413]:
        if line:
            ls = line.split('\t')
            id = int(ls[0])  
            #id = ls[1]
            name = ls[2]  
            id2name_dict[id] = name
    #print('len(id2name_dict)',len(id2name_dict))
    #print(list(id2name_dict.items())[:5])
    
    return id2name_dict

def read_relation(file_path,node_id2name_dict,rel_id2relation):  
    ht2r_dict = {}  

    with open(file_path, 'r') as file:  
        lines = file.readlines()  
        for line in lines:  
            line = line.strip()  
            if line: 

                ls = line.strip('\n').split('\t')  

                #key = (mid2entity_dict[key_value_pair[0]], mid2entity_dict[key_value_pair[2]])  
                head_tail_id = (int(ls[0]), int(ls[2]))  
                rel_id = ls[1]

                
                if head_tail_id not in ht2r_dict:  
                    ht2r_dict[head_tail_id] = rel_id  
                    
                '''
                if head_tail_id not in ht2r_dict:  
                    ht2r_dict[head_tail_id] = rel_id  
                else:
                    print(line)
                    
                    print(ht2r_dict[head_tail_id])
                    exit()
                '''
                           
    #print('len(ht2r_dict)',len(ht2r_dict))
    #print(list(ht2r_dict.items())[:5])
                    
    return ht2r_dict


from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
def find_representatives(data, k):
    data = normalize(data)

    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=0)
    kmeans.fit(data)

    labels = kmeans.labels_
    
    center_indices = []
    for i in range(k):
        indices = np.where(labels == i)[0]
        center = kmeans.cluster_centers_[i]
        distances = np.linalg.norm(data[indices] - center, axis=1)
        closest_index = indices[np.argmin(distances)]
        center_indices.append(closest_index)
    
    return center_indices
    

def bert_encode_numpy(text_list):
    sample_num = len(text_list)
    embeddings = np.zeros((sample_num, 768))

    model = SentenceTransformer('./huggingface_models/all-mpnet-base-v2')
    
    for i, text in enumerate(text_list):  
        embedding = model.encode(text)
        embeddings[i] = embedding

    return embeddings

def cluster_sample(subject_term, triplets, max_triplets_num): # triplets:[(relation, object_term)]

    text_list = []
    for (relation, object_term) in triplets:
        if 'inverse' not in relation:
            text = " [{}] is the {} of {}.".format(subject_term,relation, object_term)
        else:
            text = " [{}]'s {} is {}.".format(subject_term,relation[:-8], object_term)
        text_list.append(text)
        
    text_list_clear = []     
    for i in text_list:
        if i in text_list_clear:
            continue
        else:
            text_list_clear.append(i)
    text_list = text_list_clear
    
    if len(text_list) < max_triplets_num:
        max_triplets_num = len(text_list)


    embeddings = bert_encode_numpy(text_list)

    representative_indices = find_representatives(embeddings, max_triplets_num)

    return [triplets[i] for i in representative_indices]

def get_node_id2texts(ht2r_dict,node_id2name_dict,rel_id2relation, max_triplets_num,sampling_method):
    # import math
    cut_count = 0
    subject_id2text = {}

    #count = 0
    subject_id_2_relation2object_dict = {}
    for head_tail_id in ht2r_dict:
        
        subject_id = head_tail_id[1]
        object_term = node_id2name_dict[head_tail_id[0]]
        #object_id = head_tail_id[0]

        rel_id = ht2r_dict[head_tail_id]
        relation = rel_id2relation[rel_id]

        if subject_id not in subject_id_2_relation2object_dict:
            subject_id_2_relation2object_dict[subject_id] = {}
        
        relation2object_dict = subject_id_2_relation2object_dict[subject_id]

        if relation not in relation2object_dict:
            relation2object_dict[relation] = []

        if type(object_term) == str:
            relation2object_dict[relation].append(object_term)

    song_count = 0
    
    for subject_id in subject_id_2_relation2object_dict:
        if subject_id <= 4411:      # 4411
            subject_term = node_id2name_dict[subject_id]
            subject_id2text[subject_id] = ""

            song_count += 1

            if song_count % 10 == 0:
                print(song_count)
            relation2object_dict = subject_id_2_relation2object_dict[subject_id]

            
            triplets_num = 0
            triplets = []

            # search all the triplets
            for relation in relation2object_dict:
                object_list = relation2object_dict[relation]
                triplets_num += len(object_list)
                
                for object_term in object_list:
                    triplets.append((relation, object_term))


            if len(triplets) > max_triplets_num:
                cut_count += 1
                if sampling_method == 'random':
                    triplets = random.sample(triplets,max_triplets_num)               
                elif sampling_method == 'cluster':
                    triplets = cluster_sample(subject_term, triplets,max_triplets_num)
                else:
                    pass

            
            # wirte text
            relation2object_sample = {}
            text = ""
            triplets_text_list = []
            for relation, object_term in triplets:
                if relation not in relation2object_sample:
                    relation2object_sample[relation] = []
                relation2object_sample[relation].append(object_term)
            
            for relation in relation2object_sample:
                if 'inverse' not in relation:
                    text += " [{}] is the {} of ".format(subject_term,relation)
                else:
                    text += " [{}]'s {} is ".format(subject_term,relation[:-8])

                object_list = relation2object_sample[relation]


                text += ', '.join(object_list) + '.'


            
            subject_id2text[subject_id] = text.replace('_', ' ')

    print('Downsampling Node Ratio:{}'.format(cut_count/4411))
    
    return subject_id2text

def count_words(text):
    import re
    words = re.findall(r'\b\w+\b', text)
    if len(text) == 0:
        print(text)
    return len(words)


def get_triplets_text(subject_id2text):  
    tri_text = ""
    tri_text += 'node_id\ttriplets_text\n'
    subject_id2text = sorted(subject_id2text.items(), key=lambda x: int(x[0]))
    for node_id, text in subject_id2text: 
        tri_text += '{}\t{}\n'.format(node_id,text)
    return tri_text

def generate_overall_tsv(tri_text, des_tsv_path, output_path):
    lines = tri_text.split('\n')  
    header = lines[0].strip().split('\t')  
    triplets_data = [line.strip().split('\t') for line in lines[1:]]
    
    if triplets_data[-1] == ['']:
        triplets_data = triplets_data[:-1] 
    
    with open(des_tsv_path, 'r', encoding='utf-8') as file:  
        content = file.read()
    lines = content.split('\n')  
    attribute_data = [line.strip().split('\t') for line in lines[1:]]


    with open(output_path, 'w', encoding='utf-8') as f:  
       f.write('node_id\tentity_name\tattribute_text\ttriplets_text\n')


       for idx in range(len(attribute_data)-1):    

            triplets_row = triplets_data[idx]
            attribute_row = attribute_data[idx]


            node_id = attribute_row[0]
            entity_name = attribute_row[2]
            try:
                attribute_text = attribute_row[5]
            except:
                attribute_text = 'Null'

            try:
                if len(triplets_row)==2:
                    triplets_text = triplets_row[1]
                else:
                    triplets_text = 'Null'
            except:
                attribute_text = 'Null'    
            

            f.write(f'{node_id}\t{entity_name}\t{attribute_text}\t{triplets_text}\n')





json_path = './datasets/MUSIC10K/rel_name.json'  
rel_id2relation = read_json(json_path)

file_path='./datasets/MUSIC10K/node_info_artist_familiarity.tsv'
node_id2name_dict = get_node_id2name_dict(file_path)

file_path = "./datasets/MUSIC10K/music10k_rel-all_artist_familiarity.txt"  
ht2r_dict = read_relation(file_path,node_id2name_dict,rel_id2relation)  



import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--t", type=int, default=3)
args = parser.parse_args()

sampling_method = 'cluster'
#sampling_method='random'   

t = args.t

print('Max Triplets Nums:', t)
subject_id2text = get_node_id2texts(ht2r_dict,node_id2name_dict,rel_id2relation, max_triplets_num=t,sampling_method=sampling_method)  #cluster random

tri_text = get_triplets_text(subject_id2text)

des_path = './datasets/MUSIC10K/node_info_artist_familiarity.tsv'  
output_path = './datasets/MUSIC10K/mpnet/music10k_merge_{}_t{}.tsv'.format(sampling_method,t)  

if not os.path.exists('./datasets/MUSIC10K/mpnet'):
        os.makedirs('./datasets/MUSIC10K/mpnet')

generate_overall_tsv(tri_text, des_path,output_path)