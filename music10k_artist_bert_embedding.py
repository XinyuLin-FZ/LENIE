from sentence_transformers import SentenceTransformer
import pickle
import csv
import numpy as np
import pandas as pd
import copy


from transformers import AutoTokenizer, TransfoXLModel
import torch
import pickle




def get_attr_text_list(file_path):
    df = pd.read_csv(file_path, sep='\t', header=0)
    attr_text_list = []
    for index, row in df.iterrows():  
        attr_text_list.append(row['description'])
    print('len(attr_text_list)',len(attr_text_list))
    print(attr_text_list[0])

    return attr_text_list


def get_concat_text_list(file_path):
    
    concat_text_list = []
    with open(file_path, 'r', encoding='utf-8') as file:  
        content = file.read()
    lines = content.split('\n')  
    data = [line.strip().split('\t') for line in lines[1:-1]]
    for row in data: 
        
        des = row[2]
        kg = row[3]
        concat_text_list.append(des+kg)
    
    print('len(concat_text_list)',len(concat_text_list))
    print(concat_text_list[0])
    return concat_text_list

def get_response_list(llm_res_path):

    with open(llm_res_path, 'r', encoding='utf-8') as file:  
        lines = file.read().split('\n') 
        header = lines[0].strip().split('\t')  
        data = [line.strip().split('\t') for line in lines[1:-1]] 

        response_list = [row[1] for row in data]

        return response_list

def bert_encode_numpy_music(text_list):
    sample_num = len(text_list)
    embeddings = np.zeros((sample_num, 768))
    model = SentenceTransformer('./huggingface_models/all-mpnet-base-v2')

    for i, text in enumerate(text_list):  
        if i <= 4411:   # only for artist entity
            try:
                embedding = model.encode(text)
            except:
                embedding = model.encode('None')
            embeddings[i] = embedding
        else:
            break
        
        if i % 100 == 0:
            print('Encoded Samples: {}/{}'.format(i, 4411))


    embeddings = np.array(embeddings)
    print('movie embeddings.shape:', embeddings.shape)
    return embeddings






def save_to_pkl(embeddings, pkl_path):
    with open(pkl_path, 'wb') as f:
        pickle.dump(embeddings, f)
    print('Text embeddings saved to {}'.format(pkl_path))


def read_relation(file_path):  
    node_id2neigbour_music_id = {}  

    with open(file_path, 'r') as file:  
        lines = file.readlines()  
        for line in lines:  
            line = line.strip()  
            if line:  
                ls = line.strip('\n').split('\t')  
       
                head_id = int(ls[0])
                tail_id = int(ls[2])

                if head_id > 4411:

                    if head_id not in node_id2neigbour_music_id:  
                        node_id2neigbour_music_id[head_id] = []  
                    
                    if tail_id <= 4411:
                        node_id2neigbour_music_id[head_id].append(tail_id) 


    return node_id2neigbour_music_id


def complete_embeddings(music_embeddings, node_id2neigbour_music_id):
    
    embeddings = np.zeros((22985, music_embeddings.shape[1]))
    embeddings[:music_embeddings.shape[0]] = copy.deepcopy(music_embeddings)#.tolist()


    for node_id in range(4412,22985):

        if node_id % 5000 == 0:
            print('Encoded Samples: {}/{}'.format(node_id, 22984))

        neigbour_music_id = node_id2neigbour_music_id[node_id]
        
        e = np.mean(music_embeddings[neigbour_music_id],axis=0)
        embeddings[node_id] = e

    #embeddings = np.array(embeddings)
    print('embeddings.shape:', embeddings.shape)

    # # normalize
    mean = np.mean(embeddings, axis=0)
    var = np.std(embeddings, axis=0)
    embeddings = (embeddings-mean)/var

    return embeddings


dataset_name = 'music10k'
root_path = './text_embeddings/'

file_path = "./datasets/MUSIC10K/artist_familiarity/music10k_rel-all_artist_familiarity.txt"  
node_id2neigbour_music_id = read_relation(file_path)


sampling = 'cluster'

#'''
for max_triplets_num in [3]:   

    if sampling == 'random':
        llm_res_path = './llm_response/MUSIC10K/lxy_mpnet_llama3-1_0817/music10k_llm_res_t{}_random.tsv'.format(str(max_triplets_num)) 
        pkl_path = '{}{}/lxy/artist_mpnet_llama3-1/llm_text_mpnet_llama3-1_embeddings_random_t{}_norm.pkl'.format(root_path, dataset_name, str(max_triplets_num)) 
        response_list = get_response_list(llm_res_path)
        music_embeddings = bert_encode_numpy_music(response_list)   # 4803*768
        embeddings = complete_embeddings(music_embeddings, node_id2neigbour_music_id)
        save_to_pkl(embeddings, pkl_path)
    
    elif sampling == 'cluster':
        llm_res_path = './llm_response/MUSIC10K/lxy_mpnet_llama3-1_0817/music10k_llm_res_t{}_cluster.tsv'.format(str(max_triplets_num)) 
        pkl_path = '{}{}/lxy/artist_mpnet_llama3-1/llm_text_mpnet_llama3-1_embeddings_cluster_t{}_norm.pkl'.format(root_path, dataset_name, str(max_triplets_num)) 
        response_list = get_response_list(llm_res_path)
        music_embeddings = bert_encode_numpy_music(response_list)   # 4803*768
        embeddings = complete_embeddings(music_embeddings, node_id2neigbour_music_id)
        save_to_pkl(embeddings, pkl_path)
#'''


