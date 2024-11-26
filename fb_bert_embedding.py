from sentence_transformers import SentenceTransformer
import pickle
import csv
import numpy as np
import pandas as pd


def get_attr_text_list(file_path):
    df = pd.read_csv(file_path, sep='\t', header=0)
    attr_text_list = []
    for index, row in df.iterrows():  
        attr_text_list.append(row['node_description'])
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
        
        des = row[3]
        kg = row[4]
        concat_text_list.append(des+kg)
    
    print('len(concat_text_list)',len(concat_text_list))
    print(concat_text_list[0])
    return concat_text_list

def get_response_list(llm_res_path):

    with open(llm_res_path, 'r', encoding='utf-8') as file:  
        lines = file.read().split('\n') 
        header = lines[0].strip().split('\t')  
        data = [line.strip().split('\t') for line in lines[1:-1]]
        # print(data[-1])
        response_list = [row[1] for row in data]

        return response_list

def bert_encode_numpy(text_list):
    sample_num = len(text_list)
    embeddings = np.zeros((sample_num, 768))

    model = SentenceTransformer('./huggingface_models/all-mpnet-base-v2')
    
    for i, text in enumerate(text_list):  
        embedding = model.encode(text)
        embeddings[i] = embedding
        
        if i % 1000 == 0:
            print('Encoded Samples: {}/{}'.format(i, sample_num))

    print('embeddings.shape:', embeddings.shape)

    # # normalize
    mean = np.mean(embeddings, axis=0)
    var = np.std(embeddings, axis=0)
    embeddings = (embeddings-mean)/var

    return embeddings



def save_to_pkl(embeddings, pkl_path):
    with open(pkl_path, 'wb') as f:
        pickle.dump(embeddings, f)
    print('Text embeddings saved to {}'.format(pkl_path))


sampling = 'cluster'
dataset_name = 'fb15k'   
root_path = './text_embeddings/'


#'''
for max_triplets_num in [9]:   
    
    if sampling == 'random':
        llm_res_path = './llm_response/FB15k/lxy_mpnet_llama3-1_0804/fb15k_llm_res_t{}_random.tsv'.format(str(max_triplets_num))       
        pkl_path = '{}{}/lxy/mpnet_llama3-1/llm_text_mpnet_llama3-1_embeddings_random_t{}_norm.pkl'.format(root_path, dataset_name, str(max_triplets_num)) 
        
        response_list = get_response_list(llm_res_path)
        embeddings = bert_encode_numpy(response_list)
        save_to_pkl(embeddings, pkl_path)

    elif sampling == 'cluster':
        llm_res_path = './llm_response/FB15k/lxy_mpnet_llama3-1_0804/fb15k_llm_res_t{}_cluster.tsv'.format(str(max_triplets_num))       
        pkl_path = '{}{}/lxy/mpnet_llama3-1/llm_text_mpnet_llama3-1_embeddings_cluster_t{}_norm.pkl'.format(root_path, dataset_name, str(max_triplets_num)) 
        
        response_list = get_response_list(llm_res_path)
        embeddings = bert_encode_numpy(response_list)
        save_to_pkl(embeddings, pkl_path)
#'''

'''
pkl_path = '{}{}/lxy/mpnet_qwen2/attr_text_mpnet-embeddings_norm.pkl'.format(root_path, dataset_name)
file_path = './datasets/FB15k/fb15k_description.tsv'  
attr_text_list = get_attr_text_list(file_path)
embeddings = bert_encode_numpy(attr_text_list)
save_to_pkl(embeddings, pkl_path)

pkl_path = '{}{}/lxy/mpnet_qwen2/concat_text_mpnet-embeddings_cluster_t10_norm.pkl'.format(root_path, dataset_name)
file_path = './datasets/FB15k/lxy_0803_mpnet/fb15k_merge_cluster_t10.tsv'  
concat_text_list = get_concat_text_list(file_path)
embeddings = bert_encode_numpy(concat_text_list)
save_to_pkl(embeddings, pkl_path)
'''