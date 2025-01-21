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


    print(rel_id2relation)

    return rel_id2relation


def gen_data_info():
    '''
    node_info_file: (node_id, popularity, valid, description)
    edge_list_file: (src_id, dst_id)
    rel_file: (src_Id, rel_id, dst_id)
    '''

    # read movie info
    credit_df = pd.read_csv('./datasets/TMDB5K/tmdb_5000_credits.csv')
    movie_df = pd.read_csv('./datasets/TMDB5K/tmdb_5000_movies.csv',encoding='utf-8')

    movie_info = pd.merge(credit_df, movie_df, left_on='movie_id', right_on='id')
    node_infos = []
    id_num = 0

    print(movie_info.columns)
    # movie nodes
    movie2node = dict()
    for row in movie_info[['movie_id', 'popularity', 'overview','title_y']].itertuples():
        movie_id = getattr(row, 'movie_id')
        popularity = getattr(row, 'popularity')
        description = getattr(row, 'overview')
        node_id = id_num
        id_num += 1

        title = getattr(row, 'title_y')

        movie2node[movie_id] = node_id
        node_infos.append((node_id, title, popularity, 1, description))

    # movie genre nodes
    ## select genre node
    genre_dict = dict()
    movie_info['genres'] = movie_info['genres'].apply(literal_eval)
    for movie_item in movie_info['genres']:
        for item in movie_item:
            if item['id'] not in genre_dict:
                genre_dict[item['id']] = item['name']
    ## insert genre node
    genre2node = dict()
    for genre_id, genre_name in genre_dict.items():
        node_id = id_num
        id_num += 1
        genre2node[genre_id] = node_id

        node_infos.append((node_id, genre_name, 0, 0, '\\N'))

    # movie company nodes
    ## select company nodes
    company_dict = dict()
    movie_info['production_companies'] = movie_info['production_companies'].apply(literal_eval)
    for movie_item in movie_info['production_companies']:
        for item in movie_item:
            if item['id'] not in company_dict:
                company_dict[item['id']] = item['name']
    # insert company node
    company2node = dict()
    for comp_id, comp_name in company_dict.items():
        node_id = id_num
        id_num += 1
        company2node[comp_id] = node_id
        node_infos.append((node_id, comp_name, 0, 0, '\\N'))

    # movie country nodes
    ## select country nodes
    country_dict = dict()
    movie_info['production_countries'] = movie_info['production_countries'].apply(literal_eval)
    for movie_item in movie_info['production_countries']:
        for item in movie_item:
            if item['iso_3166_1'] not in country_dict:
                country_dict[item['iso_3166_1']] = item['name']
    ## insert country nodes
    country2node = dict()
    for country_id, country_name in country_dict.items():
        node_id = id_num
        id_num += 1
        country2node[country_id] = node_id
        node_infos.append((node_id, country_name, 0, 0, '\\N'))

    # people nodes

    ## select cast and crew nodes
    person_dict = dict()
    movie_info['cast'] = movie_info['cast'].apply(literal_eval)
    for movie_item in movie_info['cast']:
        for item in movie_item:
            person_dict[item['id']] = item['name']

    movie_info['crew'] = movie_info['crew'].apply(literal_eval)
    for movie_item in movie_info['crew']:
        for item in movie_item:
            person_dict[item['id']] = item['name']

    ## insert people nodes
    people2node = dict()
    for person_id, person_name in person_dict.items():
        node_id = id_num
        id_num += 1
        people2node[person_id] = node_id
        node_infos.append((node_id, person_name, 0, 0, '\\N'))

    # other nodes
    other2id = dict()
    other_name = ['movie', 'genre', 'company', 'country', 'person']
    for name in other_name:
        node_id = id_num
        id_num += 1
        other2id[name] = node_id
        node_infos.append((node_id, name, 0, 0, '\\N'))

    print('node_num', len(node_infos))

    # write to node_info.tsv
    columns = ['node_id','name' , 'score', 'valid', 'description']
    node_file = open('./datasets/TMDB5K/node_info_full.tsv', 'w')
    node_file.write('\t'.join(columns)+'\n')
    for node_info in node_infos:
        node_info = [str(tmp) for tmp in node_info]
        node_file.write('\t'.join(node_info)+'\n')
    node_file.close()

def get_node_id2name_dict(file_path):  
    with open(file_path, 'r') as file:  
        content = file.read()
    lines = content.split('\n')[1:]  

    id2name_dict = {}  
    for line in lines:  
        if line:
            ls = line.split('\t')
            id = int(ls[0])  
            name = ls[1]  
            id2name_dict[id] = name

    print('len(id2name_dict)',len(id2name_dict))
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
        if relation == 'is_a':
            text = " [{}] {} {}.".format(subject_term,relation, object_term)
        elif relation == 'is_a_inverse':
            text = " [{}] includes .".format(subject_term,relation, object_term)
        else:        
            if 'inverse' not in relation:
                text = " [{}] is the {} of {}.".format(subject_term,relation, object_term)
            else:
                text = " [{}]'s {} is {}.".format(subject_term,relation[:-8], object_term)
        text_list.append(text)

    
    embeddings = bert_encode_numpy(text_list)

    representative_indices = find_representatives(embeddings, max_triplets_num)

    return [triplets[i] for i in representative_indices]

def get_node_id2texts(ht2r_dict,node_id2name_dict,rel_id2relation, max_triplets_num,sampling_method):
    # import math
    cut_count = 0
    subject_id2text = {}

    subject_id_2_relation2object_dict = {}
    for head_tail_id in ht2r_dict:
        subject_id = head_tail_id[1]
        object_term = node_id2name_dict[head_tail_id[0]]

        rel_id = ht2r_dict[head_tail_id]
        relation = rel_id2relation[rel_id]

        if subject_id not in subject_id_2_relation2object_dict:
            subject_id_2_relation2object_dict[subject_id] = {}

        relation2object_dict = subject_id_2_relation2object_dict[subject_id]

        if relation not in relation2object_dict:
            relation2object_dict[relation] = []
        if type(object_term) == str:
            relation2object_dict[relation].append(object_term)

    
    movie_count = 0
    
    for subject_id in subject_id_2_relation2object_dict:
        if subject_id <= 4802: #
            subject_term = node_id2name_dict[subject_id]
            subject_id2text[subject_id] = ""

            movie_count += 1

            if movie_count % 10 == 0:
                print(movie_count)
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
                if relation == 'is_a':
                    text += " [{}] {} ".format(subject_term,relation)
                elif relation == 'is_a_inverse':
                    text += " [{}] includes ".format(subject_term,relation)
                else:
                
                    if 'inverse' not in relation:
                        text += " [{}] is the {} of ".format(subject_term,relation)
                    else:
                        text += " [{}]'s {} is ".format(subject_term,relation[:-8])

                object_list = relation2object_sample[relation]


                text += ', '.join(object_list) + '.'

                

            
            subject_id2text[subject_id] = text.replace('_', ' ')

    print('Downsampling Node Ratio:{}'.format(cut_count/4802))


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
            entity_name = attribute_row[1]
            try:
                attribute_text = attribute_row[4]
            except:
                attribute_text = 'Null'

            triplets_text = triplets_row[1]

            f.write(f'{node_id}\t{entity_name}\t{attribute_text}\t{triplets_text}\n')


           
json_path = './datasets/TMDB5K/rel_name.json'  
rel_id2relation = read_json(json_path)

file_path='./datasets/TMDB5K/node_info_full.tsv'
node_id2name_dict = get_node_id2name_dict(file_path)

file_path = "./datasets/TMDB5K/tmdb_rel-all.txt"  
ht2r_dict = read_relation(file_path,node_id2name_dict,rel_id2relation)  # 


#"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--t", type=int, default=5)
args = parser.parse_args()

sampling_method = 'cluster'
#sampling_method='random'   

t = args.t

print('Max Triplets Nums:', t)
subject_id2text = get_node_id2texts(ht2r_dict,node_id2name_dict,rel_id2relation, max_triplets_num=t,sampling_method=sampling_method)

tri_text = get_triplets_text(subject_id2text)
des_path = './datasets/TMDB5K/node_info_full.tsv'  
output_path = './datasets/TMDB5K/mpnet/tmdb5k_merge_{}_t{}.tsv'.format(sampling_method,t)  

if not os.path.exists('./datasets/TMDB5K/mpnet'):
        os.makedirs('./datasets/TMDB5K/mpnet')

generate_overall_tsv(tri_text, des_path,output_path)
#"""





