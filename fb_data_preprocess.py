import pandas as pd
import random
import numpy as np
from sentence_transformers import SentenceTransformer

def get_mid2entity_dict(file_path):  
    df = pd.read_csv(file_path, sep='\t', header=0)
    mid2entity_dict = {}  
    for index, row in df.iterrows():  
        mid2entity_dict[row['mid']] = row['entity_name'] 
    return mid2entity_dict

def get_mid2node_id(file_path):  
    df = pd.read_csv(file_path, sep='\t', header=0)
    mid2node_id_dict = {}  
    node_id2mid_dict = {}
    for index, row in df.iterrows():  
        node_id = row['node_id']
        mid = row['mid']  
        mid2node_id_dict[mid] = node_id
        node_id2mid_dict[node_id] = mid
    return mid2node_id_dict,node_id2mid_dict


def load_fb15k(data_path, dataset_name='FB15k'):
    """
    load fb15k data
    :param data_path: str, data file path
    :return:
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # edge list
    edges = data['edges']
    labels = data['labels']

    # node feats
    node_feat1 = data['features']
    node_feat2 = pickle.load(open('./datasets/fb_lang.pk', 'rb'))
    node_feat2 = torch.from_numpy(node_feat2).float()


    invalid_masks = data['invalid_masks']
    edge_types = data['edge_types']
    rel_num = (max(edge_types) + 1).item()

    # construct a heterogeneous graph
    hg = dgl.graph(edges)

    # generate edge norm
    g = hg.local_var()
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    node_norm = torch.from_numpy(norm).view(-1, 1)
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges: {'norm': edges.dst['norm']})
    edge_norm = g.edata['norm']


    return hg, edge_types, edge_norm, rel_num, node_feats, labels, train_idx, val_idx, test_idx

def get_triplet2text_dict(input_file,mid2entity_dict):  
    triplet2text_dict = {}  
    h_mid2text_dict = {}  

    with open(input_file, 'r') as file:  
        lines = file.readlines()  
        for line in lines:  
            line = line.strip()  
            if line:  # 过滤空行  
                
                key_value_pair = line.split('\t')  

                #key = (mid2entity_dict[key_value_pair[0]], mid2entity_dict[key_value_pair[2]])  
                key = (key_value_pair[0], key_value_pair[2])  
                value = key_value_pair[1].strip('\n').split('/')[-1].replace('_', ' ') 


                if key not in triplet2text_dict:  
                    triplet2text_dict[key] = value  

    return triplet2text_dict



from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

def find_representatives(data, k):
    data = normalize(data)

    # print(data)
    # print(np.linalg.norm(data, axis=1))


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
        
    #     if i % 1000 == 0:
    #         print('Encoded Samples: {}/{}'.format(i, sample_num))

    # print('embeddings.shape:', embeddings.shape)

    # # normalize
    # mean = np.mean(embeddings, axis=0)
    # var = np.std(embeddings, axis=0)
    # embeddings = (embeddings-mean)/var

    return embeddings

def cluster_sample(subject_term, triplets, max_triplets_num): # triplets:[(relation, object_term)]

    text_list = ['{} is the {} of {}.'.format(subject_term,relation,object_term) for (relation, object_term) in triplets]
    
    embeddings = bert_encode_numpy(text_list)

    representative_indices = find_representatives(embeddings, max_triplets_num)

    return [triplets[i] for i in representative_indices]

def get_node_id2texts(triplet2text_dict,mid2node_id_dict,mid2entity_dict,node_id2mid_dict, max_triplets_num, sampling_method):
    # import math
    cut_count = 0
    subject_id2text = {}

    subject_id_2_relation2object_dict = {}
    for key in triplet2text_dict:
        subject = mid2node_id_dict[key[1]]
        object_term = mid2entity_dict[key[0]]

        relation = triplet2text_dict[key]

        if subject not in subject_id_2_relation2object_dict:
            subject_id_2_relation2object_dict[subject] = {}

        relation2object_dict = subject_id_2_relation2object_dict[subject]

        if relation not in relation2object_dict:
            relation2object_dict[relation] = []
        if type(object_term) == str:
            relation2object_dict[relation].append(object_term)

    
    count = 0
    for subject_id in subject_id_2_relation2object_dict:
        count += 1
        subject_term = mid2entity_dict[node_id2mid_dict[subject_id]]
        subject_id2text[subject_id] = ""
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
            text += " \"{}\" is the {} of ".format(subject_term,relation)
            object_list = relation2object_sample[relation]
            text += ','.join(object_list) + '.'
        
        if count % 1 == 0:
            print('{}/14951'.format(count))

        
        # test
        # if subject_id == 5243:

        #     print(text)


        
        subject_id2text[subject_id] = text.replace('_', ' ')

    print('Downsampling Node Ratio:{}'.format(cut_count/len(subject_id2text.keys())))


    return subject_id2text

def count_words(text):
    import re
    words = re.findall(r'\b\w+\b', text)
    if len(text) == 0:
        print(text)
    return len(words)


def get_triplets_text(node_id2mid_dict, subject_id2text):  
    tri_text = ""
    tri_text += 'node_id\tmid\ttriplets_text\n'
    sorted_node_id2mid_dict = sorted(node_id2mid_dict.items(), key=lambda x: x[0])
    for node_id, mid in sorted_node_id2mid_dict: 
        mid_text = subject_id2text.get(node_id, '')  
        tri_text += '{}\t{}\t{}\n'.format(node_id,mid,mid_text)
    return tri_text

def generate_overall_tsv(tri_text, des_tsv_path, mid2entity_dict, output_path):
    
    lines = tri_text.split('\n')  
    header = lines[0].strip().split('\t')  
    triplets_data = [line.strip().split('\t') for line in lines[1:]]

    with open(des_tsv_path, 'r', encoding='utf-8') as file:  
        content = file.read()
    lines = content.split('\n')  
    attribute_data = [line.strip().split('\t') for line in lines[1:]]


    with open(output_path, 'w', encoding='utf-8') as f:  
       f.write('node_id\tmid\tentity_name\tattribute_text\ttriplets_text\n')

       for idx in range(len(attribute_data)-1):
            triplets_row = triplets_data[idx]
            attribute_row = attribute_data[idx]

            node_id = attribute_row[0]
            mid = attribute_row[1]
            entity_name = mid2entity_dict[mid]
            attribute_text = attribute_row[2]
            try:
                triplets_text = triplets_row[2]
            except:
                triplets_text = 'Null'

            f.write(f'{node_id}\t{mid}\t{entity_name}\t{attribute_text}\t{triplets_text}\n')


           


file_path = './datasets/FB15k/node_infos.tsv'  
mid2entity_dict = get_mid2entity_dict(file_path)  


file_path = './datasets/FB15k/fb15k_description.tsv'  
mid2node_id_dict,node_id2mid_dict = get_mid2node_id(file_path)  

file_path = "./datasets/FB15k/freebase_mtr100_mte100-all.txt"  
triplet2text_dict = get_triplet2text_dict(file_path,mid2entity_dict)  # 


'''
# random_sampling
for t in [10, 15, 20, 25, 30]:
    print('Max Triplets Nums:', t)
    subject_id2text = get_node_id2texts(triplet2text_dict,mid2node_id_dict,mid2entity_dict,node_id2mid_dict, max_triplets_num=t,sampling_method='random')

    tri_text = get_triplets_text(node_id2mid_dict, subject_id2text)
    des_path = './datasets/FB15k/fb15k_description.tsv'  
    output_path = './datasets/FB15k/fb15k_merge_random_t{}.tsv'.format(t)  
    generate_overall_tsv(tri_text, des_path, mid2entity_dict,output_path)
'''

#'''
t=10
print('Max Triplets Nums:', t)
subject_id2text = get_node_id2texts(triplet2text_dict,mid2node_id_dict,mid2entity_dict,node_id2mid_dict, max_triplets_num=t,sampling_method='random')

tri_text = get_triplets_text(node_id2mid_dict, subject_id2text)
des_path = './datasets/FB15k/fb15k_description.tsv'  
output_path = './datasets/FB15k/lxy_0803_mpnet/test/fb15k_merge_cluster_t{}.tsv'.format(t)  
generate_overall_tsv(tri_text, des_path, mid2entity_dict,output_path)
#'''
















# 输出字典的前 10 个键值对  
# for i in range(10):  
#     key = list(triplet2text_dict.keys())[i]  
#     value = triplet2text_dict[key]  
#     print(f"{key}: {value}") 


# 统计单词数量
# counts_list = []
# for key in subject_id2text:   
#     text = subject_id2text[key]  
#     counts_list.append(count_words(text))
# print(len(counts_list))
# print('Word Counts: mean:{} sum:{} max:{} min:{}'.format(np.mean(counts_list),np.sum(counts_list),np.max(counts_list),np.min(counts_list)))
