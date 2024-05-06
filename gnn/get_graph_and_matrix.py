import pickle
import torch
import pandas as pd
import scipy.sparse as sp
import json
import networkx as nx
import os
import numpy as np


def init_ds(json):
    ds= {}
    keys = json.keys()
    for k in keys:
        ds[k]= []
    return ds, keys

def read_json(file):
    dataset = {}
    keys = []
    with open(file) as file_lines:
        for count, line in enumerate(file_lines):
            data = json.loads(line.strip())
            if count ==0:
                dataset, keys = init_ds(data)
            for k in keys:
                dataset[k].append(data[k])
                
        return pd.DataFrame(dataset)


def extract_reference_categories(categories, use_only_top_level_categories=False):
    ref_categories = []

    for c in categories:
        if use_only_top_level_categories:
            if len(c['parents']) == 0:
                ref_categories.append(c['title'])
        else:
            if ('restaurants' in c['parents']) or ('food' in c['parents']):
                ref_categories.append(c['title'])

    ref_categories = np.asarray(ref_categories)

    return ref_categories


'''get 2K user graph and review matrix'''
def get_graph_and_matrix(user, item, data, years, state, label):
    print('Extracting data in ', state, ' in ', years, ' ...')
    data['date'] = pd.to_datetime(data['date'])   # review/ rating
    item = item[item['state'].str.upper().str.replace(' ', '')  == state].reset_index(drop=True) # 34039, select  business in state
    item_ids = dict(zip(item['item_id'], item.index))  # business id, index
    if label == 'rating':
        data = data[data['item_id'].isin(item_ids) & (data['rating'] > 0) & (data['date'].dt.year.isin(years))].reset_index(drop=True)
    else:
        data = data[data['item_id'].isin(item_ids) & (data['text'].apply(lambda x: len(x.split())) > 0) & (data['date'].dt.year.isin(years))].reset_index(drop=True)

    item = item[item['item_id'].isin(data['item_id'])].reset_index(drop=True)  # 28029, select business that has rating in years
    item_ids = dict(zip(item['item_id'], item.index))
    user = user[user['user_id'].isin(data['user_id'])].reset_index(drop=True)  # 197532, select user that has rating on bussiness in state in years
    user_id2index = dict(zip(user['user_id'], user.index))
    user_index2id = dict(zip(user.index, user['user_id']))
    # user_review_counts = data['user_id'].value_counts()
    # print('Reviews per user in ',state,' in ',year,':' ,data.shape[0]/user.shape[0],'.    Median:',user_review_counts[int((len(user_review_counts)-1)/2)])


    print('Getting user graph ...')
    f_row,f_col=[],[]
    for i, friends in user['friends'].items():
        for friend in friends.split(', '):
            if friend in user_id2index:
                f_row.append(i)
                f_col.append(user_id2index[friend])
    edges = list(zip(f_row, f_col))
    G = nx.Graph()
    G.add_edges_from(edges)
    print('The number of nonisolated users in the raw network:',len(list(G.nodes)))   # 78747
    print('Edges among nonisolated users in the raw network:',len(list(G.edges)))  # 419120

    print('Getting major connected components ...')
    '''index, not user_id'''
    connected_components = list(nx.connected_components(G))
    num = []
    for i in range(len(connected_components)):
        num.append(len(connected_components[i]))
    print('The number of connected components in the raw network:',len(num))   # 1225
    print('The number of users in the major cc:',len(connected_components[0]))  # 76149
        

    print('Ranking by reviews and building a subgraph of top 2K...')
    '''need to find user_id according to index'''
    major_cc = list(connected_components[0])
    major_cc_user_id = []
    for index in major_cc:
        major_cc_user_id.append(user_index2id[index])
    major_cc_review = data[data['user_id'].isin(major_cc_user_id)]
    '''get subgraph user id'''
    major_cc_user_review_counts = major_cc_review['user_id'].value_counts()  # count the number of each user_id
    subgraph_user_id = list(major_cc_user_review_counts[:2600].index)
    # print('The review counts of this subgraph',major_cc_user_review_counts[:2600])
    # print('The number of users in this subgraph',len(subgraph_user_id))


    print('Check if the subgraph is connected ...')
    subgraph_user = user[user['user_id'].isin(subgraph_user_id)].reset_index(drop=True)
    subgraph_user_id2index = dict(zip(subgraph_user['user_id'], subgraph_user.index))
    subgraph_user_index2id = dict(zip(subgraph_user.index, subgraph_user['user_id']))
    f_row_subgraph,f_col_subgraph=[],[]
    for i, friends in subgraph_user['friends'].items():
        for friend in friends.split(', '):
            if friend in subgraph_user_id2index:
                f_row_subgraph.append(i)
                f_col_subgraph.append(subgraph_user_id2index[friend])
    edges_sub = list(zip(f_row_subgraph, f_col_subgraph))
    print(len(f_row_subgraph))
    print(len(f_col_subgraph))
    sub_G = nx.Graph()
    sub_G.add_edges_from(edges_sub)
    print('The number of nonisolated users in this subgraph: ',len(sub_G.nodes))
    if len(sub_G.nodes) < 2600:
        print('The subgraph of top 2K is not connected.')
    sub_cc = list(nx.connected_components(sub_G))
    num = []
    for i in range(len(sub_cc)):
        num.append(len(sub_cc[i]))
    print('The size of the major connected component of this subgraph: ', len(sub_cc[0]))  # 2234
    
    if len(sub_cc[0]) < len(sub_G.nodes):
        print('The subgraph of top 2K is not connected.')

    print('Extracting the major connected component of this subgraph ... ')
    final_cc = list(sub_cc[0])
    final_user_id = []
    '''get final user id'''
    for index in final_cc:
        final_user_id.append(subgraph_user_index2id[index])
    print(len(final_user_id))   # 2234
    
    final_user = user[user['user_id'].isin(final_user_id)].reset_index(drop=True)
    final_user_id2index = dict(zip(final_user['user_id'], final_user.index))
    final_user_index2id = dict(zip(final_user.index,final_user['user_id']))
    final_data = data[data['user_id'].isin(final_user_id)].reset_index(drop=True)
    final_item = item[item['item_id'].isin(final_data['item_id'])].reset_index(drop=True)
    final_item_id2index = dict(zip(final_item['item_id'], final_item.index))
    final_item_index2id = dict(zip(final_item.index,final_item['item_id']))
    final_data['user_id'] = final_data['user_id'].apply(final_user_id2index.get)
    final_data['item_id'] = final_data['item_id'].apply(final_item_id2index.get)

    print('Getting final user graph ...')
    f_row_final,f_col_final=[],[]
    for i, friends in final_user['friends'].items():
        for friend in friends.split(', '):
            if friend in final_user_id2index:
                f_row_final.append(i)
                f_col_final.append(final_user_id2index[friend])
    print(len(f_row_final))
    print(len(f_col_final))
    user_graph = sp.coo_matrix(([1.0]*len(f_row_final), (f_row_final, f_col_final)), shape=(len(final_user), len(final_user)))
    # edges_final = list(zip(f_row_final, f_col_final))
    # final_G = nx.Graph()
    # final_G.add_edges_from(edges_final)
    # print(len(final_G.nodes))   # 2234
    # print(len(final_G.edges))
    A = user_graph.toarray()   # 2234
    print('shape of adjacency matrix is:', A.shape)


    print('Getting review matrix ...')
    if label == 'rating':
        final_clean_data=final_data[{'user_id','item_id','rating'}]
    else:
        final_clean_data = final_data[{'user_id','item_id','text'}]
        text_length = final_clean_data['text'].apply(lambda x: len(x.split()))
        final_clean_data.loc[:, 'text'] = text_length
    final_clean_data_mean=final_clean_data.groupby(by=['user_id','item_id']).mean()  # same user and item id, use mean rating
    final_clean_data_mean=final_clean_data_mean.reset_index()
    rm_row = final_clean_data_mean['user_id'].tolist()
    rm_col = final_clean_data_mean['item_id'].tolist()
    if label == 'rating':
        entry = final_clean_data_mean['rating'].tolist()
    else:   
        entry = final_clean_data_mean['text'].tolist()

    review_matrix = sp.coo_matrix((entry, (rm_row, rm_col)), shape=(len(final_user), len(final_item)))

    dataset = []
    X = review_matrix.toarray()
    print('shape of feature matrix is:', X.shape)
   
    dataset.append({"X": torch.FloatTensor(X), "A": torch.FloatTensor(A)})

    save_dir = '../data/Yelp/'
    filename = f'{state}_{label}_food.pickle'
    save_path = os.path.join(save_dir, filename)

    with open(save_path, 'wb') as f:
        pickle.dump((dataset), f)
  

if __name__ == '__main__':
    path = '../data/Yelp'
    print('Getting raw data ...')
    print('loading user data...')
    user = read_json(path+'/user.json')
    print('loading business data...')
    item = read_json(path+'/business.json').rename(columns={'business_id': 'item_id'})
    print('loading review data...')
    data = read_json(path+'/review.json').rename(columns={'business_id': 'item_id', 'stars': 'rating', 'text': 'text'})
    
    categories = []
    categories_path = '../data/Yelp/categories.json'
    with open(categories_path, 'r') as f:
        categories = json.load(f)

    ref_categories = extract_reference_categories(categories, use_only_top_level_categories=False)  # restaurants/food
    print(ref_categories)

    item['categories'].fillna('', inplace=True)
    pattern = '|'.join(ref_categories)
    item[item['categories'].str.contains(pattern, regex=True)]

    years=[2016,2017,2018]
    # states=['AZ','ON']
    states = ['PA', 'LA']
    labels = ['review', 'rating']
    for state in states:
        for label in labels:
            get_graph_and_matrix(user, item, data, years, state, label)
        
