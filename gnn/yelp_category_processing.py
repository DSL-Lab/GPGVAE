import numpy as np
import scipy.sparse as sp
import networkx as nx
import pickle
import yelp_parsing_utilities as ypu
import json
from collections import defaultdict
import torch
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

all_businesses = []
businesses_path = '../data/Yelp/business.json'
with open(businesses_path, 'rb') as f:
    for jsonObj in f:
        b = json.loads(jsonObj)
        all_businesses.append(b)   # len(all_businesses) = 150346

all_reviews = []
reviews_path = '../data/Yelp/review.json'
with open(reviews_path, 'rb') as f:
    for jsonObj in f:
        r = json.loads(jsonObj)
        all_reviews.append(r)   # len(all_reviews) = 6990280

all_tips = []
tips_path = '../data/Yelp/yelp_academic_dataset_tip.json'
with open(tips_path, 'rb') as f:
    for jsonObj in f:
        r = json.loads(jsonObj)
        all_tips.append(r)    # 908915

all_users = []     
users_path = '../data/Yelp/user.json'
with open(users_path, 'rb') as f:
    for jsonObj in f:
        u = json.loads(jsonObj)
        all_users.append(u)          # 1987897

categories = []
categories_path = '../data/Yelp/categories.json'
with open(categories_path, 'r') as f:
      categories = json.load(f)

ref_categories = ypu.extract_reference_categories(categories, use_only_top_level_categories=False)  # restaurants
print(ref_categories)   # 260 categories

business_to_category_dict = ypu.build_business_to_CA_categories_dict(all_businesses, ref_categories)  # keys: business ids 4416, values: categories of each business 
# state = 'PA'
# business_to_state_list = ypu.build_business_to_state(all_businesses, state)   # 34039
### use rating star  each user's rating score to each category  224831, 
# user_category_scores_dict = ypu.build_user_categories_scores_dict(all_reviews, business_to_category_dict, ref_categories)  # keys: user ids, values: average score per reference category
# user_scores_dict = ypu.build_user_scores_dict(all_reviews, business_to_state_list)  # keys: user ids, values: score per business id

### use the length of words of reviews and tips
user_category_reviews_dict = ypu.build_user_categories_review_dict(all_reviews, all_tips, business_to_category_dict, ref_categories)


user_friends_dict = ypu.build_users_friends_dict(all_users, user_category_reviews_dict)  # user ids, friends ids
# user_friends_dict = ypu.build_users_friends_dict(all_users, user_scores_dict)  # user ids, friends ids

all_possible_edges, all_weighted_edges = ypu.build_social_graph(user_friends_dict, user_category_reviews_dict, ref_categories, verbose=True, print_status_every_percentage=0.05)

# all_possible_edges, all_weighted_edges = ypu.build_social_graph(user_friends_dict, user_scores_dict, ref_categories, verbose=True, print_status_every_percentage=0.05)


G_weighted_edges = nx.Graph((x, y, {'weight': v}) for x, y, v in all_weighted_edges)
G_all_edges = nx.convert.from_edgelist(all_possible_edges)
largest_cc = max(nx.connected_components(G_all_edges), key=len)

A = nx.linalg.graphmatrix.adjacency_matrix(G_all_edges, nodelist=largest_cc)
A = A.toarray() 

dataset = []
X = [user_category_reviews_dict[user_id] for user_id in largest_cc]
# X = [user_scores_dict[user_id] for user_id in largest_cc]
X = np.stack(X, axis=0)
# dataset.append((A, X))
dataset.append({"X": torch.FloatTensor(X), "A": torch.FloatTensor(A)})

with open('../data/Yelp/CA_dataset_foodcategory_review.pickle', 'wb') as f:
    pickle.dump((dataset), f)





