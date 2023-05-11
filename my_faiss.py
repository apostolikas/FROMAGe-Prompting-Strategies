import faiss
import numpy as np
import os
import glob
import pickle
import torch
import os
from tqdm import tqdm
from fromage.utils import get_image_from_url
from datasets import load_dataset

model_dir=  './fromage_model/'
embs_paths = [s for s in glob.glob(os.path.join(model_dir, 'cc3m_embeddings*.pkl'))]
path_array = []# 2609992 instances
emb_matrix = []# 2609992 instances
for p in embs_paths:
    with open(p, 'rb') as wf:
        train_embs_data = pickle.load(wf)
        path_array.extend(train_embs_data['paths'])
        emb_matrix.append(train_embs_data['embeddings'])
    emb_matrix = np.concatenate(emb_matrix, axis=0)
# The author doesn't provide the captions so we have to find them ourselves by searching to another dataset

dataset = load_dataset('conceptual_captions') # 3318332 instances

# let's create an index based on the image embeddings
#https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
faiss_index = faiss.IndexFlatIP(emb_matrix.shape[1])
faiss.normalize_L2(emb_matrix)
faiss_index.add(emb_matrix)
#faiss.write_index(faiss_index, os.getcwd()) #I don't know how big is it

test_dummy = np.random.rand(2,emb_matrix.shape[1])
test_dummy = test_dummy.astype(np.float32)
faiss.normalize_L2(test_dummy)
D, retrieved_indices = faiss_index.search(test_dummy, k=5) 

for cur_indices in retrieved_indices: #number of test examples here 2
    for retrieved_id in cur_indices:
        retrieved_path = path_array[retrieved_id]
        hf_indices = dataset['train']['image_url'].index(retrieved_path) #takes some time
        correct_caption = dataset['train']['caption'][hf_indices]
        correct_img = get_image_from_url(retrieved_path)