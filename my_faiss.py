import faiss
import numpy as np
import os
import glob
import pickle
import torch
import os
from tqdm import tqdm
from fromage import utils
from fromage.utils import get_image_from_url
from datasets import load_dataset
# from flickr_inf import split_dictionary, display_interleaved_outputs
import pandas as pd
from PIL import Image
from numpy import asarray
import matplotlib as plt
from fromage import models

def split_dictionary(input_dict, chunk_size):
    res = []
    new_dict = {}
    for k, v in input_dict.items():
        if len(new_dict) < chunk_size:
            new_dict[k] = v
        else:
            res.append(new_dict)
            new_dict = {k: v}
    res.append(new_dict)
    return res

def display_interleaved_outputs(model_outputs, one_img_per_ret=True):
    for output in model_outputs:
        if type(output) == str:
            print(output)
        elif type(output) == list:
            if one_img_per_ret:
                plt.figure(figsize=(3, 3))
                plt.imshow(np.array(output[0]))
            else:
                fig, ax = plt.subplots(1, len(output), figsize=(3 * len(output), 3))
                for i, image in enumerate(output):
                    image = np.array(image)
                    ax[i].imshow(image)
                    ax[i].set_title(f'Retrieval #{i+1}')
            plt.show()
        elif type(output) == Image.Image:
            plt.figure(figsize=(3, 3))
            plt.imshow(np.array(output))
            plt.show()

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

df = pd.read_csv('./Flickr8k_text/ExpertAnnotations.txt',delimiter='\t')
cropped_df = df.loc[df['expert1'] == 4]
cap_df = pd.read_csv('./Flickr8k_text/Flickr8k.token.txt',delimiter='\t')
cap_dict = pd.Series(cap_df.cap.values,index=cap_df.cap_id).to_dict()
data_dict = {}
for img_id, cap_id in zip(cropped_df.image_id, cropped_df.caption_id):
    caption = cap_dict[cap_id]
    data_dict[img_id] = caption
vqa_data = split_dictionary(data_dict,3)

np.random.seed(0)
np.random.shuffle(vqa_data)
vqa_sublist = vqa_data[:15]

vqa_images_folder = './Flicker8k_Dataset/'

vqa_dict = vqa_sublist[4]
vqa_keys = list(vqa_dict.keys())
vqa_values = list(vqa_dict.values())

#! Question 
question_image_path = vqa_keys[2]
question_image = Image \
    .open(os.path.join(vqa_images_folder,question_image_path)) \
    .resize((224, 224)) \
    .convert('RGB')

#! new
# now we need to get the visual embeddings

model_dir = './fromage_model/'
model = models.load_fromage(model_dir)

visual_encoder = "openai/clip-vit-large-patch14"
feature_extractor = utils.get_feature_extractor_for_model(visual_encoder, train=False)
pixel_values = utils.get_pixel_values_for_model(feature_extractor, question_image)


pixel_values = pixel_values.to(device=model.model.logit_scale.device, dtype=model.model.logit_scale.dtype)
pixel_values = pixel_values[None, ...]

visual_embs = model.model.get_visual_embs(pixel_values, mode='retrieval')  # (1, n_visual_tokens, D)
visual_embs = torch.squeeze(visual_embs, dim=0) 
visual_embs = visual_embs.detach().cpu().numpy()
visual_embs = visual_embs.astype(np.float32)

# let's create an index based on the image embeddings
# https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
faiss_index = faiss.IndexFlatIP(emb_matrix.shape[1])
faiss.normalize_L2(emb_matrix)
faiss_index.add(emb_matrix)
#faiss.write_index(faiss_index, os.getcwd()) #I don't know how big is it
faiss.normalize_L2(visual_embs)
D, retrieved_indices = faiss_index.search(visual_embs, k= 2) 


retrieved_captions = []
retrieved_images = []
for cur_indices in retrieved_indices: #number of test examples here 1
    for retrieved_id in cur_indices:
        retrieved_path = path_array[retrieved_id]
        hf_indices = dataset['train']['image_url'].index(retrieved_path) # takes some time
        correct_caption = dataset['train']['caption'][hf_indices]
        correct_img = get_image_from_url(retrieved_path)
        retrieved_captions.append(correct_caption)
        retrieved_images.append(correct_img)

# we reverse the order so that the images with the highest similarity is closer to the question
retrieved_captions.reverse()
retrieved_images.reverse()

prompt = []
for img, caption in zip(retrieved_images, retrieved_captions):
    prompt.append(img)
    prompt.append(caption)

question = 'Give a caption'
answer = vqa_values[2]

prompt.append(question_image)
prompt.append(question)

#! Prompt and output
print("Model input : ", prompt)
model_outputs = model.generate_for_images_and_texts(prompt, num_words=20)
print("Model output :", model_outputs)
print("Ground truth :", answer)
print('\n')

