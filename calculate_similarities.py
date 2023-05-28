# %%
import numpy as np
import torch
import matplotlib.pyplot as plt

# %%
import sys
sys.path.append("..")
from tqdm import tqdm


# %% [markdown]
# Now let's calculate img2img similarities using feature extractor

# %%
from transformers import ViTImageProcessor, ViTModel
from src.image_classification.classification_utils import load_pickle, create_pickle
from fromage.data import load_real_mi
import os

processor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224-in21k') #https://huggingface.co/google/vit-large-patch16-224-in21k
model = ViTModel.from_pretrained('google/vit-large-patch16-224-in21k')
model.cuda()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_similarities(dict_model_input, num_ways):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224-in21k') #https://huggingface.co/google/vit-large-patch16-224-in21k
    model = ViTModel.from_pretrained('google/vit-large-patch16-224-in21k')
    model.to(device)

    img_sim = []
    print('now we will calculate img2img similarities')
    with torch.no_grad():
        for i in tqdm(dict_model_input):
            input = dict_model_input[i]

            images = [input[i] for i in range(0,len(input),2)]
            inputs = processor(images=images, return_tensors="pt")
            inputs=inputs.to(device)

            outputs = model(**inputs)
            pooler_output = outputs.pooler_output

            scores = [torch.nn.functional.cosine_similarity(pooler_output[i], pooler_output[-1],dim=0).item() for 
                      i in range(num_ways)]
            assert(len(scores)==num_ways)
            img_sim.append(scores)
    path = os.path.join('./src/image_classification/similarities/',f'sim_img2img_{num_ways}.pickle')
    
    create_pickle(path,img_sim)
    return img_sim

all_num_ways = [2,5]
for num_ways in all_num_ways:
    dict_model_input, dict_question_captions = load_real_mi(num_ways)
    img_sim = calculate_similarities(dict_model_input, num_ways)


