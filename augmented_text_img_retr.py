import pandas as pd 
from PIL import Image
import os
import numpy as np
from fromage import models
import matplotlib as plt
from torchvision.transforms import ToTensor
from torchmetrics.multimodal import CLIPScore


extended_captions = open("extended_captions.txt", "r")
augmented_captions = [x.rstrip("\n") for x in extended_captions.readlines()]

#! Load and preprocess data
df = pd.read_csv('./Flickr8k_text/ExpertAnnotations.txt',delimiter='\t')
cropped_df = df.loc[df['expert1'] == 4]
img_cap_list = list(zip(cropped_df.image_id, cropped_df.caption_id))
cap_df = pd.read_csv('./Flickr8k_text/Flickr8k.token.txt',delimiter='\t')
cap_dict = pd.Series(cap_df.cap.values,index=cap_df.cap_id).to_dict()
data_dict = {}
for img_id, cap_id in zip(cropped_df.image_id, cropped_df.caption_id):
    caption = cap_dict[cap_id]
    data_dict[img_id] = caption 

for i,content in enumerate(zip(data_dict.keys(),data_dict.values())):
    data_dict[content[0]] = (content[1],augmented_captions[i])

#! function to make a dictionary -> list of dictionaries (len(list)=chunk_size) (in our case 1)
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

#! final data for task
num_of_samples = 200
ic_data = split_dictionary(data_dict,1)
np.random.seed(0)
np.random.shuffle(ic_data)
ic_data = ic_data[:num_of_samples]

#! load model
model_dir = './fromage_model/'
model = models.load_fromage(model_dir)


#! function to make a dictionary -> list of dictionaries (len(list)=chunk_size) (in our case 1)
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

#! final data for task
ic_data = split_dictionary(data_dict,1)
ic_data = ic_data[:2]

#! load model
model_dir = './fromage_model/'
model = models.load_fromage(model_dir)

i=0 # counter for print 
metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
scores = []

for ic_dict in ic_data:

    i+=1
    image_path = list(ic_dict.keys())[0]
    image = Image \
        .open(os.path.join('./Flicker8k_Dataset/',image_path)) \
        .resize((224, 224)) \
        .convert('RGB') 
    caption = list(ic_dict.values())[0]

    # Retrieve image based on the augmented caption
    augmented_caption = list(ic_dict.values())[1]
    augmented_prompt = [augmented_caption[:-1] + ' [RET] ']
    model_output = model.generate_for_images_and_texts(augmented_prompt, max_img_per_ret=1, max_num_rets=1)
    retrieved_image = model_output[0]
    retrieved_image.save(str(i)+'ret_img.jpg')

    # Compare augmented caption with original and retrieved image
    transform = ToTensor()
    score_with_original_image = metric(transform(image), augmented_caption)
    score_with_new_image = metric(transform(retrieved_image), augmented_caption)
    print("Example ", i, "\t Original img score :", score_with_original_image.detach().item,"%", "\t New img score :", score_with_new_image.detatch().item,"%")
