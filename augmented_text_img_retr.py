import pandas as pd 
from PIL import Image
import os
import numpy as np
from fromage import models
import matplotlib as plt


#! Load and preprocess data
df = pd.read_csv('./Flickr8k_text/ExpertAnnotations.txt',delimiter='\t')
cropped_df = df.loc[df['expert1'] == 1]
img_cap_list = list(zip(cropped_df.image_id, cropped_df.caption_id))
cap_df = pd.read_csv('./Flickr8k_text/Flickr8k.token.txt',delimiter='\t')
cap_dict = pd.Series(cap_df.cap.values,index=cap_df.cap_id).to_dict()
data_dict = {}
for img_id, cap_id in zip(cropped_df.image_id, cropped_df.caption_id):
    caption = cap_dict[cap_id]
    data_dict[img_id] = caption


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

i=0 # counter for print 

images_dict = {}

for ic_dict in ic_data:
    i+=1
    image = list(ic_dict.keys())[0]
    caption = list(ic_dict.values())[0]

    #! First check if the model retrieves a good image based on the caption
    prompt = [caption[:-1] + ' [RET] ']
    retrieved_image = model.generate_for_images_and_texts(prompt, max_img_per_ret=1, max_num_rets=1)

    # Optional - Qualitative check on retrieved_image
    # retrieved_image[0].save(str(i) + 'retrieved_image.jpg')

    #TODO Augment caption 

    new_prompt = [augmented_caption[:-1] + ' [RET] ']
    model_output = model.generate_for_images_and_texts(prompt, max_img_per_ret=1, max_num_rets=1)
    new_retrieved_image = model_output[0]
    # Optional - Qualitative check on new retrieved_image
    # new_retrieved_image.save(str(i) + 'new_retrieved_image.jpg')

    images_dict[caption] = zip(image, new_retrieved_image)


#TODO ClipScore 
