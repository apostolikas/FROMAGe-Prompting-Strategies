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
ic_data = split_dictionary(data_dict,1)
#ic_data = ic_data[:50]

#! load model
model_dir = './fromage_model/'
model = models.load_fromage(model_dir)

i=0 # counter for print 
metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
scores_orig_img_orig_cap = []
scores_ret_img_orig_cap = []
scores_orig_img_aug_cap = []
scores_ret_img_aug_cap = []

for ic_dict in ic_data:

    i+=1
    image_path = list(ic_dict.keys())[0]
    image = Image \
        .open(os.path.join('./Flicker8k_Dataset/',image_path)) \
        .resize((224, 224)) \
        .convert('RGB') 
    caption_tuple = list(ic_dict.values())[0]

    # Retrieve image based on the augmented caption
    original_caption = caption_tuple[0]
    original_prompt = [original_caption[:-1] + ' [RET] ']
    model_output_orig = model.generate_for_images_and_texts(original_prompt, max_img_per_ret=1, max_num_rets=1, num_words=0)
    orig_img = model_output_orig[-1][0]
    #orig_img.save(str(i)+'_orig_img.jpg')

    augmented_caption = caption_tuple[1]
    augmented_prompt = [augmented_caption[:-1] + ' [RET] ']
    model_output = model.generate_for_images_and_texts(augmented_prompt, max_img_per_ret=1, max_num_rets=1, num_words=0)
    retrieved_image = model_output[-1][0]
    #retrieved_image.save(str(i)+'_ret_img.jpg')


    # Compare augmented caption with original and retrieved image
    transform = ToTensor()
    score_orig_cap_orig_img = metric(transform(image), original_caption)
    score_aug_cap_orig_img = metric(transform(image), augmented_caption)
    score_ret_img_orig_cap = metric(transform(retrieved_image), original_caption)
    score_ret_img_augm_cap = metric(transform(retrieved_image), augmented_caption)

    scores_orig_img_orig_cap.append(score_orig_cap_orig_img.detach().item())
    scores_ret_img_aug_cap.append(score_ret_img_augm_cap.detach().item())
    scores_orig_img_aug_cap.append(score_aug_cap_orig_img.detach().item())
    scores_ret_img_orig_cap.append(score_ret_img_orig_cap.detach().item())


    print("Example ", i)
    print("Original Caption  - Original img score :", score_orig_cap_orig_img.detach().item() ,"%")
    print("Original Caption  - Retrieved img score :", score_ret_img_orig_cap.detach().item() ,"%")
    print("Augmented Caption - Original img score :", score_aug_cap_orig_img.detach().item() ,"%")
    print("Augmented Caption - Retrieved img score :", score_ret_img_augm_cap.detach().item() ,"%")
    print("------------------------------------------------")

print("\nIn total:")
print("The average original caption - original img score is ", np.mean(scores_orig_img_orig_cap),"%")
print("The average original caption - retrieved img score is ", np.mean(scores_ret_img_orig_cap),"%")
print("The average augmented caption - original img score is ", np.mean(scores_orig_img_aug_cap),"%")
print("The average augmented caption - retrieved img score is ", np.mean(scores_ret_img_aug_cap),"%")
