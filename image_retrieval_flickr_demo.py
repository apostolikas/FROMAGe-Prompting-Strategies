import pandas as pd 
from PIL import Image
import os
import numpy as np
from fromage import models
from torchvision.transforms import ToTensor
import torch
import matplotlib.pyplot as plt


def split_dictionary(input_dict: dict, chunk_size: int) -> list:
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




if __name__ == '__main__':

    print('Loading the models...')
    # Load the FROMAGe model used in the paper.
    model_dir = './fromage_model/'
    model = models.load_fromage(model_dir)
    print('Models loaded successfully!')

    print('Loading data...')
    # Extended caption for text augmentation
    extended_captions = open("./src/image_retrieval_flickr/extended_captions.txt", "r")
    augmented_captions = [x.rstrip("\n") for x in extended_captions.readlines()]

    # Read the data for the image retrieval task
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
    ic_data = split_dictionary(data_dict,1)
    ic_data = [ic_data[17], ic_data[27]] 
    print('Data loaded successfully!')

    j=0

    print('Inference loop for 2 samples starts.')
    for ic_dict in ic_data:

        j+=1
        image_path = list(ic_dict.keys())[0]
        image = Image.open(os.path.join('./Flicker8k_Dataset/',image_path)).resize((224, 224)).convert('RGB') 
        caption_tuple = list(ic_dict.values())[0]

        try:

            # Retrieve image based on the original caption
            original_caption = caption_tuple[0]
            original_prompt = [original_caption[:-1] + ' [RET] ']
            model_output_orig = model.generate_for_images_and_texts(original_prompt, max_img_per_ret=1, max_num_rets=1, num_words=0)
            unaugmented_output = model_output_orig[-1][0]

            # Retrieve image based on the augmented caption
            augmented_caption = caption_tuple[1]
            augmented_prompt = [image, augmented_caption[:-1] + ' [RET] ']
            model_output = model.generate_for_images_and_texts(augmented_prompt, max_img_per_ret=1, max_num_rets=1, num_words=0)
            augmented_output = model_output[-1][0]

            # Evaluation metric for the two retrieved images (with and without using text augmentation)
            transform = ToTensor()
            model = model.to(torch.float16)
            with torch.no_grad():
                embedding_augmented = model.model.get_visual_embs((transform(augmented_output)).unsqueeze(0).half().cuda())
                embedding_unaugmented = model.model.get_visual_embs((transform(unaugmented_output)).unsqueeze(0).half().cuda())
                embedding_original = model.model.get_visual_embs((transform(image)).unsqueeze(0).half().cuda())
            model = model.bfloat16()

            similarity_score_unaugmented = torch.nn.functional.cosine_similarity(embedding_unaugmented.float(), embedding_original.float())
            similarity_score_augmented = torch.nn.functional.cosine_similarity(embedding_augmented.float(), embedding_original.float())
            print("Example ",j)
            print("Similarity of image retrieved without augmentation with the target image is : {:.3f}".format(similarity_score_unaugmented.mean(1).item()))
            print("Similarity of image retrieved using text-augmentation with the target image is : {:.3f}".format(similarity_score_augmented.mean(1).item()))
            print("\n")
            
        except:
            continue
        