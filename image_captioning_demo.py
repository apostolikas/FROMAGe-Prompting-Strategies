import pandas as pd 
from PIL import Image
import os
import numpy as np
from fromage import models
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src.image_captioning.flickr_visual_augmentation import split_dictionary, cos_sim, mean_pooling


def compare_embeddings(augmented_caption, unaugmented_caption, answer):

    # Tokenize the input
    encoded_unaugmented_input = tokenizer(unaugmented_caption, padding=True, truncation=True, return_tensors='pt')
    encoded_augmented_input = tokenizer(augmented_caption, padding=True, truncation=True, return_tensors='pt')
    encoded_target_input = tokenizer(answer, padding=True, truncation=True, return_tensors='pt')

    # FF through the model - generate embeddings
    with torch.no_grad():
        model_unaugmented_output = lm(**encoded_unaugmented_input)
        model_augmented_output = lm(**encoded_augmented_input)
        model_target_output = lm(**encoded_target_input)

    # Process the embeddings
    unaugmented_embeddings = F.normalize(mean_pooling(model_unaugmented_output, encoded_unaugmented_input['attention_mask']), p=2, dim=1)
    augmented_embeddings = F.normalize(mean_pooling(model_augmented_output, encoded_augmented_input['attention_mask']), p=2, dim=1)
    target_embeddings = F.normalize(mean_pooling(model_target_output, encoded_target_input['attention_mask']), p=2, dim=1)

    # Compute the cos sim 
    augmented_score = cos_sim(augmented_embeddings, target_embeddings)
    unaugmented_score = cos_sim(unaugmented_embeddings, target_embeddings)

    return augmented_score, unaugmented_score



if __name__ == '__main__':

    print('Loading the models...')
    # Load the FROMAGe model used in the paper.
    model_dir = './fromage_model/'
    model = models.load_fromage(model_dir)

    # Load the all-MiniLM-L6-v2 model to compare the text embeddings of the output with those of the original caption.
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    lm = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    print('Models loaded successfully!')

    print('Loading data...')
    # Prepare data for inference
    df = pd.read_csv('./Flickr8k_text/ExpertAnnotations.txt',delimiter='\t')
    cropped_df = df.loc[df['expert1'] == 4]
    cap_df = pd.read_csv('./Flickr8k_text/Flickr8k.token.txt',delimiter='\t')
    cap_dict = pd.Series(cap_df.cap.values,index=cap_df.cap_id).to_dict()
    data_dict = {}
    for img_id, cap_id in zip(cropped_df.image_id, cropped_df.caption_id):
        caption = cap_dict[cap_id]
        data_dict[img_id] = caption
    flickr_data = split_dictionary(data_dict,1)
    flickr_images_folder = './Flicker8k_Dataset/'
    flickr_data = [flickr_data[70], flickr_data[32]]
    print('Data loaded successfully!')

    print('Inference loop for 2 samples starts.\n')
    # Inference loop for 2 samples
    for flickr_dict in flickr_data:
        try:

            flickr_keys = list(flickr_dict.keys())
            flickr_values = list(flickr_dict.values())

            # Load query image & caption
            question_image_path = flickr_keys[0]
            question_image = Image.open(os.path.join(flickr_images_folder,question_image_path)).resize((224, 224)).convert('RGB')
            question = 'Caption the image.'
            answer = flickr_values[0]

            # Generate caption using the unaugmented prompt
            unaugmented_prompt = [question_image,question]
            unaugmented_caption = model.generate_for_images_and_texts(unaugmented_prompt, num_words=15, max_num_rets=0)


            # Use query image to retrieve two similar ones
            prompt_for_ret = [question_image, 'Give a similar image to the previous one [RET]']
            augmented_outputs = model.generate_for_images_and_texts(prompt_for_ret, max_img_per_ret=2) 
            for out in augmented_outputs:
                    if type(out) == str:
                        continue
                    elif type(out) == list:
                        similar_image1 = out[0]
                        similar_image2 = out[1]
                    else:
                        continue

            model_augmented_input = [similar_image1, similar_image2, question_image, question]
            augmented_caption = model.generate_for_images_and_texts(model_augmented_input, num_words=15, max_num_rets=0)

            augmented_score, unaugmented_score = compare_embeddings(augmented_caption, unaugmented_caption, answer)

            print("Caption without using augmentation :", unaugmented_caption, "-> Cos sim with target : {:.3f}".format(unaugmented_score.item()))
            print("Caption using visual augmentation :", augmented_caption, "-> Cos sim with target : {:.3f}".format(augmented_score.item()))
            print("Ground truth :", answer)
            print("\n")
            
        except:
            continue