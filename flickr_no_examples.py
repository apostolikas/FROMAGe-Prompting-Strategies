import pandas as pd 
from PIL import Image
import os
import numpy as np
from fromage import models
import matplotlib as plt
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

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

def cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


df = pd.read_csv('./Flickr8k_text/ExpertAnnotations.txt',delimiter='\t')
cropped_df = df.loc[df['expert1'] == 4]
cap_df = pd.read_csv('./Flickr8k_text/Flickr8k.token.txt',delimiter='\t')
cap_dict = pd.Series(cap_df.cap.values,index=cap_df.cap_id).to_dict()
data_dict = {}
for img_id, cap_id in zip(cropped_df.image_id, cropped_df.caption_id):
    caption = cap_dict[cap_id]
    data_dict[img_id] = caption


#Load model used in the paper.
model_dir = './fromage_model/'
model = models.load_fromage(model_dir)

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
lm = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

vqa_data = split_dictionary(data_dict,1)
np.random.seed(0)
np.random.shuffle(vqa_data)

vqa_images_folder = './Flicker8k_Dataset/'

unaugmented_scores = []
augmented_scores = []
i = 0

for vqa_dict in vqa_data:
    try:
        i+=1
        vqa_keys = list(vqa_dict.keys())
        vqa_values = list(vqa_dict.values())

        question_image_path = vqa_keys[0]
        question_image = Image \
            .open(os.path.join(vqa_images_folder,question_image_path)) \
            .resize((224, 224)) \
            .convert('RGB')
        #question_image.save('im1.jpg')
        question = 'Caption the image.'
        answer = vqa_values[0]
        # unaugmented prompt
        unaugmented_prompt = [question_image,question]
        unaugmented_output = model.generate_for_images_and_texts(unaugmented_prompt, num_words=15, max_num_rets=0)

        # augmented prompt
        prompt_for_ret = [question_image, 'Give a similar image to the previous one [RET]']
        augmented_outputs = model.generate_for_images_and_texts(prompt_for_ret, max_img_per_ret=2) 
        for out in augmented_outputs:
                if type(out) == str:
                    continue
                elif type(out) == list:
                    similar_image1 = out[0]
                    #similar_image4.save('sim1.jpg')
                    similar_image2 = out[1]
                    #similar_image5.save('sim2.jpg')
                else:
                    continue

        model_augmented_input = [similar_image1, similar_image2, question_image, question]
        augmented_output = model.generate_for_images_and_texts(model_augmented_input, num_words=15, max_num_rets=0)

        # Compute similarity metric
        encoded_unaugmented_input = tokenizer(unaugmented_output, padding=True, truncation=True, return_tensors='pt')
        encoded_augmented_input = tokenizer(augmented_output, padding=True, truncation=True, return_tensors='pt')
        encoded_target_input = tokenizer(answer, padding=True, truncation=True, return_tensors='pt')

        with torch.no_grad():
            model_unaugmented_output = lm(**encoded_unaugmented_input)
            model_augmented_output = lm(**encoded_augmented_input)
            model_target_output = lm(**encoded_target_input)

        unaugmented_embeddings = F.normalize(mean_pooling(model_unaugmented_output, encoded_unaugmented_input['attention_mask']), p=2, dim=1)
        augmented_embeddings = F.normalize(mean_pooling(model_augmented_output, encoded_augmented_input['attention_mask']), p=2, dim=1)
        target_embeddings = F.normalize(mean_pooling(model_target_output, encoded_target_input['attention_mask']), p=2, dim=1)

        augmented_score = cos_sim(augmented_embeddings, target_embeddings)
        augmented_scores.append(augmented_score.item())
        unaugmented_score = cos_sim(unaugmented_embeddings, target_embeddings)
        unaugmented_scores.append(unaugmented_score.item())


        print("Example ", i)
        print("Caption without using any augmentation", unaugmented_output, "| Cos sim with target :",unaugmented_score.item())
        print("Caption using augmentation", augmented_output, "| Cos sim with target :",augmented_score.item())
        print("Ground truth :", answer)
        print('---------------------------------')

    except:
        continue

print("Average Cosine Similarity with target using visual augmentations :",np.mean(augmented_scores))
print("Average Cosine Similarity with target without augmentations :",np.mean(unaugmented_scores))