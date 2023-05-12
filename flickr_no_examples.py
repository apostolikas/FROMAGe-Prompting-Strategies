import pandas as pd 
from PIL import Image
import os
import numpy as np
from fromage import models
import matplotlib as plt
from torchmetrics.text.bert import BERTScore

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

vqa_data = split_dictionary(data_dict,1)

np.random.seed(0)
np.random.shuffle(vqa_data)
#vqa_data = vqa_data[:5]
#vqa_sublist = vqa_data[:20]

vqa_images_folder = './Flicker8k_Dataset/'


preds = []
targets = []
i = 0

for vqa_dict in vqa_data:
#vqa_dict = vqa_sublist[3]
    i+=1
    vqa_keys = list(vqa_dict.keys())
    vqa_values = list(vqa_dict.values())

#! Input1
# image1_path = vqa_keys[0]
# image1 = Image \
#         .open(os.path.join(vqa_images_folder,image1_path)) \
#         .resize((224, 224)) \
#         .convert('RGB')          
# caption1 = vqa_values[0]
# prompt1_for_ret = caption1[:-1] + '[RET]' 
# prompt1 = [image1, prompt1_for_ret]
# outs1 = model.generate_for_images_and_texts(prompt1, max_img_per_ret=1) 
# for out1 in outs1:
#         if type(out1) == str:
#             continue
#         elif type(out1) == list:
#             similar_image1 = out1[0]
#             # similar_image2 = out1[1]
#             # similar_image3 = out1[2]
#         else:
#             continue

#! Input2
# image2_path = vqa_keys[1]
# image2 = Image \
#     .open(os.path.join(vqa_images_folder,image2_path)) \
#     .resize((224, 224)) \
#     .convert('RGB')
# caption2 = vqa_values[1]
# prompt2_for_ret = caption2[:-1] + '[RET]'
# prompt2 = [image1, prompt2_for_ret]
# outs2 = model.generate_for_images_and_texts(prompt2, max_img_per_ret=1) 
# for out2 in outs2:
#         if type(out2) == str:
#             continue
#         elif type(out2) == list:
#             similar_image4 = out2[0]
#             # similar_image5 = out2[1]
#             # similar_image6 = out2[2]
#         else:
#             continue


    #! Question 
    question_image_path = vqa_keys[0]
    question_image = Image \
        .open(os.path.join(vqa_images_folder,question_image_path)) \
        .resize((224, 224)) \
        .convert('RGB')
    question_image.save('im1.jpg')

    prompt2_for_ret = 'Give a similar image to the previous one [RET]'
    prompt2 = [question_image, prompt2_for_ret]
    try:
        outs2 = model.generate_for_images_and_texts(prompt2, max_img_per_ret=2) 
        for out2 in outs2:
                if type(out2) == str:
                    continue
                elif type(out2) == list:
                    similar_image4 = out2[0]
                    similar_image4.save('sim1.jpg')
                    similar_image5 = out2[1]
                    similar_image5.save('sim2.jpg')
                    # similar_image6 = out2[2]
                else:
                    continue
        question = 'Caption the image.'
        answer = vqa_values[0]
        model_input = [similar_image4, similar_image5, question_image, question]

        #! Prompt and output
        print("Example ", i)
        model_outputs = model.generate_for_images_and_texts(model_input, num_words=15, max_num_rets=0)
        print("Model output :", model_outputs)
        print("Ground truth :", answer)
        print('\n')

        preds.append(model_outputs[0])
        targets.append(answer)
    except:
        continue

#! Evaluation metric
bertscore = BERTScore()
score = bertscore(preds, targets)
f1_score = score['f1']
precision_score = score['precision']
recall_score = score['recall']
print("Average F1 :" ,np.mean(f1_score))
print("Average Precision :" ,np.mean(precision_score))
print("Average Recall :" ,np.mean(recall_score))