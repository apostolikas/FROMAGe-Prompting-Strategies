import pandas as pd 
from PIL import Image
import os
import numpy as np
from fromage import models
import matplotlib as plt

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

# Load model used in the paper.
model_dir = './fromage_model/'
model = models.load_fromage(model_dir)

vqa_data = split_dictionary(data_dict,3)

np.random.seed(0)
np.random.shuffle(vqa_data)
vqa_sublist = vqa_data[:15]

vqa_images_folder = './Flicker8k_Dataset/'

num_examples = 3

for i in range(num_examples, len(vqa_sublist), num_examples):

    model_input = []

    for j in range(i-num_examples, i):

        vqa_dict = vqa_sublist[j]
        vqa_keys = list(vqa_dict.keys())
        vqa_values = list(vqa_dict.values())

        #! Input 1 
        image1_path = vqa_keys[0]
        image1 = Image \
            .open(os.path.join(vqa_images_folder,image1_path)) \
            .resize((224, 224)) \
            .convert('RGB')
        caption1 = vqa_values[0]
        # Retrieve similar image 
        prompt1_for_ret = caption1[:-1] + '[RET]' 
        prompt1 = [image1, prompt1_for_ret]
        outs1 = model.generate_for_images_and_texts(prompt1, max_img_per_ret=1) 
        for out1 in outs1:
                if type(out1) == str:
                    continue
                elif type(out1) == list:
                    similar_image1 = out1[0]
                    # similar_image2 = out1[1]
                    # similar_image3 = out1[2]
                else:
                    continue
        # Give as input the original image and the retrieved one

        #! Input 2
        image2_path = vqa_keys[1]
        image2 = Image \
            .open(os.path.join(vqa_images_folder,image2_path)) \
            .resize((224, 224)) \
            .convert('RGB')
        caption2 = vqa_values[1]
        prompt2_for_ret = caption2[:-1] + '[RET]' 
        prompt2 = [image2, prompt2_for_ret]
        outs2 = model.generate_for_images_and_texts(prompt2, max_img_per_ret=1) 
        for out2 in outs2:
                if type(out2) == str:
                    continue
                elif type(out2) == list:
                    similar_image4 = out2[0]
                    # similar_image2 = out1[1]
                    # similar_image3 = out1[2]
                else:
                    continue


        question_image_path = vqa_keys[2]
        question_image = Image \
            .open(os.path.join(vqa_images_folder,question_image_path)) \
            .resize((224, 224)) \
            .convert('RGB')
        
        #! Question and Answer - Example i
        question = 'Give a caption for the image'
        answer = vqa_values[2]
        #model_input += [ question_image, 'Q: ' + question + ' A: ' + answer ]
        model_input += [ image1, similar_image1, caption1, image2, similar_image4, question_image, question, answer]


    
    #! Test scenario
    vqa_dict = vqa_sublist[i]
    vqa_keys = list(vqa_dict.keys())
    vqa_values = list(vqa_dict.values())

    #! Image 1
    image1_path = vqa_keys[0]
    image1 = Image \
        .open(os.path.join(vqa_images_folder,image1_path)) \
        .resize((224, 224)) \
        .convert('RGB')
    caption1 = vqa_values[0]
    prompt1_for_ret = caption1[:-1] + '[RET]' 
    prompt1 = [image1, prompt1_for_ret]
    outs1 = model.generate_for_images_and_texts(prompt1, max_img_per_ret=1) 
    for out1 in outs1:
            if type(out1) == str:
                continue
            elif type(out1) == list:
                similar_image1 = out1[0]
                # similar_image2 = out1[1]
                # similar_image3 = out1[2]
            else:
                continue

    #! Image 2 
    image2_path = vqa_keys[1]
    image2 = Image \
        .open(os.path.join(vqa_images_folder,image2_path)) \
        .resize((224, 224)) \
        .convert('RGB')
        
    caption2 = vqa_values[1]
    caption2 = vqa_values[1]
    prompt2_for_ret = caption2[:-1] + '[RET]' 
    prompt2 = [image2, prompt2_for_ret]
    outs2 = model.generate_for_images_and_texts(prompt2, max_img_per_ret=1) 
    for out2 in outs2:
            if type(out2) == str:
                continue
            elif type(out2) == list:
                similar_image4 = out2[0]
                # similar_image2 = out1[1]
                # similar_image3 = out1[2]
            else:
                continue

    #! Question without answer
    question_image_path = vqa_keys[2]
    question_image = Image \
        .open(os.path.join(vqa_images_folder,question_image_path)) \
        .resize((224, 224)) \
        .convert('RGB')    
    question = 'Give a caption for the image'
    answer = vqa_values[2]
    #model_input += [ question_image, 'Q: ' + question + ' A: ' ]
    model_input += [ image1, similar_image1, caption1, image2, similar_image4, question_image, question]


    #! Model output
    print("Model input : ", model_input)
    model_outputs = model.generate_for_images_and_texts(model_input, num_words=10)
    print("Model output :", model_outputs)
    print("Ground truth :", answer)
    print('\n\n')
