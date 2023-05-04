import os
import numpy as np
from fromage import models
import json 
import copy
from PIL import Image
import matplotlib as plt

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

# Load data
with open('./guided_vqa/guided_vqa_shots_1_ways_2_all_questions.json', 'r') as f:
    vqa_data = json.load(f)

# Take a few instances (the data is a list of dictionaries)
np.random.seed(0)
np.random.shuffle(vqa_data)
vqa_sublist = vqa_data[:15]

vist_images_folder = './guided_vqa'

num_examples = 5

for i in range(num_examples, len(vqa_sublist), num_examples):

    model_input = []

    
    for j in range(i-num_examples, i):

        vqa_dict = vqa_sublist[j]

        # image1_path = vqa_dict['image_1']
        # image1 = Image \
        #     .open(os.path.join(vist_images_folder,image1_path)) \
        #     .resize((224, 224)) \
        #     .convert('RGB')
            
        # caption1 = vqa_dict['caption_1']

        # image2_path = vqa_dict['image_2']
        # image2 = Image \
        #     .open(os.path.join(vist_images_folder,image2_path)) \
        #     .resize((224, 224)) \
        #     .convert('RGB')
        
        # caption2 = vqa_dict['caption_2']

        question_image_path = vqa_dict['question_image']
        question_image = Image \
            .open(os.path.join(vist_images_folder,question_image_path)) \
            .resize((224, 224)) \
            .convert('RGB')

        question = vqa_dict['question']
        
        answer = vqa_dict['answer']

        model_input += [ question_image, 'Q: ' + question + ' A: ' + answer ]
    
    
    vqa_dict = vqa_sublist[i]

    # image1_path = vqa_dict['image_1']
    # image1 = Image \
    #     .open(os.path.join(vist_images_folder,image1_path)) \
    #     .resize((224, 224)) \
    #     .convert('RGB')
        
    # caption1 = vqa_dict['caption_1']

    # image2_path = vqa_dict['image_2']
    # image2 = Image \
    #     .open(os.path.join(vist_images_folder,image2_path)) \
    #     .resize((224, 224)) \
    #     .convert('RGB')
    
    # caption2 = vqa_dict['caption_2']

    question_image_path = vqa_dict['question_image']
    question_image = Image \
        .open(os.path.join(vist_images_folder,question_image_path)) \
        .resize((224, 224)) \
        .convert('RGB')

    question = vqa_dict['question']

    answer = vqa_dict['answer']

    model_input += [ question_image, 'Q: ' + question + ' A:']

    print("Model input : ", model_input)

    model_outputs = model.generate_for_images_and_texts(model_input, num_words=10)
    
    print("Model output :", model_outputs)
    print("Ground truth :", answer)
    print('\n\n')
