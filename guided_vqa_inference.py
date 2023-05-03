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
vqa_sublist = vqa_data[:2]

vist_images_folder = './guided_vqa'


for vqa_dict in vqa_sublist:

    # Input - Output pair used as input for the model
    image1_path = vqa_dict['image_1']
    image1 = Image.open(os.path.join(vist_images_folder,image1_path))
    image1 = image1.convert('F')
    caption1 = vqa_dict['caption_1'] #+ ' [EOS]'

    # Input - Output pair used as input for the model
    image2_path = vqa_dict['image_2']
    image2 = Image.open(os.path.join(vist_images_folder,image2_path))
    image2 = image2.convert('F')
    caption2 = vqa_dict['caption_2'] #+ ' [EOS]'

    # Prompt 
    question = vqa_dict['question'] #+ ' [EOS]'
    question_image_path = vqa_dict['question_image']
    question_image = Image.open(os.path.join(vist_images_folder,question_image_path))
    question_image = question_image.convert('F')

    # # Answer
    answer = vqa_dict['answer']
    # #model_input = [caption1, caption2, question]    

    # model_input = [image1, caption1, image2, caption2, question_image, question] 
    # print("Model input : ",model_input)
    # model_outputs = model.generate_for_images_and_texts(model_input, num_words=32)
    # print("Caption 1: ", caption1, "\nCaption 2: ", caption2, "\nQuestion: ",question)
    # print("Model output :",model_outputs)
    # print("Ground truth :",answer)

    input_prompts = [image1, caption1, image2, caption2, question_image, question] 
    input_context = []
    all_outputs = []
    text = ''
    for p in input_prompts:
        # Add Q+A prefixes for prompting. This is helpful for generating dialogue.
        if type(p) == str:
            text += 'Q: ' + p + '\nA:' 
        # Concatenate image and text.
        model_prompt = input_context + [text]
        model_outputs = model.generate_for_images_and_texts(
            model_prompt, num_words=32)
        text += ' '.join([s for s in model_outputs if type(s) == str]) + '\n'
        
        # Format outputs.
        if type(model_outputs[0]) == str:
            model_outputs[0] = 'FROMAGe:  ' + model_outputs[0]
        else:
            # Image output
            model_outputs = ['FROMAGe:  '] + model_outputs[0]
        if type(p) == str:
            all_outputs.append('Input:     ' + p)
        all_outputs.extend(model_outputs)

    print("Question :" ,question)
    display_interleaved_outputs(model_outputs)
    print("Ground truth :" ,answer)
