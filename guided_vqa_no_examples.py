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
vqa_sublist = vqa_data[30:34]
vist_images_folder = './guided_vqa'
i=0
for vqa_dict in vqa_sublist:
    #vqa_dict = vqa_sublist[14]
    
    # ------------------------------ #
    i+=1
    #! Image 1
    image1_path = vqa_dict['image_1']
    image1 = Image \
        .open(os.path.join(vist_images_folder,image1_path)) \
        .resize((224, 224)) \
        .convert('RGB')
    caption1 = vqa_dict['caption_1']

    # ------------------------------ #

    #! Input2
    image2_path = vqa_dict['image_2']
    image2 = Image \
        .open(os.path.join(vist_images_folder,image2_path)) \
        .resize((224, 224)) \
        .convert('RGB')
    caption2 = vqa_dict['caption_2']

    # ------------------------------ #

    #! Question 
    question_image_path = vqa_dict['question_image']
    question_image = Image \
        .open(os.path.join(vist_images_folder,question_image_path)) \
        .resize((224, 224)) \
        .convert('RGB')
    question_image.save(str(i)+'im1.jpg')
    #question = " In the background, " + vqa_dict['question']
    #question = vqa_dict['question'] + ' Background'
    question = 'Caption the image.'
    answer = vqa_dict['answer']

    prompt2_for_ret = 'Give a similar image to the previous one [RET]'
    prompt2 = [question_image, prompt2_for_ret]
    outs2 = model.generate_for_images_and_texts(prompt2, max_img_per_ret=2) 
    for out2 in outs2:
            if type(out2) == str:
                continue
            elif type(out2) == list:
                similar_image4 = out2[0]
                similar_image4.save(str(i)+'sim1.jpg')
                similar_image5 = out2[1]
                similar_image5.save(str(i)+'sim2.jpg')
                # similar_image6 = out2[2]
            else:
                continue

    # ------------------------------ #

    #! Model output
    model_input = [ image1, 
                caption1, 
                image2, 
                caption2, 
                question_image, 
                similar_image4,
                similar_image5,
                #question]
                'Q: ' + question] #+ ' A: ']


    #! Prompt and output
    print("Model input : ", model_input)
    model_outputs = model.generate_for_images_and_texts(model_input, num_words=12, max_num_rets=0)
    print("Model output :", model_outputs)
    print("Ground truth :", answer)
    print('\n')
