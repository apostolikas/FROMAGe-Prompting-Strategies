import os
import numpy as np
from fromage import models
import json 
from PIL import Image


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
    caption1 = vqa_dict['caption_1']

    # Input - Output pair used as input for the model
    image2_path = vqa_dict['image_2']
    image2 = Image.open(os.path.join(vist_images_folder,image2_path))
    caption2 = vqa_dict['caption_2']

    # Prompt 
    question = vqa_dict['question']
    question_image_path = vqa_dict['question_image']
    question_image = Image.open(os.path.join(vist_images_folder,question_image_path))

    # Answer
    answer = vqa_dict['answer']

    model_input = [image1, caption1, image2, caption2, question_image, question]    
    model_outputs = model.generate_for_images_and_texts(model_input)
    print("Caption 1: ", caption1, "\tCaption 2: ", caption2, "\tQuestion: ",question)
    print("Model output :",model_outputs)
    print("Ground truth :",answer)
