import os
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import torch
import torch.nn.functional as F
import pandas as pd
from fromage import models
from visual_storytelling import story_in_sequence
import openai
import time 
import matplotlib.pyplot as plt
import numpy as np

openai.api_key = 'API_KEY'

assert openai.api_key != 'API_KEY', "Fill in your personal API key"

# Initialize the CLIP model and processor
model_dir = './fromage_model/'
model = models.load_fromage(model_dir)
model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
image_directory = "datasets/Archive/Flicker8k_Dataset"

df_images = pd.read_csv('filtered_capt.csv')
df_images.drop_duplicates(subset='cap', inplace=True)


def encode_image(image_path):
    # Open the image file
    with Image.open(image_path) as img:
        # Convert the image to a tensor and move it to the GPU
        image = processor(images=img, return_tensors="pt").to("cuda")
        with torch.no_grad():
            image_features = model_clip.get_image_features(**image)
    return image_features


def caption_this(model, model_output):
    prompt =  [model_output] + ['Q: Give a descriptive caption, A:']
    print('caption_prompt:',prompt)
    differences = model.generate_for_images_and_texts(prompt, max_img_per_ret=0, max_num_rets=0, num_words=15)
    print('differences: ',differences)
    return str(differences)


def gpt_addition(original_prompt,differences):
    messages = [
    {"role": "system", "content": "You receive a string, which includes caption 1 and caption 2. \
    Caption 1 is the ground truth, caption 2 is a caption of a different image. We prompt a multimodal to get an image\
    as close as possible to the ground truth. Therefore we want to have a new prompt that highlights the differences between\
    The groundtruth and caption 2. Consider descriptive properties of words. So: a dog runs in a green field and a dog runs in the grass, contain the same elements.\
    Please return just the prompt that you come up with. No introduction, no explanation, focus on major differences.\
    For example:\n\
    Caption 1: 'This is a photo of a dog'\n\
    Caption 2: 'This is a photo of a cat'\n\
    'Reply: 'The image should not contain a cat.'"
    
    }
    ]
    while True: 
        try: 
            if differences:
                messages.append({"role": "user", "content":f'caption 1: {original_prompt}, caption 2: {differences}'})
            chat = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=messages
            )
            reply = chat.choices[0].message.content
            #print('reply:', reply)
            return reply
        except: 
            time.sleep(1)



standard_path = 'visual_storytelling/images/fromage_dialogue/'
#json_df = pd.read_json('similarity_prompts.json') 
df_filtered = df_images[~df_images['image_id'].isin(json_df['original_image'])]
results = []
for index, row in df_images[:64].iterrows():
    print('i: ', index )
    p = "find an image that contains: " +row['cap']
    ret_scale_factor = 1.5  # Increase this hyperparameter to upweight the probability of FROMAGe returning an image.
    input_context = []
    text = ''
    all_outputs = []
    experiment_result = []
    image_path = os.path.join(image_directory, row['image_id'])
    ground_features = encode_image(image_path)
    
    for experiment in range(5):
        # Add Q+A prefixes for prompting. This is helpful for generating dialogue.
        text += 'Q: ' + p + '\nA:'
        # Concatenate image and text.
        model_prompt = input_context + [text]
        print('model_prompt:' , model_prompt)
        model_outputs = model.generate_for_images_and_texts(
            model_prompt, num_words=32, ret_scale_factor=ret_scale_factor, max_num_rets=1)
        #print('the list :', [s for s in model_outputs if type(s) == str])
        text += ' '.join([s for s in model_outputs if type(s) == str]) + '\n'
        #print('text:', text)
        # Format outputs.
        if type(model_outputs[0]) == str:
            print('model 0 is string')
            print(len(model_outputs[-1]))
            model_outputs[0] = 'FROMAGe:  ' + model_outputs[0]
            try: 

                img_name = f'augmenting_set_{index}_{experiment}.png'
                output_path = standard_path + img_name
                img  = model_outputs[-1][0]
                img.save(output_path)
                print('img saved')
            except:
                print('excepting')
                story_in_sequence.save_story_predictions('temp', model_outputs[-1][0], one_img_per_ret=True)

        else:
            # Image output
            
            story_in_sequence.save_story_predictions('temp', model_outputs[-1][0], one_img_per_ret=True)
            #story_in_sequence.save_story_predictions('', model_outputs[0], one_img_per_ret=True) f'dialogue_{index}'
            model_outputs = ['FROMAGe:  '] + model_outputs[0]
        all_outputs.append('Input:     ' + p)
        all_outputs.extend(model_outputs)
        #img2 = Image.open('temp.png').convert("RGB")
        # Convert the images to tensors and move them to the GPU
        #ground_image = processor(images=img1, return_tensors="pt").to("cuda")
        #retrieved_image = processor(images=img2, return_tensors="pt").to("cuda")
        output_features = encode_image(output_path)
        cos_sim =  F.cosine_similarity(ground_features, output_features)
        experiment_result.append(float(cos_sim))
        differences = caption_this(model,model_outputs[-1][0])

        extended_dialogue = gpt_addition(row['cap'] ,differences) 
        print('extended dialogue:' , extended_dialogue)
        p = extended_dialogue

    #print('displaying:')
    #img_name = f'augmenting_set_{index}'
    #display_interleaved_outputs(all_outputs, img_name )


    result = {
        "original_image": row['image_id'],
        "original_caption": row['cap'],  
        "similarity_augmenting":  experiment_result
    }
    
    results.append(result)
    df_result = pd.DataFrame(results)
    df_result.to_csv('prompt_similarity_1.csv')

