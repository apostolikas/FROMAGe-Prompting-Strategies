import json
import os
import io
import base64
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from fromage.utils import get_image_from_url

images_path = 'visual_dialog/VisualDialog_val2018/'""
dialogs_csv_path = 'dialogs.csv'

# CREATE DIALOG DATAFRAME

def create_stories(dialogs_path:str) -> pd.DataFrame:

    # Read the JSON file and load it into a Python object
    with open(dialogs_path, 'r') as f:
        dialogs = json.load(f)

    dialog_df = pd.DataFrame(columns=['id', 'round', 'image_id', 'question_id', 'question', 'answer_id', 'answer'])

    # LOAD STORY-IN-SEQUENCE JSON
    for i in range(len(dialogs['data']['dialogs'])):
        id = i

        image_id = dialogs['data']['dialogs'][i]['image_id']
        # get_image(image_id)

        for j in range(len(dialogs['data']['dialogs'][i]['dialog'])):
                round = j+1
                answer_index = dialogs['data']['dialogs'][i]['dialog'][j]['answer']
                question_index = dialogs['data']['dialogs'][i]['dialog'][j]['question']

                question = dialogs['data']['questions'][question_index]
                answer = dialogs['data']['answers'][answer_index]

                new_row = {'id': id, 'round': round, 'image_id': image_id, 'question_id': question_index, 
                           'question': question, 'answer_id': answer_index, 'answer': answer}

                # Append the new row to the DataFrame using the loc method
                dialog_df.loc[len(dialog_df)] = new_row
    
    return dialog_df

def save_stories(stories_df: pd.DataFrame, stories_csv_path: str) -> None:
    stories_df.to_csv(stories_csv_path, encoding='utf8', index=False)

def load_dialogs(stories_csv_path: str) -> pd.DataFrame:
    return pd.read_csv(stories_csv_path, encoding='utf8', dtype=str)


# Get image file by image id
def get_image(image_id, show_image=False):
    # Loop through all the files in the folder
    for file_name in os.listdir(images_path):
        # Check if the file name ends with the number and .jpg suffix
        if file_name.endswith(str(image_id) + '.jpg'):
            # Create the full path to the matching file
            # Do something with the file, e.g. print its path
            image_path = os.path.join(images_path, file_name)
            image = Image.open(image_path).resize((224, 224)).convert('RGB')
            if show_image:
                plt.figure(figsize=(3, 3))
                plt.axis('off')
                plt.imshow(np.array(image))
                plt.show()
                        
            return image
    raise ValueError(f"Value {image_id} not found in list {images_path}")

def show_image(image):
    plt.figure(figsize=(3, 3))
    plt.axis('off')
    plt.imshow(np.array(image))
    plt.show()

        
# Create prompts from dataframe
def get_prompt_list(dialogs_df, num_rows, prompt_length, ret_img=True):
    text = ""
    dialog, url_dialog, input_dialog_list, url_dialog_list = [], [], [], []
    for i in range(num_rows-1):
        if int(dialogs_df['round'][i]) == 1:
            image_id = dialogs_df['image_id'][i]
            caption = dialogs_df['caption'][i]
            text += caption
            img = get_image(image_id)
            # dialog.append(img)
        if int(dialogs_df['round'][i]) <= prompt_length:
            text += f"Q: {dialogs_df['question'][i]}, "
            text += f"A: {dialogs_df['answer'][i]}, "
        if dialogs_df['id'][i+1] != dialogs_df['id'][i]:
            text = text[:-2]
            if ret_img == True: 
                dialog.append(text)
                dialog.append(img)
                url_dialog.append(text)
                url_dialog.append(image_id)
            else:
                dialog.append(img)
                dialog.append(text)
                url_dialog.append(image_id)
                url_dialog.append(text)

            # Append the dialog when a new dialog will start next
            input_dialog_list.append(dialog)
            url_dialog_list.append(url_dialog)
            dialog, url_dialog = [], []
            text = ""

    # capture the last row
    if prompt_length == 10:
        text += f"Q: {dialogs_df['question'][num_rows]}, "
        text += f"A: {dialogs_df['answer'][num_rows]}"
    if ret_img == True: 
        url_dialog.append(text)
        url_dialog.append(image_id)
        dialog.append(text)
        dialog.append(img)
    else:
        url_dialog.append(image_id)
        url_dialog.append(text)
        dialog.append(img)
        dialog.append(text)

    url_dialog_list.append(url_dialog)
    input_dialog_list.append(dialog)
    
    return input_dialog_list, url_dialog_list

# Display the prompt and retrieve images from their ids
def display_prompt(output_list):
    for output in output_list:
        try:
            get_image(output, show_image=True)
        except:
            print(output)

# Display the output of the model, retrieve the images by their url
def display_output(story_list):
    for element in story_list:
        if type(element) == str:
            try:
                image = get_image_from_url(element)
                plt.figure(figsize=(3, 3))
                plt.axis('off')
                plt.imshow(np.array(image))
                plt.show()
            except:
                print(element)

# Save prompts and output into file with name corresponding to experiment
def save_dialogs(prompts, outputs, num_examples, num_qa, ret_img, provide_context):
    if provide_context == True:
        path = f"visual_dialog/{num_examples}_examples_{num_qa}_qa_{'ret_img' if ret_img==True else 'ret_text'}.json"
    else:
        path = f"visual_dialog/{num_examples}_examples_{num_qa}_qa_{'ret_img' if ret_img==True else 'ret_text'}_nocontext.json"
    data = {
        "prompts": prompts,
        "outputs": outputs
    }

    print(path)
    print(data)

    # Save the JSON strings to a file
    with open(path, 'w') as file:
        json.dump(data, file)