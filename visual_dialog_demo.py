from src.image_retrieval_vdialog.scripts import models # changed original code
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import openai

parser = argparse.ArgumentParser()
parser.add_argument("--num_tests", type=int, default=3)
parser.add_argument("--num_qa_per_dialog", type=int, default=5)
parser.add_argument("--openai_key", type=str, required=True)

args = parser.parse_args()

num_tests = args.num_tests
num_qa_per_dialog = args.num_qa_per_dialog
openai_key = args.openai_key

openai.api_key = openai_key

print(f"number of experiments = {num_tests}, number of qa's per dialog = {num_qa_per_dialog}")

def load_dialogs(stories_csv_path: str) -> pd.DataFrame:
    return pd.read_csv(stories_csv_path, encoding='utf8', dtype=str)

instruction = """Transform the following caption 
with a question and answer dialogue about an image 
into a caption as short as possible while capturing 
all the information that is given: """

def gpt_prompt(prompt):
    input_prompt = instruction + prompt
    # Generate text with a maximum length of 100 tokens
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt= input_prompt,
        temperature=0,
        max_tokens=100,
        n=1,
        stop=None,
    )

    adapted_prompt = response.choices[0].text.strip()
    return adapted_prompt

# Create prompts from dataframe   #SHORTEN THIS FUNCTION, REMOVE ALL UNNECESSERARY STEPS
def get_prompt_list(dialogs_df, num_rows, num_qa_per_dialog, adapt_gpt_prompt, include_Q_A):
    text = ""
    input_dialog_list = []
    for i in range(num_rows-1):
        if int(dialogs_df['round'][i]) == 1:
            caption = dialogs_df['caption'][i]
            text += f"Caption: {caption}. "
        if int(dialogs_df['round'][i]) <= num_qa_per_dialog:
            if include_Q_A == True:
                text += f"Q: {dialogs_df['question'][i]}, "
                text += f"A: {dialogs_df['answer'][i]}, "
        if dialogs_df['id'][i+1] != dialogs_df['id'][i]:
            text = text[:-2]
            # Compress the dialog + caption into one compact caption
            if adapt_gpt_prompt == True:
                text = gpt_prompt(text)
            text += ' [RET]'

            # Append the dialog when a new dialog will start next
            input_dialog_list.append([text])
            text = ""

    # capture the last row if required
    if num_qa_per_dialog == 10:
        if include_Q_A == True:
            text += f"Q: {dialogs_df['question'][num_rows]}, "
            text += f"A: {dialogs_df['answer'][num_rows]}"
    text += ' [RET]'
    input_dialog_list.append([text])
    
    return input_dialog_list

# Display the output of the model, retrieve the images by their url
def display_output(story_list):
    for element in story_list:
        if type(element) == str:
            # If a dialog is observed, display it line by line
            split_Q = element.split('Q')
            for i, line in enumerate(split_Q):
                if len(split_Q) > 1:
                    if i > 0:
                        print(f'Q{line}')
                    else:
                        print(line)
                else:
                    print(line)
        elif type(element) == Image.Image:
            print("AN IMAGE HAS BEEN FOUND AND DISPLAYED")
            plt.figure(figsize=(3, 3))
            plt.axis('off')
            plt.imshow(np.array(element))
            plt.show()
    print()

# Load model used in the paper.
model_dir = './fromage_model/'
model = models.load_fromage(model_dir)

# Load dialog csv
dialogs_csv_path = 'src/image_retrieval_vdialog/data/dialogs.csv'
dialogs_df = load_dialogs(dialogs_csv_path)

# Obtain a certain amount of dialogs (twice as much as needed to have enough data to compensate the unused data because of errors)
max_q_a_per_caption = 10
num_rows = num_tests * max_q_a_per_caption * 2

dialog_list_cap = get_prompt_list(dialogs_df, num_rows, num_qa_per_dialog, False, False)
dialog_list_cap_dialog = get_prompt_list(dialogs_df, num_rows, num_qa_per_dialog, False, True)
dialog_list_gpt = get_prompt_list(dialogs_df, num_rows, num_qa_per_dialog, True, True)

print("DIALOG LIST")
print(dialog_list_cap)

counter = 0
for i in range(len(dialog_list_cap)):
    print("Experiment num is", i)
    # Obtain the prompts for every setting
    prompt_cap = dialog_list_cap[i]
    prompt_cap_dialog = dialog_list_cap_dialog[i]
    prompt_gpt = dialog_list_gpt[i]

    # Make sure that invalid output will be skipped
    try:
        image_outputs_cap, _ = model.generate_for_images_and_texts(prompt_cap, max_img_per_ret=3, max_num_rets=1, num_words=0)
        image_outputs_cap_dialog, _ = model.generate_for_images_and_texts(prompt_cap_dialog, max_img_per_ret=3, max_num_rets=1, num_words=0)
        image_outputs_gpt, _ = model.generate_for_images_and_texts(prompt_gpt, max_img_per_ret=3, max_num_rets=1, num_words=0)

        # Add the prompts with the image id to the prompt list and the output containing urls to the model outputs
        if image_outputs_cap is not None and image_outputs_cap_dialog is not None and image_outputs_gpt is not None:
            print('\n\n')
            print("THIS IS THE INPUT + OUTPUT LIST AND INPUT TO DISPLAY FUNCTION")
            print(prompt_cap)
            print(prompt_cap_dialog)
            print(prompt_gpt)
            print('\n\n')
            print("Model output for only the caption")
            display_output(prompt_cap)
            display_output(image_outputs_cap)
            print("Model output for the caption combined with the dialog")
            display_output(prompt_cap_dialog)
            display_output(image_outputs_cap_dialog)
            print("Model output for the compressed caption and dialog by GPT-3")
            display_output(prompt_gpt)
            display_output(image_outputs_gpt)
        
        # SAVE IMAGES IN FOLDER CORRESPONDING TO TEST NUMBER AND 

        # Skip if the model did not return images
        else:
            print(f'Test {i} failed because model returned None for one of the settings.')
            continue
    except:
        print(f'Test {i} failed because of invalid model output.')
        continue
    
    # Stop running when the amount of tests is reached
    counter += 1
    if counter == num_tests:
        break