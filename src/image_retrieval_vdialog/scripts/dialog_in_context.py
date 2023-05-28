from visual_dialog import dialog_utils
from src.image_retrieval_vdialog.scripts import models # changed original code
import numpy as np
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_tests", type=int, required=True)
parser.add_argument("--num_prompts_per_test", type=int, required=True)
parser.add_argument('--ret_img', type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument('--augment_prompt', type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument('--include_Q_A', type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--num_qa_per_dialog", type=int, required=True)

args = parser.parse_args()

num_tests = args.num_tests
num_pt = args.num_prompts_per_test
ret_img = args.ret_img
num_qa_per_dialog = args.num_qa_per_dialog
adapt_prompt_gpt = args.augment_prompt
include_Q_A = args.include_Q_A

print(f"number of experiments = {num_tests}, number of dailogs per test = {num_pt}, number of qa's per dialog = {num_qa_per_dialog}, return image is {ret_img}")

# Load model used in the paper.
model_dir = './fromage_model/'
model = models.load_fromage(model_dir)

# Load dialog csv
dialogs_csv_path = 'visual_dialog/dialogs.csv'
dialogs_df = dialog_utils.load_dialogs(dialogs_csv_path)

# Obtain a certain amount of dialogs (twice as much as needed to have enough data to compensate the unused data because of errors)
q_a_per_caption = 10
num_rows = num_tests * q_a_per_caption * 2

if num_rows > len(dialogs_df):
    num_rows = len(dialogs_df)

dialog_list, url_dialog_list = dialog_utils.get_prompt_list(dialogs_df, num_rows, num_qa_per_dialog, ret_img, adapt_prompt_gpt, include_Q_A)

prompt, prompt_to_save = [], []
prompt_list, model_outputs = [], []
counter = 0

# Adjustable parameters
provide_context = False
init_prompt = False

for i in range(len(dialog_list)):
    if i % num_pt == 0:
        # Give an initial instruction to the prompt
        if init_prompt:
            if ret_img == True:
                prompt += [f"Retrieve images based on {num_qa_per_dialog} Question and Answer pairs denoted as Q and A"]
                prompt_to_save += [f"Retrieve images based on {num_qa_per_dialog} Question and Answer pairs denoted as Q and A"]
            else:
                prompt += [f"Generate {num_qa_per_dialog} question and answer pairs denoted with Q and A for each image"]
                prompt_to_save += [f"Retrieve images based on {num_qa_per_dialog} Question and Answer pairs denoted as Q and A"]
    # If the number of required examples are met, use the prompt to obtain output from the model
    if i % num_pt == num_pt-1:
        # Add the last text or image to the model, this is the actual prompt which followed the examples
        prompt += [dialog_list[i][0]]
        prompt_to_save += [url_dialog_list[i][0]]

        # Adjust the parameters based on the intention of retrieving image or text
        if ret_img == True:
            num_words = 0
            prompt += ['[RET]']
            max_num_rets=1
        else:
            num_words = 300
            max_num_rets=0

        # Print the amount of performed tests to be able to keep track of the progress
        print("Test num is", (i+1)/num_pt)

        # Make sure that invalid output will be skipped
        try:
            image_outputs, output_urls_or_caption = model.generate_for_images_and_texts(list(prompt), max_img_per_ret=3, max_num_rets=max_num_rets, num_words=num_words)
            # Add the prompts with the image id to the prompt list and the output containing urls to the model outputs
            if image_outputs is not None:
                prompt_list.append(prompt_to_save)
                model_outputs.append(output_urls_or_caption)
                prompt, prompt_to_save = [], []
            # Skip if the model did not return images
            else:
                print(f'Test {(i+1)/num_pt} failed because model returned None.')
                prompt, prompt_to_save = [], []
                continue
        except:
            print(f'Test {(i+1)/num_pt} failed because of invalid model output.')
            prompt, prompt_to_save = [], []
            continue
        
        # Stop running when the amount of tests is reached
        counter += 1
        if counter == num_tests:
            break

    # Provide examples to the model to show it how to handle certain input
    else:
        if provide_context == True:
            # Add the dialogs with url to the prompts that will be stored in a json file
            # Add the dialogs with PIL.Image objects to the prompts such that the model can handle them
            prompt_to_save += url_dialog_list[i]
            prompt += dialog_list[i]

dialog_utils.save_dialogs(prompt_list, model_outputs, num_pt, num_qa_per_dialog, ret_img, provide_context, include_Q_A, adapt_prompt_gpt)