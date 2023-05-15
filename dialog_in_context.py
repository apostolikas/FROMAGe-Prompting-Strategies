from visual_dialog import dialog_utils
from fromage import models
import numpy as np
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_tests", type=int, required=True)
parser.add_argument("--num_prompts_per_test", type=int, required=True)
parser.add_argument("--ret_img", type=bool, required=True)
parser.add_argument("--num_qa_per_dialog", type=int, required=True)

args = parser.parse_args()

num_tests = args.num_tests
num_pt = args.num_prompts_per_test
ret_img = args.ret_img
num_qa_per_dialog = args.num_qa_per_dialog

# Load model used in the paper.
model_dir = './fromage_model/'
model = models.load_fromage(model_dir)

# Load dialog csv
dialogs_csv_path = 'visual_dialog/dialogs.csv'
dialogs_df = dialog_utils.load_dialogs(dialogs_csv_path)

compensate_error_amount = 3

# Obtain a certain amount of dialogs
num_rows = num_tests * num_pt * compensate_error_amount
# num_rows = len(dialogs_df)
dialog_list, url_dialog_list = dialog_utils.get_prompt_list(dialogs_df, num_rows, num_qa_per_dialog, ret_img=ret_img)

prompt, prompt_to_save = [], []
prompt_list, model_outputs = [], []
counter = 0

# Give an initial instruction to the prompt
init_prompt = False
for i in range(len(dialog_list)):
    if init_prompt and i % num_pt == 0:
        if ret_img == True:
            prompt += ["Retrieve images based on 10 Question and Answer pairs denoted as Q and A"]
            prompt_to_save += ["Retrieve images based on 10 Question and Answer pairs denoted as Q and A"]
        else:
            prompt += ["Generate 10 question and answer pairs denoted with Q and A for each images"]
            prompt_to_save += ["Retrieve images based on 10 Question and Answer pairs denoted as Q and A"]

    if i < num_pt * num_tests:
        if i % num_pt == num_pt-1:
            prompt += [dialog_list[i][0]]
            prompt_to_save += [url_dialog_list[i][0]]

            if ret_img == True:
                num_words = 0
                prompt += ['[RET]']
            else:
                num_words = 400

            print("Test num is", (i+1)/num_pt)

            # Make sure that invalid output will be skipped
            try:
                image_outputs, output_urls = model.generate_for_images_and_texts(prompt, max_img_per_ret=3, num_words=num_words)
                # Add the prompts with the image id to the prompt list and the output containing urls to the model outputs
                if image_outputs is not None:
                    prompt_list.append(prompt_to_save)
                    model_outputs.append(output_urls)
                    prompt, prompt_to_save = [], []
                # Skip if the model did not return images
                else:
                    print(f'Test {(i+1)/num_pt} failed because model returned None.')
                    prompt, prompt_to_save = [], []
                    counter -= 1
                    continue
            except:
                print(f'Test {(i+1)/num_pt} failed because of invalid model output.')
                prompt, prompt_to_save = [], []
                continue
            
            # Stop running when the amount of tests is reached
            counter += 1
            if counter == num_tests:
                break
        # Add the dialogs with url to the prompts that will be stored in a json file
        # Add the dialogs with PIL.Image objects to the prompts such that the model can handle them
        else:
            prompt_to_save += url_dialog_list[i]
            prompt += dialog_list[i]

dialog_utils.save_dialogs(prompt_list, model_outputs, num_pt, num_qa_per_dialog, ret_img=ret_img)