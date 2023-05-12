from visual_dialog import dialog_utils
from fromage import models
import numpy as np
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_tests", type=int, required=True)
parser.add_argument("--num_prompts_per_test", type=int, required=True)
parser.add_argument("--ret_img", type=bool, required=True)

args = parser.parse_args()

num_tests = args.num_tests
num_pt = args.num_prompts_per_test
ret_img = args.ret_img

# Load model used in the paper.
model_dir = './fromage_model/'
model = models.load_fromage(model_dir)

# Load dialog csv
dialogs_csv_path = 'visual_dialog/dialogs.csv'
dialogs_df = dialog_utils.load_dialogs(dialogs_csv_path)

num_qa_per_dialog = 10
compensate_errors_amount = 5

# Obtain a certain amount of the dialogs
num_rows = num_tests * num_pt * num_qa_per_dialog * compensate_errors_amount
# num_rows = len(dialogs_df)
dialog_list = dialog_utils.get_prompt_list(dialogs_df, num_rows, ret_img=ret_img)

prompt = []
prompt_list, model_outputs = [], []
model_outputs = []
counter = 0

for i in range(len(dialog_list)):
    if i == 0: 
        if ret_img == True:
            prompt += ["Retrieve images based on 10 Question and Answer pairs denoted as Q and A"]
        else:
            prompt += ["Generate 10 question and answer pairs denoted with Q and A for each images"]

    if i < num_pt * num_tests:
        if i % num_pt == num_pt-1:
            prompt += [dialog_list[i][0]]
            if ret_img == True:
                num_words = 0
                prompt += ['[RET]']
            else:
                num_words = 400

            print("Test num is", (i+1)/3)

            # Try to obtain output of fromage model
            try:

                output = model.generate_for_images_and_texts(prompt, max_img_per_ret=3, num_words=num_words)
                if output is not None:
                    
                    # PRINT PROMPT + MODEL OUTPUT
                    print("PROMPT")
                    print(prompt)
                    print()
                    print()

                    print("MODEL OUTPUT")
                    print(output)
                    print()
                    print()
                    
                    prompt_list.append(prompt)
                    model_outputs.append(output)
                    prompt = []
                else:
                    print(f'Test {(i+1)/num_pt} failed because model returned None.')
                    prompt = []
                    counter -= 1
                    continue
            except:
                print(f'Test {(i+1)/num_pt} failed because of invalid model output.')
                prompt = []
                continue

            counter += 1
            if counter == num_tests:
                break
            
        else:
            prompt += dialog_list[i]

print("TEST LENGTHS")
print(len(prompt_list))
print(len(model_outputs))

dialog_utils.save_dialogs(prompt_list, model_outputs, num_pt, ret_img=ret_img)



# with open('visual_dialog/model_output7.txt', 'w') as f:
#     for item in model_outputs:
#         f.write(f"{item}\n")

# for output in model_outputs:
#     if type(output) == list:
#         fig, ax = plt.subplots(1, len(output), figsize=(3 * len(output), 3))
#         for i, image in enumerate(output):
#             image = np.array(image)
#             ax[i].imshow(image)
#             ax[i].set_title(f'Retrieval #{i+1}')