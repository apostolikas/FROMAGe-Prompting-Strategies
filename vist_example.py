import numpy as np
from transformers import logging
logging.set_verbosity_error()

from PIL import Image
import matplotlib.pyplot as plt

from fromage import models
from visual_storytelling import story_in_sequence


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
            # plt.show()
            plt.savefig('output1.png', facecolor='w')
        elif type(output) == Image.Image:
            plt.figure(figsize=(3, 3))
            plt.imshow(np.array(output))
            # plt.show()
            plt.savefig('output2.png', facecolor='w')


# Load model used in the paper.
model_dir = './fromage_model/'
model = models.load_fromage(model_dir)


stories_csv_path = 'visual_storytelling/stories.csv'
stories_df = story_in_sequence.load_stories(stories_csv_path)


story_prompt = story_in_sequence.create_story_list(
    stories_df, 
    story_id='45530', 
    num_captions=5,
    include_images=True,
    as_prompt=True
)

print('story_prompt')
print(story_prompt)


model_outputs = model.generate_for_images_and_texts(story_prompt, max_img_per_ret=3)


display_interleaved_outputs(model_outputs, one_img_per_ret=False)
