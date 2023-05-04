import os
import argparse

import numpy as np

from fromage import models
from visual_storytelling import story_in_sequence


parser = argparse.ArgumentParser()
parser.add_argument("--num_stories", type=int, required=True)
parser.add_argument("--num_captions", type=int, required=True)
parser.add_argument("--include_images", action="store_true")
args = parser.parse_args()

num_stories = args.num_stories
num_captions = args.num_captions
include_images = args.include_images


# Load model used in the paper.
model_dir = './fromage_model/'
model = models.load_fromage(model_dir)


# Load stories from csv
stories_csv_path = './visual_storytelling/stories.csv'
stories_df = story_in_sequence.load_stories(stories_csv_path)


# Shuffle story ids
np.random.seed(0)
story_ids = stories_df['story_id'].unique()
np.random.shuffle(story_ids)


vist_images_folder = './visual_storytelling/images'

experiment_subfolder = f'{num_captions}_captions_{"with" if include_images else "no"}_images'

# Create experiment subfolder if not exist
if not os.path.exists(f'{vist_images_folder}/{experiment_subfolder}'):
    os.makedirs(f'{vist_images_folder}/{experiment_subfolder}')


counter = 0

for story_id in story_ids:

    # Check if story already exists in experiment subfolder
    
    story_output_path = f'{vist_images_folder}/{experiment_subfolder}/{story_id}.png'

    if os.path.isfile(story_output_path):
        print(f'{story_output_path} already exists.')
        continue

    # Try to create story prompt and save model output

    story_prompt = story_in_sequence.create_story_list(
        stories_df, story_id, num_captions, include_images, as_prompt=True)
    
    if story_prompt is not None:
        
        model_outputs = model.generate_for_images_and_texts(story_prompt, max_img_per_ret=3)
        print(model_outputs)
        
        try:
            story_in_sequence.save_story_predictions(story_output_path, model_outputs, one_img_per_ret=False)
        except:
            print(f'story_id {story_id} : invalid model output.')
            continue
        
        counter += 1
        if counter == num_stories:
            break
