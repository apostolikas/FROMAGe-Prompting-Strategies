import os
import argparse

import numpy as np

from fromage import models
from frames import frames

import matplotlib.pyplot as plt


# parser = argparse.ArgumentParser()
# parser.add_argument("--num_stories", type=int, required=True)
# parser.add_argument("--num_captions", type=int, required=True)
# parser.add_argument("--include_images", action="store_true")
# args = parser.parse_args()

# num_stories = args.num_stories
# num_captions = args.num_captions
# include_images = args.include_images


# Load model used in the paper.
model_dir = './fromage_model/'
model = models.load_fromage(model_dir)

num_frames = 5

# gif_path = 'frames/bowling.gif'
# frames_path = 'frames/bowling.png'
# gif_url = 'https://33.media.tumblr.com/0346abaaf2bddc33ee1db0f84f04fb76/tumblr_nqjuk8ssUF1sfdocho1_500.gif'
# gif_caption = 'a man rolls a bowling ball along the lane.'

# gif_path = 'frames/explosion.gif'
# frames_path = 'frames/explosion.png'
# gif_url = 'https://38.media.tumblr.com/8f6110a0ec6a7409579b878ba09ae62f/tumblr_nq07jm1AQ51sja1kdo1_500.gif'
# gif_caption = 'a bald man drop one bomb and explode the local.'

# gif_path = 'frames/skating.gif'
# frames_path = 'frames/skating.png'
# gif_url = 'https://38.media.tumblr.com/88091baac04b18ef73d93e56883dccfe/tumblr_npzfsr13I51uyqfd5o1_500.gif'
# gif_caption = 'a skate boarder is doing trick on his skate board.'

# gif_path = 'frames/cats.gif'
# frames_path = 'frames/cats.png'
# gif_url = 'https://38.media.tumblr.com/6b62aa0e56c14d3604c6a205bdbc972e/tumblr_npxwgoNBQD1tht7cgo1_500.gif'
# gif_caption = 'a cat pounces on a cat's tail and scares it, which makes all the other cats run away in fear.'

gif_path = 'frames/bass.gif'
frames_path = 'frames/bass.png'
gif_url = 'https://38.media.tumblr.com/7247e34e0545868afaf1b1449c1cf263/tumblr_nfhunfZMiC1rjav34o1_500.gif'
gif_caption = 'the white man is playing a song with the bass.'

# frames.download_gif(gif_url, gif_path)

frame_list = frames.extract_frames(gif_path, num_frames)

# frames.save_frames(frame_list, frames_path)

model_input = frame_list + [ 'Video caption:' ]

print(model_input)

model_outputs = model.generate_for_images_and_texts(model_input, num_words=20, max_num_rets=0)

print(model_outputs)


# # Load stories from csv
# stories_csv_path = './visual_storytelling/stories.csv'
# stories_df = story_in_sequence.load_stories(stories_csv_path)


# # Shuffle story ids
# np.random.seed(0)
# story_ids = stories_df['story_id'].unique()
# np.random.shuffle(story_ids)


# vist_images_folder = './visual_storytelling/images'

# experiment_subfolder = f'{num_captions}_captions_{"with" if include_images else "no"}_images'

# # Create experiment subfolder if not exist
# if not os.path.exists(f'{vist_images_folder}/{experiment_subfolder}'):
#     os.makedirs(f'{vist_images_folder}/{experiment_subfolder}')


# counter = 0

# for story_id in story_ids:

#     # Check if story already exists in experiment subfolder
    
#     story_output_path = f'{vist_images_folder}/{experiment_subfolder}/{story_id}.png'

#     if os.path.isfile(story_output_path):
#         print(f'{story_output_path} already exists.')
#         continue

#     # Try to create story prompt and save model output

#     story_prompt = story_in_sequence.create_story_list(
#         stories_df, story_id, num_captions, include_images, as_prompt=True)
    
#     if story_prompt is not None:
#         try:
#             model_outputs = model.generate_for_images_and_texts(story_prompt, max_img_per_ret=3)
#         except:
#             print(f'story_id {story_id} : model cant retrieve images.')
#             continue

#         try:
#             story_in_sequence.save_story_predictions(story_output_path, model_outputs, one_img_per_ret=False)
#         except:
#             print(f'story_id {story_id} : invalid model output.')
#             continue
        
#         counter += 1
#         if counter == num_stories:
#             break
