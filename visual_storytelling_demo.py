import os

from fromage import models
from src.visual_storytelling.scripts import story_in_sequence


###### CHOOSE EXPERIMENT VARIABLES ######


# Pick one of three experiments

num_captions, include_images = 1, False
# num_captions, include_images = 5, True
# num_captions, include_images = 5, False


# Pick one of three example stories

story_id = '45569'
# story_id = '45643'
# story_id = '45672'


###### CHECK CHOSEN VARIABLES ######


vist_images_folder = './src/visual_storytelling/images'
experiment_subfolder = f'{num_captions}_captions_{"with" if include_images else "no"}_images'

# Create experiment subfolder if not exist
if not os.path.exists(f'{vist_images_folder}/{experiment_subfolder}'):
    print(f'\n {experiment_subfolder} does not exist. \n')
    exit()

# Check if story already exists in experiment subfolder
story_output_path = f'{vist_images_folder}/{experiment_subfolder}/{story_id}.png'
if not os.path.isfile(story_output_path):
    print(f'\n {story_output_path} does not exist. \n')
    exit()


###### RUN INFERENCE ON STORY ######


# Load stories from csv
stories_csv_path = './src/visual_storytelling/data/stories.csv'
stories_df = story_in_sequence.load_stories(stories_csv_path)

# Load model used in the paper.
model_dir = './fromage_model/'
model = models.load_fromage(model_dir)


story_prompt = story_in_sequence.create_story_list(
    stories_df, story_id, num_captions, include_images, as_prompt=True)

print('\n')
for elem in story_prompt:
    print(elem)
print('\n')

model_outputs = model.generate_for_images_and_texts(story_prompt, max_img_per_ret=3)

story_output_path = f'demo_outputs/visual_storytelling_output.png'
story_in_sequence.save_story_predictions(story_output_path, model_outputs, one_img_per_ret=False)
