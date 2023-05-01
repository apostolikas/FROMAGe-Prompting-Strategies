import json

import pandas as pd
from PIL import Image
from tqdm import tqdm

from fromage.utils import get_image_from_url


def create_stories(sis_json_path: str) -> pd.DataFrame:

    # LOAD STORY-IN-SEQUENCE JSON

    with open(sis_json_path, 'r', encoding='utf8') as f:
        sis = json.load(f)
    
    # CREATE IMAGES DATAFRAME
    
    images_df = pd.DataFrame(columns=['image_id', 'image_url'])

    for image in tqdm(sis['images'], desc='images'):
        image_url = image['url_o'] if 'url_o' in image.keys() else image['url_m']
        images_df.loc[len(images_df)] = [image['id'], image_url]

    # CREATE STORIES DATAFRAME
    
    stories_df = pd.DataFrame(columns=['storylet_id', 'story_id', 'image_id', 'image_url', 'original_text'])

    for annotation in tqdm(sis['annotations'], desc='annotations'):
            image_url = images_df.loc[ images_df['image_id'] == annotation[0]['photo_flickr_id'] ]['image_url'].item()
            stories_df.loc[len(stories_df)] = [
                    annotation[0]['storylet_id'], annotation[0]['story_id'], 
                    annotation[0]['photo_flickr_id'], image_url, annotation[0]['original_text'] ]
    
    return stories_df


def save_stories(stories_df: pd.DataFrame, stories_csv_path: str) -> None:
    stories_df.to_csv(stories_csv_path, encoding='utf8', index=False)


def load_stories(stories_csv_path: str) -> pd.DataFrame:
    return pd.read_csv(stories_csv_path, encoding='utf8', dtype=str)


def create_story_list(stories_df: pd.DataFrame, story_id: str, num_captions: int, include_images: bool, as_prompt: bool) -> list[str|Image.Image]:
    
    if (num_captions < 1) or (num_captions > 5):
        raise ValueError(f'num_captions {num_captions} out of range.')
    
    # reverse order to always include last caption of story
    reverse_story_df = stories_df.loc[ stories_df['story_id'] == story_id ][::-1]

    if len(reverse_story_df) == 0:
        raise ValueError(f'story_id {story_id} does not exist.')
    
    story_list = []

    for index in range(num_captions):
            
        # TEXT

        original_text = reverse_story_df.iloc[index]['original_text']

        if (index == 0) and as_prompt:
            original_text += ' [RET]' # prepare last text for prompt
        
        story_list.insert(0, original_text)

        # IMAGE

        if include_images:
            
            if (index == 0) and as_prompt:
                pass # ignore last image for prompt
            
            else:
                image = get_image_from_url(reverse_story_df.iloc[index]['image_url'])
                story_list.insert(1, image)
        
    return story_list
