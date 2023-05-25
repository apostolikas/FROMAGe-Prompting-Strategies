import os

import pandas as pd

from fromage import models
from src.video_captioning.scripts import help_functions as hf


num_gifs = 2
frames_per_gif = 5
temp_path = 'temp.gif'
model_dir = 'fromage_model/'
results_path = f'src/video_captioning/experiments/uniform_{frames_per_gif}/results.csv'
tgif_path = 'src/video_captioning/data/tgif_dataset.csv'


# Load model used in the paper.
model = models.load_fromage(model_dir)


# load gif dataframe
gif_df = pd.read_csv(tgif_path, sep=';', encoding='utf8')


# load any previous results
if os.path.isfile(results_path):
    results = pd.read_csv(results_path, sep=';', encoding='utf8')
    # start right after last seen gif based on url
    start_index = gif_df.loc[ results.iloc[-1]['gif_url'] == gif_df['gif_url'] ].index[0] + 1
else:
    results = pd.DataFrame(columns=['gif_url', 'gif_caption', 'model_caption'])
    # start from the beginning
    start_index = 0

print(f'start_index {start_index}')

for idx in range(start_index, start_index + num_gifs):

    if idx == gif_df.shape[0]:
        break

    gif_url = gif_df.iloc[idx]['gif_url']
    gif_caption = gif_df.iloc[idx]['gif_caption']
    
    try:
        hf.download_gif(gif_url, temp_path)
    except:
        print(f'Failed \t {gif_url}')
        continue
    
    frame_list = hf.extract_frames(temp_path, frames_per_gif)

    model_input = frame_list + [ 'Give caption as video.' ]
    model_outputs = model.generate_for_images_and_texts(model_input, num_words=15, max_num_rets=0)
    model_caption = model_outputs[0]

    results.loc[len(results)] = [gif_url, gif_caption, model_caption]

results.to_csv(results_path, sep=';', header=True, index=False, encoding='utf8')

os.rmdir(temp_path)

print('Done.')
