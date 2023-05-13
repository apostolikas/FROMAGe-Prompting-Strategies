from my_utils import load_pickle, create_pickle, set_seed, Mi_Dataset, examples2Timesteps, collate_fn,remove_caption_prefix
from fromage import models
import torch.utils.data as data
from transformers import GPT2Tokenizer
import math, random
import numpy as np
from tqdm import tqdm
import time
from PIL import Image, UnidentifiedImageError
import torch

tic = time.perf_counter()

set_seed(0)


# dict_model_input = load_pickle('dict_model_input_int.pickle')
keys = load_pickle('keys_rand_order.pickle')
# list_input_timestapes = load_pickle('list_input_timestape.pickle')
dict_question_captions = load_pickle('dict_question_captions.pickle')
dict_model_input = load_pickle('dict_model_input_int.pickle')

# black_img = Image.fromarray(np.zeros((224,224,3),dtype=np.uint8))

# Load model used in the paper.
model_dir = './fromage_model/'
model = models.load_fromage(model_dir)
model.calculate_black_image()

pairs=[]
for i in tqdm(dict_model_input):
  prompts = dict_model_input[i]
  ans = dict_question_captions[i]
  model_outputs = model.generate_with_content_free(prompts, num_words=12,max_num_rets=0, id=i) 
  # model_outputs = model.generate_for_images_and_texts(prompts, num_words=12,max_num_rets=0)
  print(model_outputs)
  pairs.append((model_outputs, ans))
  # if i%100==0:
  create_pickle('content_free.pickle',pairs)
  
# create_pickle('all_mini_net.pickle',pairs)