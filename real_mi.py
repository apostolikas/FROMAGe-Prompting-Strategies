from my_utils import load_pickle, create_pickle, set_seed, Mi_Dataset, examples2Timesteps, collate_fn,remove_caption_prefix
from fromage import models
import torch.utils.data as data
from transformers import GPT2Tokenizer
import math, random
from tqdm import tqdm
import time
from PIL import Image, UnidentifiedImageError

tic = time.perf_counter()

set_seed(0)

# dict_model_input = load_pickle('dict_model_input_int.pickle')
keys = load_pickle('keys_rand_order.pickle')
# list_input_timestapes = load_pickle('list_input_timestape.pickle')
dict_question_captions = load_pickle('dict_question_captions.pickle')
dict_model_input = load_pickle('dict_model_input_int.pickle')

# first = 0
# second = 0
# captions=set()
# for (id, input),(id, caption) in zip(dict_model_input.items(), dict_question_captions.items()):
#   ans1 = remove_caption_prefix(input[1])
#   ans2 = remove_caption_prefix(input[3])
#   captions.add(ans1)
#   captions.add(ans2)
#   if ans1 == caption:
#      first = first+1
#   else:
#      second = second+1
# print(captions)

# masks = {}
# k = 100

# Load model used in the paper.
model_dir = './fromage_model/'
model = models.load_fromage(model_dir)
pairs=[]
for i in tqdm(dict_model_input):
  prompts = dict_model_input[i]
  ans = dict_question_captions[i]
  model_outputs = model.generate_for_images_and_texts(prompts, num_words=14)
  pairs.append((model_outputs, ans))
  if i%100==0:
    create_pickle('all_mini_net.pickle',pairs)
  if i==1250:
    print('1250')
    break
create_pickle('first_1250_mini_net.pickle',pairs)