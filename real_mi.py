from my_utils import load_pickle, create_pickle, set_seed, Mi_Dataset, examples2Timesteps, collate_fn,remove_caption_prefix
from fromage import models
import torch.utils.data as data
from transformers import GPT2Tokenizer
import math, random
from tqdm import tqdm
import time
from PIL import Image, UnidentifiedImageError

tic = time.perf_counter()
pairs = load_pickle('rand25mi.pickle')
print(pairs)
set_seed(0)

# dict_model_input = load_pickle('dict_model_input_int.pickle')
keys = load_pickle('keys_rand_order.pickle')
# list_input_timestapes = load_pickle('list_input_timestape.pickle')
dict_question_captions = load_pickle('dict_question_captions.pickle')
dict_model_input = load_pickle('dict_model_input_int.pickle')

first = 0
second = 0
captions=set()
for (id, input),(id, caption) in zip(dict_model_input.items(), dict_question_captions.items()):
  ans1 = remove_caption_prefix(input[1])
  ans2 = remove_caption_prefix(input[3])
  captions.add(ans1)
  captions.add(ans2)
  if ans1 == caption:
     first = first+1
  else:
     second = second+1
print(captions)

masks = {}
k = 100

# Load model used in the paper.
model_dir = './fromage_model/'
model = models.load_fromage(model_dir)

k = 25
print('25 random examples')
# pairs = []
tp = 0

with open("rand25results.txt", "w") as f:
  captions25=set()
  for i, key in enumerate(tqdm(keys[:k])):
      prompts = dict_model_input[key]
      ans = dict_question_captions[key]
      print('------INPUT------')
      ex1, ex2 = prompts[1], prompts[3]
      print(ex1, ' ',ex2)
      print('------PREDICTION------')
      print(pairs[i][0])
      print('------ANSWER------')
      assert(pairs[i][1] == ans)
      print(ans)
      caption1, caption2 = remove_caption_prefix(ex1), remove_caption_prefix(ex2)
      f.write('input: '+caption1+ ' '+caption2+' \n prediction: '+pairs[i][0][0]+
               ' answer: '+ ans+'\n')
      captions25.add(caption1)
      captions25.add(caption2)
      captions25.add(ans)
      for generated in pairs[i][0]:
        if ans in generated:
            tp = tp+1
            break
      #! the model just repeats after some time so maybe I have to put a smaller number like num_words=10
      #! or no because I have to use also another experiment with bigger text input
      # model_outputs = model.generate_for_images_and_texts(prompts, num_words=20)
      # pairs.append((model_outputs, ans))
  # create_pickle('rand25mi.pickle',pairs)
toc = time.perf_counter()

print(f"Inference in 25 examples in {toc - tic:0.4f} seconds")
print(captions25)