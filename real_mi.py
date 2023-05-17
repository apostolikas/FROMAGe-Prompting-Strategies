from my_utils import load_pickle, create_pickle, set_seed, remove_caption_prefix
from fromage import models
import torch.utils.data as data
from transformers import GPT2Tokenizer
import math, random
import numpy as np
from tqdm import tqdm
import time
from PIL import Image, UnidentifiedImageError
import torch

set_seed(0)

# list_input_timestapes = load_pickle('list_input_timestape.pickle')
dict_question_captions = load_pickle('dict_question_captions.pickle')
dict_model_input = load_pickle('dict_model_input_int.pickle')

tokenizer = GPT2Tokenizer.from_pretrained('facebook/opt-6.7b')
tokenizer.pad_token = tokenizer.eos_token
# Add special tokens to the model to enable [RET].
tokenizer.add_special_tokens({"cls_token": "<|image|>"})
tokenizer.add_tokens('[RET]')

# black_img = Image.fromarray(np.zeros((224,224,3),dtype=np.uint8))
constrained_ids = [[]]# add a dummy list at the beginning because question_ids start from 1
new_line_id = tokenizer('\n', add_special_tokens=False).input_ids[0]
space_id = tokenizer.encode_plus(' ').input_ids[1]
for i in dict_model_input:
    current_constrained_ids = []
    txt1, txt2 = remove_caption_prefix(dict_model_input[i][1]), remove_caption_prefix(dict_model_input[i][3])
    txt1_new, txt2_new = ' '+txt1, ' '+txt2

    ids_1 = tokenizer.encode_plus(txt1)['input_ids'][1:]
    current_constrained_ids.extend(ids_1)
    ids_2 = tokenizer.encode_plus(txt2)['input_ids'][1:]
    current_constrained_ids.extend(ids_2)

    ids_1_new = tokenizer.encode_plus(txt1_new)['input_ids'][1:]
    current_constrained_ids.extend(ids_1_new)
    ids_2_new = tokenizer.encode_plus(txt2_new)['input_ids'][1:]
    current_constrained_ids.extend(ids_2_new)

    current_constrained_ids.append(tokenizer.eos_token_id)
    current_constrained_ids.append(new_line_id)
    current_constrained_ids.append(space_id)
    current_constrained_ids = torch.tensor(current_constrained_ids)
    # generated_ids = load_pickle(f'generated_ids_{i}.pt')

    constrained_ids.append(current_constrained_ids)

white_image = Image.fromarray(255*np.ones((224,224,3),dtype=np.uint8))

# Load model used in the paper.
model_dir = './fromage_model/'
model = models.load_fromage(model_dir)
model.calculate_black_image()
model.calculate_white_image()

pairs=[]
for i in tqdm(dict_model_input):
  prompts = dict_model_input[i]
  ans = dict_question_captions[i]
  model_outputs = model.generate_with_content_free(prompts, num_words=8,max_num_rets=0, id=i,
                                                   constrained_ids=constrained_ids[i]) 
  # model_outputs = model.generate_for_images_and_texts(prompts, num_words=12,max_num_rets=0, id=i)
  pairs.append((model_outputs, ans))
#   if i%150==0:
#     create_pickle('all_constr_content_free.pickle',pairs)

# create_pickle('all_constr_content_free.pickle',pairs)
# create_pickle('all_mini_net.pickle',pairs)