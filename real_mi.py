from my_utils import load_pickle, create_pickle, set_seed, remove_caption_prefix, change_order
from fromage import models
import torch.utils.data as data
from transformers import GPT2Tokenizer
import math, random
import numpy as np
from tqdm import tqdm
import time
from PIL import Image, UnidentifiedImageError
import torch
import argparse
from fromage.data import load_real_mi


def get_constrained_ids(dict_model_input, num_examples=2):
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
      num = -1
      for jj in range(num_examples):
        num += 2 # text every 2 items in the list
        txt = remove_caption_prefix(dict_model_input[i][num])
        new_txt = ' '+txt
        ids_1 = tokenizer.encode_plus(txt)['input_ids'][1:]
        current_constrained_ids.extend(ids_1)
        ids_2 = tokenizer.encode_plus(new_txt)['input_ids'][1:]
        current_constrained_ids.extend(ids_2)

      current_constrained_ids.append(tokenizer.eos_token_id)
      current_constrained_ids.append(new_line_id)
      current_constrained_ids.append(space_id)
      current_constrained_ids = torch.tensor(current_constrained_ids)

      constrained_ids.append(current_constrained_ids)

  return constrained_ids

def main(args):
  baseline = args.baseline
  set_seed(0)
  # list_input_timestapes = load_pickle('list_input_timestape.pickle')
  print(f'number of ways {args.num_ways}')
  if args.load_pickle:
    dict_question_captions = load_pickle(f'dict_question_captions_ways_{args.num_ways}.pickle')
    dict_model_input = load_pickle(f'dict_model_input_int_ways_{args.num_ways}.pickle')
  else:
    dict_model_input, dict_question_captions = load_real_mi(args.num_ways)
    create_pickle(f'dict_question_captions_ways_{args.num_ways}.pickle', dict_question_captions)
    create_pickle(f'dict_model_input_int_ways_{args.num_ways}.pickle', dict_model_input)

  if args.order and args.num_ways==2:
    similaties = load_pickle('clip_sim_img2img.pickle')
    print(f'{dict_model_input[1]}')
    dict_model_input = change_order(dict_model_input, similaties)
    print('changed order of examples')
    print(f'{dict_model_input[1]}')
  if args.constraint:
    constrained_ids = get_constrained_ids(dict_model_input, num_examples=args.num_ways)

    constraint_str = 'constr'
  else:
    constrained_ids = None
    constraint_str = 'NO_constr'
  # Load model used in the paper.
  print(f'Baseline {baseline}')
  model_dir = './fromage_model/'
  model = models.load_fromage(model_dir)
  if not baseline:
    model.calculate_black_image()
    model.calculate_white_image()
  baseline_str = 'base' if baseline else 'content_free'
  print(f'{constraint_str} {baseline_str} ')
  pairs=[]
  num_words = args.num_words
  print(f'num_words {num_words}')
  for i in tqdm(dict_model_input):
    prompts = dict_model_input[i]
    ans = dict_question_captions[i]
    tmp_constr_ids = constrained_ids[i] if args.constraint else None
    if baseline and not args.constraint: #baseline and full vocabulary
      model_outputs = model.generate_for_images_and_texts(prompts, num_words=num_words,max_num_rets=0, id=i)
    else:# contraint OR (content_free and full vocab)
      model_outputs = model.generate_constr_content_free(prompts, num_words=num_words,max_num_rets=0, id=i,
                                                    constrained_ids=tmp_constr_ids, baseline=baseline) 
    pairs.append((model_outputs, ans))
    if i%100==0:
      create_pickle(f'{constraint_str}_{baseline_str}_ways{args.num_ways}.pickle',pairs)

  create_pickle(f'{constraint_str}_{baseline_str}_ways{args.num_ways}.pickle',pairs)
  # create_pickle('all_mini_net.pickle',pairs)

if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Simple settings.')
  parser.add_argument('--baseline', action='store_true')
  parser.add_argument('--order', action='store_true')
  parser.add_argument('--constraint', action='store_true')
  parser.add_argument('--load_pickle', action='store_true')
  parser.add_argument('--num_words', type=int, default=20)
  parser.add_argument('--num_ways', type=int, default=2,
                      help='The number of object classes in the task. Either 2 or 5')
  args = parser.parse_args()
  main(args)

