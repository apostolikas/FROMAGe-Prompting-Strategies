from src.image_classification.classification_utils import load_pickle, create_pickle, set_seed, remove_caption_prefix, change_order
from fromage import models
import torch.utils.data as data
import math, random
import numpy as np
from tqdm import tqdm
import time
from PIL import Image, UnidentifiedImageError
import torch
import argparse
from fromage.data import load_real_mi
from src.image_classification.similarities import get_ordered_dict
from src.image_classification.classification_utils import get_constrained_ids

def main(args):
  baseline = args.baseline
  set_seed(0)
  print(f'number of ways {args.num_ways}')
  if args.load_pickle:
    dict_question_captions = load_pickle(f'dict_question_captions_ways_{args.num_ways}.pickle')
    dict_model_input = load_pickle(f'dict_model_input_int_ways_{args.num_ways}.pickle')
  else:
    dict_model_input, dict_question_captions = load_real_mi(args.num_ways)

  if args.order:
    print(f'{dict_model_input[1]}')
    dict_model_input = get_ordered_dict(dict_model_input,args.num_ways)  
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
  # useful for "calibrate before use extension"extension
  if not baseline:
    model.calculate_black_image()
    model.calculate_white_image()

  baseline_str = 'base' if baseline else 'content_free'
  print(f'{constraint_str} {baseline_str} ')
  pairs=[]
  num_words = args.num_words
  print(f'num_words {num_words}')
  order_str = '_order_sim' if args.order else ''
  for i in tqdm(dict_model_input):
    prompts = dict_model_input[i]
    ans = dict_question_captions[i]
    tmp_constr_ids = constrained_ids[i] if args.constraint else None
    if baseline and not args.constraint: #baseline and full vocabulary
      model_outputs = model.generate_for_images_and_texts(prompts, num_words=num_words,max_num_rets=0, id=i)
    else:# contraint OR (content_free and full vocab)
      model_outputs = model.other_generate(prompts, num_words=num_words,max_num_rets=0, id=i,
                                                    constrained_ids=tmp_constr_ids, baseline=baseline) 
    pairs.append((model_outputs, ans))
    

  tp=0
  print(f'{constraint_str}_{baseline_str}_ways{args.num_ways}{order_str}')
  for i,(pred,true_label) in enumerate(pairs):
    if true_label in pred[0]:
        tp+=1
  print(f'ACCURACY = {tp/len(pairs)}')

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

