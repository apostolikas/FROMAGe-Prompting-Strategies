import pickle
import torch
import numpy as np
import random
from transformers import GPT2Tokenizer

def change_order(dict_model_input, similaties):
    '''
        Change the order of the in-context examples based on the images similarities
        between the images in the input
    '''
    max_first = 0
    ordered_dict_model_input = {}
    for i,(score1, score2) in enumerate(similaties):
        key = i+1
        img_ex1, text_ex1, img_ex2, text_ex2, img_q, text_q_prompt = dict_model_input[key]
        if score1 > score2:
            max_first+=1
            ordered_dict_model_input[key] = [img_ex2, text_ex2, img_ex1, text_ex1, img_q, text_q_prompt]
        else:
            ordered_dict_model_input[key] = [img_ex1, text_ex1, img_ex2, text_ex2, img_q, text_q_prompt]

    print('Max similarity first',max_first)
    return ordered_dict_model_input

def change_order_5(dict_model_input, similaties):
    '''
        Change the order of the in-context examples based on the images similarities
        between the images in the input
    '''
    ordered_dict_model_input = {}
    for i,(scores) in enumerate(similaties):
        key = i+1
        input = dict_model_input[key]
        indices = np.argsort(scores)
        assert(indices.shape[0]*2+2 == len(input))
        new_input  = []
        for id in indices:
            new_input.extend([input[2*id],input[2*id+1]])
        new_input.extend([input[-2],input[-1]])
        ordered_dict_model_input[key] = new_input

    return ordered_dict_model_input

def create_pickle(filename, object):
    with open(filename, 'wb') as f:
        pickle.dump(object, f)

def load_pickle(filename):
    with open(filename, "rb") as input_file:
        object = pickle.load(input_file)
    return object

def get_constrained_ids(dict_model_input, num_examples=2):
  '''
  We want to generate labels that are generated with each label name
  '''
  tokenizer = GPT2Tokenizer.from_pretrained('facebook/opt-6.7b')
  tokenizer.pad_token = tokenizer.eos_token
  # Add special tokens to the model to enable [RET].
  tokenizer.add_special_tokens({"cls_token": "<|image|>"})
  tokenizer.add_tokens('[RET]')

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

def remove_caption_prefix(caption_text_q_prompt):
    '''
        Convert 'This is a {class_name}' to 'class_name'
    '''

    assert('This is a' in caption_text_q_prompt)
    only_caption_q_no_text = caption_text_q_prompt.replace('This is a ','')
    return only_caption_q_no_text

def add_caption_prefix(caption, prompt_text):
    caption = prompt_text +' '+ caption
    return caption

def set_seed(seed):
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    # Additionally, some operations on a GPU are implemented stochastic for efficiency
    # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)