import pickle
import torch
import numpy as np
import random

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

def create_pickle(filename, object):
    with open(filename, 'wb') as f:
        pickle.dump(object, f)

def load_pickle(filename):
    with open(filename, "rb") as input_file:
        object = pickle.load(input_file)
    return object

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