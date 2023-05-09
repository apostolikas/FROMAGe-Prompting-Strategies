import pickle
import torch
import numpy as np
import random

class Mi_Dataset(torch.utils.data.Dataset):
  def __init__(self, timesteps, dict_question_captions):
        self.timesteps = timesteps
        self.dict_question_captions = dict_question_captions

  def __len__(self):
        return len(self.timesteps[0])

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = [self.timesteps[i][index] for i in range(len(self.timesteps))]
        y = self.dict_question_captions[index]

        return X, y

def collate_fn(data):
    # do max padding but based on the input to the LM not the input to the Vision model
    a = 1

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

    assert('This is a' == caption_text_q_prompt)
    only_caption_q_no_text = caption_text_q_prompt.replace('This is a ','')
    return only_caption_q_no_text

def add_caption_prefix(caption, prompt_text):
    caption = prompt_text +' '+ caption
    return caption

def examples2Timesteps(examples):
    '''
    e.g. Convert input of the form
        [[ ex_1_text1, ex_1_img1, ex_1_text2, ex_1_img2 ] ,
        [ ex_2_text1, ex_2_img1, ex_2_text2, ex_2_img2 ]
        ] 
    to 
        [[ ex_1_text1, ex_2_text1]
        [ex_1_img1, ex_2_img1 ] ,
        [ ex_1_text2, ex_2_text2],
        [ex_1_img2, ex_2_img2 ]
        ] 
    '''
    timesteps = [[el[i] for el in examples] for i in range(len(examples[0]))]
    return timesteps



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