import pandas as pd 
from PIL import Image
import os
import numpy as np
from fromage import models
import matplotlib as plt
from torchvision.transforms import ToTensor
# from torchmetrics.multimodal import CLIPScore
import openai
import time
import pickle

#!put your key here
openai.api_key = "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"

gpt_version = "text-davinci-002" # https://openai.com/pricing
def prompt_llm(prompt, max_tokens = 64, temperature=0, stop=None, top_p = 1, frequency_penalty = 0):
  'do not change both temperature and top_p sampling' 
  # https://platform.openai.com/docs/api-reference/completions/create
  response = openai.Completion.create(engine=gpt_version, prompt=prompt, 
                                      max_tokens=max_tokens, temperature=temperature, stop=stop,
                                      top_p = top_p, frequency_penalty = frequency_penalty )
  return response["choices"][0]["text"].strip()


#! Load and preprocess data
df = pd.read_csv('./Flickr8k_text/ExpertAnnotations.txt',delimiter='\t')
cropped_df = df.loc[df['expert1'] == 4]
img_cap_list = list(zip(cropped_df.image_id, cropped_df.caption_id))
cap_df = pd.read_csv('./Flickr8k_text/Flickr8k.token.txt',delimiter='\t')
cap_dict = pd.Series(cap_df.cap.values,index=cap_df.cap_id).to_dict()
data_dict = {}
captions = []
for img_id, cap_id in zip(cropped_df.image_id, cropped_df.caption_id):
    caption = cap_dict[cap_id]
    data_dict[img_id] = caption 
    captions.append(caption)

instructions_dict = {}
# settings
max_generated_tokens  = 60
temperature = 0
top_p = 0.4
frequency_penalty = 1
captions = captions[:1] # how many captions to paraphrase

for i,tmp_caption in enumerate(captions):
    prompt_list = [f'Give me an equivalent sentence with the following "{tmp_caption}"']
    
    for prompt in prompt_list:
       ans = prompt_llm(prompt, max_tokens = max_generated_tokens, top_p = top_p,
                        temperature=temperature, frequency_penalty = frequency_penalty)
       instructions_dict[prompt] = ans
    #! openai doesn't like very frequent requests so maybe we need to use some sleep function after some time
    time.sleep(5) #seconds

filename = f'temp_{temperature}_topp_{top_p}_freq_{frequency_penalty}.pickle'
with open(filename, 'wb') as f:
    pickle.dump(instructions_dict, f)

for key in instructions_dict:
    print(key)
    print(instructions_dict[key])
    print('-----------------------------------')
