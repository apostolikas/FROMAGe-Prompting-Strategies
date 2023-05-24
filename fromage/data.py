"""Modified from https://github.com/mlfoundations/open_clip"""

from typing import Optional, Tuple

import collections
import logging
import os
import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
from torchvision import transforms as T
from PIL import Image, ImageFont
from torch.utils.data import Dataset
import json
import cv2#!
from fromage import utils


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def load_real_mi():

  with open('./real_name_mi/real_name_mi_shots_1_ways_2_all_questions.json', 'r') as f:
        mi_data = json.load(f)

  mi_images_folder = './real_name_mi'
  #onlyfiles = [f for f in listdir(mi_images_folder) if isfile(join(mi_images_folder, f))]

  test_images_info = []
  mi_images_folder = './real_name_mi'

  #! it seems that SAM automatically resizes the images to (1024, 1024) by using padding
  dict_model_input = {} #question_id: [img1, caption1, img2, caption2, img3]
  dict_question_captions = {} # question_id: caption
  dict_prompt_images = {} # question_id:[img1,img2]
  dict_prompt_captions = {} # question_id:[caption1, caption2]
  for i,image_info in enumerate(mi_data):
      # img_q = Image.open(os.path.join(mi_images_folder,image_info['question_image']))
      # img_prompt1 = Image.open(os.path.join(mi_images_folder,image_info['image_1']))
      # img_prompt2 = Image.open(os.path.join(mi_images_folder,image_info['image_2']))
      question_id = int(image_info['question_id'])
      assert(i+1 == question_id)
      # read the images the same way like the segment_anything paper does
      img_q = cv2.imread(os.path.join(mi_images_folder,image_info['question_image']))
      img_q = cv2.cvtColor(img_q, cv2.COLOR_BGR2RGB)
      img_q = Image.fromarray(img_q)
      img_q = img_q.resize((224,224))

      img_ex1 = cv2.imread(os.path.join(mi_images_folder,image_info['image_1']))
      img_ex1 = cv2.cvtColor(img_ex1, cv2.COLOR_BGR2RGB)
      img_ex1 = Image.fromarray(img_ex1)
      img_ex1 = img_ex1.resize((224,224))

      img_ex2 = cv2.imread(os.path.join(mi_images_folder,image_info['image_2']))
      img_ex2 = cv2.cvtColor(img_ex2, cv2.COLOR_BGR2RGB)
      img_ex2 = Image.fromarray(img_ex2)
      img_ex2 = img_ex2.resize((224,224))
      
      dict_prompt_images[question_id] = [img_ex1, img_ex2]
      
      caption_q_answer = image_info['answer']
      text_q_prompt = image_info['question']
      text_ex1 = image_info['caption_1']
      text_ex2 = image_info['caption_2']

      
      dict_question_captions[question_id] = caption_q_answer
      # only_caption_ex1_no_text = remove_caption_prefix(caption_ex1)
      # only_caption_ex2_no_text = remove_caption_prefix(caption_ex2)
      dict_prompt_captions[question_id] = [text_ex1, text_ex2]

      model_input_list = [img_ex1, text_ex1, img_ex2, text_ex2, img_q, text_q_prompt]
      dict_model_input[question_id] = model_input_list

  return dict_model_input, dict_question_captions

def preprocess_image(mi_images_folder, img):
  img_q = cv2.imread(os.path.join(mi_images_folder,img))
  img_q = cv2.cvtColor(img_q, cv2.COLOR_BGR2RGB)
  img_q = Image.fromarray(img_q)
  img_q = img_q.resize((224,224))
  return img_q

def load_real_mi(k=2):

  with open(f'./real_name_mi/real_name_mi_shots_1_ways_{k}_all_questions.json', 'r') as f:
        mi_data = json.load(f)

  mi_images_folder = './real_name_mi'
  #onlyfiles = [f for f in listdir(mi_images_folder) if isfile(join(mi_images_folder, f))]

  mi_images_folder = './real_name_mi'
  dict_model_input = {} #question_id: [img1, caption1, img2, caption2, img3]
  dict_question_captions = {} # question_id: caption
  dict_prompt_images = {} # question_id:[img1,img2]
  dict_prompt_captions = {} # question_id:[caption1, caption2]
  for i,image_info in enumerate(mi_data):
      question_id = int(image_info['question_id'])
      assert(i+1 == question_id)
      # read the images the same way like the segment_anything paper does
      img_q = preprocess_image(mi_images_folder, image_info['question_image'])
      images_examples=  []
      caption_examples = []
      model_input_list  = []
      for jj in range(k):
        num = jj+1
        image_jj = preprocess_image(mi_images_folder, image_info[f'image_{num}'])
        caption_jj = image_info[f'caption_{num}']
        images_examples.append(image_jj)
        caption_examples.append(caption_jj)

        model_input_list.append(image_jj)
        model_input_list.append(caption_jj)


      dict_prompt_images[question_id] = images_examples
      dict_prompt_captions[question_id] = caption_examples

      caption_q_answer = image_info['answer']
      text_q_prompt = image_info['question']

      dict_question_captions[question_id] = caption_q_answer
      

      model_input_list.extend([img_q, text_q_prompt])
      dict_model_input[question_id] = model_input_list

  return dict_model_input, dict_question_captions

def get_dataset(args, split: str, tokenizer, precision: str = 'fp32') -> Dataset:
  assert split in ['train', 'val'
    ], 'Expected split to be one of "train" or "val", got {split} instead.'

  dataset_paths = []
  image_data_dirs = []
  train = split == 'train'

  # Default configs for datasets.
  # Folder structure should look like:
  if split == 'train':
    if 'cc3m' in args.dataset:
      dataset_paths.append(os.path.join(args.dataset_dir, 'cc3m_train.tsv'))
      image_data_dirs.append(os.path.join(args.image_dir, 'cc3m/training/'))
    else:
      raise NotImplementedError

  elif split == 'val':
    if 'cc3m' in args.val_dataset:
      dataset_paths.append(os.path.join(args.dataset_dir, 'cc3m_val.tsv'))
      image_data_dirs.append(os.path.join(args.image_dir, 'cc3m/validation'))
    else:
      raise NotImplementedError

    assert len(dataset_paths) == len(image_data_dirs) == 1, (dataset_paths, image_data_dirs)
  else:
    raise NotImplementedError

  if len(dataset_paths) > 1:
    print(f'{len(dataset_paths)} datasets requested: {dataset_paths}')
    dataset = torch.utils.data.ConcatDataset([
      CsvDataset(path, image_dir, tokenizer, 'image',
        'caption', args.visual_model, train=train, max_len=args.max_len, precision=args.precision,
        image_size=args.image_size, retrieval_token_idx=args.retrieval_token_idx)
      for (path, image_dir) in zip(dataset_paths, image_data_dirs)])
  elif len(dataset_paths) == 1:
    dataset = CsvDataset(dataset_paths[0], image_data_dirs[0], tokenizer, 'image',
      'caption', args.visual_model, train=train, max_len=args.max_len, precision=args.precision,
      image_size=args.image_size, retrieval_token_idx=args.retrieval_token_idx)
  else:
    raise ValueError(f'There should be at least one valid dataset, got train={args.dataset}, val={args.val_dataset} instead.')
  return dataset


class CsvDataset(Dataset):
  def __init__(self, input_filename, base_image_dir, tokenizer, img_key,
               caption_key, feature_extractor_model: str,
               train: bool = True, max_len: int = 32, sep="\t", precision: str = 'fp32',
               image_size: int = 224, retrieval_token_idx: int = -1):
    logging.debug(f'Loading tsv data from {input_filename}.')
    df = pd.read_csv(input_filename, sep=sep)

    self.base_image_dir = base_image_dir
    self.images = df[img_key].tolist()
    self.captions = df[caption_key].tolist()
    assert len(self.images) == len(self.captions)

    self.feature_extractor_model = feature_extractor_model
    self.feature_extractor = utils.get_feature_extractor_for_model(
      feature_extractor_model, image_size=image_size, train=False)
    self.image_size = image_size

    self.tokenizer = tokenizer
    self.max_len = max_len
    self.precision = precision
    self.retrieval_token_idx = retrieval_token_idx

    self.font = None

    logging.debug('Done loading data.')

  def __len__(self):
    return len(self.captions)

  def __getitem__(self, idx):
    while True:
      image_path = os.path.join(self.base_image_dir, str(self.images[idx]))
      caption = str(self.captions[idx])

      try:
        img = Image.open(image_path)
        images = utils.get_pixel_values_for_model(self.feature_extractor, img)

        caption += '[RET]'
        tokenized_data = self.tokenizer(
          caption,
          return_tensors="pt",
          padding='max_length',
          truncation=True,
          max_length=self.max_len)
        tokens = tokenized_data.input_ids[0]

        caption_len = tokenized_data.attention_mask[0].sum()

        decode_caption = self.tokenizer.decode(tokens, skip_special_tokens=False)
        self.font = self.font or ImageFont.load_default()
        cap_img = utils.create_image_of_text(decode_caption.encode('ascii', 'ignore'), width=self.image_size, nrows=2, font=self.font)

        if tokens[-1] not in [self.retrieval_token_idx, self.tokenizer.pad_token_id]:
          tokens[-1] = self.retrieval_token_idx

        return image_path, images, cap_img, tokens, caption_len
      except Exception as e:
        print(f'Error reading {image_path} with caption {caption}: {e}')
        # Pick a new example at random.
        idx = np.random.randint(0, len(self)-1)
