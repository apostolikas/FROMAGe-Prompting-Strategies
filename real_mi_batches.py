from src.image_classification.classification_utils import load_pickle, set_seed, create_pickle, remove_caption_prefix
from fromage import models
import torch.utils.data as data
from transformers import GPT2Tokenizer
import math, random
from tqdm import tqdm
import torch
import pickle
#https://github.com/huggingface/transformers/issues/6535
opt_version = 'facebook/opt-6.7b'
tokenizer = GPT2Tokenizer.from_pretrained(opt_version)
tokenizer.pad_token = tokenizer.eos_token
# Add special tokens to the model to enable [RET].
tokenizer.add_special_tokens({"cls_token": "<|image|>"})
tokenizer.add_tokens('[RET]')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_acc(pairs, dict_model_input):
    tp=0
    tp_new=0
    all_counter = 0
    tp_1 = 0
    lens=[]
    small_lens=[]
    indices = []
    pr_1=0
    for i,(pr,true) in enumerate(pairs):
        if remove_caption_prefix(dict_model_input[i+1][1]) in pr[0]:
            pr_1+=1
        if true in pr[0]:
            tp+=1
        tok_=tokenizer.encode_plus(pr[0])['input_ids']
        new_pred = tokenizer.batch_decode([tok_[:7]])
        if true in new_pred[0]:
            tp_new+=1
            indices.append(i)

        lens.append(len(tok_))

    print(f'predict first {pr_1}')
    m = max(lens) if len(lens) >0 else 0
    print(f'tp {tp} tp_new {tp_new}')
    acc= tp/len(pairs)
    print(f'acc: {acc} pairs: {len(pairs)}')
pickles_names = [#'all_constr_content_free.pickle',
                 'all_mini_net.pickle','NO_constr_content_free.pickle','NO_constr_base_ways2_order_sim_last.pickle',
                 'constr_base_ways2_order_sim.pickle','constr_content_free_ways2_word7.pickle', 'all_constr_baseline.pickle',
                 'constr_content_free_ways5_word7_all.pickle','constr_base_ways5_order_sim_last_word7.pickle',
                        'constr_base_ways5_last_word7.pickle'
                 ]#,'NO_constr_base_ways2_order_sim_last.pickle']

# pickles_names = ['constr_base_ways2_order_sim.pickle',
#                  'constr_content_free_ways5.pickle']
# pickles_names = ['constr_base_ways5_order_sim.pickle',
#                  'constr_base_ways5_order_sim_last.pickle']
#dict_model_input_5 = load_pickle('dict_model_input_int_ways_5.pickle')
dict_model_input = load_pickle('dict_model_input_int_ways_2.pickle')

path = './all_pickles/'
for pickle_name in pickles_names:
    print(pickle_name)
    if '5' in pickle_name:
        pairs = load_pickle(path+pickle_name)
        calculate_acc(pairs,dict_model_input)
    else:
        pairs = load_pickle(path+pickle_name)
        calculate_acc(pairs,dict_model_input)


a=2

classes=[' electric guitar',			
' golden retriever',
' malamute',
' mixing bowl',
' cuirass',
' dalmatian',
' african hunting dog',
' lion',
' crate',
' bookshop',
' vase',
' nematode',
' hourglass',
' ant',
' king crab',
' black-footed ferret',
' scoreboard',
' theater curtain',
' school bus',
' trifle']

opt_version = 'facebook/opt-6.7b'
tokenizer = GPT2Tokenizer.from_pretrained(opt_version)
tokenizer.pad_token = tokenizer.eos_token
# Add special tokens to the model to enable [RET].
tokenizer.add_special_tokens({"cls_token": "<|image|>"})
tokenizer.add_tokens('[RET]')


#let's check content free logits



class_input_ids = []
set_class_ids = set()
for class_ in classes:
    ids = tokenizer.encode_plus(class_)['input_ids'][1:]
    class_input_ids.append(ids)
    set_class_ids.update(ids)

set_class_ids = torch.tensor(list(set_class_ids))
x= torch.rand(1,45216)
sub_x = torch.index_select(x, 1, set_class_ids)
sub_id = torch.argmax(sub_x, keepdim=True, dim=-1).item()
t=0
tokenizer.batch_decode(torch.topk(content_free[0][t], 10).indices.tolist())
a=1




tp=0
cor1 = 0
cor1_pred2 = 0
cor2 = 0
pred2=0
pred1=0
cor2_pred1 = 0
pred_strings = []
recency_bias_ids=[]
for (gen_tokens, true_label), input_id in zip(pairs[:55], dict_model_input):
    cur_input = dict_model_input[input_id]
    if true_label in cur_input[1]: #correct the first ones
        cor1 += 1
        if remove_caption_prefix(cur_input[3]) in gen_tokens[0]: #predict the second one
            cor1_pred2 += 1
            recency_bias_ids.append(input_id)

    if true_label in cur_input[3]: #correct the first ones
        cor2 += 1
        if remove_caption_prefix(cur_input[1]) in gen_tokens[0]: #predict the second one
            cor2_pred1 += 1

    if remove_caption_prefix(cur_input[3]) in gen_tokens[0]: #predict the second one
            pred2 += 1

    elif remove_caption_prefix(cur_input[1]) in gen_tokens[0]: #predict the second one
            pred1 += 1

    if true_label in gen_tokens[0]:
        tp+=1

a=1
# def tokenize(example):
#   encodings = tokenizer(example,return_tensors="pt")
#   return encodings

# dict_model_input = load_pickle('dict_model_input_int.pickle')
# keys = load_pickle('keys_rand_order.pickle')
list_input_timestapes = load_pickle('list_input_timestape.pickle')
dict_question_captions = load_pickle('dict_question_captions.pickle')
keys = load_pickle('keys_rand_order.pickle')
question_captions = list(dict_question_captions.values())

img_ex1, text_ex1, img_ex2, text_ex2, img_q, text_q_prompt = list_input_timestapes

# tokenized_text_ex1_input_ids, tokenized_text_ex2_input_ids, tokenized_text_q_prompt_input_ids = [], [], []
# tokenized_text_ex1_attention_mask, tokenized_text_ex2_attention_mask, tokenized_text_q_prompt_attention_mask = [], [], []
# for tmp_text_ex1, tmp_text_ex2, tmp_text_q_prompt in zip(text_ex1, text_ex2, text_q_prompt):
    # tok_ex1 = tokenizer(tmp_text_ex1)
    # tokenized_text_ex1_input_ids.append(tok_ex1['input_ids'])
    # # tokenized_text_ex1_attention_mask.append(tok_ex1['attention_mask'])

    # tok_ex2 = tokenizer(tmp_text_ex2)
    # tokenized_text_ex2_input_ids.append(tok_ex2['input_ids'])
    # # tokenized_text_ex2_attention_mask.append(tok_ex2['attention_mask'])

    # tok_q_prompt = tokenizer(tmp_text_q_prompt)
    # tokenized_text_q_prompt_input_ids.append(tok_q_prompt['input_ids'])
    # tokenized_text_q_prompt_attention_mask.append(tok_q_prompt['attention_mask'])

set_seed(0)
# keys = random.shuffle([i for i in range(len(tokenized_text_ex1))])
# img_ex1, text_ex1 = img_ex1[keys], text_ex1
# img_ex2, text_ex2, img_q, text_q_prompt



# Load model used in the paper.
model_dir = './fromage_model/'
model = models.load_fromage(model_dir)

batch_size = 2
n_batches = math.ceil(len(list_input_timestapes[0]) / batch_size)

pairs = []
for i in tqdm(range(n_batches)):
    print('batch size = ', batch_size)
    # tmp_img_ex1, tmp_text_ex1 = img_ex1[i*batch_size:(i+1)*batch_size], text_ex1[i*batch_size:(i+1)*batch_size]
    # tmp_img_ex2, tmp_text_ex2 = img_ex2[i*batch_size:(i+1)*batch_size], text_ex2[i*batch_size:(i+1)*batch_size]
    # tmp_img_q, tmp_text_q_prompt = img_q[i*batch_size:(i+1)*batch_size], text_q_prompt[i*batch_size:(i+1)*batch_size]
    # tmp_question_captions = question_captions[i*batch_size:(i+1)*batch_size]

    tmp_img_ex1 = [img_ex1[key-1] for key in keys[:25]]
    tmp_text_ex1 = [text_ex1[key-1] for key in keys[:25]]

    tmp_img_ex2 = [img_ex2[key-1] for key in keys[:25]]
    tmp_text_ex2 = [text_ex2[key-1] for key in keys[:25]]

    tmp_img_q = [img_q[key-1] for key in keys[:25]]
    tmp_text_q_prompt = [text_q_prompt[key-1] for key in keys[:25]]
    tmp_question_captions = [question_captions[key-1] for key in keys[:25]]
    # for a,b,c in zip(tmp_text_ex1, tmp_text_ex2, tmp_question_captions):
    #     print(remove_caption_prefix(a),' ',remove_caption_prefix(b),' ', c)

    #img1, text1, img2, text2, img_question, prompt_question
    timesteps = [tmp_img_ex1, tmp_text_ex1, tmp_img_ex2, tmp_text_ex2, tmp_img_q, tmp_text_q_prompt]#, img_ex2, text_ex2, img_q, text_q_prompt]
    

    print(type(timesteps[0][0]))
    model_outputs = model.generate_for_images_and_texts(timesteps, num_words=15, ret_scale_factor=0)
    pairs.append((model_outputs,tmp_question_captions))
    batch_size = batch_size*2
    break
create_pickle('batches_new.pickle',pairs)