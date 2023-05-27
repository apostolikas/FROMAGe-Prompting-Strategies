from my_utils import load_pickle,change_order, change_order_5
import argparse
import torch
from tqdm import tqdm
from my_utils import load_pickle, create_pickle

# from transformers import ViTImageProcessor, ViTModel
# def calculate_similarities(dict_model_input, num_ways):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     processor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224-in21k') #https://huggingface.co/google/vit-large-patch16-224-in21k
#     model = ViTModel.from_pretrained('google/vit-large-patch16-224-in21k')
#     model.to(device)

#     img_sim = []
#     print('now we will calculate img2img similarities')
#     with torch.no_grad():
#         for i in tqdm(dict_model_input):
#             input = dict_model_input[i]

#             images = [input[i] for i in range(0,len(input),2)]
#             inputs = processor(images=images, return_tensors="pt")
#             inputs=inputs.to(device)

#             outputs = model(**inputs)
#             pooler_output = outputs.pooler_output

#             scores = [torch.nn.functional.cosine_similarity(pooler_output[i], pooler_output[-1],dim=0).item() for 
#                       i in range(num_ways)]
#             assert(len(scores)==num_ways)
#             img_sim.append(scores)
        
#     create_pickle(f'sim_img2img_{num_ways}.pickle',img_sim)
#     return img_sim

def get_similarities(num_ways):
    img_sim = load_pickle(f'sim_img2img_{num_ways}.pickle')
    return img_sim

def get_ordered_dict(dict_model_input, num_ways):
    # similarities = calculate_similarities(dict_model_input, num_ways)
    similarities = get_similarities(num_ways)

    if num_ways == 2:
        new_dict_model_input = change_order(dict_model_input, similarities)
    elif num_ways == 5:
        new_dict_model_input = change_order_5(dict_model_input, similarities)
    else:
        raise ValueError('Number of ways can only be 2 or 5')
    return new_dict_model_input