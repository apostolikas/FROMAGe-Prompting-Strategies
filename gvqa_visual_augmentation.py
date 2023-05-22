import os
from fromage import models
import json 
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoProcessor, CLIPSegForImageSegmentation, OneFormerProcessor, OneFormerForUniversalSegmentation
from torchvision.transforms.functional import to_pil_image


processor_clip = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
vis_model_clip = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

# Loading a single model for all three tasks
processor_oneformer = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
vis_model_oneformer = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")

# Load model used in the paper.
model_dir = './fromage_model/'
model = models.load_fromage(model_dir)

# Load data
with open('./guided_vqa/guided_vqa_shots_1_ways_2_all_questions.json', 'r') as f:
    vqa_data = json.load(f)

# Take a few instances (the data is a list of dictionaries)
np.random.seed(0)
np.random.shuffle(vqa_data)
vqa_sublist = vqa_data[:5]
vist_images_folder = './guided_vqa'

i=0
for vqa_dict in vqa_sublist:
    
    i+=1

    image1_path = vqa_dict['image_1']
    image1 = Image \
        .open(os.path.join(vist_images_folder,image1_path)) \
        .resize((224, 224)) \
        .convert('RGB')
    caption1 = vqa_dict['caption_1']

    image2_path = vqa_dict['image_2']
    image2 = Image \
        .open(os.path.join(vist_images_folder,image2_path)) \
        .resize((224, 224)) \
        .convert('RGB')
    caption2 = vqa_dict['caption_2']

    question_image_path = vqa_dict['question_image']
    question_image = Image \
        .open(os.path.join(vist_images_folder,question_image_path)) \
        .resize((224, 224)) \
        .convert('RGB')
    question = vqa_dict['question']
    answer = vqa_dict['answer']

    # ClipSeg
    encoded_image = processor_clip(images=[question_image], return_tensors = 'pt')
    outputs = vis_model_clip(**encoded_image, conditional_pixel_values = encoded_image.pixel_values)
    segmented_image = outputs.logits.unsqueeze(0)
    segmented_pil_image = to_pil_image(segmented_image).resize((224, 224)).convert('RGB')

    # Oneformer
    # Semantic Segmentation
    semantic_inputs = processor_oneformer(images=question_image, task_inputs=["semantic"], return_tensors="pt")
    semantic_outputs = vis_model_oneformer(**semantic_inputs)
    predicted_semantic_map = processor_oneformer.post_process_semantic_segmentation(semantic_outputs, target_sizes=[question_image.size[::-1]])[0]
    semantic_map = to_pil_image(predicted_semantic_map.float())
    # Instance Segmentation
    instance_inputs = processor_oneformer(images=question_image, task_inputs=["instance"], return_tensors="pt")
    instance_outputs = vis_model_oneformer(**instance_inputs)
    predicted_instance_map = processor_oneformer.post_process_instance_segmentation(instance_outputs, target_sizes=[question_image.size[::-1]])[0]["segmentation"]
    instance_map = to_pil_image(predicted_instance_map.float())
    # Panoptic Segmentation
    panoptic_inputs = processor_oneformer(images=question_image, task_inputs=["panoptic"], return_tensors="pt")
    panoptic_outputs = vis_model_oneformer(**panoptic_inputs)
    predicted_panoptic_map = processor_oneformer.post_process_panoptic_segmentation(panoptic_outputs, target_sizes=[question_image.size[::-1]])[0]["segmentation"]
    panoptic_map = to_pil_image(predicted_panoptic_map.float())


    model_input_original = [ 
                image1, caption1, 
                image2, caption2, 
                question_image, 
                'Q: ' + question]
    model_outputs_original = model.generate_for_images_and_texts(model_input_original, num_words=15, max_num_rets=0)


    model_input_oneformer_segment = [ 
                image1, caption1, 
                image2, caption2, 
                semantic_map, instance_map, panoptic_map, 
                question_image, 
                'Q: ' + question]
    model_outputs_oneformer_segment = model.generate_for_images_and_texts(model_input_oneformer_segment, num_words=15, max_num_rets=0)

 
    model_input_clip_segment = [ 
                image1, caption1, 
                image2, caption2, 
                segmented_pil_image, question_image, 
                'Q: ' + question]
    model_outputs_clip_segment = model.generate_for_images_and_texts(model_input_clip_segment, num_words=15, max_num_rets=0)

    print("The question is :",question)
    print("Model output :", model_outputs_original)
    print("Model output using clip seg:", model_outputs_clip_segment)
    print("Model output using oneformer seg:", model_outputs_oneformer_segment)
    print("Ground truth :", answer)
    print("\n")

