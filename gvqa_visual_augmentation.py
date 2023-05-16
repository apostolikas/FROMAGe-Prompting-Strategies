import os
import numpy as np
from fromage import models
import json 
import copy
import torch
from PIL import Image
import matplotlib as plt
from torchvision import transforms
from transformers import AutoProcessor, CLIPSegForImageSegmentation,AutoTokenizer
from torchvision.transforms.functional import to_pil_image
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation


from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation


processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
vis_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

# Loading a single model for all three tasks
#processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
#vis_model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")

# image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-tiny")
# vis_model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-tiny")

# Load model used in the paper.
model_dir = './fromage_model/'
model = models.load_fromage(model_dir)


convert_tensor = transforms.ToTensor()

# Load data
with open('./guided_vqa/guided_vqa_shots_1_ways_2_all_questions.json', 'r') as f:
    vqa_data = json.load(f)

# Take a few instances (the data is a list of dictionaries)
# np.random.seed(0)
# np.random.shuffle(vqa_data)
vqa_sublist = vqa_data[1:3]
vist_images_folder = './guided_vqa'
i=0
for vqa_dict in vqa_sublist:
    #vqa_dict = vqa_sublist[14]
    
    # ------------------------------ #
    i+=1
    #! Image 1
    image1_path = vqa_dict['image_1']
    image1 = Image \
        .open(os.path.join(vist_images_folder,image1_path)) \
        .resize((224, 224)) \
        .convert('RGB')
    caption1 = vqa_dict['caption_1']
    # ------------------------------ #
    #! Input2
    image2_path = vqa_dict['image_2']
    image2 = Image \
        .open(os.path.join(vist_images_folder,image2_path)) \
        .resize((224, 224)) \
        .convert('RGB')
    caption2 = vqa_dict['caption_2']
    # ------------------------------ #
    #! Question 
    question_image_path = vqa_dict['question_image']
    question_image = Image \
        .open(os.path.join(vist_images_folder,question_image_path)) \
        .resize((224, 224)) \
        .convert('RGB')
    #question_image.save(str(i)+'im1.jpg')
    question = vqa_dict['question']
    #question = 'Caption the image.'
    answer = vqa_dict['answer']

    encoded_image = processor(images=[question_image], return_tensors = 'pt')
    #print(inputs['pixel_values'].shape)
    outputs = vis_model(**encoded_image, conditional_pixel_values = encoded_image.pixel_values)
    #print(outputs)
    segmented_image = outputs.logits.unsqueeze(0)
    print(segmented_image.shape)
    segmented_pil_image1 = to_pil_image(segmented_image).resize((224, 224)).convert('RGB')
    segmented_pil_image1.save('im1.jpg')

    # Semantic Segmentation
    # semantic_inputs = processor(images=question_image, task_inputs=["semantic"], return_tensors="pt")
    # semantic_outputs = vis_model(**semantic_inputs)
    # # pass through image_processor for postprocessing
    # predicted_semantic_map = processor.post_process_semantic_segmentation(semantic_outputs, target_sizes=[question_image.size[::-1]])[0]
    # segmented_pil_image1 = to_pil_image(predicted_semantic_map.float())
    # segmented_pil_image1.save('im1.jpg')

    # # Instance Segmentation
    # instance_inputs = processor(images=question_image, task_inputs=["instance"], return_tensors="pt")
    # instance_outputs = vis_model(**instance_inputs)
    # # pass through image_processor for postprocessing
    # predicted_instance_map = processor.post_process_instance_segmentation(instance_outputs, target_sizes=[question_image.size[::-1]])[0]["segmentation"]
    # print(predicted_instance_map.shape)
    # segmented_pil_image2 = to_pil_image(predicted_instance_map.float())
    # segmented_pil_image2.save('im2.jpg')


    # # Panoptic Segmentation
    # panoptic_inputs = processor(images=question_image, task_inputs=["panoptic"], return_tensors="pt")
    # panoptic_outputs = vis_model(**panoptic_inputs)
    # # pass through image_processor for postprocessing
    # predicted_semantic_map = processor.post_process_panoptic_segmentation(panoptic_outputs, target_sizes=[question_image.size[::-1]])[0]["segmentation"]
    # print(predicted_semantic_map.shape)
    # segmented_pil_image3 = to_pil_image(predicted_semantic_map.float())
    # segmented_pil_image3.save('im3.jpg')


    # prompt2_for_ret = 'Give a similar image to the previous one [RET]'
    # prompt2 = [question_image, prompt2_for_ret]
    # outs2 = model.generate_for_images_and_texts(prompt2, max_img_per_ret=2) 
    # for out2 in outs2:
    #         if type(out2) == str:
    #             continue
    #         elif type(out2) == list:
    #             similar_image4 = out2[0]
    #             similar_image4.save(str(i)+'sim1.jpg')
    #             similar_image5 = out2[1]
    #             similar_image5.save(str(i)+'sim2.jpg')
    #             # similar_image6 = out2[2]
    #         else:
    #             continue

    # ------------------------------ #

    #! Model output
    model_input = [ image1, 
                caption1, 
                image2, 
                caption2, 
                # segmented_pil_image2,
                # segmented_pil_image3,
                question_image, 
                segmented_pil_image1,

                #similar_image4,
                #similar_image5,
                #question]
                'Q: ' + question] #+ ' A: ']


    #! Prompt and output
    print("Model input : ", model_input)
    model_outputs = model.generate_for_images_and_texts(model_input, num_words=15, max_num_rets=0)
    print("Model output :", model_outputs)
    print("Ground truth :", answer)
    print('\n')

