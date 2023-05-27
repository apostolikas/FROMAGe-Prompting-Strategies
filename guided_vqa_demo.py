import os
from fromage import models
import json 
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, CLIPSegForImageSegmentation, OneFormerProcessor, OneFormerForUniversalSegmentation, AutoTokenizer, AutoModel
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F
from src.visual_qa.gvqa_visual_augmentation import cos_sim, mean_pooling


def clipseg_segment_image(question_image):
    encoded_image = processor_clip(images=[question_image], return_tensors = 'pt')
    outputs = vis_model_clip(**encoded_image, conditional_pixel_values = encoded_image.pixel_values)
    segmented_image = outputs.logits.unsqueeze(0)
    segmented_pil_image = to_pil_image(segmented_image).resize((224, 224)).convert('RGB')
    return segmented_pil_image
    

def oneformer_segment_image(question_image):
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
    
    return semantic_map, instance_map, panoptic_map


def compute_score(model_outputs_original, model_outputs_clip_segment,model_outputs_oneformer_segment,answer):
    # Tokenize the input
    encoded_unaugmented_input = tokenizer(model_outputs_original, padding=True, truncation=True, return_tensors='pt')
    encoded_augmented_clip_input = tokenizer(model_outputs_clip_segment, padding=True, truncation=True, return_tensors='pt')
    encoded_augmented_oneformer_input = tokenizer(model_outputs_oneformer_segment, padding=True, truncation=True, return_tensors='pt')
    encoded_target_input = tokenizer(answer, padding=True, truncation=True, return_tensors='pt')

    # FF through the model
    with torch.no_grad():
        model_unaugmented_output = lm(**encoded_unaugmented_input)
        model_augmented_clip_output = lm(**encoded_augmented_clip_input)
        model_augmented_oneformer_output = lm(**encoded_augmented_oneformer_input)
        model_target_output = lm(**encoded_target_input)

    # Process the embeddings
    unaugmented_embeddings = F.normalize(mean_pooling(model_unaugmented_output, encoded_unaugmented_input['attention_mask']), p=2, dim=1)
    augmented_clip_embeddings = F.normalize(mean_pooling(model_augmented_clip_output, encoded_augmented_clip_input['attention_mask']), p=2, dim=1)
    augmented_oneformer_embeddings = F.normalize(mean_pooling(model_augmented_oneformer_output, encoded_augmented_oneformer_input['attention_mask']), p=2, dim=1)
    target_embeddings = F.normalize(mean_pooling(model_target_output, encoded_target_input['attention_mask']), p=2, dim=1)

    # Compute cosine similarity
    augmented_clip_score = cos_sim(augmented_clip_embeddings, target_embeddings)
    augmented_onerformer_score = cos_sim(augmented_oneformer_embeddings, target_embeddings)
    unaugmented_score = cos_sim(unaugmented_embeddings, target_embeddings)

    return augmented_clip_score, augmented_onerformer_score, unaugmented_score



if __name__ == '__main__':

    print('Loading models...')
    # Load the FROMAGe model used in the paper.
    model_dir = './fromage_model/'
    model = models.load_fromage(model_dir)

    # Load the first model for segmentation of the query image (CLIPSeg)
    processor_clip = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    vis_model_clip = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    # Load the second model for segmentation of the query image (Oneformer)
    processor_oneformer = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
    vis_model_oneformer = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")

    # Load the language model to extract text embeddings (all-MiniLM-L6-v2)
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    lm = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    print('Models loaded successfully!')

    # Load data
    print('Loading data...')

    question_image_1 = Image.open(os.path.join('./src/visual_qa/demo_images/','gvqa_1_3.jpg')).resize((224, 224)).convert('RGB')
    image_1 = Image.open(os.path.join('./src/visual_qa/demo_images/','gvqa_1_1.jpg')).resize((224, 224)).convert('RGB')
    image_2 = Image.open(os.path.join('./src/visual_qa/demo_images/','gvqa_1_2.jpg')).resize((224, 224)).convert('RGB')

    question_image_2 = Image.open(os.path.join('./src/visual_qa/demo_images/','gvqa_2_3.jpg')).resize((224, 224)).convert('RGB')
    image_3 = Image.open(os.path.join('./src/visual_qa/demo_images/','gvqa_2_1.jpg')).resize((224, 224)).convert('RGB')
    image_4 = Image.open(os.path.join('./src/visual_qa/demo_images/','gvqa_2_2.jpg')).resize((224, 224)).convert('RGB')
    
    vqa_list = [
                [image_1, "This is a coat", image_2, "This is a jacket", question_image_1, "What kind of a coat is that?", "That is a blue coat."],
                [image_3, "This is a sink", image_4, "This is a cabinet", question_image_2, "What color is the sink on the left?", "Black."]
                ]
    print('Data loaded successfully!')

    print('Inference loop for 2 samples starts.\n')
    for vqa_data in vqa_list:
        
        image1 = vqa_data[0]
        caption1 = vqa_data[1]
        image2 = vqa_data[2]
        caption2 = vqa_data[3]
        question_image = vqa_data[4]
        question = vqa_data[5]
        answer = vqa_data[6]

        # ClipSeg - segment query image 
        segmented_pil_image = clipseg_segment_image(question_image)

        # Oneformer - segment query image
        semantic_map, instance_map, panoptic_map = oneformer_segment_image(question_image)

        # Generate output using the original query image
        model_input_original = [ 
                    image1, caption1, 
                    image2, caption2, 
                    question_image, 
                    'Q: ' + question]
        model_outputs_original = model.generate_for_images_and_texts(model_input_original, num_words=15, max_num_rets=0)

        # Generate output using visual augmented prompt (Oneformer)
        model_input_oneformer_segment = [ 
                    image1, caption1, 
                    image2, caption2, 
                    semantic_map, instance_map, panoptic_map, 
                    question_image, 
                    'Q: ' + question]
        model_outputs_oneformer_segment = model.generate_for_images_and_texts(model_input_oneformer_segment, num_words=15, max_num_rets=0)

        # Generate output using visual augmented prompt (CLIPSeg)
        model_input_clip_segment = [ 
                    image1, caption1, 
                    image2, caption2, 
                    segmented_pil_image, question_image, 
                    'Q: ' + question]
        model_outputs_clip_segment = model.generate_for_images_and_texts(model_input_clip_segment, num_words=15, max_num_rets=0)

        # Compute the scores by comparing the text embeddings
        augmented_clip_score, augmented_onerformer_score, unaugmented_score = compute_score(model_outputs_original, model_outputs_clip_segment,model_outputs_oneformer_segment,answer)
        
        print("\n")
        print("The question is :", question, " and the answer is : " ,answer)
        print("Cos sim between unaugmented output - answer : {:.3f}".format(unaugmented_score.item()))
        print("Cos sim between output using CLIPSeg - answer : {:.3f}".format(augmented_clip_score.item()))
        print("Cos sim between output using Oneformer - answer : {:.3f}".format(augmented_onerformer_score.item()))
        print("\n")





