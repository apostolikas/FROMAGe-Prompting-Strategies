import os
from fromage import models
import json 
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, CLIPSegForImageSegmentation, OneFormerProcessor, OneFormerForUniversalSegmentation, AutoTokenizer, AutoModel
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


if __name__ == '__main__':

    processor_clip = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    vis_model_clip = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    processor_oneformer = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
    vis_model_oneformer = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")

    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    lm = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    model_dir = './fromage_model/'
    model = models.load_fromage(model_dir)

    with open('./guided_vqa/guided_vqa_shots_1_ways_2_all_questions.json', 'r') as f:
        vqa_data = json.load(f)

    np.random.seed(0)
    np.random.shuffle(vqa_data)
    vqa_sublist = vqa_data[:300]
    vist_images_folder = './guided_vqa'

    unaugmented_scores = []
    augmented_clip_scores = []
    augmented_onerformer_scores = []

    for vqa_dict in vqa_sublist:
        
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

        encoded_unaugmented_input = tokenizer(model_outputs_original, padding=True, truncation=True, return_tensors='pt')
        encoded_augmented_clip_input = tokenizer(model_outputs_clip_segment, padding=True, truncation=True, return_tensors='pt')
        encoded_augmented_oneformer_input = tokenizer(model_outputs_oneformer_segment, padding=True, truncation=True, return_tensors='pt')
        encoded_target_input = tokenizer(answer, padding=True, truncation=True, return_tensors='pt')

        with torch.no_grad():
            model_unaugmented_output = lm(**encoded_unaugmented_input)
            model_augmented_clip_output = lm(**encoded_augmented_clip_input)
            model_augmented_oneformer_output = lm(**encoded_augmented_oneformer_input)
            model_target_output = lm(**encoded_target_input)

        unaugmented_embeddings = F.normalize(mean_pooling(model_unaugmented_output, encoded_unaugmented_input['attention_mask']), p=2, dim=1)
        augmented_clip_embeddings = F.normalize(mean_pooling(model_augmented_clip_output, encoded_augmented_clip_input['attention_mask']), p=2, dim=1)
        augmented_oneformer_embeddings = F.normalize(mean_pooling(model_augmented_oneformer_output, encoded_augmented_oneformer_input['attention_mask']), p=2, dim=1)
        target_embeddings = F.normalize(mean_pooling(model_target_output, encoded_target_input['attention_mask']), p=2, dim=1)

        augmented_clip_score = cos_sim(augmented_clip_embeddings, target_embeddings)
        augmented_clip_scores.append(augmented_clip_score.item())

        augmented_onerformer_score = cos_sim(augmented_oneformer_embeddings, target_embeddings)
        augmented_onerformer_scores.append(augmented_onerformer_score.item())

        unaugmented_score = cos_sim(unaugmented_embeddings, target_embeddings)
        unaugmented_scores.append(unaugmented_score.item())

        # print("The question is :",question)
        # print("Model output :", model_outputs_original)
        # print("Model output using clip seg:", model_outputs_clip_segment)
        # print("Model output using oneformer seg:", model_outputs_oneformer_segment)
        # print("Ground truth :", answer)
        # print("\n")

    print("Average Cosine Similarity with target using CLIP seg :",np.mean(augmented_clip_scores))
    print("Average Cosine Similarity with target using oneformer seg :",np.mean(augmented_onerformer_scores))
    print("Average Cosine Similarity with target without augmentations :",np.mean(unaugmented_scores))

