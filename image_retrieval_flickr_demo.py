from PIL import Image
import os
from fromage import models
from torchvision.transforms import ToTensor
import torch


if __name__ == '__main__':

    print('Loading the models...')
    # Load the FROMAGe model used in the paper.
    model_dir = './fromage_model/'
    model = models.load_fromage(model_dir)
    print('Models loaded successfully!')

    print('Loading data...')

    image1 = Image.open(os.path.join('./src/image_retrieval_flickr/demo_images/','ir_1.jpg')).resize((224, 224)).convert('RGB')
    caption1 = 'Three boys play around a fountain in an office building courtyard .'
    extended_caption1 = 'Three boys play around a fountain in an office building courtyard, water-splashing, energetic, joyful, playful, urban, outdoor.'

    image2 = Image.open(os.path.join('./src/image_retrieval_flickr/demo_images/','ir_2.jpg')).resize((224, 224)).convert('RGB')
    caption2 = 'The kid is on a float in the snow .'
    extended_caption2 = 'The kid is on a float in the snow, enjoying, sled, winter, cold, snowflakes, outdoor activities, recreation.'

    flickr_data = [[image1,caption1,extended_caption1],[image2,caption2,extended_caption2]]
    print('Data loaded successfully!')

    j=0

    print('Inference loop for 2 samples starts.')
    for flick_list in flickr_data:

        image = flick_list[0]
        original_caption = flick_list[1]
        augmented_caption = flick_list[2]
        j+=1

        try:
            # Retrieve image based on the original caption
            original_prompt = [original_caption[:-1] + ' [RET] ']
            model_output_orig = model.generate_for_images_and_texts(original_prompt, max_img_per_ret=1, max_num_rets=1, num_words=0)
            unaugmented_output = model_output_orig[-1][0]

            # Retrieve image based on the augmented caption
            augmented_prompt = [image, augmented_caption[:-1] + ' [RET] ']
            model_output = model.generate_for_images_and_texts(augmented_prompt, max_img_per_ret=1, max_num_rets=1, num_words=0)
            augmented_output = model_output[-1][0]

            # Evaluation metric for the two retrieved images (with and without using text augmentation)
            transform = ToTensor()
            model = model.to(torch.float16)
            with torch.no_grad():
                embedding_augmented = model.model.get_visual_embs((transform(augmented_output)).unsqueeze(0).half().cuda())
                embedding_unaugmented = model.model.get_visual_embs((transform(unaugmented_output)).unsqueeze(0).half().cuda())
                embedding_original = model.model.get_visual_embs((transform(image)).unsqueeze(0).half().cuda())
            model = model.bfloat16()

            similarity_score_unaugmented = torch.nn.functional.cosine_similarity(embedding_unaugmented.float(), embedding_original.float())
            similarity_score_augmented = torch.nn.functional.cosine_similarity(embedding_augmented.float(), embedding_original.float())
            print("Example ",j)
            print("Similarity of image retrieved without augmentation with the target image is : {:.3f}".format(similarity_score_unaugmented.mean(1).item()))
            print("Similarity of image retrieved using text-augmentation with the target image is : {:.3f}".format(similarity_score_augmented.mean(1).item()))
            print("\n")
            
        except:
            continue
        