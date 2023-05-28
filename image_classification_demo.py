from PIL import Image
import os
from src.image_classification.fromage import models
from src.image_classification.classification_utils import load_pickle

model_dir = './fromage_model/'
model = models.load_fromage(model_dir)

model.calculate_black_image()
model.calculate_white_image()
question_image_1 = Image.open(os.path.join('./demo_images/','image_classification_q.jpg')).resize((224, 224)).convert('RGB')
image_1 = Image.open(os.path.join('./demo_images/','image_classification_ex1.jpg')).resize((224, 224)).convert('RGB')
image_2 = Image.open(os.path.join('./demo_images/','image_classification_ex2.jpg')).resize((224, 224)).convert('RGB')
label = 'hourglass'
prompt =[image_1,'This is a ant', image_2,'This is a hourglass',
            question_image_1,'This is a']

# baseline unconstrained
model_outputs = model.generate_for_images_and_texts(prompt, num_words=9,max_num_rets=0)
print('--'*10)
print(f'Baseline prediction: {model_outputs[0]} TRUE: {label}')


# content free extensio unconstrained
model_outputs = model.other_generate(prompt, num_words=9,max_num_rets=0,
                                                    constrained_ids=None, baseline=False)
print('--'*10)
print(f'Content free extension prediction: {model_outputs[0]} TRUE: {label}')
score1, score2 = load_pickle(os.path.join('./src/image_classification/similarities/','sim_img2img_2.pickle'))[8]


# # image similarity extension unconstrained
sim_prompt = prompt
if score1 > score2:
    sim_prompt = [prompt[2], prompt[3], prompt[0], prompt[1], prompt[4], prompt[5] ]
model_outputs = model.generate_for_images_and_texts(sim_prompt, num_words=9,max_num_rets=0)
print('--'*10)
print(f'Image similarity extension prediction: {model_outputs[0]} TRUE: {label}')
