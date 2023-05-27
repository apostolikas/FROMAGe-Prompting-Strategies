from fromage import models
from src.gif_captioning.scripts import help_functions as hf


###### CHOOSE A GIF EXAMPLE ######


gif_url = 'https://38.media.tumblr.com/88091baac04b18ef73d93e56883dccfe/tumblr_npzfsr13I51uyqfd5o1_500.gif'
gif_caption = 'a skate boarder is doing trick on his skate board.'

# gif_url = 'https://33.media.tumblr.com/0346abaaf2bddc33ee1db0f84f04fb76/tumblr_nqjuk8ssUF1sfdocho1_500.gif'
# gif_caption = 'a man rolls a bowling ball along the lane.'

# gif_url = 'https://38.media.tumblr.com/6b62aa0e56c14d3604c6a205bdbc972e/tumblr_npxwgoNBQD1tht7cgo1_500.gif'
# gif_caption = "a cat pounces on a cat's tail and scares it, which makes all the other cats run away in fear."

# gif_url = 'https://38.media.tumblr.com/7247e34e0545868afaf1b1449c1cf263/tumblr_nfhunfZMiC1rjav34o1_500.gif'
# gif_caption = 'the white man is playing a song with the bass.'


###### CHOOSE NUMBER OF FRAMES ######


num_frames = 5


###### PREPARE GIF AS INPUT ######


gif_path = 'demo_outputs/gif_example.gif'
frames_path = 'demo_outputs/gif_example_frames.png'

# You can open the created files to view the gif and the frames.
# You can delete them after that.

hf.download_gif(gif_url, gif_path)
frame_list = hf.extract_frames(gif_path, num_frames)
hf.save_frames(frame_list, frames_path)


###### RUN INFERENCE ON FRAMES ######


model_dir = 'fromage_model/'
model = models.load_fromage(model_dir)

model_input = frame_list + [ 'Give caption as video.' ]
model_outputs = model.generate_for_images_and_texts(model_input, num_words=15, max_num_rets=0)
model_caption = model_outputs[0]

print('\n')
print(f'model_input = {model_input}\n')
print(f'model_caption = {model_caption}\n')
print(f'gif_caption = {gif_caption}\n')
