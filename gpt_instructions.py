import openai
import time

openai.api_key = "your key"
gpt_version = "text-davinci-002"
def prompt_llm(prompt, max_tokens = 64, temperature=0, stop=None):
  response = openai.Completion.create(engine=gpt_version, prompt=prompt, max_tokens=max_tokens, temperature=temperature, stop=stop)
  return response["choices"][0]["text"].strip()

captions = ['electric guitar']		

instructions_dict = {}
max_generated_tokens  = 10
for i,label_class in enumerate(captions):
    prompt_list = [#f'Q: What are useful features for distinguishing a {label_class} in a photo? A: There are several useful visual features to tell there is a {label_class} in a photo:',
                    f'Give more details for the following text: {label_class}']
    instructions_dict[label_class] = []
    for prompt in prompt_list:
       #! maybe increase temperature for more diversity but more diversity may give us more bad words
       ans = prompt_llm(prompt, max_tokens = max_generated_tokens, temperature = 0)
       instructions_dict[label_class].append(ans)
    #! openai doesn't like very frequent requests so maybe we need to use some sleep function after some time
    time.sleep(5) #seconds