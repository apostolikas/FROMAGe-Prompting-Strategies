import openai

openai_api_key = "your key"
gpt_version = ""
def prompt_llm(prompt, max_tokens=64, temperature=0, stop=None):
  response = openai.Completion.create(engine=gpt_version, prompt=prompt, max_tokens=max_tokens, temperature=temperature, stop=stop)
  return response["choices"][0]["text"].strip()

classes=['electric guitar',			
'golden retriever',
'malamute',
'mixing bowl',
'cuirass',
'dalmatian',
'african hunting dog',
'lion',
'crate',
'bookshop',
'vase',
'nematode',
'hourglass',
'ant',
'king crab',
'black-footed ferret',
'scoreboard',
'theater curtain',
'school bus',
'trifle']

instructions_dict = {}
for label_class in classes:
    prompt_list = [f'Q: What are useful features for distinguishing a {label_class} in a photo?\
                    A: There are several useful visual features to tell there is a {label_class} in a photo:',
                    f'What is a {label_class}']
    instructions_dict[label_class] =[]
    for prompt in prompt_list:
       #! maybe increase temperature for more diversity
       ans = prompt_llm(prompt,max_tokens=64, temperature=0)
       instructions_dict[label_class].append(ans)