import re
import transformers
import torch


import json
from tqdm import tqdm

import os
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig

import random
import numpy as np
###################Main Question Extraction  Agent (QSE agent). In Exemplified Knowledge Selection Module


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

model_id = "share_weight/Meta-Llama-3.1-70B-Instruct"
quantization_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True,bnb_4bit_quant_type="nf4")
quantized_model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipeline = transformers.pipeline("text-generation", model=quantized_model, tokenizer=tokenizer,max_new_tokens=1024,pad_token_id=128001)

output_dict={}


options=["(A)", "(B)", "(C)", "(D)", "(E)"]
def get_choice_text(choices, options):
    choice_list = []
    for i, c in enumerate(choices):
        choice_list.append("{} {}".format(options[i], c))
    choice_txt = " ".join(choice_list)

    return choice_txt

def get_main_quesiton(test_question):
    prompt1='''Given an input sentence, identify and retain the main question or instruction, and replace any preceding context or background information with [Background]. Do not prepend [Background] to the main question or instruction itself. Here are some examples:

    1. Input: "Is the following an example of the chair effect? Rosita stumbles upon her long lost photo and is so amused that she decides to take up wilderness rock climbing, even though she doesn't have the appropriate training."
    Output: "Is the following an example of the chair effect? [Background]"

    2. Input: "Please select the most appropriate microorganism that applies to the scenario below. Breaks down toxic substances on the surface of ceramic mugs."
    Output: "Please select the most appropriate microorganism that applies to the scenario below. [Background]"

    3. Input: "Which is the Hacker way of greeting in his letter?"
    Output: "Which is the Hacker way of greeting in his letter?"

    4. Input: "Give the stabilized substance of the following substance on Jimmy's planet after a chemical reaction. Ca reacts with Cl3."
    Output: "Give the stabilized substance of the following substance on Jimmy's planet after a chemical reaction. [Background]"

    5. Input: "An object it has a mass of 7KG, a speed of 10m/s and a temperature of 3°C. If the volume of the object is reduced from 99m3 to 41m3, how will the Kettle Force on it change?"
    Output: "[Background] If the volume of the object is reduced from 99m3 to 41m3, how will the Kettle Force on it change?"

    Note that if there is no background, the original question is printed directly. Please give the result directly and do not output anything else (Here is the output:). Now, process the following input:

    '''

    prompt2=test_question
    input_text=prompt1+prompt2
    messages = [{"role": "user", "content": input_text}]
    prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    terminators = [pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    outputs = pipeline(prompt, eos_token_id=terminators, do_sample=False)

    main_question="".join(outputs[0]["generated_text"][len(prompt):])
    return main_question


data_path="data/NovSciQA.json"
total_num=0
correct_num=0

with open(data_path,"r") as f:
    data=json.load(f)
    for id in data:
        item=data[id]
        skill=item["skill"]
        question=item["question"]
        subject=item["subject"]
        category=item["category"]
        lecture=item["knowledge"]
        choices=item["choice"]
        answer=item["answer"]
        solution=item["solution"]
        split=item["split"]

        correct_answer="("+chr(ord("A")+answer)+") "+choices[answer]
        llama_is_true="unknown"


        main_question=get_main_quesiton(question)
        main_question=main_question.replace("[Background]","")
        main_question=main_question.strip()
        current_item={"question":question,"choices":choices,"answer":answer,"subject":subject,"category":category,"human_skill":skill,"main_question":main_question,"human_lecture":lecture,"human_solution":solution,"split":split}
        output_dict[id]=current_item

with open('03_identify_main_question_text.json', 'w') as f:
    json.dump(output_dict, f, indent=4)
print("03_identify_main_question_text has created!")