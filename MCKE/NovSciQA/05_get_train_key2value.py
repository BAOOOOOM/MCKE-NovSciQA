import re
import transformers
import torch


import json
from tqdm import tqdm

########geting for key/value pairs for Question Answering Module

output_dict={}
output_list=[]

key2value_dict={}

options=["(A)", "(B)", "(C)", "(D)", "(E)"]
def get_choice_text(choices, options):
    choice_list = []
    for i, c in enumerate(choices):
        choice_list.append("{} {}".format(options[i], c))
    choice_txt = " ".join(choice_list)

    return choice_txt


data_path="02train_data_exemplified_knowledge.json"
total_num=0
correct_num=0

with open(data_path,"r") as f:
    data=json.load(f)
    for id in data:
        item=data[id]
        skill=item["human_skill"]
        question=item["question"]
        subject=item["subject"]
        category=item["category"]
        lecture=item["human_lecture"]
        choices=item["choices"]
        answer=item["answer"]
        solution=item["human_solution"]
        split=item["split"]

    
        llm_solution=item["pred_solution"]
        llama_is_true=item["llama_is_true"]


        if llama_is_true==False:
            continue

        option=get_choice_text(choices,options)
        question=question.replace("\n", " ")

        the_key=question+" "+option
        rationale="N/A"
        if llm_solution!="":
            rationale=llm_solution
        the_value="Question: "+question+" [SEP] Options: "+option+" [SEP] Rationale: "+rationale
        the_value = the_value.replace("\n", " ")

        key2value_dict[the_key]=the_value



with open('05question_key2value.json', 'w') as f:
    json.dump(key2value_dict, f, indent=4)