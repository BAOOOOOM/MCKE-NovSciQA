import re
import transformers
import torch

import json
from tqdm import tqdm


from typing import List

import os
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig

#################Question Answering Module


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


import random
import numpy as np
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


model_id = "share_weight/Meta-Llama-3.1-70B-Instruct"
quantization_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True,bnb_4bit_quant_type="nf4")
quantized_model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipeline = transformers.pipeline("text-generation", model=quantized_model, tokenizer=tokenizer,max_new_tokens=1024,pad_token_id=128001)

from sentence_transformers import SentenceTransformer, util
encode_model = SentenceTransformer('sentence-transformers/sentence-t5-large', device='cuda:0')

cosine_sim_dict={}
with open("04query_question_stem_topk.json","r") as f:
    data=json.load(f)
    cosine_sim_dict=data


key2value_list_datapath="05question_key2value.json"
key2value_dict={}

with open(key2value_list_datapath,"r") as f:
    key2value_dict=json.load(f)

train_key_list=list(key2value_dict.keys())
train_key_list_embeddings = encode_model.encode(train_key_list, convert_to_tensor=True)



output_dict={}
output_list=[]


options=["(A)", "(B)", "(C)", "(D)", "(E)"]
def get_choice_text(choices, options):
    choice_list = []
    for i, c in enumerate(choices):
        choice_list.append("{} {}".format(options[i], c))
    choice_txt = " ".join(choice_list)

    return choice_txt


data_path="data/problems_origin.json"
total_num=0
correct_num=0

with open(data_path,"r") as f:
    data=json.load(f)
    for id in data:
        item=data[id]
        question=item["question"]
        choices=item["choices"]
        answer=item["answer"]
        hint=item["hint"]
        image=item["image"]
        task=item["task"]
        grade=item["grade"]
        subject=item["subject"]
        topic=item["topic"]
        category=item["category"]
        human_skill=item["skill"]
        lecture=item["lecture"]
        solution=item["solution"]
        split=item["split"]

        correct_answer="("+chr(ord("A")+answer)+") "+choices[answer]
        if image!=None or split=="train":
            continue

        total_num+=1
        llama_is_true="unknown"

        option=get_choice_text(choices,options)
        context="N/A"
        if hint!="":
            context=hint
        question = question.replace("\n", " ")
        context=context.replace("\n", " ")

        the_query=question+" "+option
        try:
            top_k_example = cosine_sim_dict[the_query]
            example="N/A"
            for i in range(30):
                #Select similar examples from the training set, while non-training set examples will not be selected as they are not included in this dictionary.
                if top_k_example[0].replace("\n", " ")!=the_query.replace("\n", " ") and (top_k_example[i] in key2value_dict):
                    example=key2value_dict[top_k_example[i]]
                    break
            if example=="N/A":
                the_query_embedding = encode_model.encode(the_query, convert_to_tensor=True)
                current_cosine_scores = util.pytorch_cos_sim(the_query_embedding, train_key_list_embeddings)
                current_top_results = torch.topk(current_cosine_scores, k=3)
                current_top_sentences = [train_key_list[idx] for idx in current_top_results[1][0]]
                for example_key in current_top_sentences:
                    if example_key.replace("\n", " ")!=the_query.replace("\n", " ") and (example_key in key2value_dict):
                        example=key2value_dict[example_key]
                        break
        except:
            raise ValueError

        prompt1="An example of a correct rationale for a similar question will first be given below, after which a question and its associated information will be given. You need to give the rationale for answering the question in response to the information related to the question below.\n"
        prompt_example="Example: "+example+"\n"
        prompt2="Question: "+question+"\n"
        prompt3="Context: "+context+"\n"
        prompt4="Options: "+option+"\n"
        prompt5="Identify the plan for answering the question and then give the rationale for answering the question, please give the rationale directly and do not output anything else.\n"
        input_text=prompt1+prompt_example+prompt2+prompt3+prompt4+prompt5
        messages = [{"role": "user", "content": input_text}]
        prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        terminators = [pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        outputs = pipeline(prompt, eos_token_id=terminators, do_sample=False)
        new_pred_solution="".join(outputs[0]["generated_text"][len(prompt):])
        new_pred_solution = new_pred_solution.replace("\n", " ")

        prompt1="Below there is information related to the question, select the answer.\n"
        prompt2="Question: "+question+"\n"
        prompt3="Context: "+context+"\n"
        prompt4="Options: "+option+"\n"
        prompt5="Rationale: "+new_pred_solution+"\n"
        prompt6="Please give only the answer choice (e.g. (F) ) and do not output anything else (e.g. None of the above and (F) good). You must give one of [ (A), (B), (C), (D), (E) ]. The answer choice is:\n"
        input_text=prompt1+prompt2+prompt3+prompt4+prompt5+prompt6
        messages = [{"role": "user", "content": input_text}]
        prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        terminators = [pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        outputs = pipeline(prompt, eos_token_id=terminators, do_sample=False)
        llama_answer="".join(outputs[0]["generated_text"][len(prompt):])
        try:
            if ord(llama_answer[1])-ord("A")==answer:
                llama_is_true=True
                correct_num+=1
            else:
                llama_is_true=False
        except:
            print(id,llama_answer)
        current_item={"question":question,"choices":choices,"answer":answer,"hint":hint,"image":image,"context":context,"task":task,"grade":grade,"subject":subject,"topic":topic,"category":category,"human_skill":human_skill,"llm_predict_solution":new_pred_solution,"human_lecture":lecture,"human_solution":solution,"example":example,"split":split,"llama_is_true":llama_is_true}
        output_dict[id]=current_item



with open('06predict_val_test.json', 'w') as f:
    json.dump(output_dict, f, indent=4)

print("06predict_val_test has created!")
print("val and test accuracy:",correct_num/total_num)

