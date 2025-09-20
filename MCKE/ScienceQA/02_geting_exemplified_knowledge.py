import re
import transformers
import torch

import json
from tqdm import tqdm

import os
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig

######### Knowledge Exemplification Module and Exemplified Knowledge Selection Module
#########For convenience, we directly generate the instantiated knowledge for each instance in the training set, which will be convenient for later calls.

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import random
import numpy as np
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)



data_path="data/problems_origin.json"

model_id = "share_weight/Meta-Llama-3.1-70B-Instruct"
quantization_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True,bnb_4bit_quant_type="nf4")
quantized_model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipeline = transformers.pipeline("text-generation", model=quantized_model, tokenizer=tokenizer,max_new_tokens=1024,pad_token_id=128001)


options=["(A)", "(B)", "(C)", "(D)", "(E)"]
def get_choice_text(choices, options):
    choice_list = []
    for i, c in enumerate(choices):
        choice_list.append("{} {}".format(options[i], c))
    choice_txt = " ".join(choice_list)

    return choice_txt


def get_cot_without_abs_ans(question,context,option,correct_answer,lecture):
    prompt1="You are a teacher and you need to give the rationale for answering the question in response to the information related to the question below.\n"
    prompt2="Question: "+question+"\n"
    prompt3="Context: "+context+"\n"
    prompt4="Option: "+option+"\n"
    prompt5="Correct Answer: "+correct_answer+"\n"
    prompt6="Lecture: "+lecture+"\n"
    prompt7="Identify the plan for answering the question and give the rationale for answering the question (where the rationale should be given point by point). Note please give the full rationale (e.g. don't say based on what is mentioned in the lecture).\n"
    input_text=prompt1+prompt2+prompt3+prompt4+prompt5+prompt6+prompt7
    
    messages = [{"role": "user", "content": input_text}]
    prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    terminators = [pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    outputs = pipeline(prompt, eos_token_id=terminators, do_sample=False)
    new_pred_solution="".join(outputs[0]["generated_text"][len(prompt):])
    new_pred_solution = new_pred_solution.replace("\n", " ")

    return new_pred_solution


def get_cot_with_example_ans(question,context,option,correct_answer,lecture,example):
    prompt1="You are a teacher and you need to give the rationale for answering the question in response to the information related to the question below.\n"
    prompt_example="Example: "+example+"\n"
    prompt2="Question: "+question+"\n"
    prompt3="Context: "+context+"\n"
    prompt4="Option: "+option+"\n"
    prompt5="Correct Answer: "+correct_answer+"\n"
    prompt6="Lecture: "+lecture+"\n"
    prompt7="Identify the plan for answering the question and give the rationale for answering the question (where the rationale should be given point by point). Note please give the full rationale (e.g. don't say based on what is mentioned in the lecture).\n"
    input_text=prompt1+prompt_example+prompt2+prompt3+prompt4+prompt5+prompt6+prompt7
    
    messages = [{"role": "user", "content": input_text}]
    prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    terminators = [pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    outputs = pipeline(prompt, eos_token_id=terminators, do_sample=False)
    new_pred_solution="".join(outputs[0]["generated_text"][len(prompt):])
    new_pred_solution = new_pred_solution.replace("\n", " ")

    return new_pred_solution


def get_cot_with_example(question,context,option,example):
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

    return new_pred_solution


def get_answer_base_on_cot(question,context,option,rationale):
    prompt1="Below there is information related to the question, select the answer.\n"
    prompt2="Question: "+question+"\n"
    prompt3="Context: "+context+"\n"
    prompt4="Option: "+option+"\n"
    prompt5="Rationale: "+rationale+"\n"
    prompt6="Please give only the answer choice (e.g. (F) ) and do not output anything else (e.g. None of the above and (F) good). You must give one of [ (A), (B), (C), (D), (E) ]. The answer choice is:\n"
    input_text=prompt1+prompt2+prompt3+prompt4+prompt5+prompt6
    messages = [{"role": "user", "content": input_text}]
    prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    terminators = [pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    outputs = pipeline(prompt, eos_token_id=terminators, do_sample=False)
    llama_answer="".join(outputs[0]["generated_text"][len(prompt):])

    return llama_answer


#######question info
question_info_dict={}
with open(data_path,"r") as f:
    question_info_dict=json.load(f)



#######STEP1 get preliminary exemplified knowledge for each knowledge. In Knowledge Exemplification Module
skill_path="01skill2id_list_train.json"
skill_example_dict={}
with open(skill_path,"r") as f:
    skill_example_dict=json.load(f)

sample_k=5#the num of samples for preliminary exemplified knowledge
skill2sampled_k_id_list_dict={}
id2example_dict={}

for skill,current_id_list in skill_example_dict.items():
    if len(current_id_list)>=sample_k:
        sampled_k_id_list=random.sample(current_id_list,sample_k)
    else:
        sampled_k_id_list=current_id_list
    
    skill2sampled_k_id_list_dict[skill]=sampled_k_id_list
    for current_id in sampled_k_id_list:
        item=question_info_dict[current_id]
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
        skill=item["skill"]
        lecture=item["lecture"]
        solution=item["solution"]
        split=item["split"]

        correct_answer="("+chr(ord("A")+answer)+") "+choices[answer]
        option=get_choice_text(choices,options)
        context="N/A"
        if hint!="":
            context=hint
        question=question.replace("\n", " ")
        context=context.replace("\n", " ")

        new_pred_solution=get_cot_without_abs_ans(question=question,context=context,option=option,correct_answer=correct_answer,lecture=lecture)

        the_value="Question: "+question+" [SEP] Context: "+context+" [SEP] Options: "+option+" [SEP] Rationale: "+new_pred_solution
        the_value = the_value.replace("\n", " ")
        id2example_dict[current_id]=the_value


#######STEP2 test to get golden exemplified knowledge for each knowledge. In Knowledge Exemplification Module
test_m=10#test example num
skill2max_example_id_dict={}

for skill,current_id_list in skill_example_dict.items():
    selected_id_list=skill2sampled_k_id_list_dict[skill]
    abs_cot_num=len(selected_id_list)
    filtered_current_id_list = [x for x in current_id_list if x not in selected_id_list]
    try:
        now_test_id_list=random.sample(filtered_current_id_list, test_m-abs_cot_num+1)
    except:
        now_test_id_list=filtered_current_id_list
    now_test_id_list.extend(selected_id_list)

    max_correct_num=-1
    max_id=-1
    test_total_num=len(now_test_id_list)-1
    for abs_id in selected_id_list:
        current_example=id2example_dict[abs_id]
        test_correct_num=0
        for test_id in now_test_id_list:
            if test_id==abs_id:
                continue
            item=question_info_dict[test_id]
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
            skill=item["skill"]
            lecture=item["lecture"]
            solution=item["solution"]
            split=item["split"]

            correct_answer="("+chr(ord("A")+answer)+") "+choices[answer]
            option=get_choice_text(choices,options)
            context="N/A"
            if hint!="":
                context=hint
            question=question.replace("\n", " ")
            context=context.replace("\n", " ")
            
            new_pred_solution=get_cot_with_example(question=question,context=context,option=option,example=current_example)
            llama_answer=get_answer_base_on_cot(question=question,context=context,option=option,rationale=new_pred_solution)
            try:
                if ord(llama_answer[1])-ord("A")==answer:
                    llama_is_true=True
                    test_correct_num+=1
                else:
                    llama_is_true=False
            except:
                print(test_id,llama_answer)
        if test_correct_num>max_correct_num:
            max_correct_num=test_correct_num
            max_id=abs_id
    if test_total_num==0:
        skill2max_example_id_dict[skill]={"id":max_id,"acc":-1}
    else:
        skill2max_example_id_dict[skill]={"id":max_id,"acc":max_correct_num/test_total_num}



#######STEP3 generating exemplified knowledge for all training set example (To facilitate the direct call of the question answering module). In Exemplified Knowledge Selection Module.
output_dict={}

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
        skill=item["skill"]
        lecture=item["lecture"]
        solution=item["solution"]
        split=item["split"]

        correct_answer="("+chr(ord("A")+answer)+") "+choices[answer]
        if split!="train" or image!=None:
            continue

        total_num+=1
        llama_is_true="unknown"

        option=get_choice_text(choices,options)
        context="N/A"
        if hint!="":
            context=hint
        question=question.replace("\n", " ")
        context=context.replace("\n", " ")

        try:
            best_cot_id=skill2max_example_id_dict[skill]["id"]
            best_example=id2example_dict[best_cot_id]
        except:
            best_example="N/A"

        new_pred_solution=get_cot_with_example_ans(question=question,context=context,option=option,correct_answer=correct_answer,lecture=lecture,example=best_example)
        llama_answer=get_answer_base_on_cot(question=question,context=context,option=option,rationale=new_pred_solution)

        try:
            if ord(llama_answer[1])-ord("A")==answer:
                llama_is_true=True
                correct_num+=1
            else:
                llama_is_true=False
        except:
            print(id,llama_answer)
        current_item={"question":question,"choices":choices,"answer":answer,"hint":hint,"image":image,"context":context,"task":task,"grade":grade,"subject":subject,"topic":topic,"category":category,"skill":skill,"llm_solution":new_pred_solution,"human_lecture":lecture,"human_solution":solution,"split":split,"llama_is_true":llama_is_true,"selected_example":best_example}
        output_dict[id]=current_item




with open('02train_data_exemplified_knowledge.json', 'w') as f:
    json.dump(output_dict, f, indent=4)
print("02train_data_exemplified_knowledge.json has created!")
