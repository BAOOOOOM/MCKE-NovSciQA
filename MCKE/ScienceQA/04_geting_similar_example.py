from sentence_transformers import SentenceTransformer, util
import numpy as np
import json
import torch


from transformers import pipeline
import re
import transformers
import json
from tqdm import tqdm
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import torch
import re
import random
import numpy as np



###########geting Extracted question stems and similar example. In Exemplified Knowledge Selection Module

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


nlp = spacy.load('en_core_web_sm')


encoder_model = SentenceTransformer('sentence-transformers/sentence-t5-large', device=device)
ner_pipeline = pipeline("ner", model="xlm-roberta-large-finetuned-conll03-english", tokenizer="xlm-roberta-large-finetuned-conll03-english", device=device)

id2skill_dict={}


masked2skill_dict={}
masked2question_dict={}
masked2id_dict={}
id2masked_dict={}
skill_dict={}
id2question={}

def replace_entities(sentence):
    ner_results = ner_pipeline(sentence)

    entities = []
    entity_text = ""
    entity_type = ""

    for entity in ner_results:
        if entity['word'].startswith('▁'): 
            if entity_text:  
                entities.append((entity_text, entity_type))
            entity_text = entity['word'].replace('▁', '') 
            entity_type = entity['entity'].replace("B-", "").replace("I-", "")
        else:
            entity_text += entity['word']

    if entity_text:
        entities.append((entity_text, entity_type))

    result = sentence
    for entity_text, entity_type in entities:
        result = result.replace(entity_text, f'[{entity_type}]')

    pattern = r'(\[\w{3,4}\])(?:\s+\1)+'
    result = re.sub(pattern, r'\1', result)
    return result

def replace_tokens(sentence):
    sentence = replace_entities(sentence)
    doc = nlp(sentence)
    result = []
    
    for token in doc:
        if token.ent_type_ == 'PERSON':
            if not result or result[-1] != '[PER]':
                result.append('[PER]')
        elif token.ent_type_ == 'GPE':
            if not result or result[-1] != '[LOC]':
                result.append('[LOC]')
        elif token.pos_ == 'NOUN':
            if not result or result[-1] != '[NOUN]':
                result.append('[NOUN]')
        elif token.pos_ == 'NUM':
            if not result or result[-1] != '[NUM]':
                result.append('[NUM]')
        else:
            result.append(token.text)

    all_result=' '.join(result)
    pattern = r'\[\s*([A-Z]{3,4})\s*\]'
    all_result = re.sub(pattern, r'[\1]', all_result)
    all_result=all_result.replace('[PER]', '[UNK]')
    all_result=all_result.replace('[LOC]', '[UNK]')
    all_result=all_result.replace('[MISC]', '[UNK]')
    
    return all_result

options=["(A)", "(B)", "(C)", "(D)", "(E)"]
def get_choice_text(choices, options):
    choice_list = []
    for i, c in enumerate(choices):
        choice_list.append("{} {}".format(options[i], c))
    choice_txt = " ".join(choice_list)

    return choice_txt


data_path="03_identify_main_question_text.json"
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
        main_question=item["main_question"]
        
        question = question.replace("\n", " ")

        if image!=None:
            continue

        total_num+=1
        old_rationale=lecture+" "+solution

        llama_is_true="unknown"

        option=get_choice_text(choices,options)
        context="N/A"
        if old_rationale=="":
            old_rationale="N/A"
        if hint!="":
            context=hint

        id2question[id]=question+" "+option
        masked_question=replace_tokens(main_question)
        id2masked_dict[id]=masked_question

        if masked_question not in masked2skill_dict:
            masked2skill_dict[masked_question]=[skill]
        else:
            pre_skill_list=masked2skill_dict[masked_question]
            if skill not in pre_skill_list:
                pre_skill_list.append(skill)
                masked2skill_dict[masked_question]=pre_skill_list
        
        if masked_question not in masked2question_dict:
            masked2question_dict[masked_question]=[question]
            masked2id_dict[masked_question]=[id]
        else:
            pre_question_list=masked2question_dict[masked_question]
            if question not in pre_question_list:
                pre_question_list.append(question)
                masked2question_dict[masked_question]=pre_question_list
            pre_id_list=masked2id_dict[masked_question]
            pre_id_list.append(id)
            masked2id_dict[masked_question]=pre_id_list

        if skill not in skill_dict:
            skill_dict[skill]=[masked_question]
        else:
            pre_masked_question_list=skill_dict[skill]
            if masked_question not in pre_masked_question_list:
                pre_masked_question_list.append(masked_question)
                skill_dict[skill]=pre_masked_question_list


def find_top_k_similar_sentences(sentences, k=30):
    embeddings = encoder_model.encode(sentences, convert_to_tensor=True)
    cosine_similarities = util.pytorch_cos_sim(embeddings, embeddings)

    similar_sentences_dict = {}
    for i, sentence in enumerate(sentences):
        top_k_indices = cosine_similarities[i].topk(k+1)[1].cpu().numpy()
        top_k_indices = [idx for idx in top_k_indices if idx != i][:k]
        similar_sentences_dict[sentence] = [sentences[idx] for idx in top_k_indices]
    return similar_sentences_dict


sentences = list(masked2id_dict.keys())
similar_masked_sentences_dict = find_top_k_similar_sentences(sentences, k=30)



def find_top_k_similar(query: str, sentence_list: list, K: int = 30):
    query_embedding = encoder_model.encode(query, convert_to_tensor=True)
    sentence_embeddings = encoder_model.encode(sentence_list, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, sentence_embeddings)[0]
    top_k_indices = torch.topk(similarities, K).indices
    
    top_k_sentences = [sentence_list[idx] for idx in top_k_indices]
    result = top_k_sentences
    return result


output_query_dict={}
output_id_dict={}
output_list=[]

id2value_dict={}
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
        main_question=item["main_question"]
        
        question = question.replace("\n", " ")

        if image!=None:
            continue
        option=get_choice_text(choices,options)
        the_value=question+" "+option
        id2value_dict[id]=the_value



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
        main_question=item["main_question"]
        
        question = question.replace("\n", " ")

        if image!=None:
            continue
        option=get_choice_text(choices,options)
        the_query=question+" "+option

        ##############Here, we directly sampled 30 examples and then only selected the top-1 examples from the training set (in 05 part).
        mask_question=id2masked_dict[id]
        other_id_list=[]
        the_id_list=masked2id_dict[mask_question]
        for temp_id in the_id_list:
            if temp_id!=id:
                other_id_list.append(temp_id)
        other_value_list=[]
        current_top_k_question=[]
        ##############The question stem is used as a constraint condition to retrieve similar reasoning structure examples
        if len(other_id_list)>=30:
            for temp_id in other_id_list:
                other_value_list.append(id2value_dict[temp_id])
            current_top_k_question=find_top_k_similar(the_query,other_value_list,30)
        else:
            if len(other_id_list)!=0:
                for temp_id in other_id_list:
                    other_value_list.append(id2value_dict[temp_id])
                current_top_k_question=find_top_k_similar(the_query,other_value_list,len(other_id_list))

            other_mask_id_list=[]
            other_value_list=[]
            
            similar_mask_question_list=similar_masked_sentences_dict[mask_question]
            for similar_mask_question in similar_mask_question_list:
                the_id_list=masked2id_dict[similar_mask_question]
                for temp_id in the_id_list:
                    other_value_list.append(id2value_dict[temp_id])
                if len(the_id_list)>(30-len(current_top_k_question)):
                    current_top_k_question_with_other=find_top_k_similar(the_query,other_value_list,30-len(current_top_k_question))
                else:
                    current_top_k_question_with_other=find_top_k_similar(the_query,other_value_list,len(the_id_list))
                current_top_k_question.extend(current_top_k_question_with_other)
                other_value_list=[]
        if len(current_top_k_question)!=30:
            print(id)
            print(current_top_k_question)
            raise ValueError
        output_query_dict[the_query]=current_top_k_question
        output_id_dict[id]=current_top_k_question


with open('04query_question_stem_topk.json', 'w') as f:
    json.dump(output_query_dict, f, indent=4)
