import json

skill2question_id_dict={}

data_path="data/NovSciQA.json"
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

        if split!="train":
            continue

        correct_answer="("+chr(ord("A")+answer)+") "+choices[answer]
        llama_is_true="unknown"

        question = question.replace("\n", " ")
        
        if skill not in skill2question_id_dict:
            skill2question_id_dict[skill]=[id]
        else:
            temp_list=skill2question_id_dict[skill]
            temp_list.append(id)
            skill2question_id_dict[skill]=temp_list

print(len(skill2question_id_dict))

with open('01skill2id_list_train.json', 'w') as f:
    json.dump(skill2question_id_dict, f, indent=4)
