import json


#############Just raw code, there may be non-standard variables, useless variables and other phenomena. Thanks!
#########Get a list of questions for the training set (text modality) skills.
#########lecture can be regarded as knowledge

skill2question_id_dict={}

data_path="data/problems_origin.json"
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

        correct_answer=str(ord("A")+answer)+" "+choices[answer]
        if image!=None or split!="train":
            continue
        if skill not in skill2question_id_dict:
            skill2question_id_dict[skill]=[id]
        else:
            temp_list=skill2question_id_dict[skill]
            temp_list.append(id)
            skill2question_id_dict[skill]=temp_list

print(len(skill2question_id_dict))

with open('01skill2id_list_train.json', 'w') as f:
    json.dump(skill2question_id_dict, f, indent=4)
