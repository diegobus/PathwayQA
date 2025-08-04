import re
import pandas as pd

#################################################
#Title: disease_QA.py
#Author: Delaney Smith
#Inputs: generate QA prompts for disease assosciation tasks
#Output: CSV questions and answers
#################################################

disease = pd.read_csv("/scratch/groups/rbaltman/gossip_corner/sbatch_scripts/validate_disease_copy/disease.csv") #load disease information csv
pathway = pd.read_csv("/scratch/groups/rbaltman/gossip_corner/sbatch_scripts/validate_disease_copy/pathways.csv") #load pathways information csv

pathway_dict = {pathway["pathway_id"]: pathway for pathway in disease}

def form_query(instruct, p):
    query = f"{instruct}\n"
    pathway_ids = p["pathway_id"]
    if isinstance(pathway_ids, str):
        pathway_ids = [pathway_ids]
    pathway_types = p["pathway_type"]
    if isinstance(pathway_types, str):
        pathway_types = [pathway_types]
    for pid in pathway_ids:
        query += f"Pathway_id: {pid}\n"
    for ptype in pathway_types:
        query += f"Pathway_type: {ptype}\n"
    query += "Answer:"
    return query


def form_answer(p, disease_info):
    disease_info = pathway_dict.get(p["pathway_id"], {}).get("disease_relationship")
    if disease_info == "None":
        is_disease = "no"
        disease_name = "None"
    elif disease_info == "disease":
        is_disease = "yes"
        disease_name = "None"
    else:
        is_disease = "yes"
        disease_name = disease_info
    return f"{is_disease}, {disease_name}"

def get_prompts(p, disease_info):
    instruct = (
    "You are an expert on chemistry and biology. Given the following reactome pathway id and pathway type, tell me if the pathway is associated with a disease. Do not repeat the question. Provide only the answer in the following format: yes/no, <disease_name>. If there is no disease association, write 'no, None'. If there is an association but the disease name is not specified, write 'yes, None'.\n"
    "Pathway_id: R-HSA-164843\n"
    "Pathway_type: 2-LTR circle formation\n"
    "Answer: yes, Human immunodeficiency virus infectious disease\n\n"
    )
    prompts = [form_query(instruct, elem) for elem in p]
    answers = [form_answer(elem, disease_info) for elem in p]
    return prompts, answers

all = get_prompts(pathway, pathway_dict)