import argparse
import os
import pandas as pd
import json
import re

#################################################
#Title: string_match.py
#Author: Delaney Smith
#Inputs: evaluate LLM generated responses over disease tasks using regular expressions
#Output: CSV with evaluation statements of prompts
#################################################

#load LLM outputs
df = pd.read_csv("/scratch/groups/rbaltman/gossip_corner/sbatch_scripts/validate_disease_copy/biogpt_disease_gpt.csv") 

#Parse text for y/n and disease name fields
def parse_pair(text: str):
    text = re.split(r"\bPathway_id\b", text, flags=re.IGNORECASE)[0].strip()
    parts = [p.strip() for p in text.split(',', 1)]
    if len(parts) != 2:
        yes_str = parts[0].lower().strip()
        disease = False
    else:
        yes_str = parts[0].lower().strip()
        disease = parts[1] if parts[1].lower() != 'none' else None
    if yes_str.startswith('yes'):
        yes = True
    elif yes_str.startswith('no'):
        yes = False
    else:
        yes = None
    return yes, disease

#string-match (row-wise)
def strict_matches(row):
    yes_a, dis_a = parse_pair(str(row['answer']))
    yes_g, dis_g = parse_pair(str(row['generated']))
    yes_match = (yes_a == yes_g)
    if dis_a is None and dis_g is None:
        disease_match = True
    elif dis_a and dis_g:
        disease_match = dis_a.lower() in dis_g.lower() or dis_g.lower() in dis_a.lower()
    else:
        disease_match = False
    return pd.Series({'string_yes_match': yes_match, 'string_disease_match': disease_match})

df[['string_yes_match','string_disease_match']] = df.apply(strict_matches, axis=1)

#Save output
cols = ['prompt', 'answer', 'generated', 'gpt_yes_match', 'gpt_disease_match']
if 'pathway_id' in df.columns:
    cols.insert(0, 'pathway_id')
df[cols].to_csv("/scratch/groups/rbaltman/gossip_corner/sbatch_scripts/validate_disease_copy/biomedlm_disease_gpt.csv", index=False)