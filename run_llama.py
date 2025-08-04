import pandas as pd
import argparse
from vllm import LLM, SamplingParams

#################################################
#Title: run_llama.py
#Author: Delaney Smith
#Inputs: prompt & answer text file, local directory to model
#Output: CSV of LLM generated text responses to queries
#################################################

#Load prompts
df = pd.read_csv("/scratch/groups/rbaltman/gossip_corner/prompts_answers_complex_oneExample.csv")
#test = df.sample(50, random_state=42) #for small batch testing

#Load model
llm = LLM(model="/scratch/groups/rbaltman/gossip_corner/llama3.1-8b-instruct/", dtype="float16")

#Run model
sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=128)
outputs = llm.generate(test["prompt"].tolist(), sampling_params)
df["generated"] = [elem.outputs[0].text for elem in outputs]

df.to_csv("/scratch/groups/rbaltman/gossip_corner/llama_out.csv", index=False)