import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

#################################################
#Title: run_biogpt.py
#Author: Delaney Smith
#Inputs: prompt & answer text file, local directory to model
#Output: CSV of LLM generated text responses to queries
#################################################

#Path to local model
model_directory = "/scratch/groups/rbaltman/gossip_corner/microsoft/biogpt"

#Load the tokenizer and model from the local directory
tokenizer = AutoTokenizer.from_pretrained(model_directory)
model = AutoModelForCausalLM.from_pretrained(model_directory).to("cuda")

#Load prompts
df = pd.read_csv("/scratch/groups/rbaltman/gossip_corner/prompts_answers_complex_oneExample.csv")
#test = df.sample(50, random_state=42) #for small-batch testing

#Run pipeline (skipped 753 due to length)
generated_outputs = []
for _, row in df.iterrows():
    prompt = row["prompt"]
    answer = row["answer"]
    prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids
    answer_tokens = tokenizer(answer, return_tensors="pt").input_ids
    if prompt_tokens.shape[1] + answer_tokens.shape[1] > max_length:
        generated_outputs.append("prompt too long")
        continue
    input_ids = prompt_tokens.to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=1024,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    out = generated_text.rsplit("Products:", 1)[1].strip()
    print(out)
    generated_outputs.append(generated_text)

#Save to df and to csv
df["generated"] = generated_outputs
df.to_csv("/scratch/groups/rbaltman/gossip_corner/microsoft/biogpt/biogpt_out.csv", index=False)

output = pd.read_csv("/scratch/groups/rbaltman/gossip_corner/llama_out.csv")