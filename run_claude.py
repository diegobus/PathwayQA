import anthropic
import os
import re
import json
import pandas as pd
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

#################################################
#Title: run_llama.py
#Author: Delaney Smith
#Inputs: prompt & answer text file
#Output: CSV of LLM generated text responses to queries
#################################################

#Set API key
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

#load prompts
df = pd.read_csv("/scratch/groups/rbaltman/gossip_corner/prompts_answers_complex_oneExample.csv")

#for testing
#df = df.sample(n=50, random_state=42).copy()

#Anthropic example message
message = client.messages.create(
    model="claude-3-5-haiku-20241022",
    max_tokens=1024,
    temperature=0,
    system="You are a world-class poet. Respond only with short poems.",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Why is the ocean salty?"
                }
            ]
        }
    ]
)

print(message.content)

#No Batch testing
def generate_claude(prompt):
    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=1024,
        temperature=0.0,
        top_p=1.0,
        system = "You give very concise answers. You only generate lists of products and nothing else. You do not repeat enzymes or reactants.",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.content[0].text 

df["generated"] = df["prompt"].apply(generate_claude)

print(df["generated"])

#Batching
batch_requests = []

for _, row in df.iterrows():
    batch_requests.append(
        Request(
            custom_id=row["pathway_id"],
            params=MessageCreateParamsNonStreaming(
                model="claude-3-5-haiku-20241022",
                max_tokens=1024,
                temperature=0.0,
                top_p=1.0,
                system = "You give very concise answers. You do not repeat the prompt.",
                messages=[
                    {"role": "user", "content": row["prompt"]}
                ]
            )
        )
    )

#Create the batch job
batch_response = client.messages.batches.create(requests=batch_requests)

#Checking the status
batch_id = batch_response.id
#Test 2: batch_id = 'msgbatch_01WrVQHUuS7WZQPAg9JqrX2v' (done)
#One Shot: batch_id = 'msgbatch_01VXwaPXLGbpky8rfMk49d7P' (done)
#Zero Shot: batch_id = 'msgbatch_01SbXoXdWV8zcKPa8EAP31jg' (done)
#Full example: batch_id = 'msgbatch_014MwD9o4JD7J4p5NACoxSs5' (done) processed
#Disease: batch_id = 'msgbatch_014MwD9o4JD7J4p5NACoxSs5' -- didn't work
#One shot take two: batch_id = 'msgbatch_01Ena3vFgUDNZoYRbMNJbf1B' (done)
#Disease take two: batch_id = 'msgbatch_015F44nFfpy995vjALZ2b54a' (done)

batch_status = client.messages.batches.retrieve(batch_id)
print(batch_status)

#Accessing results
generated_map = {}

for result in client.messages.batches.results(batch_id):
    if result.result.type == "succeeded":
        # The text is inside: result.result.message.content[0].text
        text = ""
        for block in result.result.message.content:
            if block.type == "text":
                text += block.text
        generated_map[result.custom_id] = text

# Map to DataFrame by reaction_id
df["generated"] = df["reaction_id"].map(generated_map)

#Save df
df.to_csv("/scratch/groups/rbaltman/gossip_corner/claude_out_one.csv", index=False)

#for cleaning outputs to remove prompt from generated text
def after_last_products(text):
    if pd.isna(text):
        return None
    idx = text.rfind("Products:")
    if idx != -1:
        return text[idx + len("Products: "):].strip()
    else:
        return None
    
df["generated_cleaned"] = df["generated"].apply(after_last_products)