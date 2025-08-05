# PathwayQA
Benchmark QA dataset generation from Reactome and Evaluation with 9 LLMs <br>
Original Reactome data found on Zenodo <br>

Graphical overview of prompt generation: 
<img width="4400" height="1402" alt="overview_fig" src="https://github.com/user-attachments/assets/0f5bc903-b0d7-4a65-96f6-d148aab703f3" />

## Install dependencies

We recommend installing all dependencies in a conda environment:

```conda env create --file environment.yml```

## Download data

* Fully parsed reaction and disease data from Reactome can be found on Zenodo [link]
* Prompt and answers for the reaction and disease tasks can be found in the `/data` folder

## Run Models

The LLM models must first be downloaded from HuggingFace. The `vllm` python package is required to run the script found in `/run_models`. 

## Evaluation

The evaluation scripts are run on the output files of the LLM models in order to judge whether the generated answer matches the true answer. 
* `compare_answers_gpt.py` queries GPT 4.1 to test if the generated and true answers match. For every entity in the true answer, the model determines if it is in the generated output. `postprocess_validate.py` converts the output into a score per reaction. 
* `disease_agreement.py` performs the validation for the disease queries using the LLM. `string_match.py` performs a simple string match. 


