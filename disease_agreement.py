import pandas as pd
import os
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
import numpy as np

#################################################
#Title: disease_QA.py
#Author: Delaney Smith
#Inputs: generate QA prompts for disease assosciation tasks
#Output: graphical representation of LLM performance on disease tasks
#################################################

#Set working directory
directory = "/scratch/groups/rbaltman/gossip_corner/sbatch_scripts/validate_disease_copy"

#Evaluate agreement between GPT4.1 LLM as Judge and regex matching

results = []
all_merged = []

files = os.listdir(directory)
gpt_files = [f for f in files if f.endswith("_gpt.csv")]
string_files = [f for f in files if f.endswith("_string.csv")]

model_names = set(f.replace("_gpt.csv", "") for f in gpt_files) & set(f.replace("_string.csv", "") for f in string_files)

for model in model_names:
    gpt_path = os.path.join(directory, f"{model}_gpt.csv")
    string_path = os.path.join(directory, f"{model}_string.csv")
    df_gpt = pd.read_csv(gpt_path)
    df_string = pd.read_csv(string_path)
    merged = pd.merge(df_gpt[['pathway_id', 'gpt_disease_match']], df_string[['pathway_id', 'string_disease_match']], on='pathway_id')
    merged['gpt_disease_match'] = merged['gpt_disease_match'].astype(int)
    merged['string_disease_match'] = merged['string_disease_match'].astype(int)
    f1 = f1_score(merged['string_disease_match'], merged['gpt_disease_match'])
    mcc = matthews_corrcoef(merged['string_disease_match'], merged['gpt_disease_match'])
    num_agree = (merged['gpt_disease_match'] == merged['string_disease_match']).sum()
    percent_agree = 100 * num_agree / len(merged)
    print(model)
    print(f1)
    print(mcc)
    print(percent_agree)
    results.append({
        'model': model,
        'F1': f1,
        'MCC': mcc,
        'n': len(merged)
    })
    all_merged.append(merged)

combined = pd.concat(all_merged, ignore_index=True).dropna()
f1_all = f1_score(combined['string_disease_match'], combined['gpt_disease_match'])
mcc_all = matthews_corrcoef(combined['string_disease_match'], combined['gpt_disease_match'])
num_agree_all = (combined['gpt_disease_match'] == combined['string_disease_match']).sum()
percent_agree_all = 100 * num_agree / len(combined)
accuracy_all = accuracy_score(combined['string_disease_match'], combined['gpt_disease_match'])

print(f1_all)
print(mcc_all)
print(percent_agree_all)
print(accuracy_all)

cm = confusion_matrix(combined['string_yes_match'], combined['gpt_yes_match'])
tn, fp, fn, tp = cm.ravel()
print(f"Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

##############################

#string matching only
model_names = set(f.replace("_string.csv", "") for f in string_files)
results = []
all_merged = []

for model in model_names:
    string_path = os.path.join(directory, f"{model}_string.csv")
    df = pd.read_csv(string_path)
    df['true_label'] = df['answer'].str.strip().str.lower().str.startswith('yes')
    df['predicted_label'] = df.apply(lambda row: row['true_label'] if row['string_yes_match'] else not row['true_label'], axis=1)
    f1 = f1_score(df['true_label'], df['predicted_label'])
    mcc = matthews_corrcoef(df['true_label'], df['predicted_label'])
    accuracy = accuracy_score(df['true_label'], df['predicted_label'])
    tn, fp, fn, tp = confusion_matrix(df['true_label'], df['predicted_label']).ravel()
    print(f"Model: {model}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"MCC: {mcc:.3f}")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}\n")
    results.append({'model': model,'Accuracy': accuracy,'F1': f1, 'MCC': mcc, 'TP': tp, 'TN': tn, 'FP': fp,'FN': fn})
    all_merged.append(df)

combined_df = pd.concat(all_merged, ignore_index=True)

f1_all = f1_score(combined_df['true_label'], combined_df['predicted_label'])
mcc_all = matthews_corrcoef(combined_df['true_label'], combined_df['predicted_label'])
accuracy_all = accuracy_score(combined_df['true_label'], combined_df['predicted_label'])
tn, fp, fn, tp = confusion_matrix(combined_df['true_label'], combined_df['predicted_label']).ravel()

print("=== Combined Metrics Across All Models ===")
print(f"Accuracy: {accuracy_all:.3f}")
print(f"F1 Score: {f1_all:.3f}")
print(f"MCC: {mcc_all:.3f}")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

##############################
#gpt grading only

model_names = set(f.replace("_gpt.csv", "") for f in gpt_files)
results = []
all_merged = []

#Run this loop for y/n eval
for model in model_names:
    gpt_path = os.path.join(directory, f"{model}_gpt.csv")
    df = pd.read_csv(gpt_path)
    df['true_label'] = df['answer'].str.strip().str.lower().str.startswith('yes')
    df['predicted_label'] = df.apply(lambda row: row['true_label'] if row['gpt_yes_match'] else not row['true_label'], axis=1)
    df_yes = df
    f1 = f1_score(df_yes['true_label'], df_yes['predicted_label'])
    mcc = matthews_corrcoef(df_yes['true_label'], df_yes['predicted_label'])
    accuracy = accuracy_score(df_yes['true_label'], df_yes['predicted_label'])
    tn, fp, fn, tp = confusion_matrix(df_yes['true_label'], df_yes['predicted_label']).ravel()
    print(f"Model: {model}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"MCC: {mcc:.3f}")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}\n")
    results.append({'model': model,'Accuracy': accuracy,'F1': f1, 'MCC': mcc, 'TP': tp, 'TN': tn, 'FP': fp,'FN': fn})
    all_merged.append(df_yes)

#Run this loop for disease name eval
for model in model_names:
    gpt_path = os.path.join(directory, f"{model}_gpt.csv")
    df = pd.read_csv(gpt_path)
    df['disease_presence'] = df['answer'].str.strip().str.lower().str.startswith('yes')
    df_yes = df[df['disease_presence']].copy().astype(str) 
    df_yes['true_disease_name'] = df_yes['answer'].str.split(',', n=1, expand=True)[1].str.strip().str.lower()
    df_yes['true_label'] = (df_yes['true_disease_name'] != 'none').astype(bool)
    df_yes['gpt_disease_match'] = df_yes['gpt_disease_match'].map({'True': True, 'False': False})
    df_yes['predicted_label'] = df_yes.apply(compute_predicted_label, axis=1)
    f1 = f1_score(df_yes['true_label'], df_yes['predicted_label'])
    mcc = matthews_corrcoef(df_yes['true_label'], df_yes['predicted_label'])
    accuracy = accuracy_score(df_yes['true_label'], df_yes['predicted_label'])
    tn, fp, fn, tp = confusion_matrix(df_yes['true_label'], df_yes['predicted_label']).ravel()
    print(f"Model: {model}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"MCC: {mcc:.3f}")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}\n")
    results.append({'model': model,'Accuracy': accuracy,'F1': f1, 'MCC': mcc, 'TP': tp, 'TN': tn, 'FP': fp,'FN': fn})
    all_merged.append(df_yes)

combined_df = pd.concat(all_merged, ignore_index=True)

f1_all = f1_score(combined_df['true_label'], combined_df['predicted_label'])
mcc_all = matthews_corrcoef(combined_df['true_label'], combined_df['predicted_label'])
accuracy_all = accuracy_score(combined_df['true_label'], combined_df['predicted_label'])
tn, fp, fn, tp = confusion_matrix(combined_df['true_label'], combined_df['predicted_label']).ravel()

print("=== Combined Metrics Across All Models ===")
print(f"Accuracy: {accuracy_all:.3f}")
print(f"F1 Score: {f1_all:.3f}")
print(f"MCC: {mcc_all:.3f}")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

################
#Bar plot of results

#Set-up call
def setup_plot_style():
    plt.rcParams.update({
        "font.size": 14,            # Base font size
        "axes.titlesize": 16,       # Title font size
        "axes.labelsize": 18,       # Axis label font size
        "xtick.labelsize": 14,      # X-tick label font size
        "ytick.labelsize": 14,      # Y-tick label font size
        "legend.fontsize": 14,      # Legend font size
        "figure.figsize": (8, 6),   # Figure size in inches (width, height)
        "lines.linewidth": 2.0,     # Line width
        "axes.linewidth": 1.2,      # Axis line width
        "xtick.major.size": 8,      # Major tick size
        "ytick.major.size": 8,      # Major tick size
        "xtick.major.width": 1.2,   # Major tick width
        "ytick.major.width": 1.2,   # Major tick width
        "xtick.minor.size": 4,      # Minor tick size
        "ytick.minor.size": 4,      # Minor tick size
        "xtick.minor.width": 0.8,   # Minor tick width
        "ytick.minor.width": 0.8,   # Minor tick width
        "legend.frameon": False,    # Turn off the legend frame
    })

#Organize models
model_label_map = {
    'biogpt_disease': 'BioGPT',
    'biomedlm_disease': 'BioMedLM',
    'claude_disease': 'Claude',
    'deepseek_disease': 'DeepSeek',
    'gemma_disease': 'Gemma',
    'gpt4omini_disease': 'GPT-4o mini',
    'llama_disease': 'Llama',
    'mistral_disease': 'Mistral',
    'qwen_disease': 'Qwen'
}

#Set desired order
desired_order = [
    'biogpt_disease', 'biomedlm_disease', 'claude_disease', 'deepseek_disease',
    'gemma_disease', 'gpt4omini_disease', 'llama_disease', 'mistral_disease', 'qwen_disease'
]

results_map = {r['model']: r for r in results}
ordered_results = [results_map[m] for m in desired_order if m in results_map]

#Set-up data
model_keys = [r['model'] for r in ordered_results]
acc = [r['Accuracy'] for r in ordered_results]
f1_scores = [r['F1'] for r in ordered_results]
mcc = [r['MCC'] for r in ordered_results]
display_labels = [model_label_map.get(m, m) for m in model_keys]

#Make plot
plt.figure(figsize=(5, 4))
x = np.arange(len(model_keys))
colors = plt.cm.tab10.colors[:len(model_keys)]
width = 0.25
bars1 = plt.bar(x - width, acc, width, label='Accuracy', color=colors)
bars2 = plt.bar(x, f1_scores, width, label='F1 Score', color='white', edgecolor=colors, hatch='///')
bars3 = plt.bar(x + width, mcc, width, label='F1 Score', color='white', edgecolor=colors, hatch='....')

#Labels
plt.xlabel('Model')
plt.ylabel('Metrics')
plt.ylim(0, 1)

#Set ticks
plt.xticks(x, display_labels, rotation=45, ha='right')
plt.tick_params(axis='x', which='both', bottom=False)

#Adjust spines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)

#Add legend
legend_elements = [
    Patch(facecolor='black', label='Accuracy'),
    Patch(facecolor='white', edgecolor='black', hatch='//', label='F1 Score'),
    Patch(facecolor='white', edgecolor='black', hatch='...', label='MCC')
]
plt.legend(handles=legend_elements, loc='upper right',  bbox_to_anchor=(1.15, 1.2), fontsize=10)

#Format & Save
plt.tight_layout()
plt.savefig("/scratch/groups/rbaltman/gossip_corner/figs/yn_mcc.png", dpi = 300, bbox_inches='tight')

#Helper function to double check logic for reconstructing the predicted and true labels
def compute_predicted_label(row):
    if row['true_label'] and row['gpt_disease_match']:
        return True
    elif row['true_label'] and not row['gpt_disease_match']:
        return False
    elif not row['true_label'] and not row['gpt_disease_match']:
        return True
    else:  # not true_label and gpt_disease_match is True
        return False
    
    