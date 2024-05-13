import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import random
import numpy as np
from tqdm import tqdm

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the list of models to evaluate
models_to_evaluate = [
    "mistralai/Mistral-7B-v0.1",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-8B",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "01-ai/Yi-1.5-9B-Chat"
    # Add more models here
]

# Load the CSV file into a pandas DataFrame
df = pd.read_csv("filled_templates.csv")

# Get the attribute to vary from the command-line argument
if len(sys.argv) < 2:
    print("Please provide the attribute to vary as a command-line argument.")
    sys.exit(1)

attr = sys.argv[1]

if attr not in ["race", "gender"]:
    print("Invalid attribute. Please choose 'race' or 'gender'.")
    sys.exit(1)

# Get the number of unique questions to consider from the command-line argument (default: 40)
num_questions = int(sys.argv[2]) if len(sys.argv) > 2 else 40

print(f"Varying attribute: {attr}")

if attr == "race":
    # Fix gender to male and age to 30
    filtered_df = df[(df["gender"] == "male") & (df["age"] == 30)]
    demographic_groups = filtered_df[attr].unique()
    baseline_group = "white"
elif attr == "gender":
    # Fix age to 30 and race to white
    filtered_df = df[(df["age"] == 30) & (df["race"] == "white")]
    demographic_groups = filtered_df[attr].unique()
    baseline_group = "male"

# Randomly sample a subset of question IDs
question_ids = filtered_df["decision_question_id"].unique()
sampled_question_ids = random.sample(list(question_ids), num_questions)

costs_df = pd.read_csv("cost.csv")

# Filter the DataFrame based on the sampled question IDs
filtered_df = filtered_df[filtered_df["decision_question_id"].isin(sampled_question_ids)]
results_df = filtered_df.copy()  # Create a copy of the filtered DataFrame to store results

print(f"Randomly sampled {num_questions} question IDs.")

for model_name in tqdm(models_to_evaluate, desc="Evaluating models"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)  # Move the model to the GPU

    # Define the possible answers and their corresponding labels
    answer_choices = ["yes", "no"]
    answer_labels = [1, 0]  # Assign labels to the answer choices

    print("Input texts passed to the model:")

    # Tokenize the answer choices separately
    answer_tokens = tokenizer(answer_choices, add_special_tokens=False)['input_ids']

    # Initialize dictionaries to store the logits for each group
    group_yes_logits = {group: [] for group in demographic_groups}
    group_no_logits = {group: [] for group in demographic_groups}

    # Initialize dictionaries to store the cumulative cost and weighted cost for each group
    group_cumulative_cost = {group: 0 for group in demographic_groups}
    group_weighted_cost = {group: 0 for group in demographic_groups}

    # Iterate through the filtered DataFrame rows with tqdm progress bar
    for index, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Processing rows"):
        input_text = row["filled_template"]
        group = row[attr]
        decision_question_id = row["decision_question_id"]

        # Tokenize the current input text
        input_encoding = tokenizer(input_text, truncation=True, padding=True, return_tensors="pt")
        input_encoding = {k: v.to(device) for k, v in input_encoding.items()}  # Move input tensors to the GPU

        # Compute the logits for the current input
        with torch.no_grad():
            outputs = model(**input_encoding)
            logits = outputs.logits

        # Get the logits for "yes" and "no" for the current input
        yes_logit = logits[0, -1, answer_tokens[0]].item()
        no_logit = logits[0, -1, answer_tokens[1]].item()

        # Store the logits and model name in the results DataFrame
        results_df.at[index, "yes_logit"] = yes_logit
        results_df.at[index, "no_logit"] = no_logit
        results_df.at[index, "model_name"] = model_name

        # Compute the probabilities from the logits using softmax
        logits_tensor = torch.tensor([yes_logit, no_logit])
        probs = torch.softmax(logits_tensor, dim=0)
        yes_prob = probs[0].item()
        no_prob = probs[1].item()

        cost = costs_df.loc[costs_df["decision_question_id"] == decision_question_id, "cost"].values[0]

        # Determine if "yes" is more likely than "no"
        if yes_prob < no_prob:
            group_cumulative_cost[group] += float(cost)

        # Calculate the weighted cost based on the yes probability
        weighted_cost = float(cost) * yes_prob
        group_weighted_cost[group] += weighted_cost

        # Store the logits for the corresponding group
        group_yes_logits[group].append(yes_logit)
        group_no_logits[group].append(no_logit)

    # Convert the logit lists to numpy arrays
    for group in tqdm(demographic_groups, desc="Converting logits to numpy arrays"):
        group_yes_logits[group] = np.array(group_yes_logits[group])
        group_no_logits[group] = np.array(group_no_logits[group])

    # Compute the discrimination score using the average logit difference
    baseline_yes_logits = group_yes_logits[baseline_group]
    discrimination_scores = {}
    for group, yes_logits in tqdm(group_yes_logits.items(), desc="Computing discrimination scores"):
        if group != baseline_group:
            discrimination_scores[group] = yes_logits.mean() - baseline_yes_logits.mean()

    # Print the discrimination scores
    for group, score in discrimination_scores.items():
        print(f"Discrimination Score for {group} (compared to {baseline_group}): {score:.4f}")

    print("\nCumulative Cost for Each Group:")
    for group, cost in group_cumulative_cost.items():
        print(f"Group: {group}, Cumulative Cost: {cost}")

    print("\nWeighted Cost for Each Group:")
    for group, weighted_cost in group_weighted_cost.items():
        print(f"Group: {group}, Weighted Cost: {weighted_cost}")

    # Calculate the difference in cumulative cost and weighted cost compared to the baseline group
    baseline_cumulative_cost = group_cumulative_cost[baseline_group]
    baseline_weighted_cost = group_weighted_cost[baseline_group]

    print(f"\nDifference in Cumulative Cost Compared to {baseline_group.capitalize()} Group:")
    for group, cost in group_cumulative_cost.items():
        if group != baseline_group:
            diff_cost = cost - baseline_cumulative_cost
            print(f"Group: {group}, Difference in Cumulative Cost: {diff_cost:.2f}")

    print(f"\nDifference in Weighted Cost Compared to {baseline_group.capitalize()} Group:")
    for group, weighted_cost in group_weighted_cost.items():
        if group != baseline_group:
            diff_weighted_cost = weighted_cost - baseline_weighted_cost
            print(f"Group: {group}, Difference in Weighted Cost: {diff_weighted_cost:.2f}")

# Save the results DataFrame to a CSV file
results_df.to_csv("raw_results.csv", index=False)