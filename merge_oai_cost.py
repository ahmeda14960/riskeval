import os
import csv
import json

# Read the cost.csv file
with open('cost.csv', 'r') as file:
    csv_reader = csv.DictReader(file)
    cost_data = list(csv_reader)

# Iterate through the JSON files in the claude_json_responses directory
json_directory = 'gpt3.5_json_responses'
for filename in os.listdir(json_directory):
    if filename.endswith('.json'):
        with open(os.path.join(json_directory, filename), 'r') as json_file:
            json_data = json.load(json_file)
            
            decision_question_id = json_data['decision_question_id']
            cost = json_data['cost']
            justification = json_data['justification']
            
            # Find the corresponding row in cost_data and update it
            for row in cost_data:
                if row['decision_question_id'] == decision_question_id:
                    row['cost'] = cost
                    row['citation/explanation'] = justification
                    break

# Remove the extra "cost (yann)" column
for row in cost_data:
    del row['cost (yann)']

# Write the updated data to cost_claude.csv
fieldnames = cost_data[0].keys()
with open('cost_oai.csv', 'w', newline='') as file:
    csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
    csv_writer.writeheader()
    csv_writer.writerows(cost_data)