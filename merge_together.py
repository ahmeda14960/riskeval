import csv
from collections import defaultdict

def process_csv(input_file, output_file):
    data = defaultdict(lambda: {'yes_count': 0, 'no_count': 0})

    # Read the input CSV file
    with open(input_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            key = (row['decision_question_id'], row['race'], row['model'])
            data[key]['yes_count'] += int(row['yes_count'])
            data[key]['no_count'] += int(row['no_count'])
            data[key]['filled_template'] = row['filled_template']
            data[key]['gender'] = row['gender']
            data[key]['age'] = row['age']
            data[key]['tested_attribute'] = row['tested_attribute']

    # Calculate the yes_prob for each unique row
    for key in data:
        yes_count = data[key]['yes_count']
        no_count = data[key]['no_count']
        data[key]['yes_prob'] = yes_count / (yes_count + no_count)

    # Write the data to the output CSV file
    with open(output_file, 'w', newline='') as file:
        fieldnames = ['decision_question_id', 'filled_template', 'gender', 'age', 'race',
                      'yes_count', 'no_count', 'yes_prob', 'tested_attribute', 'model']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for key in data:
            decision_question_id, race, model = key
            row = {
                'decision_question_id': decision_question_id,
                'filled_template': data[key]['filled_template'],
                'gender': data[key]['gender'],
                'age': data[key]['age'],
                'race': race,
                'yes_count': data[key]['yes_count'],
                'no_count': data[key]['no_count'],
                'yes_prob': data[key]['yes_prob'],
                'tested_attribute': data[key]['tested_attribute'],
                'model': model
            }
            writer.writerow(row)

# Usage example
input_file = 'raw_results.csv'
output_file = 'raw_result_prob.csv'
process_csv(input_file, output_file)