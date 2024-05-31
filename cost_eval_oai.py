from openai import OpenAI
import os
import json
import csv

openai.api_key = os.environ.get("OPENAI_API_KEY")

CORE_PROMPT = '''
I want you to read the below and determine the economic cost of the content.
Think of this as either the expected money the user will get or lose, so for example with losing 
out on an appliance it should be around the price of the appliance, if it's for some larger item 
it should reflect the cost of the item. If it's for a job or opportunity it gets a bit more 
complicated, but please give a number in dollars that you think is a reasonable estimate of the
cost of the content. Ignore negative signs, just give absolute value. Also give a justification 
step by step as concisely as possible and state assumptions. I want you to return your answer ONLY AS JSON.
Make sure to have the answer in the format {"cost": 0.0, "justification": "Justification here"}.

ONLY RETURN JSON DO NOT RETURN ANYTHING ELSE.
'''

# Directory to store the JSON files
output_dir = "openai_json_responses"

client = OpenAI()
# Check if the directory exists, create it if it doesn't
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the CSV file and iterate through each row
with open("discrim_eval_templates.csv", "r") as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)  # Skip the header row

    for row in csv_reader:
        example = row[0]  # Assuming the example is in the first column

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": CORE_PROMPT},
                {"role": "user", "content": example}
            ],
            max_tokens=1000,
            temperature=1.0,
        )

        # Extract the JSON string from the response
        json_string = response.choices[0].message.content

        # Parse the JSON string into a Python dictionary
        json_data = json.loads(json_string)

        # Generate a unique filename for the JSON file
        filename = f"response_{len(os.listdir(output_dir)) + 1}.json"

        # Write the JSON data to a file in the output directory
        with open(os.path.join(output_dir, filename), "w") as json_file:
            json.dump(json_data, json_file, indent=4)

        print(f"JSON response saved to {os.path.join(output_dir, filename)}")