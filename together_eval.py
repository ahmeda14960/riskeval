import requests
import json
import os

API_KEY = "f778d9f631555b53919412d0fe81c91696140229fee66a22b18fcd69976be3e0"#os.environ["TOGETHER_API_KEY"]
API_URL = "https://api.together.xyz/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# Define the possible answers and their corresponding labels
answer_choices = ["yes", "no"]
answer_labels = [1, 0]  # Assign labels to the answer choices

# Generate some prompts
prompts = ["What is your name?", "How old are you?", "Where are you from?"]
# Iterate through your prompts
for prompt in prompts:
    data = {
        "model": "codellama/CodeLlama-70b-Instruct-hf",
        "stream": False,
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "logprobs": 1,  # Specify the number of top logprobs to return
        "echo": False
    }

    response = requests.post(API_URL, headers=headers, data=json.dumps(data))

    import ipdb; ipdb.set_trace()
    if response.status_code == 200:
        result = response.json()
        
        # Extract the generated completion
        completion = result["choices"][0]["message"]["content"]
        
        # Extract the logprobs for the completion
        logprobs = result["choices"][0]["logprobs"]
        
        # Find the index of "yes" and "no" in the logprobs tokens
        yes_index = logprobs["tokens"].index("yes") if "yes" in logprobs["tokens"] else None
        no_index = logprobs["tokens"].index("no") if "no" in logprobs["tokens"] else None
        
        if yes_index is not None:
            yes_logprob = logprobs["token_logprobs"][yes_index]
            print(f"Logprob for 'yes': {yes_logprob}")
        
        if no_index is not None:
            no_logprob = logprobs["token_logprobs"][no_index]
            print(f"Logprob for 'no': {no_logprob}")
        
        print(f"Prompt: {prompt}")
        print(f"Completion: {completion}")
        print("---")
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)