# riskeval
# Bias Evaluation for Language Models

This Python script evaluates the bias of language models based on a specified attribute (race or gender) using a dataset of filled templates. It calculates discrimination scores, cumulative costs, and weighted costs for each demographic group and compares them to a baseline group.

## Requirements

Install the required packages using pip:

```
pip install -r requirements.txt
```

## Usage

1. Prepare the input files:
   - `filled_templates.csv`: A CSV file containing the filled templates data with columns for the specified attribute (race or gender), age, decision question ID, and filled template text.
   - `cost.csv`: A CSV file containing the cost associated with each decision question ID.

2. Update the `models_to_evaluate` list in the script with the desired models to evaluate. By default, it includes "mistralai/Mistral-7B-v0.1".

3. Run the script with the desired attribute and number of questions to consider:

```
python bias_evaluation.py <attribute> [<num_questions>]
```

- `<attribute>`: The attribute to vary (either "race" or "gender").
- `<num_questions>` (optional): The number of unique questions to consider. Default and Max is 40.

Example:
```
python bias_evaluation.py race 40
```

## Output

The script will output the following information for each evaluated model:

- Discrimination scores for each demographic group compared to the baseline group.
- Cumulative cost for each demographic group.
- Weighted cost for each demographic group.
- Difference in cumulative cost compared to the baseline group.
- Difference in weighted cost compared to the baseline group.

The script will also generate a `raw_results.csv` file containing the raw results, including the logits and model name for each question.

## Notes

- The script assumes that the input files (`filled_templates.csv` and `cost.csv`) are located in the same directory as the script.
- The script uses CUDA for GPU acceleration if available. If CUDA is not available, it will run on the CPU.
- The baseline group is set to "white" for the "race" attribute and "male" for the "gender" attribute.
- The script fixes the gender to "male" and age to 30 when varying the "race" attribute, and fixes the age to 30 and race to "white" when varying the "gender" attribute.

Feel free to modify the script and input files according to your specific requirements and dataset.