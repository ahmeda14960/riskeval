import csv
import itertools
from tqdm import tqdm

def get_pronoun(gender, case):
    """
    Returns the appropriate pronoun based on the given gender and case.
    Args:
        gender (str): The gender (male or female).
        case (str): The case (subject, object, or possessive).
    Returns:
        str: The pronoun.
    """
    pronouns = {
        "male": {"subject": "he", "object": "him", "possessive": "his"},
        "female": {"subject": "she", "object": "her", "possessive": "her"},
    }
    return pronouns[gender][case]

def fill_template(template, gender, race, age):
    """
    Fills in the template with the specified demographic information.
    Args:
        template (str): The template prompt.
        gender (str): The gender (male or female).
        race (str): The race.
        age (int): The age.
    Returns:
        str: The filled template.
    """
    filled_template = template.replace("{race}", race)
    filled_template = filled_template.replace("{age}", str(age))
    words = filled_template.split()
    for i, word in enumerate(words):
        if word == "{gender}":
            if words[i + 1] in ["has", "is", "was"]:
                words[i] = get_pronoun(gender, "subject")
            elif words[i - 1] in ["to", "for", "with", "by"]:
                words[i] = get_pronoun(gender, "object")
            else:
                words[i] = get_pronoun(gender, "possessive")
    return " ".join(words)

def main():
    genders = ["male", "female"]
    races = ["white", "black", "latino", "native american", "asian"]
    ages = [20, 30, 40, 50, 60, 70]
    permutations = list(itertools.product(genders, races, ages))

    with open("risk_aware_discrim_eval - discrim_llm.csv", "r") as file:
        reader = csv.DictReader(file)
        with open("filled_templates.csv", "w", newline="") as outfile:
            fieldnames = [
                "filled_template",
                "category",
                "decision_question_id",
                "gender",
                "race",
                "age",
            ]
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            # Read the entire file into a list to prevent reading issues during iteration
            rows = list(reader)

            for row in rows:  # Loop over each row in the input CSV
                template = row['filled_template']  # Access the template column
                category = row['category']  # Access the category column
                decision_question_id = row['decision_question_id']  # Access the decision question ID
                for gender, race, age in permutations:
                    filled_template = fill_template(template, gender, race, age)
                    writer.writerow({
                        "filled_template": filled_template,
                        "category": category,
                        "decision_question_id": decision_question_id,
                        "gender": gender,
                        "race": race,
                        "age": age
                    })

if __name__ == "__main__":
    main()
