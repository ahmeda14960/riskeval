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
            elif words[i + 1] in [
                "mechanic",
                "teacher",
                "doctor",
                "engineer",
                "nurse",
                "lawyer",
                "user",
                "athlete",
                "athlete's",
                "school teacher",
                "student",
                "salesperson",
                "professor",
                "author",
                "journalist",
                "actor",
                "customer",
                "candidate",
                "applicant",
                "entrepreneur",
                "contractor",
                "business owner",
                "counselor",
                "client",
                "auto mechanic",
                "citizen",
                "accountant",
                "owner",
                "postdoctoral fellow",
                "representative",
                "grocery clerk",
                "freelance writer",
            ]:  # Add more nouns as needed
                words[i] = gender
            else:
                words[i] = get_pronoun(gender, "possessive")

    return " ".join(words)


def main():
    """
    Main function to read the CSV file, process each row, and write the filled templates to a new CSV file.
    """
    # Define the options for gender, race, and age
    genders = ["male", "female"]
    races = ["white", "black", "latino", "native american", "asian"]
    ages = [20, 30, 40, 50, 60, 70]

    # Generate all permutations of gender, race, and age
    permutations = list(itertools.product(genders, races, ages))

    # Read the CSV file and process each row
    with open("risk_aware_discrim_eval - discrim_llm.csv", "r") as file, open(
        "filled_templates.csv", "w", newline=""
    ) as outfile:
        reader = csv.DictReader(file)
        fieldnames = ["filled_template", "decision_question_id"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        total_rows = sum(1 for _ in reader)
        file.seek(0)  # Reset the file pointer to the beginning

        for row in tqdm(reader, total=total_rows, desc="Processing rows"):
            template = row["filled_template"]
            decision_question_id = row["decision_question_id"]
            for gender, race, age in permutations:
                filled_template = fill_template(template, gender, race, age)
                writer.writerow(
                    {
                        "filled_template": filled_template,
                        "decision_question_id": decision_question_id,
                    }
                )


if __name__ == "__main__":
    main()
