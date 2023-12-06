import csv

PROMPTS_TEMPLATE = './components/generators/prompts/templates/prompts.csv'

def find_prompt_by_id(target_id):
    csv_file = PROMPTS_TEMPLATE
    with open(csv_file, newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row if there is one
        for row in reader:
            id, prompt = row
            if id == target_id:
                return prompt
    return None  # Return None if the id is not found