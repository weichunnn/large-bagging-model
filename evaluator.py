import os
import weave

import instructor
from pydantic import BaseModel
from openai import OpenAI
from datetime import datetime
import json
import pickle

from dotenv import load_dotenv
load_dotenv()

weave.init('together-weave')


class DebateEvaluation(BaseModel):
    respect_for_other_team: int
    information: int
    rebuttal: int
    use_of_facts_statistics: int
    organization: int
    total_points: int
    comments: str

class DebateResult(BaseModel):
    proponent: DebateEvaluation
    opponent: DebateEvaluation


client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)


def evaluate_debate(debate_text: str) -> DebateResult:
    system_content = """
    You are a debate evaluator. Evaluate the given debate based on the rubric:

    Respect for Other Team (5 points max):
    5: All statements respectful
    4: Mostly respectful, 1-2 instances of disrespect
    3: Mostly respectful, one sarcastic remark
    2: Borderline appropriate, some sarcastic remarks
    1: Consistently disrespectful

    Information (5 points max):
    5: All information clear, accurate, thorough
    4: Most information clear, accurate, thorough
    3: Clear and accurate, but not thorough
    2: Some inaccuracies
    1: Major inaccuracies or unclear

    Rebuttal (5 points max):
    5: All counter-arguments accurate, relevant, strong
    4: Most counter-arguments accurate, relevant, strong
    3: Accurate and relevant, but some weak
    2: Some weak and irrelevant
    1: Not accurate or relevant

    Use of Facts/Statistics (5 points max):
    5: Every point well supported with facts/stats
    4: Every point adequately supported
    3: Supported, but some questionable relevance
    2: Some points supported, others not
    1: Points not supported

    Organization (5 points max):
    5: All arguments logically organized, deep understanding
    4: Most arguments logical, clear understanding
    3: Logical, understood main points
    2: Mostly logical, some difficulty in presentation
    1: Inadequate understanding of topic

    Provide scores for each category and a total score (out of 25) for both participants. Also provide brief comments explaining the scores.
    """

    user_content = f"""
    Evaluate the following debate:

    {debate_text}

    Provide scores and comments for both participants based on the rubric.
    """

    response = client.beta.chat.completions.parse(
        model="openai/gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        response_format=DebateResult,
    )

    return response

def format_debate_text(debate_data):
    debate_text = f"Topic: {debate_data['topic']}\n\n"

    # Opening statements
    for statement in debate_data['opening_statements']:
        debate_text += f"{statement['agent']}:\n{statement['statement']}\n\n"

    # Iterations
    for iteration in debate_data['iterations']:
        for argument in iteration['arguments']:
            debate_text += f"{argument['agent']}:\n{argument['argument']}\n\n"

    # Closing statements
    for statement in debate_data['closing_statements']:
        debate_text += f"{statement['agent']}:\n{statement['statement']}\n\n"

    return debate_text

# Load the sample data
# with open('sample.txt', 'r') as file:
#     sample_data = json.load(file)




# # Assuming the first debate in the list
# debate_data = sample_data['debates'][0]
# debate_text = format_debate_text(debate_data)

# print(debate_text)

# result = evaluate_debate(debate_text)

# def write_json_to_file(debate_result):
#     # Convert Pydantic model to dictionary if it's not already
#     if hasattr(debate_result, 'dict'):
#         result_dict = debate_result.dict()
#     else:
#         result_dict = debate_result

#     # Create a timestamp for the filename
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"debate_evaluation_{timestamp}.json"

#     # Write the dictionary to a JSON file
#     with open(filename, 'w') as f:
#         json.dump(result_dict, f, indent=4)

#     print(f"Evaluation result written to {filename}")

# write_json_to_file(result)


# json_response = result.choices[0].message.parsed

# ### Still need to add 

# write_json_to_file(json_response)
# print(json_response.proponent)
# print(json_response.opponent)
# print(f"Participant 1 Comments: {result.participant1.comments}")
# print(f"Participant 2 Score: {result.participant2.total_points}")
# print(f"Participant 2 Comments: {result.participant2.comments}")

def evaluate_all_debates(sample_data):
    all_evaluations = {}
    
    for index, debate in enumerate(sample_data, 1):
        debate_text = format_debate_text(debate)
        evaluation = evaluate_debate(debate_text)
        
        all_evaluations[f"debate_{index}"] = {
            "debate_text": debate_text,
            "evaluation": {
                "proponent": {
                    "score": evaluation.choices[0].message.parsed.proponent.total_points,
                    "reasoning": evaluation.choices[0].message.parsed.proponent.comments
                },
                "opponent": {
                    "score": evaluation.choices[0].message.parsed.opponent.total_points,
                    "reasoning": evaluation.choices[0].message.parsed.opponent.comments
                }
            }
        }
    
    return all_evaluations

if __name__ == "__main__":
    sample_data = pickle.load(open('my_variable.pkl', 'rb'))
    # Evaluate all debates
    all_debate_evaluations = evaluate_all_debates(sample_data)

    # Write evaluations to JSON file
    def write_evaluations_to_file(evaluations):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"all_debate_evaluations_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(evaluations, f, indent=4)
        
        print(f"All debate evaluations written to {filename}")

    write_evaluations_to_file(all_debate_evaluations)
