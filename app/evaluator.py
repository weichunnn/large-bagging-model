import os
import weave

from pydantic import BaseModel
from openai import OpenAI
from datetime import datetime
import json
import pickle

from dotenv import load_dotenv
load_dotenv()

weave.init("together-weave")


class DebateEvaluation(BaseModel):
    respect_for_other_team: int
    information: int
    relevance_of_supporting_arguments: int
    strength_of_arguments: int
    rebuttal: int
    organization: int
    preparation: int
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
    
1. Respect for Other Team (5 points)
5:

All statements are respectful and courteous.
Participants listen attentively without interrupting.
No use of personal attacks, insults, or dismissive language.
4:

Mostly respectful with 1-2 minor instances of slight disrespect.
Participants maintain a polite tone with minimal lapses.
No significant personal attacks, but occasional mild dismissive remarks.
3:

Generally respectful but includes at least one sarcastic or slightly disrespectful remark.
Participants acknowledge opposing views but may undermine them subtly.
Minimal personal attacks that do not dominate the dialogue.
2:

Borderline appropriate with multiple instances of sarcasm or disrespect.
Participants frequently use dismissive language or mild insults.
Overall tone is affected by these lapses, reducing the debate's professionalism.
1:

Consistently disrespectful throughout the debate.
Frequent personal attacks, insults, and derogatory language.
Participants undermine each other’s credibility and contributions.
2. Information and Use of Facts/Examples (5 points)
5:

All information is clear, accurate, and thorough.
Arguments are consistently supported with relevant facts and concrete examples.
Demonstrates a comprehensive understanding of the topic with detailed explanations.
4:

Most information is clear, accurate, and well-supported.
Arguments include relevant facts and examples, though some may lack depth.
Shows a strong understanding of the topic with minor gaps in detail.
3:

Information is generally clear and accurate but lacks thoroughness.
Some arguments are supported with facts and examples, while others rely on general statements.
Demonstrates a basic understanding of the topic but may miss deeper insights.
2:

Some information presented is inaccurate or unclear.
Limited use of facts and examples to support arguments, leading to weaker points.
Shows a partial understanding of the topic with noticeable gaps.
1:

Major inaccuracies or unclear information throughout.
Arguments lack factual support and rely heavily on opinions or unsubstantiated claims.
Demonstrates a poor understanding of the topic.
3. Relevance of Supporting Arguments (5 points)
5:

All supporting arguments are highly relevant to the main thesis.
Each point directly reinforces the overall position without deviating from the topic.
Maintains focus and coherence throughout the debate.
4:

Most supporting arguments are relevant with only a few minor deviations.
Arguments generally reinforce the main thesis effectively.
Maintains a clear focus with occasional slight off-topic points.
3:

Many supporting arguments are relevant, but some are only partially related.
Occasional deviations from the main thesis that slightly detract from the overall argument.
Maintains a reasonable level of focus with some inconsistencies.
2:

Few supporting arguments are relevant to the main thesis.
Several points are off-topic or do not effectively support the overall position.
Struggles to maintain a consistent focus throughout the debate.
1:

Supporting arguments are largely irrelevant or unrelated to the main thesis.
Frequent deviations from the topic that undermine the overall argument.
Lacks coherence and focus.
4. Strength of Arguments (5 points)
5:

All arguments are strong, convincing, and well-articulated.
Effectively persuades the audience with logical reasoning and compelling evidence.
Demonstrates exceptional critical thinking and depth in argumentation.
4:

Most arguments are convincing and well-presented with minor weaknesses.
Persuades the audience through logical reasoning and adequate evidence.
Shows strong critical thinking with some depth in arguments.
3:

Some arguments are convincing, while others lack impact or depth.
Persuades the audience to a moderate extent with logical reasoning.
Demonstrates basic critical thinking with limited depth.
2:

Few arguments are convincing; most are weak or unpersuasive.
Struggles to persuade the audience due to lack of logical reasoning or evidence.
Demonstrates limited critical thinking and shallow argumentation.
1:

Arguments are not convincing and fail to persuade the audience.
Lacks logical reasoning and substantive evidence.
Demonstrates poor critical thinking and weak argumentation.
5. Rebuttal (5 points)
5:

All counter-arguments are accurate, relevant, and strong.
Effectively dismantles opposing points with logic and evidence.
Demonstrates excellent understanding and anticipation of the opponent’s arguments.
4:

Most counter-arguments are accurate, relevant, and strong.
Successfully addresses opposing points with logical reasoning and evidence.
Shows good understanding of the opponent’s arguments with minor weaknesses.
3:

Counter-arguments are accurate and relevant but include some weak points.
Addresses opposing points but may lack depth or comprehensive evidence.
Demonstrates a basic understanding of the opponent’s arguments.
2:

Some rebuttals are weak or irrelevant.
Struggles to effectively counter opposing arguments due to lack of evidence or logical reasoning.
Shows limited understanding of the opponent’s arguments.
1:

Rebuttals are not accurate or relevant.
Fails to address or understand the opponent’s points.
Lacks effective counter-arguments, relying on dismissal rather than logical refutation.
6. Organization (5 points)
5:

All arguments are logically organized with a clear and coherent structure.
Presents points systematically, demonstrating a deep understanding of the topic.
Maintains a smooth flow that enhances the overall persuasiveness of the debate.
4:

Most arguments are logical and well-organized with minor issues in flow or structure.
Presents points clearly, showing a good understanding of the topic.
Maintains coherence with slight inconsistencies.
3:

Arguments are logical and demonstrate an understanding of the main points but may lack a cohesive structure.
Presents points in a generally clear manner with some organizational gaps.
Maintains a reasonable flow with occasional disruptions.
2:

Arguments are mostly logical but exhibit some difficulties in presentation or organization.
Struggles to maintain a clear structure, affecting the overall clarity.
Demonstrates partial understanding of the topic with organizational weaknesses.
1:

Arguments are disorganized and lack logical flow.
Fails to present points coherently, making the debate difficult to follow.
Demonstrates an inadequate understanding of the topic through poor organization.

7. Preparation (5 points)
5:

Student is well-prepared, demonstrating thorough understanding and readiness to defend their arguments.
Presents arguments confidently with seamless integration of research and evidence.
Anticipates and effectively addresses potential counter-arguments.
4:

Student is mostly prepared with minor gaps in readiness or understanding.
Presents arguments clearly with adequate research and evidence.
Addresses counter-arguments with minor weaknesses.
3:

Student shows some preparation but may lack depth or completeness in defending arguments.
Presents arguments with basic research and evidence but may miss key points.
Attempts to address counter-arguments but with limited effectiveness.
2:

Student needs more preparation. Arguments are underdeveloped or responses are hesitant.
Presents arguments with minimal research and evidence.
Struggles to address counter-arguments effectively.
1:
Student is unprepared to defend arguments.
Lacks understanding of the topic and fails to present coherent points.
Does not address counter-arguments or does so ineffectively.

Start evaluation now with the rubric in mind.
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
    debate_text = f"Topic: {debate_data['topic']};;"

    # Opening statements
    for statement in debate_data["opening_statements"]:
        debate_text += f"{statement['agent']}:\n{statement['statement']}';;'"

    # Iterations
    for iteration in debate_data["iterations"]:
        for argument in iteration["arguments"]:
            debate_text += f"{argument['agent']}:\n{argument['argument']};;"

    # Closing statements
    for statement in debate_data["closing_statements"]:
        debate_text += f"{statement['agent']}:\n{statement['statement']}';;'"

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
                    "score": evaluation.choices[
                        0
                    ].message.parsed.proponent.total_points,
                    "reasoning": evaluation.choices[
                        0
                    ].message.parsed.proponent.comments,
                },
                "opponent": {
                    "score": evaluation.choices[0].message.parsed.opponent.total_points,
                    "reasoning": evaluation.choices[0].message.parsed.opponent.comments,
                },
            },
        }

    return all_evaluations


def write_evaluations_to_file(evaluations):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"all_debate_evaluations_{timestamp}.json"

    # Create a 'json' folder if it doesn't exist
    json_folder = "json"
    os.makedirs(json_folder, exist_ok=True)

    # Write the file to the 'json' folder
    filepath = os.path.join(json_folder, filename)

    with open(filepath, "w") as f:
        json.dump(evaluations, f, indent=4)

    print(f"All debate evaluations written to {filepath}")

    return filepath


if __name__ == "__main__":
    sample_data = pickle.load(open("my_variable.pkl", "rb"))
    # Evaluate all debates
    all_debate_evaluations = evaluate_all_debates(sample_data)

    # Write evaluations to JSON file
    write_evaluations_to_file(all_debate_evaluations)
