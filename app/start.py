import os
from evaluator import evaluate_all_debates
import weave
import instructor
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import concurrent.futures
from style_generator import generate_style_prompt
import streamlit as st

load_dotenv()
weave.init("together-weave", key=st.secrets["WB_KEY"])


SYSTEM_CONTENT = "You are a debate moderator. Be descriptive and helpful."
MODEL = "openai/gpt-4o-2024-08-06"
QUESTION = "Assume a train is headed to 5 children tied to the tracks. You have the option to pull a lever and divert the train to a track with 1 grandma. Should you pull the lever? Why or why not?"


class Agent(BaseModel):
    persona: str
    instructions: str


class Scenario(BaseModel):
    agents: list[Agent]


def generate_user_content(question):
    return f"""
    Given this scenario below:
    
    {question}

    Break down the scenario into all of the subjects of interest involved (i.e., personas).

    Then, create a debate template that can be used to argue for or against each subject of interest (i.e., instructions).

    Note:
    1. There should always be only 2 personas in a debate: For (Proponent) and Against (Opponent).
    2. The debate should be structured in such a way that it is easy to follow and understand.
    """


config = OpenAI(
    api_key=st.secrets["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

client = instructor.from_openai(config)


def generate_scenario(question):
    return client.chat.completions.create(
        model=MODEL,
        response_model=Scenario,
        messages=[
            {"role": "system", "content": SYSTEM_CONTENT},
            {"role": "user", "content": generate_user_content(question)},
        ],
        temperature=0.7,
    )


def simulate_debate(question, num_iterations, agent_model_map, debate_styles):
    scenario = generate_scenario(question)

    debate_data = {
        "topic": question,
        "opening_statements": [],
        "iterations": [],
        "closing_statements": [],
    }

    debate_history = []

    print(f"Debate Topic: {question}")
    print("Debate Participants:")

    # Ensure we have exactly two agents and two models
    if len(scenario.agents) != 2 or len(agent_model_map) != 2:
        raise ValueError(
            "Scenario must have exactly 2 agents and 2 models for this setup."
        )

    # Assign models to agents based on their order, not their persona names
    agent_model_list = list(agent_model_map.values())
    for i, (agent, model) in enumerate(zip(scenario.agents, agent_model_list)):
        print(f"- {agent.persona}: using {model}")
        agent_model_map[
            agent.persona
        ] = model  # Update the map with the correct persona
    print()

    # Collect opening statements
    # for i, (agent, style) in enumerate(zip(scenario.agents, debate_styles), 1):
    # response = client.chat.completions.create(
    #    model=agent_model_map[agent.persona],
    #    response_model=None,
    #    messages=[
    #        {
    #            "role": "system",
    #            "content": f"You are {agent.persona}. {agent.instructions} Debate in the following style: {style}",
    #        },
    #        {
    #            "role": "user",
    #            "content": f"Provide an opening statement for the debate on the topic: {question}.",
    #        },
    #    ],
    # )
    # opening_statement = response.choices[0].message.content

    # debate_data["opening_statements"].append(
    #    {"agent": agent.persona, "statement": opening_statement}
    # )
    # debate_history.append(
    #    f"Agent {i} ({agent.persona}) Opening Statement: {opening_statement}"
    # )

    # Debate iterations
    for iteration in range(1, num_iterations + 1):
        iteration_data = {"iteration": iteration, "arguments": []}
        for i, agent in enumerate(scenario.agents, 1):
            context = "\n".join(debate_history)

            response = client.chat.completions.create(
                model=agent_model_map[agent.persona],
                response_model=None,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are {agent.persona}. {agent.instructions}",
                    },
                    {
                        "role": "user",
                        "content": f"Given the debate history:\n{context}\n\nProvide your argument for iteration {iteration} of the debate. Keep it concise and look at the above arguments to make your point stronger. Speak like a debater and like a person. IT MUST BE LESS THAN 50 WORDS",
                    },
                ],
            )
            argument = response.choices[0].message.content
            iteration_data["arguments"].append(
                {"agent": agent.persona, "argument": argument}
            )
            debate_history.append(f"Agent {i} ({agent.persona}): {argument}")

        debate_data["iterations"].append(iteration_data)

    # Closing statements
    # for i, agent in enumerate(scenario.agents, 1):
    #    context = "\n".join(debate_history)
    #    response = client.chat.completions.create(
    #        model=agent_model_map[agent.persona],
    #        response_model=None,
    #        messages=[
    #            {
    #                "role": "system",
    #                "content": f"You are {agent.persona}. {agent.instructions}",
    #            },
    #            {
    #                "role": "user",
    #                "content": f"Given the full debate history:\n{context}\n\nProvide your closing statement for the debate.",
    #            },
    #        ],
    #    )
    #    argument = response.choices[0].message.content
    #    debate_data["closing_statements"].append(
    #        {"agent": agent.persona, "statement": argument}
    #    )

    return debate_data


def run_single_debate(question, num_iterations, subset, styles):
    print(f"\nRunning debate with models: {subset}")
    scenario = generate_scenario(question)

    if len(scenario.agents) != 2:
        print("Error: Scenario must have exactly 2 agents for this setup.")
        return None

    agent_model_map = {
        "agent1": subset[0],
        "agent2": subset[1],
    }

    return simulate_debate(
        question=question,
        num_iterations=num_iterations,
        agent_model_map=agent_model_map,
        debate_styles=styles,
    )


def run_debates(num_debates, question, models, num_iterations):
    if len(models) < num_debates * 2:
        raise ValueError("Not enough models for the requested number of debates.")

    model_subsets = [models[i : i + 2] for i in range(0, num_debates * 2, 2)]
    styles = generate_style_prompt(num_debates * 2)  # Generate twice as many styles

    debate_tasks = []
    for i, subset in enumerate(model_subsets):
        debate_styles = styles.style_description[
            i * 2 : i * 2 + 2
        ]  # Get two unique styles for each debate
        debate_tasks.append((question, num_iterations, subset, debate_styles))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        all_debate_results = list(
            executor.map(lambda x: run_single_debate(*x), debate_tasks)
        )

    return [result for result in all_debate_results if result is not None]


# Example usage
FAMOUS_MODELS = [
    "openai/gpt-4o-2024-08-06",
    "meta-llama/llama-3.1-405b-instruct",
    "google/gemini-pro-1.5",
    # "cohere/command-r-plus-08-2024",
    "anthropic/claude-3.5-sonnet",
]


def main():
    custom_question = "Should artificial intelligence be given the same rights as humans? Why or why not?"
    selected_models = FAMOUS_MODELS[:4]  # Use the first 4 models from FAMOUS_MODELS
    num_debates = 2
    num_iterations = 3

    all_results = run_debates(
        num_debates, custom_question, selected_models, num_iterations
    )

    evaluate_all_debates(all_results)


if __name__ == "__main__":
    main()
