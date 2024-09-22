import os
import weave
import instructor
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import concurrent.futures
from style_generator import generate_style_prompt

load_dotenv()
weave.init("together-weave")


SYSTEM_CONTENT = "You are a debate moderator. Be descriptive and helpful."
MODEL = "openai/gpt-4o-2024-08-06"
DEBATE_ROUNDS = 1
QUESTION = "Can alternative energy effectively replace fossil fuels?"

FAMOUS_MODELS = [
    "openai/gpt-4o-2024-08-06",
    "meta-llama/llama-3.1-405b-instruct",
    "google/gemini-pro-1.5",
    # "cohere/command-r-plus-08-2024",
    "anthropic/claude-3.5-sonnet",
]


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
    api_key=os.environ.get("OPENROUTER_API_KEY"),
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


def simulate_debate(question=QUESTION, num_iterations=3, agent_model_map=None, debate_styles=None):
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
        raise ValueError("Scenario must have exactly 2 agents and 2 models for this setup.")
    
    # Assign models to agents based on their order, not their persona names
    agent_model_list = list(agent_model_map.values())
    for i, (agent, model) in enumerate(zip(scenario.agents, agent_model_list)):
        print(f"- {agent.persona}: using {model}")
        agent_model_map[agent.persona] = model  # Update the map with the correct persona
    print()

    # Collect opening statements
    for i, (agent, style) in enumerate(zip(scenario.agents, debate_styles), 1):
        response = client.chat.completions.create(
            model=agent_model_map[agent.persona],
            response_model=None,
            messages=[
                {
                    "role": "system",
                    "content": f"You are {agent.persona}. {agent.instructions} Debate in the following style: {style}",
                },
                {
                    "role": "user",
                    "content": f"Provide an opening statement for the debate on the topic: {question}.",
                },
            ],
        )
        opening_statement = response.choices[0].message.content

        debate_data["opening_statements"].append(
            {"agent": agent.persona, "statement": opening_statement}
        )
        debate_history.append(
            f"Agent {i} ({agent.persona}) Opening Statement: {opening_statement}"
        )

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
                        "content": f"Given the debate history:\n{context}\n\nProvide your argument for iteration {iteration} of the debate.",
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
                    "content": f"Given the full debate history:\n{context}\n\nProvide your closing statement for the debate.",
                },
            ],
        )
        argument = response.choices[0].message.content
        debate_data["closing_statements"].append(
            {"agent": agent.persona, "statement": argument}
        )

    return debate_data

def run_single_debate(subset, styles):
    print(f"\nRunning debate with models: {subset}")
    scenario = generate_scenario(QUESTION)

    if len(scenario.agents) != 2:
        print("Error: Scenario must have exactly 2 agents for this setup.")
        return None

    agent_model_map = {
        "agent1": subset[0],
        "agent2": subset[1],
    }

    return simulate_debate(
        question=QUESTION, num_iterations=DEBATE_ROUNDS, agent_model_map=agent_model_map, debate_styles=styles
    )

def run_debates():
    model_subsets = [FAMOUS_MODELS[i : i + 2] for i in range(0, len(FAMOUS_MODELS), 2)]
    styles = generate_style_prompt(len(model_subsets) * 2)  # Generate twice as many styles
    
    debate_tasks = []
    for i, subset in enumerate(model_subsets):
        debate_styles = styles.style_description[i*2 : i*2+2]  # Get two unique styles for each debate
        debate_tasks.append((subset, debate_styles))
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        all_debate_results = list(executor.map(lambda x: run_single_debate(*x), debate_tasks))

    return [result for result in all_debate_results if result is not None]


all_results = run_debates()

for i, result in enumerate(all_results, 1):
    print(f"\nDebate {i} Results:")
    print(result)
