import os
import weave
import instructor
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import random

load_dotenv()
weave.init("together-weave")


SYSTEM_CONTENT = "You are a debate moderator. Be descriptive and helpful."
MODEL = "openai/gpt-4o-2024-08-06"
DEBATE_ROUNDS = 1
QUESTION = "Can alternative energy effectively replace fossil fuels?"

FAMOUS_MODELS = [
    "openai/gpt-4o-2024-08-06",
    "meta-llama/llama-3.1-405b-instruct"
]


class Agent(BaseModel):
    persona: str
    instructions: str


class Scenario(BaseModel):
    agents: list[Agent]


class AgentDebate(BaseModel):
    argument: str


def generate_user_content(question):
    return f"""
    Given this scenario below:
    
    {question}

    Break down the scenario into all of the subjects of interest involved (i.e., personas).

    Then, create a debate template that can be used to argue for or against each subject of interest (i.e., instructions).

    Note:
    1. There should always be only 2 personas in a debate: For and Against.
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


def simulate_debate(question=QUESTION, num_iterations=3):
    scenario = generate_scenario(question)

    debate_data = {
        "topic": question,
        "opening_statements": [],
        "iterations": [],
        "closing_statements": [],
    }

    debate_history = []

    agent_models = random.sample(FAMOUS_MODELS, len(scenario.agents))
    agent_model_map = {
        agent.persona: model for agent, model in zip(scenario.agents, agent_models)
    }
    print(f"Debate Topic: {question}")
    print("Debate Participants:")
    for agent, model in agent_model_map.items():
        print(f"- {agent}: using {model}")
    print()

    # Collect opening statements
    for i, agent in enumerate(scenario.agents, 1):
        response = client.chat.completions.create(
            model=agent_model_map[agent.persona],
            response_model=AgentDebate,
            messages=[
                {
                    "role": "system",
                    "content": f"You are {agent.persona}. {agent.instructions}",
                },
                {
                    "role": "user",
                    "content": f"Provide an opening statement for the debate on the topic: {question}",
                },
            ],
        )
        opening_statement = response.argument
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
            # Use the full debate history (opening statements + all previous arguments)
            context = "\n".join(debate_history)

            response = client.chat.completions.create(
                model=agent_model_map[agent.persona],
                response_model=AgentDebate,
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
            argument = response.argument
            iteration_data["arguments"].append(
                {"agent": agent.persona, "argument": argument}
            )
            debate_history.append(f"Agent {i} ({agent.persona}): {argument}")

        debate_data["iterations"].append(iteration_data)

    # Closing statements
    for i, agent in enumerate(scenario.agents, 1):
        context = "\n".join(debate_history)  # Full debate history
        response = client.chat.completions.create(
            model=agent_model_map[agent.persona],
            response_model=AgentDebate,
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
        argument = response.argument
        debate_data["closing_statements"].append(
            {"agent": agent.persona, "statement": argument}
        )

    return debate_data


debate_result = simulate_debate()
print(debate_result)
