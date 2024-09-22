
import os
import weave

import instructor
from pydantic import BaseModel
from openai import OpenAI

weave.init('together-weave')

from dotenv import load_dotenv
load_dotenv()


class Agent(BaseModel):
    persona: str
    instructions: str

class Scenario(BaseModel):
    agents: list[Agent]


system_content = "You are a debate moderator. Be descriptive and helpful."

def generate_user_content(question, ):
  return f"""
  Given this scenario below:
  
  {question}

  Break down the scenario down to all of the subject of interest involved ie persona.

  Then, create a debate template that can be used to argue for or against for each of the subject of interest ie instructions

  Note:
  1.  There should always be at least 2 persona in a debate. But more can be added.
  2.  The debate should be structured in such a way that it is easy to follow and understand.
  """

config = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

client = instructor.from_openai(config)

question = "Can alternative energy effectively replace fossil fuels?"

scenario = client.chat.completions.create(
    model="openai/gpt-4o-2024-08-06",
    response_model=Scenario,
    messages=[
        {"role": "system", "content": system_content},
        {"role": "user", "content": generate_user_content(question)},
    ],
    temperature=0.7,
)

class AgentDebate(BaseModel):
    argument: str

def simulate_debate(scenario: Scenario, num_iterations=3):
    debate_data = {
        "topic": question,
        "iterations": [],
        "closing_statements": []
    }
    
    debate_history = []
    
    for iteration in range(1, num_iterations + 1):
        iteration_data = {"iteration": iteration, "arguments": []}
        for i, agent in enumerate(scenario.agents, 1):
            context = "\n".join([f"Iteration {i+1}: {msg}" for i, msg in enumerate(debate_history)])
            
            response = client.chat.completions.create(
                model="openai/gpt-4o-2024-08-06",
                response_model=AgentDebate,
                messages=[
                    {"role": "system", "content": f"You are {agent.persona}. {agent.instructions}"},
                    {"role": "user", "content": f"Given the debate history:\n{context}\n\nProvide your argument for iteration {iteration} of the debate."},
                ]
            )
            argument = response.argument
            iteration_data["arguments"].append({
                "agent": agent.persona,
                "argument": argument
            })
            debate_history.append(f"Agent {i} ({agent.persona}): {argument}")
        
        debate_data["iterations"].append(iteration_data)
    
    # Closing statements
    for i, agent in enumerate(scenario.agents, 1):
        response = client.chat.completions.create(
            model="openai/gpt-4o-2024-08-06",
            response_model=AgentDebate,
            messages=[
                {"role": "system", "content": f"You are {agent.persona}. {agent.instructions}"},
                {"role": "user", "content": f"Given the full debate history:\n{' '.join(debate_history)}\n\nProvide your closing statement for the debate."},
            ],
        )
        argument = response.argument
        debate_data["closing_statements"].append({
            "agent": agent.persona,
            "statement": argument
        })
    
    return debate_data

debate_result = simulate_debate(scenario, num_iterations=2)
print(debate_result)



# ## Debate Structure and Tips
# ### 1. Case Construction

# - Brainstorm reasons supporting your position
# - Group arguments into categories (economic, social, political)
# - Prioritize arguments and create taglines (contentions)
# - Develop 3 unique, non-contradictory contentions
# - For each contention, provide:
#   - Clear claim
#   - Supporting evidence
#   - Relevance to the topic

# ### 2. Refutation

# - Listen carefully and take notes on opponent's arguments
# - Identify claim, evidence, and relevance of each contention
# - Use "THEY SAY, I SAY BECAUSE" format to refute
# - Address all parts: claim, evidence, relation to topic
# - Avoid straw man arguments
# - Be strategic, agree where appropriate while emphasizing your case's strength

# ### 3. Case Rebuilding

# - Anticipate objections during preparation
# - Don't reveal refutations in advance
# - Use "THEY SAY, I SAY BECAUSE" structure
# - Add new analysis, evidence, and examples

# ### 4. Closing Remarks

# - Focus on new analysis of previous points, not new arguments
# - Address points of clash (economic, social, political)
# - Explain why your side prevails on these issues

# ## General Tips

# - Take thorough notes
# - Maintain confidence
# - Focus on refuting the case, not personal attacks
# - Be engaging and use appropriate body language
# - Employ humor to connect with the audience

