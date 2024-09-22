import json
import streamlit as st
from helper import sort_debates
import time


def run_dummy_debate(n_agents):
    for i in range(n_agents):
        time.sleep(0.5)
        st.write(f"Debate {i+1} is done...")


def save_to_file(data, file_type):
    filename = f"{file_type}_debates.txt"
    with open(filename, "a") as f:
        f.write(json.dumps(data) + "\n")


st.header("Debate Arena")

# Input section
debate_topic = st.text_area("Debate topic", help="Enter the topic for the debate")

k_col, agent_col = st.columns([1, 1])
k = k_col.number_input(
    "Number of outputs to rank (K)",
    min_value=1,
    value=2,
    help="Select how many top and bottom debates to display",
)

n_agents = agent_col.number_input(
    "Number of agents for the arena",
    min_value=2,
    value=4,
    help="Select the number of agents participating in the debate",
)

if "debate_run" not in st.session_state:
    st.session_state.debate_run = False

if st.button("Run Debate"):
    st.session_state.debate_run = True
    run_dummy_debate(n_agents)
    st.success("Debate completed!")

if st.session_state.debate_run:
    # Load data from all_debate_evaluations_20240921_191635.json
    with open("all_debate_evaluations_20240921_191635.json", "r") as f:
        debates = json.load(f)

    top_k = sort_debates(debates, k, sort_type="top")
    bottom_k = sort_debates(debates, k, sort_type="bottom")

    # Results section
    st.header("Results")

    tab1, tab2 = st.tabs(["Top K Debates", "Bottom K Debates"])

    def display_debate(debate, index, is_top):
        st.subheader(f"Debate {index + 1}")

        # Extract topic from debate_text
        debate_text = debate["debate_text"]
        topic = debate_text.split("\n")[0].replace("Topic: ", "")
        st.write(f"**Topic:** {topic}")

        # Split debate text into parts
        parts = debate_text.split("\n\n")[1:]  # Skip the topic

        winner = (
            "proponent"
            if debate["evaluation"]["proponent"]["score"]
            > debate["evaluation"]["opponent"]["score"]
            else "opponent"
        )

        for part in parts:
            if part.startswith("Proponent"):
                with st.chat_message("proponent"):
                    st.markdown(f"**{part}**" if winner == "proponent" else f"_{part}_")
            elif part.startswith("Opponent"):
                with st.chat_message("opponent"):
                    st.markdown(f"**{part}**" if winner == "opponent" else f"_{part}_")

        debate["winner"] = winner

        if is_top:
            if st.button("💾 Save", key=f"save_top_{index}"):
                save_to_file(debate, "good")
                st.success("Saved to good responses!")

        else:
            if st.button("💾 Save", key=f"save_bottom_{index}"):
                save_to_file(debate, "bad")
                st.success("Saved to bad responses!")

    with tab1:
        st.subheader("Top K Debates")
        for i, debate in enumerate(top_k.values()):
            display_debate(debate, i, True)

    with tab2:
        st.subheader("Bottom K Debates")
        for i, debate in enumerate(bottom_k.values()):
            display_debate(debate, i, False)
