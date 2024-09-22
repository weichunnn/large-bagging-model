import json
import streamlit as st
from helper import sort_debates
from start import run_debates
from evaluator import evaluate_all_debates, write_evaluations_to_file
from gtts import gTTS
from io import BytesIO


def save_to_file(data, file_type):
    filename = f"{file_type}_debates.txt"
    with open(filename, "a") as f:
        f.write(json.dumps(data) + "\n")


st.header("Debate Arena")

# Input section
debate_topic = st.text_area(
    "Debate topic",
    value="Should artificial intelligence be given the same rights as humans? Why or why not?",
    help="Enter the topic for the debate",
)

num_debates = st.number_input(
    "Number of debates",
    min_value=1,
    value=1,
    help="Select the number of debates to run",
)

num_iterations = st.number_input(
    "Number of iterations per debate",
    min_value=1,
    value=1,
    help="Select the number of iterations for each debate",
)

models = st.selectbox(
    "Select Proponent Model",
    options=[
        "openai/gpt-4o-2024-08-06",
        "meta-llama/llama-3.1-405b-instruct",
        "google/gemini-pro-1.5",
        "anthropic/claude-3.5-sonnet",
    ],
    help="Select the models to participate in the debates",
)

models = [models] * num_debates * 2

k = st.number_input(
    "Top K debates to display",
    min_value=1,
    value=2,
    help="Select the number of top and bottom debates to display",
)


if "debate_run" not in st.session_state:
    st.session_state.debate_run = False

if "fpath" not in st.session_state:
    st.session_state.fpath = ""

if st.button("Run Debate"):
    st.session_state.debate_run = True

    st.subheader("Running debates...")

    debates = []
    with st.spinner("Running debates..."):
        debates = run_debates(num_debates, debate_topic, models, num_iterations)

    st.success("Debate completed!")

    # Evaluate debates
    with st.spinner("Evaluating debates..."):
        all_debate_evaluations = evaluate_all_debates(debates)
        st.session_state.fpath = write_evaluations_to_file(all_debate_evaluations)

    # Write evaluations to JSON file.
    write_evaluations_to_file(all_debate_evaluations)

    st.success("Debates evaluated!")

if st.session_state.debate_run:
    if st.session_state.fpath:
        with open(st.session_state.fpath, "r") as f:
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
            parts = debate_text.split(";;")[1:]  # Skip the topic

            winner = (
                "proponent"
                if debate["evaluation"]["proponent"]["score"]
                > debate["evaluation"]["opponent"]["score"]
                else "opponent"
            )
            loser = "opponent" if winner == "proponent" else "proponent"

            for part in parts:
                if part.startswith("Proponent"):
                    with st.chat_message("proponent"):
                        st.markdown(
                            f"**{part}**"
                            if (is_top and winner == "proponent")
                            or (not is_top and loser == "proponent")
                            else f"_{part}_"
                        )

                        # sound_file = BytesIO()
                        # tts = gTTS(part, lang="en")
                        # st.audio(sound_file)
                elif part.startswith("Opponent"):
                    with st.chat_message("opponent"):
                        st.markdown(
                            f"**{part}**"
                            if (is_top and winner == "opponent")
                            or (not is_top and loser == "opponent")
                            else f"_{part}_"
                        )
                        # sound_file = BytesIO()
                        # tts = gTTS(part, lang="en")

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
