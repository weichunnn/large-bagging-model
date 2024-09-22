import json


def extract_debates(type_deb: str) -> list:
    # Open the file containing the good debates
    with open(f"{type_deb}_debates.txt", "r") as file:
        content = file.read()
    if content == "":
        return []
    debates = content.strip().split("\n")

    results = []
    for debate in debates:
        print(debate)
        debate = json.loads(debate)
        # Extract relevant information
        winner = debate["winner"]
        winner_eval = debate["evaluation"][winner.lower()]
        debate_text = debate["debate_text"]

        # Split the debate text into sections
        sections = debate_text.split("\n\n")
        # Identify the winner's sections
        winner_text = []
        for section in sections:
            if winner.capitalize() in section:
                winner_text.append(section)

        # Join the winner's sections into a single string
        winner_text = "\n\n".join(winner_text)
        # Remove the section headers
        winner_text = winner_text.replace("\n\nOpponent of Alternative Energy:", "")
        winner_text = winner_text.replace("\n\nProponent of Alternative Energy:", "")

        # Format the result
        result = f"The topic question:{sections[0]},The winning response:{winner_text}, The reasoning for the response winning: {winner_eval['reasoning']}"

        # Save the result into a list
        results.append(result)

    return results


def extract_both_debates():
    # Extract the good debates
    good_debates = extract_debates("good")
    # Extract the bad debates
    bad_debates = extract_debates("bad")
    return good_debates, bad_debates
    # return bad_debates


print(extract_both_debates())
