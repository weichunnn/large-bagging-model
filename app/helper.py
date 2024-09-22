from pprint import pprint
from operator import itemgetter
import json


def sort_debates(debates, k, sort_type="top"):
    """
    Sort debates by highest or lowest score across all proponent and opponent scores.

    :param debates: Dictionary of debate dictionaries
    :param k: Number of top/bottom debates to return
    :param sort_type: 'top' for highest scores, 'bottom' for lowest scores
    :return: Dictionary of k sorted debates in the original format
    """
    compare_func = max if sort_type == "top" else min

    def get_score(debate):
        prop_score = debate["evaluation"]["proponent"]["score"]
        op_score = debate["evaluation"]["opponent"]["score"]
        return compare_func(prop_score, op_score)

    # Sort the debates based on the score
    sorted_debates = sorted(
        debates.items(), key=lambda x: get_score(x[1]), reverse=(sort_type == "top")
    )

    # Return top/bottom k debates in the original format
    return dict(sorted_debates[:k])


def main():
    # Read the debate data from all_debate_evaluations_20240921_191635.json
    with open("all_debate_evaluations_20240921_191635.json", "r") as file:
        debates = json.load(file)

    # Test case 1: Get top 2 debates
    print("Top 2 debates:")
    top_2_debates = sort_debates(debates, k=2, sort_type="top")
    pprint(top_2_debates)

    # Test case 2: Get bottom 2 debates
    print("Bottom 2 debates:")
    bottom_2_debates = sort_debates(debates, k=2, sort_type="bottom")
    pprint(bottom_2_debates)


if __name__ == "__main__":
    main()
