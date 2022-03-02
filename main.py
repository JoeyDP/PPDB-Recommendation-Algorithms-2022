from typing import List
from pathlib import Path

import typer

import src.util as util
from src.algorithm.iknn import ItemKNN, ItemKNNIterative


app = typer.Typer()
PathArgument = typer.Argument(
    ...,
    exists=True,
    file_okay=True,
    dir_okay=False,
    writable=False,
    readable=True,
    resolve_path=True,
    help="A readable file."
)


def history_from_subset_interactions(interactions, amt_users=5) -> List[List]:
    """ Take the history of the first users in the dataset and return as list of lists"""
    user_histories = dict()
    for user_id, item_id in interactions:
        if len(user_histories) < amt_users:
            user_histories[user_id] = list()

        if user_id in user_histories:
            user_histories[user_id].append(item_id)

    return list(user_histories.values())


@app.command()
def iknn(path: Path = PathArgument, item_col: str = "article_id", user_col: str = "customer_id", top_k: int = 5,
         k: int = 20, normalize: bool = False):
    """ Example of how to train an Item KNN model and get predictions. Only for debugging. """
    alg = ItemKNNIterative(k=k, normalize=normalize)

    # This one is faster, but requires more memory
    # alg = ItemKNN(k=k, normalize=normalize)

    print("Parsing data")
    interactions = util.parse_interactions(path, item_col=item_col, user_col=user_col)

    print("fitting model")
    alg.train(interactions)

    amt_users = 5
    print(f"fetching history for {amt_users} users")
    histories = history_from_subset_interactions(interactions, amt_users=amt_users)
    print(histories)
    print("computing recommendations")
    recommendations = alg.recommend_all(histories, top_k)

    print("recommendations:")
    print(recommendations)


app()
