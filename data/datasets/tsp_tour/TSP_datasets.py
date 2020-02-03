import pandas as pd


datasets = {
    0: "ch150.tsp",
    1: "gr666.tsp",
    2: "kroC100.tsp",
    3: "tsp225.tsp",
    4: "eil101.tsp",
    5: "kroA100.tsp",
    6: "lin105.tsp",
    7: "rd100.tsp",
    8: "ulysses16.tsp",
    9: "ulysses22.tsp",
}


def get_data(dataset):
    df = pd.read_csv(
        "data/datasets/tsp_tour/" + datasets[dataset], index_col=0, header=None,
    )
    return df
