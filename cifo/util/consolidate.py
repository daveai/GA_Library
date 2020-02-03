# Consolidate the runs
# --------------------------------------------------------------------------------------------------

# save the config


def consolidate(log_name):
    # consolidate the runs information
    from os import listdir, path, mkdir
    from os.path import isfile, join
    from pandas import pandas as pd
    import numpy as np

    log_dir = f"./log/{log_name}"

    log_files = [f for f in listdir(log_dir) if isfile(join(log_dir, f))]
    print(log_files)

    fitness_runs = []
    columns_name = []
    counter = 0
    generations = []

    for log_name in log_files:
        if log_name.startswith("run_"):
            df = pd.read_excel(log_dir + "/" + log_name)
            fitness_runs.append(list(df.Fitness))
            columns_name.append(log_name.strip(".xslx"))
            counter += 1

            if not generations:
                generations = list(df["Generation"])

    # fitness_sum = [sum(x) for x in zip(*fitness_runs)]

    df = pd.DataFrame(list(zip(*fitness_runs)), columns=columns_name)

    fitness_sd = list(df.std(axis=1))
    fitness_mean = list(df.mean(axis=1))

    # df["Fitness_Sum"] = fitness_sum
    df["Generation"] = generations
    df["Fitness_SD"] = fitness_sd
    df["Fitness_Mean"] = fitness_mean
    df["Fitness_Lower"] = df["Fitness_Mean"] + df["Fitness_SD"]
    df["Fitness_Upper"] = df["Fitness_Mean"] - df["Fitness_SD"]

    if not path.exists(log_dir):
        mkdir(log_dir)

    df.to_excel(log_dir + "/all.xlsx", index=False, encoding="utf-8")

    return df
    # [sum(sublist) for sublist in itertools.izip(*myListOfLists)]
