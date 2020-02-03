from copy import deepcopy
from pandas import pandas as pd
import os


class GeneticAlgorithmLogger:
    """
        Save a file for each run (search) of a genetic algorithm
    """

    #
    # ----------------------------------------------------------------------------------------------
    def __init__(self, log_name, run):
        self._log_name = log_name
        self._run = run
        self._runs = []
        self._generations = []
        self._fitness = []
        self._populations = []

    #
    # ----------------------------------------------------------------------------------------------
    def add(self, generation, solution):
        self._runs.append(self._run)
        self._generations.append(generation)
        self._fitness.append(solution.fitness)

    def add_population(self, population):
        self._populations.append(population)

    def get_populations(self):
        return self._populations

    #
    # ----------------------------------------------------------------------------------------------
    def save(self):

        log_dir = f"./log/{self._log_name}"
        file_name = f"/run_{self._run}.xlsx"

        print(f"log: {log_dir + file_name}")

        df = pd.DataFrame(
            list(zip(self._runs, self._generations, self._fitness)),
            columns=["Run", "Generation", "Fitness"],
        )

        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        self._populations.append(self._fitness)
        # df.to_excel(log_dir + file_name, index=False, encoding="utf-8")

    def get_fitnesses(self):
        return self._fitness
