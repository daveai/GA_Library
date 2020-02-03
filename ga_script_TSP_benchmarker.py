from cifo.algorithm.genetic_algorithm import GeneticAlgorithm
from cifo.algorithm.hill_climbing import HillClimbing
from cifo.custom_problem.travel_salesman_problem import TravelSalesmanProblem
from cifo.problem.objective import ProblemObjective
from cifo.algorithm.ga_operators import (
    initialize_randomly,
    initialize_using_hc,
    initialize_using_sa,
    RouletteWheelSelection,
    RankSelection,
    TournamentSelection,
    cycle_crossover,
    pmx_crossover,
    edge_crossover,
    initialize_using_ta,
    swap_mutation,
    scramble_mutation,
    inversion_mutation,
    insert_mutation,
    elitism_replacement,
    standard_replacement,
    nwoc_crossover,
)

from cifo.util.observer import GeneticAlgorithmObserver
from data.datasets.tsp_tour import TSP_datasets
import pandas as pd
from statistics import mean

from timeit import default_timer as timer

"""
Please choose an index from one of the available datasets below, or simply
use dv which contains the TSP data provided by the instructor.
    0: 'ch150.tsp',
    1: 'gr666.tsp',
    2: 'kroC100.tsp',
    3: 'tsp225.tsp',
    4: 'eil101.tsp',
    5: 'kroA100.tsp',
    6: 'lin105.tsp',
    7: 'rd100.tsp',
    8: 'ulysses16.tsp',
    9: 'ulysses22.tsp'
"""

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

dataset = 8

# Configuration
# --------------------------------------------------------------------------------------------------
# parent selection object

parent_selection = TournamentSelection()


params = {
    # params
    "Population-Size": 20,
    "Number-of-Generations": 1000,
    "Crossover-Probability": 0.1,
    "Mutation-Probability": 0.8,
    # operators / approaches
    "Initialization-Approach": initialize_randomly,
    "Selection-Approach": parent_selection.select,
    "Tournament-Size": 20,
    "Crossover-Approach": pmx_crossover,
    "Mutation-Aproach": inversion_mutation,
    "Replacement-Approach": elitism_replacement,
}


dv = {
    "Coordinates": TSP_datasets.get_data(dataset),
    "Nodes": [i for i in range(1, len(TSP_datasets.get_data(dataset)) + 1)],
}

# Problem Instance
tsp_problem_instance = TravelSalesmanProblem(decision_variables=dv, constraints="")

# print(params)

startCum = timer()

results = []

number_of_runs = 1

# Run the same configuration many times
# --------------------------------------------------------------------------------------------------
for run in range(1, number_of_runs + 1):
    start = timer()
    # Genetic Algorithm
    ga = GeneticAlgorithm(problem_instance=tsp_problem_instance, params=params, run=run)

    ga_observer = GeneticAlgorithmObserver(ga)
    ga.register_observer(ga_observer)
    ga.search()
    results.append(ga._fittest.fitness)

print(
    f"{datasets[dataset]},{number_of_runs},{min(results)},{max(results)},{(timer()-startCum)/number_of_runs}"
)
