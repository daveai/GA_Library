from cifo.algorithm.genetic_algorithm import GeneticAlgorithm
from cifo.algorithm.hill_climbing import HillClimbing
from cifo.custom_problem.travel_salesman_problem import TravelSalesmanProblem
from cifo.problem.objective import ProblemObjective
from cifo.algorithm.ga_operators import *

from cifo.util.observer import GeneticAlgorithmObserver
from data.datasets.tsp_tour import TSP_datasets
from data.datasets.tsp_tour.tsp_data import tsp_matrix
import pandas as pd
from statistics import mean

from timeit import default_timer as timer


dv = {"Distances": tsp_matrix, "Nodes": [i for i in range(1, len(tsp_matrix) + 1)]}


# Problem Instance
tsp_problem_instance = TravelSalesmanProblem(decision_variables=dv, constraints="")

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
    "Initialization-Approach": initialize_using_hc,
    "Selection-Approach": parent_selection.select,
    "Tournament-Size": 10,
    "Crossover-Approach": pmx_crossover,
    "Mutation-Aproach": inversion_mutation,
    "Replacement-Approach": elitism_replacement,
}


number_of_runs = 5

startCum = timer()

results = []
resultsRep = []


# Run the same configuration many times
# --------------------------------------------------------------------------------------------------
for run in range(1, number_of_runs + 1):
    start = timer()
    # Genetic Algorithm
    ga = GeneticAlgorithm(problem_instance=tsp_problem_instance, params=params, run=run)

    ga_observer = GeneticAlgorithmObserver(ga)
    ga.register_observer(ga_observer)
    ga.search()
    ga.save_log()

    results.append(ga._fittest.fitness)
    resultsRep.append(ga._fittest.representation)

print(f"cum time, {timer()-startCum}")
print(
    f"total runs: {number_of_runs}, best fit: {min(results)}, worst: {max(results)}, mean: {mean(results)}"
)

for i in range(len(results)):
    print(f"\nsolution fitness: {results[i]}, representation: {resultsRep[i]}")
