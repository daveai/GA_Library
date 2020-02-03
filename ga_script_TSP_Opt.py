from cifo.algorithm.genetic_algorithm import GeneticAlgorithm
from cifo.algorithm.hill_climbing import HillClimbing
from cifo.custom_problem.travel_salesman_problem import TravelSalesmanProblem
from cifo.custom_problem.algo_optimizer_problem import AlgoOptimizerProblem
from cifo.problem.objective import ProblemObjective
from cifo.algorithm.ga_operators import (
    initialize_randomly,
    initialize_using_hc,
    initialize_using_sa,
    RouletteWheelSelection,
    RankSelection,
    TournamentSelection,
    singlepoint_crossover,
    npoint_crossover,
    uniform_crossover,
    arithmetic_crossover,
    cycle_crossover,
    pmx_crossover,
    single_point_mutation,
    multi_point_mutation,
    swap_mutation,
    scramble_mutation,
    inversion_mutation,
    insert_mutation,
    elitism_replacement,
    standard_replacement,
)
from cifo.util.terminal import Terminal, FontColor
from cifo.util.observer import GeneticAlgorithmObserver
from cifo.util.plot import plot_performance_chart
from cifo.util.consolidate import consolidate
from data.datasets.tsp_tour.tsp_data import tsp_matrix
from timeit import default_timer as timer

# Problem
# --------------------------------------------------------------------------------------------------
# Decision Variables
# from the pdf

dv = {"Distances": tsp_matrix, "Nodes": [i for i in range(1, len(tsp_matrix) + 1)]}

# burma14tsp (from http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/ )
# Optimal = 33.23 (not sure, in the website shows 3323, but that's too big)
# dv = {
#     "Nodes"    : [1, 2, 3, 4, 5, 6],
#
#     "Coordinates": [[16.47,96.10],
#                     [16.47,94.44],
#                     [20.09,92.54],
#                     [22.00,96.05],
#                     [20.47,97.02],
#                     [20.09,94.55]]
#
# }
# Problem Instance
tsp_problem_instance = TravelSalesmanProblem(decision_variables=dv, constraints="")


# Configuration
# --------------------------------------------------------------------------------------------------

# ga = GeneticAlgorithm(
#     problem_instance=tsp_problem_instance,
#     params=params,
#     run=0
#     )

dvOpt = {"Problem": tsp_problem_instance}

algo_opt_problem_instance = AlgoOptimizerProblem(
    decision_variables=dvOpt
)  # Encoding and Constraints are defined by default

parent_selection = TournamentSelection()

paramsOpt = {
    # params
    "Population-Size": 5,
    "Number-of-Generations": 2,
    "Crossover-Probability": 0.8,
    "Mutation-Probability": 0.8,
    # operators / approaches
    "Initialization-Approach": initialize_randomly,
    "Selection-Approach": parent_selection.select,
    "Tournament-Size": 3,
    "Crossover-Approach": npoint_crossover,
    "Mutation-Aproach": multi_point_mutation,
    "Replacement-Approach": elitism_replacement,
}


log_name = "TSP_Algo_opt"
start = timer()

number_of_runs = 1
# Run the same configuration many times
# --------------------------------------------------------------------------------------------------
for run in range(1, number_of_runs + 1):
    # Genetic Algorithm
    ga = GeneticAlgorithm(
        problem_instance=algo_opt_problem_instance,
        params=paramsOpt,
        run=run,
        log_name=log_name,
    )

    ga_observer = GeneticAlgorithmObserver(ga)
    ga.register_observer(ga_observer)
    fit = ga.search()
    ga.save_log()
    best_solutions = ga._population.solutions
    populations = ga.get_populations()

# df = consolidate(log_name)
# plot_performance_chart(df)
print("time:", timer() - start)

print("Fittest: " + str(fit.representation))
for sol in best_solutions:
    print("Last generation individual: " + str(sol.representation))
    print("Last generation fitness: " + str(sol.fitness))

for i, pop in enumerate(populations):
    print(i)
    for sol in pop.solutions:
        print(sol.representation)
        print(sol.fitness)
