from copy import deepcopy
from random import choice, sample, randint
import numpy as np

from cifo.algorithm.genetic_algorithm import GeneticAlgorithm
from cifo.problem.case_problem import CaseProblem
from cifo.problem.problem_template import ProblemTemplate
from cifo.problem.objective import ProblemObjective
from cifo.problem.solution import LinearSolution, Encoding
from cifo.algorithm.ga_operators import (
    initialize_randomly,
    initialize_using_hc,
    initialize_using_sa,
    initialize_using_ta,
    RouletteWheelSelection,
    RankSelection,
    TournamentSelection,
    singlepoint_crossover,
    npoint_crossover,
    uniform_crossover,
    arithmetic_crossover,
    cycle_crossover,
    pmx_crossover,
    edge_crossover,
    single_point_mutation,
    multi_point_mutation,
    swap_mutation,
    scramble_mutation,
    inversion_mutation,
    insert_mutation,
    elitism_replacement,
    standard_replacement,
)

# Each gene represents a very specific part of the problem
# - 1st is pop size
# - 2nd num of generations
# - 3rd crossover prob
# - 4th mutation prob
# - 5th initialization approach
# - 6th selection approach
# - 7th tournament size (only matters when sel approach is tournament)
# - 8th crossover approach
# - 9th mutation approach
# - 10th replacement approach
# To make things easier, all genes will be defined by integers.
# - Probabilities of 1, 2, 3, ... , 10 correspond with 10%, 20%, 30%, ..., 100%
# - Approaches will be defined by an integer (ex. single point cross is 1, multi point is 2, etc)
# Consequently, very specific rules must be defined per gene. We will define this both on
# - the Data: defining the max and min of each gene
# - the Constraints: defining which integers are available depending on the type of the problem (TSP, PIP, etc)
from cifo.util.observer import GeneticAlgorithmObserver

algo_opt_encoding_rule = {
    "Size": 10,
    "Is ordered": True,
    "Can repeat": True,
    "Data": {
        "min": [20, 1000, 0, 0, 1, 1, 2, 1, 1, 1],
        "max": [20, 1000, 10, 10, 4, 1, 20, 3, 2, 1],
    },
    "Data Type": "Interval",
    "Multi Objective": False,
}

# example for values
# 0        "Population-Size"           : 10,
# 1        "Number-of-Generations"     : 10,
# 2        "Crossover-Probability"     : 0.8,
# 3        "Mutation-Probability"      : 0.8,
# 4        "Initialization-Approach"   : initialize_randomly,
# 5        "Selection-Approach"        : parent_selection.select,
# 6        "Tournament-Size"           : 5,
# 7        "Crossover-Approach"        : pmx_crossover,
# 8        "Mutation-Aproach"          : swap_mutation,
# 9        "Replacement-Approach"      : elitism_replacement

# Translators (dicts that define Id per type of operation)
# Initialization approach
transl_init = {
    1: initialize_randomly,
    2: initialize_using_hc,
    3: initialize_using_sa,
    4: initialize_using_ta,
}
# Selection approach
transl_select = {1: TournamentSelection}
# Cross-over approach
transl_cross = {1: cycle_crossover, 2: pmx_crossover, 3: edge_crossover}
# Mutation approach
transl_mut = {1: inversion_mutation, 2: insert_mutation}
# Replacement approach
transl_repl = {1: elitism_replacement}

algo_opt_constraints_example = {
    # No constraints for 0, 1, 2, 3
    # No constraints for 4 (every init approach may be applied independently of the problem)
    # No constraints for 5 (every selection approach may be applied independently of the problem)
    # Constraints for 6 - the tournament size must be less or equal to the pop size
    6: {"MaxCompareWith": 0},
    # Constraints for 7 - if permutation, only pmx, cycle and edge may be applied
    # 7: {"ProbConditions": {"is_ordered": True, "can_repeat_elements": False}, "Allowed": [5, 6, 7]},
    # Constraints for 8 - if permutation, only swap, insert, invert and scramble may be applied
    # 8: {"ProbConditions": {"is_ordered": True, "can_repeat_elements": False}, "Allowed": [3, 4, 5, 6]},
    # No constraints for 9 (every replacement approach may be applied independently of the problem)
}

# The decision variable for this problem is the instance of the problem that must be optimized

# -------------------------------------------------------------------------------------------------
# TSP - Travel Salesman Problem
# -------------------------------------------------------------------------------------------------
class AlgoOptimizerProblem(ProblemTemplate):
    """
    """

    # Constructor
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        decision_variables,
        constraints=algo_opt_constraints_example,
        encoding_rule=algo_opt_encoding_rule,
    ):
        """
        """
        self._problem = decision_variables["Problem"]

        # Call the Parent-class constructor to store these values and to execute  any other logic to be implemented by the constructor of the super-class
        super().__init__(
            decision_variables=decision_variables,
            constraints=constraints,
            encoding_rule=encoding_rule,
        )

        # 1. Define the Name of the Problem
        self._name = "Algorithm Optimizer"

        # 2. Define the Problem Objective
        self._objective = ProblemObjective.MultiObjective
        if "Multi Objective" in encoding_rule:
            if not encoding_rule["Multi Objective"]:
                self._objective = self._problem._objective

    # Build Solution for TSP
    # ----------------------------------------------------------------------------------------------
    def build_solution(self):
        """
         Builds a linear solution based on min and max values
       """
        solution_representation = []
        encoding_data = self._encoding.encoding_data

        for iGene in range(0, self._encoding.size):
            solution_representation.append(
                randint(encoding_data["min"][iGene], encoding_data["max"][iGene])
            )

        solution = LinearSolution(
            representation=solution_representation, encoding_rule=self._encoding_rule
        )
        return solution

    # Solution Admissibility Function - is_admissible()
    # ----------------------------------------------------------------------------------------------
    def is_admissible(
        self, solution
    ):  # << use this signature in the sub classes, the meta-heuristic
        """
        Check admissibility based on the type of problem
        """
        constraints = self._constraints
        bAdm = True
        for iGene in range(0, self._encoding.size):
            # Checks if there is a specific constraint for this gene
            if iGene in constraints:
                # Checks if conditions for gene match with problem encoding
                if "ProbConditions" in constraints[iGene]:
                    bCond = True
                    for cond in constraints[iGene]["ProbConditions"]:
                        if (
                            getattr(self._problem.encoding, cond)
                            != constraints[iGene]["ProbConditions"][cond]
                        ):
                            # This condition for gene does not match the encoding
                            bCond = False
                    # If all conditions meet, then check allowed values
                    if bCond:
                        if (
                            solution.representation[iGene]
                            not in constraints[iGene]["Allowed"]
                        ):
                            bAdm = False
                            return bAdm
                elif "MaxCompareWith" in constraints[iGene]:
                    if (
                        solution.representation[iGene]
                        > solution.representation[constraints[iGene]["MaxCompareWith"]]
                    ):
                        bAdm = False
                        return bAdm
        return bAdm

    # Evaluate_solution()
    # -------------------------------------------------------------------------------------------------------------
    def evaluate_solution(self, solution, feedback=None):
        """
        Calculate the fitness of the problem solution for the specific parameters
        """

        problem = self._problem

        problem_data = {
            "Decision-Variables": problem.decision_variables,
            "Constraints": problem.constraints,
        }

        params = {}
        params.update({"Population-Size": solution.representation[0]})
        params.update({"Number-of-Generations": solution.representation[1]})
        params.update({"Crossover-Probability": solution.representation[2] / 10})
        params.update({"Mutation-Probability": solution.representation[3] / 10})
        params.update(
            {"Initialization-Approach": transl_init[solution.representation[4]]}
        )
        parent_selection = transl_select[solution.representation[5]]()
        params.update({"Selection-Approach": parent_selection.select})
        params.update({"Tournament-Size": solution.representation[6]})
        params.update({"Crossover-Approach": transl_cross[solution.representation[7]]})
        params.update({"Mutation-Aproach": transl_mut[solution.representation[8]]})
        params.update({"Replacement-Approach": transl_repl[solution.representation[9]]})

        # algo
        algo = GeneticAlgorithm

        algo_data = {"Params": params}

        case_data = {
            "Num-Runs": 10,
            "Observer": GeneticAlgorithmObserver,
            "Plot": False,
            "Consolidate": False,
        }

        cp = CaseProblem(problem, problem_data, algo, algo_data, case_data)
        print(solution.representation)
        best_fit = cp.run()
        # best_fit is a matrix where the rows matches the number of runs and columns matches the number of generations
        best_fit_avg = np.array(best_fit).mean(0)
        if self._objective == ProblemObjective.MultiObjective:
            fitness = []
            # First objective is the fitness of the average of the last generations
            fitness.append(best_fit_avg[-1])
            # Second objective is the ratio of growth of the fitness
            ratio = abs((best_fit_avg[-1] - best_fit_avg[0]) / best_fit_avg[0])
            fitness.append(ratio)
        else:
            fitness = best_fit_avg[-1]

        solution.fitness = fitness
        return solution

    def is_multi_objective(self):
        return len(self._objective_function_list) > 1
