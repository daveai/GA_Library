# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------------------
"""
Threshold Acceptance Meta-Heuristic
----------------------------------

Content: 

▶ class Threshold Acceptance (Dueck and Scheuer, 1990)

─────────────────────────────────────────────────────────────────────────

CIFO - Computation Intelligence for Optimization

"""
# -------------------------------------------------------------------------------------------------

# /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
# C O D E
# /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
import math
from copy import deepcopy
from random import choice, uniform

# -------------------------------------------------------------------------------------------------
# Class: Threshold Acceptance
# -------------------------------------------------------------------------------------------------
from cifo.problem.objective import ProblemObjective


class ThresholdAcceptance:
    """
    Classic Implementation of Simulated Annealing with some improvements.

    Improvements:
    ------------
    Accept all neighbours that are better, and those which are worse within
    a certain threshold.  The threshold is lowered periodically, as in SA.
    """

    # Constructor
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self, problem_instance, neighborhood_function, feedback=None, config={}
    ):

        """
        Threshold Acceptance Constructor

        parameters:
        -----------
        * neighborhood_function - it is expected a function that must follow the signature:
           
        <<neighborhood_function>>( solution, problem, neighborhood_size = 0 )
            
        where:
        - <neighborhood_function>  it is the name of the neighborhood function implemented for the problem_instance
        """
        self._problem_instance = problem_instance
        self._get_neighbor = neighborhood_function
        self._feedback = feedback

        self._cMethod = 1
        if "C-Method" in config:
            self._cMethod = config["C-Method"]

        self._mincMethod = 1
        if "Min-C-Method" in config:
            self._mincMethod = config["Min-C-Method"]

        self._internal_loop_iterations = 20
        if "Internal-Loop-Iterations" in config:
            self._internal_loop_iterations = config["Internal-Loop-Iterations"]

        self._max_iterations = 50
        if "Max-Iterations" in config:
            self._max_iterations = config["Max-Iterations"]

        self._rate_method = 2
        if "Rate-Method" in config:
            self._rate_method = config["Rate-Method"]

        # memory (to avoid loosing the best)
        self._best_solution = None

    # Search
    # ----------------------------------------------------------------------------------------------
    def search(self):
        """
        Threshold Acceptance Search Method
        ----------------------------------

        Algorithm:

        1: Initialize
        2: Repeat while Control Parameter >= Minimum Control Parameter 
            2.1. Internal Looping
                2.1.1: Get the best neighbor
                2.1.2: Select the best, between the current best and current best neighbor
                2.1.3: Check stop condition ***
            2.2. Update threshold 
            2.3: Check stop condition ***
        3: Return the Solution

        """
        # Step 1: Initialize
        self._initialize()
        # exist_best_neighbor = True

        # C and minimum C
        # ----------------------
        threshold = self._initialize_threshold()

        ta_solution = self._select(self._best_solution, threshold)

        # Step 2: Repeat while Control Parameter >= Minimum Control Parameter
        return ta_solution

    # Constructor
    # ----------------------------------------------------------------------------------------------
    def _initialize(self):
        """
        Initialize the initial solution
        """
        self._solution = self._problem_instance.build_solution()

        while not self._problem_instance.is_admissible(self._solution):
            self._solution = self._problem_instance.build_solution()

        self._problem_instance.evaluate_solution(
            self._solution, feedback=self._feedback
        )

        self._best_solution = self._solution

    def _select(self, solution, threshold):
        """
        Do a threshold acceptance run
        """
        solution_fit = self._problem_instance.evaluate_solution(solution).fitness
        best_solution = solution
        best_solution_fit = self._problem_instance.evaluate_solution(
            best_solution
        ).fitness

        for _ in range(self._max_iterations):
            for i in range(self._internal_loop_iterations):
                neigh = self._get_neighbor(solution, problem=self._problem_instance)
                neigh_fit = self._problem_instance.evaluate_solution(neigh).fitness
                thres = solution_fit * threshold
                if (solution_fit - neigh_fit) > (-thres):
                    solution = neigh
                    solution_fit = neigh_fit
                    # print(f"accepted: {solution_fit}")
                    if solution_fit < best_solution_fit:
                        best_solution = solution
                        best_solution_fit = solution_fit
            solution = best_solution
            solution_fit = best_solution_fit
            self._update_threshold(threshold)
        return best_solution

        # Constructor

    # ----------------------------------------------------------------------------------------------
    def _initialize_threshold(self):
        """
        Use one of the available approaches to initialize C and Minimum C
        """
        threshold = 0.15

        return threshold

    def _update_threshold(self, C):
        """
        Update the parameter C
        """
        alpha = 0.85
        threshold = C * alpha
        return threshold
