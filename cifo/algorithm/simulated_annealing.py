# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------------------
"""
Simulated Annealing Meta-Heuristic
----------------------------------

Content: 

▶ class Simulated Annealing

─────────────────────────────────────────────────────────────────────────

CIFO - Computation Intelligence for Optimization

Author: Fernando A J Peres - fperes@novaims.unl.pt - (2019) version L4.0
"""
# -------------------------------------------------------------------------------------------------

# /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
# C O D E
# /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
import math
from copy import deepcopy
from random import choice, uniform

# -------------------------------------------------------------------------------------------------
# Class: Simulated Annealing
# -------------------------------------------------------------------------------------------------
from cifo.problem.objective import ProblemObjective


class SimulatedAnnealing:
    """
    Classic Implementation of Simulated Annealing with some improvements.

    Improvements:
    ------------
    1. Memory - avoiding to lose the best
    2. C / Minimum C Calibration

    Algorithm:
    ---------
    1: Initialize
    2: Repeat while Control Parameter >= Minimum Control Parameter 
        2.1. Internal Looping
            2.1.1: Get the best neighbor
            2.1.2: Select the best, between the current best and current best neighbor
            2.1.3: Check stop condition ***
        2.2. Update C (Control Parameter)
        2.3: Check stop condition ***
    3: Return the Solution
    """

    # Constructor
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self, problem_instance, neighborhood_function, feedback=None, config={}
    ):

        """
        Simulated Annealing Constructor

        parameters:
        -----------
        * neighborhood_function - it is expected a function that must follow the signature:
           
        <<neighborhood_function>>( solution, problem, neighborhood_size = 0 )
            
        where:
        - <neighborhood_function>  it is the name of the neighborhood function implemented for the problem_instance
        """
        self._problem_instance = problem_instance
        self._get_neighbors = neighborhood_function
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

        # memory (to avoid lost the best)
        self._best_solution = None

    # Search
    # ----------------------------------------------------------------------------------------------
    def search(self):
        """
        Simulated Annealing Search Method
        ----------------------------------

        Algorithm:

        1: Initialize
        2: Repeat while Control Parameter >= Minimum Control Parameter 
            2.1. Internal Looping
                2.1.1: Get the best neighbor
                2.1.2: Select the best, between the current best and current best neighbor
                2.1.3: Check stop condition ***
            2.2. Update C (Control Parameter)
            2.3: Check stop condition ***
        3: Return the Solution

        """
        # Step 1: Initialize
        self._initialize()
        # exist_best_neighbor = True

        # C and minimum C
        # ----------------------
        C, MIN_C = self._initialize_C()

        i = 0
        j = 0
        best_solution = self._best_solution

        # Step 2: Repeat while Control Parameter >= Minimum Control Parameter
        while C >= MIN_C:
            j += 1
            # 2.1. Internal Looping
            for _ in range(0, self._internal_loop_iterations):
                i += 1

                # step 2.1.1: get the best neighbor
                # step 2.1.2: select the best, between the current best and current best neighbor
                best_solution = self._select(C)
                self._best_solution = best_solution
                # step 2.1.3: stop condition
                if i > self._max_iterations:
                    break

            if i > self._max_iterations:
                break
        # step 3
        return best_solution

    # Constructor
    # ----------------------------------------------------------------------------------------------
    def _initialize(self):
        """
        Initialize the initial solution, start C and Minimum C
        """
        self._solution = self._problem_instance.build_solution()

        while not self._problem_instance.is_admissible(self._solution):
            self._solution = self._problem_instance.build_solution()

        self._problem_instance.evaluate_solution(
            self._solution, feedback=self._feedback
        )

        self._best_solution = self._solution

    # Constructor
    # ----------------------------------------------------------------------------------------------
    def _select(self, C):
        """
        Select the solution for the next iteration
        """
        # step 2.1.1: get the best neighbor
        best_solution = self._best_solution
        neighbours = self._get_neighbors(
            solution=self._best_solution, problem=self._problem_instance
        )

        for best_neighbor in neighbours:
            fitness_best_neighbor = self._problem_instance.evaluate_solution(
                best_neighbor
            ).fitness
            fitness_best_solution = best_solution.fitness
            bProbCheck = True
            # maximization
            if self._problem_instance.objective == ProblemObjective.Maximization:
                if fitness_best_neighbor >= fitness_best_solution:
                    # replacement
                    best_solution = best_neighbor
                    fitness_best_solution = fitness_best_neighbor
                    bProbCheck = False
            # minimization
            elif self._problem_instance.objective == ProblemObjective.Minimization:
                if fitness_best_neighbor <= fitness_best_solution:
                    best_solution = best_neighbor
                    bProbCheck = False

            # probability of accepting worse
            if bProbCheck:
                # Calculate the probability of acceptance and check if it can be accepted.
                p = math.exp(-abs(fitness_best_neighbor - fitness_best_solution) / C)
                random_num = uniform(0, 1)
                if random_num <= p:
                    # Replace if it was accepted
                    best_solution = best_neighbor
                    C = self._update_C(C)

        return best_solution

    # Constructor
    # ----------------------------------------------------------------------------------------------
    def _get_random_neighbor(self, solution):
        """
        Get a random neighbor of the neighborhood (internally it will call the neighborhood provided to the algorithm)
        """
        # get neighborhood of current solution
        neighborhood = self._get_neighbors(
            solution=self._best_solution, problem=self._problem_instance
        )

        # get one randomly
        neighbor = choice(neighborhood)
        neighbor = self._problem_instance.evaluate_solution(neighbor)
        # return this neighbor
        return neighbor

        # Constructor

    # ----------------------------------------------------------------------------------------------
    def _initialize_C(self):
        """
        Use one of the available approaches to initialize C and Minimum C
        """
        C = 0
        MIN_C = 0
        k = 0.5
        kMin = 0.1
        cMethod = self._cMethod
        mincMethod = self._mincMethod

        if cMethod >= 2 or mincMethod >= 3:
            # Random walk methods. Performs a random walk
            # Length of the random walk
            lenRW = 10
            lFit = []
            # Get initial solution
            solution = self._problem_instance.build_solution
            first_solution = deepcopy(solution)
            lFit.append(self._problem_instance.evaluate_solution(solution=solution))
            for i in range(0, lenRW):
                # Get a random neighbour
                solution = self._get_random_neighbor(solution=solution)
                lFit.append(self._problem_instance.evaluate_solution(solution=solution))
            # Get the max, min and average of delta
            lDelta = [
                abs(number)
                for number in [x - lFit[i - 1] for i, x in enumerate(lFit)][1:]
            ]
            maxDelta = max(lDelta)
            minDelta = max(min(lDelta), 0.1)
            avgDelta = sum(lDelta) / len(lDelta)

        if cMethod == 1:
            # IT1 - fixed initial value
            C = 20
        elif cMethod == 2:
            # IT2 - function of initial candidate
            first_solution = self._problem_instance.evaluate_solution(
                solution=first_solution
            )
            C = k * first_solution.fitness
        elif cMethod == 3:
            # IT3 - random-walk: difference between steps
            C = k * maxDelta
        elif cMethod == 4:
            # IT4 - random walk: avergage
            C = k * avgDelta

        # minimum C - SC3
        if mincMethod == 1:
            # fixed final value
            MIN_C = min(0.1, C)
        elif mincMethod == 2:
            # function of initial temperature
            MIN_C = kMin * C
        elif mincMethod == 3:
            # IT3
            C = k * minDelta

        return C, MIN_C

    def _update_C(self, C0):
        """
        Update the parameter C
        """
        alfa = 0.95
        beta = 0.9
        C1 = 0
        if self._rate_method == 1:
            # CS1 - fixed final value
            C1 = alfa * beta ** C0
        elif self._rate_method == 2:
            # CS2 - fixed final value
            C1 = alfa * C0

        return C1
