# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------------------
"""
Hill Climbing Meta-Heuristic
----------------------------

Content: 

▶ class HillClimbing

─────────────────────────────────────────────────────────────────────────

CIFO - Computation Intelligence for Optimization

Author: Fernando A J Peres - fperes@novaims.unl.pt - (2019) version L4.0
"""
# -------------------------------------------------------------------------------------------------

from cifo.problem.problem_template import ProblemTemplate
from cifo.problem.objective import ProblemObjective
from cifo.util.observer import LocalSearchMessage


class HillClimbing:
    """
    Classic Implementation of Hill Climbing with some improvements.

    Improvements:
    ------------
    1. Stop-Conditions flexibility

    Algorithm:
    ---------
    1: Initialize
    2: Repeat while exists a better neighbor
        2.1: Get the best neighbor
        2.2: Select the best, between the current best and best neighbor    
        2.3: Check stop conditions 
    3: Return the best solution
    """

    # Constructor
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self, problem_instance, neighborhood_function, feedback=None, params={}
    ):
        """
        Hill Climbing Constructor

        Parameters:
        -----------
        ▶ problem_instance     - the instance of the problem that the algorithm will search a solution

        ▶ neighborhood_function - it is expected a function that must follow the signature:
           
            neighborhood_function( solution, problem, neighborhood_size = 0 )
        
        ▶ feedback 

        ▶ params - dictionary with configurations for Hill Climbing
            e.g.: params = { "Maximum-Iterations" : 100 , "Stop-Conditions" : "Classical", "Neighborhood-Size": -1}

            A. "Maximum-Iterations" - the number of maximum iterations (used to stop the search, even there are neighbors better than the current solution)
            
            B. "Stop-Condition" - The approach used to stop conditions
            Possible "Search-Methods" : ["Classical", "Alternative-01"]
            1. Classical - Stops, when there is no better neighbor or the number of max iterations was achieved.
            2. Alternative 1 - Stops when the number of max iterations was achieved. It can be good when the neighborhood can be different for the same solution   

            C. "Neighborhood-Size" the size of the neighborhood, the default is -1, which means the neighborhood will return all neighbors found
        """
        # set
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        self._name = "Hill Climbing"
        self._description = ""
        self._problem_instance = problem_instance
        self._get_neighbors = neighborhood_function
        self._feedback = feedback
        self._observers = []
        self._iteration = 0
        self._solution = None
        self._neighbor = None

        # parse params
        # Default params:
        # {"Maximum-Iterations": 10, "Stop-Conditions":"Classical", "Target-Fitness" : None, "Neighborhood-Size" = 0 }
        # Motivation: Enables the user to change some Hill Climbing Behaviors
        # (Flexibility)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # max iterations
        self._max_iterations = 10
        if "Maximum-Iterations" in params:
            self._max_iterations = params["Maximum-Iterations"]

        # stop condition approach
        self._stop_condition_approach = "Classical"
        self._check_stop_conditions = self._check_classical_stop_conditions
        if "Stop-Conditions" in params:
            self._stop_condition_approach = params["Stop-Conditions"]
        if self._stop_condition_approach == "Alternative-01":
            self._check_stop_conditions = self._check_alternative1_stop_conditions

        self._target_fitness = None
        if "Target-Fitness" in params:
            self._target_fitness = params["Target-Fitness"]

        self._description = f"Maximum-Iterations: {self._max_iterations} | Stop-Condition: {self._stop_condition_approach} "

        # neighborhood size
        self._neighborhood_size = 0
        if "Neighborhood-Size" in params:
            self._neighborhood_size = params["Neighborhood-Size"]

        # Prepare the internal methods for multi-objective / single-objective:
        # Motivation: Avoid in each selection step check if it is multi-single or min/max
        # (Optimization)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if self._problem_instance.is_multi_objective():
            # Multi-objective
            print("NOT IMPLEMENTED.")
        else:
            # Single-Objective
            if self._problem_instance.objective == ProblemObjective.Maximization:
                self._get_best_neighbor = self._get_best_neighbor_maximization
                self._select = self._select_maximization
            else:
                self._get_best_neighbor = self._get_best_neighbor_minimization
                self._select = self._select_minimization

    # Search Method
    # ----------------------------------------------------------------------------------------------
    def search(self):
        """
        Hill ClimbingSearch Method
        --------------------------

        Algorithm:
        ---------

        1: Initialize
        2: Repeat while exists a better neighbor
            2.1: Get the best neighbor
            2.2: Select the best, between the current best and best neighbor    
            2.3: Check stop conditions 
        3: Return the best solution

        """
        self._notify(message=LocalSearchMessage.Started)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        searching = True
        self._solution = None
        self._neighbor = None

        # Search
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # 1. Initialize
        self._initialize()
        self._notify(message=LocalSearchMessage.Initialized)

        # 2. Repeat while exists a better neighbor
        self._iteration = 0
        while searching:
            self._iteration += 1
            # 2.1: Get the best neighbor
            if self._problem_instance.objective == ProblemObjective.Maximization:
                self._get_best_neighbor = self._get_best_neighbor_maximization()
            elif self._problem_instance.objective == ProblemObjective.Minimization:
                self._get_best_neighbor = self._get_best_neighbor_minimization()

            # 2.2: Select the best, between the current best and best neighbor
            if self._problem_instance.objective == ProblemObjective.Maximization:
                changed = self._select_maximization()
            elif self._problem_instance.objective == ProblemObjective.Minimization:
                changed = self._select_minimization()
            # 2.3: Check stop conditions
            searching = self._check_stop_conditions(changed)

        # 3: Return the best solution
        self._notify(message="FINISHED")
        return self._solution

    # Initialize:  create an initial solution
    # ----------------------------------------------------------------------------------------------
    def _initialize(self):
        """
        Create a feasible initial solution
        """
        self._solution = self._problem_instance.build_solution()

        while not self._problem_instance.is_admissible(self._solution):
            self._solution = self._problem_instance.build_solution()

        self._problem_instance.evaluate_solution(
            self._solution, feedback=self._feedback
        )

    # _get_best_neighbor for maximization
    # ----------------------------------------------------------------------------------------------
    def _get_best_neighbor_maximization(self):
        """
        Get the best neighbor of the neighborhood : MAXIMIZATION
        """
        # Get Neighbors of the current solution
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        neighborhood = self._get_neighbors(
            solution=self._solution,
            problem=self._problem_instance,
            neighborhood_size=self._neighborhood_size,
        )

        best_neighbor = None

        # Find the best neighbor in neighborhood of the current solution
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        for neighbor in neighborhood:

            if self._problem_instance.is_admissible(neighbor):
                self._problem_instance.evaluate_solution(
                    solution=neighbor, feedback=self._feedback
                )

                if best_neighbor == None:
                    best_neighbor = neighbor
                else:
                    if neighbor.fitness >= best_neighbor.fitness:
                        best_neighbor = neighbor

        self._neighbor = best_neighbor
        return best_neighbor

    # _get_best_neighbor for minimization
    # ----------------------------------------------------------------------------------------------
    def _get_best_neighbor_minimization(self):
        """
        Get the best neighbor of the neighborhood : MINIMIZATION
        """
        # Get Neighbors of the current solution
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        neighborhood = self._get_neighbors(
            solution=self._solution,
            problem=self._problem_instance,
            neighborhood_size=self._neighborhood_size,
        )

        best_neighbor = None

        # Find the best neighbor in neighborhood of the current solution
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        for neighbor in neighborhood:
            self._problem_instance.evaluate_solution(
                solution=neighbor, feedback=self._feedback
            )

            if best_neighbor == None:
                best_neighbor = neighbor

            elif neighbor.fitness <= best_neighbor.fitness:
                best_neighbor = neighbor

        self._neighbor = best_neighbor
        return best_neighbor

    # _select for minimization
    # ----------------------------------------------------------------------------------------------
    def _select_minimization(self):
        """
        Select the better solution : MINIMIZATION
        
        Returns: 
        - solution: The best between them (solution and neighbor)
        - boolean : If changed the current, returns True else returns False
        """
        if self._neighbor.fitness <= self._solution.fitness:
            self._solution = self._neighbor
            self._notify(message=LocalSearchMessage.ReplacementAccepted)
            return True

        self._notify(message=LocalSearchMessage.ReplacementRejected)
        return False

    # _select for maximization
    # ----------------------------------------------------------------------------------------------
    def _select_maximization(self):
        """
        Select the better solution : MAXIMIZATION
                
        Returns: 
        - solution: The best between them (solution and neighbor)
        - boolean : If changed the current, returns True else returns False
        """
        if self._neighbor.fitness >= self._solution.fitness:
            self._solution = self._neighbor
            self._notify(message=LocalSearchMessage.ReplacementAccepted)
            return True

        self._notify(message=LocalSearchMessage.ReplacementRejected)
        return False

    # _check_classical_stop_conditions
    # ----------------------------------------------------------------------------------------------
    def _check_classical_stop_conditions(self, changed):
        """
        Classical - Stops, when there is no better neighbor or the number of max iterations was achieved.
        """
        searching = changed and (self._iteration < self._max_iterations)
        if not changed:
            self._notify(message=LocalSearchMessage.StoppedPrematurely)
        elif not searching:
            self._notify(message=LocalSearchMessage.Stopped)
        elif self._target_fitness:
            if self._solution.fitness >= self._target_fitness:
                self._notify(message=LocalSearchMessage.StoppedTargetAchieved)
                return False
        return searching

    # _select for maximization
    # ----------------------------------------------------------------------------------------------
    def _check_alternative1_stop_conditions(self, changed):
        """
        Alternative 1 - Stops when the number of max iterations was achieved. It can be good when the neighborhood can be different for the same solution   
        """
        searching = self._iteration < self._max_iterations
        if not searching:
            self._notify(message=LocalSearchMessage.Stopped)
        elif self._target_fitness:
            if self._solution.fitness >= self._target_fitness:
                self._notify(message=LocalSearchMessage.StoppedTargetAchieved)
                return False
        return searching

    # Problem Instance
    # ----------------------------------------------------------------------------------------------
    @property
    def problem(self):
        return self._problem_instance

    # Solution
    # ----------------------------------------------------------------------------------------------
    @property
    def solution(self):
        return self._solution

    # Name
    # ----------------------------------------------------------------------------------------------
    @property
    def name(self):
        return self._name

    # Description
    # ----------------------------------------------------------------------------------------------
    @property
    def description(self):
        return self._description

    # ----------------------------------------------------------------------------------------------
    # Observable Interface
    # ----------------------------------------------------------------------------------------------

    # _select for maximization
    # ----------------------------------------------------------------------------------------------
    def get_state(self):
        return self._state

    # _select for maximization
    # ----------------------------------------------------------------------------------------------
    def register_observer(self, observer):
        self._observers.append(observer)

    # _select for maximization
    # ----------------------------------------------------------------------------------------------
    def unregister_observer(self, observer):
        self._observers.remove(observer)

    # _select for maximization
    # ----------------------------------------------------------------------------------------------
    def _notify(self, message, neighbors=None):

        self._state = {
            "iteration": self._iteration,
            "message": message,
            "neighbor": self._neighbor,
            "neighbors": neighbors,
        }

        for observer in self._observers:
            observer.update()
