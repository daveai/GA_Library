# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------------------
"""
Problem Template
----------------
Content:

▶ class ProblemTemplate

─────────────────────────────────────────────────────────────────────────

CIFO - Computation Intelligence for Optimization

Author: Fernando A J Peres - fperes@novaims.unl.pt - (2019) version L4.0

"""
# -------------------------------------------------------------------------------------------------


from random import choice
from copy import deepcopy, copy
from cifo.problem.solution import LinearSolution, Encoding
from cifo.problem.objective import ProblemObjective

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Class: Problem Template Base Class (Super-class)
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
class ProblemTemplate:
    """
    Problem Template Class
    Remark: It should be seen as an abstract class, so please do not instantiate this directly
    """

    # Constructor
    # -------------------------------------------------------------------------------------------------------------
    def __init__(self, decision_variables, constraints={}, encoding_rule={}):
        """
        The constructor of the Problem Template. 
        """
        self._name = "Problem Template"
        self._decision_variables = decision_variables
        self._constraints = constraints
        self._encoding_rule = encoding_rule
        # self._objective_function_list   = []
        # self._objective_list            = []
        self._objective = ProblemObjective.Maximization

        # parse the encoding rule # read the encoding rules
        # member variables
        self._encoding = Encoding(encoding_rule)

    @property
    def name(self):
        return self._name

    # Decision Variables
    # -------------------------------------------------------------------------------------------------------------
    @property
    def decision_variables(self):
        "Problem Decision Variables"
        return self._decision_variables

    # Constraints
    # -------------------------------------------------------------------------------------------------------------
    @property
    def constraints(self):
        "Problem Constraints"
        return self._constraints

    # Encoding Rule
    # -------------------------------------------------------------------------------------------------------------
    @property
    def encoding_rule(self):
        """
        Encoding Rule (Python Dictionary) used to create the Encoding Object
        """
        return self._encoding_rule

    # Encoding
    # -------------------------------------------------------------------------------------------------------------
    @property
    def encoding(self):
        """
        Solution Encoding Object used by the problem
        """
        return self._encoding

    @encoding.setter
    def encoding(self, encoding_rule):
        self._encoding = Encoding(encoding_rule)

    # Build Solution Function - build_solution()
    # -------------------------------------------------------------------------------------------------------------
    # It should be seen as an abstract method
    def build_solution(
        self,
    ):  # << use this signature in the sub classes, the meta-heuristics
        print(
            " build_solution - It is an abstract method! Must be extended / implemented in the child class"
        )
        return None

    # Solution Admissibility Function - is_admissible()
    # -------------------------------------------------------------------------------------------------------------
    def is_admissible(
        self, solution
    ):  # << use this signature in the sub classes, the meta-heuristics
        print(
            " is_admissible - It is an abstract method! Must be extended / implemented in the child class"
        )
        pass

    # Evaluate_solution()
    # -------------------------------------------------------------------------------------------------------------
    # It should be seen as an abstract method
    def evaluate_solution(
        self, solution, feedback=None
    ):  # << This method does not need to be extended, it already automated solutions evaluation, for Single-Objective and for Multi-Objective
        """
        For the current solution, run all objective functions (objective function list) added in a specific problem and store these fitness values calculated in the solution. Default is one Objective Function (Single-Objective Problem)  

        Remark:
        -------
        This method does not need to be extended, it already automated solutions evaluation, for Single-Objective and for Multi-Objective     
        """
        ###for objective_function in self._objective_function_list:
        ###    solution.reset_fitness()
        ###    fitness_value =  objective_function( solution = solution, decision_variables = self._decision_variables, feedback = feedback )
        ###    solution.fitness_list.append( fitness_value )
        ###return solution
        print(
            " is_admissible - It is an abstract method! Must be extended / implemented in the child class"
        )
        pass

    # This function exists to improve the speed of computations
    # Instead of using deepcopy, we limit the copy to the representaiton
    def fast_solution_copy(
        self, representation, feedback=None
    ):  # << This method does not need to be extended, it already automated solutions evaluation, for Single-Objective and for Multi-Objective
        encoding_data = self._encoding.encoding_data

        solution_copy = LinearSolution(
            representation=copy(representation), encoding_rule=self._encoding_rule
        )
        return solution_copy

    # Solution Admissibility Function - is_admissible()
    # -------------------------------------------------------------------------------------------------------------
    ###@property
    ###def is_multi_objective( self ):
    ###    """
    ###    Returns if the problem is multi-objective or not (False == Single Objective)
    ###    """
    ###    return (len(self._objective_function_list) > 1)

    # problem Objectives
    # -------------------------------------------------------------------------------------------------------------
    ###@property
    ###def objectives(self):
    ###    """
    ###    Returns the problem's list of objectives.
    ###    Sing-Objective Problem only will have one element ProblemObjective.Minimization or ProblemObjective.Maximization
    ###    Multi-objective Problem will have a list of objectives, e.g.: [ProblemObjective.Minimization , ProblemObjective.Maximization, ProblemObjective.Minimization  ]
    ###
    ###    REMARK:
    ###    -------
    ###    The order is important, the objective function list and objective list are linked to the same objective. So they must be in the same position in both lists. In this way, the meta-heuristic will know that the first objective is maximization, so the first objective function will be related to this objective. The solution fitness list also will follow this same order. The first fitness value will be measured by maximization. Consequently, in this case, the meta-heuristic when need to select a set of solutions based on the first fitness must select the highest fitness value, once it is a maximization.
    ###    """
    ###    return self._objective_list

    # problem Objective (for single-objective Problem)
    # -------------------------------------------------------------------------------------------------------------
    @property
    def objective(self):
        """
        It returns the first (or unique) objective of the objective list. It is useful for Single-Objective Problems
        """
        return self._objective
