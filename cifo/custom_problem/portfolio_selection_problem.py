from copy import deepcopy
from random import choice, randint, random, shuffle, sample
import statistics

from cifo.problem.problem_template import ProblemTemplate
from cifo.problem.objective import ProblemObjective
from cifo.problem.solution import LinearSolution, Encoding


# -------------------------------------------------------------------------------------------------
# PIP - Portfolio Investment Problem
# -------------------------------------------------------------------------------------------------
class PortfolioSelectionProblem(ProblemTemplate):
    """
    """

    # Constructor
    # ----------------------------------------------------------------------------------------------
    def __init__(self, decision_variables, constraints, encoding_rule):
        """
        """
        # optimize the access to the decision variables
        # optimize the access to the decision variables
        self._prices = []
        if "Price" in decision_variables:
            self._prices = decision_variables["Price"]

        self._exp_ret_3m = []
        if "Exp-return-3m" in decision_variables:
            self._exp_ret_3m = decision_variables["Exp-return-3m"]

        self._std_dev = []
        if "Std-dev" in decision_variables:
            self._std_dev = decision_variables["Std-dev"]

        self._company_names = []
        if "Company" in decision_variables:
            self._company_names = decision_variables["Company"]

        if encoding_rule["Size"] == -1:
            encoding_rule["Size"] = len(self._prices)

        # Call the Parent-class constructor to store these values and to execute  any other logic to be implemented by the constructor of the super-class
        super().__init__(
            decision_variables=decision_variables,
            constraints=constraints,
            encoding_rule=encoding_rule,
        )

        # 1. Define the Name of the Problem
        self._name = "Portfolio Selection Problem"

        # 2. Define the Problem Objective
        self._objective = ProblemObjective.Maximization

    # Build Solution for PI Problem
    # ----------------------------------------------------------------------------------------------
    def build_solution(self):
        """
        """

        solution_representation = sample(
            self._encoding.encoding_data, self._encoding.size
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
        We encode constraints into the fitness fucntion, to penalize solutions which do not comply with 
        our contraints.
        """
        return True

    # Evaluate_solution()
    # -------------------------------------------------------------------------------------------------------------
    # It should be seen as an abstract method
    def evaluate_solution(
        self, solution, feedback=None
    ):  # << This method does not need to be extended, it already automated solutions evaluation, for Single-Objective and for Multi-Objective
        """
        The evaluation is made only based on the expected return
        """
        sharpeSafe = 1.57
        fitness = 0
        ## Monetary return
        perReturn = self.portfolio_return(solution)
        money = self._decision_variables["Budget"] / self._decision_variables["Size"]
        monReturn = sum([money / 100 * x for x in perReturn])

        ## Calc Sharpe Ratio
        sharpe = (
            sum([x - sharpeSafe for x in perReturn])
            / len(perReturn)
            / self.std_portfolio_return(solution)
        )

        ## Multiply together = Fitness
        solution.fitness = monReturn * sharpe
        return solution

    # -------------------------------------------------------------------------------------------------
    # OPTIONAL - it onlu+y is needed if you will implement Local Search Methods
    #            (Hill Climbing and Simulated Annealing)
    # -------------------------------------------------------------------------------------------------
    def get_neighbors(self, solution, problem, neighborhood_size=0):

        neighbors = []

        folio = solution.representation

        for i in range(solution._encoding.size):
            n0 = folio[:]
            n1 = folio[:]
            n0[i] = (n0[i] + 1) % len(self._encoding._encoding_data)
            n1[i] = n1[i] - 1
            n0 = self.fast_solution_copy(n0)
            n1 = self.fast_solution_copy(n1)
            neighbors.append(n0)
            neighbors.append(n1)

        return neighbors

    # Is_multi_objective()
    # -------------------------------------------------------------------------------------------------------------
    # It should return if multi obj or not
    def is_multi_objective(self):
        return False

    def portfolio_return(self, solution):
        pr = [self._exp_ret_3m[x] for x in solution.representation]
        return pr

    def mean_portfolio_return(self, solution):
        pr = self.portfolio_return(solution)
        mpr = sum(pr) / len(pr)
        return mpr

    def std_portfolio_return(self, solution):
        pr = self.portfolio_return(solution)
        # spr = statistics.stdev(pr)
        spr = statistics.stdev(pr)
        return spr
