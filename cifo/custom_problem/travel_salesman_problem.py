from copy import deepcopy, copy
from random import choice, sample, randint, shuffle

from cifo.problem.problem_template import ProblemTemplate
from cifo.problem.objective import ProblemObjective
from cifo.problem.solution import LinearSolution, Encoding

tsp_encoding_rule = {
    "Size": -1,  # It must be defined by the size of DV (Number of products)
    "Is ordered": True,
    "Can repeat": False,
    "Data": [0, 0],  # must be defined by the data
    "Data Type": "Choices",
}


# REMARK: There is no constraint

# -------------------------------------------------------------------------------------------------
# TSP - Travel Salesman Problem
# -------------------------------------------------------------------------------------------------
class TravelSalesmanProblem(ProblemTemplate):
    """
    """

    # Constructor
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self, decision_variables, constraints, encoding_rule=tsp_encoding_rule
    ):
        """
        """
        # optimize the access to the decision variables
        self._values = []
        if "Nodes" in decision_variables:
            self._values = decision_variables["Nodes"]

        if "Distances" in decision_variables:
            self._distances = decision_variables["Distances"]
        elif "Coordinates" in decision_variables:
            self._distances = coordsToDist(decision_variables["Coordinates"])

        if encoding_rule["Size"] == -1:
            encoding_rule["Size"] = len(self._values)

        if encoding_rule["Data"] == [0, 0]:
            encoding_rule["Data"] = self._values

        # Call the Parent-class constructor to store these values and to execute  any other logic to be implemented by the constructor of the super-class
        super().__init__(
            decision_variables=decision_variables,
            constraints=constraints,
            encoding_rule=encoding_rule,
        )

        # 1. Define the Name of the Problem
        self._name = "Travel Salesman Problem"

        # 2. Define the Problem Objective
        self._objective = ProblemObjective.Minimization

    # Build Solution for TSP
    # ----------------------------------------------------------------------------------------------
    def build_solution(self):
        """
         Builds a linear solution for TSP with a specific order of cities
       """
        solution_representation = []
        encoding_data = self._encoding.encoding_data
        # Choose an initial random city
        solution_representation.append(randint(1, len(encoding_data)))
        # Build a solution by always appending the closest next city
        for i in range(len(encoding_data)):
            dists = [x for x in self._distances[solution_representation[-1] - 1]]
            sorDist = sorted(dists)
            for j in sorDist:
                if (dists.index(j) + 1) not in solution_representation:
                    solution_representation.append(dists.index(j) + 1)
                    break

        solution = LinearSolution(
            representation=solution_representation, encoding_rule=self._encoding_rule
        )
        return solution

        # solution_representation = []
        # encoding_data = self._encoding.encoding_data
        # solution_representation = sample(
        #     range(1, len(encoding_data) + 1), len(encoding_data)
        # )
        # solution = LinearSolution(
        #     representation=solution_representation, encoding_rule=self._encoding_rule
        # )
        # return solution

    # Solution Admissibility Function - is_admissible()
    # ----------------------------------------------------------------------------------------------
    def is_admissible(
        self, solution
    ):  # << use this signature in the sub classes, the meta-heuristic
        """
        The only constraint is that cities cannot be repeated. This is taken into account by the algorithm
        So there is no need to check admissibility
        """
        return True

    # Evaluate_solution()
    # -------------------------------------------------------------------------------------------------------------
    # It should be seen as an abstract method
    def evaluate_solution(
        self, solution, feedback=None
    ):  # << This method does not need to be extended, it already automated solutions evaluation, for Single-Objective and for Multi-Objective
        """
        Calculate the distances of the nodes based on the orders
        """

        fitness = 0

        for i in range(1, len(solution.representation)):
            fitness += self._distances[solution.representation[i - 1] - 1][
                solution.representation[i] - 1
            ]

        fitness += self._distances[solution.representation[-1] - 1][
            solution.representation[0] - 1
        ]
        solution.fitness = fitness

        return solution

    # Is_multi_objective()
    # -------------------------------------------------------------------------------------------------------------
    # It should return if multi obj or not
    def is_multi_objective(self):
        return False

    # -------------------------------------------------------------------------------------------------
    # OPTIONAL - it is only needed if you will implement Local Search Methods
    #            (Hill Climbing and Simulated Annealing)
    # -------------------------------------------------------------------------------------------------
    def get_neighbors(self, solution, problem, neighborhood_size=20):

        neighbors = []

        tour = solution.representation

        # Neighbourhood function inspired by Lin-N-Opt, take edges,
        # and create new neighbour by inverting one side and swapping them
        # quicker and better performing than simple exchange neighbour

        # We create a new solution with the LinearSolution template as
        # deepcopy is very computationally intensive. This performs over
        # 20% quicker than deepcopying
        iters = min(round(len(tour) / 2), 100)
        indexes = sample([i for i in range(2, len(tour))], iters)
        j = 3

        for i in indexes:
            # n = tour[:]
            n1 = tour[:]
            # n[:i], n[-i:] = tour[-i:][::-1], tour[:i]
            n1[i - j : i], n1[-i : -i + j] = tour[-i : -i + j][::-1], tour[i - j : i]
            # neigh = self.fast_solution_copy(n)
            neigh1 = self.fast_solution_copy(n1)

            # neighbors.append(neigh)
            neighbors.append(neigh1)

        # for i in range(0, len(solution.representation) - 1):
        #     n = deepcopy(solution)
        #     n.representation[i], n.representation[i+1] = n.representation[i+1], n.representation[i]
        #     neighbors.append(n)

        return neighbors

    def get_single_neighbor(self, solution, problem):
        i = randint(0, len(solution.representation))
        j = 3

        n = solution.representation[:]
        n[i - j : i] = n[i - j : i][::-1]

        neigh = self.fast_solution_copy(n)

        return neigh


# Util function to transform a table of coords into a matrix of distances
def coordsToDist(coords):
    from timeit import default_timer as timer

    start = timer()
    """
    Util function to transform a table of coords into a matrix of distances
    :param coords: list of N rows and 2 columns (x and y)
    :return: matrix of NxN distances
    """
    from scipy.spatial.distance import squareform, pdist
    import numpy

    coordinates_array = numpy.array(coords)
    dist_array = pdist(coordinates_array)
    return squareform(dist_array)
