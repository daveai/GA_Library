from random import (
    uniform,
    randint,
    choice,
    sample,
    random,
    shuffle,
    sample,
    getrandbits,
)
from copy import deepcopy

from cifo.problem.objective import ProblemObjective
from cifo.problem.solution import EncodingDataType
from cifo.problem.population import Population

from cifo.algorithm.hill_climbing import HillClimbing
from cifo.algorithm.simulated_annealing import SimulatedAnnealing
from cifo.algorithm.threshold_acceptance import ThresholdAcceptance


###################################################################################################
# INITIALIZATION APPROACHES
###################################################################################################

# (!) REMARK:
# Initialization signature: <method_name>( problem, population_size ):

# -------------------------------------------------------------------------------------------------
# Random Initialization
# -------------------------------------------------------------------------------------------------
def initialize_randomly(problem, population_size):
    """
    Initialize a population of solutions (feasible solution) for an evolutionary algorithm
    
    Required:
    
    @ problem - problem's build solution function knows how to create an individual in accordance with the encoding.
    
    @ population_size - to define the size of the population to be returned. 
    """
    solution_list = []

    i = 0
    # generate a population of admissible solutions (individuals)
    for _ in range(0, population_size):
        s = problem.build_solution()

        # check if the solution is admissible
        while not problem.is_admissible(s):
            s = problem.build_solution()

        s.id = [0, i]
        i += 1
        problem.evaluate_solution(s)

        solution_list.append(s)

    # print([x.fitness for x in solution_list])

    population = Population(
        problem=problem, maximum_size=population_size, solution_list=solution_list
    )

    return population


# -------------------------------------------------------------------------------------------------
# Initialization using Hill Climbing
# -------------------------------------------------------------------------------------------------
def initialize_using_hc(problem, population_size):
    params = {
        "Maximum-Iterations": 200,
        "Stop-Conditions": "Classical",
        "Neighborhood-Size": -1,
    }

    hc = HillClimbing(
        problem_instance=problem,
        neighborhood_function=problem.get_neighbors,
        feedback=None,
        params=params,
    )
    solution_list = []
    for i in range(0, population_size):
        nn = hc.search()
        nn.id = [0, i]
        solution_list.append(nn)

    population = Population(
        problem=problem, maximum_size=population_size, solution_list=solution_list
    )

    return population


# -------------------------------------------------------------------------------------------------
# Initialization using Simulated Annealing
# -------------------------------------------------------------------------------------------------
def initialize_using_sa(problem, population_size):
    params = {"Internal-Loop-Iterations": 20, "Max-Iterations": 250}
    sa = SimulatedAnnealing(
        problem_instance=problem,
        neighborhood_function=problem.get_neighbors,
        feedback=None,
        config=params,
    )
    solution_list = []

    population = Population(
        problem=problem, maximum_size=population_size, solution_list=solution_list
    )

    return population


# -------------------------------------------------------------------------------------------------
# Initialization using Threshold Acceptance
# -------------------------------------------------------------------------------------------------
def initialize_using_ta(problem, population_size):
    params = {"Internal-Loop-Iterations": 50, "Max-Iterations": 800}
    ta = ThresholdAcceptance(
        problem_instance=problem,
        neighborhood_function=problem.get_single_neighbor,
        feedback=None,
        config=params,
    )
    solution_list = []
    for i in range(0, population_size):
        nn = ta.search()
        nn.id = [0, i]
        solution_list.append(nn)

    population = Population(
        problem=problem, maximum_size=population_size, solution_list=solution_list
    )

    return population


###################################################################################################
# SELECTION APPROACHES
###################################################################################################
# -------------------------------------------------------------------------------------------------
# class RouletteWheelSelection
# -------------------------------------------------------------------------------------------------
class RouletteWheelSelection:
    """
    Main idea: better individuals get higher chance
    The chances are proportional to the fitness
    Implementation: roulette wheel technique
    Assign to each individual a part of the roulette wheel
    Spin the wheel n times to select n individuals

    REMARK: This implementation does not consider minimization problem
    """

    def select(self, population, objective, params):
        """
        select two different parents using roulette wheel
        """
        index1 = self._select_index(population=population, objective=objective)
        index2 = index1

        while index2 == index1:
            index2 = self._select_index(population=population, objective=objective)

        return population.get(index1), population.get(index2)

    def _select_index(self, population, objective):

        # Get the Total Fitness (all solutions in the population) to calculate the chances proportional to fitness
        total_fitness = 0
        total_fitness_min = 0
        if objective == ProblemObjective.Maximization:
            for solution in population.solutions:
                total_fitness += solution.fitness
        elif objective == ProblemObjective.Minimization:
            # Checks if there are <0 fitness values
            count_non_pos = sum(
                1 for solution in population.solutions if solution.fitness <= 0
            )
            if count_non_pos > 0:
                # Creates a list with the fitness values
                lFit = []
            for solution in population.solutions:
                total_fitness_min += 1 / solution.fitness
                if count_non_pos > 0:
                    lFit.append(solution.fitness)

            # Need to compute min and max fitness and normalizes the fitness
            if count_non_pos > 0:
                # Normalizes the fitness list
                fit_min = min(lFit)
                fit_max = max(lFit)
                lFitNorm_inverse = [
                    1 / ((float(i) - fit_min) / (fit_max - fit_min) + 1 / fit_max)
                    for i in lFit
                ]
                sum_fit_norm_inverse = sum(lFitNorm_inverse)

        # spin the wheel
        wheel_position = uniform(0, 1)

        # calculate the position which wheel should stop
        stop_position = 0
        index = 0
        for solution in population.solutions:
            if objective == ProblemObjective.Maximization:
                stop_position += solution.fitness / total_fitness
            elif objective == ProblemObjective.Minimization:
                if count_non_pos > 0:
                    stop_position += lFitNorm_inverse[index] / sum(lFitNorm_inverse)
                else:
                    stop_position += 1 / solution.fitness / total_fitness_min
            if stop_position > wheel_position:
                break
            index += 1

        return index


# -------------------------------------------------------------------------------------------------
# class RankSelection
# -------------------------------------------------------------------------------------------------
class RankSelection:
    """
    Rank Selection sorts the population first according to fitness value and ranks them. Then every chromosome is allocated selection probability with respect to its rank. Individuals are selected as per their selection probability. Rank selection is an exploration technique of selection.
    """

    def select(self, population, objective, params):
        # Step 1: Sort / Rank
        sortedIndexes = self._sort(population, objective)

        # Step 2: Create a rank list [0, 1, 1, 2, 2, 2, ...]
        rank_list = []

        for index in sortedIndexes:
            for _ in range(0, index + 1):
                rank_list.append(index)

        # Step 3: Select solution index
        index1 = randint(0, len(rank_list) - 1)
        index2 = index1

        while index2 == index1:
            index2 = randint(0, len(rank_list) - 1)

        return population.get(rank_list[index1]), population.get(rank_list[index2])

    def _sort(self, population, objective):

        # Create a dict, in which we'll sort indexes based on fitness
        fitDict = {}

        for i in range(population.size):
            fitDict[i] = population.solutions[i].fitness

        # Get list of indexes in order of fitness
        newIndexs = sorted(fitDict, key=fitDict.get)

        # Reverse the list if we are doing Minimization (because of
        # how the rest of the class is code that way. Seems counter-
        # intuitive to do so.)
        if objective == ProblemObjective.Minimization:
            newIndexs.reverse()

        return newIndexs


# -------------------------------------------------------------------------------------------------
# class TournamentSelection
# -------------------------------------------------------------------------------------------------
class TournamentSelection:
    """
    """

    def select(self, population, objective, params):
        tournament_size = 2
        if "Tournament-Size" in params:
            tournament_size = params["Tournament-Size"]

        fitness_scores = [x.fitness for x in population.solutions]

        if objective == ProblemObjective.Maximization:
            index1 = fitness_scores.index(max(sample(fitness_scores, tournament_size)))
            fitness_scores[index1] = fitness_scores[index1] / 2
            index2 = fitness_scores.index(max(sample(fitness_scores, tournament_size)))

        elif objective == ProblemObjective.Minimization:
            index1 = fitness_scores.index(min(sample(fitness_scores, tournament_size)))
            fitness_scores[index1] = fitness_scores[index1] * 2
            index2 = fitness_scores.index(min(sample(fitness_scores, tournament_size)))
        elif objective == ProblemObjective.MultiObjective:
            index1 = self._select_index(objective, population, tournament_size)
            index2 = index1
            while index2 == index1:
                index2 = self._select_index(objective, population, tournament_size)

        return population.solutions[index1], population.solutions[index2]

    def _select_index(self, objective, population, tournament_size):

        index_temp = -1
        index_selected = randint(0, population.size - 1)

        for _ in range(0, tournament_size):
            index_temp = randint(0, population.size - 1)
            if objective == ProblemObjective.Maximization:
                if (
                    population.solutions[index_temp].fitness
                    > population.solutions[index_selected].fitness
                ):
                    index_selected = index_temp
            elif objective == ProblemObjective.Minimization:
                if (
                    population.solutions[index_temp].fitness
                    < population.solutions[index_selected].fitness
                ):
                    index_selected = index_temp
            elif objective == ProblemObjective.MultiObjective:
                # Only picks up if it's on a better front
                if (
                    population.solutions[index_temp].fitness[0]
                    <= population.solutions[index_selected].fitness[0]
                    and population.solutions[index_temp].fitness[1]
                    >= population.solutions[index_selected].fitness[1]
                ):
                    index_selected = index_temp

        return index_selected


###################################################################################################
# CROSSOVER APPROACHES
###################################################################################################
# -------------------------------------------------------------------------------------------------
# Singlepoint crossover
# -------------------------------------------------------------------------------------------------
def singlepoint_crossover(problem, solution1, solution2):
    singlepoint = randint(0, len(solution1.representation) - 1)
    # print(f" >> singlepoint: {singlepoint}")

    offspring1 = deepcopy(solution1)  # solution1.clone()
    offspring2 = deepcopy(solution2)  # .clone()

    for i in range(singlepoint, len(solution2.representation)):
        offspring1.representation[i] = solution2.representation[i]
        offspring2.representation[i] = solution1.representation[i]

    return offspring1, offspring2


# -------------------------------------------------------------------------------------------------
# N-point crossover
# -------------------------------------------------------------------------------------------------
def npoint_crossover(problem, solution1, solution2, n=4):
    sol1, sol2 = solution1.representation, solution2.representation
    # Offspring placeholders - None values make it easy to debug for errors
    offspring1 = [None] * len(sol1)
    offspring2 = [None] * len(sol1)

    if n > len(sol1):
        n = 1
        print(
            "ERROR: Please note that n provided was too large, function will",
            "return a single-point crossover, please resubmit with a valid n.",
        )

    cP_possibilities = [i for i in range(1, len(sol1))]
    cP = sample(cP_possibilities, n)
    cP.sort()

    # Fill in all values inbetween the crossover points, alternating between parents
    # at each iteration
    for i in cP:
        start = offspring1.index(None)
        offspring1[start:i] = sol1[start:i]
        offspring2[start:i] = sol2[start:i]
        sol1, sol2 = sol2, sol1
    # Fill in remaining missing values from the parents - ensure parents are picked
    # correctly to perform n-point crossover
    offspring1[offspring1.index(None) :] = sol1[offspring1.index(None) :]
    offspring2[offspring2.index(None) :] = sol2[offspring2.index(None) :]

    solution1.representation, solution2.representation = offspring1, offspring2

    return solution1, solution2


# -------------------------------------------------------------------------------------------------
# Uniform crossover
# -------------------------------------------------------------------------------------------------
def uniform_crossover(problem, solution1, solution2):
    sol1, sol2 = solution1.representation, solution2.representation
    # Offspring placeholders - None values make it easy to debug for errors
    offspring1 = [None] * len(sol1)
    offspring2 = [None] * len(sol1)
    # Uniform crossover, based on p, if p larger than 50 parent 1 givens gene i
    # to offspring 1, if lower, parent 2 gives gene i to offspring 1. Second
    # offspring is the reverse mapping of offspring 1.
    for i in range(len(sol1)):
        p = randint(0, 100)
        if p >= 50:
            offspring1[i] = sol1[i]
            offspring2[i] = sol2[i]
        else:
            offspring1[i] = sol2[i]
            offspring2[i] = sol1[i]

    solution1.representation, solution2.representation = offspring1, offspring2

    return solution1, solution2


# -------------------------------------------------------------------------------------------------
# Arithmetic crossover
# -------------------------------------------------------------------------------------------------
def arithmetic_crossover(problem, solution1, solution2, alpha=random()):
    sol1, sol2 = solution1.representation, solution2.representation
    # Offspring placeholders - None values make it easy to debug for errors
    offspring1 = [None] * len(sol1)
    offspring2 = [None] * len(sol1)
    # Keep variable integer False, it will switch if loop finds ints in the
    # submitted solution
    integer = False
    # Take weighted sum of two parents, invert alpha for second offspring
    for i in range(len(sol1)):
        offspring1[i] = sol1[i] * alpha + (1 - alpha) * sol2[i]
        offspring2[i] = sol2[i] * alpha + (1 - alpha) * sol1[i]
        if type(sol1[i]) is int:
            integer = True
    # If values are detected as integers, round the floats into ints
    if integer:
        offspring1 = [round(i) for i in offspring1]
        offspring2 = [round(i) for i in offspring2]

    solution1.representation, solution2.representation = offspring1, offspring2

    return solution1, solution2


# -------------------------------------------------------------------------------------------------
# Partially Mapped Crossover
# -------------------------------------------------------------------------------------------------
# DONE: implemented Partially Mapped Crossover
def pmx_crossover(problem, solution1, solution2):
    # Create two random crossover points
    cP1 = randint(0, len(solution1.representation))
    cP2 = randint(0, len(solution1.representation))

    # Ensure crossover points are not the same
    while cP1 == cP2:
        cP2 = randint(0, len(solution1.representation))

    # Invert the order, if crossover point 2 is smaller than point 1
    if cP1 > cP2:
        cP1, cP2 = cP2, cP1

    def PMXoffspring(x, y):
        # Offspring placeholder - None values make it easy to debug for errors
        offspring = [None] * len(x)

        # Copy crossover part into offspring
        offspring[cP1:cP2] = x[cP1:cP2]

        # Fill in elements of crossover which are not in offspring, according to
        # PMX crossover method
        for i in list(set(y[cP1:cP2]) - set(x[cP1:cP2])):
            temp = i
            index = y.index(x[y.index(temp)])
            while offspring[index] != None:
                temp = index
                index = y.index(x[temp])
            offspring[index] = i

        # Fill in remaining parts
        for i in range(0, len(x)):
            if offspring[i] == None:
                offspring[i] = y[i]
        return offspring

    solution1.representation, solution2.representation = (
        PMXoffspring(solution1.representation, solution2.representation),
        PMXoffspring(solution2.representation, solution1.representation),
    )

    return solution1, solution2


# -------------------------------------------------------------------------------------------------
# Cycle Crossover
# -------------------------------------------------------------------------------------------------
# Done: implemented Cycle Crossover
def cycle_crossover(problem, solution1, solution2):
    sol1, sol2 = solution1.representation, solution2.representation
    # Offsprint placeholders - None values make it easy to debug for errors
    offspring1 = [None] * len(sol1)
    offspring2 = [None] * len(sol1)
    # While there are still None values in offspring, get the first index of
    # None and start a "cycle" according to the cycle crossover method
    while None in offspring1:
        index = offspring1.index(None)
        # alternate parents between cycles beginning on second cycle
        if index != 0:
            sol1, sol2 = sol2, sol1
        val1 = sol1[index]
        val2 = sol2[index]

        while val1 != val2:
            offspring1[index] = sol1[index]
            offspring2[index] = sol2[index]
            val2 = sol2[index]
            index = sol1.index(val2)
        # In case last values share the same index, fill them in each offspring
        offspring1[index] = sol1[index]
        offspring2[index] = sol2[index]

    solution1.representation, solution2.representation = offspring1, offspring2

    return solution1, solution2


# -------------------------------------------------------------------------------------------------
# Edge Crossover
# -------------------------------------------------------------------------------------------------
# Done: implemented Edge Crossover
def edge_crossover(problem, solution1, solution2):
    def getOffspring(sol1, sol2):
        # Create an edge table. We follow Whitley edge-3 crossover in
        # this implementation
        edgeTable = {}
        listNums = sol1[:]

        for i in range(1, len(sol1) + 1):
            neighList = []
            neighList.append(sol1[sol1.index(i) - 1]), neighList.append(
                sol1[sol1.index(i) - len(sol1) + 1]
            )
            neighList.append(sol2[sol2.index(i) - 1]), neighList.append(
                sol2[sol2.index(i) - len(sol2) + 1]
            )
            edgeTable[i] = neighList

        def rmValDict(dictionary, value):
            for v in dictionary.values():
                if value in v:
                    v.remove(value)
                    try:
                        v.remove(value)
                    except ValueError:
                        pass

        def getDuplicate(l):
            duplicates = list(set([x for x in l if l.count(x) > 1]))
            return duplicates

        def getLongest(l):
            longest = []
            length = 0
            for element in l:
                if len(edgeTable[element]) >= length:
                    longest.append(element)
                    length = len(edgeTable[element])

            return longest

        offspring = [None] * len(sol1)

        element = randint(1, len(sol1) - 1)
        index = 0
        offspring[index] = element

        while None in offspring:
            element = offspring[index]
            listNums.remove(element)
            rmValDict(edgeTable, element)
            index = offspring.index(None)
            if len(edgeTable[element]) > len(set(edgeTable[element])):
                offspring[index] = choice(getDuplicate(edgeTable[element]))
            elif len(edgeTable[element]) == 1:
                offspring[index] = edgeTable[element][0]
            elif len(edgeTable[element]) > 0:
                offspring[index] = choice(getLongest(edgeTable[element]))
            else:
                offspring[index] = choice(listNums)

        return offspring

    solution1.representation, solution2.representation = (
        getOffspring(solution1.representation, solution2.representation),
        getOffspring(solution2.representation, solution1.representation),
    )

    return solution1, solution2


# -------------------------------------------------------------------------------------------------
# Non-Wrapping Order Crossover
# -------------------------------------------------------------------------------------------------
# DONE: implemented Non-Wrapping Order Crossover
def nwoc_crossover(problem, solution1, solution2):
    # Create two random crossover points
    cP1 = randint(0, len(solution1.representation))
    cP2 = randint(0, len(solution1.representation))

    # Ensure crossover points are not the same
    while cP1 == cP2:
        cP2 = randint(0, len(solution1.representation))

    # Invert the order, if crossover point 2 is smaller than point 1
    if cP1 > cP2:
        cP1, cP2 = cP2, cP1

    def NWOCoffspring(x, y):
        # Offspring placeholders - None values make it easy to debug for errors
        offspring = [None] * len(x)
        offspring[:] = x[:]

        # Remove values in crossover, shift and resinsert as per NWOC

        for i in list(set(y[cP1:cP2]) - set(x[cP1:cP2])):
            offspring.remove(i)

        count = 0

        for i in y[cP1:cP2]:
            if i not in offspring:
                offspring.insert(cP1 + count, i)
                count += 1

        return offspring

    solution1.representation, solution2.representation = (
        NWOCoffspring(solution1.representation, solution2.representation),
        NWOCoffspring(solution2.representation, solution1.representation),
    )

    return solution1, solution2


###################################################################################################
# MUTATION APPROACHES
###################################################################################################
# -------------------------------------------------------------------------------------------------
# Singlepoint mutation
# -----------------------------------------------------------------------------------------------
def single_point_mutation(problem, solution):
    encoding = problem.encoding
    solution.representation = _single_point_mutation(
        solution.representation, encoding.encoding_type, encoding.encoding_data
    )
    return solution


def _single_point_mutation(solRep, probEncType, probEncodingData):
    singlepoint = randint(0, len(solRep) - 1)
    # print(f" >> singlepoint: {singlepoint}")

    try:
        if probEncType == EncodingDataType.choices:
            solRep = point_mutation(solRep, probEncodingData, singlepoint)
        elif probEncType == EncodingDataType.min_max:
            # Computes list of possible data for this gene
            probEncodingInterval = list(
                range(
                    probEncodingData["min"][singlepoint],
                    probEncodingData["max"][singlepoint] + 1,
                )
            )
            solRep = point_mutation(solRep, probEncodingInterval, singlepoint)
    except:
        print("(!) Error: singlepoint mutation encoding.data issues)")
    return solRep

    # -------------------------------------------------------------------------------------------------


# Multipoint mutation
# -----------------------------------------------------------------------------------------------
def multi_point_mutation(problem, solution):
    encoding = problem.encoding
    solution.representation = _multi_point_mutation(
        solution.representation, encoding.encoding_type, encoding.encoding_data
    )
    return solution


def _multi_point_mutation(solRep, probEncType, probEncodingData):
    # Number of points (or genes) to be mutated
    numOfPoints = randint(1, len(solRep))
    # Position of genes to be mutated
    posOfPoints = sample(range(len(solRep)), numOfPoints)
    # print(f" >> multipoint: {posOfPoints}")

    try:
        for iPos in posOfPoints:
            if probEncType == EncodingDataType.choices:
                solRep = point_mutation(solRep, probEncodingData, iPos)
            elif probEncType == EncodingDataType.min_max:
                # Computes list of possible data for this gene
                probEncodingInterval = list(
                    range(
                        probEncodingData["min"][iPos], probEncodingData["max"][iPos] + 1
                    )
                )
                solRep = point_mutation(solRep, probEncodingInterval, iPos)
    except:
        print("(!) Error: multipoint mutation encoding.data issues)")

    return solRep


# -------------------------------------------------------------------------------------------------
# Mutation utils
# -----------------------------------------------------------------------------------------------
def point_mutation(rep, encoding_data, iPos):
    if len(encoding_data) == 1:
        # There's only one option, so it cannot mutate and returns the same value
        return rep

    temp = deepcopy(encoding_data)

    temp.remove(rep[iPos])

    gene = temp[0]
    if len(temp) > 1:
        gene = choice(temp)

    rep[iPos] = gene
    return rep


# -------------------------------------------------------------------------------------------------
# Swap mutation
# -----------------------------------------------------------------------------------------------
def swap_mutation(problem, solution):
    encoding = problem.encoding
    solution.representation = _swap_mutation(
        solution.representation, encoding.encoding_type
    )
    return solution


def _swap_mutation(solRep, probEncType):
    if len(solRep) < 2:
        print(
            "ERROR: Please note that the size of the representation must be at least 2"
        )

    # Position of the two genes to be swapped
    posOfPoints = sample(range(len(solRep)), 2)
    # print(f" >> swap: {posOfPoints}")

    try:
        if (
            probEncType == EncodingDataType.choices
            or probEncType == EncodingDataType.min_max
        ):
            solRep[posOfPoints[0]], solRep[posOfPoints[1]] = (
                solRep[posOfPoints[1]],
                solRep[posOfPoints[0]],
            )
    except:
        print("(!) Error: swap mutation encoding.data issues)")

    return solRep


# -------------------------------------------------------------------------------------------------
# Insert mutation
# -----------------------------------------------------------------------------------------------
def insert_mutation(problem, solution):
    encoding = problem.encoding
    solution.representation = _insert_mutation(
        solution.representation, encoding.encoding_type
    )
    return solution


def _insert_mutation(solRep, probEncType):
    if len(solRep) < 2:
        print(
            "ERROR: Please note that the size of the representation must be at least 2"
        )

    # Position of the two genes to be swapped
    posOfPoints = sample(range(len(solRep)), 2)
    # This method assumes that the second point is after (on the right of) the first one
    # So, need to sort the list
    posOfPoints.sort()
    # print(f" >> insert: {posOfPoints}")

    if (
        probEncType == EncodingDataType.choices
        or probEncType == EncodingDataType.min_max
    ):
        try:
            solRep.insert(posOfPoints[0], solRep.pop(posOfPoints[1]))
            return solRep
        except:
            print("(!) Error: insert mutation encoding.data issues)")


# -------------------------------------------------------------------------------------------------
# Inversion mutation
# -----------------------------------------------------------------------------------------------
def inversion_mutation(problem, solution):
    encoding = problem.encoding
    solution.representation = _inversion_mutation(
        solution.representation, encoding.encoding_type
    )
    return solution


def _inversion_mutation(solRep, probEncType):
    if len(solRep) < 2:
        print(
            "ERROR: Please note that the size of the representation must be at least 2"
        )

    # Position of the start and end of substring
    posOfPoints = sample(range(len(solRep) + 1), 2)
    # This method assumes that the second point is after (on the right of) the first one
    # So, need to sort the list
    posOfPoints.sort()
    # If the positions are next to each other, then the substring has only one object
    # To avoid this, checks the substring size and changes it if needed
    if posOfPoints[1] - posOfPoints[0] < 2:
        if posOfPoints[1] != len(solRep):
            posOfPoints[1] = posOfPoints[1] + 1
        else:
            posOfPoints[0] = posOfPoints[0] - 1
    # print(f" >> inversion: {posOfPoints}")

    if (
        probEncType == EncodingDataType.choices
        or probEncType == EncodingDataType.min_max
    ):
        try:
            solRep[posOfPoints[0] : posOfPoints[1]] = solRep[
                posOfPoints[0] : posOfPoints[1]
            ][::-1]
            return solRep
        except:
            print("(!) Error: inversion mutation encoding.data issues)")


# -------------------------------------------------------------------------------------------------
# Scramble mutation
# -----------------------------------------------------------------------------------------------
def scramble_mutation(problem, solution):
    encoding = problem.encoding
    solution.representation = _scramble_mutation(
        solution.representation, encoding.encoding_type
    )
    return solution


def _scramble_mutation(solRep, probEncType):
    if len(solRep) < 2:
        print(
            "ERROR: Please note that the size of the representation must be at least 2"
        )

    # Position of the start and end of substring
    posOfPoints = sample(range(len(solRep) + 1), 2)
    # This method assumes that the second point is after (on the right of) the first one
    # So, need to sort the list
    posOfPoints.sort()
    # If the positions are next to each other, then the substring has only one object
    # To avoid this, checks the substring size and changes it if needed
    if posOfPoints[1] - posOfPoints[0] < 2:
        if posOfPoints[1] != len(solRep):
            posOfPoints[1] = posOfPoints[1] + 1
        else:
            posOfPoints[0] = posOfPoints[0] - 1
    # print(f" >> scramble: {posOfPoints}")

    if (
        probEncType == EncodingDataType.choices
        or probEncType == EncodingDataType.min_max
    ):
        try:
            # This is a random shuffle, so there may be a chance that there is no shuffle (big chance if substring is small)
            # So, need to guarantee that the shuffle really changes the substring
            sub = solRep[posOfPoints[0] : posOfPoints[1]]
            while sub == solRep[posOfPoints[0] : posOfPoints[1]]:
                shuffle(sub)
            solRep[posOfPoints[0] : posOfPoints[1]] = sub

            return solRep
        except:
            print("(!) Error: scramble mutation encoding.data issues)")


def portfolio_mutation(problem, solution):

    for i in range(len(solution.representation)):
        if getrandbits(1):
            newShare = choice(problem._encoding.encoding_data)
            while newShare in solution.representation:
                newShare = choice(problem._encoding.encoding_data)
            solution.representation[i] = newShare
    return solution


###################################################################################################
# REPLACEMENT APPROACHES
###################################################################################################
# -------------------------------------------------------------------------------------------------
# Standard replacement
# -----------------------------------------------------------------------------------------------
def standard_replacement(problem, current_population, new_population):
    return new_population


# -------------------------------------------------------------------------------------------------
# Elitism replacement
# -----------------------------------------------------------------------------------------------
def elitism_replacement(problem, current_population, new_population):

    if problem.objective == ProblemObjective.Minimization:
        if current_population.fittest.fitness < new_population.fittest.fitness:
            new_population.replace_leastfit(current_population.fittest)

    elif problem.objective == ProblemObjective.Maximization:
        if current_population.fittest.fitness > new_population.fittest.fitness:
            new_population.replace_leastfit(current_population.fittest)
    elif problem.objective == ProblemObjective.MultiObjective:
        new_population.replace_leastfit(current_population.fittest)

    return new_population
