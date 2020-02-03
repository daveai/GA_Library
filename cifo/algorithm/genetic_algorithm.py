# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------------------
"""
Genetic  Meta-Heuristic
----------------------------

Content: 

▶ class Genetic Algorithm (Incomplete version)

─────────────────────────────────────────────────────────────────────────

CIFO - Computation Intelligence for Optimization

Author: Fernando A J Peres - fperes@novaims.unl.pt - (2019) version L4.0
"""
# -------------------------------------------------------------------------------------------------

# import
from random import random
from copy import deepcopy

from cifo.algorithm.ga_operators import (
    initialize_randomly,
    RankSelection,
    RouletteWheelSelection,
    TournamentSelection,
    singlepoint_crossover,
    initialize_using_hc,
    single_point_mutation,
    cycle_crossover,
    standard_replacement,
    elitism_replacement,
)

from cifo.problem.population import Population

from cifo.util.terminal import Terminal, FontColor

from cifo.util.logger import GeneticAlgorithmLogger


# default params
default_params = {
    "Population-Size": 5,
    "Number-of-Generations": 10,
    "Crossover-Probability": 0.8,
    "Mutation-Probability": 0.5,
    "Initialization-Approach": initialize_randomly,
    "Selection-Approach": RouletteWheelSelection,
    "Tournament-Size": 5,
    "Crossover-Approach": singlepoint_crossover,
    "Mutation-Aproach": single_point_mutation,
    "Replacement-Approach": elitism_replacement,
}

# -------------------------------------------------------------------------------------------------
# Genetic Algorithm
# REMARK: Incomplete Version
# -------------------------------------------------------------------------------------------------
class GeneticAlgorithm:
    """
    A genetic algorithm is a search heuristic that is inspired by Charles Darwin’s theory of 
    natural evolution. This algorithm reflects the process of natural selection where the fittest 
    individuals are selected for reproduction in order to produce offspring of the next generation.

    The process of natural selection starts with the selection of fittest individuals from a population. 
    They produce offspring which inherit the characteristics of the parents and will be added to the 
    next generation. If parents have better fitness, their offspring will be better than parents and 
    have a better chance at surviving. This process keeps on iterating and at the end, a generation with 
    the fittest individuals will be found.
    
    This notion can be applied for a search problem. We consider a set of solutions for a problem and 
    select the set of best ones out of them.
    
    Six phases are considered in a genetic algorithm.

    1. Initial population
    2. Fitness function
    3. Selection
    4. Crossover
    5. Mutation
    6. Replacement
    """

    # Constructor
    # ---------------------------------------------------------------------------------------------
    def __init__(self, problem_instance, params=default_params, run=0, log_name="temp"):
        self._text = ""
        self._generation = 0
        self._run = run
        self._fittest = None
        self._problem_instance = problem_instance
        self._population = None
        self._observers = []

        self._parse_params(params)

        self._logger = GeneticAlgorithmLogger(log_name, run)

        self._observers = []

    # search
    # ---------------------------------------------------------------------------------------------
    def search(self):
        """
            Genetic Algorithm - Search Algorithm
            1. Initial population

            2. Repeat n generations )(#1 loop )
                
                2.1. Repeat until generate the next generation (#2 loop )
                    1. Selection
                    2. Try Apply Crossover (depends on the crossover probability)
                    3. Try Apply Mutation (depends on the mutation probability)
                
                2.2. Replacement

            3. Return the best solution    
        """
        problem = self._problem_instance
        select = self._selection_approach
        cross = self._crossover_approach
        mutate = self._mutation_approach
        replace = self._replacement_approach
        is_admissible = self._problem_instance.is_admissible

        self._generation = 0
        self._notify(message="Genetic Algorithm")

        # 1. Initial population
        self._population = self._initialize(problem, self._population_size)
        self._fittest = self._population.fittest

        self._notify()
        self._store_population()

        # 2. Repeat n generations )(#1 loop )
        for self._generation in range(1, self._number_of_generations + 1):

            new_population = Population(
                problem=problem, maximum_size=self._population_size, solution_list=[]
            )
            i = 0

            # 2.1. Repeat until generate the next generation (#2 loop )
            while new_population.has_space:
                # 2.1.1. Selection
                parent1, parent2 = select(
                    self._population, problem.objective, self._params
                )
                offspring1 = problem.fast_solution_copy(
                    parent1.representation
                )  # parent1.clone()
                offspring2 = problem.fast_solution_copy(
                    parent2.representation
                )  # parent1.clone()
                # offspring1 = problem.build_solution(parent1.representation[:]) # parent2.clone()
                # offspring2 = problem.build_solution(parent2.representation[:]) # parent2.clone()
                # 2.1.2. Try Apply Crossover (depends on the crossover probability)
                if self.apply_crossover:
                    offspring1, offspring2 = cross(
                        problem, offspring1, offspring2
                    )  # I guess this was wrong here, I've replaced 'solution' by 'offspring'
                    offspring1.id = [self._generation, i]
                    i += 1
                    offspring2.id = [self._generation, i]
                    i += 1

                # 2.1.3. Try Apply Mutation (depends on the mutation probability)
                if self.apply_mutation:
                    offspring1 = mutate(problem, offspring1)
                    offspring1.id = [self._generation, i]
                    i += 1
                if self.apply_mutation:
                    offspring2 = mutate(problem, offspring2)
                    offspring2.id = [self._generation, i]
                    i += 1

                # # Don't allow Twins
                # solReps = [x.representation for x in new_population.solutions]
                # if offspring1.representation in solReps:
                #     offspring1 = problem.get_single_neighbor(offspring1, problem)

                # if offspring2.representation in solReps:
                #     offspring2 = problem.get_single_neighbor(offspring2, problem)

                # add the offsprings in the new population (New Generation)
                if new_population.has_space and is_admissible(offspring1):
                    problem.evaluate_solution(offspring1)
                    new_population.solutions.append(offspring1)

                if new_population.has_space and is_admissible(offspring2):
                    problem.evaluate_solution(offspring2)
                    new_population.solutions.append(offspring2)
                    # print(f'Added O2 - {offspring2.id}-{offspring2.representation}')

            self._population = replace(problem, self._population, new_population)

            self._fittest = self._population.fittest
            self._notify()
            self._store_population()

        self._notify(message="Fittest Solution")
        return self._fittest

    def __str__(self):
        self._text = ""

    @property
    def apply_crossover(self):
        chance = random()
        return chance < self._crossover_probability

    @property
    def apply_mutation(self):
        chance = random()
        return chance < self._mutation_probability

    # initialize
    # ---------------------------------------------------------------------------------------------
    def _parse_params(self, params):
        self._params = params
        self._text = ""

        self._population_size = 10
        if "Population-Size" in params:
            self._population_size = params["Population-Size"]
        self._text += "\nPopulation-Size " + str(self._population_size)

        self._crossover_probability = 0.8
        if "Crossover-Probability" in params:
            self._crossover_probability = params["Crossover-Probability"]
        self._text += "\nCrossover-Probability " + str(self._crossover_probability)

        self._mutation_probability = 0.5
        if "Mutation-Probability" in params:
            self._mutation_probability = params["Mutation-Probability"]

        self._number_of_generations = 5
        if "Number-of-Generations" in params:
            self._number_of_generations = params["Number-of-Generations"]

        # Initialization signature: <method_name>( problem, population_size ):
        self._initialize = None
        if "Initialization-Approach" in params:
            self._initialize = params["Initialization-Approach"]
        else:
            print("Undefined Initialization approach. The default will be used.")
            self._initialize = initialize_randomly

        # Selection
        self._selection_approach = None
        if "Selection-Approach" in params:
            self._selection_approach = params["Selection-Approach"]
        else:
            print("Undefined Selection approach. The default will be used.")
            parent_selection = TournamentSelection()
            self._selection_approach = parent_selection.select

        # tournament size
        self._tournament_size = 5
        if "Tournament-Size" in params:
            self._tournament_size = params["Tournament-Size"]

        # Crossover
        self._crossover_approach = None
        if "Crossover-Approach" in params:
            self._crossover_approach = params["Crossover-Approach"]
        else:
            print("Undefined Crossover-Approach. The default will be used.")
            self._crossover_approach = singlepoint_crossover

        # Mutation
        self._mutation_approach = None
        if "Mutation-Aproach" in params:
            self._mutation_approach = params["Mutation-Aproach"]
        else:
            print("Undefined Mutation-Aproach. The default will be used.")
            self._mutation_approach = single_point_mutation

        # Replacement
        self._replacement_approach = None
        if "Replacement-Approach" in params:
            self._replacement_approach = params["Replacement-Approach"]
        else:
            print("Undefined Replacement-Approach. The default will be used.")
            self._replacement_approach = elitism_replacement

        self._notify(message="Configuration", content=self._text)

    # register observer
    # ----------------------------------------------------------------------------------------------
    def register_observer(self, observer):
        self._observers.append(observer)

    # unregister observer
    # ----------------------------------------------------------------------------------------------
    def unregister_observer(self, observer):
        self._observers.remove(observer)

    # notify
    # ----------------------------------------------------------------------------------------------
    def _notify(self, message="", content=""):

        self._state = {
            "iteration": self._generation,
            "message": message,
            "content": content,
            "fittest": self._fittest,
        }

        if message == "":
            self._logger.add(generation=self._generation, solution=self._fittest)

        # if message == "Fittest Solution" :
        #    self._logger.save()

        for observer in self._observers:
            observer.update()

    # get state
    # ----------------------------------------------------------------------------------------------
    def get_state(self):
        return self._state

    def save_log(self):
        self._logger.save()

    def get_best_fits(self):
        return self._logger.get_fitnesses()

    def _store_population(self):
        self._logger.add_population(self._population)

    def get_populations(self):
        return self._logger.get_populations()

    # def calc_weights( self, solution ):
    #    weights = self._problem_instance.decision_variables["Weights"]
    #
    #    weight = 0
    #    for  i in range(0, len( weights )):
    #        if solution.representation[ i ] == 1:
    #            weight += weights[ i ]
    #
    #    return weight
