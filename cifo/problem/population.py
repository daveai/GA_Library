from cifo.problem.objective import ProblemObjective
from cifo.util.MO_utils import (
    crowding_distance,
    fast_non_dominated_sort,
    sort_by_values,
)

# -------------------------------------------------------------------------------------------------
# Population Class
# -------------------------------------------------------------------------------------------------
class Population:
    """
    Population - 
    """

    # ---------------------------------------------------------------------------------------------
    def __init__(self, problem, maximum_size, solution_list):
        self._problem = problem
        self._objective = problem.objective
        self._max_size = maximum_size
        self._list = solution_list
        self._fittest = None
        self._sorted = False

    # ---------------------------------------------------------------------------------------------
    @property
    def fittest(self):
        self.sort()
        if len(self._list) > 0:
            return self._list[-1]
        return None
        # if self._objective == ProblemObjective.MultiObjective:
        #     self.sort()
        #     if len(self._list) > 0 :
        #         return self._list[ -1 ]
        #     return None
        # else:
        #     fitness_scores = [x.fitness for x in self._list]
        #     if len(fitness_scores) > 0:
        #         if self._objective == ProblemObjective.Maximization:
        #             return self._list[fitness_scores.index(max(fitness_scores))]
        #         elif self._objective == ProblemObjective.Minimization:
        #             return self._list[fitness_scores.index(min(fitness_scores))]
        #     return None

    @property
    def least_fit(self):
        self.sort()
        if len(self._list) > 0:
            return self._list[0]
        return None
        # if self._objective == ProblemObjective.MultiObjective:
        #     self.sort()
        #     if len(self._list) > 0 :
        #         return self._list[ 0 ]
        #     return None
        # else:
        #     fitness_scores = [x.fitness for x in self._list]
        #     if len(fitness_scores) > 0:
        #         if self._objective == ProblemObjective.Maximization:
        #             return self._list[fitness_scores.index(min(fitness_scores))]
        #         elif self._objective == ProblemObjective.Minimization:
        #             return self._list[fitness_scores.index(max(fitness_scores))]
        #     return None

    def replace_leastfit(self, solution):
        self.sort()
        self._list[0] = solution
        # if self._objective == ProblemObjective.MultiObjective:
        #     self.sort()
        #     self._list[ 0 ] = solution
        # else:
        #     fitness_scores = [x.fitness for x in self._list]
        #     if self._objective == ProblemObjective.Maximization:
        #         self._list[fitness_scores.index(min(fitness_scores))] = solution
        #     elif self._objective == ProblemObjective.Minimization:
        #         self._list[fitness_scores.index(max(fitness_scores))] = solution

    @property
    def size(self):
        return len(self._list)

    @property
    def has_space(self):
        return len(self._list) < self._max_size

    @property
    def is_full(self):
        return len(self._list) >= self._max_size

    def add(self, solution):
        self._list.append(solution)

    def get(self, index):
        """
        It returns a solution of the population according to the index
        """
        if index >= 0 and index < len(self._list):
            return self._list[index]
        else:
            return None

    @property
    def solutions(self):
        """
        Solution list (of the population)
        """
        return self._list

    def sort(self):
        """
        it sorts the population in ascending order of fittest solution in accordance with the objective

        @ objective
        - Maximization
        - Minimization
        - Multi-objective { set of objectives }
        """

        if self._objective == ProblemObjective.Maximization:
            for i in range(0, len(self._list)):
                for j in range(i, len(self._list)):
                    if self._list[i].fitness > self._list[j].fitness:
                        swap = self._list[j]
                        self._list[j] = self._list[i]
                        self._list[i] = swap

        elif self._objective == ProblemObjective.Minimization:
            for i in range(0, len(self._list)):
                for j in range(i, len(self._list)):
                    if self._list[i].fitness < self._list[j].fitness:
                        swap = self._list[j]
                        self._list[j] = self._list[i]
                        self._list[i] = swap
        elif self._objective == ProblemObjective.MultiObjective:
            # pop_fit = [solution.fitness for solution in self._list]
            # front_no, max_front = nd_sort(pop_fit)
            # crowd_dis = crowding_distance(pop_fit, front_no)
            pop_fit1 = [solution.fitness[0] for solution in self._list]
            pop_fit2 = [solution.fitness[1] for solution in self._list]
            non_dominated_sorted_solution = fast_non_dominated_sort(pop_fit1, pop_fit2)
            crowding_distance_values = []
            temp_list = self._list
            self._list = []
            for i in range(len(non_dominated_sorted_solution) - 1, -1, -1):
                cd = crowding_distance(
                    pop_fit1, pop_fit2, non_dominated_sorted_solution[i][:]
                )
                ordered_front = [
                    x for _, x in sorted(zip(cd, non_dominated_sorted_solution[i][:]))
                ]
                for j in ordered_front:
                    self._list.append(temp_list[j])

        self._sorted = True
