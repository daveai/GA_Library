# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------------------
"""
Case Problem
----------------
Content:

▶ class CaseProblem

─────────────────────────────────────────────────────────────────────────

CIFO - Computation Intelligence for Optimization

Author: Fernando A J Peres - fperes@novaims.unl.pt - (2019) version L4.0

"""
# -------------------------------------------------------------------------------------------------


from random import choice
from copy import deepcopy
from cifo.problem.solution import LinearSolution, Encoding
from cifo.problem.objective import ProblemObjective
from cifo.util.plot import plot_performance_chart
from cifo.util.consolidate import consolidate
from cifo.custom_problem.travel_salesman_problem import TravelSalesmanProblem


class CaseProblem:
    """
    Case Problem
    """

    # Constructor
    # -------------------------------------------------------------------------------------------------------------
    def __init__(self, problem, problem_data, algo, algo_data, case_data):
        """
        The constructor of the Case Problem
        """
        self._name = "Case Problem"
        self._problem = problem
        self._problem_data = problem_data
        self._algo = algo
        self._algo_data = algo_data
        self._case_data = case_data
        self._best_fitness = 0
        self._best_fit_individual = []

        # if "Decision-Variables" in self._problem_data:
        #     self._decision_variables = self._problem_data[ "Decision-Variables"]
        #
        # if "Constraints" in self._problem_data:
        #     self._constraints = self._problem_data["Constraints"]

        if "Params" in self._algo_data:
            self._params = self._algo_data["Params"]

        if "Num-Runs" in self._case_data:
            self._num_runs = self._case_data["Num-Runs"]

        if "Observer" in self._case_data:
            self._observer = self._case_data["Observer"]

        if "Plot" in self._case_data:
            self._plot = self._case_data["Plot"]
        else:
            self._plot = False

        if "Consolidate" in self._case_data:
            self._consolidate = self._case_data["Consolidate"]
        else:
            self._consolidate = False

        if "Log" in self._case_data:
            self._log = self._case_data["Log"]
        else:
            self._log = False

    @property
    def name(self):
        return self._name

    @property
    def logname(self):
        return self._problem.name

    def run(self):

        # problem_instance = self._problem(
        #     decision_variables=self._decision_variables,
        #     constraints=self._constraints)
        problem_instance = self._problem
        best_solution_hist = []
        for irun in range(1, self._num_runs + 1):
            al = self._algo(
                problem_instance=problem_instance,
                params=self._params,
                run=irun,
                log_name=self.logname,
            )

            al_observer = self._observer(al)
            al.register_observer(al_observer)
            al.search()
            if self._log:
                al.save_log()
            best_solution_hist.append(al.get_best_fits())

        if self._consolidate:
            df = consolidate(self.logname)
        if self._plot:
            plot_performance_chart(df)
        print(best_solution_hist)
        return best_solution_hist
