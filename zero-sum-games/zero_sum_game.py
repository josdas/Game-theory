import numpy as np
import scipy.optimize


class ZeroSumGame:
    def __init__(self, get_score, first_actions, second_actions):
        """
        Calculate Nash equilibrium in mixed strategies for zero-sum games.
        :param get_score: get_score(first_action, second_action) is a function that returns score of the game.
        :param first_actions: first_actions is a list of possible actions for the first player.
        :param second_actions: second_actions is a list of possible actions for the second player.
        """
        self._get_score = get_score
        self._first_actions = first_actions
        self._second_actions = second_actions
        self.optimal_policy = None
        self.game_price = self.calc_best_policy_for_the_first_player()

    def calc_best_policy_for_the_first_player(self):
        n = len(self._first_actions)
        m = len(self._second_actions)

        price_matrix = np.array(
            [[self._get_score(f_act, s_act) for f_act in self._first_actions]
             for s_act in self._second_actions])

        # First n variables are probabilities of actions for
        # the first players.
        # Introduced price of the game for the first player
        # with the last index.
        c = np.zeros(n + 1)
        c[n] = -1

        # By min-max theorem the first player should get at
        # least the price of the game against each pure strategy
        # of the second player.
        A_ub = np.hstack((-price_matrix, np.ones((m, 1))))
        b_ub = np.zeros(m)

        # Normalization of probabilities.
        # sum(probability_i-th_action) = 1
        A_eq = np.ones((1, n + 1))
        A_eq[0][n] = 0
        b_eq = np.ones(1)

        # Probabilities are non-negative.
        # Price of the game can be negative.
        bounds = [(0, None) for _ in range(n)] + [(None, None)]

        # Used simplex method to solve linear programming problem.
        # In the worst case asymptotic is exponential but in
        # real-life games it's almost polynomial.
        res = scipy.optimize.linprog(c=c,
                                     A_ub=A_ub, b_ub=b_ub,
                                     A_eq=A_eq, b_eq=b_eq,
                                     bounds=bounds)

        # Policy is a distribution over actions for the first player.
        self.optimal_policy = res.x[:n]
        return -res.fun

    def gen_action(self):
        """
        :return: Generate sample of the optimal distribution over actions for the first player.
        """
        return np.random.choice(self._first_actions, p=self.optimal_policy)
