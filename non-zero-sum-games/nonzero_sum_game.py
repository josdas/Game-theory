import numpy as np
import scipy.optimize


class NonZeroSumGame:
    def __init__(self, get_score, first_actions, second_actions):
        """
        Calculate Nash equilibrium in mixed strategies for non-zero-sum games.
        :param get_score: get_score(first_action, second_action) is a function
        that returns pair of scores for the game.
        :param first_actions: first_actions is a list of possible actions for the first player.
        :param second_actions: second_actions is a list of possible actions for the second player.
        """
        self._get_score = get_score
        self._actions = [first_actions, second_actions]
        self.optimal_policy = None
        self.game_price = self.calc_optimal_policy()

    def calc_optimal_policy(self):
        n = len(self._actions[0])
        m = len(self._actions[1])

        price_matrix = np.array(
            [[self._get_score(f_act, s_act) for s_act in self._actions[1]]
             for f_act in self._actions[0]]).transpose((2, 0, 1))

        total_price = price_matrix[0] + price_matrix[1]

        vars_count = n + m + 2

        def loss(x):
            first, second = x[:n], x[n:n + m]
            alpha, betta = x[n + m], x[n + m + 1]
            return -first @ total_price @ second + alpha + betta

        def jac(x):
            first, second = x[:n], x[n:n + m]
            first_jac = -total_price @ second
            second_jac = -first @ total_price
            alpha_jac = [1]
            betta_jac = [1]
            return np.concatenate((first_jac, second_jac,
                                   alpha_jac, betta_jac))

        # Gx >= 0
        G = np.zeros((n + m, vars_count))
        for i in range(n):
            G[i][n:n + m] = -price_matrix[0][i]
            G[i][n + m] = 1
        for i in range(m):
            G[n + i][:n] = -price_matrix[1][:, i]
            G[n + i][n + m + 1] = 1

        # Ax - b = 0
        A = np.zeros((2, vars_count))
        A[0][:n] = 1
        A[1][n:n + m] = 1
        b = np.ones(2)

        constraints = [
            {
                'type': 'ineq',
                'fun': lambda x: G @ x,
                'jac': lambda x: G,
            },
            {
                'type': 'eq',
                'fun': lambda x: A @ x - b,
                'jac': lambda x: A
            }
        ]

        bounds = [(0, None) for _ in range(n + m)]
        bounds += [(None, None), (None, None)]

        x0 = np.random.random(vars_count)
        x0[:n] /= x0[:n].sum()
        x0[n:n + m] /= x0[n:n + m].sum()
        x0[n + m:n + m + 2] *= total_price.sum()

        res = scipy.optimize.minimize(x0=x0, fun=loss, jac=jac, method='SLSQP',
                                      constraints=constraints,
                                      bounds=bounds)

        # Policy is a distribution over actions for the each players.
        self.optimal_policy = res.x[:n], res.x[n:n + m]

        game_price = [self.optimal_policy[0] @ price_matrix[player] @ self.optimal_policy[1]
                      for player in (0, 1)]
        return game_price

    def gen_action(self, player):
        assert player in {1, 2}
        player -= 1
        return np.random.choice(self._actions[player], p=self.optimal_policy[player])
