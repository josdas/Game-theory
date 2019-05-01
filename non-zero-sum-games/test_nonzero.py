import numpy as np

from nonzero_sum_game import NonZeroSumGame

SEED = 1337


def simulate_game(get_score, first_actions, second_actions, iters=10 ** 4, comments=None, verbosity=1):
    game = NonZeroSumGame(get_score, first_actions, second_actions)

    score_first, score_second = 0, 0
    for it in range(iters):
        f_act = game.gen_action(1)
        s_act = game.gen_action(2)
        f_score, s_score = get_score(f_act, s_act)
        score_first += f_score
        score_second += s_score

    if verbosity >= 1:
        if comments is not None:
            print(comments)
        print(game.optimal_policy[0])
        print(f'Game price first: {game.game_price[0]}')
        print(f'Average score first: {score_first / iters}')
        print()
        print(game.optimal_policy[1])
        print(f'Game price second: {game.game_price[1]}')
        print(f'Average score second: {score_second / iters}')
        print('-' * 80)
    return game.optimal_policy, game.game_price


def rock_paper_scissors(first_act, second_act):
    FIGHTS = {
        ('rock', 'scissors'),
        ('paper', 'rock'),
        ('scissors', 'paper')
    }

    ACTIONS = ('rock', 'paper', 'scissors')

    assert first_act in ACTIONS and second_act in ACTIONS
    if (first_act, second_act) in FIGHTS:
        return 1, -1
    if (second_act, first_act) in FIGHTS:
        return -1, 1
    return 0, 0


def prisoners_dilemma(first_act, second_act):
    ACTIONS = ['silent', 'betray']
    assert first_act in ACTIONS and second_act in ACTIONS
    if first_act == 'silent' and second_act == 'silent':
        return -1, -1
    if first_act == 'silent' and second_act == 'betray':
        return -3, 0
    if first_act == 'betray' and second_act == 'betray':
        return -2, -2,
    if first_act == 'betray' and second_act == 'silent':
        return 0, -3


def battle_of_the_sexes(first_act, second_act):
    ACTIONS = ['opera', 'football']
    assert first_act in ACTIONS and second_act in ACTIONS
    if first_act == 'opera' and second_act == 'opera':
        return 2, 1
    if first_act == 'opera' and second_act == 'football':
        return -1, -1
    if first_act == 'football' and second_act == 'football':
        return 1, 2
    if first_act == 'football' and second_act == 'opera':
        return -1, -1


if __name__ == '__main__':
    np.random.seed(SEED)

    simulate_game(rock_paper_scissors,
                  ['rock', 'paper', 'scissors'],
                  ['rock', 'paper', 'scissors'],
                  comments="Fair rock paper scissors with zero price")

    simulate_game(rock_paper_scissors,
                  ['rock', 'paper', 'scissors'],
                  ['rock', 'paper'],
                  comments="Unfair rock paper scissors where the second player can't use 'scissors'")

    simulate_game(rock_paper_scissors,
                  ['rock', 'paper', 'scissors'],
                  ['rock'],
                  comments="Unfair rock paper scissors where the second player can't use 'scissors' and 'paper'")

    simulate_game(prisoners_dilemma,
                  ['silent', 'betray'],
                  ['silent', 'betray'],
                  comments="Prisoner's dilemma")

    for i in range(10):
        simulate_game(battle_of_the_sexes,
                      ['opera', 'football'],
                      ['opera', 'football'],
                      comments="Battle of the Sexes #{}".format(i))
