import numpy as np

from zero_sum_game import ZeroSumGame

SEED = 1337


def simulate_game(get_score, first_actions, second_actions, iters=10 ** 4, comments=None):
    def reverse_score(first_act, second_act):
        return -get_score(second_act, first_act)

    first = ZeroSumGame(get_score, first_actions, second_actions)
    second = ZeroSumGame(reverse_score, second_actions, first_actions)

    score = 0
    for it in range(iters):
        a = first.gen_action()
        b = second.gen_action()
        score += get_score(a, b)

    if comments is not None:
        print(comments)
    print(f'Average score: {score / iters}')
    print(f'Expected game price: {first.price}')
    print('-' * 80)


FIGHTS = {
    ('rock', 'scissors'),
    ('paper', 'rock'),
    ('scissors', 'paper')
}

ACTIONS = ('rock', 'paper', 'scissors')


def rock_paper_scissors(first_act, second_act):
    assert first_act in ACTIONS and second_act in ACTIONS
    if (first_act, second_act) in FIGHTS:
        return 1
    if (second_act, first_act) in FIGHTS:
        return -1
    return 0


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
