from grid_maze.dynamics import (
    DOWN, LEFT, RIGHT, UP, cell_to_xy, move, parse_observation, reward_and_done, xy_to_cell,
)


def test_up_decreases_y_and_right_increases_x():
    assert move(2, 2, UP, 5) == (2, 1)
    assert move(2, 2, DOWN, 5) == (2, 3)
    assert move(2, 2, LEFT, 5) == (1, 2)
    assert move(2, 2, RIGHT, 5) == (3, 2)


def test_walls_keep_agent_in_place():
    assert move(0, 0, UP, 5) == (0, 0)
    assert move(0, 0, LEFT, 5) == (0, 0)
    assert move(4, 4, DOWN, 5) == (4, 4)
    assert move(4, 4, RIGHT, 5) == (4, 4)


def test_cell_index_roundtrip():
    for c in range(25):
        assert xy_to_cell(*cell_to_xy(c, 5), 5) == c


def test_rewards():
    assert reward_and_done((1, 1), [(1, 1)], [(2, 2)]) == (1.0, True)
    assert reward_and_done((2, 2), [(1, 1)], [(2, 2)]) == (-1.0, True)
    assert reward_and_done((0, 0), [(1, 1)], [(2, 2)]) == (0.0, False)


def test_parse_observation():
    obs = [0, 1, 2, 3, 4, 0, 1, 1]
    assert parse_observation(obs, 1, 2) == ((0, 1), [(2, 3)], [(4, 0), (1, 1)])
