import numpy as np

from pygtm.physical import physical_space


def test_uniform_grid():
    nx, ny = physical_space.uniform_grid([0, 10], [0, 10], 10)
    assert (nx, ny) == (10, 10)

    nx, ny = physical_space.uniform_grid([0, 10], [0, 20], 20)
    assert (nx, ny) == (10, 20)

    nx, ny = physical_space.uniform_grid([0, 20], [0, 10], 20)
    assert (nx, ny) == (20, 10)

    nx, ny = physical_space.uniform_grid([0, 30], [0, 10], 20)
    assert (nx, ny) == (20, 6)

    nx, ny = physical_space.uniform_grid([0, 10], [0, 30], 20)
    assert (nx, ny) == (6, 20)

    nx, ny = physical_space.uniform_grid([-15, 15], [-2, 8], 20)
    assert (nx, ny) == (20, 6)

    nx, ny = physical_space.uniform_grid([-9, 1], [-19, 11], 20)
    assert (nx, ny) == (6, 20)


def test_create_grid():
    # create a small grid of 6 elements (3 in lon and 2 in lat)
    coords, elements, x, y, dx, dy = physical_space.create_grid([0, 10], [0, 2], 4, 3)
    assert coords[:, 0].tolist() == [
        0,
        10 / 3,
        20 / 3,
        10,
        0,
        10 / 3,
        20 / 3,
        10,
        0,
        10 / 3,
        20 / 3,
        10,
    ]
    assert coords[:, 1].tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    assert elements.tolist() == [
        [0, 1, 4, 5],
        [1, 2, 5, 6],
        [2, 3, 6, 7],
        [4, 5, 8, 9],
        [5, 6, 9, 10],
        [6, 7, 10, 11],
    ]
    assert (x.tolist(), y.tolist()) == (
        np.linspace(0, 10, 4, endpoint=True).tolist(),
        np.linspace(0, 2, 3, endpoint=True).tolist(),
    )
    assert (dx, dy) == (10 / 3, 1)

    # negative values in lon lat
    # create a small grid of 8 elements (4 in lon and 2 in lat)
    coords, elements, x, y, dx, dy = physical_space.create_grid([-5, 5], [-1, 1], 5, 3)
    assert coords[:, 0].tolist() == [
        -5,
        -2.5,
        0,
        2.5,
        5,
        -5,
        -2.5,
        0,
        2.5,
        5,
        -5,
        -2.5,
        0,
        2.5,
        5,
    ]
    assert coords[:, 1].tolist() == [
        -1,
        -1,
        -1,
        -1,
        -1,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
    ]
    assert elements.tolist() == [
        [0, 1, 5, 6],
        [1, 2, 6, 7],
        [2, 3, 7, 8],
        [3, 4, 8, 9],
        [5, 6, 10, 11],
        [6, 7, 11, 12],
        [7, 8, 12, 13],
        [8, 9, 13, 14],
    ]
    assert (x.tolist(), y.tolist()) == (
        np.linspace(-5, 5, 5, endpoint=True).tolist(),
        np.linspace(-1, 1, 3, endpoint=True).tolist(),
    )
    assert dx == 2.5
    assert dy == 1


def test_find_element():
    lon = [0, 3]
    lat = [0, 3]
    n = 4
    d = physical_space(lon, lat, n)

    # one element
    x = np.array([0])
    y = np.array([0])
    assert d.find_element(x, y).tolist() == [0]

    # test one bin
    x = np.array([0, 0.5, 0.25])
    y = np.array([0, 0.5, 0.75])
    assert d.find_element(x, y).tolist() == [0, 0, 0]

    x = np.array([2.5, 2.7])
    y = np.array([2.5, 2.2])
    assert d.find_element(x, y).tolist() == [8, 8]

    # top right corners
    x = np.array([0, 1, 2, 3])
    y = np.array([0, 1, 2, 3])
    assert d.find_element(x, y).tolist() == [0, 0, 4, 8]

    x = np.array([0.5, 1.5, 2.5])
    y = np.array([0.5, 0.5, 0.5])
    assert d.find_element(x, y).tolist() == [0, 1, 2]

    # left boundaries
    x = np.array([0, 0, 0, 0])
    y = np.array([0.5, 1.5, 3, 5])
    assert d.find_element(x, y).tolist() == [0, 3, 6, -1]

    # right boundaries
    x = np.array([3, 3, 3, 3.1])
    y = np.array([0.5, 1.1, 2.34, 5])
    assert d.find_element(x, y).tolist() == [2, 5, 8, -1]

    # bottom boundaries
    x = np.array([0.5, 1.5, 3, 5])
    y = np.array([0, 0, 0, 0])
    assert d.find_element(x, y).tolist() == [0, 1, 2, -1]

    # top boundaries
    x = np.array([0.3, 1.2, 2.67, -0.5])
    y = np.array([3, 3, 3, 3])
    assert d.find_element(x, y).tolist() == [6, 7, 8, -1]

    # random
    x = np.array([2.1, 3, 1.6, 3])
    y = np.array([2.4, 1.3, 3, 3.5])
    assert d.find_element(x, y).tolist() == [8, 5, 7, -1]

    # outside with inside points
    x = np.array([-10, 10, 1])
    y = np.array([-10, 10, 1])
    assert d.find_element(x, y).tolist() == [-1, -1, 0]

    # all outside
    x = np.array([-10, 10])
    y = np.array([-10, 10])
    assert d.find_element(x, y).tolist() == [-1, -1]
