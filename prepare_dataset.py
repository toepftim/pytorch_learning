from manifpy import SE2
import numpy as np


def griper_position(box_position):
    x, y, phi = box_position
    b = SE2(x, y, phi)
    t = SE2(-1, 0, 0)
    g = b*t
    return g.x(), g.y(), g.angle()


def make_random_data(n: int) -> np.ndarray:
    data = np.random.rand(n, 3).astype(np.float32)
    data[:, :2] *= 10
    data[:, 2] -= 0.5
    data[:, 2] *= 2*np.pi
    return data


def get_labels(data: np.ndarray):
    n = data.shape[0]
    labels = np.zeros(data.shape, np.float32)
    for i in range(n):
        labels[i, :] = griper_position(data[i, :])
    return labels


if __name__ == '__main__':
    data = make_random_data(2)
    labels = get_labels(data)
    print(data, labels)

