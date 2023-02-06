from manifpy import SE2
import numpy as np


def griper_position(box_position):
    b = SE2(*box_position)
    rot = SE2(0, 0, np.pi)
    t = SE2(-1, 0, 0)
    g = (b*rot)*t
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
    import matplotlib.pyplot as plt
    example_count = 10
    data = make_random_data(example_count)
    labels = get_labels(data)
    print(data)
    print(labels)

    fig, ax = plt.subplots()
    for i in range(example_count):
        ax.arrow(data[i, 0], data[i, 1], np.cos(data[i, 2]), np.sin(data[i, 2]), width=0.1, length_includes_head=True, color="brown")
        ax.arrow(labels[i, 0], labels[i, 1], np.cos(labels[i, 2]), np.sin(labels[i, 2]), width=0.1, length_includes_head=True, color="gray")
    ax.set(xlim=(-1, 10), ylim=(-1, 10))
    plt.show()

