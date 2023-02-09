from manifpy import SE2
import numpy as np

from visualize import visualize_prediction

space_size = 7


def griper_positions(box_position):
    results = []
    b = SE2(*box_position)
    for i in range(4):
        rot_angle = (i/2)*np.pi
        rot = SE2(0, 0, rot_angle)
        t = SE2(-0.5, 0, 0)
        g = (b*rot)*t
        if 0.6 < g.x() < space_size - 0.6 and 0.6 < g.y() < space_size - 0.6:
            results.append((g.x(), g.y(), g.angle()))
    # if len(results) ==  1:
    #     results *= 4
    return results


def make_random_data(n: int) -> np.ndarray:
    random_data = np.random.rand(n, 3).astype(np.float32)
    random_data[:, :2] *= 3
    random_data[:, :2] += 0.5
    random_data[:, 2] -= 0.5
    random_data[:, 2] *= 2*np.pi
    return random_data


def make_grid_data():
    objs = np.array([(2*row+0.5, 2*col+0.5, 0) for row in range((space_size+1)//2) for col in range((space_size+1)//2)], np.float32)
    return objs


def get_data_and_labels(data: np.ndarray):
    n = data.shape[0]
    new_data = []
    labels = []
    for i in range(n):
        for label in griper_positions(data[i, :]):
            new_data.append(data[i, :])
            labels.append(label)
    return np.array(new_data, np.float32), np.array(labels, np.float32)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    data = make_grid_data()
    data, labels = get_data_and_labels(data)

    fig, ax = plt.subplots()
    visualize_prediction(ax, data, correct_grippers=labels)
    ax.set(xlim=(0, space_size), ylim=(0, space_size))
    plt.show()
