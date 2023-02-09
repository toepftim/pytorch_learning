from typing import Optional
from manifpy import SE2
import numpy as np
import matplotlib.pyplot as plt

box_size = 1
gripper_length = 0.3
gripper_opening = 0.5


def draw_box(axs, x, y, angle, color=None, width=None):
    color = color if color is not None else "gray"
    width = width if width is not None else 4

    coords = [
        (x+0.5*box_size*np.sqrt(2)*np.cos(angle+(2*i+1)*np.pi/4),
         y+0.5*box_size*np.sqrt(2)*np.sin(angle+(2*i+1)*np.pi/4))
        for i in range(4)
    ]

    axs.add_patch(plt.Polygon(coords, fc=color, ec="white", linewidth=width))


def gripper_lines(flange: SE2):
    """Return tuple of lines (start-end point) that are used to plot gripper attached to the flange frame."""
    return (
        (flange * SE2(0, -gripper_opening, 0)).translation(),
        (flange * SE2(0, +gripper_opening, 0)).translation()
    ), (
        (flange * SE2(0, -gripper_opening, 0)).translation(),
        (flange * SE2(gripper_length, -gripper_opening, 0)).translation()
    ), (
        (flange * SE2(0, +gripper_opening, 0)).translation(),
        (flange * SE2(gripper_length, +gripper_opening, 0)).translation()
    ), (
        (flange * SE2(-gripper_length, 0, 0)).translation(),
        flange.translation()
    )


def draw_gripper(axs, x, y, angle, color=None, width=None):
    color = color if color is not None else "gray"
    width = width if width is not None else 4
    for line in gripper_lines(SE2(x, y, angle)):
        axs.plot((line[0][0], line[1][0]), (line[0][1], line[1][1]), c=color, linewidth=width)


def visualize_prediction(axs, objects: iter, correct_grippers: Optional[iter] = None, predicted_grippers: Optional[iter] = None):
    for box in objects:
        draw_box(axs, *box, width=6)
    if correct_grippers is not None:
        for gripper in correct_grippers:
            draw_gripper(axs, *gripper, color="green", width=6)
    if predicted_grippers is not None:
        for gripper in predicted_grippers:
            draw_gripper(axs, *gripper, color="red", width=4)
    axs.set_title(f"{len(objects)} vzorků")


if __name__ == '__main__':
    fig, axs = plt.subplots()
    visualize_prediction(axs, ((1, 1, 0),), ((0.5, 1, 0),), ((0.4, 1.1, 0.1),))
    plt.xlim((0, 5))
    plt.ylim((0, 5))
    plt.show()
