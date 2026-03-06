import numpy as np


class RandomController:
    def __init__(self):
        pass

    def __call__(self, model, t):
        """Compute control action.

        Args:
            model: SCONE model
            t: Current simulation time

        Returns:
            input_array: Control inputs for the exoskeleton actuators"""
        # x = model.dofs()[0].pos()

        input_array = np.zeros(4)
        input_array[0] = 1 * np.random.uniform(-1, 1)  # exo hip flexion right
        input_array[1] = 1 * np.random.uniform(-1, 1)  # exo hip flexion left
        input_array[2] = 1 * np.random.uniform(-1, 1)  # exo knee flexion right
        input_array[3] = 1 * np.random.uniform(-1, 1)  # exo knee flexion left

        input_array = np.zeros(4)
        input_array[0] = -100

        return input_array
