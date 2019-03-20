import random as rdm
import numpy as np


class Policy:
    def action(self, p, s):
        pass


class StaticPolicy(Policy):
    def __init__(self, u):
        self.u = u

    def action(self, p, s):
        return self.u


class RandomPolicy(Policy):
    def action(self, p, s):
        return rdm.choice([-4, 4])


class OptimalPolicy(Policy):
    def __init__(self, function_qn_approximate):
        self.function_Qn_approximate = function_qn_approximate

    def action(self, p, s):
        right = self.function_Qn_approximate.predict(np.array([p, s, 4]).reshape(1, -1))[0]
        left = self.function_Qn_approximate.predict(np.array([p, s, -4]).reshape(1, -1))[0]

        if type(right) is list():       # scikit-learn classifier and keras classifier have not the same structure
            right = right[0]
            left = left[0]

        if right >= left:
            return 4
        else:
            return -4
