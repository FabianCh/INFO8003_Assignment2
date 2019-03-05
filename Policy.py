import random as rdm


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
