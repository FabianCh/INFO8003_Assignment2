import math


class Domain:
    ACTION_SPACE = [-4, 4]

    def __init__(self, discount_factor=0.95):
        self.gamma = discount_factor
        self.integration_time_step = 0.001
        self.discretizing_time_step = 0.100
        self.m = 1
        self.g = 9.81
        self.B = 1

    @staticmethod
    def is_valid_state_space(p, s):
        """method to know if a state is valid."""
        if abs(p) <= 1 and abs(s) <= 3:
            return True
        else:
            return False

    @staticmethod
    def is_terminal_state(p, s):
        """method to know if a terminal state is reach"""
        if abs(p) > 1 or abs(s) > 3:
            return True
        else:
            return False

    @staticmethod
    def hill(p):
        """method to compute the position of the hill for a given position od the car"""
        if p < 0:
            return p**2 + p
        else:
            return p/((1 + 5 * p**2)**(1/2))

    @staticmethod
    def hill_prime(p):
        """method to compute the derivative of the position of the hill for a given position od the car"""
        if p < 0:
            return 2*p + 1
        else:
            return 1 / ((1 + 5 * p**2)**(3 / 2))

    @staticmethod
    def hill_second(p):
        """method to compute the second derivative of the position of the hill for a given position od the car"""
        if p < 0:
            return 2
        else:
            return (-15 * p) / (1 + 5 * p**2)**(5 / 2)

    def derivative(self, p, s, u):
        """method to compute the acceleration for a given action"""
        return s, (u / (self.m * (1 + self.hill_prime(p)**2))) - \
               ((self.g * self.hill_prime(p)) / (1 + self.hill_prime(p)**2)) - \
               ((s**2 * self.hill_prime(p) * self.hill_second(p)) / (1 + self.hill_prime(p)**2))

    def next_state(self, p, s, u):
        """method to return the next state for a given state and a given action"""

        def euler_method(pt, st, ut):
            """function to return the next position for a given state and a given action"""
            p_prime, s_prime = self.derivative(pt, st, ut)
            return pt + p_prime * self.integration_time_step, st + s_prime * self.integration_time_step, ut
        
        time_counter = 0
        while time_counter < self.discretizing_time_step:
            p, s, u = euler_method(p, s, u)
            time_counter += self.integration_time_step
        return p, s

    def reward(self, pt, st, ut):
        """method to compute the reward for a given state and action"""
        next_pt, next_st = self.next_state(pt, st, ut)
        if next_pt < -1 or abs(next_st) > 3:
            return -1
        elif next_pt > 1 and abs(next_st) <= 3:
            return 1
        else:
            return 0
