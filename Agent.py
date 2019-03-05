from display_caronthehill import *


class Agent:
    def __init__(self):
        self.p = 0
        self.s = 1

    def play(self, domain, policy):
        p, s = -0.5, 0
        """method to return the last reward of a play"""
        u = policy.action(p, s)
        while domain.is_terminal_state(p, s) is not True:
            u = policy.action(p, s)
            r = domain.reward(p, s, u)
            p, s = domain.next_state(p, s, u)
        return r

    def show(self, domain, policy):
        save_caronthehill_image(0, 1, "out.jpeg")

    def expected_return(self, domain, policy, n=100):
        """method to return the expected value with a policy in a domain"""
        cumulated_reward = 0
        for _ in range(n):
            cumulated_reward += self.play(domain, policy)
            print(cumulated_reward)
        cumulated_reward /= n
        return cumulated_reward
