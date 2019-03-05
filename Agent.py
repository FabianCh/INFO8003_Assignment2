from display_caronthehill import *


class Agent:
    def __init__(self):
        self.p = 0
        self.s = 1

    def play(self, domain, policy):
        u = policy.action(self.p, self.s)
        while domain.is_terminal_state(self.p, self.s) is not True:
            u = policy.action(self.p, self.s)
            self.p, self.s = domain.next_state(self.p, self.s, u)
        return domain.reward(self.p, self.s, u)

    def show(self, domain, policy):
        save_caronthehill_image(self.p, self.s, "out.jpeg")

    def expected_return_iterated(self, domain, policy, n):
        """method to return the Expected value after N turn with a policy in a domain"""
        expected_return = 0
        i = 0
        while i < n:
            u = policy.action(self.p, self.s)
            reward = domain.reward(self.p, self.s, u)
            self.p, self.s = domain.next_state(self.p, self.s, u)
            expected_return = reward + domain.gamma * expected_return
        return expected_return

    def expected_return(self, domain, policy, error=0.01):
        n = 0
        while ((domain.gamma ** n) * domain.B) / (1 - domain.gamma) > error:
            n += 1
        return self.expected_return_iterated(domain, policy, n)
