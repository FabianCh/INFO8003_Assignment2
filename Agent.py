class Agent:
    def __init__(self):
        self.p = 0
        self.s = 1

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
