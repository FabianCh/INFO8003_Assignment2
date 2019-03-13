from display_caronthehill import *
from Domain import *
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.animation as animation


class Agent:
    def __init__(self, discount_factor=0.95):
        self.domain = Domain()
        self.gamma = discount_factor

    def play(self, policy):
        """method to return the last reward of a play"""
        p, s = -0.5, 0
        r = 0
        while self.domain.is_terminal_state(p, s) is not True:
            u = policy.action(p, s)
            r = self.domain.reward(p, s, u)
            p, s = self.domain.next_state(p, s, u)
        return r

    def one_step_transition_generator(self, policy):
        """method to return the list of one step transition of a play"""
        p, s = -0.5, 0
        list_one_step_transition = []
        while self.domain.is_terminal_state(p, s) is not True:
            u = policy.action(p, s)
            r = self.domain.reward(p, s, u)
            one_step_transition = [[p, s], u, r]
            p, s = self.domain.next_state(p, s, u)
            one_step_transition.append([p, s])
            list_one_step_transition.append(one_step_transition)
        return list_one_step_transition

    def show(self, policy):
        """method to create the animation of the result of a given policy"""
        p, s = -0.5, 0
        fig = plt.figure()

        # Generation of images
        counter = 0
        while self.domain.is_terminal_state(p, s) is not True:
            u = policy.action(p, s)
            save_caronthehill_image(p, s, "image\\state" + str(counter) + ".png")
            p, s = self.domain.next_state(p, s, u)
            counter += 1

        # Loading of images
        ims = []
        for i in range(counter):
            image = img.imread("image\\state" + str(i) + ".png")
            im = plt.imshow(image, animated=True)
            ims.append([im])

        # Creation of the animation
        ani = animation.ArtistAnimation(fig, ims, interval=100)
        ani.save("movie.html", writer='html')
        plt.show()

    def expected_return(self, policy, n=100):
        """method to return the expected value with a policy in a domain"""
        cumulative_reward = 0
        for _ in range(n):
            cumulative_reward += self.play(policy)
        cumulative_reward /= n
        return cumulative_reward
