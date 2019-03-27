from display_caronthehill import *
from Domain import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.animation as animation
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD


class Agent:
    def __init__(self, discount_factor=0.95):
        self.domain = Domain()
        self.gamma = discount_factor
        self.approximation_function_Qn = []
        self.approximation_parametric_function_Qn = []
        self.buffer_one_step_transition = []

    def play(self, policy):
        """method to return the last reward of a play"""
        p, s = -0.5, 0
        r = 0
        number_of_action = 0
        while self.domain.is_terminal_state(p, s) is not True:
            u = policy.action(p, s)
            r = self.domain.reward(p, s, u)
            p, s = self.domain.next_state(p, s, u)
            number_of_action += 1
        return r, number_of_action

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
            self.buffer_one_step_transition.append(one_step_transition)
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

    def fitted_q_iteration(self, n, method="Linear_regression"):
        """
        method to learn the estimator of the q function
        :param n: index if the Qn needed
        :param method: string to choose the learning method (Linear_regression, Extremely_Randomized_Trees
                                                                                                or Neural_network)
        :return: The classifier Qn
        """

        # look if the function is already compute
        if len(self.approximation_function_Qn) > n:
            return self.approximation_function_Qn[n]
        else:
            i = len(self.approximation_function_Qn)

            while i < n:

                # Creation of the training set
                X = []
                y = []

                list_one_step_transition = self.buffer_one_step_transition

                if i == 0:
                    for one_step_transition in list_one_step_transition:
                        X.append([one_step_transition[0][0], one_step_transition[0][1], one_step_transition[1]])
                        y.append(one_step_transition[2])
                else:
                    for one_step_transition in list_one_step_transition:
                        X.append([one_step_transition[0][0], one_step_transition[0][1], one_step_transition[1]])
                        p = one_step_transition[3][0]
                        s = one_step_transition[3][1]
                        y.append(one_step_transition[2] + self.gamma *
                                 max(self.approximation_function_Qn[-1].predict(np.array([p, s, -4])
                                                                                .reshape(1, -1))[0],
                                     self.approximation_function_Qn[-1].predict(np.array([p, s, +4])
                                                                                .reshape(1, -1))[0]))

                # Learning the q function with the method asked
                if method == "Linear_regression":
                    self.approximation_function_Qn.append(LinearRegression().fit(X, y))

                elif method == "Extremely_Randomized_Trees":
                    self.approximation_function_Qn.append(ExtraTreesRegressor(n_estimators=10).fit(X, y))

                elif method == "Neural_network":

                    model = Sequential()
                    model.add(Dense(5, input_dim=3, activation='relu'))
                    model.add(Dense(5, activation='relu'))
                    model.add(Dense(5, activation='relu'))
                    model.add(Dense(5, activation='relu'))

                    if i == 0:
                        model.add(Dense(1, activation='sigmoid'))
                        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                    else:
                        model.add(Dense(1, activation='linear'))
                        opt = SGD(lr=0.01, momentum=0.9)
                        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])

                    X = np.array(X)
                    y = np.array(y)

                    model.fit(X, y, epochs=10, batch_size=10, verbose=0)

                    self.approximation_function_Qn.append(model)

                else:
                    raise ValueError("method unknown")

                i += 1
                print("Q" + str(i))

            return self.approximation_function_Qn[-1]

    # def parametric_q_learning(self, n, mathod="neural_network"):
    #     """
    #     method to learn the estimator of the q function
    #     :param n: index if the Qn needed
    #     :return: Qn
    #     """
    #     if len(self.approximation_parametric_function_Qn) > n:
    #         return self.approximation_parametric_function_Qn[n]
    #     else:
    #         i = len(self.approximation_parametric_function_Qn)
    #
    #         while i < n:
    #             pass

    def display_q_function(self, n, method="Linear_regression"):
        """
        method to display the estimator of the q function for the action 'forward', 'backward' and the max of
        the 'forward' and 'backward'
        :param n: index if the Qn to display
        :param method: string to choose the learning method (Linear_regression, Extremely_Randomized_Trees
                                                                                                or Neural_network)
        :return: plot three images
        """
        x = np.linspace(-1, 1, 200)
        y = np.linspace(-3, 3, 200)
        qn = self.fitted_q_iteration(n, method)
        vector_img_right = np.zeros((200, 200))
        vector_img_left = np.zeros((200, 200))
        vector_img = np.zeros((200, 200))

        for i in range(200):
            for j in range(200):
                vector_img_right[199 - j, i] = qn.predict(np.array([x[i], y[j], 4]).reshape(1, -1))[0]
                vector_img_left[199 - j, i] = qn.predict(np.array([x[i], y[j], -4]).reshape(1, -1))[0]
                if vector_img_left[199 - j, i] <= vector_img_right[199 - j, i]:
                    vector_img[199 - j, i] = 4
                else:
                    vector_img[199 - j, i] = -4

        cs = plt.contourf(x, y, vector_img_right, cmap='Spectral')
        plt.colorbar(cs)
        plt.xlabel("position")
        plt.ylabel("speed")
        plt.show()
        plt.clf()

        cs2 = plt.contourf(x, y, vector_img_left, cmap='Spectral')
        plt.colorbar(cs2)
        plt.xlabel("position")
        plt.ylabel("speed")
        plt.show()
        plt.clf()

        cs3 = plt.contourf(x, y, vector_img, cmap='Spectral')
        plt.colorbar(cs3)
        plt.xlabel("position")
        plt.ylabel("speed")
        plt.show()
