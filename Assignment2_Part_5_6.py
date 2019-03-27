from Agent import *
from Domain import *
from Policy import *


domain = Domain()
agent = Agent()
print("\n")

print("Generation of the buffer...")
random_policy = RandomPolicy()

for _ in range(30):
    agent.one_step_transition_generator(random_policy)
print("buffer generated")
static_policy = StaticPolicy(4)
agent.one_step_transition_generator(static_policy)

len_buffer = len(agent.buffer_one_step_transition)
print(len_buffer)

print("learning Q function...")
# agent.fitted_q_iteration(25, method="Linear_regression")
agent.fitted_q_iteration(24, method="Extremely_Randomized_Trees")
# agent.fitted_q_iteration(25, method="Neural_network")
print("Q function learnt")

optimal_policy = OptimalPolicy(agent.approximation_function_Qn[-1])
print(agent.play(optimal_policy))
agent.show(optimal_policy)


# agent.display_q_function(25, method="Linear_regression")
agent.display_q_function(25, method="Extremely_Randomized_Trees")
# agent.display_q_function(25, method="Neural_network")

