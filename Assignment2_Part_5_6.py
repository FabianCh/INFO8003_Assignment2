from Agent import *
from Domain import *
from Policy import *


domain = Domain()
agent = Agent()
print("\n")

print("Generation of the buffer...")
random_policy = RandomPolicy()

for _ in range(20):
    agent.one_step_transition_generator(random_policy)
print("buffer generated")
static_policy = StaticPolicy(4)
agent.one_step_transition_generator(static_policy)

len_buffer = len(agent.buffer_one_step_transition)
print(len_buffer)

print("learning Q function...")
# agent.learning_q_iteration(300)
# agent.learning_q_iteration(24, method="Extremely_Randomized_Trees")
agent.learning_q_iteration(21, method="Neural_network")
print("Q function learnt")

print(agent.approximation_function_Qn[0].predict(np.array([0, 1, 4]).reshape(1, -1))[0])
print(agent.approximation_function_Qn[-1].predict(np.array([0, 1, 4]).reshape(1, -1))[0])

optimal_policy = OptimalPolicy(agent.approximation_function_Qn[-1])
print(agent.play(optimal_policy))
agent.show(optimal_policy)
