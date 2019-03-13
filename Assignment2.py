from Agent import *
from Domain import *
from Policy import *

domain = Domain()
agent = Agent()
print("\n")

policy = StaticPolicy(4)
print("Final result of a right policy :")
print("     " + str(agent.play(policy)) + "\n")

policy = RandomPolicy()
print("Expected return of a random policy :")
print("     " + str(agent.expected_return(policy)) + "\n")

print("list of one step transition of a random policy")
print("     " + str(agent.one_step_transition_generator(policy)) + "\n")

# print("Generating the animation ...")
# policy = RandomPolicy()
# agent.show(policy)
# print("Generation ended")
