from Agent import *
from Domain import *
from Policy import *

domain = Domain()
agent = Agent()

policy = StaticPolicy(4)
print("Final result of a right policy :")
print("     " + str(agent.play(policy)))

policy = RandomPolicy()
print("Expected return of a random policy :")
print("     " + str(agent.expected_return(policy)))

# print("Generating the animation ...")
# policy = RandomPolicy()
# print(agent.show(policy))
# print("Generation ended")
