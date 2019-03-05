from Agent import *
from Domain import *
from Policy import *

domain = Domain()
agent = Agent()
#policy = StaticPolicy(4)
policy = RandomPolicy()

print(agent.play(domain, policy))
print(agent.expected_return(domain, policy))
