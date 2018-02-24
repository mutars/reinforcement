import gym

env = gym.make('Breakout-v0')
env.reset()
env.step(1)

while True:
    observ, reward, done, info = env.step(2)
    env.step(1)
    env.render()
    if done:
        break
