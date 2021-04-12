import gym
import imageio as im
env = gym.envs.make('Asterix-v0')
img = env.render(mode="rgb_array")
img = im.imread(img)
im.imsave(img, './sample.png')
obs = env.reset()
print(obs[2])