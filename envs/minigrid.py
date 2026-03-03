import gymnasium as gym
import numpy as np


TASK_MAP = {
    'keycorridor': 'MiniGrid-DoorKey-16x16-v0',
    'keycorridor8': 'MiniGrid-DoorKey-8x8-v0',
    'keycorridor6': 'MiniGrid-DoorKey-6x6-v0',
    'fourrooms': 'MiniGrid-FourRooms-v0',
    'memory13': 'MiniGrid-MemoryS13-v0',
    'empty16': 'MiniGrid-Empty-16x16-v0',
}


class MiniGrid(gym.Env):
    metadata = {}

    def __init__(self, task, size=(64, 64), seed=0):
        import pyglet
        pyglet.options['headless'] = True
        from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
        import gym as old_gym

        env_id = TASK_MAP[task]
        self._env = ImgObsWrapper(RGBImgPartialObsWrapper(old_gym.make(env_id)))
        self._env.seed(seed)
        self._size = size
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        img_shape = self._size + (3,)
        return gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, img_shape, dtype=np.uint8),
        })

    @property
    def action_space(self):
        return gym.spaces.Discrete(self._env.action_space.n)

    def step(self, action):
        image, reward, done, info = self._env.step(action)
        image = self._resize(image)
        obs = {
            "image": image,
            "is_first": False,
            "is_last": done,
            "is_terminal": done,
        }
        return obs, np.float32(reward), done, info

    def reset(self):
        image = self._env.reset()
        image = self._resize(image)
        return {
            "image": image,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
        }

    def _resize(self, image):
        from PIL import Image
        img = Image.fromarray(image)
        img = img.resize(self._size, Image.NEAREST)
        return np.array(img)
