import collections

import gymnasium as gym
import numpy as np


class PinPad(gym.Env):
    metadata = {}

    COLORS = {
        '1': (255,   0,   0),
        '2': (  0, 255,   0),
        '3': (  0,   0, 255),
        '4': (255, 255,   0),
        '5': (255,   0, 255),
        '6': (  0, 255, 255),
        '7': (128,   0, 128),
        '8': (  0, 128, 128),
    }

    def __init__(self, task, length=2000, seed=0):
        assert length > 0
        layout = {
            'three': LAYOUT_THREE,
            'four': LAYOUT_FOUR,
            'five': LAYOUT_FIVE,
            'six': LAYOUT_SIX,
            'seven': LAYOUT_SEVEN,
            'eight': LAYOUT_EIGHT,
        }[task]
        self.layout = np.array([list(line) for line in layout.split('\n')]).T
        assert self.layout.shape == (16, 14), self.layout.shape
        self.length = length
        self.random = np.random.RandomState(seed)
        self.pads = set(self.layout.flatten().tolist()) - set('* #\n')
        self.target = tuple(sorted(self.pads))
        self.spawns = []
        for (x, y), char in np.ndenumerate(self.layout):
            if char != '#':
                self.spawns.append((x, y))
        print(f'Created PinPad env with sequence: {"->".join(self.target)}')
        self.sequence = collections.deque(maxlen=len(self.target))
        self.player = None
        self.steps = None
        self.done = None
        self.countdown = None
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        return gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
        })

    @property
    def action_space(self):
        return gym.spaces.Discrete(5)

    def step(self, action):
        reward = 0.0
        if self.countdown:
            self.countdown -= 1
            if self.countdown == 0:
                self.player = self.spawns[self.random.randint(len(self.spawns))]
                self.sequence.clear()
        move = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)][action]
        x = np.clip(self.player[0] + move[0], 0, 15)
        y = np.clip(self.player[1] + move[1], 0, 13)
        tile = self.layout[x][y]
        if tile != '#':
            self.player = (x, y)
        if tile in self.pads:
            if not self.sequence or self.sequence[-1] != tile:
                self.sequence.append(tile)
        if tuple(self.sequence) == self.target and not self.countdown:
            reward += 10.0
            self.countdown = 10
        self.steps += 1
        self.done = self.done or (self.steps >= self.length)
        obs = {
            "image": self._render_obs(),
            "display_image": self._render_display(),
            "is_first": False,
            "is_last": self.done,
            "is_terminal": False,
        }
        return obs, np.float32(reward), self.done, {}

    def reset(self):
        self.player = self.spawns[self.random.randint(len(self.spawns))]
        self.sequence.clear()
        self.steps = 0
        self.done = False
        self.countdown = 0
        return {
            "image": self._render_obs(),
            "display_image": self._render_display(),
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
        }

    def _render_grid(self):
        """Render the base 16x16 grid without progress bar."""
        grid = np.zeros((16, 16, 3), np.uint8) + 255
        white = np.array([255, 255, 255])
        if self.countdown:
            grid[:] = (223, 255, 223)
        current = self.layout[self.player[0]][self.player[1]]
        for (x, y), char in np.ndenumerate(self.layout):
            if char == '#':
                grid[x, y] = (192, 192, 192)
            elif char in self.pads:
                color = np.array(self.COLORS[char])
                color = color if char == current else (10 * color + 90 * white) / 100
                grid[x, y] = color
        grid[self.player] = (0, 0, 0)
        # Fill bar area with wall color (agent never sees progress)
        grid[:, -2:] = (192, 192, 192)
        return grid

    def _render_obs(self):
        """Agent observation: 64x64, no progress bar."""
        grid = self._render_grid()
        image = np.repeat(np.repeat(grid, 4, 0), 4, 1)
        return image.transpose((1, 0, 2))

    def _render_display(self):
        """Display image for videos: agent view + red separator + progress bar.

        Returns (64, 76, 3) uint8 — wider than the agent's observation to make
        it obvious the progress strip is not part of the agent's input.
        """
        grid = self._render_grid()
        # Build progress bar strip (16x2)
        bar = np.full((16, 2, 3), 48, dtype=np.uint8)  # dark background
        for i, char in enumerate(self.sequence):
            bar[2 * i + 1, 0] = self.COLORS[char]
        # Separator column (16x1) in red
        sep = np.full((16, 1, 3), 0, dtype=np.uint8)
        sep[..., 0] = 255  # red
        # Concatenate: grid (16x16) | sep (16x1) | bar (16x2)
        combined = np.concatenate([grid, sep, bar], axis=1)  # (16, 19, 3)
        image = np.repeat(np.repeat(combined, 4, 0), 4, 1)  # (64, 76, 3)
        return image.transpose((1, 0, 2))

    def render(self):
        return self._render_display()


LAYOUT_THREE = """
################
#1111      3333#
#1111      3333#
#1111      3333#
#1111      3333#
#              #
#              #
#              #
#              #
#     2222     #
#     2222     #
#     2222     #
#     2222     #
################
""".strip('\n')

LAYOUT_FOUR = """
################
#1111      4444#
#1111      4444#
#1111      4444#
#1111      4444#
#              #
#              #
#              #
#              #
#3333      2222#
#3333      2222#
#3333      2222#
#3333      2222#
################
""".strip('\n')

LAYOUT_FIVE = """
################
#          4444#
#111       4444#
#111       4444#
#111           #
#111        555#
#           555#
#           555#
#333        555#
#333           #
#333       2222#
#333       2222#
#          2222#
################
""".strip('\n')

LAYOUT_SIX = """
################
#111        555#
#111        555#
#111        555#
#              #
#33          66#
#33          66#
#33          66#
#33          66#
#              #
#444        222#
#444        222#
#444        222#
################
""".strip('\n')

LAYOUT_SEVEN = """
################
#111        444#
#111        444#
#11          44#
#              #
#33          55#
#33          55#
#33          55#
#33          55#
#              #
#66          22#
#666  7777  222#
#666  7777  222#
################
""".strip('\n')

LAYOUT_EIGHT = """
################
#111  8888  444#
#111  8888  444#
#11          44#
#              #
#33          55#
#33          55#
#33          55#
#33          55#
#              #
#66          22#
#666  7777  222#
#666  7777  222#
################
""".strip('\n')
