"""PyGame rendering for the Grid Maze.

Supports both Gymnasium render modes:
  * "human"     -> opens a window and draws in real time
  * "rgb_array" -> draws off-screen and returns an (H, W, 3) uint8 array
                   (required by the RecordVideo wrapper)
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")  # we never play sound; silences ALSA warnings

import numpy as np  # noqa: E402
import pygame  # noqa: E402

WHITE = (255, 255, 255)
GRID = (60, 60, 60)
START = (255, 235, 59)   # yellow  "S"
GOAL = (46, 160, 67)     # green   "G"
BAD = (220, 38, 38)      # red     "X"
AGENT = (37, 99, 235)    # blue circle


class GridMazeRenderer:
    def __init__(self, size: int, render_mode: str, fps: int = 4, cell_px: int = 96):
        self.size, self.render_mode, self.fps, self.cell = size, render_mode, fps, cell_px
        self.px = size * cell_px  # 5 * 96 = 480 (divisible by 16 -> clean video encoding)

        pygame.init()
        pygame.font.init()
        self.font = pygame.font.Font(None, int(cell_px * 0.6))
        self.surface = pygame.Surface((self.px, self.px))
        self.window = self.clock = None
        if render_mode == "human":
            pygame.display.init()
            pygame.display.set_caption("Grid Maze")
            self.window = pygame.display.set_mode((self.px, self.px))
            self.clock = pygame.time.Clock()

    # ---------------------------------------------------------------- drawing
    def _rect(self, cell):
        x, y = cell
        return pygame.Rect(x * self.cell, y * self.cell, self.cell, self.cell)

    def _label(self, cell, text, color):
        pygame.draw.rect(self.surface, color, self._rect(cell))
        img = self.font.render(text, True, (0, 0, 0))
        self.surface.blit(img, img.get_rect(center=self._rect(cell).center))

    def draw(self, agent, start, goals, bads):
        self.surface.fill(WHITE)
        self._label(start, "S", START)
        for g in goals:
            self._label(g, "G", GOAL)
        for b in bads:
            self._label(b, "X", BAD)

        for i in range(self.size + 1):  # grid lines
            p = i * self.cell
            pygame.draw.line(self.surface, GRID, (p, 0), (p, self.px), 2)
            pygame.draw.line(self.surface, GRID, (0, p), (self.px, p), 2)

        pygame.draw.circle(self.surface, AGENT, self._rect(agent).center, int(self.cell * 0.28))

        if self.render_mode == "human":
            self.window.blit(self.surface, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.fps)
            return None
        # pygame surfaces are (W, H, 3); video/image libraries want (H, W, 3)
        return np.transpose(pygame.surfarray.array3d(self.surface), (1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
        pygame.quit()
