from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from shapely import affinity
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree


class SafeDrivingWrapper(gym.Wrapper):
    """
    Cria uma 'zona segura' ao redor da pista e aplica o reward shaping:

    - fora da zona segura: penalidade grande e fim imediato do episódio;
    - dentro: bônus proporcional à velocidade e penalidade se estiver parado.

    Os polígonos da pista são construídos uma única vez por `reset` (STRtree),
    e a checagem roda uma única vez por passo (resultado em `info["on_track"]`).
    """
    def __init__(
        self, env, border_width=0.5, off_track_penalty=100.0,
        speed_bonus_scale=0.1, low_speed_threshold=1.0, low_speed_penalty=1.0
    ):
        super().__init__(env)
        self.border_width = border_width
        self.off_track_penalty = off_track_penalty
        self.speed_bonus_scale = speed_bonus_scale
        self.low_speed_threshold = low_speed_threshold
        self.low_speed_penalty = low_speed_penalty
        self._tree = None
        self._on_track = True

    def _build_track(self):
        polygons = []
        for poly in self.unwrapped.road_poly:
            polygon = Polygon(poly[0])
            if self.border_width > 0:
                scale = 1 + self.border_width
                polygon = affinity.scale(polygon, xfact=scale, yfact=scale)
            polygons.append(polygon)
        self._tree = STRtree(polygons)

    def _check_on_track(self):
        x, y = self.unwrapped.car.hull.position
        return len(self._tree.query(Point(x, y), predicate="within")) > 0

    def car_on_track(self):
        """Resultado da checagem do último passo (sem recalcular)."""
        return self._on_track

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._build_track()
        self._on_track = self._check_on_track()
        info["on_track"] = self._on_track
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._on_track = self._check_on_track()
        info["on_track"] = self._on_track

        if not self._on_track:
            reward -= self.off_track_penalty
            terminated = True
        else:
            speed = float(np.linalg.norm(self.unwrapped.car.hull.linearVelocity))
            reward += max(0.0, speed * self.speed_bonus_scale)
            if speed < self.low_speed_threshold:
                reward -= self.low_speed_penalty

        return obs, reward, terminated, truncated, info


class FrameStackSkipWrapper(gym.Wrapper):
    """
    Pré-processamento de imagem: escala de cinza (uint8), frame skip e
    empilhamento dos últimos `stack` frames. Dá ao agente noção de movimento.
    Observação resultante: (96, 96, stack) uint8.
    """
    _GRAY = np.array([0.299, 0.587, 0.114], dtype=np.float32)

    def __init__(self, env, frame_skip=4, stack=4):
        super().__init__(env)
        self.frame_skip = frame_skip
        self.frames = deque(maxlen=stack)
        h, w = env.observation_space.shape[:2]
        self.observation_space = spaces.Box(0, 255, (h, w, stack), dtype=np.uint8)

    def _gray(self, frame):
        return (frame[..., :3] @ self._GRAY).astype(np.uint8)

    def _stacked(self):
        return np.stack(self.frames, axis=-1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        gray = self._gray(obs)
        for _ in range(self.frames.maxlen):
            self.frames.append(gray)
        return self._stacked(), info

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        self.frames.append(self._gray(obs))
        return self._stacked(), total_reward, terminated, truncated, info
