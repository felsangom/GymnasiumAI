import gymnasium as gym
from shapely import affinity
from shapely.geometry import Point, Polygon


class SafeDrivingWrapper(gym.Wrapper):
    """
    A wrapper to create a 'safe zone' around the track and penalize the agent
    for going too far off-road.
    """
    def __init__(self, env, border_width=0.5):
        super(SafeDrivingWrapper, self).__init__(env)
        self.border_width = border_width

    def car_on_track(self):
        car_on_track = False
        x, y = self.unwrapped.car.hull.position
        point = Point(x, y)
        for poly in self.unwrapped.road_poly:
            polygon = Polygon(poly[0])
            # Create a larger polygon representing the track + safe border
            if self.border_width > 0:
                border_scale = 1 + self.border_width
                polygon = affinity.scale(polygon, xfact=border_scale, yfact=border_scale)

            if polygon.contains(point):
                car_on_track = True
                break
        return car_on_track

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)

        # Apply a heavy penalty if the car is outside the safe zone
        if not self.car_on_track():
            reward -= 100
            terminated = True # End the episode immediately

        return next_state, reward, terminated, truncated, info

# We will apply this wrapper later when we instantiate the environment.
