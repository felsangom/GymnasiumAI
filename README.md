# GymnasiumAI

Here you'll find my solutions to some of the Python Gymnasium environments.

Install the dependencies bellow before trying to run the code:

```bash
pip install numpy tensorflow shapely "gymnasium[all]"
```

Each folder contains the weights file with pre trained data.

## CarRacing

For the car_racing problem, I've changed the environment a bit, by adding a smaller border around the track, so the car don't roam freely through the map.
This way, I can achieve faster training by reseting the env when the car goes off the track by a small margin.

To do this, you need to change the `gymnasium/envs/box2d/car_racing.py` file and change the following code:

1. Import the needed libs
```python
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely import affinity
```

2. Create the method `car_on_track` in the class `CarRacing`:
```python
    def car_on_track(self, border_width=0):
        car_in_track  = False
        x, y = self.car.hull.position
        point = Point(x, y)
        for poly in self.road_poly:
            polygon = Polygon(poly[0])

            if border_width > 0:
                border = 1 + border_width
                polygon = affinity.scale(polygon, xfact=border, yfact=border)

            if polygon.contains(point):
                car_in_track = True
                break

        return car_in_track
```

3. Call the `car_on_track` method inside the `step` method, and apply a negative reward if the car go out of the bounds of the specified track border.
```python
    def step(self, action: Union[np.ndarray, int]):
        # existing code...

        step_reward = 0
        terminated = False
        truncated = False
        info = {}
        if action is not None:  # First step without action, called from reset()
            self.reward -= 0.1
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward

            if not self.car_on_track(0.5):
                step_reward  -= 100
                terminated = True
                info["lap_finished"] = False

            # remaining of the code remains untouched
```

I'm adding the file here in case you just want to download and replace your own.
