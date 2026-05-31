# simulator

Simulates dynamic multicopter behavior with configurable architecture, mass properties, propulsion units, and thrust-vector actuation.

## Features

- Define mass distribution with `MassElement` and estimate:
  - total mass
  - center of gravity (CoG)
  - inertia tensor
- Define any number of `PropulsionUnit`s with:
  - position relative to CoG or a separate reference frame
  - thrust axis
  - configurable thrust response curve (`signal -> thrust`)
  - configurable power curve (`thrust -> kW`)
- Optional `ThrustVectorActuator` per propulsion unit:
  - configurable rotation axis
  - configurable actuator response curve (`signal -> angle`)
- Simulate translational and rotational dynamics with quaternions:
  - position, velocity, acceleration
  - orientation, angular velocity, angular acceleration
  - total power draw

## Quick example

```python
from simulator import (
    LinearCurve,
    MulticopterSimulator,
    PropulsionUnit,
)

units = [
    PropulsionUnit(
        name="m1",
        position_m=(0.25, 0.25, 0.0),
        thrust_axis_body=(0.0, 0.0, 1.0),
        thrust_curve_n=LinearCurve(0.0, 100.0, 0.0, 15.0),
        power_curve_kw=LinearCurve(0.0, 100.0, 0.0, 1.5),
    ),
    PropulsionUnit(
        name="m2",
        position_m=(-0.25, 0.25, 0.0),
        thrust_axis_body=(0.0, 0.0, 1.0),
        thrust_curve_n=LinearCurve(0.0, 100.0, 0.0, 15.0),
        power_curve_kw=LinearCurve(0.0, 100.0, 0.0, 1.5),
    ),
]

sim = MulticopterSimulator(
    mass_kg=1.5,
    inertia_kg_m2=((0.03, 0.0, 0.0), (0.0, 0.03, 0.0), (0.0, 0.0, 0.06)),
    propulsion_units=units,
)

state = sim.step({"m1": (60.0, 0.0), "m2": (60.0, 0.0)}, dt_s=0.02)
print(state.position_m, state.orientation, state.total_power_kw)
```

## Tests

Run:

```bash
python -m unittest
```
