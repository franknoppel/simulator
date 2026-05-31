import math
import subprocess
import sys
import unittest

from simulator import (
    LinearCurve,
    MassElement,
    MulticopterSimulator,
    PropulsionUnit,
    Quaternion,
    ThrustVectorActuator,
    estimate_inertia_tensor,
    estimate_mass_and_cog,
    quaternion_to_euler_deg,
)


class SimulatorTests(unittest.TestCase):
    def test_estimate_mass_cog_and_inertia(self):
        elements = [
            MassElement(1.0, (-1.0, 0.0, 0.0)),
            MassElement(1.0, (1.0, 0.0, 0.0)),
        ]
        mass, cog = estimate_mass_and_cog(elements)
        inertia = estimate_inertia_tensor(elements, cog)

        self.assertEqual(mass, 2.0)
        self.assertAlmostEqual(cog[0], 0.0)
        self.assertAlmostEqual(cog[1], 0.0)
        self.assertAlmostEqual(cog[2], 0.0)
        self.assertEqual(inertia[1][1], 2.0)
        self.assertEqual(inertia[2][2], 2.0)

    def test_thrust_vector_actuation(self):
        actuator = ThrustVectorActuator(
            rotation_axis_body=(1.0, 0.0, 0.0),
            angle_curve_rad=LinearCurve(-100.0, 100.0, -math.pi / 2.0, math.pi / 2.0),
        )
        unit = PropulsionUnit(
            name="tilt",
            position_m=(0.0, 0.0, 0.0),
            thrust_axis_body=(0.0, 0.0, 1.0),
            thrust_curve_n=LinearCurve(0.0, 100.0, 0.0, 100.0),
            actuator=actuator,
        )

        force = unit.thrust_vector(100.0, 100.0)
        self.assertAlmostEqual(force[0], 0.0, places=5)
        self.assertAlmostEqual(force[1], -100.0, places=5)
        self.assertAlmostEqual(force[2], 0.0, places=5)

    def test_power_sum_and_upward_motion(self):
        units = [
            PropulsionUnit(
                name="m1",
                position_m=(1.0, 0.0, 0.0),
                thrust_axis_body=(0.0, 0.0, 1.0),
                thrust_curve_n=LinearCurve(0.0, 100.0, 0.0, 20.0),
                power_curve_kw=LinearCurve(0.0, 20.0, 0.0, 2.0),
            ),
            PropulsionUnit(
                name="m2",
                position_m=(-1.0, 0.0, 0.0),
                thrust_axis_body=(0.0, 0.0, 1.0),
                thrust_curve_n=LinearCurve(0.0, 100.0, 0.0, 20.0),
                power_curve_kw=LinearCurve(0.0, 20.0, 0.0, 2.0),
            ),
        ]
        sim = MulticopterSimulator(
            mass_kg=2.0,
            inertia_kg_m2=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            propulsion_units=units,
        )

        state = sim.step({"m1": (100.0, 0.0), "m2": (100.0, 0.0)}, dt_s=0.1)
        self.assertGreater(state.acceleration_m_s2[2], 0.0)
        self.assertGreater(state.position_m[2], 0.0)
        self.assertAlmostEqual(state.total_power_kw, 4.0)

    def test_quaternion_to_euler_deg(self):
        q = Quaternion.from_axis_angle((1.0, 0.0, 0.0), math.pi / 2.0)
        roll, pitch, yaw = quaternion_to_euler_deg(q)
        self.assertAlmostEqual(roll, 90.0, places=4)
        self.assertAlmostEqual(pitch, 0.0, places=4)
        self.assertAlmostEqual(yaw, 0.0, places=4)

    def test_headless_runner_outputs_telemetry(self):
        completed = subprocess.run(
            [sys.executable, "run_simulation.py", "--headless", "--duration", "0.1", "--dt", "0.05"],
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertIn("Simulation complete", completed.stdout)
        self.assertIn("Position [m]:", completed.stdout)
        self.assertIn("Speed [m/s]:", completed.stdout)
        self.assertIn("Attitude [deg]", completed.stdout)
        self.assertIn("Power [kW]:", completed.stdout)


if __name__ == "__main__":
    unittest.main()
