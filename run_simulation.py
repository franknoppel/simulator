from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, Tuple

from simulator import LinearCurve, MulticopterSimulator, PropulsionUnit, quaternion_to_euler_deg


Vector3 = Tuple[float, float, float]


@dataclass
class PilotCommand:
    throttle: float = 55.0
    pitch: float = 0.0
    roll: float = 0.0
    yaw: float = 0.0


def _build_demo_simulator() -> MulticopterSimulator:
    arm = 0.35
    units = [
        PropulsionUnit(
            name="front_left",
            position_m=(arm, arm, 0.0),
            thrust_axis_body=(0.04, -0.04, 1.0),
            thrust_curve_n=LinearCurve(0.0, 100.0, 0.0, 25.0),
            power_curve_kw=LinearCurve(0.0, 25.0, 0.0, 2.0),
        ),
        PropulsionUnit(
            name="front_right",
            position_m=(arm, -arm, 0.0),
            thrust_axis_body=(-0.04, -0.04, 1.0),
            thrust_curve_n=LinearCurve(0.0, 100.0, 0.0, 25.0),
            power_curve_kw=LinearCurve(0.0, 25.0, 0.0, 2.0),
        ),
        PropulsionUnit(
            name="rear_left",
            position_m=(-arm, arm, 0.0),
            thrust_axis_body=(0.04, 0.04, 1.0),
            thrust_curve_n=LinearCurve(0.0, 100.0, 0.0, 25.0),
            power_curve_kw=LinearCurve(0.0, 25.0, 0.0, 2.0),
        ),
        PropulsionUnit(
            name="rear_right",
            position_m=(-arm, -arm, 0.0),
            thrust_axis_body=(-0.04, 0.04, 1.0),
            thrust_curve_n=LinearCurve(0.0, 100.0, 0.0, 25.0),
            power_curve_kw=LinearCurve(0.0, 25.0, 0.0, 2.0),
        ),
    ]
    return MulticopterSimulator(
        mass_kg=1.8,
        inertia_kg_m2=((0.05, 0.0, 0.0), (0.0, 0.05, 0.0), (0.0, 0.0, 0.09)),
        propulsion_units=units,
    )


def _mix_controls(command: PilotCommand) -> Dict[str, Tuple[float, float]]:
    t = command.throttle
    pitch = command.pitch * 0.25
    roll = command.roll * 0.25
    yaw = command.yaw * 0.2

    return {
        "front_left": (max(0.0, min(100.0, t - pitch + roll + yaw)), 0.0),
        "front_right": (max(0.0, min(100.0, t - pitch - roll - yaw)), 0.0),
        "rear_left": (max(0.0, min(100.0, t + pitch + roll - yaw)), 0.0),
        "rear_right": (max(0.0, min(100.0, t + pitch - roll + yaw)), 0.0),
    }


def _speed_magnitude(v: Vector3) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _body_points(arm_m: float = 0.35):
    return {
        "center": (0.0, 0.0, 0.0),
        "front_left": (arm_m, arm_m, 0.0),
        "front_right": (arm_m, -arm_m, 0.0),
        "rear_left": (-arm_m, arm_m, 0.0),
        "rear_right": (-arm_m, -arm_m, 0.0),
    }


def _project(point: Vector3, width: int, height: int, scale: float = 140.0):
    x, y, z = point
    depth = z + 4.5
    if depth < 0.5:
        depth = 0.5
    px = width * 0.5 + x * scale / depth
    py = height * 0.65 - y * scale / depth
    return (px, py)


def _run_headless(duration_s: float, dt_s: float) -> None:
    sim = _build_demo_simulator()
    command = PilotCommand()
    steps = max(1, int(duration_s / dt_s))
    for idx in range(steps):
        phase = idx / steps
        command.pitch = 35.0 * math.sin(phase * math.pi * 2.0)
        command.roll = 20.0 * math.sin(phase * math.pi)
        command.yaw = 25.0 * math.cos(phase * math.pi * 2.0)
        state = sim.step(_mix_controls(command), dt_s=dt_s)

    roll, pitch, yaw = quaternion_to_euler_deg(state.orientation)
    speed = _speed_magnitude(state.velocity_m_s)
    print("Simulation complete")
    print(f"Position [m]: {state.position_m[0]:.2f}, {state.position_m[1]:.2f}, {state.position_m[2]:.2f}")
    print(f"Speed [m/s]: {speed:.2f}")
    print(f"Attitude [deg] roll={roll:.1f}, pitch={pitch:.1f}, yaw={yaw:.1f}")
    print(f"Power [kW]: {state.total_power_kw:.2f}")


def _run_gui(dt_s: float) -> None:
    import tkinter as tk

    sim = _build_demo_simulator()
    command = PilotCommand()

    root = tk.Tk()
    root.title("Multicopter Simulator")
    root.geometry("1000x720")

    left = tk.Frame(root)
    left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    right = tk.Frame(root, padx=8, pady=8)
    right.pack(side=tk.RIGHT, fill=tk.Y)

    canvas = tk.Canvas(left, bg="#111", width=760, height=700)
    canvas.pack(fill=tk.BOTH, expand=True)

    telemetry = {
        "position": tk.StringVar(value="Position [m]: 0.00, 0.00, 0.00"),
        "speed": tk.StringVar(value="Speed [m/s]: 0.00"),
        "attitude": tk.StringVar(value="Attitude [deg]: roll 0.0 | pitch 0.0 | yaw 0.0"),
        "power": tk.StringVar(value="Power [kW]: 0.00"),
    }

    for key in ("position", "speed", "attitude", "power"):
        tk.Label(right, textvariable=telemetry[key], anchor="w", width=44).pack(fill=tk.X, pady=2)

    tk.Label(right, text="Throttle").pack(anchor="w", pady=(12, 0))
    throttle_scale = tk.Scale(right, from_=0, to=100, orient=tk.HORIZONTAL, length=220)
    throttle_scale.set(command.throttle)
    throttle_scale.pack()

    tk.Label(right, text="Pitch Command").pack(anchor="w", pady=(8, 0))
    pitch_scale = tk.Scale(right, from_=-100, to=100, orient=tk.HORIZONTAL, length=220)
    pitch_scale.set(command.pitch)
    pitch_scale.pack()

    tk.Label(right, text="Roll Command").pack(anchor="w", pady=(8, 0))
    roll_scale = tk.Scale(right, from_=-100, to=100, orient=tk.HORIZONTAL, length=220)
    roll_scale.set(command.roll)
    roll_scale.pack()

    tk.Label(right, text="Yaw Command").pack(anchor="w", pady=(8, 0))
    yaw_scale = tk.Scale(right, from_=-100, to=100, orient=tk.HORIZONTAL, length=220)
    yaw_scale.set(command.yaw)
    yaw_scale.pack()

    instructions = (
        "Use sliders to command throttle, pitch, roll and yaw.\n"
        "Model view: green front arm, blue rear arm, red dots motors."
    )
    tk.Label(right, text=instructions, justify=tk.LEFT, wraplength=230).pack(anchor="w", pady=(10, 0))

    points = _body_points()

    def tick() -> None:
        command.throttle = float(throttle_scale.get())
        command.pitch = float(pitch_scale.get())
        command.roll = float(roll_scale.get())
        command.yaw = float(yaw_scale.get())

        state = sim.step(_mix_controls(command), dt_s=dt_s)

        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())

        world_points = {}
        for name, body_point in points.items():
            rotated = state.orientation.rotate(body_point)
            world_points[name] = (
                state.position_m[0] + rotated[0],
                state.position_m[1] + rotated[1],
                state.position_m[2] + rotated[2],
            )

        projected = {name: _project(p, width, height) for name, p in world_points.items()}

        canvas.delete("all")
        canvas.create_line(*projected["front_left"], *projected["front_right"], fill="#46e36f", width=3)
        canvas.create_line(*projected["rear_left"], *projected["rear_right"], fill="#63a5ff", width=3)
        canvas.create_line(*projected["center"], *projected["front_left"], fill="#7ae88f", width=2)
        canvas.create_line(*projected["center"], *projected["front_right"], fill="#7ae88f", width=2)
        canvas.create_line(*projected["center"], *projected["rear_left"], fill="#85b8ff", width=2)
        canvas.create_line(*projected["center"], *projected["rear_right"], fill="#85b8ff", width=2)
        for motor in ("front_left", "front_right", "rear_left", "rear_right"):
            px, py = projected[motor]
            canvas.create_oval(px - 6, py - 6, px + 6, py + 6, fill="#ef5350", outline="")

        center_x, center_y = projected["center"]
        canvas.create_oval(center_x - 5, center_y - 5, center_x + 5, center_y + 5, fill="#fefefe", outline="")

        roll, pitch, yaw = quaternion_to_euler_deg(state.orientation)
        speed = _speed_magnitude(state.velocity_m_s)
        telemetry["position"].set(
            f"Position [m]: {state.position_m[0]:.2f}, {state.position_m[1]:.2f}, {state.position_m[2]:.2f}"
        )
        telemetry["speed"].set(f"Speed [m/s]: {speed:.2f}")
        telemetry["attitude"].set(f"Attitude [deg]: roll {roll:.1f} | pitch {pitch:.1f} | yaw {yaw:.1f}")
        telemetry["power"].set(f"Power [kW]: {state.total_power_kw:.2f}")

        root.after(max(1, int(dt_s * 1000)), tick)

    tick()
    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multicopter simulator demo")
    parser.add_argument("--headless", action="store_true", help="Run without GUI and print telemetry at the end")
    parser.add_argument("--duration", type=float, default=10.0, help="Headless duration in seconds")
    parser.add_argument("--dt", type=float, default=0.02, help="Simulation step size in seconds")
    args = parser.parse_args()

    if args.headless:
        _run_headless(duration_s=args.duration, dt_s=args.dt)
        return

    try:
        _run_gui(dt_s=args.dt)
    except Exception as exc:
        print(f"GUI unavailable ({exc}). Falling back to headless mode.")
        _run_headless(duration_s=args.duration, dt_s=args.dt)


if __name__ == "__main__":
    main()
