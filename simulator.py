from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, pi, sin, sqrt
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

Vector3 = Tuple[float, float, float]
Matrix3 = Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]


def _add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(v: Vector3, s: float) -> Vector3:
    return (v[0] * s, v[1] * s, v[2] * s)


def _dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vector3, b: Vector3) -> Vector3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(v: Vector3) -> float:
    return sqrt(_dot(v, v))


def _normalize(v: Vector3) -> Vector3:
    n = _norm(v)
    if n == 0:
        raise ValueError("zero-length vector")
    return _scale(v, 1.0 / n)


def _mat_vec_mul(m: Matrix3, v: Vector3) -> Vector3:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def _invert_matrix3(m: Matrix3) -> Matrix3:
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]

    cofactor00 = e * i - f * h
    cofactor01 = -(d * i - f * g)
    cofactor02 = d * h - e * g
    cofactor10 = -(b * i - c * h)
    cofactor11 = a * i - c * g
    cofactor12 = -(a * h - b * g)
    cofactor20 = b * f - c * e
    cofactor21 = -(a * f - c * d)
    cofactor22 = a * e - b * d

    determinant = a * cofactor00 + b * cofactor01 + c * cofactor02
    if determinant == 0:
        raise ValueError("inertia matrix is singular")

    inv_det = 1.0 / determinant
    return (
        (cofactor00 * inv_det, cofactor10 * inv_det, cofactor20 * inv_det),
        (cofactor01 * inv_det, cofactor11 * inv_det, cofactor21 * inv_det),
        (cofactor02 * inv_det, cofactor12 * inv_det, cofactor22 * inv_det),
    )


@dataclass(frozen=True)
class LinearCurve:
    input_min: float
    input_max: float
    output_min: float
    output_max: float
    clamp: bool = True

    def __call__(self, signal: float) -> float:
        if self.input_max == self.input_min:
            raise ValueError("input range cannot be zero")
        s = signal
        if self.clamp:
            s = min(max(s, self.input_min), self.input_max)
        ratio = (s - self.input_min) / (self.input_max - self.input_min)
        return self.output_min + ratio * (self.output_max - self.output_min)


@dataclass(frozen=True)
class MassElement:
    mass_kg: float
    position_m: Vector3


@dataclass(frozen=True)
class Quaternion:
    w: float
    x: float
    y: float
    z: float

    @staticmethod
    def identity() -> "Quaternion":
        return Quaternion(1.0, 0.0, 0.0, 0.0)

    @staticmethod
    def from_axis_angle(axis: Vector3, angle_rad: float) -> "Quaternion":
        half = angle_rad * 0.5
        ax = _normalize(axis)
        s = sin(half)
        return Quaternion(cos(half), ax[0] * s, ax[1] * s, ax[2] * s).normalized()

    def conjugate(self) -> "Quaternion":
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def normalized(self) -> "Quaternion":
        n = sqrt(self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z)
        if n == 0:
            raise ValueError("zero-length quaternion")
        return Quaternion(self.w / n, self.x / n, self.y / n, self.z / n)

    def multiply(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )

    def rotate(self, vector: Vector3) -> Vector3:
        q_vec = Quaternion(0.0, vector[0], vector[1], vector[2])
        rotated = self.multiply(q_vec).multiply(self.conjugate())
        return (rotated.x, rotated.y, rotated.z)


@dataclass(frozen=True)
class ThrustVectorActuator:
    rotation_axis_body: Vector3
    angle_curve_rad: Callable[[float], float] = field(
        default_factory=lambda: LinearCurve(-100.0, 100.0, -pi / 2.0, pi / 2.0)
    )

    def rotate(self, base_vector_body: Vector3, signal: float) -> Vector3:
        q = Quaternion.from_axis_angle(self.rotation_axis_body, self.angle_curve_rad(signal))
        return q.rotate(base_vector_body)


@dataclass(frozen=True)
class PropulsionUnit:
    name: str
    position_m: Vector3
    thrust_axis_body: Vector3
    thrust_curve_n: Callable[[float], float] = field(
        default_factory=lambda: LinearCurve(0.0, 100.0, 0.0, 100.0)
    )
    power_curve_kw: Callable[[float], float] = field(
        default_factory=lambda: LinearCurve(0.0, 100.0, 0.0, 1.0)
    )
    position_reference: str = "cog"
    actuator: Optional[ThrustVectorActuator] = None

    def thrust_vector(self, thrust_signal: float, actuator_signal: float = 0.0) -> Vector3:
        direction = _normalize(self.thrust_axis_body)
        if self.actuator is not None:
            direction = _normalize(self.actuator.rotate(direction, actuator_signal))
        thrust = self.thrust_curve_n(thrust_signal)
        return _scale(direction, thrust)

    def power_draw_kw(self, thrust_signal: float) -> float:
        thrust = self.thrust_curve_n(thrust_signal)
        return self.power_curve_kw(thrust)


@dataclass
class VehicleState:
    position_m: Vector3 = (0.0, 0.0, 0.0)
    velocity_m_s: Vector3 = (0.0, 0.0, 0.0)
    acceleration_m_s2: Vector3 = (0.0, 0.0, 0.0)
    orientation: Quaternion = field(default_factory=Quaternion.identity)
    angular_velocity_rad_s: Vector3 = (0.0, 0.0, 0.0)
    angular_acceleration_rad_s2: Vector3 = (0.0, 0.0, 0.0)
    total_power_kw: float = 0.0


ControlInput = Mapping[str, Tuple[float, float]]


@dataclass
class MulticopterSimulator:
    mass_kg: float
    inertia_kg_m2: Matrix3
    propulsion_units: Sequence[PropulsionUnit]
    cog_in_reference_m: Vector3 = (0.0, 0.0, 0.0)
    gravity_m_s2: Vector3 = (0.0, 0.0, -9.81)
    state: VehicleState = field(default_factory=VehicleState)

    def __post_init__(self) -> None:
        if self.mass_kg <= 0:
            raise ValueError("mass must be positive")
        self._inertia_inv = _invert_matrix3(self.inertia_kg_m2)
        self._units = {unit.name: unit for unit in self.propulsion_units}

    def _position_relative_to_cog(self, unit: PropulsionUnit) -> Vector3:
        if unit.position_reference == "cog":
            return unit.position_m
        if unit.position_reference == "reference":
            return _sub(unit.position_m, self.cog_in_reference_m)
        raise ValueError(f"unsupported position_reference: {unit.position_reference}")

    def step(self, control_input: ControlInput, dt_s: float) -> VehicleState:
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")

        net_force_body: Vector3 = (0.0, 0.0, 0.0)
        net_torque_body: Vector3 = (0.0, 0.0, 0.0)
        total_power_kw = 0.0

        for name, unit in self._units.items():
            thrust_signal, actuator_signal = control_input.get(name, (0.0, 0.0))
            force = unit.thrust_vector(thrust_signal, actuator_signal)
            arm = self._position_relative_to_cog(unit)
            torque = _cross(arm, force)
            net_force_body = _add(net_force_body, force)
            net_torque_body = _add(net_torque_body, torque)
            total_power_kw += unit.power_draw_kw(thrust_signal)

        net_force_world = self.state.orientation.rotate(net_force_body)
        accel_world = _add(_scale(net_force_world, 1.0 / self.mass_kg), self.gravity_m_s2)

        vel = _add(self.state.velocity_m_s, _scale(accel_world, dt_s))
        pos = _add(self.state.position_m, _scale(vel, dt_s))

        omega = self.state.angular_velocity_rad_s
        i_omega = _mat_vec_mul(self.inertia_kg_m2, omega)
        coriolis = _cross(omega, i_omega)
        angular_accel = _mat_vec_mul(self._inertia_inv, _sub(net_torque_body, coriolis))
        new_omega = _add(omega, _scale(angular_accel, dt_s))

        omega_quat = Quaternion(0.0, new_omega[0], new_omega[1], new_omega[2])
        q_dot = self.state.orientation.multiply(omega_quat)
        orientation = Quaternion(
            self.state.orientation.w + 0.5 * q_dot.w * dt_s,
            self.state.orientation.x + 0.5 * q_dot.x * dt_s,
            self.state.orientation.y + 0.5 * q_dot.y * dt_s,
            self.state.orientation.z + 0.5 * q_dot.z * dt_s,
        ).normalized()

        self.state = VehicleState(
            position_m=pos,
            velocity_m_s=vel,
            acceleration_m_s2=accel_world,
            orientation=orientation,
            angular_velocity_rad_s=new_omega,
            angular_acceleration_rad_s2=angular_accel,
            total_power_kw=total_power_kw,
        )
        return self.state


def estimate_mass_and_cog(elements: Iterable[MassElement]) -> Tuple[float, Vector3]:
    total_mass = 0.0
    weighted = (0.0, 0.0, 0.0)
    for element in elements:
        if element.mass_kg < 0:
            raise ValueError("mass cannot be negative")
        total_mass += element.mass_kg
        weighted = _add(weighted, _scale(element.position_m, element.mass_kg))

    if total_mass <= 0:
        raise ValueError("total mass must be positive")
    return total_mass, _scale(weighted, 1.0 / total_mass)


def estimate_inertia_tensor(elements: Iterable[MassElement], cog_m: Optional[Vector3] = None) -> Matrix3:
    elements_list = list(elements)
    if not elements_list:
        raise ValueError("elements cannot be empty")

    if cog_m is None:
        _, cog_m = estimate_mass_and_cog(elements_list)

    i_xx = i_yy = i_zz = 0.0
    i_xy = i_xz = i_yz = 0.0

    for element in elements_list:
        x, y, z = _sub(element.position_m, cog_m)
        m = element.mass_kg
        i_xx += m * (y * y + z * z)
        i_yy += m * (x * x + z * z)
        i_zz += m * (x * x + y * y)
        i_xy -= m * x * y
        i_xz -= m * x * z
        i_yz -= m * y * z

    return (
        (i_xx, i_xy, i_xz),
        (i_xy, i_yy, i_yz),
        (i_xz, i_yz, i_zz),
    )
