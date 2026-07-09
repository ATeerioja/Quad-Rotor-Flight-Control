"""6-DOF rigid-body quadrotor dynamics, integrated with RK4.

Pure numpy/scipy physics core, no Gym dependency, so it is testable in
isolation (see quad_rl/tests/test_dynamics.py).

State vector (13 dims), in this order:
    position          (3)  -- world frame, meters
    velocity          (3)  -- world frame, m/s
    quaternion        (4)  -- unit quaternion, scalar-first [w, x, y, z],
                               body -> world rotation. NOT Euler angles,
                               to avoid gimbal lock.
    angular_velocity  (3)  -- body frame, rad/s

Frames: world frame is Z-up. Body frame at identity orientation is
x-forward, y-left, z-up (FLU) -- the only right-handed frame consistent
with x-forward and a Z-up world, so it isn't a free choice once those two
are fixed. Thrust always acts along the body +z axis.

Action: 4 motor commands in [0, 1] (clipped defensively), mapped to
thrust via a static curve `thrust_i = thrust_coefficient * command_i`.
No actuator lag is modeled in this stage -- the state vector above has
no room for per-motor lag state -- so `motor_time_constant` is accepted
in `params` but currently unused; a future stage can add it either as
extra state or as an env-level filter.

Motor mixing: quadrotor X-configuration. Motors are numbered 1-4 as
front-right, rear-right, rear-left, front-left, each at distance
`arm_length` from the center of mass at +-45 degrees from the body
x-axis. Diagonal pairs (1, 3) and (2, 4) spin in opposite directions so
that yaw reaction torques cancel at equal thrust -- the standard X-quad
convention. With F_i = thrust of motor i:

    F_total = F1 + F2 + F3 + F4
    tau_x   = (arm_length / sqrt(2)) * (-F1 - F2 + F3 + F4)
    tau_y   = (arm_length / sqrt(2)) * (-F1 + F2 + F3 - F4)
    tau_z   = yaw_torque_coefficient * (F1 - F2 + F3 - F4)

`yaw_torque_coefficient` is not among the parameters listed in the
Stage 0.2 prompt, but yaw torque has no other source in this model (roll
and pitch come from thrust differences via arm_length; yaw only comes
from motor reaction torque), so it's added to the config rather than
hardcoded, to keep "no magic numbers in the physics code" true for all
three axes.

Config `params` dict (flat, physics-only -- not the full YAML document;
bridging default.yaml's nested `physics.inertia.{ixx,iyy,izz}` into the
flat array form below is the job of the future env config loader):
    mass                    -- kg
    inertia                 -- array-like [ixx, iyy, izz], kg*m^2 (diagonal)
    arm_length               -- m
    thrust_coefficient        -- thrust = thrust_coefficient * command
    drag_coefficient           -- linear velocity damping
    yaw_torque_coefficient      -- yaw reaction torque = coeff * thrust
    gravity                      -- m/s^2
    motor_time_constant            -- s (unused, reserved)
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

POS = slice(0, 3)
VEL = slice(3, 6)
QUAT = slice(6, 10)
OMEGA = slice(10, 13)

STATE_DIM = 13
ACTION_DIM = 4

_SQRT2 = np.sqrt(2.0)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Return q rescaled to unit norm."""
    return q / np.linalg.norm(q)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 * q2, both scalar-first [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Body -> world rotation matrix for scalar-first quaternion q."""
    w, x, y, z = q
    # scipy uses scalar-last [x, y, z, w] quaternions.
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def _motor_mixing(thrusts: np.ndarray, params: dict) -> tuple[float, np.ndarray]:
    """Map 4 motor thrusts to (total_thrust, body_torque) for an X-frame."""
    f1, f2, f3, f4 = thrusts
    arm = params["arm_length"]
    k_yaw = params["yaw_torque_coefficient"]

    total_thrust = f1 + f2 + f3 + f4
    tau_x = (arm / _SQRT2) * (-f1 - f2 + f3 + f4)
    tau_y = (arm / _SQRT2) * (-f1 + f2 + f3 - f4)
    tau_z = k_yaw * (f1 - f2 + f3 - f4)
    return total_thrust, np.array([tau_x, tau_y, tau_z])


def _state_derivative(state: np.ndarray, action: np.ndarray, params: dict) -> np.ndarray:
    """d(state)/dt for the 13-dim state, given a 4-dim motor command."""
    velocity = state[VEL]
    quat = state[QUAT]
    omega = state[OMEGA]

    command = np.clip(action, 0.0, 1.0)
    thrusts = params["thrust_coefficient"] * command
    total_thrust, torque = _motor_mixing(thrusts, params)

    mass = params["mass"]
    gravity = params["gravity"]
    drag = params["drag_coefficient"]

    rotmat = _quat_to_rotmat(quat)
    thrust_world = rotmat @ np.array([0.0, 0.0, total_thrust])
    gravity_world = np.array([0.0, 0.0, -gravity])

    dpos = velocity
    dvel = thrust_world / mass + gravity_world - (drag / mass) * velocity

    omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])
    dquat = 0.5 * quat_multiply(quat, omega_quat)

    inertia = np.asarray(params["inertia"], dtype=float)
    domega = (torque - np.cross(omega, inertia * omega)) / inertia

    dstate = np.empty(STATE_DIM)
    dstate[POS] = dpos
    dstate[VEL] = dvel
    dstate[QUAT] = dquat
    dstate[OMEGA] = domega
    return dstate


def step(state: np.ndarray, action: np.ndarray, dt: float, params: dict) -> np.ndarray:
    """Advance state by dt using RK4, with action held constant (zero-order
    hold) across the four RK4 stages. Renormalizes the quaternion after
    integration, since RK4 does not conserve the unit-quaternion
    constraint on its own.
    """
    k1 = _state_derivative(state, action, params)
    k2 = _state_derivative(state + 0.5 * dt * k1, action, params)
    k3 = _state_derivative(state + 0.5 * dt * k2, action, params)
    k4 = _state_derivative(state + dt * k3, action, params)

    next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    next_state[QUAT] = quat_normalize(next_state[QUAT])
    return next_state
