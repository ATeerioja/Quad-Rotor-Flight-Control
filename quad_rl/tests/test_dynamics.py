"""Pytest suite for quad_rl.envs.dynamics -- physics invariants that
would catch a broken rigid-body model before it's wrapped in a Gym env.

Tests:
    1. Hover equilibrium: all 4 motors at the thrust command that exactly
       balances gravity (computed analytically from mass and
       thrust_coefficient) keeps velocity and angular velocity at zero
       for many steps, within numerical tolerance.
    2. Free fall: zero thrust produces vertical acceleration matching
       -g within tolerance.
    3. Symmetry: a small positive roll torque produces roll angular
       acceleration of the correct sign and expected magnitude (from
       the inertia tensor), with no cross-coupling into pitch/yaw for a
       diagonal inertia tensor.
    4. Quaternion normalization: the orientation quaternion's norm stays
       within 1e-6 of 1.0 after many integration steps.
    5. Angular momentum conservation: for a torque-free, thrust-free
       case, angular velocity stays constant (no artificial damping
       bugs in the integrator).
    6. External force: an optional external_force adds a pure
       translational acceleration with no torque coupling, and omitting
       it entirely is identical to passing external_force=None.
"""

from pathlib import Path

import numpy as np
import pytest

from quad_rl.config.loader import load_config
from quad_rl.envs import dynamics as dyn

CONFIG_PATH = Path(__file__).resolve().parents[1] / "envs" / "configs" / "default.yaml"


@pytest.fixture
def params():
    return load_config(CONFIG_PATH).physics.as_params()


@pytest.fixture
def dt():
    return load_config(CONFIG_PATH).simulation.dt


def _level_state():
    """State at the origin, at rest, upright, non-rotating."""
    state = np.zeros(dyn.STATE_DIM)
    state[dyn.QUAT] = [1.0, 0.0, 0.0, 0.0]
    return state


def _hover_command(params):
    return params["mass"] * params["gravity"] / (4.0 * params["thrust_coefficient"])


def test_hover_equilibrium(params, dt):
    hover_command = _hover_command(params)
    assert 0.0 < hover_command < 1.0, (
        "hover command out of actuator range [0, 1] -- check thrust_coefficient"
    )

    action = np.full(4, hover_command)
    state = _level_state()
    for _ in range(500):
        state = dyn.step(state, action, dt, params)

    assert np.allclose(state[dyn.VEL], 0.0, atol=1e-6)
    assert np.allclose(state[dyn.OMEGA], 0.0, atol=1e-6)


def test_free_fall(params):
    state = _level_state()
    action = np.zeros(4)
    # Instantaneous derivative at v=0, where drag is exactly zero --
    # a finite difference over a full RK4 step would pick up a small,
    # physically correct drag contribution from the velocity gained
    # during the step, which isn't what this test is checking.
    dstate = dyn._state_derivative(state, action, params)
    assert dstate[dyn.VEL][2] == pytest.approx(-params["gravity"], abs=1e-9)


def test_roll_symmetry_no_cross_coupling(params):
    hover_command = _hover_command(params)
    delta = 0.01
    # Motor order is front-right, rear-right, rear-left, front-left
    # (see dynamics.py's mixing matrix). Decreasing motors 1,2 and
    # increasing 3,4 by the same amount produces a pure roll torque:
    # the pitch and yaw torque terms cancel by construction.
    action = np.array([
        hover_command - delta,
        hover_command - delta,
        hover_command + delta,
        hover_command + delta,
    ])
    state = _level_state()
    dstate = dyn._state_derivative(state, action, params)

    thrusts = params["thrust_coefficient"] * action
    _, torque = dyn._motor_mixing(thrusts, params)
    expected_roll_accel = torque[0] / params["inertia"][0]

    assert dstate[dyn.OMEGA][0] > 0
    assert dstate[dyn.OMEGA][0] == pytest.approx(expected_roll_accel, rel=1e-9)
    assert dstate[dyn.OMEGA][1] == pytest.approx(0.0, abs=1e-12)
    assert dstate[dyn.OMEGA][2] == pytest.approx(0.0, abs=1e-12)


def test_quaternion_stays_normalized(params, dt):
    hover_command = _hover_command(params)
    # Asymmetric thrust so the vehicle keeps rotating throughout the
    # rollout, rather than sitting at a fixed orientation.
    action = np.array([
        hover_command * 1.1,
        hover_command * 0.9,
        hover_command * 1.05,
        hover_command * 0.95,
    ])
    state = _level_state()
    for _ in range(2000):
        state = dyn.step(state, action, dt, params)
        assert abs(np.linalg.norm(state[dyn.QUAT]) - 1.0) < 1e-6


def test_angular_velocity_conserved_without_torque(params, dt):
    state = _level_state()
    # Angular velocity aligned with a single principal axis is an exact
    # equilibrium of Euler's equations for a diagonal inertia tensor
    # (cross(omega, I @ omega) == 0 identically). A generic omega
    # direction would legitimately precess given this inertia tensor
    # (ixx == iyy != izz, a symmetric top) -- that's correct physics,
    # not damping, but it would make "constant angular velocity" the
    # wrong invariant to check. Axis alignment isolates the "no
    # artificial damping" property this test is actually after.
    state[dyn.OMEGA] = [0.5, 0.0, 0.0]
    action = np.zeros(4)
    omega0 = state[dyn.OMEGA].copy()

    for _ in range(200):
        state = dyn.step(state, action, dt, params)

    assert np.allclose(state[dyn.OMEGA], omega0, atol=1e-9)


def test_external_force_adds_pure_translational_acceleration(params):
    state = _level_state()
    action = np.zeros(4)
    external_force = np.array([1.0, -2.0, 3.0])

    dstate = dyn._state_derivative(state, action, params, external_force=external_force)

    gravity_term = np.array([0.0, 0.0, -params["gravity"]])
    expected_dvel = gravity_term + external_force / params["mass"]
    assert dstate[dyn.VEL] == pytest.approx(expected_dvel, abs=1e-9)
    assert dstate[dyn.OMEGA] == pytest.approx(0.0, abs=1e-12)


def test_external_force_default_none_matches_no_kwarg_call(params):
    state = _level_state()
    action = np.zeros(4)

    with_default = dyn._state_derivative(state, action, params, external_force=None)
    without_kwarg = dyn._state_derivative(state, action, params)

    assert np.array_equal(with_default, without_kwarg)
