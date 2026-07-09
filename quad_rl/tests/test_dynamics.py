"""Pytest suite for quad_rl.envs.dynamics -- physics invariants that
would catch a broken rigid-body model before it's wrapped in a Gym env.

Planned tests:
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
"""
