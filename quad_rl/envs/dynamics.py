"""6-DOF rigid-body quadrotor dynamics, integrated with RK4.

This module will implement the physics core, with no Gym dependency so it
stays testable in isolation (see quad_rl/tests/test_dynamics.py).

State vector (13 dims), in this order:
    position         (3)  -- world frame
    linear velocity  (3)  -- world frame
    orientation      (4)  -- unit quaternion (body -> world), NOT Euler
                              angles, to avoid gimbal lock
    angular velocity (3)  -- body frame

Action: 4 motor thrust commands normalized to [0, 1], mapped through a
configurable thrust curve (thrust = k_thrust * command), optionally with
first-order motor lag (time constant exposed via config).

All physical parameters are driven by a config dict -- no hardcoded
magic numbers -- so the model is ready for domain randomization in a
later stage:
    mass, inertia tensor (diagonal), arm_length, thrust_coefficient,
    drag_coefficient (linear velocity damping), gravity,
    motor_time_constant.

Planned contents:
    - quaternion kinematics: dq/dt from body angular velocity
    - Newton-Euler equations for translational and rotational dynamics
    - motor mixing: 4 thrusts -> total thrust + roll/pitch/yaw torque
      (quadrotor X or + configuration -- to be documented once chosen)
    - RK4 integrator, default timestep 0.01 s (100 Hz), configurable
    - step(state, action, dt, params) -> next_state as the public
      interface, implemented as a stateless function or small class
"""
