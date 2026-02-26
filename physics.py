from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class EnvironmentParams:
    board_distance: float
    center_height: float
    release_height: float
    board_diameter: float
    outer_bull_diameter: float
    bullseye_diameter: float
    gravity: float = 9.81


@dataclass(frozen=True)
class AeroParams:
    air_density: float
    drag_coefficient: float
    reference_area: float
    mass: float

    @property
    def drag_k(self) -> float:
        # a_drag = -k * ||v_rel|| * v_rel
        return 0.5 * self.air_density * self.drag_coefficient * self.reference_area / self.mass

env = EnvironmentParams(
    board_distance=2.37,
    center_height=1.73,
    release_height=1.9,
    board_diameter=0.451,
    outer_bull_diameter=0.032,
    bullseye_diameter=0.0127,
    )

    # Approximate dart aerodynamics (tune these for your specific dart).
aero = AeroParams(
    air_density=1.2,
    drag_coefficient=0.85,
    reference_area=2.8e-4,
    mass=0.022,
    )

def simulate_throw(v0, theta, env, aero, dt, t_max):
    # Unpack environment parameters
    D  = env.board_distance
    g  = env.gravity
    y0 = env.release_height

    # Unpack aero
    k  = aero.drag_k

    # Initial position
    x = 0
    y = y0

    # Initial velocity components
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)

    t = 0

    while (t < t_max):
        # Save previous state for board-crossing interpolation
        x_prev  = x
        y_prev  = y
        vx_prev = vx
        vy_prev = vy
        t_prev  = t
        # Compute accelerations        
        ax, ay = compute_accel(vx, vy, g, k)
        # Update state
        x, y, vx, vy = step_state(x_prev, y_prev, vx_prev, vy_prev, ax, ay, dt)
        t += dt
        if check_board_crossing(x_prev, x, D):
            y_hit, t_hit, vx_hit, vy_hit = interpolate_hit(
                (x_prev, y_prev, vx_prev, vy_prev, t_prev),
                (x, y, vx, vy, t),
                D
            )
            return y_hit, t_hit, vx_hit, vy_hit


def compute_accel(vx, vy, g, k):
    speed = np.sqrt(vx**2 + vy**2)
    ax = -k * speed * vx
    ay = -g - k * speed * vy
    return ax, ay

def step_state(x, y, vx, vy, ax, ay, dt):
    # Semi-implicit Euler update
    vx_new = vx + ax * dt
    vy_new = vy + ay * dt
    x_new = x + vx_new * dt
    y_new = y + vy_new * dt
    return x_new, y_new, vx_new, vy_new

def check_board_crossing(x_prev, x, D):
    return (x_prev < D) and (x >= D)

def interpolate_hit(prev_state, curr_state, D):
    x_prev, y_prev, vx_prev, vy_prev, t_prev = prev_state
    x, y, vx, vy, t = curr_state
    alpha = (D - x_prev) / (x - x_prev)
    if alpha < 1e-6:
        return y_prev, t_prev, vx_prev, vy_prev
    t_hit = t_prev + alpha * (t - t_prev)
    y_hit = y_prev + alpha * (y - y_prev)
    vx_hit = vx_prev + alpha * (vx - vx_prev)
    vy_hit = vy_prev + alpha * (vy - vy_prev)
    return y_hit, t_hit, vx_hit, vy_hit