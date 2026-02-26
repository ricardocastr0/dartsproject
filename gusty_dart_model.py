from dataclasses import dataclass

import matplotlib.pyplot as plt
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


@dataclass(frozen=True)
class ControlBounds:
    v_min: float
    v_max: float
    theta_min_deg: float
    theta_max_deg: float


@dataclass(frozen=True)
class GustParams:
    tau: float
    sigma_x: float
    sigma_y: float


@dataclass(frozen=True)
class LaunchNoiseParams:
    sigma_v: float
    sigma_theta_deg: float


@dataclass(frozen=True)
class SimulationParams:
    dt: float
    max_time: float


@dataclass(frozen=True)
class MonteCarloParams:
    n_rollouts: int
    seed: int


def score_from_y_hit(y_hit: float, env: EnvironmentParams):
    miss_distance = abs(y_hit - env.center_height)
    on_board = miss_distance <= env.board_diameter / 2

    if miss_distance <= env.bullseye_diameter / 2:
        return 50, "bullseye", miss_distance
    if miss_distance <= env.outer_bull_diameter / 2:
        return 25, "outer bull", miss_distance
    if on_board:
        return 1, "board", miss_distance
    return 0, "miss", miss_distance


def step_wind(wind, gust: GustParams, dt: float, rng):
    # Ornstein-Uhlenbeck wind with stationary std (sigma_x, sigma_y)
    decay = np.exp(-dt / gust.tau)
    std_scale = np.sqrt(max(0.0, 1.0 - decay**2))

    wx_next = decay * wind[0] + gust.sigma_x * std_scale * rng.normal()
    wy_next = decay * wind[1] + gust.sigma_y * std_scale * rng.normal()
    return np.array([wx_next, wy_next], dtype=float)


def step_dynamics(state, wind, env: EnvironmentParams, aero: AeroParams, dt: float):
    px, py, vx, vy = state

    v_rel = np.array([vx - wind[0], vy - wind[1]], dtype=float)
    rel_speed = np.linalg.norm(v_rel)
    a_drag = -aero.drag_k * rel_speed * v_rel

    ax = a_drag[0]
    ay = a_drag[1] - env.gravity

    vx_next = vx + ax * dt
    vy_next = vy + ay * dt
    px_next = px + vx_next * dt
    py_next = py + vy_next * dt

    return np.array([px_next, py_next, vx_next, vy_next], dtype=float)


def sample_launch_control(u_nominal, bounds: ControlBounds, launch_noise: LaunchNoiseParams, rng):
    v_nom, theta_nom = u_nominal
    v_noisy = np.clip(v_nom + rng.normal(0.0, launch_noise.sigma_v), bounds.v_min, bounds.v_max)
    theta_noisy = np.clip(
        theta_nom + rng.normal(0.0, launch_noise.sigma_theta_deg),
        bounds.theta_min_deg,
        bounds.theta_max_deg,
    )
    return float(v_noisy), float(theta_noisy)


def rollout(u_nominal, env, aero, gust, sim, bounds, launch_noise, rng, return_history=False):
    v0, theta_deg = sample_launch_control(u_nominal, bounds, launch_noise, rng)
    theta_rad = np.radians(theta_deg)

    if v0 <= 0.0 or np.cos(theta_rad) <= 1e-6:
        return {
            "reached_board": False,
            "y_hit": np.nan,
            "score": 0,
            "zone": "invalid",
            "miss_distance": np.inf,
            "error": np.inf,
            "cost": 1e6,
        }

    state = np.array([0.0, env.release_height, v0 * np.cos(theta_rad), v0 * np.sin(theta_rad)], dtype=float)
    wind = np.array([0.0, 0.0], dtype=float)

    t = 0.0
    max_steps = int(sim.max_time / sim.dt)

    if return_history:
        history = {
            "t": [t],
            "x": [state[0]],
            "y": [state[1]],
            "wx": [wind[0]],
            "wy": [wind[1]],
        }

    for _ in range(max_steps):
        prev_state = state.copy()

        wind = step_wind(wind, gust, sim.dt, rng)
        state = step_dynamics(state, wind, env, aero, sim.dt)
        t += sim.dt

        if return_history:
            history["t"].append(t)
            history["x"].append(state[0])
            history["y"].append(state[1])
            history["wx"].append(wind[0])
            history["wy"].append(wind[1])

        if prev_state[0] < env.board_distance <= state[0]:
            frac = (env.board_distance - prev_state[0]) / (state[0] - prev_state[0])
            y_hit = prev_state[1] + frac * (state[1] - prev_state[1])
            t_hit = t - sim.dt + frac * sim.dt

            score, zone, miss_distance = score_from_y_hit(y_hit, env)
            error = y_hit - env.center_height

            out = {
                "reached_board": True,
                "y_hit": float(y_hit),
                "t_hit": float(t_hit),
                "score": score,
                "zone": zone,
                "miss_distance": float(miss_distance),
                "error": float(error),
                "cost": float(error**2),
                "u_nominal": tuple(map(float, u_nominal)),
                "u_applied": (float(v0), float(theta_deg)),
            }
            if return_history:
                out.update(
                    {
                        "t": np.asarray(history["t"]),
                        "x": np.asarray(history["x"]),
                        "y": np.asarray(history["y"]),
                        "wx": np.asarray(history["wx"]),
                        "wy": np.asarray(history["wy"]),
                    }
                )
            return out

    return {
        "reached_board": False,
        "y_hit": np.nan,
        "score": 0,
        "zone": "no_hit",
        "miss_distance": np.inf,
        "error": np.inf,
        "cost": 1e6,
        "u_nominal": tuple(map(float, u_nominal)),
        "u_applied": (float(v0), float(theta_deg)),
    }


def estimate_expected_cost(u, env, aero, gust, sim, bounds, launch_noise, mc, rng):
    costs = []
    scores = []
    hit_flags = []
    bull_flags = []

    sample_rollout = None
    for i in range(mc.n_rollouts):
        r = rollout(u, env, aero, gust, sim, bounds, launch_noise, rng, return_history=(i == 0))
        if sample_rollout is None:
            sample_rollout = r

        costs.append(r["cost"])
        scores.append(r["score"])
        hit_flags.append(r["reached_board"])
        bull_flags.append(r["score"] == 50)

    return {
        "u": u,
        "J_hat": float(np.mean(costs)),
        "mean_score": float(np.mean(scores)),
        "hit_rate": float(np.mean(hit_flags)),
        "bullseye_rate": float(np.mean(bull_flags)),
        "sample_rollout": sample_rollout,
    }


def grid_search_robust(env, aero, gust, sim, bounds, launch_noise, mc, v_step=0.25, theta_step=0.25):
    rng = np.random.default_rng(mc.seed)

    v_values = np.arange(bounds.v_min, bounds.v_max + v_step, v_step)
    theta_values = np.arange(bounds.theta_min_deg, bounds.theta_max_deg + theta_step, theta_step)

    best = None
    all_results = []

    for theta in theta_values:
        for v in v_values:
            u = (float(v), float(theta))
            e = estimate_expected_cost(u, env, aero, gust, sim, bounds, launch_noise, mc, rng)
            all_results.append(e)

            if best is None:
                best = e
                continue

            # Primary objective: minimize expected squared miss distance.
            # Tie-breaker: higher bullseye rate, then higher expected score.
            if e["J_hat"] < best["J_hat"]:
                best = e
            elif np.isclose(e["J_hat"], best["J_hat"]):
                if e["bullseye_rate"] > best["bullseye_rate"]:
                    best = e
                elif np.isclose(e["bullseye_rate"], best["bullseye_rate"]) and e["mean_score"] > best["mean_score"]:
                    best = e

    return best, all_results


def plot_sample_rollout(rollout_result, env):
    if "x" not in rollout_result or "y" not in rollout_result:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(rollout_result["x"], rollout_result["y"], color="steelblue")
    plt.xlabel("Distance (m)")
    plt.ylabel("Height (m)")
    plt.title("Gusty Flight: Sample Rollout")

    y_max = max(rollout_result["y"])
    upper = y_max + 4 if env.center_height + env.board_diameter / 2 < y_max < 5 else env.center_height + env.board_diameter / 2 + 0.5
    plt.xlim(0.0, env.board_distance + 0.5)
    plt.ylim(0.0, upper)

    x_board = env.board_distance
    c = env.center_height
    plt.vlines(x_board, c - env.board_diameter / 2, c + env.board_diameter / 2, color="red", linewidth=4)
    plt.vlines(x_board, c - env.outer_bull_diameter / 2, c + env.outer_bull_diameter / 2, color="orange", linewidth=6)
    plt.vlines(x_board, c - env.bullseye_diameter / 2, c + env.bullseye_diameter / 2, color="green", linewidth=8)

    if np.isfinite(rollout_result.get("y_hit", np.nan)):
        plt.scatter([x_board], [rollout_result["y_hit"]], color="blue", zorder=5)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()


def plot_cost_contours(all_results):
    v_values = sorted({r["u"][0] for r in all_results})
    theta_values = sorted({r["u"][1] for r in all_results})

    cost_grid = np.full((len(theta_values), len(v_values)), np.nan)
    v_index = {v: i for i, v in enumerate(v_values)}
    theta_index = {th: i for i, th in enumerate(theta_values)}

    for r in all_results:
        v, theta = r["u"]
        cost_grid[theta_index[theta], v_index[v]] = r["J_hat"]

    V, T = np.meshgrid(v_values, theta_values)

    plt.figure(figsize=(8, 5))
    filled = plt.contourf(V, T, cost_grid, levels=20, cmap="viridis")
    plt.contour(V, T, cost_grid, levels=10, colors="white", linewidths=0.7, alpha=0.8)
    plt.colorbar(filled, label="Cost J_hat")
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Angle (deg)")
    plt.title("Cost Contours: Robust Objective")
    plt.tight_layout()


def plot_rollout_variances(u_nominal, env, aero, gust, sim, bounds, launch_noise, n_rollouts, seed):
    rng = np.random.default_rng(seed)
    x_common = np.linspace(0.0, env.board_distance, 250)
    y_traces = []
    y_hits = []

    for _ in range(n_rollouts):
        r = rollout(
            u_nominal,
            env,
            aero,
            gust,
            sim,
            bounds,
            launch_noise,
            rng,
            return_history=True,
        )
        if not r["reached_board"]:
            continue

        y_interp = np.interp(x_common, r["x"], r["y"])
        y_traces.append(y_interp)
        y_hits.append(r["y_hit"])

    if not y_traces:
        return

    y_traces = np.asarray(y_traces)
    y_hits = np.asarray(y_hits)
    y_var_by_x = np.var(y_traces, axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7))

    axes[0].plot(x_common, y_var_by_x, color="darkblue")
    axes[0].set_xlabel("Distance x (m)")
    axes[0].set_ylabel("Var[y(x)]")
    axes[0].set_title("Trajectory Variance Across Rollouts")
    axes[0].grid(alpha=0.25)

    axes[1].hist(y_hits, bins=20, color="teal", edgecolor="black", alpha=0.8)
    axes[1].axvline(env.center_height, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Landing Height y_hit (m)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Landing Distribution (var={np.var(y_hits):.6f})")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()


def main():
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

    bounds = ControlBounds(
        v_min=8.0,
        v_max=22.0,
        theta_min_deg=5.0,
        theta_max_deg=55.0,
    )

    gust = GustParams(
        tau=0.35,
        sigma_x=0.35,
        sigma_y=0.10,
    )

    launch_noise = LaunchNoiseParams(
        sigma_v=0.35,
        sigma_theta_deg=0.8,
    )

    sim = SimulationParams(
        dt=0.002,
        max_time=2.5,
    )

    mc = MonteCarloParams(
        n_rollouts=40,
        seed=40,
    )

    best, all_results = grid_search_robust(
        env,
        aero,
        gust,
        sim,
        bounds,
        launch_noise,
        mc,
        v_step=1.0,
        theta_step=1.0,
    )

    print(f"Best nominal control (v, theta_deg): {best['u']}")
    print(f"J_hat (expected squared miss): {best['J_hat']:.6f}")
    print(f"Mean score: {best['mean_score']:.3f}")
    print(f"Bullseye rate: {best['bullseye_rate']:.3f}")
    print(f"Hit rate: {best['hit_rate']:.3f}")

    sample = best["sample_rollout"]
    print(f"Sample rollout zone: {sample['zone']}, score={sample['score']}")
    print(f"Sample nominal control: {sample['u_nominal']}")
    print(f"Sample applied control: {sample['u_applied']}")
    if np.isfinite(sample["y_hit"]):
        print(f"Sample y_hit: {sample['y_hit']:.4f} m, miss={sample['miss_distance']:.4f} m")

    plot_sample_rollout(sample, env)
    plt.savefig("gusty_trajectory.png")

    plot_cost_contours(all_results)
    plt.savefig("gusty_cost_contours.png")

    plot_rollout_variances(
        best["u"],
        env,
        aero,
        gust,
        sim,
        bounds,
        launch_noise,
        n_rollouts=200,
        seed=mc.seed + 1000,
    )
    plt.savefig("gusty_rollout_variance.png")


if __name__ == "__main__":
    main()
