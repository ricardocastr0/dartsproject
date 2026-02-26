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
class ControlBounds:
    v_min: float
    v_max: float
    theta_min_deg: float
    theta_max_deg: float


@dataclass(frozen=True)
class NoiseParams:
    sigma_v: float
    sigma_theta_deg: float

    @property
    def sigma_theta_rad(self) -> float:
        return np.radians(self.sigma_theta_deg)


@dataclass(frozen=True)
class MonteCarloParams:
    n_rollouts: int
    seed: int


# Layer A: deterministic physics map
# input: u = (v, theta_deg)
# output: y_hit (+ optional path details)
def simulate_deterministic(u, env: EnvironmentParams, n_steps=100, return_path=False):
    v, theta_deg = u
    theta_rad = np.radians(theta_deg)

    t_flight = env.board_distance / (v * np.cos(theta_rad))
    t = np.linspace(0.0, t_flight, num=n_steps)
    x = np.linspace(0.0, env.board_distance, num=n_steps)
    y = env.release_height + v * np.sin(theta_rad) * t - 0.5 * env.gravity * t**2
    y_hit = y[-1]

    result = {
        "v": v,
        "theta_deg": theta_deg,
        "y_hit": y_hit,
        "t_flight": t_flight,
    }
    if return_path:
        result["t"] = t
        result["x"] = x
        result["y"] = y

    return result


def score_from_y_hit(y_hit, env: EnvironmentParams):
    miss_distance = abs(y_hit - env.center_height)
    on_board = miss_distance <= env.board_diameter / 2

    if miss_distance <= env.bullseye_diameter / 2:
        return 50, "bullseye", miss_distance
    if miss_distance <= env.outer_bull_diameter / 2:
        return 25, "outer bull", miss_distance
    if on_board:
        return 1, "board", miss_distance
    return 0, "miss", miss_distance


# Layer B: noise injection model
# epsilon ~ N(0, Sigma), u_tilde = u + epsilon, then clip to bounds.
def sample_noisy_control(u, bounds: ControlBounds, noise: NoiseParams, rng):
    v_nom, theta_nom_deg = u

    eps_v = rng.normal(0.0, noise.sigma_v)
    eps_theta_rad = rng.normal(0.0, noise.sigma_theta_rad)

    v_noisy = np.clip(v_nom + eps_v, bounds.v_min, bounds.v_max)
    theta_noisy_deg = np.clip(
        theta_nom_deg + np.degrees(eps_theta_rad),
        bounds.theta_min_deg,
        bounds.theta_max_deg,
    )

    return (float(v_noisy), float(theta_noisy_deg)), (float(eps_v), float(eps_theta_rad))


def run_one_projectile(u_nominal, env, bounds, noise, rng, n_steps=100, return_path=False):
    u_noisy, epsilon = sample_noisy_control(u_nominal, bounds, noise, rng)
    sim = simulate_deterministic(u_noisy, env, n_steps=n_steps, return_path=return_path)

    score, zone, miss_distance = score_from_y_hit(sim["y_hit"], env)
    error = sim["y_hit"] - env.center_height

    result = {
        "u_nominal": u_nominal,
        "u_noisy": u_noisy,
        "epsilon": epsilon,
        "y_hit": sim["y_hit"],
        "error": error,
        "cost": error**2,
        "score": score,
        "zone": zone,
        "miss_distance": miss_distance,
        "t_flight": sim["t_flight"],
    }
    if return_path:
        result["t"] = sim["t"]
        result["x"] = sim["x"]
        result["y"] = sim["y"]

    return result


# Layer C: Monte Carlo estimator of expected cost
# J_hat(u) = (1/N) * sum_i (e_i)^2
def estimate_expected_cost(u_nominal, env, bounds, noise, mc, rng):
    rollouts = []
    errors = []

    for i in range(mc.n_rollouts):
        rollout = run_one_projectile(
            u_nominal,
            env,
            bounds,
            noise,
            rng,
            return_path=(i == 0),
        )
        rollouts.append(rollout)
        errors.append(rollout["error"])

    errors = np.asarray(errors)
    j_hat = float(np.mean(errors**2))

    score_mean = float(np.mean([r["score"] for r in rollouts]))
    bullseye_rate = float(np.mean([r["score"] == 50 for r in rollouts]))

    return {
        "u_nominal": u_nominal,
        "J_hat": j_hat,
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "mean_score": score_mean,
        "bullseye_rate": bullseye_rate,
        "sample_rollout": rollouts[0],
    }


# Layer D: optimizer (grid search version)
def grid_search_robust(env, bounds, noise, mc, v_step=0.1, theta_step=0.1):
    rng = np.random.default_rng(mc.seed)

    v_values = np.arange(bounds.v_min, bounds.v_max + v_step, v_step)
    theta_values = np.arange(bounds.theta_min_deg, bounds.theta_max_deg + theta_step, theta_step)

    best_eval = None
    all_evals = []

    for theta_deg in theta_values:
        for v in v_values:
            u = (float(v), float(theta_deg))
            evaluation = estimate_expected_cost(u, env, bounds, noise, mc, rng)
            all_evals.append(evaluation)

            if best_eval is None:
                best_eval = evaluation
                continue

            if evaluation["J_hat"] < best_eval["J_hat"]:
                best_eval = evaluation

    return best_eval, all_evals


def plot_single_projectile(rollout, env):
    if "x" not in rollout or "y" not in rollout:
        sim = simulate_deterministic(rollout["u_noisy"], env, return_path=True)
        x, y = sim["x"], sim["y"]
    else:
        x, y = rollout["x"], rollout["y"]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, "--", color="steelblue")
    plt.xlabel("Distance (m)")
    plt.ylabel("Height (m)")
    plt.title("Single Projectile (Noisy Rollout)")

    y_max = max(y)
    y_upper = y_max if env.center_height + env.board_diameter / 2 < y_max < 5 else env.center_height + env.board_diameter / 2 + 0.5
    plt.xlim(0, env.board_distance + 0.5)
    plt.ylim(0, y_upper)

    x_board = env.board_distance
    c = env.center_height

    plt.vlines(x_board, c - env.board_diameter / 2, c + env.board_diameter / 2, color="red", linewidth=4)
    plt.vlines(
        x_board,
        c - env.outer_bull_diameter / 2,
        c + env.outer_bull_diameter / 2,
        color="orange",
        linewidth=6,
    )
    plt.vlines(
        x_board,
        c - env.bullseye_diameter / 2,
        c + env.bullseye_diameter / 2,
        color="green",
        linewidth=8,
    )

    plt.scatter([x_board], [rollout["y_hit"]], color="blue", zorder=5)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()


def main():
    # Parameters to define before optimization
    env = EnvironmentParams(
        board_distance=2.37,
        center_height=1.73,
        release_height=1.0,
        board_diameter=0.451,
        outer_bull_diameter=0.032,
        bullseye_diameter=0.0127,
    )

    bounds = ControlBounds(
        v_min=10.0,
        v_max=20.0,
        theta_min_deg=10.0,
        theta_max_deg=50.0,
    )

    noise = NoiseParams(
        sigma_v=0.25,
        sigma_theta_deg=0.6,
    )

    mc = MonteCarloParams(
        n_rollouts=80,
        seed=7,
    )

    best_eval, _ = grid_search_robust(
        env,
        bounds,
        noise,
        mc,
        v_step=0.5,
        theta_step=0.5,
    )

    best_u = best_eval["u_nominal"]
    print(f"Best nominal control (v, theta_deg): {best_u}")
    print(f"Robust objective J_hat: {best_eval['J_hat']:.6f}")
    print(f"Mean score: {best_eval['mean_score']:.3f}")
    print(f"Bullseye rate: {best_eval['bullseye_rate']:.3f}")
    print(f"Mean error: {best_eval['mean_error']:.4f} m, Std error: {best_eval['std_error']:.4f} m")

    rollout = best_eval["sample_rollout"]
    print(f"Sample rollout score: {rollout['score']} ({rollout['zone']})")
    print(f"Sample rollout y_hit: {rollout['y_hit']:.4f} m, miss: {rollout['miss_distance']:.4f} m")

    plot_single_projectile(rollout, env)
    plt.savefig("trajectory.png")


if __name__ == "__main__":
    main()
