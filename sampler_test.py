"""
MC sampler test script.
"""

import os

import corner
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.qmc import LatinHypercube

OUTPUT_DIR_SAMPLER = './media/img/param'
N_DIMS = 19
N_SAMPLES = 10000
BOUNDS = np.array(
    [
        [0.00, 0.99],  # ecc 1
        [0.00, 0.99],  # ecc 2
        [0.00, 0.99],  # ecc 3
        [2.00, 7.00],  # sep 1
        [2.00, 7.00],  # sep 2
        [2.00, 7.00],  # sep 3
        [0.10, 1.00],  # v_mag 1
        [0.10, 1.00],  # v_mag 2
        [0.00, 5.00],  # impact_parameter 1
        [0.00, 5.00],  # impact_parameter 2
        [0.00, np.pi / 2],  # theta 1
        [0.00, np.pi / 2],  # theta 2
        [0.00, 2 * np.pi],  # phi 1
        [0.00, 2 * np.pi],  # phi 2
        [0.00, 2 * np.pi],  # psi 1
        [0.00, 2 * np.pi],  # psi 2
        [0.00, 2 * np.pi],  # true_anomalies 1
        [0.00, 2 * np.pi],  # true_anomalies 2
        [0.00, 2 * np.pi],  # true_anomalies 3
    ]
)


def sampler_nd(
    n_samples: int = N_SAMPLES,
    n_dims: int = N_DIMS,
    bounds: np.ndarray = BOUNDS,
    save_dir: str = OUTPUT_DIR_SAMPLER,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Param sampler supporting `n` dimensions with two methods.
    - Uniform: `numpy.random.uniform()`
    - LHS: `scipy.stats.qmc.LatinHyperCube()`

    Args:
        n_samples (int, optional): Defaults to N_SAMPLES.
        n_dims (int, optional): Defaults to N_DIMS.
        bounds (np.ndarray, optional): Defaults to BOUNDS.
        save_dir (str, optional): Defaults to OUTPUT_DIR_SAMPLER.

    Returns:
        tuple[np.ndarray, np.ndarray]: Samples (Uniform), Samples (LHS)
    """
    # init
    os.makedirs(save_dir, exist_ok=True)

    # local repo
    low, high = sampler_nd_bounds(bounds)

    # numpy sampler
    samples_np = np.random.uniform(low, high, (n_samples, n_dims))

    # scipy sampler <- choose this
    sampler_sp = LatinHypercube(d=n_dims)
    samples_sp = sampler_sp.random(n=n_samples) * (high - low) + low

    return samples_np, samples_sp


def sampler_nd_bounds(bounds: np.ndarray = BOUNDS) -> tuple[np.ndarray, np.ndarray]:
    """
    Get `low` and `high` param bounds.

    Args:
        bounds (np.ndarray, optional): Defaults to BOUNDS.

    Returns:
        tuple[np.ndarray, np.ndarray]: bounds (`low`), bounds (`high`)
    """
    low, high = bounds[:, 0], bounds[:, 1]
    return low, high


def sampler_nd_coverage(
    samples: np.ndarray, n_dims: int = N_DIMS, bounds: np.ndarray = BOUNDS
) -> list:
    """
    Get param sapce `coverage` for some `samples`

    Args:
        samples (np.ndarray): Generated samples.
        n_dims (int, optional): Defaults to N_DIMS.
        bounds (np.ndarray, optional): Defaults to BOUNDS.

    Returns:
        list: Coverage of some `samples`
    """
    low, high = sampler_nd_bounds(bounds)
    coverage = [np.ptp(samples[:, i]) / (high[i] - low[i]) for i in range(n_dims)]
    return coverage


def sampler_nd_coverage_plot(
    samples_np: np.ndarray,
    samples_sp: np.ndarray,
    n_samples: int = N_SAMPLES,
    n_dims: int = N_DIMS,
    bounds: np.ndarray = BOUNDS,
    save_dir: str = OUTPUT_DIR_SAMPLER,
) -> None:
    """
    General plotter for sample coverage visualization.

    Args:
        samples_np (np.ndarray): Uniform samples from `numpy.random.uniform()`.
        samples_sp (np.ndarray): LHS samples from `scipy.stats.qmc.LatinHyperCube()`
        n_samples (int, optional): Defaults to N_SAMPLES.
        n_dims (int, optional): Defaults to N_DIMS.
        bounds (np.ndarray, optional): Defaults to BOUNDS.
        save_dir (str, optional): Defaults to OUTPUT_DIR_SAMPLER.
    """

    # local repo
    low, high = sampler_nd_bounds(bounds)
    coverage_np = sampler_nd_coverage(samples_np)
    coverage_sp = sampler_nd_coverage(samples_sp)

    # fig init
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Uniform coverage
    bars_np = ax1.bar(range(n_dims), coverage_np, color='blue', alpha=0.7, label='Uniform')
    ax1.axhline(1.0, color='red', ls='--', lw=2, label='Perfect 100%')
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel('Parameter Index')
    ax1.set_ylabel('Range Coverage')
    ax1.set_title('Uniform: Coverage')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: LHS coverage
    bars_sp = ax2.bar(range(n_dims), coverage_sp, color='green', alpha=0.8, label='LHS')
    ax2.axhline(1.0, color='red', ls='--', lw=2, label='Perfect 100%')
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel('Parameter Index')
    ax2.set_ylabel('Range Coverage')
    ax2.set_title('LHS: Coverage')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 1d param converage comparison
    for i, cov in enumerate(coverage_sp):
        if cov > 0.99:
            ax2.text(i, cov + 0.01, '✓', ha='center', va='bottom', fontsize=14, color='red')

    plt.suptitle('Parameter Space Coverage Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/1d_param_coverage_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # corner
    fig_left = plt.figure(figsize=(20, 20))
    corner.corner(
        samples_np,
        fig=fig_left,
        color='blue',
        alpha=0.6,
        labels=[f'p{i}' for i in range(n_dims)],
        range=[(low[i], high[i]) for i in range(n_dims)],
        quantiles=[0.25, 0.5, 0.75],
        smooth=0.05,
        plot_datapoints=False,
        plot_density=False,
    )

    fig_left.suptitle(f'Uniform Corner Plot ({n_samples} samples)', fontsize=16)
    plt.savefig(f'{save_dir}/uniform.png', dpi=100, bbox_inches='tight')
    plt.close(fig_left)

    fig_right = plt.figure(figsize=(20, 20))
    corner.corner(
        samples_sp,
        fig=fig_right,
        color='green',
        alpha=0.7,
        labels=[f'p{i}' for i in range(n_dims)],
        range=[(low[i], high[i]) for i in range(n_dims)],
        quantiles=[0.25, 0.5, 0.75],
        smooth=0.05,
        plot_datapoints=False,
    )  # SAFE

    fig_right.suptitle(f'LHS Corner Plot ({n_samples} samples)', fontsize=16)
    plt.savefig(f'{save_dir}/lhs.png', dpi=100, bbox_inches='tight')
    plt.close(fig_right)

    # Summary stats
    print("\n" + "=" * 50)
    print("1D COVERAGE SUMMARY")
    print("=" * 50)
    print(f"Uniform sampling avg: {np.mean(coverage_np):.1%} ± {np.std(coverage_np):.1%}")
    print(f"LHS     sampling avg: {np.mean(coverage_sp):.1%} ± {np.std(coverage_sp):.1%}")
    print(f"LHS          perfect: {sum(1 for x in coverage_sp if x > 0.99)}/19")
    print(f"Uniform      perfect: {sum(1 for x in coverage_np if x > 0.99)}/19")
    print("=" * 50)


if __name__ == "__main__":
    samples_np, samples_sp = sampler_nd()
    sampler_nd_coverage_plot(samples_np, samples_sp)
