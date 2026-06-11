import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simcentralconnect
from pyDOE import lhs
from scipy.stats import norm


class MonteCarloSimulator:
    def __init__(self):
        self.p = {
            "x0": np.array([0]),  # Initial values for x, adjust as needed
        }
        # CAGR parameters
        self.cagr_mean = 0.0282  # 2.82%
        self.cagr_std = 0.011  # 1.1%

        # Initial demand for 2024 (kg/h) - known value
        self.initial_demand_2024 = 240

        # Years for projection
        self.base_year = 2024  # Base year with known demand
        self.start_year = 2025  # First projection year
        self.end_year = 2045

    def calculate_demand_projection(self, cagr_sample: float) -> dict:
        """Calculate yearly demand from 2024 to 2045 using D_t = D_{t-1}(1 + CAGR_t)
        Starting from known 2024 demand of 318.5 kg/h"""
        years = range(self.start_year, self.end_year + 1)
        demand_projection = {}

        # Include the base year (2024) with known demand
        demand_projection[self.base_year] = self.initial_demand_2024

        # Calculate 2025 demand from 2024 base
        demand_projection[self.start_year] = self.initial_demand_2024 * (
            1 + cagr_sample
        )  # 2025 demand

        # Calculate demand for each subsequent year (2026-2045)
        for year in years[1:]:  # Skip the first year (2025) as it's already calculated
            previous_year = year - 1
            demand_projection[year] = demand_projection[previous_year] * (
                1 + cagr_sample
            )

        return demand_projection

    def run_monte_carlo_simulation(self, num_samples: int, results_dir: str):
        """Run Monte Carlo simulations by varying CAGR and calculating demand projections using Latin Hypercube Sampling."""

        # Generate CAGR samples using Latin Hypercube Sampling
        np.random.seed(0)

        # Generate LHS samples in [0,1] space
        lhs_samples = lhs(1, samples=num_samples, criterion="maximin")

        # Transform to normal distribution using inverse CDF
        self.cagr_samples = norm.ppf(
            lhs_samples[:, 0], loc=self.cagr_mean, scale=self.cagr_std
        )

        # Sort the CAGR values in ascending order
        self.cagr_samples.sort()

        # Print some statistics about CAGR values
        print("\nCAGR Values Statistics:")
        print(f"Minimum: {min(self.cagr_samples) * 100:.3f}%")
        print(f"Maximum: {max(self.cagr_samples) * 100:.3f}%")
        print(f"Mean: {np.mean(self.cagr_samples) * 100:.3f}%")
        print(f"Median: {np.median(self.cagr_samples) * 100:.3f}%")
        print(f"Standard Deviation: {np.std(self.cagr_samples) * 100:.3f}%")

        # Print first few and last few values
        print("\nFirst 5 CAGR values:")
        print([f"{val * 100:.3f}%" for val in self.cagr_samples[:5]])
        print("\nLast 5 CAGR values:")
        print([f"{val * 100:.3f}%" for val in self.cagr_samples[-5:]])

        # Save the generated CAGR samples for reproducibility (Conventional)
        try:
            cagr_samples_df = pd.DataFrame({"CAGR": self.cagr_samples})
            cagr_samples_file = os.path.join(
                results_dir, "CAGR_Samples_Conventional_Fixed.csv"
            )
            cagr_samples_df.to_csv(cagr_samples_file, index=False)
        except Exception:
            # If saving fails, just continue — not critical
            pass

        # Calculate demand projections for each CAGR sample
        self.demand_projections = []
        self.final_demands = []  # Demand values for 2045

        for i, cagr_sample in enumerate(self.cagr_samples):
            demand_projection = self.calculate_demand_projection(cagr_sample)
            self.demand_projections.append(demand_projection)
            self.final_demands.append(demand_projection[self.end_year])

        # Convert to numpy array for easier handling
        self.final_demands = np.array(self.final_demands)

        # Calculate min/max demand across ALL years for if-loop definition
        years = list(range(self.start_year, self.end_year + 1))
        all_demands = []

        for projection in self.demand_projections:
            for year in years:
                all_demands.append(projection[year])

        min_demand_all = min(all_demands)
        max_demand_all = max(all_demands)

        print("\nDEMAND RANGE (All Years 2025-2045):")
        print(f"Minimum: {min_demand_all:.1f} kg/h")
        print(f"Maximum: {max_demand_all:.1f} kg/h")

        # Print statistics about final demands (2045)
        print("\nDemand in 2045 Statistics:")
        print(f"Minimum: {min(self.final_demands):.2f} kg/h")
        print(f"Maximum: {max(self.final_demands):.2f} kg/h")
        print(f"Mean: {np.mean(self.final_demands):.2f} kg/h")
        print(f"Median: {np.median(self.final_demands):.2f} kg/h")
        print(f"Standard Deviation: {np.std(self.final_demands):.2f} kg/h")

        # Store results
        results = []

        for i in range(num_samples):
            try:
                # Use the complete demand projection for 20 years (2025-2045)
                demand_projection = self.demand_projections[i]

                # Run the AVEVA simulation with the 20-year demand projection
                obj, simulationstatus, demand_2045 = self.MCsimulator(
                    self.p["x0"], demand_projection
                )
                results.append(
                    (obj, simulationstatus, demand_2045, self.cagr_samples[i])
                )

                print(f"Completed sample {i + 1}/{num_samples} - LCOP: {obj:.2f} ¤/t")

            except Exception as e:
                print(f"Error occurred at point {i + 1}: {str(e)}")
                results.append(
                    (None, False, self.final_demands[i], self.cagr_samples[i])
                )
                continue

        return results

    def MCsimulator(self, x: np.ndarray, demand_projection: dict) -> float:
        """AVEVA Process Simulation interface - runs simulation for 20 years with varying demand
        NON-MODULAR VERSION: Fixed reactor modules, no demand-based adjustments"""
        # Connect to AVEVA Process Simulation
        sc = simcentralconnect.connect().Result
        var_manager = sc.GetService("IVariableManager")
        sim_manager = sc.GetService("ISimulationManager")

        # Setup simulation
        sim_name1 = "IbuprofenProcessSimulationConventional"
        TCI = 10755108  # Capital Investment in ¤

        # Initialize totals for 20-year calculation
        total_discounted_opex = 0
        total_discounted_product = 0
        r = 0.1  # Discount rate

        # Simulate each year from 2025 to 2045 (20 years)
        years = list(range(self.start_year, self.end_year + 1))
        simulation_successful = True

        for year_idx, year in enumerate(years):
            # Get demand for this specific year
            pu = demand_projection[year]
            t = year_idx + 1  # Time index for discounting (1 to 20)

            try:
                # NON-MODULAR: Open simulation with fixed reactor configuration
                # No demand-based adjustments to reactor modules
                sim_manager.OpenSimulation(sim_name1).Result

                # Set demand for this year (only variable that changes)
                var_manager.SetVariableValue(
                    sim_name1, "Var104", pu, "kg/h", 90000
                ).Result

                # Get annual results
                AnnualOPEX = var_manager.GetVariableValue(
                    sim_name1, "EconSummary1.TotalOperatingCost", "¤/yr", 90000
                ).Result
                AnnualLabor = var_manager.GetVariableValue(
                    sim_name1, "EconSummary1.AnnualLaborCost", "¤", 90000
                ).Result
                AnnualMaintenance = var_manager.GetVariableValue(
                    sim_name1, "MaintenanceCost", "¤", 90000
                ).Result
                AnnualProduct = var_manager.GetVariableValue(
                    sim_name1, "IBU_crystals.W", "kg/h", 90000
                ).Result
                ss = sim_manager.GetSimulationStatus(sim_name1).Result

                # Check if simulation was successful for this year
                if not ss[2]:
                    print(
                        f"WARNING: Simulation failed for year {year} (Demand: {pu:.1f} kg/h)"
                    )
                    simulation_successful = False
                    # Continue without reverting snapshot - use current values anyway

                # Calculate discounted values for this year
                try:
                    annual_total_opex = float(
                        AnnualOPEX + AnnualLabor + AnnualMaintenance
                    )
                    annual_total_product = float(AnnualProduct * 24 * 330)  # kg/year
                except (TypeError, ValueError):
                    annual_total_opex = 0
                    annual_total_product = 0

                # Calculate and add discounted values to totals
                discount_factor = (1 + r) ** t
                discounted_opex = annual_total_opex / discount_factor
                discounted_product = annual_total_product / discount_factor

                total_discounted_opex += discounted_opex
                total_discounted_product += discounted_product

            except Exception:
                simulation_successful = False
                # Continue without reverting snapshot - try to proceed anyway
                continue

        # Calculate LCOP using 20-year totals

        # Ensure all values are floats and handle potential conversion issues
        try:
            tci_float = float(TCI)
            opex_float = float(total_discounted_opex)
            product_float = float(total_discounted_product)

            LCOP = (
                (tci_float + opex_float) / product_float
            ) * 1000  # Convert from ¤/kg to ¤/t
        except Exception:
            LCOP = float("nan")

        return (
            LCOP,
            simulation_successful,
            demand_projection[self.end_year],
        )  # Return 2045 demand for reference


if __name__ == "__main__":
    # Save results in the same directory as this script (portable across machines)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = script_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    parser = argparse.ArgumentParser(
        description="Run Monte Carlo simulations and optional convergence analysis."
    )
    parser.add_argument(
        "--samples",
        default="100",
        help="Comma-separated list of sample sizes to run (e.g. 20,50,100).",
    )
    parser.add_argument(
        "--convergence",
        action="store_true",
        help="Run permutation-based convergence analysis on the obtained LCOP values.",
    )
    parser.add_argument(
        "--conv-step",
        type=int,
        default=5,
        help="Step size for convergence sample sizes (default: 5).",
    )
    parser.add_argument(
        "--conv-repeats",
        type=int,
        default=200,
        help="Number of permutation repeats for convergence (default: 200).",
    )
    parser.add_argument(
        "--conv-out",
        default=None,
        help="Output prefix for convergence files (default: uses results_dir + suffix).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible permutations (default: 0).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=9000.0,
        help="LCOP threshold for ruin-probability (default: 9000).",
    )

    args = parser.parse_args()

    # Allow automated sequence of sample sizes when requested
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Run automated sample sizes [20,40,60,80,100,120].",
    )

    parser.add_argument(
        "--conv-maxsum",
        action="store_true",
        help="Also produce cumulative max-to-sum and max-to-mean convergence plots.",
    )

    # re-parse to include the new option when script is executed directly
    args = parser.parse_args()

    if args.auto:
        sample_list = [20, 40, 60, 80, 100, 120]
    else:
        sample_list = [int(s) for s in args.samples.split(",") if s.strip()]

    simulator = MonteCarloSimulator()

    def convergence_analysis(values, threshold=9000.0, step=5, repeats=200):
        values = np.asarray(values)
        n = values.size
        if n == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        step = max(1, int(step))
        sizes = np.arange(step, n + 1, step)
        if sizes.size == 0:
            sizes = np.array([n])
        all_mean = np.zeros((repeats, sizes.size), dtype=float)
        all_ruin = np.zeros((repeats, sizes.size), dtype=float)
        for r in range(repeats):
            perm = np.random.permutation(n)
            perm_vals = values[perm]
            for i, s in enumerate(sizes):
                sample = perm_vals[:s]
                all_mean[r, i] = np.nanmean(sample)
                all_ruin[r, i] = np.sum(sample > threshold) / float(s)
        mean_means = np.nanmean(all_mean, axis=0)
        mean_stds = np.nanstd(all_mean, axis=0, ddof=1)
        ruin_means = np.nanmean(all_ruin, axis=0)
        ruin_stds = np.nanstd(all_ruin, axis=0, ddof=1)
        return sizes, mean_means, mean_stds, ruin_means, ruin_stds

    def run_convergence_and_save(values, out_prefix, threshold, step, repeats, seed):
        np.random.seed(seed)
        sizes, mean_means, mean_stds, ruin_means, ruin_stds = convergence_analysis(
            values, threshold=threshold, step=step, repeats=repeats
        )
        if sizes.size == 0:
            print("No valid LCOP values for convergence analysis.")
            return
        # Plot mean convergence
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 1, 1)
        plt.plot(sizes, mean_means, "-o")
        plt.fill_between(
            sizes, mean_means - mean_stds, mean_means + mean_stds, alpha=0.2
        )
        plt.xlabel("Sample size n")
        plt.ylabel("Mean LCOP (¤/t)")
        plt.title("Convergence of Mean LCOP")
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(sizes, ruin_means, "-o")
        plt.fill_between(
            sizes, ruin_means - ruin_stds, ruin_means + ruin_stds, alpha=0.2
        )
        plt.xlabel("Sample size n")
        plt.ylabel(f"Ruin P(LCOP>{threshold})")
        plt.title("Convergence of Ruin Probability")
        plt.grid(True)

        plt.tight_layout()
        out_plot = f"{out_prefix}.png"
        plt.savefig(out_plot, dpi=300, bbox_inches="tight")
        print(f"Saved convergence plot to: {out_plot}")

        # Save CSV
        rows = []
        for i, s in enumerate(sizes):
            rows.append(
                {
                    "size": int(s),
                    "mean": mean_means[i],
                    "mean_std": mean_stds[i],
                    "ruin": ruin_means[i],
                    "ruin_std": ruin_stds[i],
                }
            )
        out_csv = f"{out_prefix}.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"Saved convergence CSV to: {out_csv}")

    def cumulative_maxsum_analysis(values, step=1):
        """Deterministic cumulative analysis: for increasing sample sizes compute
        cumulative mean, std, max/sum and max/mean to check whether a single
        sample dominates the total."""
        values = np.asarray(values)
        n = values.size
        if n == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        step = max(1, int(step))
        sizes = np.arange(step, n + 1, step)
        if sizes.size == 0:
            sizes = np.array([n])
        cum_means = np.zeros(sizes.size, dtype=float)
        cum_stds = np.zeros(sizes.size, dtype=float)
        max_over_sum = np.zeros(sizes.size, dtype=float)
        max_over_mean = np.zeros(sizes.size, dtype=float)
        for i, s in enumerate(sizes):
            sample = values[:s]
            cum_means[i] = np.nanmean(sample)
            cum_stds[i] = np.nanstd(sample, ddof=1) if s > 1 else 0.0
            total = np.nansum(sample)
            m = np.nanmax(sample)
            max_over_sum[i] = m / total if total != 0 else np.nan
            max_over_mean[i] = m / cum_means[i] if cum_means[i] != 0 else np.nan
        return sizes, cum_means, cum_stds, max_over_sum, max_over_mean

    def run_cumulative_maxsum_and_save(values, out_prefix, step=1):
        sizes, cum_means, cum_stds, max_over_sum, max_over_mean = (
            cumulative_maxsum_analysis(values, step=step)
        )
        if sizes.size == 0:
            print("No valid LCOP values for cumulative max/sum analysis.")
            return

        plt.figure(figsize=(10, 10))
        plt.subplot(3, 1, 1)
        plt.plot(sizes, cum_means, "-o")
        plt.fill_between(sizes, cum_means - cum_stds, cum_means + cum_stds, alpha=0.2)
        plt.xlabel("Sample size n")
        plt.ylabel("Cumulative mean LCOP (¤/t)")
        plt.title("Cumulative Mean LCOP")
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(sizes, max_over_sum, "-o", color="purple")
        plt.xlabel("Sample size n")
        plt.ylabel("Max / Sum")
        plt.title("Max-to-Sum Ratio (cumulative)")
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(sizes, max_over_mean, "-o", color="brown")
        plt.xlabel("Sample size n")
        plt.ylabel("Max / Mean")
        plt.title("Max-to-Mean Ratio (cumulative)")
        plt.grid(True)

        plt.tight_layout()
        out_plot = f"{out_prefix}_maxsum.png"
        plt.savefig(out_plot, dpi=300, bbox_inches="tight")
        print(f"Saved cumulative max/sum plot to: {out_plot}")

        rows = []
        for i, s in enumerate(sizes):
            rows.append(
                {
                    "size": int(s),
                    "cum_mean": cum_means[i],
                    "cum_std": cum_stds[i],
                    "max_over_sum": max_over_sum[i],
                    "max_over_mean": max_over_mean[i],
                }
            )
        out_csv = f"{out_prefix}_maxsum.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"Saved cumulative max/sum CSV to: {out_csv}")

    def cumulative_ms_moments(values, step=1):
        """Compute max-to-sum (M/S) ratios for mean, variance, skewness, kurtosis
        for cumulative sample sizes."""
        values = np.asarray(values)
        n = values.size
        if n == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        step = max(1, int(step))
        sizes = np.arange(step, n + 1, step)
        if sizes.size == 0:
            sizes = np.array([n])

        ms_mean = np.zeros(sizes.size, dtype=float)
        ms_var = np.zeros(sizes.size, dtype=float)
        ms_skew = np.zeros(sizes.size, dtype=float)
        ms_kurt = np.zeros(sizes.size, dtype=float)

        for i, s in enumerate(sizes):
            sample = values[:s]
            # Mean contributions (use absolute contributions to avoid cancellation)
            abs_vals = np.abs(sample)
            denom_mean = np.nansum(abs_vals)
            ms_mean[i] = np.nanmax(abs_vals) / denom_mean if denom_mean != 0 else np.nan

            # Central moments using sample mean at size s
            mean_s = np.nanmean(sample)
            dev = sample - mean_s
            contrib_var = dev**2
            denom_var = np.nansum(contrib_var)
            ms_var[i] = np.nanmax(contrib_var) / denom_var if denom_var != 0 else np.nan

            contrib_skew = np.abs(dev**3)
            denom_skew = np.nansum(contrib_skew)
            ms_skew[i] = (
                np.nanmax(contrib_skew) / denom_skew if denom_skew != 0 else np.nan
            )

            contrib_kurt = np.abs(dev**4)
            denom_kurt = np.nansum(contrib_kurt)
            ms_kurt[i] = (
                np.nanmax(contrib_kurt) / denom_kurt if denom_kurt != 0 else np.nan
            )

        return sizes, ms_mean, ms_var, ms_skew, ms_kurt

    def run_ms_moments_and_save(values, out_prefix, step=1):
        sizes, ms_mean, ms_var, ms_skew, ms_kurt = cumulative_ms_moments(
            values, step=step
        )
        if sizes.size == 0:
            print("No valid LCOP values for M/S moment analysis.")
            return

        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.plot(sizes, ms_mean, "-o")
        plt.xlabel("Sample size n")
        plt.ylabel("M/S (mean)")
        plt.title("Max-to-Sum: Mean")
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(sizes, ms_var, "-o", color="orange")
        plt.xlabel("Sample size n")
        plt.ylabel("M/S (variance)")
        plt.title("Max-to-Sum: Variance")
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.plot(sizes, ms_skew, "-o", color="purple")
        plt.xlabel("Sample size n")
        plt.ylabel("M/S (skewness)")
        plt.title("Max-to-Sum: Skewness")
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.plot(sizes, ms_kurt, "-o", color="brown")
        plt.xlabel("Sample size n")
        plt.ylabel("M/S (kurtosis)")
        plt.title("Max-to-Sum: Kurtosis")
        plt.grid(True)

        plt.tight_layout()
        out_plot = f"{out_prefix}_ms_moments.png"
        plt.savefig(out_plot, dpi=300, bbox_inches="tight")
        print(f"Saved M/S moments plot to: {out_plot}")

        rows = []
        for i, s in enumerate(sizes):
            rows.append(
                {
                    "size": int(s),
                    "ms_mean": ms_mean[i],
                    "ms_var": ms_var[i],
                    "ms_skew": ms_skew[i],
                    "ms_kurt": ms_kurt[i],
                }
            )
        out_csv = f"{out_prefix}_ms_moments.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"Saved M/S moments CSV to: {out_csv}")

    # Run simulations for each requested sample size
    for num_samples in sample_list:
        print(f"\nRunning Monte Carlo with num_samples={num_samples}")
        results = simulator.run_monte_carlo_simulation(num_samples, results_dir)

        results_df = pd.DataFrame(
            {
                "Sample_Number": range(1, num_samples + 1),
                "CAGR": simulator.cagr_samples,
                "Demand_2045": simulator.final_demands,
                "LCOP": [r[0] for r in results],
                "Simulation_Status": [r[1] for r in results],
            }
        )

        successful_lcop = results_df.loc[results_df["Simulation_Status"], "LCOP"]
        valid_lcop = successful_lcop[np.isfinite(successful_lcop)]

        # Save results CSV per sample size
        out_results = os.path.join(
            results_dir, f"Conventional_MC_results_Fixed_N{num_samples}.csv"
        )
        results_df.to_csv(out_results, index=False)
        print(f"Saved simulation results to: {out_results}")

        if args.convergence:
            # Use deterministic cumulative convergence (no permutation-based analysis)
            out_pref = args.conv_out or os.path.join(
                results_dir, f"Conventional_Convergence_N{num_samples}"
            )
            # Save cumulative M/S analysis for moments (mean, var, skew, kurt)
            run_ms_moments_and_save(
                valid_lcop.to_numpy(), out_pref, step=args.conv_step
            )

    # Create a DataFrame to store results
    results_df = pd.DataFrame(
        {
            "Sample_Number": range(1, num_samples + 1),
            "CAGR": simulator.cagr_samples,
            "Demand_2045": simulator.final_demands,
            "LCOP": [r[0] for r in results],
            "Simulation_Status": [r[1] for r in results],
        }
    )

    # Calculate Monte Carlo statistics for LCOP (successful simulations only)
    successful_lcop = results_df.loc[results_df["Simulation_Status"], "LCOP"]

    # Filter out NaN and infinite values
    valid_lcop = successful_lcop[np.isfinite(successful_lcop)]

    if len(valid_lcop) > 0:
        mean_lcop = valid_lcop.mean()
        std_lcop = valid_lcop.std()

        print("\n" + "=" * 40)
        print("MONTE CARLO UNCERTAINTY ANALYSIS - NON-MODULAR")
        print("=" * 40)
        print(f"Successful Simulations: {len(successful_lcop)}/{num_samples}")
        print(f"Valid LCOP Values: {len(valid_lcop)}/{num_samples}")
        print(f"LCOP Mean: {mean_lcop:.2f} ¤/t")
        print(f"LCOP Standard Deviation: {std_lcop:.2f} ¤/t")
        print(f"LCOP Minimum: {valid_lcop.min():.2f} ¤/t")
        print(f"LCOP Maximum: {valid_lcop.max():.2f} ¤/t")
        print("=" * 40)

        # Add statistics to the results DataFrame
        results_df["LCOP_Mean"] = mean_lcop
        results_df["LCOP_StdDev"] = std_lcop
    else:
        print("\nWarning: No successful simulations to analyze!")

    # Save results to CSV
    results_file = os.path.join(results_dir, "Conventional_MC_results_Fixed.csv")
    results_df.to_csv(results_file, index=False)

    # Create demand projection DataFrame for detailed analysis
    years = list(
        range(simulator.base_year, simulator.end_year + 1)
    )  # Include 2024-2045
    demand_proj_df = pd.DataFrame(index=range(num_samples), columns=years)

    for i, projection in enumerate(simulator.demand_projections):
        for year in years:
            demand_proj_df.loc[i, year] = projection[year]

    # Save demand projections to CSV
    demand_proj_file = os.path.join(
        results_dir,
        "Demand_Projections_Conventional_Fixed_2024_2045.csv",
    )
    demand_proj_df.to_csv(demand_proj_file, index=True)

    # Create the main results plot
    plt.figure(figsize=(18, 12))

    # First subplot: CAGR and Demand 2045
    plt.subplot(2, 3, 1)
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(
        results_df["Sample_Number"],
        results_df["CAGR"] * 100,
        color="green",
        label="CAGR (%)",
        linewidth=1.5,
    )
    ax2.plot(
        results_df["Sample_Number"],
        results_df["Demand_2045"],
        color="orange",
        label="Demand 2045 (kg/h)",
        linewidth=1.5,
    )

    ax1.set_ylabel("CAGR (%)", fontweight="bold", fontsize=10, color="green")
    ax2.set_ylabel("Demand 2045 (kg/h)", fontweight="bold", fontsize=10, color="orange")
    ax1.set_xlabel("Monte Carlo Sample Number", fontweight="bold", fontsize=10)
    plt.title(
        "CAGR and Projected Demand for 2045 (Conventional)",
        fontweight="bold",
        fontsize=12,
    )
    ax1.tick_params(axis="y", labelcolor="green")
    ax2.tick_params(axis="y", labelcolor="orange")

    # Second subplot: LCOP Results
    plt.subplot(2, 3, 2)
    # Masks
    successful_mask = results_df["Simulation_Status"]
    failed_mask = ~results_df["Simulation_Status"]

    # Plot successful (blue circles)
    plt.plot(
        results_df.loc[successful_mask, "Sample_Number"],
        results_df.loc[successful_mask, "LCOP"],
        marker="o",
        linestyle="-",
        color="b",
        markersize=3,
        label="Successful",
    )
    # Plot failed (red x)
    plt.plot(
        results_df.loc[failed_mask, "Sample_Number"],
        results_df.loc[failed_mask, "LCOP"],
        marker="x",
        linestyle="None",
        color="r",
        markersize=5,
        label="Failed",
    )

    plt.ylabel("LCOP (¤/t)", fontweight="bold", fontsize=10)
    plt.xlabel("Monte Carlo Sample Number", fontweight="bold", fontsize=10)
    plt.legend()
    plt.title("LCOP Results (Conventional)", fontweight="bold", fontsize=12)

    # Third subplot: LCOP Distribution (Histogram)
    plt.subplot(2, 3, 3)
    if len(valid_lcop) > 0:
        n_bins = int(1 + 3.222 * np.log10(len(valid_lcop)))
        plt.hist(
            valid_lcop,
            bins=n_bins,
            color="purple",
            alpha=0.7,
            edgecolor="black",
            density=True,
        )

        # Add vertical lines for mean and std dev
        mean_lcop = valid_lcop.mean()
        std_lcop = valid_lcop.std()
        plt.axvline(
            mean_lcop,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_lcop:.2f}",
        )
        plt.axvline(
            mean_lcop + std_lcop,
            color="orange",
            linestyle=":",
            linewidth=2,
            label=f"+1σ: {mean_lcop + std_lcop:.2f}",
        )
        plt.axvline(
            mean_lcop - std_lcop,
            color="orange",
            linestyle=":",
            linewidth=2,
            label=f"-1σ: {mean_lcop - std_lcop:.2f}",
        )

        plt.ylabel("Probability Density", fontweight="bold", fontsize=10)
        plt.xlabel("LCOP (¤/t)", fontweight="bold", fontsize=10)
        plt.title(
            "LCOP Uncertainty Distribution (Conventional)",
            fontweight="bold",
            fontsize=12,
        )
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)

    # Fourth subplot: Demand Growth Over Time (sample trajectories)
    plt.subplot(2, 1, 2)

    # Plot several sample trajectories
    sample_indices = np.linspace(0, num_samples - 1, min(20, num_samples), dtype=int)

    for idx in sample_indices:
        projection = simulator.demand_projections[idx]
        demands = [projection[year] for year in years]
        plt.plot(years, demands, alpha=0.3, color="blue", linewidth=0.8)

    # Plot mean trajectory
    mean_demands = []
    for year in years:
        year_demands = [proj[year] for proj in simulator.demand_projections]
        mean_demands.append(np.mean(year_demands))

    plt.plot(years, mean_demands, color="red", linewidth=3, label="Mean Trajectory")

    plt.ylabel("Throughput (kg/h)", fontweight="bold", fontsize=10)
    plt.xlabel("Year", fontweight="bold", fontsize=10)
    plt.title(
        "Demand Projections 2024-2045 (Conventional)", fontweight="bold", fontsize=12
    )
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    plot_file = os.path.join(results_dir, "Conventional_MC_CAGR_results_Fixed.png")
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.show()
