import argparse
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simcentralconnect
from pyDOE import lhs
from scipy.stats import norm


class MonteCarloSimulator:
    def __init__(self):
        self.p = {
            "x0": np.array([0]),  
        }
        # CAGR parameters
        self.cagr_mean = 0.0282  # 2.82%
        self.cagr_std = 0.011  # 1.1%

        # Initial demand for 2024 (kg/h) - known value
        self.initial_demand_2024 = 240

        # Years for projection
        self.base_year = 2024  
        self.start_year = 2025  
        self.end_year = 2040   # Truncated to 2040 (15-year horizon)

    def calculate_demand_projection(self, cagr_samples_per_year: np.ndarray) -> dict:
        """Calculate yearly demand from 2024 to 2040 using D_t = D_{t-1}(1 + CAGR_t)
        CAGR varies dynamically per year across the 15-year horizon."""
        years = list(range(self.start_year, self.end_year + 1))
        demand_projection = {}

        # Include the base year (2024) with known demand
        demand_projection[self.base_year] = self.initial_demand_2024

        # Calculate demand for each year using year-specific CAGR
        for year_idx, year in enumerate(years):
            cagr_t = cagr_samples_per_year[year_idx]  # Unique growth rate for this year
            previous_year = year - 1
            demand_projection[year] = demand_projection[previous_year] * (1 + cagr_t)

        return demand_projection

    def run_monte_carlo_simulation(self, num_samples: int, results_dir: str):
        """Run Monte Carlo simulations by varying CAGR per year using Latin Hypercube Sampling."""
        np.random.seed(0)

        num_years = self.end_year - self.start_year + 1  # Exactly 16 years (2025-2040 inclusive)

        # Generate LHS samples: num_samples x num_years
        lhs_samples = lhs(num_years, samples=num_samples, criterion="maximin")

        # Transform each column to normal distribution using inverse CDF
        self.cagr_samples = np.zeros_like(lhs_samples)
        for year_idx in range(num_years):
            self.cagr_samples[:, year_idx] = norm.ppf(
                lhs_samples[:, year_idx], loc=self.cagr_mean, scale=self.cagr_std
            )

        # Print statistics about CAGR values (per year)
        print("\nCAGR Values Statistics (per year):")
        print(f"Shape: {self.cagr_samples.shape} (samples x years)")
        print(f"Global Min: {np.min(self.cagr_samples) * 100:.3f}%")
        print(f"Global Max: {np.max(self.cagr_samples) * 100:.3f}%")
        print(f"Global Mean: {np.mean(self.cagr_samples) * 100:.3f}%")
        print(f"Global Std: {np.std(self.cagr_samples) * 100:.3f}%")

        # Print per-year statistics
        print("\nPer-year CAGR statistics (first 5 years and last 5 years):")
        years = list(range(self.start_year, self.end_year + 1))
        for i in range(min(5, num_years)):
            print(f"Year {years[i]}: mean={np.mean(self.cagr_samples[:, i]) * 100:.3f}%, std={np.std(self.cagr_samples[:, i]) * 100:.3f}%")
        print("...")
        for i in range(-min(5, num_years), 0):
            print(f"Year {years[i]}: mean={np.mean(self.cagr_samples[:, i]) * 100:.3f}%, std={np.std(self.cagr_samples[:, i]) * 100:.3f}%")

        # Save the generated CAGR sample matrix
        try:
            cagr_samples_df = pd.DataFrame(
                self.cagr_samples, columns=[f"Year_{year}" for year in years]
            )
            cagr_samples_file = os.path.join(results_dir, "CAGR_Samples_Conventional_TimeDependent.csv")
            cagr_samples_df.to_csv(cagr_samples_file, index=False)
        except Exception:
            pass

        # Calculate demand projections for each MC sample trajectory
        self.demand_projections = []
        self.final_demands = []  # Demand values for 2040

        for i in range(num_samples):
            cagr_sample_per_year = self.cagr_samples[i, :]  
            demand_projection = self.calculate_demand_projection(cagr_sample_per_year)
            self.demand_projections.append(demand_projection)
            self.final_demands.append(demand_projection[self.end_year])

        self.final_demands = np.array(self.final_demands)

        # Calculate min/max demand bounds across ALL elements
        all_demands = []
        for projection in self.demand_projections:
            for year in years:
                all_demands.append(projection[year])

        print("\nDEMAND RANGE (All Years 2025-2040):")
        print(f"Minimum: {min(all_demands):.1f} kg/h")
        print(f"Maximum: {max(all_demands):.1f} kg/h")

        print("\nDemand in 2040 Statistics:")
        print(f"Minimum: {min(self.final_demands):.2f} kg/h")
        print(f"Maximum: {max(self.final_demands):.2f} kg/h")
        print(f"Mean: {np.mean(self.final_demands):.2f} kg/h")
        print(f"Median: {np.median(self.final_demands):.2f} kg/h")
        print(f"Standard Deviation: {np.std(self.final_demands):.2f} kg/h")

        results = []
        for i in range(num_samples):
            try:
                demand_projection = self.demand_projections[i]
                obj, simulationstatus, demand_2040 = self.MCsimulator(
                    self.p["x0"], demand_projection
                )
                mean_cagr = np.mean(self.cagr_samples[i, :])
                results.append((obj, simulationstatus, demand_2040, mean_cagr))
                print(f"Completed sample {i + 1}/{num_samples} - LCOP: {obj:.2f} ¤/t")

            except Exception as e:
                print(f"Error occurred at point {i + 1}: {str(e)}")
                mean_cagr = np.mean(self.cagr_samples[i, :])
                results.append((None, False, self.final_demands[i], mean_cagr))
                continue

        return results

    def MCsimulator(self, x: np.ndarray, demand_projection: dict) -> float:
        """AVEVA Process Simulation interface - runs simulation for 15 years with varying demand"""
        sc = simcentralconnect.connect().Result
        var_manager = sc.GetService("IVariableManager")
        sim_manager = sc.GetService("ISimulationManager")
        snap_manager = sc.GetService("ISnapshotManager")

        sim_name1 = "IbuprofenProcessSimulationConventional"
        snapshot_name = "Pro 1"
        TCI = 9276106

        total_discounted_opex = 0
        total_discounted_product = 0
        r = 0.1  

        years = list(range(self.start_year, self.end_year + 1))
        simulation_successful = True

        # Open full simulation workspace snapshot file ONCE at the start of the trajectory path
        try:
            sim_manager.OpenSimulation(sim_name1).Result
        except Exception as e:
            print(f"CRITICAL ERROR: Simulation '{sim_name1}' could not be opened: {str(e)}")
            return float("nan"), False, demand_projection[self.end_year]

        for year_idx, year in enumerate(years):
            pu = demand_projection[year]
            t = year_idx + 1  

            try:
                # Set dynamic throughput demand for this specific year
                var_manager.SetVariableValue(sim_name1, "Var104", pu, "kg/h", 90000).Result

                # Extract calculated economic parameters
                AnnualOPEX = var_manager.GetVariableValue(sim_name1, "EconSummary1.TotalOperatingCost", "¤/yr", 90000).Result
                AnnualLabor = var_manager.GetVariableValue(sim_name1, "EconSummary1.AnnualLaborCost", "¤", 90000).Result
                AnnualMaintenance = var_manager.GetVariableValue(sim_name1, "MaintenanceCost", "¤", 90000).Result
                AnnualProduct = var_manager.GetVariableValue(sim_name1, "IBU_crystals.W", "kg/h", 90000).Result
                ss = sim_manager.GetSimulationStatus(sim_name1).Result

                if not ss[2]:
                    print(f"WARNING: Simulation failed for year {year} (Demand: {pu:.1f} kg/h)")
                    simulation_successful = False

                try:
                    annual_total_opex = float(AnnualOPEX + AnnualLabor + AnnualMaintenance)
                    annual_total_product = float(AnnualProduct * 24 * 330)  
                except (TypeError, ValueError):
                    annual_total_opex = 0
                    annual_total_product = 0

                # Accumulate multi-year discounted cash flow totals
                discount_factor = (1 + r) ** t
                total_discounted_opex += annual_total_opex / discount_factor
                total_discounted_product += annual_total_product / discount_factor

            except Exception:
                simulation_successful = False
                continue  # Original logic: march forward to next years even if this year zeroed out

        # Revert snapshot memory back to base canvas baseline for the next independent scenario path
        try:
            snap_manager.RevertSnapshot(sim_name1, snapshot_name, 180000).Result
        except Exception as e:
            print(f"WARNING: Snapshot revert failed: {str(e)}")

        # Calculate single unified LCOP using 15-year horizon totals
        try:
            tci_float = float(TCI)
            opex_float = float(total_discounted_opex)
            product_float = float(total_discounted_product)

            LCOP = ((tci_float + opex_float) / product_float) * 1000  
        except Exception:
            LCOP = float("nan")

        return LCOP, simulation_successful, demand_projection[self.end_year]


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = script_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    num_samples = 100  
    simulator = MonteCarloSimulator()

    # Run simulations and get results
    results = simulator.run_monte_carlo_simulation(num_samples, results_dir)

    mean_cagr_per_sample = [r[3] for r in results]  
    results_df = pd.DataFrame(
        {
            "Sample_Number": range(1, num_samples + 1),
            "CAGR_Mean": mean_cagr_per_sample,  
            "Demand_2040": simulator.final_demands,
            "LCOP": [r[0] for r in results],
            "Simulation_Status": [r[1] for r in results],
        }
    )

    successful_lcop = results_df.loc[results_df["Simulation_Status"], "LCOP"]
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

        results_df["LCOP_Mean"] = mean_lcop
        results_df["LCOP_StdDev"] = std_lcop
    else:
        print("\nWarning: No successful simulations to analyze!")

    # Save results to CSV
    results_file = os.path.join(results_dir, "Conventional_MC_results_TimeDependent.csv")
    results_df.to_csv(results_file, index=False)

    # Create demand projection DataFrame for detailed analysis
    years = list(range(simulator.base_year, simulator.end_year + 1))  # 2024-2040
    demand_proj_df = pd.DataFrame(index=range(num_samples), columns=years)

    for i, projection in enumerate(simulator.demand_projections):
        for year in years:
            demand_proj_df.loc[i, year] = projection[year]

    demand_proj_file = os.path.join(results_dir, "Demand_Projections_Conventional_TimeDependent_2024_2040.csv")
    demand_proj_df.to_csv(demand_proj_file, index=True)

    # --- Plotting Phase ---
    plt.figure(figsize=(18, 12))

    # First subplot: CAGR Mean and Demand 2040
    plt.subplot(2, 3, 1)
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(results_df["Sample_Number"], results_df["CAGR_Mean"] * 100, color="green", label="CAGR Mean (%)", linewidth=1.5)
    ax2.plot(results_df["Sample_Number"], results_df["Demand_2040"], color="orange", label="Demand 2040 (kg/h)", linewidth=1.5)

    ax1.set_ylabel("CAGR Mean (%)", fontweight="bold", fontsize=10, color="green")
    ax2.set_ylabel("Demand 2040 (kg/h)", fontweight="bold", fontsize=10, color="orange")
    ax1.set_xlabel("Monte Carlo Sample Number", fontweight="bold", fontsize=10)
    plt.title("CAGR and Projected Demand for 2040 (Conventional)", fontweight="bold", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="green")
    ax2.tick_params(axis="y", labelcolor="orange")

    # Second subplot: LCOP Results
    plt.subplot(2, 3, 2)
    successful_mask = results_df["Simulation_Status"]
    failed_mask = ~results_df["Simulation_Status"]

    plt.plot(results_df.loc[successful_mask, "Sample_Number"], results_df.loc[successful_mask, "LCOP"],
             marker="o", linestyle="-", color="b", markersize=3, label="Successful")
    plt.plot(results_df.loc[failed_mask, "Sample_Number"], results_df.loc[failed_mask, "LCOP"],
             marker="x", linestyle="None", color="r", markersize=5, label="Failed")

    plt.ylabel("LCOP (¤/t)", fontweight="bold", fontsize=10)
    plt.xlabel("Monte Carlo Sample Number", fontweight="bold", fontsize=10)
    plt.legend()
    plt.title("LCOP Results (Conventional)", fontweight="bold", fontsize=12)

    # Third subplot: LCOP Distribution (Histogram)
    plt.subplot(2, 3, 3)
    if len(valid_lcop) > 0:
        n_bins = int(1 + 3.222 * np.log10(len(valid_lcop)))
        plt.hist(valid_lcop, bins=n_bins, color="purple", alpha=0.7, edgecolor="black", density=True)

        mean_lcop = valid_lcop.mean()
        std_lcop = valid_lcop.std()
        plt.axvline(mean_lcop, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_lcop:.2f}")
        plt.axvline(mean_lcop + std_lcop, color="orange", linestyle=":", linewidth=2, label=f"+1σ: {mean_lcop + std_lcop:.2f}")
        plt.axvline(mean_lcop - std_lcop, color="orange", linestyle=":", linewidth=2, label=f"-1σ: {mean_lcop - std_lcop:.2f}")

        plt.ylabel("Probability Density", fontweight="bold", fontsize=10)
        plt.xlabel("LCOP (¤/t)", fontweight="bold", fontsize=10)
        plt.title("LCOP Uncertainty Distribution (Conventional)", fontweight="bold", fontsize=12)
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)

    # Fourth subplot: Demand Growth Over Time (sample trajectories)
    plt.subplot(2, 1, 2)
    sample_indices = np.linspace(0, num_samples - 1, min(20, num_samples), dtype=int)

    for idx in sample_indices:
        projection = simulator.demand_projections[idx]
        demands = [projection[year] for year in years]
        plt.plot(years, demands, alpha=0.3, color="blue", linewidth=0.8)

    mean_demands = []
    for year in years:
        year_demands = [proj[year] for proj in simulator.demand_projections]
        mean_demands.append(np.mean(year_demands))

    plt.plot(years, mean_demands, color="red", linewidth=3, label="Mean Trajectory")

    plt.ylabel("Throughput (kg/h)", fontweight="bold", fontsize=10)
    plt.xlabel("Year", fontweight="bold", fontsize=10)
    plt.title("Demand Projections 2024-2040 (Conventional)", fontweight="bold", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = os.path.join(results_dir, "Conventional_MC_CAGR_results_TimeDependent.png")
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.show()