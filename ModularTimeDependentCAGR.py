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
            "x0": np.array([0]),  
        }
        # CAGR parameters
        self.cagr_mean = 0.0282  # 2.82%
        self.cagr_std = 0.011    # 1.1%

        # Initial demand for 2024 (kg/h) - using 7.5% of world demand scenario
        self.initial_demand_2024 = 240

        # Years for projection
        self.base_year = 2024  
        self.start_year = 2025  
        self.end_year = 2040   # 16 years (2025-2040 inclusive)

    def calculate_demand_projection(self, cagr_samples_per_year: np.ndarray) -> dict:
        """Calculate yearly demand from 2024 to 2040 using D_t = D_{t-1}(1 + CAGR_t)"""
        years = list(range(self.start_year, self.end_year + 1))
        demand_projection = {}

        # Include the base year (2024) with known demand
        demand_projection[self.base_year] = self.initial_demand_2024

        # Calculate demand for each year using year-specific CAGR
        for year_idx, year in enumerate(years):
            cagr_t = cagr_samples_per_year[year_idx]  # Year-specific CAGR sample
            previous_year = year - 1
            demand_projection[year] = demand_projection[previous_year] * (1 + cagr_t)

        return demand_projection

    def run_monte_carlo_simulation(self, num_samples: int, results_dir: str):
        """Run Monte Carlo simulations by varying CAGR per year using Latin Hypercube Sampling."""
        np.random.seed(0)
        num_years = self.end_year - self.start_year + 1  # 16 years

        # Generate LHS samples: num_samples x num_years
        lhs_samples = lhs(num_years, samples=num_samples, criterion="maximin")

        # Transform each column to normal distribution using inverse CDF
        self.cagr_samples = np.zeros_like(lhs_samples)
        for year_idx in range(num_years):
            self.cagr_samples[:, year_idx] = norm.ppf(
                lhs_samples[:, year_idx], loc=self.cagr_mean, scale=self.cagr_std
            )

        print("\nCAGR Values Statistics (per year):")
        print(f"Shape: {self.cagr_samples.shape} (samples x years)")
        print(f"Global Min: {np.min(self.cagr_samples) * 100:.3f}%")
        print(f"Global Max: {np.max(self.cagr_samples) * 100:.3f}%")
        print(f"Global Mean: {np.mean(self.cagr_samples) * 100:.3f}%")

        # Save the generated CAGR samples for reproducibility (Modular)
        try:
            years = list(range(self.start_year, self.end_year + 1))
            cagr_samples_df = pd.DataFrame(self.cagr_samples, columns=[f"Year_{year}" for year in years])
            cagr_samples_file = os.path.join(results_dir, "CAGR_Samples_Modular_TimeDependent.csv")
            cagr_samples_df.to_csv(cagr_samples_file, index=False)
        except Exception:
            pass

        self.demand_projections = []
        self.final_demands = []  # Demand values for 2040

        for i in range(num_samples):
            cagr_sample_per_year = self.cagr_samples[i, :]
            demand_projection = self.calculate_demand_projection(cagr_sample_per_year)
            self.demand_projections.append(demand_projection)
            self.final_demands.append(demand_projection[self.end_year])

        self.final_demands = np.array(self.final_demands)

        # Calculate results
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
        """AVEVA Process Simulation interface - runs simulation for 16 years with dynamic switching logic"""
        sc = simcentralconnect.connect().Result
        var_manager = sc.GetService("IVariableManager")
        sim_manager = sc.GetService("ISimulationManager")
        snap_manager = sc.GetService("ISnapshotManager")

        sim_name1 = "IbuprofenProcessSimulationModular_Onecarbo"
        sim_name2 = "IbuprofenProcessSimulationModular_Twocarbo"
        snapshot_name = "Pro 1"
        TCI = 11185631  # Capital Investment in ¤

        total_discounted_opex = 0
        total_discounted_product = 0
        r = 0.1  # Discount rate

        years = list(range(self.start_year, self.end_year + 1))
        simulation_successful = True
        active_sims_opened = set()

        for year_idx, year in enumerate(years):
            pu = demand_projection[year]
            t = year_idx + 1  # Time index for discounting (1 to 16)

            try:
                # --- SUPERVISORY LAYER SWITCHING LOGIC ---
                if pu > 335 and pu < 490:
                    active_sim = sim_name2
                    if active_sim not in active_sims_opened:
                        sim_manager.OpenSimulation(active_sim).Result
                        active_sims_opened.add(active_sim)
                    var_manager.SetVariableValue(active_sim, "SP3.OutRatio[S45]", 0.001, "fraction", 90000).Result
                    var_manager.SetVariableValue(active_sim, "SP4.OutRatio[S54]", 0.999, "fraction", 90000).Result
                elif pu >= 490:
                    active_sim = sim_name2
                    if active_sim not in active_sims_opened:
                        sim_manager.OpenSimulation(active_sim).Result
                        active_sims_opened.add(active_sim)
                    var_manager.SetVariableValue(active_sim, "SP3.OutRatio[S45]", 0.999, "fraction", 90000).Result
                    var_manager.SetVariableValue(active_sim, "SP4.OutRatio[S54]", 0.999, "fraction", 90000).Result
                else:  # pu <= 335
                    active_sim = sim_name1
                    if active_sim not in active_sims_opened:
                        sim_manager.OpenSimulation(active_sim).Result
                        active_sims_opened.add(active_sim)
                    var_manager.SetVariableValue(active_sim, "SP3.OutRatio[S45]", 0.001, "fraction", 90000).Result
                    var_manager.SetVariableValue(active_sim, "SP4.OutRatio[S54]", 0.001, "fraction", 90000).Result

                # Set dynamic demand parameter
                var_manager.SetVariableValue(active_sim, "Var104", pu, "kg/h", 90000).Result

                # Fetch parameters from the active workspace
                AnnualOPEX = var_manager.GetVariableValue(active_sim, "EconSummary1.TotalOperatingCost", "¤/yr", 90000).Result
                AnnualLabor = var_manager.GetVariableValue(active_sim, "EconSummary1.AnnualLaborCost", "¤", 90000).Result
                AnnualMaintenance = var_manager.GetVariableValue(active_sim, "MaintenanceCost", "¤", 90000).Result
                AnnualProduct = var_manager.GetVariableValue(active_sim, "IBU_crystals.W", "kg/h", 90000).Result
                ss = sim_manager.GetSimulationStatus(active_sim).Result

                if not ss[2]:
                    print(f"WARNING: Simulation step failed to converge for year {year}")
                    simulation_successful = False
                    continue

                annual_total_opex = float(AnnualOPEX + AnnualLabor + AnnualMaintenance)
                annual_total_product = float(AnnualProduct * 24 * 330)

                # Add discounted values to totals
                total_discounted_opex += annual_total_opex / (1 + r) ** t
                total_discounted_product += annual_total_product / (1 + r) ** t

            except Exception as e:
                print(f"Error in simulation step for year {year}: {str(e)}")
                simulation_successful = False
                continue

        # --- MANDATORY SNAPSHOT RECOVERY AT END OF TRAJECTORY LOOP ---
        for opened_sim in active_sims_opened:
            try:
                snap_manager.RevertSnapshot(opened_sim, snapshot_name, 180000).Result
            except Exception as e:
                print(f"Warning: Failed memory reset for '{opened_sim}': {str(e)}")

        # Calculate LCOP using accumulated years metrics
        if total_discounted_product > 0:
            LCOP = ((TCI + total_discounted_opex) / total_discounted_product) * 1000
        else:
            LCOP = float("nan")

        return LCOP, simulation_successful, demand_projection[self.end_year]


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = script_dir

    num_samples = 100  
    simulator = MonteCarloSimulator()

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

    successful_lcop = results_df.loc[results_df["Simulation_Status"] == True, "LCOP"]

    if len(successful_lcop) > 0:
        mean_lcop = successful_lcop.mean()
        std_lcop = successful_lcop.std()

        print("\n" + "=" * 40)
        print("MONTE CARLO UNCERTAINTY ANALYSIS (TIME-DEPENDENT)")
        print("=" * 40)
        print(f"Successful Simulations: {len(successful_lcop)}/{num_samples}")
        print(f"LCOP Mean: {mean_lcop:.2f} ¤/t")
        print(f"LCOP Standard Deviation: {std_lcop:.2f} ¤/t")
        print("=" * 40)

        results_df["LCOP_Mean"] = mean_lcop
        results_df["LCOP_StdDev"] = std_lcop
    else:
        print("\nWarning: No pristine converged simulations to analyze!")

    # Save results ledger to CSV
    results_file = os.path.join(results_dir, "Modular_MC_results_TimeDependent.csv")
    results_df.to_csv(results_file, index=False)

    years = list(range(simulator.base_year, simulator.end_year + 1))  
    demand_proj_df = pd.DataFrame(index=range(num_samples), columns=years)

    for i, projection in enumerate(simulator.demand_projections):
        for year in years:
            demand_proj_df.loc[i, year] = projection[year]

    demand_proj_file = os.path.join(results_dir, "Demand_Projections_Modular_TimeDependent_2024_2040.csv")
    demand_proj_df.to_csv(demand_proj_file, index=True)

    # --- PLOT ANALYSIS PANEL ---
    plt.figure(figsize=(18, 12))

    # 1. CAGR and Demand 2040
    plt.subplot(2, 3, 1)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(results_df["Sample_Number"], results_df["CAGR_Mean"] * 100, color="green", linewidth=1.5)
    ax2.plot(results_df["Sample_Number"], results_df["Demand_2040"], color="orange", linewidth=1.5)
    ax1.set_ylabel("CAGR Mean (%)", fontweight="bold", fontsize=10, color="green")
    ax2.set_ylabel("Demand 2040 (kg/h)", fontweight="bold", fontsize=10, color="orange")
    ax1.set_xlabel("Monte Carlo Sample Number", fontweight="bold", fontsize=10)
    plt.title("CAGR and Projected Demand for 2040", fontweight="bold", fontsize=12)

    # 2. LCOP Scatter Profiles
    plt.subplot(2, 3, 2)
    successful_mask = results_df["Simulation_Status"] == True
    failed_mask = ~successful_mask
    plt.plot(results_df.loc[successful_mask, "Sample_Number"], results_df.loc[successful_mask, "LCOP"], marker="o", linestyle="-", color="b", markersize=3, label="Successful")
    plt.plot(results_df.loc[failed_mask, "Sample_Number"], results_df.loc[failed_mask, "LCOP"], marker="x", linestyle="None", color="r", markersize=5, label="Failed")
    plt.ylabel("LCOP (¤/t)", fontweight="bold", fontsize=10)
    plt.xlabel("Monte Carlo Sample Number", fontweight="bold", fontsize=10)
    plt.legend()
    plt.title("LCOP Results", fontweight="bold", fontsize=12)

    # 3. LCOP Distribution Histogram
    plt.subplot(2, 3, 3)
    if len(successful_lcop) > 0:
        plt.hist(successful_lcop, bins="fd", color="purple", alpha=0.7, edgecolor="black", density=True)
        plt.axvline(mean_lcop, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_lcop:.2f}")
        plt.ylabel("Probability Density", fontweight="bold", fontsize=10)
        plt.xlabel("LCOP (¤/t)", fontweight="bold", fontsize=10)
        plt.title("LCOP Uncertainty Distribution", fontweight="bold", fontsize=12)
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)

    # 4. Trajectories over time
    plt.subplot(2, 1, 2)
    sample_indices = np.linspace(0, num_samples - 1, min(20, num_samples), dtype=int)
    for idx in sample_indices:
        projection = simulator.demand_projections[idx]
        demands = [projection[year] for year in years]
        plt.plot(years, demands, alpha=0.3, color="blue", linewidth=0.8)

    mean_demands = [np.mean([proj[year] for proj in simulator.demand_projections]) for year in years]
    plt.plot(years, mean_demands, color="red", linewidth=3, label="Mean Trajectory")
    plt.ylabel("Demand (kg/h)", fontweight="bold", fontsize=10)
    plt.xlabel("Year", fontweight="bold", fontsize=10)
    plt.title("Demand Projections (2024-2040)", fontweight="bold", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = os.path.join(results_dir, "Modular_MC_CAGR_results_TimeDependent.png")
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.show()