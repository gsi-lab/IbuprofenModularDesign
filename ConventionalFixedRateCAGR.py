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
        self.end_year = 2040   # 15-year horizon (2025-2040)

    def calculate_demand_projection(self, cagr_sample: float) -> dict:
        years = range(self.start_year, self.end_year + 1)
        demand_projection = {}
        demand_projection[self.base_year] = self.initial_demand_2024

        demand_projection[self.start_year] = self.initial_demand_2024 * (1 + cagr_sample)

        for year in years[1:]:  
            previous_year = year - 1
            demand_projection[year] = demand_projection[previous_year] * (1 + cagr_sample)

        return demand_projection

    def run_monte_carlo_simulation(self, num_samples: int, results_dir: str):
        np.random.seed(0)
        lhs_samples = lhs(1, samples=num_samples, criterion="maximin")
        self.cagr_samples = norm.ppf(lhs_samples[:, 0], loc=self.cagr_mean, scale=self.cagr_std)
        self.cagr_samples.sort()

        print("\nCAGR Values Statistics:")
        print(f"Minimum: {min(self.cagr_samples) * 100:.3f}%")
        print(f"Maximum: {max(self.cagr_samples) * 100:.3f}%")

        try:
            cagr_samples_df = pd.DataFrame({"CAGR": self.cagr_samples})
            cagr_samples_file = os.path.join(results_dir, "CAGR_Samples_Conventional_Fixed.csv")
            cagr_samples_df.to_csv(cagr_samples_file, index=False)
        except Exception:
            pass

        self.demand_projections = []
        self.final_demands = []  

        for i, cagr_sample in enumerate(self.cagr_samples):
            demand_projection = self.calculate_demand_projection(cagr_sample)
            self.demand_projections.append(demand_projection)
            self.final_demands.append(demand_projection[self.end_year])

        self.final_demands = np.array(self.final_demands)

        results = []
        for i in range(num_samples):
            try:
                demand_projection = self.demand_projections[i]
                obj, simulationstatus, demand_2040 = self.MCsimulator(
                    self.p["x0"], demand_projection
                )
                results.append((obj, simulationstatus, demand_2040, self.cagr_samples[i]))
                print(f"Completed sample {i + 1}/{num_samples} - LCOP: {obj:.2f} ¤/t")
            except Exception as e:
                print(f"Error occurred at point {i + 1}: {str(e)}")
                results.append((None, False, self.final_demands[i], self.cagr_samples[i]))
                continue

        return results

    def MCsimulator(self, x: np.ndarray, demand_projection: dict) -> float:
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

        # --- FIX 1: Open the simulation ONCE at the start of the trajectory ---
        try:
            sim_manager.OpenSimulation(sim_name1).Result
        except Exception as e:
            print(f"CRITICAL ERROR: Simulation '{sim_name1}' could not be opened: {str(e)}")
            return float("nan"), False, demand_projection[self.end_year]

        # --- STEP 2: RUN CHRONOLOGICAL TIMELINE STEPS ---
        for year_idx, year in enumerate(years):
            pu = demand_projection[year]
            t = year_idx + 1  

            try:
                # Set demand for this year
                var_manager.SetVariableValue(sim_name1, "Var104", pu, "kg/h", 90000).Result

                # Get annual results
                AnnualOPEX = var_manager.GetVariableValue(sim_name1, "EconSummary1.TotalOperatingCost", "¤/yr", 90000).Result
                AnnualLabor = var_manager.GetVariableValue(sim_name1, "EconSummary1.AnnualLaborCost", "¤", 90000).Result
                AnnualMaintenance = var_manager.GetVariableValue(sim_name1, "MaintenanceCost", "¤", 90000).Result
                AnnualProduct = var_manager.GetVariableValue(sim_name1, "IBU_crystals.W", "kg/h", 90000).Result
                ss = sim_manager.GetSimulationStatus(sim_name1).Result

                if not ss[2]:
                    print(f"WARNING: Simulation non-convergence for year {year}")
                    simulation_successful = False

                try:
                    annual_total_opex = float(AnnualOPEX + AnnualLabor + AnnualMaintenance)
                    annual_total_product = float(AnnualProduct * 24 * 330)  
                except (TypeError, ValueError):
                    annual_total_opex = 0
                    annual_total_product = 0

                discount_factor = (1 + r) ** t
                total_discounted_opex += annual_total_opex / discount_factor
                total_discounted_product += annual_total_product / discount_factor

            except Exception as year_error:
                print(f"  -> Year {year} threw an error: {str(year_error)}")
                simulation_successful = False
                continue  # Original logic: march forward to next years even if this one failed

        # --- FIX 2: REVERT SNAPSHOT AFTER THE 15-YEAR TRAJECTORY IS DONE ---
        try:
            snap_manager.RevertSnapshot(sim_name1, snapshot_name, 180000).Result
        except Exception as e:
            print(f"WARNING: Snapshot revert failed: {str(e)}")

        # Calculate LCOP using gathered totals
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

    parser = argparse.ArgumentParser(description="Run Monte Carlo simulations.")
    parser.add_argument("--samples", default="100")
    parser.add_argument("--convergence", action="store_true")
    parser.add_argument("--conv-step", type=int, default=5)
    parser.add_argument("--conv-repeats", type=int, default=200)
    parser.add_argument("--conv-out", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=9000.0)
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--conv-maxsum", action="store_true")

    args = parser.parse_args()

    if args.auto:
        sample_list = [20, 40, 60, 80, 100, 120]
    else:
        sample_list = [int(s) for s in args.samples.split(",") if s.strip()]

    simulator = MonteCarloSimulator()

    for num_samples in sample_list:
        print(f"\nRunning Monte Carlo with num_samples={num_samples}")
        results = simulator.run_monte_carlo_simulation(num_samples, results_dir)

        results_df = pd.DataFrame({
            "Sample_Number": range(1, num_samples + 1),
            "CAGR": simulator.cagr_samples,
            "Demand_2040": simulator.final_demands,
            "LCOP": [r[0] for r in results],
            "Simulation_Status": [r[1] for r in results],
        })

        out_results = os.path.join(results_dir, f"Conventional_MC_results_Fixed_N{num_samples}.csv")
        results_df.to_csv(out_results, index=False)

    successful_lcop = results_df.loc[results_df["Simulation_Status"], "LCOP"]
    valid_lcop = successful_lcop[np.isfinite(successful_lcop)]

    if len(valid_lcop) > 0:
        print("\n" + "=" * 40)
        print("MONTE CARLO UNCERTAINTY ANALYSIS")
        print("=" * 40)
        print(f"LCOP Mean: {valid_lcop.mean():.2f} ¤/t")
        print(f"LCOP Standard Deviation: {valid_lcop.std():.2f} ¤/t")
        print("=" * 40)

    results_file = os.path.join(results_dir, "Conventional_MC_results_Fixed.csv")
    results_df.to_csv(results_file, index=False)

    years = list(range(simulator.base_year, simulator.end_year + 1))
    demand_proj_df = pd.DataFrame(index=range(num_samples), columns=years)
    for i, projection in enumerate(simulator.demand_projections):
        for year in years:
            demand_proj_df.loc[i, year] = projection[year]

    demand_proj_file = os.path.join(results_dir, "Demand_Projections_Conventional_Fixed_2024_2040.csv")
    demand_proj_df.to_csv(demand_proj_file, index=True)

    # --- Plotting Phase ---
    plt.figure(figsize=(18, 12))
    plt.subplot(2, 3, 1)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(results_df["Sample_Number"], results_df["CAGR"] * 100, color="green", label="CAGR (%)")
    ax2.plot(results_df["Sample_Number"], results_df["Demand_2040"], color="orange", label="Demand 2040 (kg/h)")
    plt.title("CAGR and Projected Demand for 2040")

    plt.subplot(2, 3, 2)
    plt.plot(results_df.loc[results_df["Simulation_Status"], "Sample_Number"], results_df.loc[results_df["Simulation_Status"], "LCOP"], marker="o", color="b")
    plt.title("LCOP Results")

    plt.subplot(2, 3, 3)
    if len(valid_lcop) > 0:
        plt.hist(valid_lcop, bins=10, color="purple", edgecolor="black", density=True)
        plt.title("LCOP Uncertainty Distribution")

    plt.subplot(2, 1, 2)
    for idx in np.linspace(0, num_samples - 1, min(20, num_samples), dtype=int):
        plt.plot(years, [simulator.demand_projections[idx][y] for year in years for y in [year]], alpha=0.3, color="blue")
    plt.title("Demand Projections 2024-2040")
    plt.tight_layout()

    png_file = os.path.join(results_dir, "Conventional_MC_CAGR_Results_Fixed.png")
    plt.savefig(png_file, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {png_file}")
    plt.show()