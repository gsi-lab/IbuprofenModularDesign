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

        # Initial demand for 2024 (kg/h) - using 7.5% of world demand scenario
        self.initial_demand_2024 = 240

        # Years for projection
        self.base_year = 2024  # Base year with known demand
        self.start_year = 2025  # First projection year
        self.end_year = 2045

    def calculate_demand_projection(self, cagr_sample: float) -> dict:
        """Calculate yearly demand from 2024 to 2045 using D_t = D_{t-1}(1 + CAGR_t)"""
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

        # Save the generated CAGR samples for reproducibility (Modular)
        try:
            cagr_samples_df = pd.DataFrame({"CAGR": self.cagr_samples})
            cagr_samples_file = os.path.join(
                results_dir, "CAGR_Samples_Modular_Fixed.csv"
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
        """AVEVA Process Simulation interface - runs simulation for 20 years with varying demand"""
        # Connect to AVEVA Process Simulation
        sc = simcentralconnect.connect().Result
        var_manager = sc.GetService("IVariableManager")
        sim_manager = sc.GetService("ISimulationManager")
        snapshot_manager = sc.GetService("ISnapshotManager")

        # Setup simulation
        sim_name1 = "IbuprofenProcessSimulationModular"
        snapshot_name = "Pro 1"  # Add snapshot name
        TCI = 12161600  # Capital Investment in ¤

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
                # Determine simulation settings based on demand level
                if pu >= 380 and pu < 420:
                    sim_manager.OpenSimulation(sim_name1).Result
                    var_manager.SetVariableValue(
                        sim_name1, "SP3.OutRatio[S45]", 0.001, "fraction", 90000
                    ).Result
                    var_manager.SetVariableValue(
                        sim_name1, "SP4.OutRatio[S54]", 0.001, "fraction", 90000
                    ).Result
                elif pu >= 420:
                    sim_manager.OpenSimulation(sim_name1).Result
                    var_manager.SetVariableValue(
                        sim_name1, "SP3.OutRatio[S45]", 0.001, "fraction", 90000
                    ).Result
                    var_manager.SetVariableValue(
                        sim_name1, "SP4.OutRatio[S54]", 0.999, "fraction", 90000
                    ).Result
                elif pu < 380:
                    sim_manager.OpenSimulation(sim_name1).Result
                    var_manager.SetVariableValue(
                        sim_name1, "SP3.OutRatio[S45]", 1, "fraction", 90000
                    ).Result
                    var_manager.SetVariableValue(
                        sim_name1, "SP4.OutRatio[S54]", 0.001, "fraction", 90000
                    ).Result

                # Set demand for this year
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
                    simulation_successful = False
                    # User requested not to revert to snapshot; continue without reverting
                    print(
                        f"Simulation unsuccessful for year {year} (no snapshot revert)."
                    )
                    continue

                # Calculate discounted values for this year
                annual_total_opex = float(AnnualOPEX + AnnualLabor + AnnualMaintenance)
                annual_total_product = float(AnnualProduct * 24 * 330)  # kg/year

                # Add discounted values to totals
                total_discounted_opex += annual_total_opex / (1 + r) ** t
                total_discounted_product += annual_total_product / (1 + r) ** t

            except Exception as e:
                print(
                    f"Error in simulation for year {year}: {str(e)} (continuing without snapshot revert)"
                )
                simulation_successful = False
                # Do not revert snapshot per user preference; continue to next year
                continue

        # Calculate LCOP using 20-year totals
        if total_discounted_product > 0:
            LCOP = ((TCI + total_discounted_opex) / total_discounted_product) * 1000
        else:
            LCOP = float("inf")  # Handle division by zero

        return (
            LCOP,
            simulation_successful,
            demand_projection[self.end_year],
        )  # Return 2045 demand for reference


if __name__ == "__main__":
    # Save results in the same directory as this script (portable across machines)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = script_dir

    num_samples = 100  # Define the number of Monte Carlo samples
    simulator = (
        MonteCarloSimulator()
    )  # Create an instance of the MonteCarloSimulator class

    # Run simulations and get results
    results = simulator.run_monte_carlo_simulation(
        num_samples, results_dir
    )  # Call the method on the instance

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

    if len(successful_lcop) > 0:
        mean_lcop = successful_lcop.mean()
        std_lcop = successful_lcop.std()

        print("\n" + "=" * 40)
        print("MONTE CARLO UNCERTAINTY ANALYSIS")
        print("=" * 40)
        print(f"Successful Simulations: {len(successful_lcop)}/{num_samples}")
        print(f"LCOP Mean: {mean_lcop:.2f} ¤/t")
        print(f"LCOP Standard Deviation: {std_lcop:.2f} ¤/t")
        print("=" * 40)

        # Add statistics to the results DataFrame
        results_df["LCOP_Mean"] = mean_lcop
        results_df["LCOP_StdDev"] = std_lcop
    else:
        print("\nWarning: No successful simulations to analyze!")

    # Save results to CSV
    results_file = os.path.join(results_dir, "Modular_MC_results_Fixed.csv")
    results_df.to_csv(results_file, index=False)

    # Create demand projection DataFrame for detailed analysis
    years = list(
        range(simulator.base_year, simulator.end_year + 1)
    )  # Include 2024-2045
    demand_proj_df = pd.DataFrame(index=range(num_samples), columns=years)

    for i, projection in enumerate(simulator.demand_projections):
        for year in years:
            demand_proj_df.loc[i, year] = projection[year]

    # Save demand projections to CSV (Modular-specific name to avoid overwriting Nonmodular)
    demand_proj_file = os.path.join(
        results_dir, "Demand_Projections_Modular_Fixed_2024_2045.csv"
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
    plt.title("CAGR and Projected Demand for 2045", fontweight="bold", fontsize=12)
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
    plt.title("LCOP Results", fontweight="bold", fontsize=12)

    # Third subplot: LCOP Distribution (Histogram)
    plt.subplot(2, 3, 3)
    if len(successful_lcop) > 0:
        # Use Freedman-Diaconis rule for LCOP histogram as well to get data-driven bins
        plt.hist(
            successful_lcop,
            bins="fd",
            color="purple",
            alpha=0.7,
            edgecolor="black",
            density=True,
        )

        # Add vertical lines for mean and std dev
        mean_lcop = successful_lcop.mean()
        std_lcop = successful_lcop.std()
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
        plt.title("LCOP Uncertainty Distribution", fontweight="bold", fontsize=12)
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

    plt.ylabel("Demand (kg/h)", fontweight="bold", fontsize=10)
    plt.xlabel("Year", fontweight="bold", fontsize=10)
    plt.title("Demand Projections 2024-2045", fontweight="bold", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    plot_file = os.path.join(results_dir, "Modular_MC_CAGR_results_Fixed.png")
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.show()

    # Results are already saved directly in the Paper2codes directory (results_dir)
