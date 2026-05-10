# =========================================================
# Advanced Bayesian Inference using MCMC
# =========================================================
# Features Added:
# ✔ Burn-in Period
# ✔ Acceptance Rate
# ✔ Trace Plots
# ✔ Joint Posterior Scatter Plot
# ✔ Dark Theme Visualization
# ✔ Confidence Interval
# ✔ Progress Bar
# ✔ Save Results Automatically
# ✔ Better Proposal Distribution
# ✔ Professional Scientific Graphs
# =========================================================

# Required Libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm

# Dark Theme
plt.style.use("dark_background")

# =========================================================
# Generate Synthetic Gaussian Data
# =========================================================
np.random.seed(42)

true_mean = 5
true_std = 1

# Generate Data
data = np.random.normal(true_mean, true_std, size=100)

# =========================================================
# Prior Parameters
# =========================================================
prior_mean = 0
prior_std = 10

# =========================================================
# Likelihood Function
# =========================================================
def likelihood(data, mean, std):
    return np.prod(norm.pdf(data, loc=mean, scale=std))

# =========================================================
# Prior Function
# =========================================================
def prior(mean, std):
    mean_prior = norm.pdf(mean, loc=prior_mean, scale=prior_std)
    std_prior = norm.pdf(std, loc=1, scale=1)
    return mean_prior * std_prior

# =========================================================
# Posterior Function
# =========================================================
def posterior(data, mean, std):
    return likelihood(data, mean, std) * prior(mean, std)

# =========================================================
# Metropolis-Hastings Algorithm
# =========================================================
def metropolis_hastings(data, initial_mean, initial_std, iterations):

    samples = []

    current_mean = initial_mean
    current_std = initial_std

    accepted = 0
    rejected = 0

    for _ in tqdm(range(iterations)):

        # Better proposal distribution with wider movement
        proposed_mean = np.random.normal(current_mean, 0.5)
        proposed_std = np.abs(np.random.normal(current_std, 0.15))

        # Current and proposed posterior
        current_posterior = posterior(data, current_mean, current_std)
        proposed_posterior = posterior(data, proposed_mean, proposed_std)

        # Acceptance ratio
        acceptance_ratio = proposed_posterior / current_posterior

        # Accept or Reject
        if np.random.rand() < acceptance_ratio:
            current_mean = proposed_mean
            current_std = proposed_std
            accepted += 1
        else:
            rejected += 1

        samples.append((current_mean, current_std))

    acceptance_rate = accepted / iterations

    print("\n============================")
    print("Acceptance Rate:", round(acceptance_rate, 3))
    print("Accepted Samples:", accepted)
    print("Rejected Samples:", rejected)
    print("============================")

    return np.array(samples)

# =========================================================
# Run MCMC
# =========================================================
initial_mean = 0
initial_std = 1
iterations = 5000

samples = metropolis_hastings(
    data,
    initial_mean,
    initial_std,
    iterations
)

# =========================================================
# Burn-in Removal
# =========================================================
burn_in = 1000
samples = samples[burn_in:]

# =========================================================
# Extract Samples
# =========================================================
means = samples[:, 0]
stds = samples[:, 1]

# =========================================================
# Estimated Parameters
# =========================================================
estimated_mean = np.mean(means)
estimated_std = np.mean(stds)

print(f"\nEstimated Mean: {estimated_mean:.3f}")
print(f"Estimated Std Dev: {estimated_std:.3f}")

# =========================================================
# Confidence Intervals
# =========================================================
mean_low = np.percentile(means, 2.5)
mean_high = np.percentile(means, 97.5)

std_low = np.percentile(stds, 2.5)
std_high = np.percentile(stds, 97.5)

print("\n95% Credible Interval for Mean:")
print(f"[{mean_low:.3f}, {mean_high:.3f}]")

print("\n95% Credible Interval for Std Dev:")
print(f"[{std_low:.3f}, {std_high:.3f}]")

# =========================================================
# Posterior Distribution Histograms
# =========================================================
plt.figure(figsize=(16, 6))

# Mean Histogram
plt.subplot(1, 2, 1)
plt.hist(means, bins=60, density=True, alpha=0.7, color='lime')
plt.axvline(true_mean, color='red', linestyle='dashed', linewidth=2)
plt.title('Posterior Distribution of Mean')
plt.xlabel('Mean')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

# Std Histogram
plt.subplot(1, 2, 2)
plt.hist(stds, bins=60, density=True, alpha=0.7, color='cyan')
plt.axvline(true_std, color='red', linestyle='dashed', linewidth=2)
plt.title('Posterior Distribution of Std Dev')
plt.xlabel('Standard Deviation')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("posterior_distribution.png")
plt.show()

# =========================================================
# Trace Plots
# =========================================================
plt.figure(figsize=(16, 6))

# Mean Trace Plot
plt.subplot(1, 2, 1)
plt.plot(means)
plt.title("Trace Plot of Mean")
plt.xlabel("Iteration")
plt.ylabel("Mean")

# Std Trace Plot
plt.subplot(1, 2, 2)
plt.plot(stds)
plt.title("Trace Plot of Std Dev")
plt.xlabel("Iteration")
plt.ylabel("Std Dev")

plt.tight_layout()
plt.savefig("trace_plots.png")
plt.show()

# =========================================================
# Joint Posterior Scatter Plot
# =========================================================
plt.figure(figsize=(8, 6))
plt.scatter(means, stds, alpha=0.5)
plt.title("Joint Posterior Distribution")
plt.xlabel("Mean")
plt.ylabel("Std Dev")
plt.savefig("joint_posterior.png")
plt.show()

# =========================================================
# Heatmap Density Plot
# =========================================================
plt.figure(figsize=(8, 6))
plt.hexbin(means, stds, gridsize=30)
plt.title("Posterior Density Heatmap")
plt.xlabel("Mean")
plt.ylabel("Std Dev")
plt.colorbar(label='Density')
plt.savefig("posterior_heatmap.png")
plt.show()

# =========================================================
# Correlation Between Mean and Std
# =========================================================
correlation = np.corrcoef(means, stds)

print("\nCorrelation Matrix:")
print(correlation)

# =========================================================
# Save Complete Results
# =========================================================
np.savez(
    'bayesian_inference_results.npz',
    means=means,
    stds=stds,
    true_mean=true_mean,
    true_std=true_std,
    estimated_mean=estimated_mean,
    estimated_std=estimated_std
)

print("\nResults Saved Successfully!")
print("Generated Files:")
print("- posterior_distribution.png")
print("- trace_plots.png")
print("- joint_posterior.png")
print("- posterior_heatmap.png")
print("- bayesian_inference_results.npz")

# =========================================================
# End of Program
# =========================================================
print("\nAdvanced Bayesian MCMC Analysis Completed Successfully!")