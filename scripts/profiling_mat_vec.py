import pandas as pd
import matplotlib.pyplot as plt

# === Read CSV ===
filename = "data/foam.csv"  # Assicurati che sia nel path corretto
df = pd.read_csv(filename)

# === Group for sparsity ===
plt.figure(figsize=(10, 6))
for sparsity, group in df.groupby("sparsity"):
    group_sorted = group.sort_values("n")
    plt.plot(group_sorted["n"], group_sorted["speedup_time"], marker='o', label=f"Sparsity {sparsity:.2f}")

# Ideal speedup line
plt.axhline(y=2.0, color='red', linestyle='--', linewidth=2)
plt.text(df["n"].max(), 2.03, "Theoretical max (2x)", fontsize=9, color='red', ha='right')

plt.xlabel("Matrix size (n)")
plt.ylabel("Speedup (scalar vs vectorized)")
plt.title("Speedup vs Matrix Size at Varying Sparsity Levels")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
