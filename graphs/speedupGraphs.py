import numpy as np
import matplotlib.pyplot as plt

def amdahl_speedup(alpha, N):
    return 1 / ((1 - alpha) + (alpha / N))

N_values = np.arange(1, 33, 1)
alpha_values = [0.25, 0.50, 0.75, 0.95]


fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for alpha in alpha_values:
    speedup = [amdahl_speedup(alpha, N) for N in N_values]
    axes[0].plot(N_values, speedup, label=f'Alpha = {alpha}')

axes[0].set_title("Amdahl's Law: Speedup vs Number of Processors")
axes[0].set_xlabel("Number of Processors (N)")
axes[0].set_ylabel("Speedup")
axes[0].grid(True)
axes[0].legend()

for alpha in alpha_values:
    speedup = [amdahl_speedup(alpha, N) for N in N_values]
    efficiency = [s / N for s, N in zip(speedup, N_values)]
    axes[1].plot(N_values, efficiency, label=f'Alpha = {alpha}')

axes[1].set_title("Amdahl's Law: Efficiency vs Number of Processors")
axes[1].set_xlabel("Number of Processors (N)")
axes[1].set_ylabel("Efficiency")
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.show()


def gustafson_speedup(alpha, N):
    return N - alpha * (N - 1)

N_values = np.arange(1, 33, 1)
alpha_values = [0.25, 0.50, 0.75, 0.95]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for alpha in alpha_values:
    speedup = [gustafson_speedup(alpha, N) for N in N_values]
    axes[0].plot(N_values, speedup, label=f'Alpha = {alpha}')

axes[0].set_title("Gustafson's Law: Speedup vs Number of Processors")
axes[0].set_xlabel("Number of Processors (N)")
axes[0].set_ylabel("Speedup")
axes[0].grid(True)
axes[0].legend()

for alpha in alpha_values:
    speedup = [gustafson_speedup(alpha, N) for N in N_values]
    efficiency = [s / N for s, N in zip(speedup, N_values)]
    axes[1].plot(N_values, efficiency, label=f'Alpha = {alpha}')

axes[1].set_title("Gustafson's Law: Efficiency vs Number of Processors")
axes[1].set_xlabel("Number of Processors (N)")
axes[1].set_ylabel("Efficiency")
axes[1].grid(True)
axes[1].legend()

def house1_time(pizzas):
    return pizzas * 10

def house2_time(pizzas):
    return (pizzas // 5) * 10 + (10 if pizzas % 5 != 0 else 0)

def house3_time(pizzas):
    return 10 + 2 * (pizzas - 1)


pizzas = np.arange(1, 33)

house1_times = [house1_time(p) for p in pizzas]  # Sequential
house2_times = [house2_time(p) for p in pizzas]  # Parallel
house3_times = [house3_time(p) for p in pizzas]  # Pipelined


fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns

# Plot 1: Comparing all three methods
axes[0].plot(pizzas, house1_times, label="Sequential (1 Oven)", color="blue", marker='o')
axes[0].plot(pizzas, house2_times, label="Parallel (5 Ovens)", color="green", marker='s')
axes[0].plot(pizzas, house3_times, label="Pipelined (Conveyor Belt)", color="red", marker='^')
axes[0].set_title("Sequential, Parallel, and Pipelined")
axes[0].set_xlabel("Number of Pizzas")
axes[0].set_ylabel("Baking Time (minutes)")
axes[0].grid(True)
axes[0].set_xticks(np.arange(1, 33, 2))
axes[0].set_yticks(np.arange(1, 300, 25))
axes[0].legend()

# Plot 2: Comparing Parallel and Pipelined only
axes[1].plot(pizzas, house2_times, label="Parallel (5 Ovens)", color="green", marker='s')
axes[1].plot(pizzas, house3_times, label="Pipelined (Conveyor Belt)", color="red", marker='^')
axes[1].set_title("Parallel vs Pipelined")
axes[1].set_xlabel("Number of Pizzas")
axes[1].set_ylabel("Baking Time (minutes)")
axes[1].grid(True)
axes[1].set_xticks(np.arange(1, 33, 1))
axes[1].set_yticks(np.arange(1, 80, 5))
axes[1].legend()

plt.tight_layout()
plt.show()
