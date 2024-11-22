import matplotlib.pyplot as plt
import numpy as np

# Set the style
plt.style.use('seaborn')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Create figure and axis with a larger size
fig, ax = plt.subplots(figsize=(12, 8))

# Sample data structure (you'll need to replace this with your actual data)
models = ['llama3.1', 'llama3.2', 'mistral', 'nemetron', 'phi3.5', 'smollm']
n_values = np.arange(1, 15, 1)

# Color palette - modern, distinguishable colors
colors = ['#2E86AB', '#A23B72', '#1B998B', '#F46036', '#949AA6', '#4C3549']

# Marker styles
markers = ['o', 's', '^', 'D', 'v', 'p']

# Plot for each model
for i, model in enumerate(models):
    # Replace these with your actual data arrays
    eq_data = np.random.rand(len(n_values)) * 0.7 + 0.2  # Dummy data
    opt_data = eq_data + 0.1
    
    # Plot with enhanced styling
    ax.plot(n_values, eq_data, 
            color=colors[i], 
            marker=markers[i],
            markersize=8,
            label=f'{model} (EQ)',
            linestyle='--',
            linewidth=2,
            alpha=0.8)
    
    ax.plot(n_values, opt_data,
            color=colors[i],
            marker=markers[i],
            markersize=8,
            label=f'{model} (OPT)',
            linewidth=2,
            alpha=0.8)

# Customize the grid
ax.grid(True, linestyle='--', alpha=0.7)

# Customize the axis
ax.set_xlabel('n', fontsize=12, fontweight='bold')
ax.set_ylabel('Utility', fontsize=12, fontweight='bold')
ax.set_title('Optimal and Nash Equilibrium Utility by Model', 
             fontsize=14, 
             fontweight='bold', 
             pad=20)

# Enhance the legend
ax.legend(bbox_to_anchor=(1.05, 1),
          loc='upper left',
          borderaxespad=0.,
          frameon=True,
          fontsize=10,
          ncol=1)

# Set axis limits with some padding
ax.set_xlim(0.5, 14.5)
ax.set_ylim(0.15, 0.95)

# Add minor ticks
ax.minorticks_on()

# Tight layout to prevent label cutoff
plt.tight_layout()

# Save the figure with high DPI
# plt.savefig('enhanced_utility_plot.png', 
            # dpi=300, 
            # bbox_inches='tight')

plt.show()
