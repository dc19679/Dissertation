import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Generate random data for demonstration purposes
num_time_steps = 100
LacI_values = np.random.randint(0, 1000, size=num_time_steps)
TetR_values = np.random.randint(0, 1000, size=num_time_steps)

# Set up the plot
fig, ax = plt.subplots()

# Choose a colormap for the segments
colormap = cm.get_cmap('viridis', num_time_steps)

# Plot each line segment with a different color
for i in range(1, num_time_steps):
    ax.plot(
        LacI_values[i-1:i+1],
        TetR_values[i-1:i+1],
        color=colormap(i),
        lw=2,
        marker='o'
    )

# Add a colorbar
sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=num_time_steps - 1))
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.set_label("Time step")

# Set axis labels
ax.set_xlabel("LacI")
ax.set_ylabel("TetR")

# Set title
ax.set_title("Cell Path in State Space")

# Display the plot
plt.show()
