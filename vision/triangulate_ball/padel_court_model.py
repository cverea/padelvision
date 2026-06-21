import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Function to generate the keypoints of a padel court
def generate_padel_court_coordinates():
    # Court dimensions
    length = 20  # meters
    width = 10   # meters
    height_wall = 3  # height of the solid/glass wall
    height_fence = 4  # total height including fence
    distance_from_back_wall = 4  # Distance from back wall where the vertical lines are placed


    # Coordinates are relative to (0, 0, 0) at one corner of the court
    
    # Court outline (floor level)
    court_outline = [
        [0, 0, 0],         # Bottom-left corner
        [length, 0, 0],    # Bottom-right corner
        [length, width, 0],  # Top-right corner
        [0, width, 0],     # Top-left corner
        [0, 0, 0]          # Closing the loop
    ]
    
    # Service lines (parallel to the net)
    service_lines = [
        [3, 0, 0],         # Service line left half
        [3, width, 0],     # Service line right half
        [length - 3, 0, 0],    # Service line left half (far side)
        [length - 3, width, 0]  # Service line right half (far side)
    ]
    
    # Center service lines (perpendicular to the net)
    center_lines = [
        [3, width / 2, 0],  # Center service line (near)
        [length - 3, width / 2, 0]  # Center service line (far)
    ]
    
    # Net line
    net_line = [
        [length / 2, 0, 0],     # Net line left
        [length / 2, width, 0]  # Net line right
    ]
    
    # Back walls (3 meters high)
    back_walls = [
        [0, 0, 0], [0, 0, height_wall],  # Bottom-left wall
        [0, width, 0], [0, width, height_wall],  # Top-left wall
        
        [length, 0, 0], [length, 0, height_wall],  # Bottom-right wall
        [length, width, 0], [length, width, height_wall]  # Top-right wall
    ]
    
    # Side walls
    side_walls = [
        [0, 0, height_wall], [length / 2, 0, height_wall],  # Side wall along bottom (left half)
        [0, width, height_wall], [length / 2, width, height_wall],  # Side wall along top (left half)
        
        [length / 2, 0, height_wall], [length, 0, height_wall],  # Side wall along bottom (right half)
        [length / 2, width, height_wall], [length, width, height_wall]  # Side wall along top (right half)
    ]
    
    # Fencing (extends to 4 meters high)
    fencing = [
        [0, 0, height_wall], [0, 0, height_fence],
        [0, width, height_wall], [0, width, height_fence],
        
        [length, 0, height_wall], [length, 0, height_fence],
        [length, width, height_wall], [length, width, height_fence],
        
        [0, 0, height_fence], [length, 0, height_fence],
        [0, width, height_fence], [length, width, height_fence],
        
        # Add fencing along the side walls, near the back walls (4 meters length, 4 meters high)
        [0, 0, height_wall], [0, 0, height_fence],  # Bottom-left side fence section
        [0, width, height_wall], [0, width, height_fence],  # Top-left side fence section

        [length, 0, height_wall], [length, 0, height_fence],  # Bottom-right side fence section
        [length, width, height_wall], [length, width, height_fence],  # Top-right side fence section

        # Side fence near the back walls (4 meters length, 4 meters high)
        [0, 0, 0], [distance_from_back_wall, 0, height_fence],  # Bottom-left side fence (along the length, near the back wall)
        [0, width, 0], [4, width, height_fence],  # Top-left side fence (along the length, near the back wall)
        
        [length, 0, 0], [length - distance_from_back_wall, 0, height_fence],  # Bottom-right side fence (along the length, near the back wall)
        [length, width, 0], [length - distance_from_back_wall, width, height_fence]  # Top-right side fence (along the length, near the back wall)
    ]
    
    # Combine all points
    all_points = np.array(court_outline + service_lines + center_lines + net_line + back_walls + side_walls + fencing)
    
    return all_points

# Generate the array of 3D coordinates
padel_court_coordinates = generate_padel_court_coordinates()

# Convert the numpy array into a pandas DataFrame for better visualization
df_padel_court_coordinates = pd.DataFrame(padel_court_coordinates, columns=["X", "Y", "Z"])

# Extract the X, Y, Z coordinates
X = df_padel_court_coordinates["X"]
Y = df_padel_court_coordinates["Y"]
Z = df_padel_court_coordinates["Z"]
# Court dimensions
length = 20  # meters
width = 10   # meters
height_wall = 3  # height of the solid/glass wall
height_fence = 4  # total height including fence
distance_from_back_wall = 4  # Distance from back wall where the vertical lines are placed

# Create a 3D plot and add lines connecting the key points for better visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the key points
ax.scatter(X, Y, Z, color='blue')

# Court outline (floor level)
court_outline = np.array([
    [0, 0, 0], [length, 0, 0], [length, width, 0], [0, width, 0], [0, 0, 0]
])
ax.plot(court_outline[:, 0], court_outline[:, 1], court_outline[:, 2], color='black')

# Service lines
ax.plot([3, 3], [0, width], [0, 0], color='green')
ax.plot([length - 3, length - 3], [0, width], [0, 0], color='green')

# Net line
ax.plot([length / 2, length / 2], [0, width], [0, 0], color='red')

# Center service lines (perpendicular)
ax.plot([3, length - 3], [width / 2, width / 2], [0, 0], color='blue')

# Back walls (3 meters high)
# Left back wall
ax.plot([0, 0], [0, 0], [0, height_wall], color='orange')  # Vertical line (bottom-left)
ax.plot([0, 0], [width, width], [0, height_wall], color='orange')  # Vertical line (top-left)
ax.plot([0, 0], [0, width], [height_wall, height_wall], color='orange')  # Horizontal line at the top (left wall)

# Right back wall
ax.plot([length, length], [0, 0], [0, height_wall], color='orange')  # Vertical line (bottom-right)
ax.plot([length, length], [width, width], [0, height_wall], color='orange')  # Vertical line (top-right)
ax.plot([length, length], [0, width], [height_wall, height_wall], color='orange')  # Horizontal line at the top (right wall)

# Side walls (3 meters high)
# Left side wall
ax.plot([0, length / 2], [0, 0], [height_wall, height_wall], color='orange')  # Horizontal line along bottom-left side
ax.plot([0, length / 2], [width, width], [height_wall, height_wall], color='orange')  # Horizontal line along top-left side

# Right side wall
ax.plot([length / 2, length], [0, 0], [height_wall, height_wall], color='orange')  # Horizontal line along bottom-right side
ax.plot([length / 2, length], [width, width], [height_wall, height_wall], color='orange')  # Horizontal line along top-right side

# Fencing lines (top fencing at 4 meters high)
ax.plot([0, 0], [0, width], [height_fence, height_fence], color='grey')  # Left back fence top
ax.plot([length, length], [0, width], [height_fence, height_fence], color='grey')  # Right back fence top

# Fencing lines (side fence at 4 meters high near back walls)
ax.plot([0, distance_from_back_wall], [0, 0], [height_fence, height_fence], color='grey')  # Left bottom fence along length (near back wall)
ax.plot([0, distance_from_back_wall], [width, width], [height_fence, height_fence], color='grey')  # Left top fence along length (near back wall)
ax.plot([length - distance_from_back_wall, length], [0, 0], [height_fence, height_fence], color='grey')  # Right bottom fence along length (near back wall)
ax.plot([length - distance_from_back_wall, length], [width, width], [height_fence, height_fence], color='grey')  # Right top fence along length (near back wall)

# Fencing vertical lines (side fence near back walls)
ax.plot([0, 0], [0, 0], [height_wall, height_fence], color='grey')  # Vertical line left bottom
ax.plot([0, 0], [width, width], [height_wall, height_fence], color='grey')  # Vertical line left top
ax.plot([length, length], [0, 0], [height_wall, height_fence], color='grey')  # Vertical line right bottom
ax.plot([length, length], [width, width], [height_wall, height_fence], color='grey')  # Vertical line right top

# Vertical lines from 4m fencing to 3m side wall on both sides of the court, placed distance_from_back_wall from the back wall
# Left side of the court, closer to net (distance_from_back_wall meters from back wall)
ax.plot([distance_from_back_wall, distance_from_back_wall], [0, 0], [height_fence, height_wall], color='purple')  # Bottom-left vertical line (closer to net)
ax.plot([distance_from_back_wall, distance_from_back_wall], [width, width], [height_fence, height_wall], color='purple')  # Top-left vertical line (closer to net)

# Right side of the court, closer to net (distance_from_back_wall meters from back wall)
ax.plot([length - distance_from_back_wall, length - distance_from_back_wall], [0, 0], [height_fence, height_wall], color='purple')  # Bottom-right vertical line (closer to net)
ax.plot([length - distance_from_back_wall, length - distance_from_back_wall], [width, width], [height_fence, height_wall], color='purple')  # Top-right vertical line (closer to net)

# Add labels to the axes
ax.set_xlabel('X (Length)')
ax.set_ylabel('Y (Width)')
ax.set_zlabel('Z (Height)')

# Set title
ax.set_title("3D Padel Court Keypoints with Vertical Fencing Connections Closer to Net")

# Set equal scaling for all axes
max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0

mid_x = (X.max() + X.min()) * 0.5
mid_y = (Y.max() + Y.min()) * 0.5
mid_z = (Z.max() + Z.min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Show the plot
plt.show()
