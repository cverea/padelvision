import matplotlib.pyplot as plt
import numpy as np  

# Example 3D points for multiple frames
points_3D = np.array([
 [-31391.92415685, -17504.01380118,  49064.93771028],
 [-33385.11508581,-18613.61023394,  52182.44545213],
 [-44288.47038659, -24691.01623532,  69229.49504025],
 [-34588.63230064, -19281.7810773,   54068.02714018],
 [-28511.48067465, -15892.08792295,  44568.49811528],
 [-30488.90091403, -16992.29469943,  47660.85515458],
 [-32969.01108603, -18373.51701299,  51541.39095081],
 [-30567.42592257, -17032.70897007,  47787.96830328],
 [-34708.9574216,  -19339.83395433,  54265.4751453 ],
 [-29357.36715594, -16356.21726256,  45899.33480142]
    # ... add more points for each frame
])

# Plot the 3D trajectory of the ball
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(points_3D[:, 0], points_3D[:, 1], points_3D[:, 2], marker='o')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

plt.show()
