import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def arc_length_parametrization(waypoints, num_samples):
    """
    Parametrize a trajectory by arc length instead of time.
    
    Args:
        waypoints: Array of waypoints (n, 3) with x,y,z coordinates
        num_samples: Number of desired samples for the output
        
    Returns:
        theta_values: Progress parameter (normalized arc length) [0,1]
        x_values: X-coordinates at the corresponding theta values
        y_values: Y-coordinates at the corresponding theta values
        z_values: Z-coordinates at the corresponding theta values
    """

    
    # Create parameters for the original waypoints (evenly distributed)
    t_orig = np.linspace(0, 1, len(waypoints))
    
    # Create cubic splines through the waypoints
    cs_x = CubicSpline(t_orig, waypoints[:, 0])
    cs_y = CubicSpline(t_orig, waypoints[:, 1])
    cs_z = CubicSpline(t_orig, waypoints[:, 2])

    

    
    # Generate dense sampling for arc length calculation
    # More points = more accurate arc length calculation
    num_dense_samples = num_samples
    t_dense = np.linspace(0, 1, num_dense_samples)
    
    # Calculate the points on the spline for this dense sampling
    x_dense = cs_x(t_dense)
    y_dense = cs_y(t_dense)
    z_dense = cs_z(t_dense)
    points_dense = np.column_stack((x_dense, y_dense, z_dense))


    

    



    # Calculate the arc length between consecutive points
    # np.diff returns the difference between adjacent elements
    diffs = np.diff(points_dense, axis=0)  # Differences in x, y, z
    
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))  # Euclidean distances

    # print(segment_lengths)
    # Calculate the cumulative arc length up to each point
    cumulative_length = np.zeros(num_dense_samples)
    cumulative_length[1:] = np.cumsum(segment_lengths)




    # Total length of the trajectory
    total_length = cumulative_length[-1]

    
    # Normalize the cumulative length to [0,1] for the progress parameter theta
    theta_dense = cumulative_length / total_length
    
    # Generate evenly distributed theta values for the output
    theta_values = np.linspace(0, 1, num_samples)

    # Interpolate t-values for the evenly distributed theta values
    # This is the key step: we convert from uniform arc lengths
    # back to the corresponding parameters on the original spline
    t_interp = np.interp(theta_values, theta_dense, t_dense)



    
    # Calculate the final x,y,z coordinates for these t-values
    x_values = cs_x(t_interp)
    y_values = cs_y(t_interp)
    z_values = cs_z(t_interp)



    # Return the results
    return theta_values, x_values, y_values, z_values, x_dense, y_dense, z_dense, t_dense, t_interp

# Test code that runs when this file is executed as a script
if __name__ == "__main__":
    # Create 10 waypoints in 2D (with z=0)
    # These waypoints form a curved path
    waypoints_2d = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.5, 0.0],
        [2.0, 1.5, 0.0],
        [3.0, 1.0, 0.0],
        [4.0, 2.0, 0.0],
        [5.0, 3.5, 0.0],
        [6.0, 3.0, 0.0],
        [7.0, 2.0, 0.0],
        [8.0, 1.0, 0.0],
        [9.0, 0.5, 0.0]
    ])
    
    # Number of output points after arc length parametrization
    num_samples = 50
    
    # Apply arc length parametrization and get all data for plotting
    theta, x_arc, y_arc, z_arc, x_dense, y_dense, z_dense, t_dense, t_interp = arc_length_parametrization(waypoints_2d, num_samples)
    
    # Create a figure with two separate plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # First plot: Original spline with dense sampling points
    ax1.plot(waypoints_2d[:, 0], waypoints_2d[:, 1], 'ro', markersize=8, label='Original Waypoints')
    ax1.plot(x_dense, y_dense, 'b-', linewidth=1.5, label='Cubic Spline Curve')
    ax1.plot(x_dense, y_dense, 'bo', markersize=4, label='Dense Sampling Points')
    ax1.set_title('Original Spline with Uneven Point Distribution', fontsize=14)
    ax1.set_xlabel('X-Coordinate')
    ax1.set_ylabel('Y-Coordinate')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Second plot: Arc length parametrized points
    ax2.plot(waypoints_2d[:, 0], waypoints_2d[:, 1], 'ro', markersize=8, label='Original Waypoints')
    ax2.plot(x_dense, y_dense, 'b-', linewidth=1.5, label='Cubic Spline Curve')
    ax2.plot(x_arc, y_arc, 'go', markersize=6, label='Arc Length Parametrized Points')
    ax2.set_title('Arc Length Parametrized with Even Point Distribution', fontsize=14)
    ax2.set_xlabel('X-Coordinate')
    ax2.set_ylabel('Y-Coordinate')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("two_separate_plots.png")
    plt.show()
    
    # Create a third plot to show segment lengths comparison
    plt.figure(figsize=(10, 6))
    
    # Plot the lengths of the segments between original dense points
    segment_indices_dense = np.arange(len(x_dense)-1)
    diffs_dense = np.diff(np.column_stack((x_dense, y_dense)), axis=0)
    distances_dense = np.sqrt(np.sum(diffs_dense**2, axis=1))
    
    plt.subplot(2, 1, 1)
    plt.bar(segment_indices_dense, distances_dense, width=1.0, color='blue', alpha=0.6)
    plt.title('Segment Lengths from Original Spline Sampling', fontsize=12)
    plt.ylabel('Segment Length', fontsize=10)
    
    # Plot the lengths of the segments between arc length parametrized points
    segment_indices_arc = np.arange(len(x_arc)-1)
    diffs_arc = np.diff(np.column_stack((x_arc, y_arc)), axis=0)
    distances_arc = np.sqrt(np.sum(diffs_arc**2, axis=1))
    
    plt.subplot(2, 1, 2)
    plt.bar(segment_indices_arc, distances_arc, width=0.5, color='green', alpha=0.6)
    plt.title('Segment Lengths from Arc Length Parametrized Points', fontsize=12)
    plt.xlabel('Segment Index', fontsize=10)
    plt.ylabel('Segment Length', fontsize=10)
    
    plt.tight_layout()
    plt.savefig("segment_lengths_comparison.png")
    plt.show()