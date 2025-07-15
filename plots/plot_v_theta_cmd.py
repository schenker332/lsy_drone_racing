#!/usr/bin/env python3
"""
Plot v_theta and v_theta_cmd over time from drone racing logs.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_v_theta_comparison(log_dir):
    """
    Plot v_theta (actual) and dv_theta_cmd (command) over time.
    
    Args:
        log_dir: Path to the log directory containing state_log.csv and control_log.csv
    """
    
    # File paths
    state_file = os.path.join(log_dir, 'state_log.csv')
    control_file = os.path.join(log_dir, 'control_log.csv')
    
    # Check if files exist
    if not os.path.exists(state_file):
        raise FileNotFoundError(f"State log file not found: {state_file}")
    if not os.path.exists(control_file):
        raise FileNotFoundError(f"Control log file not found: {control_file}")
    
    # Load data
    print("Loading data...")
    state_df = pd.read_csv(state_file)
    control_df = pd.read_csv(control_file)
    
    print(f"State data: {len(state_df)} rows, columns: {list(state_df.columns)}")
    print(f"Control data: {len(control_df)} rows, columns: {list(control_df.columns)}")
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: v_theta (actual) from state log
    ax1.plot(state_df['time'], state_df['v_theta'], 
             linewidth=2, color='blue', alpha=0.8)
    ax1.set_ylabel('v_theta [rad/s]', fontsize=12)
    ax1.set_title('v_theta (Actual Angular Velocity)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics for v_theta
    v_theta_mean = state_df['v_theta'].mean()
    v_theta_std = state_df['v_theta'].std()
    v_theta_min = state_df['v_theta'].min()
    v_theta_max = state_df['v_theta'].max()
    
    ax1.text(0.02, 0.98, 
             f'μ={v_theta_mean:.4f}, σ={v_theta_std:.4f}\n'
             f'min={v_theta_min:.4f}, max={v_theta_max:.4f}',
             transform=ax1.transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             fontsize=10)
    
    # Plot 2: dv_theta_cmd (command) from control log
    ax2.plot(control_df['time'], control_df['dv_theta_cmd'], 
             linewidth=2, color='red', alpha=0.8)
    ax2.set_xlabel('Time [s]', fontsize=12)
    ax2.set_ylabel('dv_theta_cmd [rad/s²]', fontsize=12)
    ax2.set_title('dv_theta_cmd (Angular Acceleration Command)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics for dv_theta_cmd
    dv_theta_cmd_mean = control_df['dv_theta_cmd'].mean()
    dv_theta_cmd_std = control_df['dv_theta_cmd'].std()
    dv_theta_cmd_min = control_df['dv_theta_cmd'].min()
    dv_theta_cmd_max = control_df['dv_theta_cmd'].max()
    
    ax2.text(0.02, 0.98, 
             f'μ={dv_theta_cmd_mean:.4f}, σ={dv_theta_cmd_std:.4f}\n'
             f'min={dv_theta_cmd_min:.4f}, max={dv_theta_cmd_max:.4f}',
             transform=ax2.transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(log_dir, 'v_theta_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    plt.show()

if __name__ == "__main__":
    # Log directory
    log_dir = "/Users/niclas/VS_Code/ADR/lsy_drone_racing/logs/run_20250712_121750"

    try:
        plot_v_theta_comparison(log_dir)
    except Exception as e:
        print(f"Error: {e}")
