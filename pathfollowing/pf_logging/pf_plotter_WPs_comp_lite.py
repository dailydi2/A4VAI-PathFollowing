#!/usr/bin/env python3
"""
Copy the latest CSV from /home/lyj801/Documents/A4VAI-SITL/ROS2/logs
to pf_logging/pf_data, and draw all subplots in one figure (summary_plot.png).
"""

import os
import glob
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === Directory paths ===
LOG_DIR  = "/home/lyj801/Documents/A4VAI-SITL/ROS2/logs"
DEST_DIR = "/home/lyj801/Documents/A4VAI-SITL/ROS2/ros2_ws/src/pathfollowing/pathfollowing/pf_logging/pf_data"

os.makedirs(DEST_DIR, exist_ok=True)

# === Find latest CSV ===
csvs = sorted(glob.glob(os.path.join(LOG_DIR, "*.csv")), key=os.path.getmtime)
if not csvs:
    raise FileNotFoundError(f"No CSV found in {LOG_DIR}")
src_csv = csvs[-1]

# === Copy to DEST_DIR ===
dst_csv = os.path.join(DEST_DIR, os.path.basename(src_csv))
shutil.copy2(src_csv, dst_csv)
print(f"[INFO] Copied: {src_csv} -> {dst_csv}")

# === Load CSV ===
df = pd.read_csv(dst_csv)
print(f"[INFO] Loaded columns: {list(df.columns)}")

# === Time vector ===
t = df["time"].to_numpy() if "time" in df.columns else np.arange(len(df))

# === Create a single figure with multiple subplots ===
fig, axs = plt.subplots(1, 3, figsize=(10, 3))
# fig.suptitle("Pathfollowing Log Summary", fontsize=14, fontweight="bold")

# ---- (1) XY trajectory ----
# ax = axs[0]
# if {"pos_x", "pos_y"}.issubset(df.columns):
#     ax.plot(df["pos_x"], df["pos_y"], color='tab:blue')
# ax.set_title("XY Trajectory")
# ax.set_xlabel("pos_x [m]")
# ax.set_ylabel("pos_y [m]")
# ax.axis("equal")
# ax.grid(True)

# 3d plot
fig, axs = plt.subplots(1, 3, figsize=(10, 3))
# ?: ? ?? 3D? ??
fig.delaxes(axs[0])                              # 2D ? ??
ax1 = fig.add_subplot(1, 3, 1, projection='3d')  # ?? ??(1??)? 3D ??
ax2 = axs[1]
ax3 = axs[2]
if {"pos_x", "pos_y", "pos_z"}.issubset(df.columns):
    ax1.plot(df["pos_x"], df["pos_y"], -df["pos_z"], color='tab:blue')
ax1.set_title("3D Trajectory")
ax1.set_xlabel("pos_x [m]")
ax1.set_ylabel("pos_y [m]")
ax1.set_zlabel("pos_z [m]")
ax1.view_init(elev=20, azim=30)

# ax2.plot(df["pos_x"], df["pos_y"], color='tab:gray')
# ax2.set_title("XY Projection")
# ax2.set_xlabel("pos_x [m]")
# ax2.set_ylabel("pos_y [m]")
# ax2.axis("equal")
# ax2.grid(True)

# ---- (2) Altitude ----
ax = axs[1]
if "pos_z" in df.columns:
    ax.plot(t, -df["pos_z"], color='tab:orange')
ax.set_title("Altitude")
ax.set_xlabel("time [s]")
ax.set_ylabel("pos_z [m]")
ax.grid(True)



# ---- (5) Cross Tracking Error + Velocity Error ----
ax = axs[2]

ax.set_title("Error")
ax.set_xlabel("time [s]")
ax.set_ylabel("error")
ax.grid(True)

# Cross tracking error
if "cross-tracking_err" in df.columns:
    y_ct = df["cross-tracking_err"].replace(1e12, np.nan)
    ax.plot(t, y_ct, label="cross-track err[m]", color='tab:blue')

# Velocity error
if "vel_err" in df.columns:
    ax.plot(t, df["vel_err"], label="velocity err[m/s]", color='tab:orange')

ax.legend()


if "MPPI_time" in df.columns:
    mean_time = df["MPPI_time"].mean()
    std_time  = df["MPPI_time"].std() 
    text = f"Average MPPI compute time:\n{mean_time:.4f} ± {std_time:.4f} s"
    print(text)  
else:
    text = "No MPPI_time data"

# === Layout adjustment ===
plt.tight_layout(rect=[0, 0, 1, 0.97])  # leave space for main title

# === Save one summary figure ===
csv_name = os.path.splitext(os.path.basename(src_csv))[0]
out_path = os.path.join(DEST_DIR, f"summary_plot_lite_{csv_name}.png")
plt.savefig(out_path, dpi=200)
plt.close(fig)

print(f"[INFO] Saved summary figure: {out_path}")
