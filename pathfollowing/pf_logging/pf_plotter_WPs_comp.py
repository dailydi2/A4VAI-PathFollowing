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
fig, axs = plt.subplots(3, 2, figsize=(10, 10))
# fig.suptitle("Pathfollowing Log Summary", fontsize=14, fontweight="bold")

# ---- (1) XY trajectory ----
ax = axs[0, 0]
if {"pos_x", "pos_y"}.issubset(df.columns):
    ax.plot(df["pos_x"], df["pos_y"], color='tab:blue')
ax.set_title("XY Trajectory")
ax.set_xlabel("pos_x [m]")
ax.set_ylabel("pos_y [m]")
ax.axis("equal")
ax.grid(True)

# ---- (2) Altitude ----
ax = axs[0, 1]
if "pos_z" in df.columns:
    ax.plot(t, -df["pos_z"], color='tab:orange')
ax.set_title("Altitude")
ax.set_xlabel("time [s]")
ax.set_ylabel("pos_z [m]")
ax.grid(True)

# ---- (3) Speed and velocity components ----
ax = axs[1, 0]
if {"vel_x", "vel_y", "vel_z","des_spd"}.issubset(df.columns):
    speed = np.sqrt(df["vel_x"]**2 + df["vel_y"]**2 + df["vel_z"]**2)
    ax.plot(t, df["vel_x"], label="vel_x", color="tab:blue", alpha=0.7)
    ax.plot(t, df["vel_y"], label="vel_y", color="tab:orange", alpha=0.7)
    ax.plot(t, df["vel_z"], label="vel_z", color="tab:green", alpha=0.7)
    ax.plot(t, speed, label="|v|", color="black", linewidth=1.5)
    ax.plot(t, df["des_spd"], label="des_|v|", color="tab:red", alpha=0.5)
ax.set_title("Velocity and Speed")
ax.set_xlabel("time [s]")
ax.set_ylabel("velocity [m/s]")
ax.legend()
ax.grid(True)

# ---- (4) Euler angles ----
ax = axs[1, 1]
for col, label in [("eul_roll", "roll"), ("eul_pitch", "pitch"), ("eul_yaw", "yaw")]:
    if col in df.columns:
        ang = np.degrees(np.unwrap(df[col].to_numpy()))
        ax.plot(t, ang, label=label)

ax.set_title("Euler Angles")
ax.set_xlabel("time [s]")
ax.set_ylabel("angle [deg]")
ax.legend()
ax.grid(True)


# ---- (5) Cross Tracking Error + Velocity Error ----
ax = axs[2, 0]

ax.set_title("Cross Tracking & Velocity Error")
ax.set_xlabel("time [s]")
ax.set_ylabel("error")
ax.grid(True)

# Cross tracking error
if "cross-tracking_err" in df.columns:
    y_ct = df["cross-tracking_err"].replace(9999, np.nan)
    ax.plot(t, y_ct, label="cross-track err[m]", color='tab:blue')

# Velocity error
if "vel_err" in df.columns:
    ax.plot(t, df["vel_err"], label="velocity err[m/s]", color='tab:orange')

ax.legend()

# ---- (6) MPPI result ----
ax = axs[2, 1]
for col, label in [("u_0", "Ax_cmd"), ("u_1", "guid_gain")]:
    if col in df.columns:
        ax.plot(t, df[col], label=label)
ax.set_title("Control Input from MPPI")
ax.set_xlabel("time [s]")
ax.set_ylabel("u")
ax.legend()
ax.grid(True)

# ---- (8) MPPI computation time ----
# ax = axs[3, 0]
# if "MPPI_time" in df.columns:
#     ax.plot(t, df["MPPI_time"], color='tab:red')
# ax.set_title("MPPI compute time")
# ax.set_xlabel("time [s]")
# ax.set_ylabel("sec")
# ax.grid(True)

# ax = axs[3, 1]

if "MPPI_time" in df.columns:
    mean_time = df["MPPI_time"].mean()
    std_time  = df["MPPI_time"].std() 
    text = f"Average MPPI compute time:\n{mean_time:.4f} ± {std_time:.4f} s"
    print(text)  
else:
    text = "No MPPI_time data"

# ax.text(0.5, 0.5, text,
#         ha='center', va='center',
#         transform=ax.transAxes,
#         fontsize=12, fontweight='bold')

# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_xlabel("") 
# ax.set_ylabel("")
# ax.set_title("MPPI compute time")
# # for spine in ax.spines.values():
# #     spine.set_visible(False)

# ax.grid(False) 

# === Layout adjustment ===
plt.tight_layout(rect=[0, 0, 1, 0.97])  # leave space for main title

# === Save one summary figure ===
csv_name = os.path.splitext(os.path.basename(src_csv))[0]
out_path = os.path.join(DEST_DIR, f"summary_plot_{csv_name}.png")
plt.savefig(out_path, dpi=200)
plt.close(fig)

print(f"[INFO] Saved summary figure: {out_path}")
