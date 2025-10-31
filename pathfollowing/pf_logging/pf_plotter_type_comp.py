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
fig, axs = plt.subplots(6, 3, figsize=(15, 10))
gs = axs[0, 0].get_gridspec()

for a in axs.ravel():
    a.remove()

ax_00_a = fig.add_subplot(gs[0:3, 0])  # col=0, rows 0~2
ax_00_b = fig.add_subplot(gs[3:6, 0])  # col=0, rows 3~5

ax_01_a = fig.add_subplot(gs[0:2, 1])   # col=1, rows 0~1
ax_01_b = fig.add_subplot(gs[2:4, 1])   # col=1, rows 2~3
ax_01_c = fig.add_subplot(gs[4:6, 1])   # col=1, rows 4~5

ax_02_a = fig.add_subplot(gs[0:2, 2])   # col=2, rows 0~1
ax_02_b = fig.add_subplot(gs[2:4, 2])   # col=2, rows 2~3
ax_02_c = fig.add_subplot(gs[4:6, 2])   # col=2, rows 4~5

# ---- (1) XY trajectory ----
ax = ax_00_a
if {"pos_x", "pos_y"}.issubset(df.columns):
    ax.plot(df["pos_x"], df["pos_y"], color='tab:blue')
ax.set_title("XY Trajectory")
ax.set_xlabel("pos_x [m]")
ax.set_ylabel("pos_y [m]")
ax.axis("equal")
ax.grid(True)

# ---- (2) Altitude  X-Z plane----
ax = ax_00_b
if {"pos_x", "pos_z"}.issubset(df.columns):
    ax.plot(df["pos_x"], -df["pos_z"], color='tab:blue')
ax.set_title("XZ Trajectory")
ax.set_xlabel("pos_x [m]")
ax.set_ylabel("pos_z [m]")
ax.axis("equal")
ax.grid(True)


# ---- (4) Euler angles ----
# # ---- roll
ax = ax_01_a
pairs = [("att_cmd_r", "cmd", "tab:red"),
         ("eul_roll",  "res", "tab:blue")]   # cmd???, res???
for col, label, color in pairs:
    if col in df.columns:
        ang = np.degrees(np.unwrap(df[col].to_numpy()))
        ax.plot(t, ang, label=label, color=color, linewidth=1.5)

ax.set_title("Attitude - Roll")
ax.set_xlabel("time [s]")
ax.set_ylabel("Roll [deg]")
ax.legend()
ax.grid(True)

# ---- pitch
ax = ax_01_b
pairs = [("att_cmd_p", "cmd", "tab:red"),
         ("eul_pitch", "res", "tab:blue")]
for col, label, color in pairs:
    if col in df.columns:
        ang = np.degrees(np.unwrap(df[col].to_numpy()))
        ax.plot(t, ang, label=label, color=color, linewidth=1.5)

ax.set_title("Attitude - Pitch")
ax.set_xlabel("time [s]")
ax.set_ylabel("Pitch [deg]")
ax.legend()
ax.grid(True)

# ---- yaw
ax = ax_01_c
pairs = [("att_cmd_y", "cmd", "tab:red"),
         ("eul_yaw",   "res", "tab:blue")]
for col, label, color in pairs:
    if col in df.columns:
        ang = np.degrees(np.unwrap(df[col].to_numpy()))
        ax.plot(t, ang, label=label, color=color, linewidth=1.5)

ax.set_title("Attitude - Yaw")
ax.set_xlabel("time [s]")
ax.set_ylabel("Yaw [deg]")
ax.legend()
ax.grid(True)


# ---- (5) Velocity  ----
ax = ax_02_a

if {"vel_x", "vel_y", "vel_z","des_spd"}.issubset(df.columns):
    speed = np.sqrt(df["vel_x"]**2 + df["vel_y"]**2 + df["vel_z"]**2)
    # ax.plot(t, df["vel_x"], label="vel_x", color="tab:blue", alpha=0.7)
    # ax.plot(t, df["vel_y"], label="vel_y", color="tab:orange", alpha=0.7)
    # ax.plot(t, df["vel_z"], label="vel_z", color="tab:green", alpha=0.7)
    ax.plot(t, df["des_spd"], label="des_|v|", color="tab:red", alpha=1)
    ax.plot(t, speed, label="|v|", color="tab:blue", linewidth=1.5)
ax.set_title("Speed")
ax.set_xlabel("time [s]")
ax.set_ylabel("Speed [m/s]")
ax.legend()
ax.grid(True)


# ---- (5) Cross Tracking Error + Velocity Error ----
ax = ax_02_b
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
ax = ax_02_c
for col, label in [("u_0", "Ax_cmd"), ("u_1", "guid_gain")]:
    if col in df.columns:
        ax.plot(t, df[col], label=label)
ax.set_title("Control Input from MPPI")
ax.set_xlabel("time [s]")
ax.set_ylabel("u")
ax.legend()
ax.grid(True)

# === Layout adjustment ===
plt.tight_layout(rect=[0, 0, 1, 0.97])  # leave space for main title

# === Save one summary figure ===
csv_name = os.path.splitext(os.path.basename(src_csv))[0]
out_path = os.path.join(DEST_DIR, f"summary_plot__type_comp_{csv_name}.png")
plt.savefig(out_path, dpi=200)
plt.close(fig)

print(f"[INFO] Saved summary figure: {out_path}")
