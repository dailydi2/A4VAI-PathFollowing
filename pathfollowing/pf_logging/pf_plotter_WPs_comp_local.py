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
DEST_DIR = "/home/lyj801/Documents/A4VAI-SITL/ROS2/ros2_ws/src/pathfollowing/pathfollowing/pf_logging/pf_data"

os.makedirs(DEST_DIR, exist_ok=True)



def load_table(path: str) -> pd.DataFrame:
    name, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)            # ???? sheet_name="Sheet1"
    elif ext == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"???? ?? ???: {ext}")

def plot_summary(df: pd.DataFrame, out_path: str):

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
        y_ct = df["cross-tracking_err"].replace(1e12, np.nan)
        ax.plot(t, y_ct, label="cross-track err[m]", color='tab:blue')

    # Velocity error
    if "vel_err" in df.columns:
        ax.plot(t, df["vel_err"], label="velocity err[m/s]", color='tab:orange')

    ax.legend()

    # ---- (6) MPPI result ----
    ax = axs[2, 1]
    for col, label in [("u_0", "Ax_cmd"), ("u_1", "guid_gain"), ("T_norm", "T_norm")]:
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
    else:
        text = "No MPPI_time data"
    print(text)  


def main():
    if len(sys.argv) < 2:
        raise SystemExit("???: python plot_pf_summary.py <????(.xlsx/.xls/.csv)>")

    input_path = sys.argv[1]
    if not os.path.isabs(input_path):
        # ?? ???? ?? ????? ??
        input_path = os.path.abspath(input_path)

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"??? ????: {input_path}")

    df = load_table(input_path)
    base = os.path.basename(input_path)
    name, _ = os.path.splitext(base)
    out_path = os.path.join(DEST_DIR, f"summary_plot_{name}.png")

    print(f"[INFO] Load: {input_path}")
    print(f"[INFO] Columns: {list(df.columns)}")
    if "MPPI_time" in df.columns:
        print(f"[INFO] MPPI time = {df['MPPI_time'].mean():.4f} ± {df['MPPI_time'].std():.4f} s")

    plot_summary(df, out_path)
    print(f"[INFO] Saved: {out_path}")

if __name__ == "__main__":
    main()