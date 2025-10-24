import os
import csv
from datetime import datetime
from zoneinfo import ZoneInfo

class Logger:
    def __init__(self):
        log_dir = "/home/user/workspace/ros2/logs"
        # log_dir = "/home/lyj801/Documents/A4VAI-SITL/ROS2/ros2_ws/src/pathfollowing/pathfollowing/log"

        timestamp = datetime.now(tz=ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d_%H-%M-%S")
        self.filename = os.path.join(log_dir, f"{timestamp}.csv")

        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "time",
                "pos_x", "pos_y", "pos_z",
                "vel_x", "vel_y", "vel_z",
                "eul_roll", "eul_pitch", "eul_yaw",
                "u_0", "u_1",
                "MPPI_time",
                "cross-tracking_err", "vel_err"
            ])

    def update(self, t, pos, vel, eul, gain, cal_t, ct_err, vel_err):
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                t,
                pos[0], pos[1], pos[2],
                vel[0], vel[1], vel[2],
                eul[0], eul[1], eul[2],
                gain[0], gain[1], 
                cal_t,
                ct_err, vel_err
            ])


logger = Logger()

