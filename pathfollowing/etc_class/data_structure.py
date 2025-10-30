############################################################
#
#   - Name : data_structure.py
#
#                   -   Created by E. T. Jeong, 2024.02.01
#
############################################################


#.. Library
# pulbic libs.
import numpy as np


# private libs.

#.. State_Var
class State_Var():
    #.. initialize an instance of the class
    def __init__(self) -> None:
        self.Ri         =   np.array([0., 0., 0.])
        self.Vi         =   np.array([0., 0., 0.])
        self.att_ang    =   np.array([0., 0., 0.])
        self.Ai         =   np.array([0., 0., 0.])
        self.cI_B       =   np.identity(3)
        pass
    pass

#.. Path_Following_Var
class Path_Following_Var():
    #.. initialize an instance of the class
    def __init__(self) -> None:
        self.VT_Ri                       =   np.array([0., 0., 0.])
        self.WP_idx_passed               =   0
        self.WP_idx_heading              =   1
        self.PF_goal                     =   False
        self.PF_done                     =   False
        self.WP_manual                   =   0
        self.reWP_flag                   =   0 # 20240914 diy
        self.reset_flag2mppi             =   0 # 241223 diy
        self.stop_flag                   =   0
        self.intr_flag                   =   0
        self.intr_prev                   =   0
        self.coli_flag                   =   0
        self.tran_flag                   =   0
        self.point_closest_on_path_i     =   np.array([0., 0., 0.])
        self.dist_to_path                =   9999.
        self.unit_Rw1w2                  =   np.array([1., 0., 0.])
        self.cost_arr                    =   np.array([0., 0., 0.])
        self.total_cost                  =   0.
        self.terminal_cost               =   0.
        self.init_WP_idx_passed          =   0
        self.final_WP_idx_passed         =   0
        self.init_point_closest_on_path  =   np.array([1., 0., 0.])
        self.final_point_closest_on_path =   np.array([1., 0., 0.])
        self.init_time                   =   0.
        self.final_time                  =   0.
        pass
    pass

#.. Guid_Var
class Guid_Var():
    #.. initialize an instance of the class
    def __init__(self) -> None:
        self.Ai_cmd             =   np.array([0., 0., 0.])
        self.Ai_cmd_compensated =   np.array([0., 0., 0.])
        self.Ai_disturbance     =   np.array([0., 0., 0.])
        self.T_cmd              =   9.81
        self.norm_T_cmd         =   0
        self.att_ang_cmd        =   np.array([0., 0., 0.])
        self.out_NDO            =   np.array([0., 0., 0.])
        self.z_NDO              =   np.array([0., 0., 0.])
        self.Ai_rotor_drag      =   np.array([0., 0., 0.])
        self.MPPI_ctrl_input    =   np.array([0.5, 2.])
        self.MPPI_calc_time     =   0.
        self.qd_cmd             =   np.array([0., 0., 0., 0.])
        self.guid_type_used     =   0
        pass
    pass
