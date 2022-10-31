import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib import cm
import localization
import path_classification

# Params
budgets = [
    ["System I", 4, 8], ["System II", 8, 16]
]
set = "office-walls"
dumb_plot = True
architecture = "HH"
link = "up"
samples = 218
cases = [
    ("SMOMP", 512, -40),
    ("SMOMP", 512, -30),
    ("SMOMP", 512, -20),
    ("SMOMP", 512, -10),
    ("SMOMP", 512, 0),
    ("SMOMP", 512, 10),
    ("SMOMP", 512, 20),
    ("SMOMP", 512, 30),
    ("SMOMP", 512, 40),
]
figsize = (4, 3)
# Power
#p_t_dBm = 20            # dBm
# Noise related
T = 15                  # C
k_B = 1.38064852e-23    # Boltzmanz's constant
# Speed of light
c = 3e8
colors = cm.get_cmap('viridis')(np.linspace(0.2, 0.8, len(budgets)))
print(colors)
fig = plt.figure("Line", figsize=figsize)
ax = fig.add_subplot(111)
axins = zoomed_inset_axes(ax, 80, loc="upper right", borderpad=1)
for (budget_name, N_UE, N_AP), color in zip(budgets, colors):
    # Antennas
    #N_UE = 8                # Number of UE antennas in each dimension
    #N_AP = 16                # Number of AP antennas in each dimension
    N_RF_UE = N_UE if architecture == "HH" else 1   # Number of UE RF-chains in total
    N_RF_AP = N_AP          # Number of AP RF-chains in total
    # Carriers
    f_c = 60                # GHz
    B = 2                   # GHz
    K = 64                  # Number of frequency carriers
    Q = 64
    # Measurements
    N_M_UE = N_UE              # Number of UE measurements in each dimension
    N_M_AP = N_AP              # Number of AP measurements in each dimension
    # Estimation
    N_est = 5               # Number of estimated paths
    K_res_lr = 2            # Coarse resolution factor

    # Initialization
    Quartiles_loc = []
    Quartiles_dist = []
    Quartiles_angle = []
    Quartiles_loc_angle = []

    # Load data
    with open("data/{}/AP_pos.txt".format(set)) as f:
        AP_pos = [[float(el) for el in line.split()] for line in f.read().split("\n")[1:-1]]
    with open("data/{}/UE_pos.txt".format(set)) as f:
        UE_pos = [[float(el) for el in line.split()] for line in f.read().split("\n")[1:-1]]
    with open("data/{}/AP_selected.txt".format(set)) as f:
        AP_selected = [int(a) for a in f.read().split("\n")[1].split()]

    # Crop data
    UE_pos, AP_selected = [X[:samples] for X in [UE_pos, AP_selected]]

    # Label script
    def str_label(N_M_UE, N_M_AP):
        if architecture == "AH":
            return "$M={}$".format((N_M_UE*N_M_AP)**2//N_RF_AP)
        elif architecture == "HH":
            return "$M={}$".format((N_M_UE*N_M_AP)**2//(N_RF_AP*N_RF_UE))

    # Loop
    Pt_vec = []
    Best_50_Error_loc = []
    Success = []
    for method, K_res, p_t_dBm in cases:
        with open(
            "data/{}/{}/paths/{}_{}_{}_{}_{}_{}_{}_{}.json".format(
                set, architecture, method, N_UE, N_AP, N_M_UE, N_M_AP, Q, p_t_dBm, int(10*K_res)),
            "r") as f:
            estimations = json.loads(f.read())

        # Localize
        UE_pos_est = []
        for estimation, ue_pos, ap, ii_ue in zip(estimations, UE_pos, AP_selected, range(len(UE_pos))):
            Power = np.asarray(estimation['Power'])
            DoA = np.asarray(estimation['DoA'])
            DoD = np.asarray(estimation['DoD'])
            DDoF = np.asarray(estimation['DDoF'])
            classi = path_classification.RX_TXF(DoA, DoD, th_az=0.12, th_el=0.12)
            pos = localization.localization_single_ap(AP_pos[ap], DoA, DDoF, classi=classi, weights=np.power(10, (Power-np.min(Power))/20))
            if pos is not None:
                UE_pos_est.append(pos.tolist())
            else:
                UE_pos_est.append(None)

        UE_pos_crop = np.asarray([ue_pos for ii_ue, ue_pos in enumerate(UE_pos) if UE_pos_est[ii_ue] is not None])
        UE_pos_est_crop = np.asarray([ue_pos for ue_pos in UE_pos_est if ue_pos is not None])
        success = np.sum([ue_pos is not None for ue_pos in UE_pos_est])
        if success > 1:
            # Statistical information
            AP_sel_crop = np.asarray([AP_pos[ap] for ii_ue, ap in enumerate(AP_selected) if UE_pos_est[ii_ue] is not None])

            ### Location error
            Error_loc = np.linalg.norm(UE_pos_crop-UE_pos_est_crop, ord = 2, axis = 1)
            Error_loc[np.isnan(Error_loc)] = np.Inf

            ### Distance error
            Error_dist = np.abs(np.linalg.norm(UE_pos_crop-AP_sel_crop, ord = 2, axis = 1)-np.linalg.norm(UE_pos_est_crop-AP_sel_crop, ord = 2, axis = 1))
            Error_dist[np.isnan(Error_dist)] = np.Inf

            ### Angle error
            Error_angle = (180/np.pi)*np.arccos(np.sum((UE_pos_crop-AP_sel_crop)*(UE_pos_est_crop-AP_sel_crop), axis = 1)/(np.linalg.norm(UE_pos_crop-AP_sel_crop, ord = 2, axis = 1)*np.linalg.norm(UE_pos_est_crop-AP_sel_crop, ord = 2, axis = 1)))
            Error_loc_angle = Error_angle*(np.pi/180)*np.linalg.norm(UE_pos_crop-AP_sel_crop, ord = 2, axis = 1)
            Error_angle[np.isnan(Error_angle)] = np.Inf
            Error_loc_angle[np.isnan(Error_loc_angle)] = np.Inf

            # Collect vector information
            Pt_vec.append(p_t_dBm)
            #Best_50_Error_loc.append([err for err in Error_loc if err < np.quantile(Error_loc, 0.5*samples/success)])
            Best_50_Error_loc.append(Error_loc)
            Success.append(success)

    # Build statistical plots
    plt.figure("Line", figsize=figsize)
    if dumb_plot:
        for axis in [ax, axins]:
            axis.plot(Pt_vec, [np.quantile(err_vec, 0.05) for err_vec in Best_50_Error_loc], "--", color=color, label=budget_name+" best 5%")
            axis.plot(Pt_vec, [np.quantile(err_vec, 0.5) for err_vec in Best_50_Error_loc], "-", color=color, label=budget_name+" best 50%")
            axis.plot(Pt_vec, [np.quantile(err_vec, 0.80) for err_vec in Best_50_Error_loc], ":", color=color, label=budget_name+" best 80%")
    else:
        for axis in [ax, axins]:
            axis.fill_between(Pt_vec, [np.quantile(err_vec, 0.25) for err_vec in Best_50_Error_loc], [np.quantile(err_vec, 0.75) for err_vec in Best_50_Error_loc], color=color, alpha=0.2)
            axis.plot(Pt_vec, [np.quantile(err_vec, 0.5) for err_vec in Best_50_Error_loc], "-", color=color, label=budget_name)

plt.figure("Line", figsize=figsize)
ax.set_xticks(Pt_vec)
ax.set_xlabel("Transmitted power [dBm]")
#ax.set_yscale("log")
ax.set_ylim([0, 10])
ax.set_ylabel("Localization error [m]")
ax.legend(loc="upper left")
if dumb_plot:
    axins.set_xlim(-0.1, 0.1)
    axins.set_ylim(0, 0.02)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
plt.tight_layout()
plt.savefig("data\\{}\\{}\\figures\\LineLocErr_Pt.png".format(set, architecture), bbox_inches='tight')
plt.savefig("data\\{}\\{}\\figures\\LineLocErr_Pt.pdf".format(set, architecture), bbox_inches='tight')