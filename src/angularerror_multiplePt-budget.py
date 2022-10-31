import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib import cm

# Params
budgets = [
    ["System I", 4, 8], ["System II", 8, 16]
]
set = "office-walls"
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
figsize = (5, 3.5)
# Power
#p_t_dBm = 20            # dBm
# Noise related
T = 15                  # C
k_B = 1.38064852e-23    # Boltzmanz's constant
# Speed of light
c = 3e8
colors = cm.get_cmap('viridis')(np.linspace(0.2, 0.8, len(budgets)))
print(colors)
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
        AP_pos = np.array([[float(el) for el in line.split()] for line in f.read().split("\n")[1:-1]])
    with open("data/{}/UE_pos.txt".format(set)) as f:
        UE_pos = np.array([[float(el) for el in line.split()] for line in f.read().split("\n")[1:-1]])
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
    Error = []
    for method, K_res, p_t_dBm in cases:
        Pt_vec.append(p_t_dBm)
        with open(
            "data/{}/{}/paths/{}_{}_{}_{}_{}_{}_{}_{}.json".format(
                set, architecture, method, N_UE, N_AP, N_M_UE, N_M_AP, Q, p_t_dBm, int(10*K_res)),
            "r") as f:
            estimations = json.loads(f.read())

        # Compute error
        Error_this = []
        for estimation, ue_pos, ap, ii_ue in zip(estimations, UE_pos, AP_selected, range(len(UE_pos))):
            Power = np.asarray(estimation['Power'])
            DoA = np.asarray(estimation['DoA'])
            DoD = np.asarray(estimation['DoD'])
            DDoF = np.asarray(estimation['DDoF'])
            doa_real = ue_pos - AP_pos[ap]
            doa_real /= np.linalg.norm(doa_real)
            doa_error = np.rad2deg(np.arccos(np.dot(DoA[0], doa_real)))
            Error_this.append(doa_error)
        Error.append(Error_this)

    # Build statistical plots
    plt.figure("Line", figsize=figsize)
    plt.fill_between(Pt_vec, [np.quantile(err_vec, 0.25) for err_vec in Error], [np.quantile(err_vec, 0.75) for err_vec in Error], color=color, alpha=0.2)
    plt.plot(Pt_vec, [np.quantile(err_vec, 0.5) for err_vec in Error], "-", color=color, label=budget_name)

plt.figure("Line", figsize=figsize)
plt.xticks(Pt_vec)
plt.xlabel("Transmitted power [dBm]")
#plt.yscale("log")
plt.xlim([np.min(Pt_vec), np.max(Pt_vec)])
plt.ylim([0, 2])
plt.ylabel("Localization error [°]")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("data\\{}\\{}\\figures\\LineA_Pt.png".format(set, architecture), bbox_inches='tight')
plt.savefig("data\\{}\\{}\\figures\\LineA_Pt.pdf".format(set, architecture), bbox_inches='tight')
plt.show()