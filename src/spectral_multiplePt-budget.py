import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib import cm
import pywarraychannels

# Params
budgets = [
    ["System I", 4, 8], ["System II", 8, 16], ["System I - Perfect CSI", 4, 8], ["System II - Perfect CSI", 8, 16]
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
T += 273.1              # K
k_B = 1.38064852e-23    # Boltzmanz's constant
# Speed of light
c = 3e8
colors = cm.get_cmap('viridis')([0.2, 0.8, 0.2, 0.8])
styles = ["--", "--", "-", "-"]
print(colors)
fig = plt.figure("Line", figsize=figsize)
ax = fig.add_subplot(111)
#axins = zoomed_inset_axes(ax, 2.5, loc="upper right", borderpad=0.5)
channels = {}
for (budget_name, N_UE, N_AP), color, style in zip(budgets, colors, styles):
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
    orientations_AP = [pywarraychannels.uncertainties.Static(tilt=np.pi/2), pywarraychannels.uncertainties.Static(tilt=-np.pi/2)]
    # Estimation
    N_est = 5               # Number of estimated paths

    # Transform params to natural units
    f_c *= 1e9
    B *= 1e9

    # Compute noise level
    p_n = k_B*T*B

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
    with open("data/{}/Info_selected.txt".format(set)) as f:
        Rays = [pywarraychannels.em.Geometric([[float(p) for p in line.split()] for line in ue_block.split("\n")], bool_flip_RXTX=link=="up") for ue_block in f.read()[:-1].split("\n<ue>\n")]

    # Crop data
    UE_pos, AP_selected = [X[:samples] for X in [UE_pos, AP_selected]]

    # Label script
    def str_label(N_M_UE, N_M_AP):
        if architecture == "AH":
            return "$M={}$".format((N_M_UE*N_M_AP)**2//N_RF_AP)
        elif architecture == "HH":
            return "$M={}$".format((N_M_UE*N_M_AP)**2//(N_RF_AP*N_RF_UE))

    # Define pulse shape filter
    filter = pywarraychannels.filters.RCFilter(early_samples=8, late_samples=8)

    # Define antennas
    antenna_UE = pywarraychannels.antennas.RectangularAntenna((N_UE, N_UE), z_positive=True)
    antenna_AP = pywarraychannels.antennas.RectangularAntenna((N_AP, N_AP), z_positive=True)

    # Define channel Geometric
    if link == "up":
        channel_Geometric = pywarraychannels.channels.Geometric(
            antenna_AP, antenna_UE, f_c=f_c,
            B=B, K=K, filter=filter, bool_sync=True)
    else:
        channel_Geometric = pywarraychannels.channels.Geometric(
            antenna_UE, antenna_AP, f_c=f_c,
            B=B, K=K, filter=filter, bool_sync=True)

    # Loop
    Pt_vec = []
    SE = []
    for method, K_res, p_t_dBm in cases:
        p_t = np.power(10, (p_t_dBm-30)/10)
        Pt_vec.append(p_t_dBm)
        with open(
            "data/{}/{}/paths/{}_{}_{}_{}_{}_{}_{}_{}.json".format(
                set, architecture, method, N_UE, N_AP, N_M_UE, N_M_AP, Q, p_t_dBm, int(10*K_res)),
            "r") as f:
            estimations = json.loads(f.read())

        # Compute spectral efficiency
        SE_this = []
        budget_prop = budget_name.split(" ")[1]
        print(budget_prop)
        bool_compute_channel = not budget_prop in channels
        print(bool_compute_channel)
        if bool_compute_channel:
            channels[budget_prop] = []
        for ii_pos, (estimation, rays, ap) in enumerate(zip(estimations, Rays, AP_selected)):
            antenna_AP.uncertainty = orientations_AP[ap]
            N_S = np.min([N_RF_AP, N_RF_UE, len(estimation['Power'])])
            if bool_compute_channel:
                channel_Geometric.build(rays)
                channels[budget_prop].append(channel_Geometric.channel)
            else:
                channel_Geometric.channel = channels[budget_prop][ii_pos]
            if budget_name.split(" ")[-2] == "Perfect":
                Power = np.power(10, (rays.ray_info[:, 2]-30)/10)
                DoA = np.asarray([pywarraychannels.em.polar2cartesian(az, el) for az, el in np.deg2rad(rays.ray_info[:, 3:5])])
                DoD = np.asarray([pywarraychannels.em.polar2cartesian(az, el) for az, el in np.deg2rad(rays.ray_info[:, 5:7])])
                I = np.bitwise_and(antenna_AP.uncertainty.apply_inverse(DoA)[:, 2] >= 0, antenna_UE.uncertainty.apply_inverse(DoD)[:, 2] >= 0)
                if np.sum(I) == 0:
                    I[0] = True
                Power = Power[I][:N_S]
                DoA = DoA[I][:N_S]
                DoD = DoD[I][:N_S]
                if link == "up":
                    antenna_AP.set_codebook(antenna_AP.steering_vector(DoA).T)
                    antenna_UE.set_codebook(antenna_UE.steering_vector(DoD).T)
                else:
                    antenna_AP.set_codebook(antenna_AP.steering_vector(DoD).T)
                    antenna_UE.set_codebook(antenna_UE.steering_vector(DoA).T)
                channel_freq = np.fft.fft(channel_Geometric.channel, K, axis=2, norm="ortho")/np.sqrt(K)
                se = 0
                for ch in channel_freq.transpose([2, 0, 1]):
                    if True:
                        u, s, v = np.linalg.svd(ch, full_matrices=False)
                        v = np.conj(v).T
                        s = s[:N_S]
                        u = u[..., :N_S]
                        v = v[..., :N_S]
                    else:
                        u, s, v = antenna_AP.codebook, np.sqrt(Power*(N_AP*N_UE)**2), np.conj(antenna_UE.codebook).T
                        v = np.conj(v).T
                        s = s[:N_S]
                        u, _, _ = np.linalg.svd(u[..., :N_S], full_matrices=False)
                        v = v[..., :N_S]
                    if np.sum(s) > 0:
                        power_alloc = pywarraychannels.utils.water_filling(p_n/s**2, p_t, return_cells=True)
                        measurement_ii_freq = np.dot(np.conj(u).T, np.dot(ch, v*np.sqrt(power_alloc)[None, :]))
                        eig = np.linalg.svd(measurement_ii_freq, compute_uv=False)**2
                        se += np.sum(np.log2(1+eig/p_n))
                SE_this.append(se)
            else:
                Power = np.power(10, (np.asarray(estimation['Power'])[:N_S]-30)/10)/p_t
                DoA = np.asarray(estimation['DoA'])[:N_S]
                DoD = np.asarray(estimation['DoD'])[:N_S]
                if link == "up":
                    antenna_AP.set_codebook(antenna_AP.steering_vector(DoA).T)
                    antenna_UE.set_codebook(antenna_UE.steering_vector(DoD).T)
                else:
                    antenna_AP.set_codebook(antenna_AP.steering_vector(DoD).T)
                    antenna_UE.set_codebook(antenna_UE.steering_vector(DoA).T)
                #measurement = channel_Geometric.measure([1])
                """
                measurement = np.tensordot(np.tensordot(channel_Geometric.channel, np.conj(antenna_AP.codebook), axes=(0, 0)), antenna_UE.codebook, axes=(0, 0)).transpose([1, 2, 0])
                measurement_freq = np.fft.fft(measurement, K, axis=2, norm="ortho")
                power_alloc = pywarraychannels.utils.water_filling(p_n/Power, p_t, return_cells=True)
                #power_alloc = np.ones(N_S)*p_t/N_S
                measurement_freq *= np.sqrt(power_alloc)[None, :, None]
                se = 0
                for ii_freq in range(measurement_freq.shape[2]):
                    measurement_ii_freq = measurement_freq[:, :, ii_freq]
                    eig = np.linalg.svd(measurement_ii_freq, full_matrices=False, compute_uv=False)**2
                    se += np.sum(np.log2(1+eig/p_n))
                """
                channel_freq = np.fft.fft(channel_Geometric.channel, K, axis=2, norm="ortho")/np.sqrt(K)
                se = 0
                for ch in channel_freq.transpose([2, 0, 1]):
                    u, s, v = antenna_AP.codebook, np.sqrt(Power*(N_AP*N_UE)**2), np.conj(antenna_UE.codebook).T
                    v = np.conj(v).T
                    s = s[:N_S]
                    u, _, _ = np.linalg.svd(u[..., :N_S], full_matrices=False)
                    v = v[..., :N_S]
                    v /= np.linalg.norm(v, ord=2, axis=0, keepdims=True)
                    if np.sum(s) > 0:
                        power_alloc = pywarraychannels.utils.water_filling(p_n/s**2, p_t, return_cells=True)
                        measurement_ii_freq = np.dot(np.conj(u).T, np.dot(ch, v*np.sqrt(power_alloc)[None, :]))
                        eig = np.linalg.svd(measurement_ii_freq, compute_uv=False)**2
                        se += np.sum(np.log2(1+eig/p_n))
                SE_this.append(se)
            if ii_pos == 0:
                print(budget_name)
                print(p_t_dBm)
                print(power_alloc)
                print(eig/p_n)
                print(s**2/p_n)
                print(np.linalg.norm(ch))
                print(se)
        SE.append(SE_this)

    # Build statistical plots
    if dumb_plot:
        for axis in [ax]:
            #axis.plot(Pt_vec, [np.quantile(se_vec, 0.95) for se_vec in SE], "--", color=color, label=budget_name+" best 5%")
            #axis.plot(Pt_vec, [np.quantile(se_vec, 0.5) for se_vec in SE], "-", color=color, label=budget_name+" best 50%")
            #axis.plot(Pt_vec, [np.quantile(se_vec, 0.2) for se_vec in SE], ":", color=color, label=budget_name+" best 80%")
            axis.plot(Pt_vec, [np.mean(se_vec) for se_vec in SE], style, color=color, label=budget_name)
    else:
        for axis in [ax]:
            ax.fill_between(Pt_vec, [np.quantile(se_vec, 0.25) for se_vec in SE], [np.quantile(se_vec, 0.75) for se_vec in SE], color=color, alpha=0.2)
            ax.plot(Pt_vec, [np.quantile(se_vec, 0.5) for se_vec in SE], "-", color=color, label=budget_name)

ax.set_xticks(Pt_vec)
ax.set_xlabel("Transmitted power [dBm]")
ax.set_xlim([np.min(Pt_vec), np.max(Pt_vec)])
#ax.set_ylim([0, 2])
ax.set_ylabel("Spectral efficiency [Gbps/GHz]")
ax.legend(loc="upper left")
#if dumb_plot:
#    axins.set_xlim(5, 15)
#    axins.set_ylim(0, 600)
#    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
plt.tight_layout()
plt.savefig("data\\{}\\{}\\figures\\LineSE_Pt.png".format(set, architecture), bbox_inches='tight')
plt.savefig("data\\{}\\{}\\figures\\LineSE_Pt.pdf".format(set, architecture), bbox_inches='tight')
#plt.yscale("log")
#plt.legend(loc="lower right")
#plt.savefig("data\\{}\\{}\\figures\\LineSE_Pt_logy.png".format(set, architecture), bbox_inches='tight')
#plt.savefig("data\\{}\\{}\\figures\\LineSE_Pt_logy.pdf".format(set, architecture), bbox_inches='tight')
plt.show()