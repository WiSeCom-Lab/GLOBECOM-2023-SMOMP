import localization
import numpy as np

# Generate devices
AP = np.random.randn(3)
UE = np.random.randn(3)
Walls_dist = np.asarray([-3, -3, -3])
Walls_dist_2 = np.asarray([5, 4, 3])

# Compute virtual images
V_UE = np.ones((6, 1))*UE[np.newaxis, :]
V_AP = np.ones((6, 1))*AP[np.newaxis, :]
for ii in range(3):
    V_UE[ii, ii] += 2*(Walls_dist[ii]-V_UE[ii, ii])
    V_AP[ii, ii] += 2*(Walls_dist[ii]-V_AP[ii, ii])
for ii in range(3):
    V_UE[ii+3, ii] += 2*(Walls_dist_2[ii]-V_UE[ii+3, ii])
    V_AP[ii+3, ii] += 2*(Walls_dist_2[ii]-V_AP[ii+3, ii])

# Compute path information
DoA = np.concatenate([(UE-AP)[np.newaxis, :], V_UE-AP[np.newaxis, :]])
DoD = np.concatenate([(AP-UE)[np.newaxis, :], V_AP-UE[np.newaxis, :]])
DoA = DoA/np.linalg.norm(DoA, ord=2, axis=1)[:, np.newaxis]
DoD = DoD/np.linalg.norm(DoD, ord=2, axis=1)[:, np.newaxis]
DDoF = np.concatenate([
    np.linalg.norm((AP-UE)[np.newaxis, :], ord=2, axis=1),
    np.linalg.norm(V_AP-UE[np.newaxis, :], ord=2, axis=1)])
DDoF -= DDoF[0]

print("Error (ADoA+ADoD+DDoF) classic: {:.2f}".format(
    np.linalg.norm(
        UE - localization.localization_single_ap_classic(AP, DoA, DoD, DDoF))))

print("Error (AoA+AoD+DDoF): {:.2f}".format(
    np.linalg.norm(
        UE - localization.localization_single_ap(AP, DoA, DDoF, dirs_D=DoD))))

print("Error (AoA+DDoF): {:.2f}".format(
    np.linalg.norm(
        UE - localization.localization_single_ap(AP, DoA, DDoF))))

print("Error (AoA+AoD+DDoF, NLoS): {:.2f}".format(
    np.linalg.norm(
        UE - localization.localization_single_ap_NLoS(AP, DoA[1:], DoD[1:], DDoF[1:]))))

print("Error (LoS noclassi): {:.2f}".format(
    np.linalg.norm(
        UE - localization.localization_LoS_noclassi(AP, DoA, DDoF))))

print("Error (LoS noclassi Anchors): {:.2f}".format(
    np.linalg.norm(
        UE - localization.localization_LoS_noclassiAnchors(np.concatenate([AP[np.newaxis], V_AP[3::-1]], axis=0), DoA, DDoF))))

print("Error (2Anchors): {:.2f}".format(
    np.linalg.norm(
        UE - localization.localization_2Anchors(np.concatenate([AP[np.newaxis], V_AP], axis=0), DoA, DDoF, weights=[1]+6*[1]))))

print("Error (3Anchors): {:.2f}".format(
    np.linalg.norm(
        UE - localization.localization_3Anchors(np.concatenate([AP[np.newaxis], V_AP], axis=0), DoA, DDoF, weights=[1]+6*[1]))))