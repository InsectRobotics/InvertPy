from invertpy.brain.centralcomplex.fanshapedbody import WindGuidedLayer

from scipy.special import expit

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    w = WindGuidedLayer()

    tn1 = np.array([0.5, 1.])  # wind direction
    tan = np.array([0., 1.])  # differential MBON input

    # a = lambda x: expit(5 * x)
    a = lambda x: x / np.max(np.absolute(x))

    epg = -np.cos(np.linspace(0, 4 * np.pi, 16))  # heading direction
    tn1 = tn1 / np.linalg.norm(tn1)

    w(epg=(epg + 1) / 2, nod=tn1, mbon=np.r_[tan, tan, tan])

    # a_epg = a(epg)
    a_epg = w.r_epg

    # pfn = tn1.dot(w.w_nod2pfn) * epg.dot(w.w_epg2pfn)
    # a_pfn = a(pfn)
    a_pfn = w.r_pfn
    # hdc = a_pfn.dot(w.w_pfn2hdc) * tan.dot(w.w_tan2hdc)
    # a_hdc = a(hdc)
    a_hdc = w.r_hdc
    # pfl3 = epg.dot(w.w_epg2pfl3) + 0.3 * a_pfn.dot(w.w_pfn2pfl3) + 0.7 * a_hdc.dot(w.w_hdc2pfl3)
    # a_pfl3 = a(pfl3)
    a_pfl3 = w.r_pfl3
    # pfl2 = epg.dot(w.w_epg2pfl2) + 0.3 * a_pfn.dot(w.w_pfn2pfl2) + 0.7 * a_hdc.dot(w.w_hdc2pfl2)
    # a_pfl2 = a(pfl2)
    a_pfl2 = w.r_pfl2

    r_motor = np.maximum(a_pfl3 - .5, 0).reshape((2, -1)).mean(axis=1) + .5
    print(r_motor)

    plt.figure('test-wind-circuit', figsize=(10, 10))

    plt.subplot(2, 2, 1)
    plt.plot((a_epg[:8] + a_epg[8:]) / 2, 'k--', label='E-PG')
    plt.plot(a_pfn[8:], 'C0', label='PFN_R')
    plt.plot(a_pfn[:8], 'C1', label='PFN_L')
    plt.ylim([0, 1])
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot((a_epg[:8] + a_epg[8:]) / 2, 'k--')
    plt.plot((a_pfn[8:] + a_pfn[:8]) / 2, 'C6--', label='PFN')
    plt.plot((a_hdc[:8] + a_hdc[8:]) / 2, 'C3', label='hDC')
    plt.ylim([0, 1])
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot((a_pfn[8:] + a_pfn[:8]) / 2, 'C6--', label='PFN')
    plt.plot(a_pfl3[8:], 'C2', label='PFL3_R')
    plt.plot(a_pfl3[:8], 'C4', label='PFL3_L')
    plt.ylim([0, 1])
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot((a_pfn[8:] + a_pfn[:8]) / 2, 'C6--', label='PFN')
    plt.plot(a_pfl2[8:], 'C2', label='PFL2_R')
    plt.plot(a_pfl2[:8], 'C4', label='PFL2_L')
    plt.ylim([0, 1])
    plt.legend()

    plt.tight_layout()
    plt.show()
