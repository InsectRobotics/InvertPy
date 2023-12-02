from invertpy.brain.mushroombody import VectorMemoryMB

import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    m = VectorMemoryMB(nb_cs=2, nb_us=2, nb_kc=10, cs_magnitude=1, ltm_charging_speed=.5)
    print(m)

    plt.figure("familiarity-circuit", figsize=(15, 4))

    plt.subplot(141)
    plt.title("$w_{m2m}$")
    plt.imshow(m.w_m2m, vmin=-1, vmax=1, cmap='coolwarm')
    print(m.mbon_names)
    plt.yticks(np.arange(m.nb_mbon), [f"${name}$" for name in m.mbon_names])
    plt.xticks(np.arange(m.nb_mbon), [f"${name}$" for name in m.mbon_names], rotation="vertical")

    plt.subplot(142)
    plt.title("$w_{m2d}$")
    plt.imshow(m.w_m2d, vmin=-1, vmax=1, cmap='coolwarm')
    plt.yticks(np.arange(m.nb_mbon), [f"${name}$" for name in m.mbon_names])
    plt.xticks(np.arange(m.nb_dan), [f"${name}$" for name in m.dan_names], rotation="vertical")

    plt.subplot(143)
    plt.title("$w_{d2m}$")
    plt.imshow(m.w_d2m, vmin=-1, vmax=1, cmap='coolwarm')
    plt.yticks(np.arange(m.nb_dan), [f"${name}$" for name in m.dan_names])
    plt.xticks(np.arange(m.nb_mbon), [f"${name}$" for name in m.mbon_names], rotation="vertical")

    plt.subplot(144)
    plt.title("$w_{d2d}$")
    plt.imshow(m.w_d2d, vmin=-1, vmax=1, cmap='coolwarm')
    plt.yticks(np.arange(m.nb_dan), [f"${name}$" for name in m.dan_names])
    plt.xticks(np.arange(m.nb_dan), [f"${name}$" for name in m.dan_names], rotation="vertical")

    plt.tight_layout()

    # plt.figure("P2K", figsize=(15, 4))
    # plt.imshow(m.w_c2k, vmin=-1, vmax=1, cmap='coolwarm')
    # plt.tight_layout()

    plt.show()
