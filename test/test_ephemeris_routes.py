from sky import Sky
from sensor import Compass
from antworld import load_routes
from centralcomplex import CX

from datetime import datetime, timedelta

import numpy as np


if __name__ == '__main__':
    noise = 0.
    dx = 1e-02  # meters
    dt = 1 / 30  # minutes
    delta = timedelta(minutes=dt)
    routes = load_routes()
    flow = dx * np.ones(2) / np.sqrt(2)

    compass = Compass()
    sky = Sky(phi_s=np.pi, theta_s=np.pi / 3)

    stats = {
        "max_alt": [],
        "noise": [],
        "opath": [],
        "ipath": [],
        "d_x": [],
        "d_c": [],
        "tau": []
    }

    avg_time = timedelta(0.)

    for enable_ephemeris in [True, False]:
        if enable_ephemeris:
            print("Foraging with a circadian mechanism.")
        else:
            print("Foraging without a circadian mechanism.")

        # stats
        d_x = []  # logarithmic distance
        d_c = []
        tau = []  # tortuosity
        ri = 0

        print("Routes: ", end="")

        for route in routes[::2]:
            net = CX(noise=0., pontin=False)
            net.update = True

            # sun position
            cur = datetime(2020, 8, 20, 6, 0)
            sev

    r_tcl = compass(sky)


