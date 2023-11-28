import invertpy.brain.centralcomplex.fanshapedbody_dye as fb
import invertpy.brain.centralcomplex.stone as stone


class DyeMemoryCX(stone.StoneCX):

    def __init__(self, *args, epsilon=0.1e+04, length=1e-03, k=0.0001, beta=0.3, phi=0.00045, c_tot=0.3,
                 # volume=1e-18, wavelength=750, W_max=1e-15,
                 gain=1.0, noise=0.0, mem_initial=None, **kwargs):
        stone.StoneCX.__init__(self, *args, noise=noise, **kwargs)

        self["memory"] = fb.PathIntegrationDyeLayer(
            nb_tb1=self.nb_tb1, nb_cpu4=self.nb_cpu4, nb_tn1=self.nb_tn1, nb_tn2=self.nb_tn2,
            epsilon=epsilon, length=length, k=k, beta=beta, phi=phi, c_tot=c_tot,
            # volume=volume, wavelength=wavelength, W_max=W_max,
            noise=noise, gain=gain, mem_initial=mem_initial,
        )
