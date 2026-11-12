import argparse
import numpy as np
from pathlib import Path
import time
start_time = time.time()
import matplotlib.pyplot as plt

import xobjects as xo
import xtrack as xt
import xcoll as xc


parser = argparse.ArgumentParser()
parser.add_argument(
    "line",
    type=str,
    help="path to xsuite line to track"
)

def main():
    args = parser.parse_args()
    line_path = args.line

    nemitt_x = 2.5e-6
    nemitt_y = 2.5e-6
    sigma_z = 0.076

    line = xt.load(line_path)

    '''
    Twiss without beam-beam
    '''
    line["beambeam_scale"] = 0
    tw = line.twiss(method="6d")
    b_size = tw.get_beam_covariance(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                                    gemitt_zeta=sigma_z**2 / tw.bets0)
    line["beambeam_scale"] = 1.0
    tw_bb = line.twiss(method="6d")
    b_size_bb = tw.get_beam_covariance(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                                       gemitt_zeta=sigma_z**2 / tw.bets0)

    print(tw.qx, tw.qy, tw.dqx, tw.dqy)
    print(tw_bb.qx, tw_bb.qy, tw_bb.dqx, tw_bb.dqy)

    fs = 18
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), layout="tight")
    ax0 = axes[0]
    ax1 = axes[1]

    ax0.plot(tw.s, tw.betx, color='black', linestyle='-', label=r"$x$, no BB")
    ax0.plot(tw_bb.s, tw_bb.betx, color='gray', linestyle='--', label=r"$x$, with BB")
    ax0.plot(tw.s, tw.bety, color='red', linestyle='-', label=r"$y$, no BB")
    ax0.plot(tw_bb.s, tw_bb.bety, color='coral', linestyle='--', label=r"$y$, with BB")
    ax0.set_xlabel(r"$s$ [m]", fontsize=fs)
    ax0.set_ylabel(r"$\beta$ [m]", fontsize=fs)
    ax0.tick_params(axis="both", labelsize=fs - 2)
    ax0.legend(loc="best", fontsize=fs-2)

    ax1.plot(tw.s, tw.x * 1e3, color='black', linestyle='-', label=r"$x$, no BB")
    ax1.plot(tw_bb.s, tw_bb.x * 1e3, color='gray', linestyle='--', label=r"$x$, with BB")
    ax1.plot(tw.s, tw.y * 1e3, color='red', linestyle='-', label=r"$y$, no BB")
    ax1.plot(tw_bb.s, tw_bb.y * 1e3, color='coral', linestyle='--', label=r"$y$, with BB")
    ax1.set_xlabel(r"$s$ [m]", fontsize=fs)
    ax1.set_ylabel(r"orbit [mm]", fontsize=fs)
    ax1.tick_params(axis="both", labelsize=fs - 2)
    ax1.legend(loc="best", fontsize=fs-2)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), layout="tight")
    ax0 = axes[0]
    ax1 = axes[1]

    ax0.plot(tw.s, (tw_bb.betx-tw.betx)/tw.betx*100, color='black', linestyle='-', label=r"$x$")
    ax0.plot(tw.s, (tw_bb.bety-tw.bety)/tw.bety*100, color='red', linestyle='-', label=r"$y$")
    ax0.set_xlabel(r"$s$ [m]", fontsize=fs)
    ax0.set_ylabel(r"$\Delta\beta$ [%]", fontsize=fs)
    ax0.tick_params(axis="both", labelsize=fs - 2)
    ax0.legend(loc="best", fontsize=fs-2)

    ax1.plot(tw.s, (tw_bb.x/b_size_bb.sigma_x-tw.x/b_size.sigma_x), color='black', linestyle='-', label=r"$x$")
    ax1.plot(tw.s, (tw_bb.y/b_size_bb.sigma_y-tw.y/b_size.sigma_y), color='red', linestyle='-', label=r"$y$")
    ax1.set_xlabel(r"$s$ [m]", fontsize=fs)
    ax1.set_ylabel(r"$\Delta$ orbit [$\sigma$]", fontsize=fs)
    ax1.tick_params(axis="both", labelsize=fs - 2)
    ax1.legend(loc="best", fontsize=fs-2)

    plt.show()


    '''
    Get tune footprint
    '''
    line["beambeam_scale"] = 0
    fp = line.get_footprint(nemitt_x=nemitt_x, nemitt_y=nemitt_y)
    line["beambeam_scale"] = 1.0
    fp_bb = line.get_footprint(
        nemitt_x=nemitt_x, nemitt_y=nemitt_y, 
        linear_rescale_on_knobs=[
            xt.LinearRescale(knob_name='beambeam_scale', v0=0.0, dv=0.1)
        ]
    )

    plt.figure(1)
    fp.plot(color='green', label=f'no BB')
    fp_bb.plot(color='purple', label=f'with BB')
    plt.legend()

    plt.show()



if __name__ == "__main__":
    main()