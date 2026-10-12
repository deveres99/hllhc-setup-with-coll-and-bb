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

lhc_collimation_metadata = {"warm_regions": [[0.0, 22.5365], [54.853, 152.489], [172.1655, 192.4], [199.4847, 224.3], [3095.45428, 3155.62858], [3167.74008, 3188.43308], [3211.44458, 3263.86758], [3309.90008, 3354.97408], [3401.00558, 3453.42858], [3476.44008, 3494.06558], [3505.88528, 3568.31858], [6405.4088, 6457.9138], [6468.7785, 6859.5138], [6870.3785, 6923.5338], [9735.90702, 9824.73052], [9830.83202, 9861.73052], [9878.73202, 9939.98552], [9950.54802, 10043.462], [10054.0245, 10115.278], [10132.2795, 10163.9705], [10170.072, 10257.603], [13104.9892, 13129.8045], [13136.8892, 13157.1237], [13176.8002, 13271.6472], [13306.7527, 13351.8257], [13386.9312, 13481.7782], [13501.4547, 13522.7845], [13529.8692, 13554.6845], [16394.6378, 16450.8713], [16456.9728, 16487.2713], [16493.3728, 16830.8713], [16836.9728, 16867.2713], [16873.3728, 16928.2948], [19734.8504, 19760.6997], [19771.5644, 20217.9087], [20228.7734, 20252.9744], [23089.9797, 23138.577], [23150.3967, 23171.3755], [23194.387, 23246.81], [23292.8425, 23337.9155], [23383.948, 23436.371], [23459.3825, 23480.0825], [23492.194, 23553.116], [26433.4879, 26458.3032], [26465.3879, 26486.7177], [26506.3942, 26601.2412], [26636.3467, 2.66588832]], "s_range": {"full": [0.0, 26658.883176584142], "betatron": [19394, 20594], "momentum": [6065, 7265], "dump": [16062, 17262], "injection": [2890, 3775], "IP1": [26216, 443], "IP2": [2890, 3775], "IP5": [12886, 13773], "IP8": [22884, 23770]}}


def track_lm(line_path, colldb_path, beam, plane, num_turns, num_particles):
    line = xt.load(line_path)

    colldb = xc.CollimatorDatabase.from_yaml(colldb_path, beam=beam)
    colldb.install_everest_collimators(line=line, verbose=True)

    tw = line.twiss()
    line.xcoll.collimators.assign_optics(twiss=tw)

    tcp  = f"tcp.{'c' if plane=='H' else 'd'}6{'l' if f'{beam}'=='1' else 'r'}7.b{beam}"
    part = line[tcp].generate_pencil(num_particles)

    line.discard_tracker()
    line.build_tracker(_context=xo.ContextCpu(omp_num_threads='auto'))

    line.xcoll.scattering.enable()
    line.track(part, num_turns=num_turns, time=True, with_progress=1)
    line.xcoll.scattering.disable()
    print(f"Done tracking in {line.time_last_track:.1f}s.")

    line.discard_tracker()
    line.build_tracker(_context=xo.ContextCpu())

    lm_time = time.time()
    line_is_reversed = True if f'{beam}' == '2' else False
    ThisLM = xc.LossMap(line, line_is_reversed=line_is_reversed, part=part)
    ThisLM.update_metadata(lhc_collimation_metadata)
    print(f"Loss map created in {time.time()-lm_time:.1f}s.")

    ThisLM.plot(savefig=f'lossmap_B{beam}{plane}.pdf', zoom='betatron')
    plt.show()


def main():
    args = parser.parse_args()
    line_path = args.line
    colldb_path = "./example_data/CollDB_tight_150mm.yaml"

    if "hllhc_b1" in line_path:
        beam = 1
    elif "hllhc_b2" in line_path:
        beam = 2
    else:
        raise ValueError("Wrong line, beam cannot be determined!")

    track_lm(line_path, colldb_path, beam, "H", 200, 50000)
    track_lm(line_path, colldb_path, beam, "V", 200, 50000)


if __name__ == "__main__":
    main()