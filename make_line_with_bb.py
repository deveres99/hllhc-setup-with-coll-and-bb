import argparse
import json
import re
import numpy as np

import xtrack as xt

from xingKnobs import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "isthick",
    type=int,
    help="0 if thin, 1 if thick"
)
parser.add_argument(
    "acc_models_tag",
    type=str,
    help="tag for acc-models to load from"
)
parser.add_argument(
    "optics_path",
    type=str,
    help="path to folder containing the optics file within strengths folder on acc-models"
)
parser.add_argument(
    "part_of_cycle",
    type=str,
    help="name of the part of the cycle, e.g. ramp, collapse, lumiramp, levelling"
)
parser.add_argument(
    "step_in_part",
    type=int,
    help="number of the step in the specific part of the cycle"
)
parser.add_argument(
    "time_step_in_part",
    type=int,
    help="time [s] of the step in the specific part of the cycle"
)
parser.add_argument(
    "betastar",
    type=str,
    help="value of beta* in cm in ATLAS and CMS"
)

parser.add_argument(
    "--betastar_lhcb",
    type=str,
    default=150, 
    help="value of beta* in cm in LHCb"
)
parser.add_argument(
    "--on_sep2h",
    type=float,
    default=None,
    help="horizontal separation knob in IP2"
)
parser.add_argument(
    "--on_sep2v",
    type=float,
    default=None,
    help="vertical separation knob in IP2"
)
parser.add_argument(
    "--on_sep8h",
    type=float,
    default=None,
    help="horizontal separation knob in IP8"
)
parser.add_argument(
    "--on_sep8v",
    type=float,
    default=None,
    help="vertical separation knob in IP8"
)

# Beam-beam inputs: only used (and then required) if beta* <= 98 cm, i.e. in collision
parser.add_argument(
    "--filling_scheme",
    type=str,
    help="path to filling scheme json file (with 'beam1' and 'beam2' arrays)"
)
parser.add_argument(
    "--bunch",
    type=int,
    help="index of the bunch to simulate (used for both beams unless --bunch_b2 is given)"
)
parser.add_argument(
    "--bunch_b2",
    type=int,
    default=None,
    help="index of the bunch to simulate for beam 2 (default: same as --bunch)"
)
parser.add_argument(
    "--correction_bunch",
    type=int,
    default=None,
    help="index of the bunch in beam 1 on which the IP orbit is corrected back to the "
         "reference with beam-beam on (default: same as --bunch)"
)
parser.add_argument(
    "--correction_bunch_b2",
    type=int,
    default=None,
    help="index of the bunch in beam 2 on which the IP orbit is corrected back to the "
         "reference with beam-beam on (default: same as --correction_bunch if given, "
         "otherwise the simulated bunch of beam 2)"
)
parser.add_argument(
    "--intensity",
    type=float,
    help="bunch intensity in number of particles per bunch, e.g. 2.2e11"
)
parser.add_argument(
    "--ips",
    type=str,
    default="1258",
    help="IPs with beam-beam, as a string of IP numbers in any order, e.g. 1258 or 15 (default: 1258)"
)

# Machine settings, used with and without beam-beam. Tunes and chromaticities
# are matched with the octupoles on, before installing beam-beam.
parser.add_argument(
    "--i_mo",
    type=float,
    required=True,
    help="Landau octupole current in A, same for both beams, e.g. -300"
)
parser.add_argument(
    "--qx",
    type=float,
    required=True,
    help="horizontal tune, e.g. 62.31"
)
parser.add_argument(
    "--qy",
    type=float,
    required=True,
    help="vertical tune, e.g. 60.32"
)
parser.add_argument(
    "--dqx",
    type=float,
    required=True,
    help="horizontal chromaticity, e.g. 15"
)
parser.add_argument(
    "--dqy",
    type=float,
    required=True,
    help="vertical chromaticity, e.g. 15"
)

# Beam-beam parameters (same as in studies/template_jobs config.yaml)
NEMITT_X = 2.5e-6
NEMITT_Y = 2.5e-6
SIGMA_Z = 0.076
NUM_SLICES_HEAD_ON = 11
NUM_LONG_RANGE_ENCOUNTERS_PER_SIDE = {"ip1": 25, "ip2": 18, "ip5": 25, "ip8": 18}
# Bunch slot of beam 2 colliding with bunch 0 of beam 1 at each IP
DELAY_AT_IPS_SLOTS = {"ip1": 0, "ip2": 891, "ip5": 0, "ip8": 2670}
BUNCH_SPACING_BUCKETS = 10
HARMONIC_NUMBER = 35640
# Keep the closed orbit change from the beam-beam kicks (weak-strong: the
# other beam stays on its orbit without beam-beam). xfields removes it by default.
BB_ORBIT_EFFECT = True
# Collimator placeholders (drifts) that beam-beam lenses must not be installed into
COLLIMATOR_PLACEHOLDERS = "(tc|td).*"
# Minimum distance in m of the beam-beam lenses from the placeholder edges, as
# xcoll removes elements sitting at the edges when installing the collimators
COLLIMATOR_MARGIN = 1e-3


def parse_ips(ips):
    # "1258", "51", ... -> ["ip1", "ip5"], ... (always ordered as ip1, ip2, ip5, ip8)
    if len(ips) == 0 or not set(ips) <= set("1258") or len(set(ips)) != len(ips):
        parser.error(f"--ips must be a non-empty combination of the digits 1, 2, 5, 8 without repetitions, got '{ips}'")
    return [f"ip{ii}" for ii in "1258" if ii in ips]


def extract_lines(lhc, beam):
    # Move both lines to a new environment, dropping the /b1, /b2 suffixes of
    # the line to be saved (as in make_line_with_apertures.py). The other line
    # keeps its suffixes, so there are no name clashes, and is only needed as
    # the counter-rotating beam for the beam-beam configuration.
    rename_shared = {}
    for name in lhc[f"b{beam}"].element_names:
        if f"/b{beam}" in name:
            rename_shared[name] = name.replace(f"/b{beam}", "")

    other = 2 if int(beam) == 1 else 1

    env = xt.Environment()
    env.import_line(lhc[f"b{beam}"].copy(), line_name=f"b{beam}", rename_elements=rename_shared)
    env.import_line(lhc[f"b{other}"].copy(), line_name=f"b{other}")

    env.b1.twiss_default["method"] = "6d"
    env.b2.twiss_default["method"] = "6d"
    env.b2.twiss_default["reverse"] = False

    return env, env[f"b{beam}"]


def collimator_table(line):
    tt = line.get_table().rows[COLLIMATOR_PLACEHOLDERS]
    return tt.rows[tt.element_type == "Drift"]


def install_beam_beam(env, ip_names, intensity):
    # Lines start at IP1, so lenses left of IP1 would land at negative s:
    # install from IP3 and cycle back to keep s consistent with the aperture files
    start_names = {nn: env[nn].element_names[0] for nn in ["b1", "b2"]}
    env.b1.cycle("ip3")
    env.b2.cycle("ip3")

    collimators = {nn: collimator_table(env[nn]) for nn in ["b1", "b2"]}

    env.xfields.install_beambeam_interactions(
        clockwise_line="b1",
        anticlockwise_line="b2",
        ip_names=ip_names,
        num_long_range_encounters_per_side=NUM_LONG_RANGE_ENCOUNTERS_PER_SIDE,
        num_slices_head_on=NUM_SLICES_HEAD_ON,
        harmonic_number=HARMONIC_NUMBER,
        bunch_spacing_buckets=BUNCH_SPACING_BUCKETS,
        sigmaz=SIGMA_Z,
        delay_at_ips_slots=[DELAY_AT_IPS_SLOTS[ip] for ip in ip_names],
    )

    # No lens may be inside a collimator placeholder or within COLLIMATOR_MARGIN
    # of its edges, otherwise the collimator cannot be installed (e.g. with xcoll)
    for nn in ["b1", "b2"]:
        lenses = env[nn].get_table().rows["bb_.*"]
        tt = collimators[nn]
        clashing = [cc for cc, s0, s1 in zip(tt.name, tt.s_start, tt.s_end)
                    if np.any((lenses.s > s0 - COLLIMATOR_MARGIN) & (lenses.s < s1 + COLLIMATOR_MARGIN))]
        assert not clashing, (f"Beam-beam lenses in or too close to collimator placeholders in {nn}: "
                              f"{clashing}, reduce NUM_LONG_RANGE_ENCOUNTERS_PER_SIDE")

    env.b1.cycle(start_names["b1"])
    env.b2.cycle(start_names["b2"])

    env.xfields.configure_beambeam_interactions(
        num_particles=intensity,
        nemitt_x=NEMITT_X,
        nemitt_y=NEMITT_Y,
    )

    # configure_beambeam_interactions stores the kick of each lens on the closed
    # orbit without beam-beam in post_subtract_*, which the lens subtracts from
    # every particle, so that beam-beam leaves the closed orbit unchanged.
    # Setting them to zero restores the orbit effect.
    if BB_ORBIT_EFFECT:
        for ll in [env.b1, env.b2]:
            for name in ll.get_table().rows.match("BeamBeam.*", col="element_type").name:
                for kk in ["x", "px", "y", "py", "zeta", "pzeta"]:
                    if hasattr(ll[name], f"post_subtract_{kk}"):
                        setattr(ll[name], f"post_subtract_{kk}", 0)


def ip_orbit_vary_and_targets(beam, tw_ref):
    # x, y at the IPs back to the reference tw_ref with the van der Meer knobs
    # on_[x,y]ip[1,2,5,8]b[1,2]
    ips = ["ip1", "ip2", "ip5", "ip8"]
    vary = xt.VaryList([f"on_{pp}ip{ip[-1]}b{beam}" for ip in ips for pp in "xy"], step=1e-4) # in mm
    targets = [xt.TargetSet(["x", "y"], value=tw_ref, at=ip, tol=1e-8) for ip in ips]
    return vary, targets


def match_tune_chroma_and_ip_orbit(line, beam, tw_ref, qx, qy, chroma_x, chroma_y):
    # Tunes and chromaticities with dq[x,y].b[1,2] and dqp[x,y].b[1,2], and
    # the IP orbit, as the octupole feed-down from the orbit bumps changes all of them.
    # The default step is too small for the chromaticity knobs (does not converge)
    vary_ip, targets_ip = ip_orbit_vary_and_targets(beam, tw_ref)
    line.match(
        vary=[xt.VaryList([f"dqx.b{beam}", f"dqy.b{beam}"], step=1e-5),
              xt.VaryList([f"dqpx.b{beam}", f"dqpy.b{beam}"], step=1e-2),
              vary_ip],
        targets=[xt.TargetSet(qx=qx, qy=qy, tol=1e-5),
                 xt.TargetSet(dqx=chroma_x, dqy=chroma_y, tol=1e-2)]
                + targets_ip,
    )


def match_ip_orbit(line, beam, tw_ref):
    vary, targets = ip_orbit_vary_and_targets(beam, tw_ref)
    line.match(vary=vary, targets=targets)


def print_twiss(line, title):
    tw = line.twiss()
    print("-------------------------------------------------------------------")
    print(title)
    print(f"Qx={tw.qx}, Qy={tw.qy}")
    print(f"dQx={tw.dqx}, dQy={tw.dqy}")
    print(tw.rows["ip[1258]"].cols["betx bety x px y py"])
    print("-------------------------------------------------------------------")


def main():
    args = parser.parse_args()
    isthick = args.isthick
    acc_models_tag = args.acc_models_tag
    optics_path = args.optics_path
    part_of_cycle = args.part_of_cycle
    step_in_part = args.step_in_part
    time_step_in_part = args.time_step_in_part
    betastar = args.betastar
    betastar_lhcb = args.betastar_lhcb

    # No collisions at injection and flat-top (beams separated), so beam-beam
    # is only installed for beta* <= 98 cm
    with_bb = float(betastar) <= 98
    if with_bb:
        missing = [nn for nn in ["filling_scheme", "bunch", "intensity"]
                   if getattr(args, nn) is None]
        if missing:
            parser.error(f"beam-beam is installed for beta*={betastar} mm, "
                         f"so these options are required: {', '.join('--' + nn for nn in missing)}")
        ip_names = parse_ips(args.ips)
        bunch_b1 = args.bunch
        bunch_b2 = args.bunch if args.bunch_b2 is None else args.bunch_b2
        corr_bunch_b1 = bunch_b1 if args.correction_bunch is None else args.correction_bunch
        corr_bunch_b2 = (args.correction_bunch_b2 if args.correction_bunch_b2 is not None
                         else bunch_b2 if args.correction_bunch is None else args.correction_bunch)
        with open(args.filling_scheme, "r") as fid:
            filling = json.load(fid)
    else:
        print(f"beta*={betastar} mm > 98 cm: no beam-beam installed, beam-beam inputs are ignored")

    if isthick:
        json_url = f"https://gitlab.cern.ch/acc-models/acc-models-lhc/-/raw/{acc_models_tag}/xsuite/lhc.json"
        optics_url = f"https://gitlab.cern.ch/acc-models/acc-models-lhc/-/raw/{acc_models_tag}/strengths/{optics_path}/{part_of_cycle}/{part_of_cycle}_A{betastar}cmC{betastar}cmA10mL{betastar_lhcb}cm_{step_in_part}.madx"
    else:
        json_url = f"https://gitlab.cern.ch/acc-models/acc-models-lhc/-/raw/{acc_models_tag}/xsuite/lhc_thin.json"
        optics_url = f"https://gitlab.cern.ch/acc-models/acc-models-lhc/-/raw/{acc_models_tag}/strengths/{optics_path}/{part_of_cycle}/{part_of_cycle}_A{betastar}cmC{betastar}cmA10mL{betastar_lhcb}cm_{step_in_part}_thin.madx"
    lhc = xt.load(json_url)
    lhc.vars.load(optics_url)
    settings_url = f"https://acc-models.web.cern.ch/acc-models/lhc/{acc_models_tag}/scenarios/cycle/pp/{part_of_cycle}/{time_step_in_part}/settings.madx"
    lhc.vars.load(settings_url)
    lhc.new_particle("beam1", mass0=938272046.0, q0=1, p0c="p0c")
    lhc.b1.set_particle_ref("beam1")
    lhc.new_particle("beam2", mass0=938272046.0, q0=1, p0c="p0c")
    lhc.b2.set_particle_ref("beam2")

    print("Loaded line:", json_url)
    print("Loaded optics:", optics_url)
    print("Loaded settings:", settings_url)

    # IP2 and 8 custom separation
    for nn in ["on_sep2h", "on_sep2v", "on_sep8h", "on_sep8v"]:
        if getattr(args, nn) is None:
            continue
        lhc[nn] = getattr(args, nn)

    lhc.b2.twiss_default["reverse"] = False
    for beam in [1, 2]:
        lhc[f"b{beam}"].twiss_default["method"] = "6d"

    for beam in [1, 2]:
        print_twiss(lhc[f"b{beam}"], f"Beam{beam} without beam-beam and apertures as loaded from acc-models:")

    # Reference orbit at the IPs, before changing the octupole current
    tw_ref = {beam: lhc[f"b{beam}"].twiss() for beam in [1, 2]}

    # The orbit bumps through the octupoles give a tune, chromaticity and IP
    # orbit change from feed-down, which is compensated by the matching below
    lhc["i_mo.b1"] = args.i_mo
    lhc["i_mo.b2"] = args.i_mo
    print(f"Octupole current: i_mo.b1 = i_mo.b2 = {args.i_mo} A")

    for beam in [1, 2]:
        print_twiss(lhc[f"b{beam}"], f"Beam{beam} without beam-beam and apertures after MO change:")

    # Tunes, chromaticities and IP orbit, with the octupoles on and before beam-beam
    for beam in [1, 2]:
        match_tune_chroma_and_ip_orbit(lhc[f"b{beam}"], beam, tw_ref[beam],
                                       args.qx, args.qy, args.dqx, args.dqy)

    for beam in [1, 2]:
        print_twiss(lhc[f"b{beam}"], f"Beam{beam} without beam-beam and apertures after correction:")

    # ---------------------------------------------------------------------------
    # Beam-beam and apertures, per beam, and save
    # ---------------------------------------------------------------------------
    for beam in [1, 2]:
        env, line = extract_lines(lhc, beam)

        if with_bb:
            install_beam_beam(env, ip_names, args.intensity)

            # Emulate the operators correcting the IP orbit back to the reference
            # on the signal of a selected bunch (no bunch-by-bunch correction):
            # correct with the beam-beam pattern of the correction bunches, then
            # switch to the pattern of the simulated bunches
            if BB_ORBIT_EFFECT:
                env.xfields.apply_filling_pattern(
                    filling_pattern_cw=filling["beam1"],
                    filling_pattern_acw=filling["beam2"],
                    i_bunch_cw=corr_bunch_b1,
                    i_bunch_acw=corr_bunch_b2,
                )
                print_twiss(line, f"Beam{beam} with beam-beam of the correction bunch "
                                  f"(b1={corr_bunch_b1}, b2={corr_bunch_b2}) before IP orbit correction:")
                match_ip_orbit(line, beam, tw_ref[beam])
                print_twiss(line, f"Beam{beam} with beam-beam of the correction bunch "
                                  f"(b1={corr_bunch_b1}, b2={corr_bunch_b2}) after IP orbit correction:")

            env.xfields.apply_filling_pattern(
                filling_pattern_cw=filling["beam1"],
                filling_pattern_acw=filling["beam2"],
                i_bunch_cw=bunch_b1,
                i_bunch_acw=bunch_b2,
            )
            print_twiss(line, f"Beam{beam} with beam-beam in {', '.join(ip_names)} "
                              f"(bunch b1={bunch_b1}, b2={bunch_b2}, intensity={args.intensity:.3g}):")

        print_twiss(line, f"Beam{beam} with {'beam-beam' if with_bb else ''}:")

        imo = line[f"i_mo.b{beam}"]
        save_name = f"hllhc_b{beam}_noaper_{acc_models_tag.replace('.', 'p')}_{optics_path}_{part_of_cycle}{step_in_part}_{betastar}cm_qx{int(round(args.qx*1e3)-62000)}qy{int(round(args.qy*1e3)-60000)}_dqx{round(args.dqx)}dqy{round(args.dqy)}_imo{int(imo)}"
        if with_bb:
            bunch = bunch_b1 if beam == 1 else bunch_b2
            save_name += f"_bb_ip{''.join(ip[-1] for ip in ip_names)}_bunch{bunch}"
        line.to_json(save_name + ".json")
        print("Saved:", save_name + ".json")


if __name__ == "__main__":
    main()
