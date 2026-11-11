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
parser.add_argument(
    "--sep2_sigma",
    type=float,
    default=0.0,
    help="half separation in IP2 in sigma"
)
parser.add_argument(
    "--sep8_sigma",
    type=float,
    default=0.0,
    help="half separation in IP8 in sigma"
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
# Collimator placeholders (drifts) that beam-beam lenses must not be installed into
COLLIMATOR_PLACEHOLDERS = "(tc|td).*"
# Minimum distance in m of the beam-beam lenses from the placeholder edges, as
# xcoll removes elements sitting at the edges when installing the collimators
COLLIMATOR_MARGIN = 1e-3
# Aperture model, only available on the hl19 branch for now
APERTURE_URL = "https://gitlab.cern.ch/acc-models/acc-models-lhc/-/raw/hl19/aperture/aperture.json"
# Spacing in m of the regular grid of aperture limits (on top of the ones at
# the aperture transitions and around the elements). Sets how well the aperture
# interpolated between the limits (in loss map refinement and plots) follows
# the model: 5 m gives a p99 error of ~0.08 mm, below the polygon discretisation
APERTURE_STEP = 5.0
# LHCb VELO apertures (not in the aperture model), in m from IP8 along beam 1
VELO_S_FROM_IP8 = {"velo.a_aper": -0.8, "velo.b_aper": -0.4, "velo.c_aper": -0.2, "velo.d_aper": 0.4}


def check_aperture(line):
    tt = line.get_table()

    excluded_element_type = "|Marker|Limit.*|.*Drift.*"
    checked_indices = tt.rows.indices.match_not(
        excluded_element_type, col="element_type"
    )

    violations = []

    elements_in_need = []
    aper_up_to_insert = []
    aper_do_to_insert = []

    for ii in checked_indices:
        name = tt.name[ii]
        element_type = tt.element_type[ii]

        previous_name = "<START>"
        previous_type = "<START>"
        if ii > 0:
            previous_name = tt.name[ii - 1]
            previous_type = tt.element_type[ii - 1]

        next_name = "<END>"
        next_type = "<END>"
        if ii + 1 < len(tt):
            next_name = tt.name[ii + 1]
            next_type = tt.element_type[ii + 1]

        has_limit_before = previous_type.startswith("Limit")
        has_limit_after = next_type.startswith("Limit")

        if not (has_limit_before and has_limit_after):

            # Find closest Limit before
            closest_limit_before = "<START>"
            for jj in range(ii - 1, -1, -1):
                if tt.element_type[jj].startswith("Limit"):
                    closest_limit_before = tt.name[jj]
                    break

            # Find closest Limit after
            closest_limit_after = "<END>"
            for jj in range(ii + 1, len(tt)):
                if tt.element_type[jj].startswith("Limit"):
                    closest_limit_after = tt.name[jj]
                    break

            violations.append(
                {
                    "row": ii,
                    "name": name,
                    "element_type": element_type,
                    "previous_name": previous_name,
                    "previous_type": previous_type,
                    "next_name": next_name,
                    "next_type": next_type,
                    "closest_limit_before": closest_limit_before,
                    "closest_limit_after": closest_limit_after,
                }
            )

            elements_in_need.append(name)
            aper_up_to_insert.append(closest_limit_before)
            aper_do_to_insert.append(closest_limit_after)

    if violations:
        print()
        print("Elements without a Limit.* element immediately before and after:")
        for vv in violations:
            print(
                f"row={vv['row']} "
                f"name={vv['name']} "
                f"type={vv['element_type']} "
                f"previous=({vv['previous_name']}, {vv['previous_type']}) "
                f"next=({vv['next_name']}, {vv['next_type']}) "
                f"closest_limit_before={vv['closest_limit_before']} "
                f"closest_limit_after={vv['closest_limit_after']}"
            )
    else:
        print("All checked elements have Limit.* neighbors immediately before and after.")

    return elements_in_need, aper_up_to_insert, aper_do_to_insert


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


def install_beam_beam(env, ip_names, intensity, filling, bunch_b1, bunch_b2):
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

    env.xfields.apply_filling_pattern(
        filling_pattern_cw=filling["beam1"],
        filling_pattern_acw=filling["beam2"],
        i_bunch_cw=bunch_b1,
        i_bunch_acw=bunch_b2,
    )


def load_aperture_model(acc_models_tag):
    # The aperture model is defined on the thick lattice (pipes are placed
    # w.r.t. thick element names), so it is always built on the thick lattice,
    # also when the thin one is used: s is the same in both
    aperture_env = xt.load(f"https://gitlab.cern.ch/acc-models/acc-models-lhc/-/raw/{acc_models_tag}/xsuite/lhc.json")
    aperture_env.aperture.from_json(APERTURE_URL, timeout=60)
    print("Loaded aperture model:", APERTURE_URL)
    return aperture_env


def install_apertures(env, line, beam, betastar, aperture):
    tt = line.get_table()

    # Limits directly upstream and downstream of all elements other than drifts
    # and markers, and of the collimator placeholders (used by xcoll when
    # installing the collimators). For thin elements the downstream limit is a
    # replica of the upstream one, except for collimators as xcoll needs actual
    # limit elements.
    need = tt.rows.match_not("|Marker|Limit.*|.*Drift.*", col="element_type")
    colls = collimator_table(line)
    need_names = np.concatenate([need.name, colls.name])
    s_start = np.concatenate([need.s_start, colls.s_start])
    s_end = np.concatenate([need.s_end, colls.s_end])

    # Sampling for the loss location granularity: around the aperture transitions
    # and on a regular grid, only inside drifts that are not collimator placeholders
    s_grid = aperture.get_s_around_transitions(resolution=APERTURE_STEP)
    drifts = tt.rows.match(".*Drift.*", col="element_type")
    drifts = drifts.rows[~np.isin(drifts.name, colls.name)]
    ii = np.searchsorted(drifts.s_start, s_grid, side="right") - 1
    s_grid = s_grid[(ii >= 0) & (s_grid > drifts.s_start[ii] + aperture.s_tol)
                    & (s_grid < drifts.s_end[ii] - aperture.s_tol)]

    limits = aperture.get_limit_elements_at_s(np.unique(np.concatenate([s_start, s_end, s_grid])))

    apers_to_insert_at_name = []
    for name, s0, s1 in zip(need_names, s_start, s_end):
        env.elements[f"{name}_aper_up"] = limits[s0]
        if s1 == s0 and name not in colls.name:
            env.new(f"{name}_aper_do", "Replica", parent_name=f"{name}_aper_up")
        else:
            env.elements[f"{name}_aper_do"] = limits[s1]
        apers_to_insert_at_name.append(env.place(f"{name}_aper_up", at=0, from_=f"{name}@start"))
        apers_to_insert_at_name.append(env.place(f"{name}_aper_do", at=0, from_=f"{name}@end"))

    apers_to_insert_at_s = []
    for ii, s in enumerate(s_grid):
        env.elements[f"aper.{ii}"] = limits[s]
        apers_to_insert_at_s.append(env.place(f"aper.{ii}", at=s))

    # The LHCb VELO is not in the aperture model
    sign = 1 if int(beam) == 1 else -1
    for velo, ds in VELO_S_FROM_IP8.items():
        env.elements[velo] = xt.LimitEllipse(a=0.0029, b=0.0029)
        apers_to_insert_at_s.append(env.place(velo, at=tt["s", "ip8"] + sign * ds))

    line.insert(apers_to_insert_at_name, s_tol=1e-6)
    line.insert(apers_to_insert_at_s, s_tol=1e-6)

    tw = line.twiss()

    env["velo_at_parking"] = 1 if float(betastar) > 98 else 0
    velo_shift_x = tw.x[tw.name == "ip8"][0]
    velo_shift_y = tw.y[tw.name == "ip8"][0]
    env["velo_shift_x"] = f"0*velo_at_parking+({velo_shift_x})*(1-velo_at_parking)"
    env["velo_shift_y"] = f"0*velo_at_parking+({velo_shift_y})*(1-velo_at_parking)"
    env["velo_aper_a"] = f"0.0232*velo_at_parking+0.0029*(1-velo_at_parking)"
    env["velo_aper_b"] = f"0.0232*velo_at_parking+0.0029*(1-velo_at_parking)"
    for velo in ["velo.a_aper", "velo.b_aper", "velo.c_aper", "velo.d_aper"]:
        line.element_refs[velo].a = env.vars["velo_aper_a"]
        line.element_refs[velo].b = env.vars["velo_aper_b"]
        line.element_refs[velo].shift_x = env.vars["velo_shift_x"]
        line.element_refs[velo].shift_y = env.vars["velo_shift_y"]

    _ = check_aperture(line)


def trim_tune_and_chroma(env, beam, qx, qy, chroma_x, chroma_y,
                         tol_q=1e-5, tol_dq=1e-2, max_iter=10):
    # dq[x,y].b[1,2] and dqp[x,y].b[1,2] are deltas on top of the current tunes
    # and chromaticities. Iterate, as tune and chromaticity are coupled e.g. by
    # the octupole feed-down from the orbit bumps.
    for _ in range(max_iter):
        tw = env[f"b{beam}"].twiss()
        delta = {f"dqx.b{beam}": qx - tw.qx, f"dqy.b{beam}": qy - tw.qy,
                 f"dqpx.b{beam}": chroma_x - tw.dqx, f"dqpy.b{beam}": chroma_y - tw.dqy}
        if all(abs(vv) < (tol_dq if kk.startswith("dqp") else tol_q)
               for kk, vv in delta.items()):
            return
        for kk, vv in delta.items():
            env[kk] += vv
    print(f"WARNING: tune/chromaticity trim of beam {beam} not converged, "
          f"remaining deltas: {delta}")


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
        with open(args.filling_scheme, "r") as fid:
            filling = json.load(fid)
    else:
        print(f"beta*={betastar} mm > 98 cm: no beam-beam installed, beam-beam inputs are ignored")

    if isthick:
        json_url = f"https://gitlab.cern.ch/acc-models/acc-models-lhc/-/raw/{acc_models_tag}/xsuite/lhc.json"
    else:
        json_url = f"https://gitlab.cern.ch/acc-models/acc-models-lhc/-/raw/{acc_models_tag}/xsuite/lhc_thin.json"
    lhc = xt.load(json_url)
    optics_url = f"https://gitlab.cern.ch/acc-models/acc-models-lhc/-/raw/{acc_models_tag}/strengths/{optics_path}/{part_of_cycle}/{part_of_cycle}_A{betastar}cmC{betastar}cmA10mL{betastar_lhcb}cm_{step_in_part}.madx"
    lhc.vars.load(optics_url)
    settings_url = f"https://gitlab.cern.ch/acc-models/acc-models-lhc/{acc_models_tag}/scenarios/cycle/pp/{part_of_cycle}/{time_step_in_part}/settings.madx"
    lhc.vars.load(settings_url)
    lhc.new_particle("beam1", mass0=938272046.0, q0=1, p0c="p0c")
    lhc.b1.set_particle_ref("beam1")
    lhc.new_particle("beam2", mass0=938272046.0, q0=1, p0c="p0c")
    lhc.b2.set_particle_ref("beam2")

    print("Loaded line:", json_url)
    print("Loaded optics:", optics_url)
    print("Loaded settings:", settings_url)

    aperture_env = load_aperture_model(acc_models_tag)

    # Octupoles first: the orbit bumps through them give a tune shift from
    # feed-down, which is compensated by the tune/chromaticity trim below
    lhc["i_mo.b1"] = args.i_mo
    lhc["i_mo.b2"] = args.i_mo
    print(f"Octupole current: i_mo.b1 = i_mo.b2 = {args.i_mo} A")

    lhc.b2.twiss_default["reverse"] = False
    for beam in [1, 2]:
        lhc[f"b{beam}"].twiss_default["method"] = "6d"

    # Tunes and chromaticities, with the octupoles on and before beam-beam
    for beam in [1, 2]:
        trim_tune_and_chroma(lhc, beam, args.qx, args.qy, args.dqx, args.dqy)

    for beam in [1, 2]:
        print_twiss(lhc[f"b{beam}"], f"Beam{beam} without beam-beam and apertures:")

    # ---------------------------------------------------------------------------
    # Beam-beam and apertures, per beam, and save
    # ---------------------------------------------------------------------------
    for beam in [1, 2]:
        env, line = extract_lines(lhc, beam)

        if with_bb:
            install_beam_beam(env, ip_names, args.intensity, filling, bunch_b1, bunch_b2)
            print_twiss(line, f"Beam{beam} with beam-beam in {', '.join(ip_names)} "
                              f"(bunch b1={bunch_b1}, b2={bunch_b2}, intensity={args.intensity:.3g}):")

        install_apertures(env, line, beam, betastar, aperture_env[f"b{beam}"].aperture)
        print_twiss(line, f"Beam{beam} with {'beam-beam and ' if with_bb else ''}apertures:")

        imo = line[f"i_mo.b{beam}"]
        save_name = f"hllhc_b{beam}_{acc_models_tag.replace('.', 'p')}_{'_'.join(re.split('/|.madx', optics_path)[:-1])}_qx{int(round(args.qx*1e3)-62000)}qy{int(round(args.qy*1e3)-60000)}_dqx{round(args.dqx)}dqy{round(args.dqy)}_imo{int(imo)}"
        if with_bb:
            bunch = bunch_b1 if beam == 1 else bunch_b2
            save_name += f"_bb_ip{''.join(ip[-1] for ip in ip_names)}_bunch{bunch}"
        line.to_json(save_name + ".json")
        print("Saved:", save_name + ".json")


if __name__ == "__main__":
    main()
