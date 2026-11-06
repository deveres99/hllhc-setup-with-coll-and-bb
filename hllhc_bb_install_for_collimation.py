# Minimal example: HL-LHC lines with beam-beam installed such that no lens
# lands inside (and slices) a collimator placeholder drift, so that xcoll can
# later install the collimators in place of the placeholders.
#
# env.xfields.install_beambeam_interactions(...) does not allow moving
# individual lenses, so install_beam_beam() below reproduces it with these
# xfields internals, moving the lenses in between:
#   - config_tools.generate_set_of_bb_encounters_1beam
#   - config_tools.install_dummy_bb_lenses
#   - config_tools._compute_delays
#   - env._bb_config (set by hand, read by configure_beambeam_interactions
#     and apply_filling_pattern)
import json

import xtrack as xt
from xfields.config_tools.beambeam_config_tools import config_tools as bb_tools

from xingKnobs import *

# Settings (as in 001_create_line.sh)
ACC_MODELS_TAG = "vhl19.5"
OPTICS_PATH = "cycle_round_v0/opt_150_thin.madx"
BETASTAR = "150"
FILLING_SCHEME = "example_data/25ns_2760b_2748_2492_2574_288bpi_13inj_800ns_bs200ns.json"
BUNCH_B1 = 169
BUNCH_B2 = 170
INTENSITY = 1.1e11
IP_NAMES = ["ip1", "ip2", "ip5", "ip8"]
SEP2_SIGMA = 1.8
SEP8_SIGMA = 0.96
I_MO = -300
QX = 62.31
QY = 60.32
DQX = 15
DQY = 15

# Beam-beam parameters (same as in studies/template_jobs config.yaml)
NEMITT_X = 2.5e-6
NEMITT_Y = 2.5e-6
SIGMA_Z = 0.076
NUM_SLICES_HEAD_ON = 11
NUM_LONG_RANGE_ENCOUNTERS_PER_SIDE = {"ip1": 25, "ip2": 20, "ip5": 25, "ip8": 20}
# Bunch slot of beam 2 colliding with bunch 0 of beam 1 at each IP
DELAY_AT_IPS_SLOTS = {"ip1": 0, "ip2": 891, "ip5": 0, "ip8": 2670}
BUNCH_SPACING_BUCKETS = 10
HARMONIC_NUMBER = 35640
# Collimator placeholders (drifts) that beam-beam lenses must not be installed into
COLLIMATOR_PLACEHOLDERS = "(tc|td).*"
# Minimum distance in m of the beam-beam lenses from the placeholder edges, as
# xcoll removes elements sitting at the edges when installing the collimators
COLLIMATOR_MARGIN = 1e-3


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


def collimator_at(tt, s):
    # Collimator placeholder drift containing s, or closer than COLLIMATOR_MARGIN/2
    # to its edges (half, so that lenses moved to COLLIMATOR_MARGIN outside the
    # edges are safely clear of it), as (name, s_start, s_end)
    for name, s0, s1 in zip(tt.name, tt.s_start, tt.s_end):
        if s0 - COLLIMATOR_MARGIN / 2 < s < s1 + COLLIMATOR_MARGIN / 2:
            return name, s0, s1
    return None


def move_lenses_out_of_collimators(env, bb_dfs):
    # Long-range lenses falling inside a collimator placeholder would slice it,
    # so that the collimator can no longer be installed (e.g. with xcoll).
    # Such lenses are moved to just outside the closest edge of the placeholder. Both
    # partner lenses are moved by the same amount, so that they stay at the
    # same physical location. atPosition is w.r.t. the IP, along b1 for the b1
    # line and along b4 (opposite sign) for the b2 line.
    tt = {nn: collimator_table(env[nn]) for nn in ["b1", "b2"]}
    s_ip = {nn: {ip: env[nn].get_table()["s", ip] for ip in set(bb_dfs[nn].ip_name)}
            for nn in ["b1", "b2"]}
    sign = {"b1": 1, "b2": -1}

    df_b1 = bb_dfs["b1"]
    for name in df_b1.index[df_b1.label == "bb_lr"]:
        partner = df_b1.loc[name, "other_elementName"]
        ip = df_b1.loc[name, "ip_name"]
        d = df_b1.loc[name, "atPosition"]

        def clashes(d):
            out = []
            for nn in ["b1", "b2"]:
                coll = collimator_at(tt[nn], s_ip[nn][ip] + sign[nn] * d)
                if coll is not None:
                    out.append((nn, coll))
            return out

        found = clashes(d)
        if not found:
            continue

        # Candidate positions: just outside the edges of the clashing placeholders, closest first
        candidates = sorted({sign[nn] * (s_edge - s_ip[nn][ip])
                             for nn, (_, s0, s1) in found
                             for s_edge in (s0 - COLLIMATOR_MARGIN, s1 + COLLIMATOR_MARGIN)},
                            key=lambda dd: abs(dd - d))
        d_new = next((dd for dd in candidates if not clashes(dd)), None)
        if d_new is None:
            raise ValueError(f"Cannot move {name} out of {[coll[0] for _, coll in found]}")

        print(f"Moving {name} and {partner} by {d_new - d:+.3f} m "
              f"out of {', '.join(sorted(set(coll[0] for _, coll in found)))}")
        bb_dfs["b1"].loc[name, "atPosition"] = d_new
        bb_dfs["b2"].loc[partner, "atPosition"] = -d_new


def install_beam_beam(env, ip_names, intensity, filling, bunch_b1, bunch_b2):
    # Lines start at IP1, so lenses left of IP1 would land at negative s:
    # install from IP3 and cycle back to keep the original s
    start_names = {nn: env[nn].element_names[0] for nn in ["b1", "b2"]}
    env.b1.cycle("ip3")
    env.b2.cycle("ip3")

    # Same as env.xfields.install_beambeam_interactions(...), but with the
    # lenses moved out of the collimator placeholders before inserting them
    bb_dfs = {}
    for nn, other in [("b1", "b2"), ("b2", "b1")]:
        bb_dfs[nn] = bb_tools.generate_set_of_bb_encounters_1beam(
            circumference=env[nn].get_length(),
            harmonic_number=HARMONIC_NUMBER,
            bunch_spacing_buckets=BUNCH_SPACING_BUCKETS,
            numberOfHOSlices=NUM_SLICES_HEAD_ON,
            bunch_particle_charge=env[nn].particle_ref.q0,
            sigt=SIGMA_Z,
            relativistic_beta=env[nn].particle_ref.beta0[0],
            ip_names=ip_names,
            numberOfLRPerIRSide=[NUM_LONG_RANGE_ENCOUNTERS_PER_SIDE[ip] for ip in ip_names],
            beam_name=nn,
            other_beam_name=other,
        )
    bb_dfs["b2"]["atPosition"] = -bb_dfs["b2"]["atPosition"] # b2 line is in the b4 frame

    move_lenses_out_of_collimators(env, bb_dfs)

    keep_columns = ["beam", "other_beam", "ip_name", "elementName", "other_elementName", "label",
                    "self_particle_charge", "self_relativistic_beta", "self_frac_of_bunch",
                    "identifier", "s_crab"]
    for nn in ["b1", "b2"]:
        collimators = collimator_table(env[nn]).name
        bb_tools.install_dummy_bb_lenses(bb_df=bb_dfs[nn], line=env[nn])
        bb_dfs[nn] = bb_dfs[nn][keep_columns].copy()
        sliced = [cc for cc in collimators if cc not in env[nn].element_names]
        assert not sliced, f"Beam-beam lenses sliced collimator placeholders in {nn}: {sliced}"

    bb_tools._compute_delays(bb_dfs["b1"], bb_dfs["b2"],
                             [DELAY_AT_IPS_SLOTS[ip] for ip in ip_names], ip_names,
                             HARMONIC_NUMBER, BUNCH_SPACING_BUCKETS)

    env._bb_config = {
        "dataframes": {"clockwise": bb_dfs["b1"], "anticlockwise": bb_dfs["b2"]},
        "ip_names": ip_names,
        "clockwise_line": "b1",
        "anticlockwise_line": "b2",
        "bunch_spacing_buckets": BUNCH_SPACING_BUCKETS,
        "harmonic_number": HARMONIC_NUMBER,
    }

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
    with open(FILLING_SCHEME, "r") as fid:
        filling = json.load(fid)

    json_url = f"https://gitlab.cern.ch/acc-models/acc-models-lhc/-/raw/{ACC_MODELS_TAG}/xsuite/lhc_thin.json"
    lhc = xt.load(json_url)
    optics_url = f"https://gitlab.cern.ch/acc-models/acc-models-lhc/-/raw/{ACC_MODELS_TAG}/strengths/{OPTICS_PATH}"
    lhc.vars.load(optics_url)

    print("Loaded line:", json_url)
    print("Loaded optics:", optics_url)

    # Octupoles first: the orbit bumps through them give a tune shift from
    # feed-down, which is compensated by the tune/chromaticity trim below
    lhc["i_mo.b1"] = I_MO
    lhc["i_mo.b2"] = I_MO
    print(f"Octupole current: i_mo.b1 = i_mo.b2 = {I_MO} A")


    # ---------------------------------------------------------------------------
    # Machine settings (same as make_line_with_apertures.py, for both beams)
    # ---------------------------------------------------------------------------
    lhc.b2.twiss_default["reverse"] = False
    for beam in [1, 2]:
        lhc[f"b{beam}"].twiss_default["method"] = "6d"

    lhc["on_sep1"] = 0
    lhc["on_sep2"] = 0
    lhc["on_sep5"] = 0
    lhc["on_sep8"] = 0
    for key, val in xingKnobs[BETASTAR].items():
        lhc[key] = val

    # Get beam size at IP2 and IP8 and set separation according to user input
    tw = lhc.b1.twiss()
    beam_sizes = tw.get_beam_covariance(nemitt_x=NEMITT_X, nemitt_y=NEMITT_Y,
                                        gemitt_zeta=SIGMA_Z**2 / tw.bets0)
    sigma_x_IP2 = beam_sizes.rows["ip2"].sigma_x[0] * 1e3 # in mm
    sigma_x_IP8 = beam_sizes.rows["ip8"].sigma_x[0] * 1e3 # in mm
    lhc["sep2_sigma"] = SEP2_SIGMA
    lhc["sep8_sigma"] = SEP8_SIGMA
    lhc.vars["on_sep2h"] = lhc.vars["on_sep2h"]._expr + sigma_x_IP2 * lhc.vars["sep2_sigma"]
    lhc.vars["on_sep8h"] = lhc.vars["on_sep8h"]._expr + sigma_x_IP8 * lhc.vars["sep8_sigma"]

    # Tunes and chromaticities, with the octupoles on and before beam-beam
    for beam in [1, 2]:
        trim_tune_and_chroma(lhc, beam, QX, QY, DQX, DQY)

    for beam in [1, 2]:
        print_twiss(lhc[f"b{beam}"], f"Beam{beam} without beam-beam:")

    # ---------------------------------------------------------------------------
    # Beam-beam, per beam, and save
    # ---------------------------------------------------------------------------
    for beam in [1, 2]:
        env, line = extract_lines(lhc, beam)

        install_beam_beam(env, IP_NAMES, INTENSITY, filling, BUNCH_B1, BUNCH_B2)
        print_twiss(line, f"Beam{beam} with beam-beam in {', '.join(IP_NAMES)} "
                          f"(bunch b1={BUNCH_B1}, b2={BUNCH_B2}, intensity={INTENSITY:.3g}):")

        save_name = f"hllhc_b{beam}_bb_for_collimation.json"
        line.to_json(save_name)
        print("Saved:", save_name)


if __name__ == "__main__":
    main()
