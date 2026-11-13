# hllhc-setup-with-coll-and-bb
A small example repo to show how to set up HL-LHC simulations with both collimation and beam-beam in xsuite.

## Requriements to run
To run examples from this repo you need a python environment with the classic scientific computing packages, and xsuite 
(with all the core packages: xobjects, xdeps, xtrack, xpart, xfields, xcoll). The code in the filling_scheme_inspector 
folder here is a copy of the bb_toolbox by Matteo Rufolo. I was forced to do this, as the package is un-installable in 
its current state.

## Example workflow
1) Execute `000_inspect_filling_scheme.py`. This will print the following information:
        Most representative bunch family in B1 with 692/2760 bunches:
                # of LR in ATLAS/CMS: 50
                # of LR in ALICE: 36
                # of LR in LHCB: 36
        -- example bunches: [142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157] 

        Most representative bunch family in B2 with 692/2760 bunches:
                # of LR in ATLAS/CMS: 50
                # of LR in ALICE: 36
                # of LR in LHCB: 36
        -- example bunches: [139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154]
It will also show a plot of the population of different bunch families as well as their number of LR encounters in each 
IP. Based on this, pick a bunch for each beam to be used. The bunch numbers are slot indices in the filling scheme 
(0-3563), which is also what `make_line_with_apertures_and_bb.py` expects.

2) Execute `001_create_line.sh` (or the commands therein) with the desired inputs
This bash script runs the `make_line_with_apertures_and_bb.py` python script with some example inputs.
In general the python script needs the following inputs:
Positional:
- `isthick`: must be 0 (i.e. thin) if you want BB
- `acc_models_tag`: the name of the tag on the acc-models gitlab to pull the line, optics and settings from, e.g. 
  `vhl19.7`
- `optics_path`: folder of the optics within the strengths folder on the acc-models gitlab, e.g. `cycle_round_v0` for 
  the baseline optics without the new IR7 optics, or `cycle_round_v3` to have the same with new IR7 optics included
- `part_of_cycle`: name of the part of the cycle, e.g. `ramp`, `collapse`, `lumiramp`, `levelling`
- `step_in_part`: number of the optics step in that part of the cycle, e.g. `14`
- `time_step_in_part`: time in s of the step in that part of the cycle, used to load the machine settings (crossing 
  angles, separations, etc.) from `scenarios/cycle/pp/<part_of_cycle>/<time_step_in_part>/settings.madx`, e.g. `21000`
- `betastar`: value of beta* in cm in ATLAS and CMS for the given optics, e.g. `15`. BB is only installed if this is 
  at most 98 cm (i.e. in collision)

  The optics file loaded is 
  `strengths/<optics_path>/<part_of_cycle>/<part_of_cycle>_A<betastar>cmC<betastar>cmA10mL<betastar_lhcb>cm_<step_in_part>_thin.madx` 
  (without `_thin` for the thick line), e.g. `cycle_round_v0/levelling/levelling_A15cmC15cmA10mL150cm_14_thin.madx` for 
  the end of levelling.
Named and required:
- `i_mo`: Landau octupole current in A, same for both beams (Attention! You must set this to the desired value, you 
  can't just make a line with 0 octupole current and then set it later in the simulation, because it produces several % 
  beta change around the ring and thus makes the BB lenses no longer compatible with the optics)
- `qx`: horizontal tune of the bare machine without BB, but with the octupole current turned on, the BB will move this
- `qy`: vertical tune of the bare machine without BB, but with the octupole current turned on, the BB will move this
- `dqx`: horizontal chroma of the bare machine without BB, but with the octupole current turned on, the BB will move this
- `dqy`: vertical chroma of the bare machine without BB, but with the octupole current turned on, the BB will move this

  After the octupole current is set, the tunes, chromas and the `x`, `y` orbit at IP1, 2, 5 and 8 are matched back 
  (the orbit to its value before the octupole current change, using the van der Meer knobs `on_xip<n>b<beam>`, 
  `on_yip<n>b<beam>`).
Named and optional:
- `betastar_lhcb`: value of beta* in cm in LHCb, default is `150`
- `on_sep2h`, `on_sep2v`, `on_sep8h`, `on_sep8v`: horizontal and vertical separation knobs in IP2 and IP8, i.e. the 
  offset of each beam at the IP in mm (the two beams move in opposite directions, so the full separation is twice the 
  value). Each one that is given overrides the value from the loaded optics and settings; the ones not given are left as loaded. They are set right after loading the optics and settings, before 
  the reference orbit is taken, so the IP orbit matching keeps them.
Named and optional (only needed if BB is to be installed):
- `filling_scheme`: path to a json file containing the filling scheme (see for example the file in the `example_data` 
  folder)
- `bunch`: bunch no. in B1 for which the BB lenses should be switched on according to the filling scheme
- `bunch_b2`: bunch no. in B2 for which the BB lenses should be switched on according to the filling scheme, if not 
  specified will be same as `bunch`
- `correction_bunch`: bunch no. in B1 on which the IP orbit is corrected (see below), if not specified will be same as 
  `bunch`
- `correction_bunch_b2`: bunch no. in B2 on which the IP orbit is corrected (see below), if not specified will be same 
  as `correction_bunch` if that is given, otherwise same as `bunch_b2`

  All bunch numbers are slot indices in the filling scheme (0-3563, one slot per 25 ns), not the count of filled 
  bunches, i.e. the same numbering as the example bunches printed by `000_inspect_filling_scheme.py`. The slot must be 
  filled in the corresponding beam, otherwise the script stops with `AssertionError: Selected bunch is not in the 
  filling scheme`. Note that B1 and B2 are filled in different slots (e.g. in the example filling scheme B1 starts at 
  slot 26 and B2 at slot 14).
- `intensity`: bunch intensity to use to calculate strength of BB lenses
- `ips`: list of numbers corresponding to IPs where BB lenses should be installed (any permutation is accepted), 
  default is `1258`

The separations in IP2 and IP8 come from the machine settings of the selected step, unless overridden with `on_sep2h`, 
`on_sep2v`, `on_sep8h` and `on_sep8v` (see above).

Orbit effect from BB: by default (`BB_ORBIT_EFFECT = True` in the script) the orbit effect of the BB kicks is included 
in a weak-strong way (the other beam stays on its orbit without BB). To emulate an operator correcting the IP orbit on 
the signal of a selected bunch (there is no bunch-by-bunch orbit correction), the `x`, `y` orbit at IP1, 2, 5 and 8 is 
first matched back to the reference with the BB lenses of `correction_bunch` (`correction_bunch_b2`) switched on, and 
then the BB lenses of `bunch` (`bunch_b2`) are switched on. A bunch with a different LR pattern than the correction 
bunch (e.g. at the end of a train) will thus have a small residual IP orbit offset. Pick a bunch in the middle of a 
train (most representative family from step 1) as correction bunch. With `BB_ORBIT_EFFECT = False` the BB does not 
change the closed orbit and no correction is done.

The number of LR encounters per side installed at each IP is set by `NUM_LONG_RANGE_ENCOUNTERS_PER_SIDE` in the script 
(25 in IP1 and IP5, 18 in IP2 and IP8, so that no BB lens is installed in a collimator placeholder drift).

3) Execute `002_inspect_line.py`
This is just a quick test that the line is valid and the installed BB makes sense.
This script takes the path to a line as input and produces 3 pop-up plots:
- Plot of betas and orbit around ring with and without BB
- Plot of relative change in betas and change of orbit in sigmas around ring with BB turned on vs with BB turned off
- Tune footprint with and without BB

4) Execute `003_example_lossmap.py`
This is just a quick test that the line is compatible with collimation.
This script takes the path to a line as input and produces 2 saved plots:
- Horizontal lossmap with pencil on primary
- Vertical lossmap with pencil on primary