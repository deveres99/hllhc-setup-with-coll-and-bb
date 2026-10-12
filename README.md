# hllhc-setup-with-coll-and-bb
A small example repo to show how to set up HL-LHC simulations with both collimation and beam-beam in xsuite.

## Requriements to run
To run examples from this repo you need a python environment with the classic scientific computing packages, and xsuite 
(with all the core packages: xobjects, xdeps, xtrack, xpart, xfields, xcoll). The code in the filling_scheme_inspector 
folder here is a copy of the bb_toolbox by Matteo Rufolo. I was forced to do this, as the package is un-installable in 
its current state.

## Example workflow
1) Execute `000_inspect_filling_scheme.py`. This will print the following information:
        Most representative bunch family in B1 with 1044/2760 bunches:
                # of LR in ATLAS/CMS: 59
                # of LR in ALICE: 59
                # of LR in LHCB: 59
        -- example bunches: [169 170 171 172 173 174 175 176 177 178] 

        Most representative bunch family in B2 with 1044/2760 bunches:
                # of LR in ATLAS/CMS: 59
                # of LR in ALICE: 59
                # of LR in LHCB: 59
        -- example bunches: [166 167 168 169 170 171 172 173 174 175]
It will also show a plot of the population of different bunch families as well as their number of LR encounters in each 
IP. Based on this, pick a bunch for each beam to be used.

2) Execute `001_create_line.sh` (or the command therein) with the desired inputs
This bash script runs the `make_line_with_apertures_and_bb.py` python script with some example inputs.
In general the python script needs the following inputs:
Positional:
- `isthick`: must be 0 (i.e. thin) if you want BB
- `acc_models_tag`: the name of the tag on the acc-models gitlab to pull the optics from, e.g. `vhl19.5`
- `optics_path`: path to optics file within strengths folder on the acc-models gitlab, 
  e.g. `cycle_round_v0/opt_150_thin.madx` for the baseline optics without the new IR7 optics at 150mm beta* (i.e. end 
  of levelling), or `cycle_round_v3/opt_150_thin.madx` to have the same with new IR7 optics included
- `betastar`: value of beta* in mm for the given optics, e.g. `150`
Named and required:
- `imo`: Landau octupole current in A (Attention! You must set this to the desired value, you can't just make a line 
  with 0 octupole current and then set it later in the simulation, because it produces several % beta change around the 
  ring and thus makes the BB lenses no longer compatible with the optics)
- `qx`: horizontal tune of the bare machine without BB, but with the octupole current turned on, the BB will move this
- `qy`: vertical tune of the bare machine without BB, but with the octupole current turned on, the BB will move this
- `dqx`: horizontal chroma of the bare machine without BB, but with the octupole current turned on, the BB will move this
- `dqy`: vertical chroma of the bare machine without BB, but with the octupole current turned on, the BB will move this
Named and optional (only needed if BB is to be installed):
- `filling_scheme`: path to a json file containing the filling scheme (see for example the file in the `example_data` 
  folder)
- `bunch`: bunch no. in B1 for which the strength of BB lenses should be computed
- `bunch_b2`: bunch no. in B2 for which the strength of BB lenses should be computed, if not specified will be same as 
  `bunch`
- `intensity`: bunch intensity to use to calculate strength of BB lenses
- `ips`: list of numbers corresponding to IPs where BB lenses should be installed (any permutation is accepted), 
  default is `1258`
- `sep2_sigma`: half separation in IP2 in units of sigma, default is `0` (i.e. IP2 head-on)
- `sep8_sigma`: half separation in IP8 in units of sigma, default is `0` (i.e. IP8 head-on)
ATTENTION: orbit effect from BB is not included!

3) Execute `002_inspect_line.py`
This is just a quick test that the line is valid and the installed BB makes sense.
This script takes the path to a line as input and produces 3 pop-up plots:
- Plot of betas and orbit around ring with and without BB
- Plot of relative change in betas and orbit around ring with BB turned on vs with BB turned off
- Tune footprint with and without BB

4) Execute `003_example_lossmap.py`
This is just a quick test that the line is compatible with collimation.
This script takes the path to a line as input and produces 2 saved plots:
- Horizontal lossmap with pencil on primary
- Vertical lossmap with pencil on primary