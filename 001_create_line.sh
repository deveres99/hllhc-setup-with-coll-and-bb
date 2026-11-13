#!/bin/bash

# Most representative bunch, HO in all IPs
python make_line_with_apertures_and_bb.py 0 vhl19.7 cycle_round_v0 levelling 14 21000 15 --on_sep2h 0.14143767799901877 --on_sep2v 0.0 --on_sep8h -0.018488092951750332 --on_sep8v 0.014364431134589522 --filling_scheme example_data/25ns_2760b_2748_2492_2574_288bpi_13inj_800ns_bs200ns.json --bunch 150 --bunch_b2 150 --correction_bunch 150 --correction_bunch_b2 150 --intensity 1.2e11 --ips 1258 --i_mo -300 --qx 62.317 --qy 60.322 --dqx 5 --dqy 5