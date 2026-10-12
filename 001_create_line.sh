#!/bin/bash

python make_line_with_apertures_and_bb.py 0 vhl19.5 cycle_round_v0/opt_150_thin.madx 150 --filling_scheme example_data/25ns_2760b_2748_2492_2574_288bpi_13inj_800ns_bs200ns.json --bunch 169 --bunch_b2 170 --intensity 1.1e11 --ips 1258 --sep2_sigma 1.8 --sep8_sigma 0.96 --i_mo -300 --qx 62.31 --qy 60.32 --dqx 15 --dqy 15