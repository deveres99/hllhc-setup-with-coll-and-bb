xingKnobs = \
    {"150": 
    {
            # Crossing angles [murad]
            "on_x1hs": 250, "on_x5vs": 250, "on_x2v": 170, "on_x8v": -200,
    
            # Beam separations at crossing point [mm]
            "on_sep1v": "-0.55 * on_sep1", "on_sep5h": "0.55 * on_sep5", "on_sep2h": "0.143 * on_sep2", "on_sep8h": "0.043 * on_sep8",
    
            # Solenoid switches [0/1]
            "on_sol_atlas": 1, "on_sol_alice": 1, "on_sol_cms": 1,
    
            # Spectrometer dipoles in ALICE and LHCb [0,1; 1 is for 7 TeV beam]
            "on_alice": 1, "on_lhcb": 1,
    
            # Add crossing angle in non-nominal plane, keep at 0 [murad]
            #"on_a1": 0, "on_a2": 0, "on_a5": 0, "on_a8": 0,
    
            # Add offset - plane depends on phi_IRx
            #"on_o1": 0, "on_o2": 0, "on_o5": 0, "on_o8": 0,
    
            # Crab cavity switches [murad]
            "on_crab1": -190, "on_crab5": -190,
    
            # Switch for arc orbit correctors (dispersion), leave at 1 [0/1]
            "on_dsep1v":"on_sep1v * on_sep1", "on_dx1hs": "on_x1hs + on_x1hl", "on_dsep5h": "on_sep5h * on_sep5", "on_dx5vs": "on_x5vs + on_x5vl",
        },
    "200":
    {
            # Crossing angles [murad]
            "on_x1hs": 250, "on_x5vs": 250, "on_x2v": 170, "on_x8v": -200,
    
            # Beam separations at crossing point [mm]
            "on_sep1v": "-0.55 * on_sep1", "on_sep5h": "0.55 * on_sep5", "on_sep2h": "0.143 * on_sep2", "on_sep8h": "0.043 * on_sep8",
    
            # Solenoid switches [0/1]
            "on_sol_atlas": 1, "on_sol_alice": 1, "on_sol_cms": 1,
    
            # Spectrometer dipoles in ALICE and LHCb [0,1; 1 is for 7 TeV beam]
            "on_alice": 1, "on_lhcb": 1,
    
            # Add crossing angle in non-nominal plane, keep at 0 [murad]
            #"on_a1": 0, "on_a2": 0, "on_a5": 0, "on_a8": 0,
    
            # Add offset - plane depends on phi_IRx
            #"on_o1": 0, "on_o2": 0, "on_o5": 0, "on_o8": 0,
    
            # Crab cavity switches [murad]
            "on_crab1": -190, "on_crab5": -190,
    
            # Switch for arc orbit correctors (dispersion), leave at 1 [0/1]
            "on_dsep1v":"on_sep1v * on_sep1", "on_dx1hs": "on_x1hs + on_x1hl", "on_dsep5h": "on_sep5h * on_sep5", "on_dx5vs": "on_x5vs + on_x5vl",
        },
    "640":
    {
            # Crossing angles [murad]
            "on_x1hs": 250, "on_x5vs": 250, "on_x2v": 170, "on_x8v": 200,
    
            # Beam separations at crossing point [mm]
            "on_sep1v": "-0.55 * on_sep1", "on_sep5h": "0.55 * on_sep5", "on_sep2h": "0.143 * on_sep2", "on_sep8h": "0.043 * on_sep8",
    
            # Solenoid switches [0/1]
            "on_sol_atlas": 1, "on_sol_alice": 1, "on_sol_cms": 1,
    
            # Spectrometer dipoles in ALICE and LHCb [0,1; 1 is for 7 TeV beam]
            "on_alice": 1, "on_lhcb": 1,
    
            # Add crossing angle in non-nominal plane, keep at 0 [murad]
            #"on_a1": 0, "on_a2": 0, "on_a5": 0, "on_a8": 0,
    
            # Add offset - plane depends on phi_IRx
            #"on_o1": 0, "on_o2": 0, "on_o5": 0, "on_o8": 0,
    
            # Crab cavity switches [murad]
            "on_crab1": -190, "on_crab5": -190,
    
            # Switch for arc orbit correctors (dispersion), leave at 1 [0/1]
            "on_dsep1v":"on_sep1v * on_sep1", "on_dx1hs": "on_x1hs + on_x1hl", "on_dsep5h": "on_sep5h * on_sep5", "on_dx5vs": "on_x5vs + on_x5vl",
        },
    "1000":
    {
            # Crossing angles [murad]
            "on_x1hs": 250, "on_x5vs": 250, "on_x2v": 170, "on_x8v": 200,
    
            # Beam separations at crossing point [mm]
            "on_sep1v": "-0.55 * on_sep1", "on_sep5h": "0.55 * on_sep5", "on_sep2h": "0.143 * on_sep2", "on_sep8h": "0.043 * on_sep8",
    
            # Solenoid switches [0/1]
            "on_sol_atlas": 1, "on_sol_alice": 1, "on_sol_cms": 1,
    
            # Spectrometer dipoles in ALICE and LHCb [0,1; 1 is for 7 TeV beam]
            "on_alice": 1, "on_lhcb": 1,
    
            # Add crossing angle in non-nominal plane, keep at 0 [murad]
            #"on_a1": 0, "on_a2": 0, "on_a5": 0, "on_a8": 0,
    
            # Add offset - plane depends on phi_IRx
            #"on_o1": 0, "on_o2": 0, "on_o5": 0, "on_o8": 0,
    
            # Crab cavity switches [murad]
            "on_crab1": -190, "on_crab5": -190,
    
            # Switch for arc orbit correctors (dispersion), leave at 1 [0/1]
            "on_dsep1v":"on_sep1v * on_sep1", "on_dx1hs": "on_x1hs + on_x1hl", "on_dsep5h": "on_sep5h * on_sep5", "on_dx5vs": "on_x5vs + on_x5vl",
        },
    "6000": # injection? TODO -- cross-check energy and xing angle for IP8
    {
            # Crossing angles [murad]
            "on_x1hs": 500, "on_x5vs": 500, "on_x2v": 170, "on_x8v": 170,
    
            # Beam separations at crossing point [mm]
            "on_sep1v": "2.00 * on_sep1", "on_sep5h": "-2.00 * on_sep5", "on_sep2h": "3.50 * on_sep2", "on_sep8h": "3.50 * on_sep8",
    
            # Solenoid switches [0/1]
            "on_sol_atlas": 1, "on_sol_alice": 1, "on_sol_cms": 1,
    
            # Spectrometer dipoles in ALICE and LHCb [0,1]
            "on_alice": 1, "on_lhcb": 1,
    
            # Add crossing angle in non-nominal plane, keep at 0 [murad]
            #"on_a1": 0, "on_a2": 0, "on_a5": 0, "on_a8": 0,
    
            # Add offset - plane depends on phi_IRx
            #"on_o1": 0, "on_o2": 0, "on_o5": 0, "on_o8": 0,
    
            # Crab cavity switches [murad]
            "on_crab1": 0, "on_crab5": 0,
    
            # Switch for arc orbit correctors (dispersion), leave at 1 [0/1]
            "on_dsep1v": 0, "on_dx1hs": 0, "on_dsep5h": 0, "on_dx5vs": 0,
        },      
    }
