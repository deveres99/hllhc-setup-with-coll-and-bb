## Copied from https://gitlab.cern.ch/mrufolo/fillingstudies by Matteo Rufolo ##

import numpy as np
from numpy import linalg as LA
import pandas as pd
import copy
import json
from matplotlib import pyplot as plt
import filling_scheme_inspector.tool_box_to_bool as tbtb
import filling_scheme_inspector.bb_toolbox as bbt
import random




def prefill_algorithm_NOINDIV_ALICE(bunches, empty, len_batch, bunches_ns = 25, bunches_tune_shift = 12, bunches_tune_shift_ns = 25, tune_shift_gap = 48, collisions_flag = 0, inj_space = 31):
    '''
    Having as input the number of bunches for each PS_batch, the number of empties for the empty gap
    between PS batches, and how many PS_batches there are inside an SPS_batch, this script gives as output
    the two filling scheme of the two beams with the desired constraints given in input, with the number of collisions 
    in ATLAS/CMS/ALICE maximized.
    Hypothesys:
        - in this script we are considering that the "twelve" bunches for one of the two beams 
        are positioned in 0 and for the other in inj_twelve given as input
        - we are also considering that the ABK is optimized as best posssible in order to contatin perfectly the
        longest SPS batch, and it is completely filled
    Args:
        bunches: number of bunches for every PS_batch
        empty: number of empty slots for every empty gap between PS_batches
        len_batch: maximum of number of PS_batches for every SPS_batch
        bunches_tune_shift: how many bunches, for the first injection for the tune shift, set as 12
        tune_shift_gap: the space where you could find all the first bunches for the tune measurement\, set as 48
        inj_space: number of empty slots fot every empty gap between SPS_batches
    Return: 
        np.array(collisions_max); a vector with the number of collisions using the desired filling scheme
        np.array(B1_max): a boolean vector that represent the slots filled by bunches for the desired beam1
        np.array(B2_max): a boolean vector that represent the slots filled by bunches for the desired beam2
    '''
    
    
    collisions =np.array([0,0,0])
    collisions_max1 = np.array([0,0,0])
    collisions_new = np.array([0,0,0])
    B1_max1 = np.zeros(3564)
    B2_max1 = np.zeros(3564)
    B1_max2 = np.zeros(3564)
    B2_max2 = np.zeros(3564)
    B1_new2 = np.zeros(3564)
    B2_new2 = np.zeros(3564)
    B1_new = -np.ones(3564)
    B2_new = -np.ones(3564)
    #B1 = -np.ones(3564)
    #B2 = -np.ones(3564)
    initial_B1 = -np.ones(3564)
    initial_B2 = -np.ones(3564)
    

    Dict_ns_len = {'25ns': 1, '50ns':2, '75ns':3,'100ns':4}
    assert bunches_ns in [25,50,75,100], "bunches_ns accepted in LHC are 25 or 50 or 75 or 100"
    len_ns = Dict_ns_len[f'{bunches_ns}ns']
    assert bunches_tune_shift_ns in [25,50,75,100], "bunches_tune_shift_ns accepted in LHC are 25 or 50 or 75 or 100"
    len_ns_measure_tune = Dict_ns_len[f'{bunches_tune_shift_ns}ns']

    #bunches in ns for tune measurement
    if bunches_tune_shift!=0 and tune_shift_gap !=0:
        bunch_in_ns_measure_tune = np.zeros(len_ns_measure_tune)
        bunch_in_ns_measure_tune[0] = 1
        bunches_in_ns_measure_tune = np.zeros((bunches_tune_shift-1)*(1+(len_ns_measure_tune-1))+1)
        for i in range(bunches_tune_shift-1):
            bunches_in_ns_measure_tune[(i*len_ns):((i+1)*len_ns)] = bunch_in_ns_measure_tune
        bunches_in_ns_measure_tune[-1] = 1
        if collisions_flag == 0 and bunches_tune_shift_ns == 25:
            assert tune_shift_gap>=2*len(bunches_in_ns_measure_tune), "there is not enough space in the tune shift gap to fill the bunches"
        elif collisions_flag == 0 and bunches_tune_shift_ns > 25:
            assert tune_shift_gap>=len(bunches_in_ns_measure_tune)+1, "there is not enough space in the tune shift gap to fill the bunches"
        elif collisions_flag == 1 and bunches_tune_shift_ns >= 25:
            assert tune_shift_gap>=len(bunches_in_ns_measure_tune), "there is not enough space in the tune shift gap to fill the bunches"

    #bunch in the ns 
    bunch_in_ns = np.zeros(len_ns)
    bunch_in_ns[0] = 1
    #PS batch in ns, that finish with one, because there is later the inj_space
    PS_batch_in_ns = np.zeros((bunches-1)*(1+(len_ns-1))+1)
    for i in range(bunches-1):
        PS_batch_in_ns[(i*len_ns):((i+1)*len_ns)] = bunch_in_ns
    PS_batch_in_ns[-1] = 1
    # vector that represent the batches trains, in counts the inj space only at the end!!
    vec = [[]]
    vec[0] = np.zeros((bunches-1)*(1+(len_ns-1))+1+inj_space)
    vec[0][0:(bunches-1)*(1+(len_ns-1))+1] = PS_batch_in_ns
    for j in np.arange(len_batch-1)+2:#-np.arange(len_batch)):
        v= np.zeros((j-1)*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1)+(inj_space))
        for i in np.arange(j):
            v[(i)*(bunches*(1+(len_ns-1))+empty-len_ns+1):(i)*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1)] = PS_batch_in_ns
        vec.append(v)

    

    # fill the final part, considering the AB and ABK 
    slot = 3564-121-((len_batch-1)*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1))
    initial_B1[slot:3564-121+inj_space] = vec[len_batch-1]
    initial_B2[slot:3564-121+inj_space] = vec[len_batch-1]
    slot = 3564-121-2*((len_batch-1)*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1))-inj_space
    #fill the most possible the filling scheme with the biggest trains respecting the maximum given as input by thr user
    i = 1
    while slot > 2673-121+31:
        initial_B1[slot:slot+((len_batch-1)*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1))+inj_space] = vec[len_batch-1]
        initial_B2[slot:slot+((len_batch-1)*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1))+inj_space] = vec[len_batch-1]
        slot_New = slot
        i += 1
        slot = 3564-120-(i+1)*((len_batch-1)*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1))-i*inj_space-1
    
    #adding the bunches for the tune measurement 
    last_bunch_measure_tune = 0
    if bunches_tune_shift!=0 and tune_shift_gap !=0:
        for i in range(tune_shift_gap-len(bunches_in_ns_measure_tune)+1):
            initial2_B1 = copy.copy(initial_B1)
            if i!=0:
                initial2_B1[0:i] = np.zeros(i)
                B1_new[0:i] = np.zeros(i)
            initial2_B1[i:i+len(bunches_in_ns_measure_tune)] = bunches_in_ns_measure_tune
            B1_new[i:i+len(bunches_in_ns_measure_tune)] = bunches_in_ns_measure_tune
            if i != tune_shift_gap-len(bunches_in_ns_measure_tune):
                initial2_B1[i+len(bunches_in_ns_measure_tune):tune_shift_gap] = np.zeros(tune_shift_gap-(i+len(bunches_in_ns_measure_tune)))
                B1_new[i+len(bunches_in_ns_measure_tune):tune_shift_gap] = np.zeros(tune_shift_gap-(i+len(bunches_in_ns_measure_tune)))
            for m in range(tune_shift_gap-len(bunches_in_ns_measure_tune)+1):
                B1 = copy.copy(initial2_B1)
                B2 = copy.copy(initial_B2)
                exit_flag_next_m = 0
                if collisions_flag == 1:
                    if i!=0:
                        B2[0:i] = np.zeros(i)
                        B2_new[0:i] = np.zeros(i)
                    B2[i:i+len(bunches_in_ns_measure_tune)] = bunches_in_ns_measure_tune
                    B2_new[i:i+len(bunches_in_ns_measure_tune)] = bunches_in_ns_measure_tune
                    if i != tune_shift_gap-len(bunches_in_ns_measure_tune):
                        B2[i+len(bunches_in_ns_measure_tune):tune_shift_gap] = np.zeros(tune_shift_gap-(i+len(bunches_in_ns_measure_tune)))
                        B2_new[i+len(bunches_in_ns_measure_tune):tune_shift_gap] = np.zeros(tune_shift_gap-(i+len(bunches_in_ns_measure_tune)))
                    last_bunch_measure_tune = i+len(bunches_in_ns_measure_tune)
                else:
                    if m in np.linspace(i-len(bunches_in_ns_measure_tune),i+len(bunches_in_ns_measure_tune),num = 2*len(bunches_in_ns_measure_tune)+1):
                        if np.abs(m-i)%len_ns_measure_tune != 0:
                            if m!=0:
                                B2[0:m] = np.zeros(m)
                                B2_new[0:m] = np.zeros(m)
                            B2[m:m+len(bunches_in_ns_measure_tune)] = bunches_in_ns_measure_tune
                            B2_new[m:m+len(bunches_in_ns_measure_tune)] = bunches_in_ns_measure_tune
                            if m != tune_shift_gap-len(bunches_in_ns_measure_tune):
                                B2[m+len(bunches_in_ns_measure_tune):tune_shift_gap] = np.zeros(tune_shift_gap-(m+len(bunches_in_ns_measure_tune)))
                                B2_new[m+len(bunches_in_ns_measure_tune):tune_shift_gap] = np.zeros(tune_shift_gap-(m+len(bunches_in_ns_measure_tune)))
                            last_bunch_measure_tune = max(i,m)+len(bunches_in_ns_measure_tune)
                        else:
                            exit_flag_next_m = 1
                    else:
                        if m!=0:
                            B2[0:m] = np.zeros(m)
                            B2_new[0:m] = np.zeros(m)
                        B2[m:m+len(bunches_in_ns_measure_tune)] = bunches_in_ns_measure_tune
                        B2_new[m:m+len(bunches_in_ns_measure_tune)] = bunches_in_ns_measure_tune
                        if m != tune_shift_gap-len(bunches_in_ns_measure_tune):
                            B2[m+len(bunches_in_ns_measure_tune):tune_shift_gap] = np.zeros(tune_shift_gap-(m+len(bunches_in_ns_measure_tune)))
                            B2_new[m+len(bunches_in_ns_measure_tune):tune_shift_gap] = np.zeros(tune_shift_gap-(m+len(bunches_in_ns_measure_tune)))
                        last_bunch_measure_tune = max(i,m)+len(bunches_in_ns_measure_tune)
                
                #print(B1[:70])
                if exit_flag_next_m == 0:
                    #copy the last quarter in the first one
                    #PS: Be careful in copying only the part of the last quarter that can fill after the 12 bunches in the first quarter
                    first_slot_where_inject = max(last_bunch_measure_tune, tune_shift_gap-inj_space)
                    if slot_New-2673>first_slot_where_inject+inj_space:
                        slot_New_pos = slot_New
                    else:
                        slot_New_pos = slot_New + (len_batch-1)*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1)+(inj_space)
                        #print(f"uau{B1[slot_New_pos]}")
                        #print(f"uau2{B1[slot_New:slot_New + (len_batch-1)*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1)+(inj_space)+1]}")
                    B1[slot_New_pos-2673:891-121+inj_space] = B1[slot_New_pos:3564-121+inj_space]
                    B2[slot_New_pos-2673:891-121+inj_space] = B2[slot_New_pos:3564-121+inj_space]
                    #print(f"uau3{B1[:slot_New_pos-2673]}")
                    

                    flag = 0
                    for j in (len_batch-1)-np.arange(len_batch-1):
                        frac = ((slot_New)-(2673)+(121)-(2*inj_space))/((j-1)*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1))
                        # check if you can fill the empty space between quarters with a SPS batch of the maximum length possible
                        if frac >1 and flag == 0:
                            print(f"inserted len {j}")
                            # check that shifting the longest trains of some slots, you could insert in the empty space between quarters a SPS batch
                            # with a PS batch more, once entered in this shift, this situation is already preferred because you increase ATLAS/CMS...

                            # POSSIBLE IMPROVEMENT: now you see if shifting of a quantity, you could increase the SPS batch by one, and then stop. But..
                            # you could check also if shifting by >quantity and then position that SPS batch in different positions, considering also the SPS batch after
                            # the twelve that you will insert later, if ALICE increases.
                            max_number_shift = int((slot_New_pos-2673-(first_slot_where_inject+inj_space))/3)
                            check = (slot_New)-(2673)+(121)-(2*inj_space)+(1+np.arange(max_number_shift))>=(j*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1))

                            if check.any():
                                slot_shift = 1+np.arange(max_number_shift)[np.where(check)[0][0]]

                                if (B1[slot_New-2673-3*slot_shift]==-1):
                                    # here you are considering that in the different quarters there are different trains, so if the last train
                                    # is shifted by a quantity "tot" so the train in the previous quarters is shifted by two times "tot" because he is shifted
                                    # but is affected by the shift of the train ahead
                                    init_slots_shift = [slot_New-2673-3*slot_shift,(891)-slot_shift*2,(1782)-slot_shift]
                                    quarter_slot_shifted = (2673)-slot_shift-121+inj_space
                                    B1_new[init_slots_shift[0]:891-3*slot_shift-121+inj_space] = B1[slot_New_pos:3564-121+inj_space]
                                    B2_new[init_slots_shift[0]:891-3*slot_shift-121+inj_space] = B2[slot_New_pos:3564-121+inj_space]
                                    B1_new[2673-121+inj_space:3564-121+31] = B1[2673-121+inj_space:3564-121+31]
                                    B2_new[2673-121+inj_space:3564-121+31] = B2[2673-121+inj_space:3564-121+31]
                                    B1_new[quarter_slot_shifted:quarter_slot_shifted+(j)*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1)+(inj_space)] = vec[j]
                                    B2_new[quarter_slot_shifted:quarter_slot_shifted+(j)*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1)+(inj_space)] = vec[j]

                                    for k in np.arange(2)+1:
                                        B1_new[init_slots_shift[k]-slot_shift-121+inj_space:(init_slots_shift[k])+891-121+31] = B1_new[quarter_slot_shifted:3564-121+31]
                                        B2_new[init_slots_shift[k]-slot_shift-121+inj_space:(init_slots_shift[k])+891-121+31] = B2_new[quarter_slot_shifted:3564-121+31]
                                        
                                    # (*)
                                    # here we are filling the initial empty space, after the first twelve bunches, checking all the position available to see 
                                    # which one is the best possible in terms of collisions in ALICE and save it
                                    flag_init = 0
                                    # in the following cycle we are considering as maximum SPS batch the one inserted between quarters, would be useless do the cycle
                                    # for all length available because this is limited by the twelve bunches that there aren't in the other quarters and also becasue before 
                                    # we fill the empty space between quarters, here before 0 there is the ABORT GAP where must be neither one bunch, so the space that we can fill
                                    # is "half" of the spaces considered before
                                    for l in (len_batch-1)-np.arange(len_batch-1):
                                        initial_frac = (init_slots_shift[0]-first_slot_where_inject-(2*inj_space))/((l-1)*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1))
                                        if initial_frac >=1 and flag_init == 0:
                                            print(f"inserted in the beggining len {l}")
                                            flag_init = 1
                                            B1_check = copy.copy(B1_new)
                                            B2_check = copy.copy(B2_new)
                                            for k in np.arange(init_slots_shift[0]-inj_space-(first_slot_where_inject+inj_space))+first_slot_where_inject+inj_space:
                                                indec = k
                                                end_indec = indec+(l-1)*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1)
                                                if (B1_check[indec] == -1) and (B1_check[end_indec] == -1) and end_indec<=(init_slots_shift[0]-inj_space):
                                                    B1_new[indec:end_indec+inj_space] = vec[l-1]
                                                    B2_new[indec:end_indec+inj_space] = vec[l-1]                                  
                                                    B1_new[np.where(B1_new==-1)] = 0
                                                    B2_new[np.where(B2_new==-1)] = 0
                                                    collisions_new = tbtb.Head_on(B1_new,B2_new,np.array([0,891,2670]))
                                                    #print(collisions_new)
                                                    if collisions_new[1]>collisions_max1[1]:
                                                        collisions_max1 = collisions_new
                                                        vec_B1 = copy.copy(B1_new)
                                                        vec_B2 = copy.copy(B2_new)
                                                        B1_max1 = vec_B1
                                                        B2_max1 = vec_B2 
                                                    B1_new = copy.copy(B1_check)
                                                    B2_new = copy.copy(B2_check)
                                            
                                    if flag_init == 0:
                                        B1_new[np.where(B1_new==-1)] = 0
                                        B2_new[np.where(B2_new==-1)] = 0
                                        collisions_new = tbtb.Head_on(B1_new,B2_new,np.array([0,891,2670]))
                                        collisions_max1 = collisions_new
                                        B1_max1 = copy.copy(B1_new)
                                        B2_max1 = copy.copy(B2_new)        
                                    # in this configuration you have maximized ATLAS/CMS more than all the other possible solutions, and also with that number 
                                    # of collisions of ATLAS/CMS you have been checked
                                    flag = 2

                            
                            if flag != 2:
                                # now you couldn't increase the number of PS batch inside the SPS batch, so we maximize the the number of collisions in ALICE trying to
                                # insert the SPS batch in the best position possible!
                                gap_frac = (slot_New+1-2673+121-2*inj_space)/((j-1)*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1))
                                flag_gap = 1
                                if gap_frac>=1 and flag_gap ==1:
                                    flag_gap = 1
                                    B1_new2[:] = copy.copy(B1[:])
                                    B2_new2[:] = copy.copy(B2[:])
                                    for k in np.arange(slot_New+1-inj_space-(2673-121+inj_space))+2673-121+inj_space:
                                        indec = k
                                        end_indec = indec+(j-1)*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1)+(inj_space)
                                        if (B1_new2[indec] == -1) and (B1_new2[end_indec] == -1):
                                            B1_new2[indec:indec+(j-1)*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1)+(inj_space)] = vec[j-1]
                                            B2_new2[indec:indec+(j-1)*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1)+(inj_space)] = vec[j-1]
                                        # we are maximizing ALICE, so we are copying and pasting the SPS batches perfectly on quarters of all the ring
                                        for l in [891,1782]:
                                            B1_new2[l-121+inj_space:l+891-121+31] = B1_new2[2673-121+inj_space:3564-121+31]
                                            B2_new2[l-121+inj_space:l+891-121+31] = B2_new2[2673-121+inj_space:3564-121+31]

                                        # (*)
                                        # here we are filling the initial empty space, after the first twelve bunches, checking all the position available to see 
                                        # which one is the best possible in terms of collisions in ALICE and save it
                                        flag_init = 0
                                        # in the following cycle we are considering as maximum SPS batch the one inserted between quarters, would be useless do the cycle
                                        # for all length available because this is limited by the twelve bunches that there aren't in the other quarters and also becasue before 
                                        # we fill the empty space between quarters, here before 0 there is the ABORT GAP where must be neither one bunch, so the space that we can fill
                                        # is "half" of the spaces considered before
                                        for l in (len_batch-1)-np.arange(len_batch-1):
                                            initial_frac = (slot_New_pos-2673-first_slot_where_inject-(2*inj_space))/((l-1)*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1))
                                            
                                            if initial_frac >=1 and flag_init == 0:#
                                                
                                                #print(l)
                                                #print(f"beh{initial_frac}")
                                                #print(f"mah{slot_New_pos}")
                                                #print(first_slot_where_inject)
                                                flag_init = 1
                                                B1_check = copy.copy(B1_new2)
                                                B2_check = copy.copy(B2_new2)
                                                for k in np.arange(slot_New_pos-2673-inj_space-(first_slot_where_inject+inj_space))+first_slot_where_inject+inj_space:
                                                    #print(B1_check[first_slot_where_inject+inj_space:slot_New_pos-2673-inj_space])
                                                    indec = k
                                                    end_indec = indec+(l-1)*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1)
                                                    #print(B1_check[indec])
                                                    #print(B1_check[end_indec])
                                                    if (B1_check[indec] == -1) and (B1_check[end_indec] == -1) and end_indec<=(slot_New_pos-2673-inj_space):
                                                        B1_new2[indec:end_indec+inj_space] = vec[l-1]
                                                        B2_new2[indec:end_indec+inj_space] = vec[l-1]
                                                        B1_new2[np.where(B1_new2==-1)] = 0
                                                        B2_new2[np.where(B2_new2==-1)] = 0
                                                        collisions = tbtb.Head_on(B1_new2,B2_new2,np.array([0,891,2670]))
                                                        #print(collisions[1])
                                                        #print(collisions_max1[1]) # gia salvato da vecchia iterazione, toglilo fai dei copy e cambia tutt!!
                                                        if collisions[1]>collisions_max1[1]:
                                                            collisions_max1 = collisions
                                                            vec_B1 = copy.copy(B1_new2)
                                                            vec_B2 = copy.copy(B2_new2)
                                                            B1_max1 = vec_B1
                                                            B2_max1 = vec_B2  
                                                            print(f"inserted in the beggining len {l}")
                                                        B1_new2 = copy.copy(B1_check)
                                                        B2_new2 = copy.copy(B2_check)
                                                #print(B2[:last_bunch_measure_tune])
                                                
                                        if flag_init == 0:
                                            B1_new2[np.where(B1_new2==-1)] = 0
                                            B2_new2[np.where(B2_new2==-1)] = 0
                                            collisions = tbtb.Head_on(B1_new2,B2_new2,np.array([0,891,2670]))
                                            collisions_max1 = collisions
                                            B1_max1 = copy.copy(B1_new2)
                                            B2_max1 = copy.copy(B2_new2) 
                                    
                                # in this for we are going from the longest SPS batch to the smaller ones, so once you checked that one batch can be inserted,
                                # surely the following steps will be smaller so you would lose in number of collisions in ATLAS/CMS, so you use the flag in order to exit from the cycle
                                flag = 1

                    # this is the case where you couldn't insert any SPS batch between quarters
                    if flag == 0:
                        print(f"not inserted")
                        B1_new2[:] = B1[:]
                        B2_new2[:] = B2[:]
                        for l in [891,1782]:
                            B1_new2[l-121+inj_space:l+891-121+31] = B1_new2[2673-121+inj_space:3564-121+31]
                            B2_new2[l-121+inj_space:l+891-121+31] = B2_new2[2673-121+inj_space:3564-121+31]
                        
                        # (*)
                        # here we are filling the initial empty space, after the first twelve bunches, checking all the position available to see 
                        # which one is the best possible in terms of collisions in ALICE and save it
                        flag_init = 0
                        # in the following cycle we are considering as maximum SPS batch the one inserted between quarters, would be useless do the cycle
                        # for all length available because this is limited by the twelve bunches that there aren't in the other quarters and also becasue before 
                        # we fill the empty space between quarters, here before 0 there is the ABORT GAP where must be neither one bunch, so the space that we can fill
                        # is "half" of the spaces considered before
                        for l in (len_batch-1)-np.arange(len_batch-1):
                            initial_frac = (slot_New_pos-2673-first_slot_where_inject-(2*inj_space))/((l-1)*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1))
                            if initial_frac >=1 and flag_init == 0:
                                print(f"inserted in the beggining len {l}")
                                flag_init = 1
                                B1_check = copy.copy(B1_new2)
                                B2_check = copy.copy(B2_new2)
                                for k in np.arange(slot_New_pos-2673-inj_space-(first_slot_where_inject+inj_space))+first_slot_where_inject+inj_space:
                                    indec = k
                                    end_indec = indec+(l-1)*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1)
                                    if (B1_check[indec] == -1) and (B1_check[end_indec] == -1) and end_indec<=(slot_New_pos-2673-inj_space):
                                        B1_new2[indec:end_indec+inj_space] = vec[l-1]
                                        B2_new2[indec:end_indec+inj_space] = vec[l-1]
                                        B1_new2[np.where(B1_new2==-1)] = 0
                                        B2_new2[np.where(B2_new2==-1)] = 0
                                        collisions = tbtb.Head_on(B1_new2,B2_new2,np.array([0,891,2670]))
                                        if collisions[1]>collisions_max1[1]:
                                            collisions_max1 = collisions
                                            vec_B1 = copy.copy(B1_new2)
                                            vec_B2 = copy.copy(B2_new2)
                                            B1_max1 = vec_B1
                                            B2_max1 = vec_B2
                                        B1_new2 = copy.copy(B1_check)
                                        B2_new2 = copy.copy(B2_check)
                                
                        if flag_init == 0:
                            B1_new2[np.where(B1_new2==-1)] = 0
                            B2_new2[np.where(B2_new2==-1)] = 0
                            collisions = tbtb.Head_on(B1_new2,B2_new2,np.array([0,891,2670]))
                            collisions_max1 = collisions
                            B1_max1 = copy.copy(B1_new2)
                            B2_max1 = copy.copy(B2_new2) 
    else:
        B1 = copy.copy(initial_B1)
        B2 = copy.copy(initial_B2)
        first_slot_where_inject = max(last_bunch_measure_tune, tune_shift_gap-inj_space)
        if slot_New-2673>first_slot_where_inject:
            slot_New_pos = slot_New
        else:
            slot_New_pos = slot_New + (len_batch-1)*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1)+(inj_space)
            #print(f"uau{B1[slot_New_pos]}")
            #print(f"uau2{B1[slot_New:slot_New + (len_batch-1)*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1)+(inj_space)+1]}")
        B1[slot_New_pos-2673:891-121+inj_space] = B1[slot_New_pos:3564-121+inj_space]
        B2[slot_New_pos-2673:891-121+inj_space] = B2[slot_New_pos:3564-121+inj_space]
        #print(f"uau3{B1[:slot_New_pos-2673]}")
        

        flag = 0
        for j in (len_batch-1)-np.arange(len_batch-1):
            frac = ((slot_New)-(2673)+(121)-(2*inj_space))/((j-1)*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1))
            # check if you can fill the empty space between quarters with a SPS batch of the maximum length possible
            if frac >1 and flag == 0:
                print(f"inserted len {j}")
                # check that shifting the longest trains of some slots, you could insert in the empty space between quarters a SPS batch
                # with a PS batch more, once entered in this shift, this situation is already preferred because you increase ATLAS/CMS...

                # POSSIBLE IMPROVEMENT: now you see if shifting of a quantity, you could increase the SPS batch by one, and then stop. But..
                # you could check also if shifting by >quantity and then position that SPS batch in different positions, considering also the SPS batch after
                # the twelve that you will insert later, if ALICE increases.
                max_number_shift = int((slot_New_pos-2673-(first_slot_where_inject))/3)
                check = (slot_New_pos)-(2673)+(121)-(2*inj_space)+(1+np.arange(max_number_shift))>=(j*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1))

                if 0==1:
                    slot_shift = 1+np.arange(max_number_shift)[np.where(check)[0][0]]

                    if (B1[slot_New_pos-2673-3*slot_shift]==-1):
                        # here you are considering that in the different quarters there are different trains, so if the last train
                        # is shifted by a quantity "tot" so the train in the previous quarters is shifted by two times "tot" because he is shifted
                        # but is affected by the shift of the train ahead
                        init_slots_shift = [slot_New_pos-2673-3*slot_shift,(891)-slot_shift*2,(1782)-slot_shift]
                        #print(init_slots_shift)
                        quarter_slot_shifted = (2673)-slot_shift-121+inj_space
                        B1_new[init_slots_shift[0]:891-3*slot_shift-121+inj_space] = B1[slot_New_pos:3564-121+inj_space]
                        B2_new[init_slots_shift[0]:891-3*slot_shift-121+inj_space] = B2[slot_New_pos:3564-121+inj_space]
                        B1_new[2673-121+inj_space:3564-121+31] = B1[2673-121+inj_space:3564-121+31]
                        B2_new[2673-121+inj_space:3564-121+31] = B2[2673-121+inj_space:3564-121+31]
                        B1_new[quarter_slot_shifted:quarter_slot_shifted+(j)*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1)+(inj_space)] = vec[j]
                        B2_new[quarter_slot_shifted:quarter_slot_shifted+(j)*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1)+(inj_space)] = vec[j]

                        for k in np.arange(2)+1:
                            B1_new[init_slots_shift[k]-slot_shift-121+inj_space:(init_slots_shift[k])+891-121+31] = B1_new[quarter_slot_shifted:3564-121+31]
                            B2_new[init_slots_shift[k]-slot_shift-121+inj_space:(init_slots_shift[k])+891-121+31] = B2_new[quarter_slot_shifted:3564-121+31]
                            
                        # (*)
                        # here we are filling the initial empty space, after the first twelve bunches, checking all the position available to see 
                        # which one is the best possible in terms of collisions in ALICE and save it
                        flag_init = 0
                        # in the following cycle we are considering as maximum SPS batch the one inserted between quarters, would be useless do the cycle
                        # for all length available because this is limited by the twelve bunches that there aren't in the other quarters and also becasue before 
                        # we fill the empty space between quarters, here before 0 there is the ABORT GAP where must be neither one bunch, so the space that we can fill
                        # is "half" of the spaces considered before
                        for l in (len_batch-1)-np.arange(len_batch-1):
                            initial_frac = (init_slots_shift[0]-first_slot_where_inject-(inj_space))/((l-1)*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1))
                            #print(initial_frac)
                            #print(l)
                            #print(j)
                            if initial_frac >=1 and flag_init == 0:
                                print(f"inserted in the beggining len {l}")
                                flag_init = 1
                                B1_check = copy.copy(B1_new)
                                B2_check = copy.copy(B2_new)
                                for k in np.arange(init_slots_shift[0]-inj_space-(first_slot_where_inject))+first_slot_where_inject:
                                    indec = k
                                    end_indec = indec+(l-1)*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1)
                                    if (B1_check[indec] == -1) and (B1_check[end_indec] == -1) and end_indec<=(init_slots_shift[0]-inj_space):
                                        B1_new[indec:end_indec+inj_space] = vec[l-1]
                                        B2_new[indec:end_indec+inj_space] = vec[l-1]                                  
                                        B1_new[np.where(B1_new==-1)] = 0
                                        B2_new[np.where(B2_new==-1)] = 0
                                        collisions_new = tbtb.Head_on(B1_new,B2_new,np.array([0,891,2670]))
                                        #print(collisions_new)
                                        if collisions_new[1]>collisions_max1[1]:
                                            collisions_max1 = collisions_new
                                            vec_B1 = copy.copy(B1_new)
                                            vec_B2 = copy.copy(B2_new)
                                            B1_max1 = vec_B1
                                            B2_max1 = vec_B2 
                                        B1_new = copy.copy(B1_check)
                                        B2_new = copy.copy(B2_check)
                                    
                        if flag_init == 0:
                            B1_new[np.where(B1_new==-1)] = 0
                            B2_new[np.where(B2_new==-1)] = 0
                            collisions_new = tbtb.Head_on(B1_new,B2_new,np.array([0,891,2670]))
                            collisions_max1 = collisions_new
                            B1_max1 = copy.copy(B1_new)
                            B2_max1 = copy.copy(B2_new)        
                        # in this configuration you have maximized ATLAS/CMS more than all the other possible solutions, and also with that number 
                        # of collisions of ATLAS/CMS you have been checked
                        flag = 2

                
                if flag != 2:
                    # now you couldn't increase the number of PS batch inside the SPS batch, so we maximize the the number of collisions in ALICE trying to
                    # insert the SPS batch in the best position possible!
                    gap_frac = (slot_New+1-2673+121-2*inj_space)/((j-1)*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1))
                    flag_gap = 1
                    if gap_frac>=1 and flag_gap ==1:
                        flag_gap = 1
                        B1_new2[:] = copy.copy(B1[:])
                        B2_new2[:] = copy.copy(B2[:])
                        for k in np.arange(slot_New+1-inj_space-(2673-121+inj_space))+2673-121+inj_space:
                            indec = k
                            end_indec = indec+(j-1)*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1)+(inj_space)
                            if (B1_new2[indec] == -1) and (B1_new2[end_indec] == -1):
                                B1_new2[indec:indec+(j-1)*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1)+(inj_space)] = vec[j-1]
                                B2_new2[indec:indec+(j-1)*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1)+(inj_space)] = vec[j-1]
                            # we are maximizing ALICE, so we are copying and pasting the SPS batches perfectly on quarters of all the ring
                            for l in [891,1782]:
                                B1_new2[l-121+inj_space:l+891-121+31] = B1_new2[2673-121+inj_space:3564-121+31]
                                B2_new2[l-121+inj_space:l+891-121+31] = B2_new2[2673-121+inj_space:3564-121+31]

                            # (*)
                            # here we are filling the initial empty space, after the first twelve bunches, checking all the position available to see 
                            # which one is the best possible in terms of collisions in ALICE and save it
                            flag_init = 0
                            # in the following cycle we are considering as maximum SPS batch the one inserted between quarters, would be useless do the cycle
                            # for all length available because this is limited by the twelve bunches that there aren't in the other quarters and also becasue before 
                            # we fill the empty space between quarters, here before 0 there is the ABORT GAP where must be neither one bunch, so the space that we can fill
                            # is "half" of the spaces considered before
                            for l in (len_batch-1)-np.arange(len_batch-1):
                                initial_frac = (slot_New_pos-2673-first_slot_where_inject-(inj_space))/((l-1)*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1))
                                
                                if initial_frac >=1 and flag_init == 0:#
                                    
                                    #print(l)
                                    #print(f"beh{initial_frac}")
                                    #print(f"mah{slot_New_pos}")
                                    #print(first_slot_where_inject)
                                    flag_init = 1
                                    B1_check = copy.copy(B1_new2)
                                    B2_check = copy.copy(B2_new2)
                                    for k in np.arange(slot_New_pos-2673-inj_space-(first_slot_where_inject))+first_slot_where_inject:
                                        #print(B1_check[first_slot_where_inject:slot_New_pos-2673-inj_space])
                                        indec = k
                                        end_indec = indec+(l-1)*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1)
                                        #print(B1_check[indec])
                                        #print(B1_check[end_indec])
                                        if (B1_check[indec] == -1) and (B1_check[end_indec] == -1) and end_indec<=(slot_New_pos-2673-inj_space):
                                            B1_new2[indec:end_indec+inj_space] = vec[l-1]
                                            B2_new2[indec:end_indec+inj_space] = vec[l-1]
                                            B1_new2[np.where(B1_new2==-1)] = 0
                                            B2_new2[np.where(B2_new2==-1)] = 0
                                            collisions = tbtb.Head_on(B1_new2,B2_new2,np.array([0,891,2670]))
                                            #print(collisions[1])
                                            #print(collisions_max1[1]) # gia salvato da vecchia iterazione, toglilo fai dei copy e cambia tutt!!
                                            if collisions[1]>collisions_max1[1]:
                                                collisions_max1 = collisions
                                                vec_B1 = copy.copy(B1_new2)
                                                vec_B2 = copy.copy(B2_new2)
                                                B1_max1 = vec_B1
                                                B2_max1 = vec_B2  
                                                print(f"inserted in the beggining len {l}")
                                            B1_new2 = copy.copy(B1_check)
                                            B2_new2 = copy.copy(B2_check)
                                    #print(B2[:last_bunch_measure_tune])
                                    
                            if flag_init == 0:
                                B1_new2[np.where(B1_new2==-1)] = 0
                                B2_new2[np.where(B2_new2==-1)] = 0
                                collisions = tbtb.Head_on(B1_new2,B2_new2,np.array([0,891,2670]))
                                collisions_max1 = collisions
                                B1_max1 = copy.copy(B1_new2)
                                B2_max1 = copy.copy(B2_new2) 
                        
                    # in this for we are going from the longest SPS batch to the smaller ones, so once you checked that one batch can be inserted,
                    # surely the following steps will be smaller so you would lose in number of collisions in ATLAS/CMS, so you use the flag in order to exit from the cycle
                    flag = 1

        # this is the case where you couldn't insert any SPS batch between quarters
        if flag == 0:
            print(f"not inserted")
            B1_new2[:] = B1[:]
            B2_new2[:] = B2[:]
            for l in [891,1782]:
                B1_new2[l-121+inj_space:l+891-121+31] = B1_new2[2673-121+inj_space:3564-121+31]
                B2_new2[l-121+inj_space:l+891-121+31] = B2_new2[2673-121+inj_space:3564-121+31]
            
            # (*)
            # here we are filling the initial empty space, after the first twelve bunches, checking all the position available to see 
            # which one is the best possible in terms of collisions in ALICE and save it
            flag_init = 0
            # in the following cycle we are considering as maximum SPS batch the one inserted between quarters, would be useless do the cycle
            # for all length available because this is limited by the twelve bunches that there aren't in the other quarters and also becasue before 
            # we fill the empty space between quarters, here before 0 there is the ABORT GAP where must be neither one bunch, so the space that we can fill
            # is "half" of the spaces considered before
            for l in (len_batch-1)-np.arange(len_batch-1):
                initial_frac = (slot_New_pos-2673-first_slot_where_inject-(inj_space))/((l-1)*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1))
                if initial_frac >=1 and flag_init == 0:
                    print(f"inserted in the beggining len {l}")
                    flag_init = 1
                    B1_check = copy.copy(B1_new2)
                    B2_check = copy.copy(B2_new2)
                    for k in np.arange(slot_New_pos-2673-inj_space-(first_slot_where_inject))+first_slot_where_inject:
                        indec = k
                        end_indec = indec+(l-1)*(bunches*(1+(len_ns-1))+empty-len_ns+1)+((bunches-1)*(1+(len_ns-1))+1)
                        if (B1_check[indec] == -1) and (B1_check[end_indec] == -1) and end_indec<=(slot_New_pos-2673-inj_space):
                            B1_new2[indec:end_indec+inj_space] = vec[l-1]
                            B2_new2[indec:end_indec+inj_space] = vec[l-1]
                            B1_new2[np.where(B1_new2==-1)] = 0
                            B2_new2[np.where(B2_new2==-1)] = 0
                            collisions = tbtb.Head_on(B1_new2,B2_new2,np.array([0,891,2670]))
                            if collisions[1]>collisions_max1[1]:
                                collisions_max1 = collisions
                                vec_B1 = copy.copy(B1_new2)
                                vec_B2 = copy.copy(B2_new2)
                                B1_max1 = vec_B1
                                B2_max1 = vec_B2
                            B1_new2 = copy.copy(B1_check)
                            B2_new2 = copy.copy(B2_check)
                        
            if flag_init == 0:
                B1_new2[np.where(B1_new2==-1)] = 0
                B2_new2[np.where(B2_new2==-1)] = 0
                collisions = tbtb.Head_on(B1_new2,B2_new2,np.array([0,891,2670]))
                collisions_max1 = collisions
                B1_max1 = copy.copy(B1_new2)
                B2_max1 = copy.copy(B2_new2) 
    #print(flag)
                
    # if collisions_flag == 1:    
    #     # I want that the bunches for tune measurement of B1 and B2 collide in ATLAS/CMS, so i will fill both the beams at the same time
    #     for i in range(tune_shift_gap-len(bunches_in_ns_measure_tune)+1):
    #         if i!=0:
    #             B1_new[0:i] = np.zeros(i)
    #             B2_new[0:i] = np.zeros(i)
    #         B1_new[i:i+len(bunches_in_ns_measure_tune)] = bunches_in_ns_measure_tune
    #         B2_new[i:i+len(bunches_in_ns_measure_tune)] = bunches_in_ns_measure_tune
    #         if i != tune_shift_gap-len(bunches_in_ns_measure_tune):
    #             B1_new[i+len(bunches_in_ns_measure_tune):tune_shift_gap] = bunches_in_ns_measure_tune
    #             B2_new[i+len(bunches_in_ns_measure_tune):tune_shift_gap] = bunches_in_ns_measure_tune
    # elif collisions_flag == 0 and bunches_tune_shift_ns:
    #     # I don't want that the bunches for tune measurement of B1 and B2 collide in ATLAS/CMS, 
    #     # so i will fill firstly B1 seeing all the option and fixed B1 I will fill also B2 in order to have neither 
    #     # one collision in ATLAS and CMS and see all the option and choose the best one
    #     for i in range(tune_shift_gap-len(bunches_in_ns_measure_tune)+1):
    #         if i!=0:
    #             B1_new[0:i] = np.zeros(i)
    #         B1_new[i:i+len(bunches_in_ns_measure_tune)] = bunches_in_ns_measure_tune
    #         if i != tune_shift_gap-len(bunches_in_ns_measure_tune):
    #             B1_new[i+len(bunches_in_ns_measure_tune):tune_shift_gap] = bunches_in_ns_measure_tune
    #         for m in range(tune_shift_gap-len(bunches_in_ns_measure_tune)+1):
    #             # if the m is chosen inside the other initial bunches, but in the empty spaces, you could use that in order to solve
    #             if m in np.linspace(i-len(bunches_in_ns_measure_tune),i+len(bunches_in_ns_measure_tune),num = 2*len(bunch_in_ns_measure_tune)+1):
    #                 if (m-i)%len_ns_measure_tune != 0:
    #                     if m!=0:
    #                         B2_new[0:m] = np.zeros(m)
    #                     B2_new[m:m+len(bunches_in_ns_measure_tune)] = bunches_in_ns_measure_tune
    #                     if m != tune_shift_gap-len(bunches_in_ns_measure_tune):
    #                         B2_new[m+len(bunches_in_ns_measure_tune):tune_shift_gap] = bunches_in_ns_measure_tune
    #             else:
    #                 if m!=0:
    #                     B2_new[0:m] = np.zeros(m)
    #                 B2_new[m:m+len(bunches_in_ns_measure_tune)] = bunches_in_ns_measure_tune
    #                 if m != tune_shift_gap-len(bunches_in_ns_measure_tune):
    #                     B2_new[m+len(bunches_in_ns_measure_tune):tune_shift_gap] = bunches_in_ns_measure_tune
    
    
    


    
    # fill the space in a clever way, either trying to fill the remaining empty space with the maximum train possible
    # but consider that losing some collisions, shifting these longest trains, you could insert a train tha contain one batch more, 
    # and this would lead to a better configuration possible for ATLAS/CMS

    # either here than (*) I use the flag because there are some specific approaches to lead to the best solution, and once done this
    # you wanna exit, because the other approaches would lead to worst solutions!!
    

    # save the best solution possible considering the twelve bunches detached from 0 for both the beams, and see wich configuration is the best possible
    B1_max2 = copy.copy(B1_max1)
    B2_max2 = copy.copy(B2_max1)
    B2_max2[0:tune_shift_gap] = np.zeros(tune_shift_gap)
    B2_max2[tune_shift_gap:tune_shift_gap+bunches_tune_shift] = np.ones(bunches_tune_shift)
    B2_max2[tune_shift_gap+bunches_tune_shift:tune_shift_gap+bunches_tune_shift+inj_space] = np.zeros(inj_space)
    B1_max2[0:bunches_tune_shift] = np.ones(bunches_tune_shift)    
    B1_max2[bunches_tune_shift:tune_shift_gap+bunches_tune_shift+inj_space] = np.zeros(tune_shift_gap+inj_space)
    collisions_max2 = tbtb.Head_on(B1_max2,B2_max2,np.array([0,891,2670]))

    if collisions_max1[1]>collisions_max2[1]:
        B1_max = B1_max1
        B2_max = B2_max1
        collisions_max = collisions_max1
        #print("prefered B1") 
    else:
        B1_max = B1_max2
        B2_max = B2_max2
        collisions_max = collisions_max2
        #print("prefered B2") 
    #print(collisions_max1)
    #print(collisions_max2)
    return collisions_max,B1_max,B2_max


def prefill_algorithm_NOINDIV_LHCB(bunches,empty,len_batch, twelve = 12, inj_twelve = 12, inj_space = 31):
    '''
    Having as input the number of bunches for each PS_batch, the number of empties for the empty gap
    between PS batches, and how many PS_batches there are inside an SPS_batch, this script gives as output
    the two filling scheme of the two beams with the desired constraints given in input, with the number of collisions 
    in ATLAS/CMS/LHCb maximized.
    Hypothesys:
        - in this script we are considering that the "twelve" bunches for one of the two beams 
        are positioned in 0 and for the other in inj_twelve given as input
        - we are also considering that the ABK is optimized as best posssible in order to contatin perfectly the
        longest SPS batch, and it is completely filled
    Args:
        bunches: number of bunches for every PS_batch
        empty: number of empty slots for every empty gap between PS_batches
        len_batch: number of PS_batches for every SPS_batch
        twelve: how many bunches, for the first "twelve" bunches, set as 12
        inj_twelve: where to inject the "twelve" bunches of the detached beam, set as 12
        inj_space: number of empty slots fot every empty gap between SPS_batches
    Return: 
        np.array(collisions_max); a vector with the number of collisions using the desired filling scheme
        np.array(B1_max): a boolean vector that represent the slots filled by bunches for the desired beam1
        np.array(B2_max): a boolean vector that represent the slots filled by bunches for the desired beam2
    '''
    vec = [[]]
    vec = [[]]
    vec[0] = np.zeros(bunches+inj_space)
    vec[0][0:bunches] = np.ones(bunches)
    collisions =np.array([0,0,0])
    collisions_max1 = np.array([0,0,0])
    collisions_new = np.array([0,0,0])
    B1_max1 = np.zeros(3564)
    B2_max1 = np.zeros(3564)
    B1_max2 = np.zeros(3564)
    B2_max2 = np.zeros(3564)
    B1_new2 = np.zeros(3564)
    B2_new2 = np.zeros(3564)
    B1_new = -np.ones(3564)
    B2_new = -np.ones(3564)
    B1 = -np.ones(3564)
    B2 = -np.ones(3564)
    # vector that represent the batches trains, in counts the inj space only at the end!!
    for j in np.arange(len_batch-1)+2:
        v= np.zeros((j)*(bunches+empty)+(inj_space-empty))
        v[(j)*(bunches+empty):(j)*(bunches+empty)+(inj_space-empty)] = np.zeros((inj_space-empty))
        for i in np.arange(j):
            v[(i)*(bunches+empty):(i)*(bunches+empty)+bunches] = np.ones(bunches)
        vec.append(v)
    
    # fill the initial part with the twelve bunches and the final part, considering the AB and ABK 
    B1[0:inj_twelve] = np.zeros(inj_twelve)
    B1[inj_twelve:inj_twelve+twelve] = np.ones(twelve)
    B1[inj_twelve+twelve:inj_twelve+twelve+inj_space] = np.zeros(inj_space)
    B2[0:twelve] = np.ones(twelve)    
    B2[twelve:inj_twelve+twelve+inj_space] = np.zeros(inj_twelve+inj_space)
    B1_new[0:inj_twelve] = np.zeros(inj_twelve)
    B1_new[inj_twelve:inj_twelve+twelve] = np.ones(twelve)
    B1_new[inj_twelve+twelve:inj_twelve+twelve+inj_space] = np.zeros(inj_space)
    B2_new[0:twelve] = np.ones(twelve)    
    B2_new[twelve:inj_twelve+twelve+inj_space] = np.zeros(inj_twelve+inj_space)
    slot = 3564-121-((len_batch-1)*(bunches+empty)+bunches)
    B1[slot:3564-121+inj_space] = vec[len_batch-1]
    B2[slot:3564-121+inj_space] = vec[len_batch-1]
    slot = 3564-121-2*((len_batch-1)*(bunches+empty)+bunches)-31
    #fill the most possible the filling scheme with the biggest trains respecting the maximum given as input by thr user
    i = 1
    while slot > 2673-121+31:
        B1[slot:slot+((len_batch-1)*(bunches+empty)+bunches)+inj_space] = vec[len_batch-1]
        B2[slot:slot+((len_batch-1)*(bunches+empty)+bunches)+inj_space] = vec[len_batch-1]
        slot_New = slot
        i += 1
        slot = 3564-120-(i+1)*((len_batch-1)*(bunches+empty)+bunches)-i*31-1
    
    #copy the last quarter in the first one (shifting of 3 times of 3 slots (in order to optimize LHCb))
    #PS: Be careful in copying only the part of the last quarter that can fill after the 12 bunches in the first quarter
    if slot_New-2673-9>inj_twelve+twelve+inj_space:
        slot_New_pos = slot_New
    else:
        slot_New_pos = slot_New + j*(bunches+empty)+(inj_space-empty)
    B1[slot_New_pos-2673-9:891-9-121+inj_space] = B1[slot_New_pos:3564-121+inj_space]
    B2[slot_New_pos-2673-9:891-9-121+inj_space] = B2[slot_New_pos:3564-121+inj_space]
    


    # fill the space in a clever way, either trying to fill the remaining empty space with the maximum train possible
    # but consider that losing some collisions in LHCb, shifting these longest trains, you could insert a train tha contain one batch more, 
    # and this would lead to a better configuration possible for ATLAS/CMS

    # either here than (*) I use the flag because there are some specific approaches to lead to the best solution, and once done this
    # you wanna exit, because the other approaches would lead to worst solutions!!
    flag = 0
    for j in (len_batch-1)-np.arange(len_batch-1):
        frac = ((slot_New)-(2673-3)+(121)-(2*inj_space))/((j-1)*(bunches+empty)+bunches)
        # check if you can fill the empty space between quarters with a SPS batch of the maximum length possible
        if frac >1 and flag == 0:
            # check that shifting the longest trains of some slots, you could insert in the empty space between quarters a SPS batch
            # with a PS batch more, once entered in this shift, this situation is already preferred because you increase ATLAS/CMS...
            # we are doing the check using 2673-3 because the trains in the 3 quarter are shifted automatically by three slots on left, so 
            # there is more space 

            # POSSIBLE IMPROVEMENT: now you see if shifting of a quantity, you could increase the SPS batch by one, and then stop. But..
            # you could check also if shifting by >quantity and then position that SPS batch in different positions, considering also the SPS batch after
            # the twelve that you will insert later, if LHCb increases.
            max_number_shift = int((slot_New_pos-2673-9-(inj_twelve+twelve+inj_space))/3)
            check = (slot_New)-(2673-3)+(121)-(2*inj_space)+(1+np.arange(max_number_shift))>=(j*(bunches+empty)+bunches)

            if check.any():
                slot_shift = 1+np.arange(max_number_shift)[np.where(check)[0][0]]
                if (B1[slot_New-2673-9-3*slot_shift]==-1):
                    # here you are considering that in the different quarters there are different trains, so if the last train
                    # is shifted by a quantity "tot" so the train in the previous quarters is shifted by two times "tot" because he is shifted
                    # but is affected by the shift of the train ahead
                    # and you could see this shift of 3 slots that we talked before also here, always in order to maximize LHCb
                    init_slots_shift = [(slot_New-2673-9)-3*slot_shift,(885-3)-slot_shift*2,(1779-3)-slot_shift]
                    quarter_slot_shifted = (2673-3)-slot_shift-121+inj_space
                    B1_new[init_slots_shift[0]:891-9-3*slot_shift-121+inj_space] = B1[slot_New_pos:3564-121+inj_space]
                    B2_new[init_slots_shift[0]:891-9-3*slot_shift-121+inj_space] = B2[slot_New_pos:3564-121+inj_space]
                    B1_new[2673-121+inj_space:3564-121+31] = B1[2673-121+inj_space:3564-121+31]
                    B2_new[2673-121+inj_space:3564-121+31] = B2[2673-121+inj_space:3564-121+31]
                    B1_new[quarter_slot_shifted:quarter_slot_shifted+(j+1)*(bunches+empty)+(inj_space-empty)] = vec[j]
                    B2_new[quarter_slot_shifted:quarter_slot_shifted+(j+1)*(bunches+empty)+(inj_space-empty)] = vec[j]
                    for k in np.arange(2)+1:
                        # this 3 here, is ispired at the spaces which is shifted  the neirby quarter, before we subtracted 3 slots because the quarter on left
                        # was shifting 3 times more, but the trains that we were considering weren't shifted by these 3 slots so now I have to sum them again
                        B1_new[init_slots_shift[k]-slot_shift-121+inj_space:(init_slots_shift[k]+3)+891-121+31] = B1_new[quarter_slot_shifted:3564-121+31]
                        B2_new[init_slots_shift[k]-slot_shift-121+inj_space:(init_slots_shift[k]+3)+891-121+31] = B2_new[quarter_slot_shifted:3564-121+31]
                    

                    # (*)
                    # here we are filling the initial empty space, after the first twelve bunches, checking all the position available to see 
                    # which one is the best possible in terms of collisions in ALICE and save it
                    flag_init = 0
                    # in the following cycle we are considering as maximum SPS batch the one inserted between quarters, would be useless do the cycle
                    # for all length available because this is limited by the twelve bunches that there aren't in the other quarters and also becasue before 
                    # we fill the empty space between quarters, here before 0 there is the ABORT GAP where must be neither one bunch, so the space that we can fill
                    # is "half" of the spaces considered before
                    for l in j-np.arange(j):
                        initial_frac = (init_slots_shift[0]-inj_twelve-twelve-(2*inj_space))/((l-1)*(bunches+empty)+bunches)
                        if initial_frac >=1 and flag_init == 0:
                            flag_init = 1
                            B1_check = copy.copy(B1_new)
                            B2_check = copy.copy(B2_new)
                            for k in np.arange(init_slots_shift[0]-inj_space-(inj_twelve+twelve+inj_space))+inj_twelve+twelve+inj_space:
                                indec = k
                                end_indec = indec+(l-1)*(bunches+empty)+bunches
                                if (B1_check[indec] == -1) and (B1_check[end_indec] == -1) and end_indec<=(init_slots_shift[0]-inj_space):
                                    B1_new[indec:end_indec+inj_space] = vec[l-1]
                                    B2_new[indec:end_indec+inj_space] = vec[l-1]
                                    B1_new[np.where(B1_new==-1)] = 0
                                    B2_new[np.where(B2_new==-1)] = 0
                                    collisions_new = tbtb.Head_on(B1_new,B2_new,np.array([0,891,2670]))
                                    print(collisions_new)
                                    if collisions_new[2]>collisions_max1[2]:
                                        collisions_max1 = collisions_new
                                        vec_B1 = copy.copy(B1_new)
                                        vec_B2 = copy.copy(B2_new)
                                    B1_new = copy.copy(B1_check)
                                    B2_new = copy.copy(B2_check)
                            B1_max1 = vec_B1
                            B2_max1 = vec_B2  
                    if flag_init == 0:
                        B1_new[np.where(B1_new==-1)] = 0
                        B2_new[np.where(B2_new==-1)] = 0
                        collisions_new = tbtb.Head_on(B1_new,B2_new,np.array([0,891,2670]))
                        collisions_max1 = collisions_new
                        B1_max1 = copy.copy(B1_new)
                        B2_max1 = copy.copy(B2_new)        
                    # in this configuration you have maximized ATLAS/CMS more than all the other possible solutions, and also with that number 
                    # of collisions of ATLAS/CMS you have been checked
                    flag = 2

            
            if flag != 2:
                # now you couldn't increase the number of PS batch inside the SPS batch, so we maximize the the number of collisions in LHCb trying to
                # insert the SPS batch in the best position possible!
                gap_frac = (slot_New+1-2673+121-2*inj_space)/((j-1)*(bunches+empty)+bunches)
                flag_gap = 1
                if gap_frac>=1 and flag_gap ==1:
                    flag_gap = 1
                    B1_new2[:] = copy.copy(B1[:])
                    B2_new2[:] = copy.copy(B2[:])
                    for k in np.arange(slot_New+1-inj_space-(2673-121+inj_space))+2673-121+inj_space:
                        indec = k
                        end_indec = indec+j*(bunches+empty)+(inj_space-empty)
                        if (B1_new2[indec] == -1) and (B1_new2[end_indec] == -1):
                            B1_new2[indec:indec+j*(bunches+empty)+(inj_space-empty)] = vec[j-1]
                            B2_new2[indec:indec+j*(bunches+empty)+(inj_space-empty)] = vec[j-1]
                        # we are maximizing LHCb, so we are copying and pasting the SPS batches not perfectly on quarters of all the ring, so 
                        # we are shifting all by three slots, and we are ading up all the shifts
                        for l in [885,1779]:
                            B1_new2[l-121+inj_space:l+891-121+31] = B1_new2[2673-121+inj_space:3564-121+31]
                            B2_new2[l-121+inj_space:l+891-121+31] = B2_new2[2673-121+inj_space:3564-121+31]
                        
                        # (*)
                        # here we are filling the initial empty space, after the first twelve bunches, checking all the position available to see 
                        # which one is the best possible in terms of collisions in ALICE and save it
                        flag_init = 0
                        # in the following cycle we are considering as maximum SPS batch the one inserted between quarters, would be useless do the cycle
                        # for all length available because this is limited by the twelve bunches that there aren't in the other quarters and also becasue before 
                        # we fill the empty space between quarters, here before 0 there is the ABORT GAP where must be neither one bunch, so the space that we can fill
                        # is "half" of the spaces considered before
                        for l in j-np.arange(j):
                            initial_frac = (slot_New_pos-2673-9-inj_twelve-twelve-(2*inj_space))/((l-1)*(bunches+empty)+bunches)
                            if initial_frac >=1 and flag_init == 0:
                                flag_init = 1
                                B1_check = copy.copy(B1_new2)
                                B2_check = copy.copy(B2_new2)
                                for k in np.arange(slot_New_pos-2673-9-inj_space-(inj_twelve+twelve+inj_space))+inj_twelve+twelve+inj_space:
                                    indec = k
                                    end_indec = indec+(l-1)*(bunches+empty)+bunches
                                    if (B1_check[indec] == -1) and (B1_check[end_indec] == -1) and end_indec<=(slot_New_pos-2673-9-inj_space):
                                        B1_new2[indec:end_indec+inj_space] = vec[l-1]
                                        B2_new2[indec:end_indec+inj_space] = vec[l-1]
                                        B1_new2[np.where(B1_new2==-1)] = 0
                                        B2_new2[np.where(B2_new2==-1)] = 0
                                        collisions = tbtb.Head_on(B1_new2,B2_new2,np.array([0,891,2670]))
                                        if collisions[2]>collisions_max1[2]:
                                            collisions_max1 = collisions
                                            vec_B1 = copy.copy(B1_new2)
                                            vec_B2 = copy.copy(B2_new2)
                                        B1_new2 = copy.copy(B1_check)
                                        B2_new2 = copy.copy(B2_check)
                                B1_max1 = vec_B1
                                B2_max1 = vec_B2  
                        if flag_init == 0:
                            B1_new2[np.where(B1_new2==-1)] = 0
                            B2_new2[np.where(B2_new2==-1)] = 0
                            collisions = tbtb.Head_on(B1_new2,B2_new2,np.array([0,891,2670]))
                            collisions_max1 = collisions
                            B1_max1 = copy.copy(B1_new2)
                            B2_max1 = copy.copy(B2_new2) 
                # in this for we are going from the longest SPS batch to the smaller ones, so once you checked that one batch can be inserted,
                # surely the following steps will be smaller so you would lose in number of collisions in ATLAS/CMS, so you use the flag in order to exit from the cycle
                flag = 1
    
    # this is the case where you couldn't insert any SPS batch between quarters
    if flag == 0:
        B1_new2[:] = B1[:]
        B2_new2[:] = B2[:]
        for l in [885,1779]:
            B1_new2[l-121+inj_space:l+891-121+31] = B1_new2[2673-121+inj_space:3564-121+31]
            B2_new2[l-121+inj_space:l+891-121+31] = B2_new2[2673-121+inj_space:3564-121+31]
        
        # (*)
        # here we are filling the initial empty space, after the first twelve bunches, checking all the position available to see 
        # which one is the best possible in terms of collisions in ALICE and save it
        flag_init = 0
        # in the following cycle we are considering as maximum SPS batch the one inserted between quarters, would be useless do the cycle
        # for all length available because this is limited by the twelve bunches that there aren't in the other quarters and also becasue before 
        # we fill the empty space between quarters, here before 0 there is the ABORT GAP where must be neither one bunch, so the space that we can fill
        # is "half" of the spaces considered before
        for l in j-np.arange(j):
            initial_frac = (slot_New_pos-2673-9-inj_twelve-twelve-(2*inj_space))/((l-1)*(bunches+empty)+bunches)
            if initial_frac >=1 and flag_init == 0:
                flag_init = 1
                B1_check = copy.copy(B1_new2)
                B2_check = copy.copy(B2_new2)
                for k in np.arange(slot_New_pos-2673-9-inj_space-(inj_twelve+twelve+inj_space))+inj_twelve+twelve+inj_space:
                    indec = k
                    end_indec = indec+(l-1)*(bunches+empty)+bunches
                    if (B1_check[indec] == -1) and (B1_check[end_indec] == -1) and end_indec<=(slot_New_pos-2673-9-inj_space):
                        B1_new2[indec:end_indec+inj_space] = vec[l-1]
                        B2_new2[indec:end_indec+inj_space] = vec[l-1]
                        B1_new2[np.where(B1_new2==-1)] = 0
                        B2_new2[np.where(B2_new2==-1)] = 0
                        collisions = tbtb.Head_on(B1_new2,B2_new2,np.array([0,891,2670]))
                        if collisions[1]>collisions_max1[1]:
                            collisions_max1 = collisions
                            vec_B1 = copy.copy(B1_new2)
                            vec_B2 = copy.copy(B2_new2)
                        B1_new2 = copy.copy(B1_check)
                        B2_new2 = copy.copy(B2_check)
                B1_max1 = vec_B1
                B2_max1 = vec_B2  
        if flag_init == 0:
            B1_new2[np.where(B1_new2==-1)] = 0
            B2_new2[np.where(B2_new2==-1)] = 0
            collisions = tbtb.Head_on(B1_new2,B2_new2,np.array([0,891,2670]))
            collisions_max1 = collisions
            B1_max1 = copy.copy(B1_new2)
            B2_max1 = copy.copy(B2_new2) 

    # save the best solution possible considering the twelve bunches detached from 0 for both the beams, and see wich configuration is the best possible
    B1_max2 = copy.copy(B1_max1)
    B2_max2 = copy.copy(B2_max1)
    B2_max2[0:inj_twelve] = np.zeros(inj_twelve)
    B2_max2[inj_twelve:inj_twelve+twelve] = np.ones(twelve)
    B2_max2[inj_twelve+twelve:inj_twelve+twelve+inj_space] = np.zeros(inj_space)
    B1_max2[0:twelve] = np.ones(twelve)    
    B1_max2[twelve:inj_twelve+twelve+inj_space] = np.zeros(inj_twelve+inj_space)
    collisions_max2 = tbtb.Head_on(B1_max2,B2_max2,np.array([0,891,2670]))

    if collisions_max1[2]>collisions_max2[2]:
        B1_max = B1_max1
        B2_max = B2_max1
        collisions_max = collisions_max1 
        print("prefered B1") 
    else:
        B1_max = B1_max2
        B2_max = B2_max2
        collisions_max = collisions_max2
        print("prefered B2") 
    print(collisions_max)
    return collisions_max,B1_max,B2_max


#@profile
def MC_shift_preserve_ATLAS_optimized(slots_to_shift, slots_init, beam, n_sample, bunches, slots_to_shift_check, bunches_ns = 25, bunches_tune_shift = 12, bunches_tune_shift_ns = 25, empties = 7, shift_from_ABK = 0):
    '''
    a MC method to shift the trains in order to get a configuration with more collisions in LHCb or ALICE, the shift will done of 1 slot 
    singularly at every train, and the script will save all the configurations found
    Hypothesis:
        - considering that the wo beams are so similar, in "slots_to_shift" are represented the (first slot-1) and the (last slot +1) for every SPS_batch, only 
        for one of thetwo beams, which one is represented in "beam". The information of the position of the "twelve" bunches fot the other beam is represented in "slots_init"
        - for sake of semplicity we will call B1 as the beam that is represented by slots_to_shift
    Args:
        slots_to_shift: positions of the (first element -1) and (last element+1) of every SPS_batch for B1
        slots_init: initial position of the twelve bunches of the other beam (B2)
        beam: give the number of the beam connected with slots_to_shift (can be deleted)
        bunches: number of bunches for every PS batch (can be computed I think)
        slots_to_shift_check: the beam with whom compare the distance of MC simulation
        empties: number of empty slots for every empty gap between PS_batches, set as 7
        twelve: where  the "twelve" bunches of the detached beam, set as 0 (is this slots_to_shift-11?) control see if works
        ABK: free space before the Abort Gap that has the beam in case is not perfectly positioned in the Abort Gap Keeper
    Return: 
        pd.DataFrame: that contains the slots_to_shift, distance wrt slots_to_shift_check, slots_init, beam, events_in_ALICE, events_in_LHCb
        of all the MC simulation
    '''
    #Dictionary IPs
    Dict_IPN = {'IP1':0,'IP5':0,'IP8':2670,'IP2':891}
    #dictionary that converts time in space
    Dict_ns_len = {'25ns': 1, '50ns':2, '75ns':3,'100ns':4}
    assert bunches_ns in [25,50,75,100], "bunches_ns accepted in LHC are 25 or 50 or 75 or 100"
    len_ns = Dict_ns_len[f'{bunches_ns}ns']
    assert bunches_tune_shift_ns in [25,50,75,100], "bunches_tune_shift_ns accepted in LHC are 25 or 50 or 75 or 100"
    len_ns_measure_tune = Dict_ns_len[f'{bunches_tune_shift_ns}ns']
    #bunches in ns for tune measurement
    if bunches_tune_shift != 0:
        bunch_in_ns_measure_tune = np.zeros(len_ns_measure_tune)
        bunch_in_ns_measure_tune[0] = 1
        bunches_in_ns_measure_tune = np.zeros((bunches_tune_shift-1)*(1+(len_ns_measure_tune-1))+1)
        for i in range(bunches_tune_shift-1):
            bunches_in_ns_measure_tune[(i*len_ns_measure_tune):((i+1)*len_ns_measure_tune)] = bunch_in_ns_measure_tune
        bunches_in_ns_measure_tune[-1] = 1


    #bunch in the ns 
    bunch_in_ns = np.zeros(len_ns)
    bunch_in_ns[0] = 1
    #PS batch in ns, that finish with one, because there is later the inj_space
    PS_batch_in_ns = np.zeros((bunches-1)*(1+(len_ns-1))+1)
    for i in range(bunches-1):
        PS_batch_in_ns[(i*len_ns):((i+1)*len_ns)] = bunch_in_ns
    PS_batch_in_ns[-1] = 1

    #vector that contains the number of remaining slots in empty gap that are more than 800ns 
    B1_more800ns = []
    index_INDIV = []

    B1 = np.zeros(3564)
    B1_more800ns = np.zeros(len(slots_to_shift))
    #if slots_init-1 !=slots_to_shift[0] and bunches_tune_shift!=0:
    B1_more800ns[0] = -(slots_to_shift[0])
    B1_more800ns[-1] = shift_from_ABK

    #computation of the two beams and the B1_more800ns
    for i in np.arange(int((len(slots_to_shift)-2)/2))+1:
        more800_ns = slots_to_shift[2*i]-slots_to_shift[2*i-1]-32+3
        B1_more800ns[2*i-1] = more800_ns
        B1_more800ns[2*i] = -more800_ns
        slot = slots_to_shift[2*i-2]+1
        flag = 0
        while flag == 0:
            if slots_to_shift[2*i-1]-slots_to_shift[2*i-2] == 3:
                index_INDIV = np.append(index_INDIV, 2*i-2)
                B1[slot] = 1
                flag = 1
            else:
                if bunches_tune_shift != 0:
                    if 2*i-2 ==0:
                        B1[slot:slot+len(bunches_in_ns_measure_tune)] = bunches_in_ns_measure_tune
                        flag = 1
                    else:
                        B1[slot:slot+len(PS_batch_in_ns)] = PS_batch_in_ns
                        slot += len(PS_batch_in_ns)+empties
                        if (slot>=slots_to_shift[2*i-1]-1+empties):
                            flag = 1
                else:
                    B1[slot:slot+len(PS_batch_in_ns)] = PS_batch_in_ns
                    slot += len(PS_batch_in_ns)+empties
                    if (slot>=slots_to_shift[2*i-1]-1+empties):
                        flag = 1
    flag = 0
    slot = slots_to_shift[-2]+1
    while flag == 0:
        B1[slot:slot+len(PS_batch_in_ns)] = PS_batch_in_ns
        slot += len(PS_batch_in_ns)+empties
        if (slot>=slots_to_shift[-1]-1+empties):
            flag = 1
    B2 = copy.copy(B1)
    # if starting from the MonteCarlo, the two bunches for tune measurement are in the same slots, I will leave those there
    if slots_init-1 !=slots_to_shift[0] and bunches_tune_shift!=0:
        init_B2 = np.zeros(slots_to_shift[2]+1)
        init_B2[slots_init:slots_init+len(bunches_in_ns_measure_tune)] = bunches_in_ns_measure_tune
        B2[:slots_to_shift[2]+1] = init_B2


    # MC method, you compute all the shifts that are possible, in order to have an improvement on the number of collisions
    vector = np.zeros(100*n_sample)
    # list that contains all the interesting information of the MonteCarlo
    list_informations= [ii for ii in vector]
    informations = np.append(slots_to_shift,[tbtb.events_in_IPN(np.roll(B1,0),B2,'IP8')[1],tbtb.events_in_IPN(np.roll(B1,0),B2,'IP2')[1],slots_init,beam])
    list_informations[0] = [ii for ii in informations]

    
    #take the informations of how many possible samples of the MonteCarlo can be considered
    count = 1
    if len(B1_more800ns>0):
        
        #repeat 100 times the same MC simulation with the same number of possible shifts
        for index in np.arange(100):
            # copy in order to not overwrite the initial data
            B1_new = copy.copy(B1)
            B2_new = copy.copy(B2)
            B1_more800ns_new = copy.copy(B1_more800ns)
            slots_to_shift_new = copy.copy(slots_to_shift)
            # - (how many slots it is detached the first of beam1)
            init_shift_B1 = -slots_to_shift[0]
            for iter in np.arange(n_sample):
                # random choice of the position where have the shift (could be optimized this, sth to do)
                pos_shift = np.random.randint(len(B1_more800ns_new))
                
            

                # flag that in cases you cannot move all the INDIVs it surpass this step, and didn't move anything
                exit_flag_INDIV = 0
                if B1_more800ns_new[pos_shift]!=0:
                    count += 1
                    #shift value is the sign of the possible movement of the SPS batch
                    shift_value = int(np.sign(B1_more800ns_new[pos_shift]))
                    # if the pos shift is odd it means that the shift of SPS batch will be on right
                    if pos_shift%2 == 1:
                        if pos_shift-1 in index_INDIV:
                            # I'm moving all the INDIVs at the same time in order to not affect the number of collisions
                            # Pick all the INDIV except the pos_shift of the first INDIV
                            vec_INDIV = index_INDIV[~(pos_shift == index_INDIV+1)]+1
                            vec_INDIV = [int(i) for i in vec_INDIV]
                            #check if the movements are available
                            if np.array([B1_more800ns_new[i] !=0 for i in vec_INDIV]).all():
                                # do the shift for every indiv with except of the pos_shift that we'll do later
                                # and change all the information of the slots_to_shift_new
                                for i in vec_INDIV:
                                    pos_shift_2 = i
                                    
                                    B1_new[slots_to_shift_new[pos_shift_2-1]:slots_to_shift_new[pos_shift_2]] = np.roll(B1_new[slots_to_shift_new[pos_shift_2-1]:slots_to_shift_new[pos_shift_2]],shift_value)
                                    B2_new[slots_to_shift_new[pos_shift_2-1]:slots_to_shift_new[pos_shift_2]] = np.roll(B2_new[slots_to_shift_new[pos_shift_2-1]:slots_to_shift_new[pos_shift_2]],shift_value)
                                    slots_to_shift_new[pos_shift_2-1] += shift_value
                                    B1_more800ns_new[pos_shift_2] -= shift_value
                                    B1_more800ns_new[pos_shift_2-1] -= shift_value
                                    if pos_shift_2!=len(B1_more800ns_new)-1:
                                        B1_more800ns_new[pos_shift_2+shift_value] += shift_value
                                    if pos_shift_2 != 1:
                                        B1_more800ns_new[pos_shift_2-2] += shift_value
                                    slots_to_shift_new[pos_shift_2] += shift_value
                                    
                                    
                            else:
                                #if you cannot change the INDIV
                                exit_flag_INDIV = 1

                        #if we haven't pick the INDIV in pos_shift and also if we picked it, we could move all the other INDIVs
                        if exit_flag_INDIV == 0:

                            #shift of B1
                            B1_new[slots_to_shift_new[pos_shift-1]:slots_to_shift_new[pos_shift]] = np.roll(B1_new[slots_to_shift_new[pos_shift-1]:slots_to_shift_new[pos_shift]],shift_value)
                            #check in orrder to see if the there are bunches for tune measurement, in case affermative if they collide!
                            if slots_init-1 !=slots_to_shift[0] and bunches_tune_shift == 0:
                                #in this case we move both the beams at the same time
                                B2_new[slots_to_shift_new[pos_shift-1]:slots_to_shift_new[pos_shift]] = np.roll(B2_new[slots_to_shift_new[pos_shift-1]:slots_to_shift_new[pos_shift]],shift_value)
                                B1_more800ns_new[pos_shift-2] += shift_value
                                if i ==1:
                                    slots_init += shift_value
                            else:
                                if pos_shift!= 1:
                                    # shift of B2 if we are not moving the first 12 bunches
                                    B2_new[slots_to_shift_new[pos_shift-1]:slots_to_shift_new[pos_shift]] = np.roll(B2_new[slots_to_shift_new[pos_shift-1]:slots_to_shift_new[pos_shift]],shift_value)
                                    B1_more800ns_new[pos_shift-2] += shift_value
                                else:
                                    # if we are moving the bunches for the tune measurent, we have to be careful because in this case we don't want collisions in ATLAS/CMS
                                    #change of parameters 
                                    init_shift_B1 -= shift_value
                                    #change of slots_init for the B2 chosen randomly among the positions available after the shift of B1
                                    if B1_more800ns_new[1]>=12:
                                        c = np.random.randint(2)
                                        slots_init = np.random.randint([0,slots_to_shift_new[1]+shift_value+1],[slots_to_shift_new[0]-10+shift_value,slots_to_shift_new[2]-10])[c]
                                    else:
                                        slots_init = np.random.randint([0],[slots_to_shift_new[0]-10+shift_value])[0]
                                    
                                    init_B2 = np.zeros(slots_to_shift_new[2]+1)
                                    init_B2[slots_init:slots_init+len(bunches_in_ns_measure_tune)] = bunches_in_ns_measure_tune
                                    B2_new[:slots_to_shift_new[2]+1] = init_B2

                            #change of all the parameters
                            slots_to_shift_new[pos_shift-1] += shift_value
                            B1_more800ns_new[pos_shift] -= shift_value
                            B1_more800ns_new[pos_shift-1] -= shift_value
                            if pos_shift!=len(B1_more800ns_new)-1:
                                B1_more800ns_new[pos_shift+shift_value] += shift_value
                            slots_to_shift_new[pos_shift] += shift_value
                    # shift of SPS batch will be on right, we will do the same passages but considering the opposite direction
                    else:
                        if pos_shift in index_INDIV:
                            # I'm moving all the INDIVs at the same time in order to not affect the number of collisions
                            # Pick all the INDIV except the pos_shift of the first INDIV
                            vec_INDIV = index_INDIV[~(pos_shift == index_INDIV)]
                            vec_INDIV = [int(i) for i in vec_INDIV]
                            #check if the movements are available
                            if np.array([B1_more800ns_new[i] !=0 for i in vec_INDIV]).all():
                                # do the shift for every indiv with except of the pos_shift that we'll do later
                                # and change all the information of the slots_to_shift_new
                                for i in vec_INDIV:
                                    pos_shift_2 = i
                                    
                                    B1_new[slots_to_shift_new[pos_shift_2]:slots_to_shift_new[pos_shift_2+1]] = np.roll(B1_new[slots_to_shift_new[pos_shift_2]:slots_to_shift_new[pos_shift_2+1]],shift_value)
                                    B2_new[slots_to_shift_new[pos_shift_2]:slots_to_shift_new[pos_shift_2+1]] = np.roll(B2_new[slots_to_shift_new[pos_shift_2]:slots_to_shift_new[pos_shift_2+1]],shift_value)
                                    
                                    slots_to_shift_new[pos_shift_2+1] += shift_value
                                    B1_more800ns_new[pos_shift_2] -= shift_value
                                    B1_more800ns_new[pos_shift_2+1] -= shift_value
                                    if pos_shift_2!=len(B1_more800ns_new)-2:
                                        B1_more800ns_new[pos_shift_2+2] += shift_value 
                                    if pos_shift_2 !=0:   
                                        B1_more800ns_new[pos_shift_2+shift_value] += shift_value
                                    slots_to_shift_new[pos_shift_2] += shift_value
                                  
                            else:
                                #if you cannot change the INDIV
                                exit_flag_INDIV = 1
                        #if we haven't pick the INDIV in pos_shift and also if we picked it, we could move all the other INDIVs        
                        if exit_flag_INDIV == 0:
                            #shift of B1
                            B1_new[slots_to_shift_new[pos_shift]:slots_to_shift_new[pos_shift+1]] = np.roll(B1_new[slots_to_shift_new[pos_shift]:slots_to_shift_new[pos_shift+1]],shift_value)
                            #check in orrder to see if the there are bunches for tune measurement, in case affermative if they collide!
                            if slots_init-1 !=slots_to_shift[0] and bunches_tune_shift == 0:
                                B2_new[slots_to_shift_new[pos_shift]:slots_to_shift_new[pos_shift+1]] = np.roll(B2_new[slots_to_shift_new[pos_shift]:slots_to_shift_new[pos_shift+1]],shift_value)
                                B1_more800ns_new[pos_shift+shift_value] += shift_value
                                if i == 0:
                                    slots_init += shift_value
                            else:
                                if pos_shift!=0:
                                    # shift of B2 if we are not moving the bunches for tune shift 
                                    B2_new[slots_to_shift_new[pos_shift]:slots_to_shift_new[pos_shift+1]] = np.roll(B2_new[slots_to_shift_new[pos_shift]:slots_to_shift_new[pos_shift+1]],shift_value)
                                    B1_more800ns_new[pos_shift+shift_value] += shift_value
                                else:
                                    #change of parameters 
                                    init_shift_B1 -= shift_value
                                    #change of slots_init for the B2 chosen randomly among the positions available after the shift of B1
                                    if B1_more800ns_new[1]>=12:
                                        c = np.random.randint(2)
                                        slots_init = np.random.randint([0,slots_to_shift_new[1]+shift_value+1],[slots_to_shift_new[0]-10+shift_value,slots_to_shift_new[2]-10])[c]
                                    else:
                                        slots_init = np.random.randint([0],[slots_to_shift_new[0]-10+shift_value])[0]
                                    init_B2 = np.zeros(slots_to_shift_new[2]+1)
                                    init_B2[slots_init:slots_init+12] = np.ones(12)
                                    B2_new[:slots_to_shift_new[2]+1] = init_B2
                                    
                            #change of all the parameters
                            slots_to_shift_new[pos_shift+1] += shift_value
                            B1_more800ns_new[pos_shift] -= shift_value
                            B1_more800ns_new[pos_shift+1] -= shift_value
                            if pos_shift!=len(B1_more800ns_new)-2:
                                B1_more800ns_new[pos_shift+2] += shift_value    
                            slots_to_shift_new[pos_shift] += shift_value

                    #once done the shift save the information that you need in order to restart the MonteCarlo
                    events_LHCb_new = tbtb.events_in_IPN(np.roll(B1_new,0),B2_new,'IP8')[1]
                    events_ALICE_new = tbtb.events_in_IPN(np.roll(B1_new,0),B2_new,'IP2')[1]
                    informations_new = np.append(slots_to_shift_new,[events_LHCb_new,events_ALICE_new,slots_init,beam])
                    list_informations[count-1] = [ii for ii in informations_new]
            
        # put all the informations, sectioned, that we want in the DataFrame 
        del list_informations[count:]
        df_fill_schemes2 = pd.DataFrame(index = np.arange(count))
        df_fill_schemes2['informations'] = list_informations
        print(f'aaa{df_fill_schemes2.iloc[0]["informations"]}')
        df_fill_schemes = pd.DataFrame(np.unique(df_fill_schemes2), columns=df_fill_schemes2.columns)
        df_help = pd.DataFrame(df_fill_schemes2["informations"].values.tolist())
        df_fill_schemes = pd.DataFrame(index = np.arange(len(df_help.index)))
        vec_string_slot = df_help[df_help.columns[0:len(slots_to_shift)]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1)
        help_vec = [list(ii.split(" ")) for ii in vec_string_slot]
        df_fill_schemes['slots_to_shift'] = [[int(eval(ii)) for ii in i] for i in help_vec]
        df_fill_schemes['distance'] = [LA.norm(ii-slots_to_shift_check[0]) for ii in df_fill_schemes['slots_to_shift']]
        df_fill_schemes['events_LHCb'] = df_help.iloc[:,-4]
        df_fill_schemes['events_ALICE'] = df_help.iloc[:,-3]
        df_fill_schemes['slots_init'] = [int(ii) for ii in df_help.iloc[:,-2]]
        df_fill_schemes['beam'] = df_help.iloc[:,-1]

    return df_fill_schemes

def MC_hybrid_shift_preserve_ATLAS_optimized(slots_to_shift, slots_init, beam, n_sample, bunches, slots_to_shift_check, bunches_ns = 25, bunches_tune_shift = 12, bunches_tune_shift_ns = 25, empties = 7, bunches2 = np.nan, empties2 = np.nan, len_batches = np.nan, shift_from_ABK = 0):
    '''
    a MC method to shift the trains in order to get a configuration with more collisions in LHCb or ALICE, the shift will done of 1 slot 
    singularly at every train, and the script will save all the configurations found
    Hypothesis:
        - considering that the wo beams are so similar, in "slots_to_shift" are represented the (first slot-1) and the (last slot +1) for every SPS_batch, only 
        for one of thetwo beams, which one is represented in "beam". The information of the position of the "twelve" bunches fot the other beam is represented in "slots_init"
        - for sake of semplicity we will call B1 as the beam that is represented by slots_to_shift
    Args:
        slots_to_shift: positions of the (first element -1) and (last element+1) of every SPS_batch for B1
        slots_init: initial position of the twelve bunches of the other beam (B2)
        beam: give the number of the beam connected with slots_to_shift (can be deleted)
        bunches: number of bunches for every PS batch (can be computed I think)
        slots_to_shift_check: the beam with whom compare the distance of MC simulation
        empties: number of empty slots for every empty gap between PS_batches, set as 7
        twelve: where  the "twelve" bunches of the detached beam, set as 0 (is this slots_to_shift-11?) control see if works
        ABK: free space before the Abort Gap that has the beam in case is not perfectly positioned in the Abort Gap Keeper
    Return: 
        pd.DataFrame: that contains the slots_to_shift, distance wrt slots_to_shift_check, slots_init, beam, events_in_ALICE, events_in_LHCb
        of all the MC simulation
    '''

    if np.isnan(bunches2):
        assert bunches2 is empties2 is len_batches, "if there are not hybrid you have to put bunches2 empties and len_batches as Nan"
    else:
        assert empties2!=np.nan and len_batches != np.nan, "if it is hybrid you have to put bunches2 empties and len_batches all different of Nan"

    #Dictionary IPs
    Dict_IPN = {'IP1':0,'IP5':0,'IP8':2670,'IP2':891}
    #dictionary that converts time in space
    Dict_ns_len = {'25ns': 1, '50ns':2, '75ns':3,'100ns':4}
    assert bunches_ns in [25,50,75,100], "bunches_ns accepted in LHC are 25 or 50 or 75 or 100"
    len_ns = Dict_ns_len[f'{bunches_ns}ns']
    assert bunches_tune_shift_ns in [25,50,75,100], "bunches_tune_shift_ns accepted in LHC are 25 or 50 or 75 or 100"
    len_ns_measure_tune = Dict_ns_len[f'{bunches_tune_shift_ns}ns']
    #bunches in ns for tune measurement
    if bunches_tune_shift != 0:
        bunch_in_ns_measure_tune = np.zeros(len_ns_measure_tune)
        bunch_in_ns_measure_tune[0] = 1
        bunches_in_ns_measure_tune = np.zeros((bunches_tune_shift-1)*(1+(len_ns_measure_tune-1))+1)
        for i in range(bunches_tune_shift-1):
            bunches_in_ns_measure_tune[(i*len_ns_measure_tune):((i+1)*len_ns_measure_tune)] = bunch_in_ns_measure_tune
        bunches_in_ns_measure_tune[-1] = 1

    if np.isnan(bunches2):
        #bunch in the ns 
        bunch_in_ns = np.zeros(len_ns)
        bunch_in_ns[0] = 1
        #PS batch in ns, that finish with one, because there is later the inj_space
        PS_batch_in_ns = np.zeros((bunches-1)*(1+(len_ns-1))+1)
        for i in range(bunches-1):
            PS_batch_in_ns[(i*len_ns):((i+1)*len_ns)] = bunch_in_ns
        PS_batch_in_ns[-1] = 1

        #vector that contains the number of remaining slots in empty gap that are more than 800ns 
        B1_more800ns = []
        index_INDIV = []

        B1 = np.zeros(3564)
        B1_more800ns = np.zeros(len(slots_to_shift))
        #if slots_init-1 !=slots_to_shift[0] and bunches_tune_shift!=0:
        B1_more800ns[0] = -(slots_to_shift[0])
        B1_more800ns[-1] = shift_from_ABK

        #check if you are working on hybrid filling scheme
        

        #computation of the two beams and the B1_more800ns
        for i in np.arange(int((len(slots_to_shift)-2)/2))+1:
            more800_ns = slots_to_shift[2*i]-slots_to_shift[2*i-1]-32+3
            B1_more800ns[2*i-1] = more800_ns
            B1_more800ns[2*i] = -more800_ns
            slot = slots_to_shift[2*i-2]+1
            flag = 0
            while flag == 0:
                if slots_to_shift[2*i-1]-slots_to_shift[2*i-2] == 3:
                    index_INDIV = np.append(index_INDIV, 2*i-2)
                    B1[slot] = 1
                    flag = 1
                else:
                    if bunches_tune_shift != 0:
                        if 2*i-2 ==0:
                            B1[slot:slot+len(bunches_in_ns_measure_tune)] = bunches_in_ns_measure_tune
                            flag = 1
                        else:
                            B1[slot:slot+len(PS_batch_in_ns)] = PS_batch_in_ns
                            slot += len(PS_batch_in_ns)+empties
                            if (slot>=slots_to_shift[2*i-1]-1+empties):
                                flag = 1
                    else:
                        B1[slot:slot+len(PS_batch_in_ns)] = PS_batch_in_ns
                        slot += len(PS_batch_in_ns)+empties
                        if (slot>=slots_to_shift[2*i-1]-1+empties):
                            flag = 1
        flag = 0
        slot = slots_to_shift[-2]+1
        while flag == 0:
            B1[slot:slot+len(PS_batch_in_ns)] = PS_batch_in_ns
            slot += len(PS_batch_in_ns)+empties
            if (slot>=slots_to_shift[-1]-1+empties):
                flag = 1
        B2 = copy.copy(B1)
        # if starting from the MonteCarlo, the two bunches for tune measurement are in the same slots, I will leave those there
        if slots_init-1 !=slots_to_shift[0] and bunches_tune_shift!=0:
            init_B2 = np.zeros(slots_to_shift[2]+1)
            init_B2[slots_init:slots_init+len(bunches_in_ns_measure_tune)] = bunches_in_ns_measure_tune
            B2[:slots_to_shift[2]+1] = init_B2
    else:
        #bunch in the ns 
        bunch_in_ns = np.zeros(len_ns)
        bunch_in_ns[0] = 1
        PS_batch1_in_ns = np.zeros((bunches-1)*(1+(len_ns-1))+1)
        for i in range(bunches-1):
            PS_batch1_in_ns[(i*len_ns):((i+1)*len_ns)] = bunch_in_ns
        PS_batch1_in_ns[-1] = 1
        PS_batch2_in_ns = np.zeros((bunches2-1)*(1+(len_ns-1))+1)
        for i in range(bunches2-1):
            PS_batch2_in_ns[(i*len_ns):((i+1)*len_ns)] = bunch_in_ns
        PS_batch2_in_ns[-1] = 1
        #PS batch in ns, that finish with one, because there is later the inj_space
        SPS_batch_in_ns = np.zeros((len_batches[0]-1)*((bunches-1)*(1+(len_ns-1))+1+empties)+((bunches-1)*(1+(len_ns-1))+1+empties2)+\
            (len_batches[1]-1)*((bunches2-1)*(1+(len_ns-1))+1+empties2)+((bunches2-1)*(1+(len_ns-1))+1))
        for i in range(len_batches[0]):
            SPS_batch_in_ns[i*((bunches-1)*(1+(len_ns-1))+1+empties):i*((bunches-1)*(1+(len_ns-1))+1+empties)+((bunches-1)*(1+(len_ns-1))+1)] = PS_batch1_in_ns
        for i in range(len_batches[1]-1):
            print(i*((bunches2-1)*(1+(len_ns-1))+1+empties2))
            SPS_batch_in_ns[(len_batches[0]-1)*((bunches-1)*(1+(len_ns-1))+1+empties)+((bunches-1)*(1+(len_ns-1))+1+empties2)+i*((bunches2-1)*(1+(len_ns-1))+1+empties2)\
                :(len_batches[0]-1)*((bunches-1)*(1+(len_ns-1))+1+empties)+((bunches-1)*(1+(len_ns-1))+1+empties2)+i*((bunches2-1)*(1+(len_ns-1))+1+empties2)+((bunches2-1)*(1+(len_ns-1))+1)] = PS_batch2_in_ns
        SPS_batch_in_ns[-((bunches2-1)*(1+(len_ns-1))+1):] = PS_batch2_in_ns


        if bunches_tune_shift != 0:
            bunch_in_ns_measure_tune = np.zeros(len_ns_measure_tune)
            bunch_in_ns_measure_tune[0] = 1
            bunches_in_ns_measure_tune = np.zeros((bunches_tune_shift-1)*(1+(len_ns_measure_tune-1))+1)
            for i in range(bunches_tune_shift-1):
                bunches_in_ns_measure_tune[(i*len_ns_measure_tune):((i+1)*len_ns_measure_tune)] = bunch_in_ns_measure_tune
            bunches_in_ns_measure_tune[-1] = 1

        #vector that contains the number of remaining slots in empty gap that are more than 800ns 
        B1_more800ns = []
        index_INDIV = []

        B1 = np.zeros(3564)
        B1_more800ns = np.zeros(len(slots_to_shift))
        #if slots_init-1 !=slots_to_shift[0] and bunches_tune_shift!=0:
        B1_more800ns[0] = -(slots_to_shift[0])
        B1_more800ns[-1] = 0

        #check if you are working on hybrid filling scheme


        #computation of the two beams and the B1_more800ns
        for i in np.arange(int((len(slots_to_shift)-2)/2))+1:
            more800_ns = slots_to_shift[2*i]-slots_to_shift[2*i-1]-32+3
            B1_more800ns[2*i-1] = more800_ns
            B1_more800ns[2*i] = -more800_ns
            slot = int(slots_to_shift[2*i-2]+1)
            if slots_to_shift[2*i-1]-slots_to_shift[2*i-2] == 3:
                index_INDIV = np.append(index_INDIV, 2*i-2)
                B1[slot] = 1
            else:
                if bunches_tune_shift != 0:
                    if 2*i-2 ==0:
                        B1[slot:slot+len(bunches_in_ns_measure_tune)] = bunches_in_ns_measure_tune
                    else:
                        B1[slot:slot+len(SPS_batch_in_ns)] = SPS_batch_in_ns
                else:
                    B1[slot:slot+len(SPS_batch_in_ns)] = SPS_batch_in_ns
        slot = int(slots_to_shift[-2]+1)
        B1[slot:slot+len(SPS_batch_in_ns)] = SPS_batch_in_ns

        B2 = copy.copy(B1)
        # if starting from the MonteCarlo, the two bunches for tune measurement are in the same slots, I will leave those there
        if slots_init-1 !=int(slots_to_shift[0]) and bunches_tune_shift!=0:
            init_B2 = np.zeros(slots_to_shift[2]+1)
            print("buuu")
            init_B2[slots_init:slots_init+len(bunches_in_ns_measure_tune)] = bunches_in_ns_measure_tune
            B2[:slots_to_shift[2]+1] = init_B2
            


    # MC method, you compute all the shifts that are possible, in order to have an improvement on the number of collisions
    vector = np.zeros(10*n_sample)
    # list that contains all the interesting information of the MonteCarlo
    list_informations= [ii for ii in vector]
    informations = np.append(slots_to_shift,[tbtb.events_in_IPN(np.roll(B1,0),B2,'IP8')[1],tbtb.events_in_IPN(np.roll(B1,0),B2,'IP2')[1],slots_init,beam])
    print(informations)
    print(tbtb.events_in_IPN(B1,B2,'IP8')[1])
    list_informations[0] = [ii for ii in informations]

    
    #take the informations of how many possible samples of the MonteCarlo can be considered
    count = 1
    if len(B1_more800ns>0):
        
        #repeat 100 times the same MC simulation with the same number of possible shifts
        for index in np.arange(10):
            # copy in order to not overwrite the initial data
            B1_new = copy.copy(B1)
            B2_new = copy.copy(B2)
            B1_more800ns_new = copy.copy(B1_more800ns)
            slots_to_shift_new = copy.copy(slots_to_shift)
            # - (how many slots it is detached the first of beam1)
            init_shift_B1 = -slots_to_shift[0]
            for iter in np.arange(n_sample):
                # random choice of the position where have the shift (could be optimized this, sth to do)
                pos_shift = np.random.randint(len(B1_more800ns_new))
                
            

                # flag that in cases you cannot move all the INDIVs it surpass this step, and didn't move anything
                exit_flag_INDIV = 0
                if B1_more800ns_new[pos_shift]!=0:
                    count += 1
                    #shift value is the sign of the possible movement of the SPS batch
                    shift_value = int(np.sign(B1_more800ns_new[pos_shift]))
                    # if the pos shift is odd it means that the shift of SPS batch will be on right
                    if pos_shift%2 == 1:
                        if pos_shift-1 in index_INDIV:
                            # I'm moving all the INDIVs at the same time in order to not affect the number of collisions
                            # Pick all the INDIV except the pos_shift of the first INDIV
                            vec_INDIV = index_INDIV[~(pos_shift == index_INDIV+1)]+1
                            vec_INDIV = [int(i) for i in vec_INDIV]
                            #check if the movements are available
                            if np.array([B1_more800ns_new[i] !=0 for i in vec_INDIV]).all():
                                # do the shift for every indiv with except of the pos_shift that we'll do later
                                # and change all the information of the slots_to_shift_new
                                for i in vec_INDIV:
                                    pos_shift_2 = i
                                    
                                    B1_new[slots_to_shift_new[pos_shift_2-1]:slots_to_shift_new[pos_shift_2]] = np.roll(B1_new[slots_to_shift_new[pos_shift_2-1]:slots_to_shift_new[pos_shift_2]],shift_value)
                                    B2_new[slots_to_shift_new[pos_shift_2-1]:slots_to_shift_new[pos_shift_2]] = np.roll(B2_new[slots_to_shift_new[pos_shift_2-1]:slots_to_shift_new[pos_shift_2]],shift_value)
                                    slots_to_shift_new[pos_shift_2-1] += shift_value
                                    B1_more800ns_new[pos_shift_2] -= shift_value
                                    B1_more800ns_new[pos_shift_2-1] -= shift_value
                                    if pos_shift_2!=len(B1_more800ns_new)-1:
                                        B1_more800ns_new[pos_shift_2+shift_value] += shift_value
                                    if pos_shift_2 != 1:
                                        B1_more800ns_new[pos_shift_2-2] += shift_value
                                    slots_to_shift_new[pos_shift_2] += shift_value
                                    
                                    
                            else:
                                #if you cannot change the INDIV
                                exit_flag_INDIV = 1

                        #if we haven't pick the INDIV in pos_shift and also if we picked it, we could move all the other INDIVs
                        if exit_flag_INDIV == 0:

                            #shift of B1
                            B1_new[slots_to_shift_new[pos_shift-1]:slots_to_shift_new[pos_shift]] = np.roll(B1_new[slots_to_shift_new[pos_shift-1]:slots_to_shift_new[pos_shift]],shift_value)
                            #check in orrder to see if the there are bunches for tune measurement, in case affermative if they collide!
                            if slots_init-1 !=slots_to_shift[0] and bunches_tune_shift == 0:
                                #in this case we move both the beams at the same time
                                B2_new[slots_to_shift_new[pos_shift-1]:slots_to_shift_new[pos_shift]] = np.roll(B2_new[slots_to_shift_new[pos_shift-1]:slots_to_shift_new[pos_shift]],shift_value)
                                B1_more800ns_new[pos_shift-2] += shift_value
                                if i ==1:
                                    slots_init += shift_value
                            else:
                                if pos_shift!= 1:
                                    # shift of B2 if we are not moving the first 12 bunches
                                    B2_new[slots_to_shift_new[pos_shift-1]:slots_to_shift_new[pos_shift]] = np.roll(B2_new[slots_to_shift_new[pos_shift-1]:slots_to_shift_new[pos_shift]],shift_value)
                                    B1_more800ns_new[pos_shift-2] += shift_value
                                else:
                                    # if we are moving the bunches for the tune measurent, we have to be careful because in this case we don't want collisions in ATLAS/CMS
                                    #change of parameters 
                                    init_shift_B1 -= shift_value
                                    #change of slots_init for the B2 chosen randomly among the positions available after the shift of B1
                                    if B1_more800ns_new[1]>=12 and np.abs(B1_more800ns_new[0])>=12:
                                        c = np.random.randint(2)
                                        print([slots_to_shift_new[0],slots_to_shift_new[1],slots_to_shift_new[2]])
                                        print(B1_more800ns_new[1])
                                        print(slots_to_shift_new[1]-1)
                                        print(slots_to_shift_new[0]-10+shift_value)
                                        print(slots_to_shift_new[2]-(bunches_tune_shift)-31)
                                        slots_init = np.random.randint([0,slots_to_shift_new[1]-1],[slots_to_shift_new[0]-10+shift_value+1,slots_to_shift_new[2]-(bunches_tune_shift-2)-31])[c]
                                    elif B1_more800ns_new[1]<12 and np.abs(B1_more800ns_new[0])>=12:
                                        slots_init = np.random.randint([0],[slots_to_shift_new[0]-10+shift_value+1])[0]
                                    elif B1_more800ns_new[1]>=12 and np.abs(B1_more800ns_new[0])<12:
                                        slots_init = np.random.randint([slots_to_shift_new[1]-1],[slots_to_shift_new[2]-(bunches_tune_shift-2)-31])[0]
                                    init_B2 = np.zeros(slots_to_shift_new[2]+1)
                                    init_B2[slots_init:slots_init+len(bunches_in_ns_measure_tune)] = bunches_in_ns_measure_tune
                                    B2_new[:slots_to_shift_new[2]+1] = init_B2

                            #change of all the parameters
                            slots_to_shift_new[pos_shift-1] += shift_value
                            B1_more800ns_new[pos_shift] -= shift_value
                            B1_more800ns_new[pos_shift-1] -= shift_value
                            if pos_shift!=len(B1_more800ns_new)-1:
                                B1_more800ns_new[pos_shift+shift_value] += shift_value
                            slots_to_shift_new[pos_shift] += shift_value
                    # shift of SPS batch will be on right, we will do the same passages but considering the opposite direction
                    else:
                        if pos_shift in index_INDIV:
                            # I'm moving all the INDIVs at the same time in order to not affect the number of collisions
                            # Pick all the INDIV except the pos_shift of the first INDIV
                            vec_INDIV = index_INDIV[~(pos_shift == index_INDIV)]
                            vec_INDIV = [int(i) for i in vec_INDIV]
                            #check if the movements are available
                            if np.array([B1_more800ns_new[i] !=0 for i in vec_INDIV]).all():
                                # do the shift for every indiv with except of the pos_shift that we'll do later
                                # and change all the information of the slots_to_shift_new
                                for i in vec_INDIV:
                                    pos_shift_2 = i
                                    
                                    B1_new[slots_to_shift_new[pos_shift_2]:slots_to_shift_new[pos_shift_2+1]] = np.roll(B1_new[slots_to_shift_new[pos_shift_2]:slots_to_shift_new[pos_shift_2+1]],shift_value)
                                    B2_new[slots_to_shift_new[pos_shift_2]:slots_to_shift_new[pos_shift_2+1]] = np.roll(B2_new[slots_to_shift_new[pos_shift_2]:slots_to_shift_new[pos_shift_2+1]],shift_value)
                                    
                                    slots_to_shift_new[pos_shift_2+1] += shift_value
                                    B1_more800ns_new[pos_shift_2] -= shift_value
                                    B1_more800ns_new[pos_shift_2+1] -= shift_value
                                    if pos_shift_2!=len(B1_more800ns_new)-2:
                                        B1_more800ns_new[pos_shift_2+2] += shift_value 
                                    if pos_shift_2 !=0:   
                                        B1_more800ns_new[pos_shift_2+shift_value] += shift_value
                                    slots_to_shift_new[pos_shift_2] += shift_value
                                  
                            else:
                                #if you cannot change the INDIV
                                exit_flag_INDIV = 1
                        #if we haven't pick the INDIV in pos_shift and also if we picked it, we could move all the other INDIVs        
                        if exit_flag_INDIV == 0:
                            #shift of B1
                            B1_new[slots_to_shift_new[pos_shift]:slots_to_shift_new[pos_shift+1]] = np.roll(B1_new[slots_to_shift_new[pos_shift]:slots_to_shift_new[pos_shift+1]],shift_value)
                            #check in orrder to see if the there are bunches for tune measurement, in case affermative if they collide!
                            if slots_init-1 !=slots_to_shift[0] and bunches_tune_shift == 0:
                                B2_new[slots_to_shift_new[pos_shift]:slots_to_shift_new[pos_shift+1]] = np.roll(B2_new[slots_to_shift_new[pos_shift]:slots_to_shift_new[pos_shift+1]],shift_value)
                                B1_more800ns_new[pos_shift+shift_value] += shift_value
                                if i == 0:
                                    slots_init += shift_value
                            else:
                                if pos_shift!=0:
                                    # shift of B2 if we are not moving the bunches for tune shift 
                                    B2_new[slots_to_shift_new[pos_shift]:slots_to_shift_new[pos_shift+1]] = np.roll(B2_new[slots_to_shift_new[pos_shift]:slots_to_shift_new[pos_shift+1]],shift_value)
                                    B1_more800ns_new[pos_shift+shift_value] += shift_value
                                else:
                                    #change of parameters 
                                    init_shift_B1 -= shift_value
                                    #change of slots_init for the B2 chosen randomly among the positions available after the shift of B1
                                    if B1_more800ns_new[1]>=12 and np.abs(B1_more800ns_new[0])>=12:
                                        c = np.random.randint(2)
                                        print([slots_to_shift_new[0],slots_to_shift_new[1],slots_to_shift_new[2]])
                                        print(B1_more800ns_new[1])
                                        print(slots_to_shift_new[1]-1)
                                        print(slots_to_shift_new[0]-10+shift_value)
                                        print(slots_to_shift_new[2]-(bunches_tune_shift-1)-31)
                                        slots_init = np.random.randint([0,slots_to_shift_new[1]-1],[slots_to_shift_new[0]-10+shift_value+1,slots_to_shift_new[2]-(bunches_tune_shift-2)-31])[c]
                                    elif B1_more800ns_new[1]<12 and np.abs(B1_more800ns_new[0])>=12:
                                        slots_init = np.random.randint([0],[slots_to_shift_new[0]-10+shift_value+1])[0]
                                    elif B1_more800ns_new[1]>=12 and np.abs(B1_more800ns_new[0])<12:
                                        slots_init = np.random.randint([slots_to_shift_new[1]-1],[slots_to_shift_new[2]-(bunches_tune_shift-2)-31])[0]
                                    init_B2 = np.zeros(slots_to_shift_new[2]+1)
                                    init_B2[slots_init:slots_init+12] = np.ones(12)
                                    B2_new[:slots_to_shift_new[2]+1] = init_B2
                                    
                            #change of all the parameters
                            slots_to_shift_new[pos_shift+1] += shift_value
                            B1_more800ns_new[pos_shift] -= shift_value
                            B1_more800ns_new[pos_shift+1] -= shift_value
                            if pos_shift!=len(B1_more800ns_new)-2:
                                B1_more800ns_new[pos_shift+2] += shift_value    
                            slots_to_shift_new[pos_shift] += shift_value

                    #once done the shift save the information that you need in order to restart the MonteCarlo
                    events_LHCb_new = tbtb.events_in_IPN(np.roll(B1_new,0),B2_new,'IP8')[1]
                    events_ALICE_new = tbtb.events_in_IPN(np.roll(B1_new,0),B2_new,'IP2')[1]
                    informations_new = np.append(slots_to_shift_new,[events_LHCb_new,events_ALICE_new,slots_init,beam])
                    list_informations[count-1] = [ii for ii in informations_new]
            
        # put all the informations, sectioned, that we want in the DataFrame 
        del list_informations[count:]
        df_fill_schemes2 = pd.DataFrame(index = np.arange(count))
        df_fill_schemes2['informations'] = list_informations
        print(df_fill_schemes2.iloc[0]['informations'])
        df_fill_schemes = pd.DataFrame(np.unique(df_fill_schemes2), columns=df_fill_schemes2.columns)
        df_help = pd.DataFrame(df_fill_schemes2["informations"].values.tolist())
        df_fill_schemes = pd.DataFrame(index = np.arange(len(df_help.index)))
        vec_string_slot = df_help[df_help.columns[0:len(slots_to_shift)]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1)
        help_vec = [list(ii.split(" ")) for ii in vec_string_slot]
        df_fill_schemes['slots_to_shift'] = [[int(eval(ii)) for ii in i] for i in help_vec]
        df_fill_schemes['distance'] = [LA.norm(ii-slots_to_shift_check[0]) for ii in df_fill_schemes['slots_to_shift']]
        df_fill_schemes['events_LHCb'] = df_help.iloc[:,-4]
        df_fill_schemes['events_ALICE'] = df_help.iloc[:,-3]
        df_fill_schemes['slots_init'] = [int(ii) for ii in df_help.iloc[:,-2]]
        df_fill_schemes['beam'] = df_help.iloc[:,-1]

    return df_fill_schemes


def MC_shift(slots_to_shift1, slots_to_shift2, n_sample, bunches, bunches_ns = 25, bunches_tune_shift = 12, bunches_tune_shift_ns = 25, empties = 7, shift_from_ABK = 0):
    '''
    a MC method to shift the trains in order to get a configuration with more collisions in LHCb or ALICE, the shift will done of 1 slot 
    singularly at every train, and the script will save all the configurations found
    Hypothesis:
        - considering that the wo beams are so similar, in "slots_to_shift" are represented the (first slot-1) and the (last slot +1) for every SPS_batch, only 
        for one of thetwo beams, which one is represented in "beam". The information of the position of the "twelve" bunches fot the other beam is represented in "slots_init"
        - for sake of semplicity we will call B1 as the beam that is represented by slots_to_shift
    Args:
        slots_to_shift: positions of the (first element -1) and (last element+1) of every SPS_batch for B1
        slots_init: initial position of the twelve bunches of the other beam (B2)
        beam: give the number of the beam connected with slots_to_shift (can be deleted)
        bunches: number of bunches for every PS batch (can be computed I think)
        slots_to_shift_check: the beam with whom compare the distance of MC simulation
        empties: number of empty slots for every empty gap between PS_batches, set as 7
        twelve: where  the "twelve" bunches of the detached beam, set as 0 (is this slots_to_shift-11?) control see if works
        ABK: free space before the Abort Gap that has the beam in case is not perfectly positioned in the Abort Gap Keeper
    Return: 
        pd.DataFrame: that contains the slots_to_shift, distance wrt slots_to_shift_check, slots_init, beam, events_in_ALICE, events_in_LHCb
        of all the MC simulation
    '''

    #dictionary that converts time in space
    Dict_ns_len = {'25ns': 1, '50ns':2, '75ns':3,'100ns':4}
    assert bunches_ns in [25,50,75,100], "bunches_ns accepted in LHC are 25 or 50 or 75 or 100"
    len_ns = Dict_ns_len[f'{bunches_ns}ns']
    assert bunches_tune_shift_ns in [25,50,75,100], "bunches_tune_shift_ns accepted in LHC are 25 or 50 or 75 or 100"
    len_ns_measure_tune = Dict_ns_len[f'{bunches_tune_shift_ns}ns']
    #bunches in ns for tune measurement
    if bunches_tune_shift != 0:
        bunch_in_ns_measure_tune = np.zeros(len_ns_measure_tune)
        bunch_in_ns_measure_tune[0] = 1
        bunches_in_ns_measure_tune = np.zeros((bunches_tune_shift-1)*(1+(len_ns_measure_tune-1))+1)
        for i in range(bunches_tune_shift-1):
            bunches_in_ns_measure_tune[(i*len_ns):((i+1)*len_ns)] = bunch_in_ns_measure_tune
        bunches_in_ns_measure_tune[-1] = 1


    #bunch in the ns 
    bunch_in_ns = np.zeros(len_ns)
    bunch_in_ns[0] = 1
    #PS batch in ns, that finish with one, because there is later the inj_space
    PS_batch_in_ns = np.zeros((bunches-1)*(1+(len_ns-1))+1)
    for i in range(bunches-1):
        PS_batch_in_ns[(i*len_ns):((i+1)*len_ns)] = bunch_in_ns
    PS_batch_in_ns[-1] = 1


    #vector that contains the number of remaining slots in empty gap that are more than 800ns 
    B1_more800ns = []
    index_INDIV1 = []

    B2_more800ns = []
    index_INDIV2 = []

    B1 = np.zeros(3564)
    B1_more800ns = np.zeros(len(slots_to_shift1))
    #if slots_init-1 !=slots_to_shift[0] and bunches_tune_shift!=0:
    B1_more800ns[0] = -(slots_to_shift1[0])-1
    # 3444 is the last bunch possible, after it there is the Abort Gap
    shift_from_ABK1 = 3444-slots_to_shift1[-1]
    B1_more800ns[-1] = shift_from_ABK1

    B2 = np.zeros(3564)
    B2_more800ns = np.zeros(len(slots_to_shift2))
    #if slots_init-1 !=slots_to_shift[0] and bunches_tune_shift!=0:
    B2_more800ns[0] = -(slots_to_shift2[0])-1
    shift_from_ABK2 = 3444-slots_to_shift2[-1]
    B2_more800ns[-1] = shift_from_ABK2

    #computation of the two beams and the B1_more800ns
    for i in np.arange(int((len(slots_to_shift1)-2)/2))+1:
        more800_ns1 = slots_to_shift1[2*i]-slots_to_shift1[2*i-1]-32+3
        more800_ns2 = slots_to_shift2[2*i]-slots_to_shift2[2*i-1]-32+3

        B1_more800ns[2*i-1] = more800_ns1
        B1_more800ns[2*i] = -more800_ns1

        B2_more800ns[2*i-1] = more800_ns2
        B2_more800ns[2*i] = -more800_ns2

        slot1 = np.mod(slots_to_shift1[2*i-2]+1,3564)
        slot2 = np.mod(slots_to_shift2[2*i-2]+1,3564)
        flag1 = 0
        flag2 = 0
        while flag1 == 0:
            if slots_to_shift1[2*i-1]-slots_to_shift1[2*i-2] == 3:
                index_INDIV1 = np.append(index_INDIV1, 2*i-2)
                B1[slot1] = 1
                flag1 = 1
            else:
                if bunches_tune_shift != 0:
                    if 2*i-2 ==0:
                        if slot1+len(bunches_in_ns_measure_tune)>3564:
                            B1[slot1:] = bunches_in_ns_measure_tune[:3564-slot1]
                            B1[:len(bunches_in_ns_measure_tune)-(3564-slot1)] = bunches_in_ns_measure_tune[3564-slot1:]
                        else:
                            B1[slot1:slot1+len(bunches_in_ns_measure_tune)] = bunches_in_ns_measure_tune
                        flag1 = 1
                    else:
                        if slot1+len(PS_batch_in_ns)>3564:
                            B1[slot1:] = PS_batch_in_ns[:3564-slot1]
                            B1[:len(PS_batch_in_ns)-(3564-slot1)] = PS_batch_in_ns[3564-slot1:]
                        else:
                            B1[slot1:slot1+len(PS_batch_in_ns)] = PS_batch_in_ns
                        slot1 += len(PS_batch_in_ns)+empties
                        if (slot1>=np.mod(slots_to_shift1[2*i-1],3564)-1+empties):
                            flag1 = 1
                else:
                    if slot1+len(PS_batch_in_ns)>3564:
                        B1[slot1:] = PS_batch_in_ns[:3564-slot1]
                        B1[:len(PS_batch_in_ns)-(3564-slot1)] = PS_batch_in_ns[3564-slot1:]
                    else:
                        B1[slot1:slot1+len(PS_batch_in_ns)] = PS_batch_in_ns
                    slot1 += len(PS_batch_in_ns)+empties
                    if (slot1>=np.mod(slots_to_shift1[2*i-1],3564)-1+empties):
                        flag1 = 1
        while flag2 == 0:
            if slots_to_shift2[2*i-1]-slots_to_shift2[2*i-2] == 3:
                index_INDIV2 = np.append(index_INDIV2, 2*i-2)
                B2[slot2] = 1
                flag2 = 1
            else:
                if bunches_tune_shift != 0:
                    if 2*i-2 ==0:
                        if slot2+len(bunches_in_ns_measure_tune)>3564:
                            B2[slot2:] = bunches_in_ns_measure_tune[:3564-slot2]
                            B2[:len(bunches_in_ns_measure_tune)-(3564-slot2)] = bunches_in_ns_measure_tune[3564-slot2:]
                        else:
                            B2[slot2:slot2+len(bunches_in_ns_measure_tune)] = bunches_in_ns_measure_tune
                        flag2 = 1
                    else:

                        if slot2+len(PS_batch_in_ns)>3564:
                            B2[slot2:] = PS_batch_in_ns[:3564-slot2]
                            B2[:len(PS_batch_in_ns)-(3564-slot2)] = PS_batch_in_ns[3564-slot2:]
                        else:
                            B2[slot2:slot2+len(PS_batch_in_ns)] = PS_batch_in_ns
                        slot2 += len(PS_batch_in_ns)+empties
                        if (slot2>=np.mod(slots_to_shift2[2*i-1],3564)-1+empties):
                            flag2 = 1
                else:
                    if slot2+len(PS_batch_in_ns)>3564:
                        B2[slot2:] = PS_batch_in_ns[:3564-slot2]
                        B2[:len(PS_batch_in_ns)-(3564-slot2)] = PS_batch_in_ns[3564-slot2:]
                    else:
                        B2[slot2:slot2+len(PS_batch_in_ns)] = PS_batch_in_ns
                    slot2 += len(PS_batch_in_ns)+empties
                    if (slot2>=np.mod(slots_to_shift2[2*i-1],3564)-1+empties):
                        flag2 = 1
    
    flag1 = 0
    slot1 = slots_to_shift1[-2]+1
    while flag1 == 0:
        if slot1+len(PS_batch_in_ns)>3564:
            B1[slot1:] = PS_batch_in_ns[:3564-slot1]
            B1[:len(PS_batch_in_ns)-(3564-slot1)] = PS_batch_in_ns[3564-slot1:]
        else:
            B1[slot1:slot1+len(PS_batch_in_ns)] = PS_batch_in_ns
        slot1 += len(PS_batch_in_ns)+empties
        if (slot1>=np.mod(slots_to_shift1[-1],3564)-1+empties):
            flag1 = 1

    flag2 = 0
    slot2 = slots_to_shift2[-2]+1
    while flag2 == 0:
        if slot2+len(PS_batch_in_ns)>3564:
            B2[slot2:] = PS_batch_in_ns[:3564-slot2]
            B2[:len(PS_batch_in_ns)-(3564-slot2)] = PS_batch_in_ns[3564-slot2:]
        else:
            B2[slot2:slot2+len(PS_batch_in_ns)] = PS_batch_in_ns
        slot2 += len(PS_batch_in_ns)+empties
        if (slot2>=np.mod(slots_to_shift2[-1],3564)-1+empties):
            flag2 = 1
    


    # MC method, you compute all the shifts that are possible, in order to have an improvement on the number of collisions
    vector = np.zeros(1*n_sample)
    # list that contains all the interesting information of the MonteCarlo
    list_informations= [ii for ii in vector]
    slots_to_shift_dict = { 'beam1' :slots_to_shift1,
                        'beam2': slots_to_shift2
                    }
    B_more800ns_dict = {
                    'beam1':B1_more800ns,
                    'beam2':B2_more800ns
    }
    beams_dict = {
            'beam1':B1,
            'beam2':B2
    }

    index_INDIV_dict = {
                    'beam1': index_INDIV1,
                    'beam2': index_INDIV2
    }
    
    add_info = np.append(slots_to_shift2,[tbtb.events_in_IPN(B1,B2,'IP1')[1],tbtb.events_in_IPN(B1,B2,'IP8')[1],tbtb.events_in_IPN(B1,B2,'IP2')[1]])
    informations = np.append(slots_to_shift1,[add_info[ii] for ii in range(len(add_info))])
    list_informations[0] = [ii for ii in informations]

    
    #take the informations of how many possible samples of the MonteCarlo can be considered
    count = 0
    if len(B1_more800ns>0) or len(B2_more800ns)>0:
        
        #repeat 100 times the same MC simulation with the same number of possible shifts
        for index in np.arange(1):
            # copy in order to not overwrite the initial data
            beams_new_dict = copy.copy(beams_dict)
            B_more800ns_new_dict = copy.copy(B_more800ns_dict)
            slots_to_shift_new_dict = copy.copy(slots_to_shift_dict)

            for iter in np.arange(n_sample):
                n_beam = np.random.randint(2)+1
                other_beam = [1,2][np.where([1,2]!=n_beam*np.ones(2))[0][0]]
                
                B_new = copy.copy(beams_new_dict[f'beam{n_beam}'])
                other_B = copy.copy(beams_new_dict[f'beam{other_beam}'])
                B_more800ns_new = copy.copy(B_more800ns_new_dict[f'beam{n_beam}'])
                slots_to_shift_new = copy.copy(slots_to_shift_new_dict[f'beam{n_beam}'])
                other_slots_to_shift = copy.copy(slots_to_shift_new_dict[f'beam{other_beam}'])
                index_INDIV= copy.copy(index_INDIV_dict[f'beam{n_beam}'])
                # random choice of the position where have the shift (could be optimized this, sth to do)
                pos_shift = random.sample([int(ii) for ii in np.where(B_more800ns_new!=np.zeros(len(B_more800ns_new)))[0]],1)[0]
                #pos_shift = np.random.randint(len(B_more800ns_new))
                
            
                # flag that in cases you cannot move all the INDIVs it surpass this step, and didn't move anything
                exit_flag_INDIV = 0
                if B_more800ns_new[pos_shift]!=0:
                    count += 1
                    #shift value is the sign of the possible movement of the SPS batch
                    shift_value = int(np.sign(B_more800ns_new[pos_shift]))
                    # if the pos shift is odd it means that the shift of SPS batch will be on right
                    if pos_shift%2 == 1:
                        if pos_shift-1 in index_INDIV:
                            # I'm moving all the INDIVs at the same time in order to not affect the number of collisions
                            # Pick all the INDIV except the pos_shift of the first INDIV
                            vec_INDIV = index_INDIV[~(pos_shift == index_INDIV+1)]+1
                            vec_INDIV = [int(i) for i in vec_INDIV]
                            #check if the movements are available
                            if np.array([B_more800ns_new[i] !=0 for i in vec_INDIV]).all():
                                # do the shift for every indiv with except of the pos_shift that we'll do later
                                # and change all the information of the slots_to_shift_new
                                for i in vec_INDIV:
                                    pos_shift_2 = i
                                    
                                    B_new[slots_to_shift_new[pos_shift_2-1]:slots_to_shift_new[pos_shift_2]] = np.roll(B_new[slots_to_shift_new[pos_shift_2-1]:slots_to_shift_new[pos_shift_2]],shift_value)
                                    slots_to_shift_new[pos_shift_2-1] += shift_value
                                    B_more800ns_new[pos_shift_2] -= shift_value
                                    B_more800ns_new[pos_shift_2-1] -= shift_value
                                    if pos_shift_2!=len(B_more800ns_new)-1:
                                        B_more800ns_new[pos_shift_2+shift_value] += shift_value
                                    if pos_shift_2 != 1:
                                        B_more800ns_new[pos_shift_2-2] += shift_value
                                    slots_to_shift_new[pos_shift_2] += shift_value
                    
                        
                            else:
                                #if you cannot change the INDIV
                                exit_flag_INDIV = 1

                        #if we haven't pick the INDIV in pos_shift and also if we picked it, we could move all the other INDIVs
                        if exit_flag_INDIV == 0:

                            #shift of B1
                            #print(shift_value)
                            #print([slots_to_shift_new])
                            B_new[slots_to_shift_new[pos_shift-1]:slots_to_shift_new[pos_shift]] = np.roll(B_new[slots_to_shift_new[pos_shift-1]:slots_to_shift_new[pos_shift]],shift_value)
                            
                            if pos_shift!= 1:
                                B_more800ns_new[pos_shift-2] += shift_value
                            #change of all the parameters
                            slots_to_shift_new[pos_shift-1] += shift_value
                            B_more800ns_new[pos_shift] -= shift_value
                            B_more800ns_new[pos_shift-1] -= shift_value
                            if pos_shift!=len(B_more800ns_new)-1:
                                B_more800ns_new[pos_shift+shift_value] += shift_value
                            slots_to_shift_new[pos_shift] += shift_value
                    # shift of SPS batch will be on right, we will do the same passages but considering the opposite direction
                    else:
                        if pos_shift in index_INDIV:
                            # I'm moving all the INDIVs at the same time in order to not affect the number of collisions
                            # Pick all the INDIV except the pos_shift of the first INDIV
                            vec_INDIV = index_INDIV[~(pos_shift == index_INDIV)]
                            vec_INDIV = [int(i) for i in vec_INDIV]
                            #check if the movements are available
                            if np.array([B_more800ns_new[i] !=0 for i in vec_INDIV]).all():
                                # do the shift for every indiv with except of the pos_shift that we'll do later
                                # and change all the information of the slots_to_shift_new
                                for i in vec_INDIV:
                                    pos_shift_2 = i
                                    
                                    B_new[slots_to_shift_new[pos_shift_2]:slots_to_shift_new[pos_shift_2+1]] = np.roll(B_new[slots_to_shift_new[pos_shift_2]:slots_to_shift_new[pos_shift_2+1]],shift_value)
                                    
                                    slots_to_shift_new[pos_shift_2+1] += shift_value
                                    B_more800ns_new[pos_shift_2] -= shift_value
                                    B_more800ns_new[pos_shift_2+1] -= shift_value
                                    if pos_shift_2!=len(B_more800ns_new)-2:
                                        B_more800ns_new[pos_shift_2+2] += shift_value 
                                    if pos_shift_2 !=0:   
                                        B_more800ns_new[pos_shift_2+shift_value] += shift_value
                                    slots_to_shift_new[pos_shift_2] += shift_value
                                  
                            else:
                                #if you cannot change the INDIV
                                exit_flag_INDIV = 1
                        #if we haven't pick the INDIV in pos_shift and also if we picked it, we could move all the other INDIVs        
                        if exit_flag_INDIV == 0:
                            #shift of B1

                            #print(shift_value)
                            #print([slots_to_shift_new])
                            B_new[slots_to_shift_new[pos_shift]:slots_to_shift_new[pos_shift+1]] = np.roll(B_new[slots_to_shift_new[pos_shift]:slots_to_shift_new[pos_shift+1]],shift_value)
                            
                            #check in orrder to see if the there are bunches for tune measurement, in case affermative if they collide!
                            if pos_shift!=0:
                                B_more800ns_new[pos_shift+shift_value] += shift_value
                            #change of all the parameters
                            slots_to_shift_new[pos_shift+1] += shift_value
                            B_more800ns_new[pos_shift] -= shift_value
                            B_more800ns_new[pos_shift+1] -= shift_value
                            if pos_shift!=len(B_more800ns_new)-2:
                                B_more800ns_new[pos_shift+2] += shift_value    
                            slots_to_shift_new[pos_shift] += shift_value

                    #once done the shift save the information that you need in order to restart the MonteCarlo
                    beams_new_dict[f'beam{n_beam}'] = B_new
                    B_more800ns_new_dict[f'beam{n_beam}'] = B_more800ns_new
                    slots_to_shift_new_dict[f'beam{n_beam}'] = slots_to_shift_new
                    if n_beam == 1:
                        events_ATLAS_new = tbtb.events_in_IPN(B_new,other_B,'IP1')[1]
                        events_LHCb_new = tbtb.events_in_IPN(B_new,other_B,'IP8')[1]
                        events_ALICE_new = tbtb.events_in_IPN(B_new,other_B,'IP2')[1]
                        add_info = np.append(other_slots_to_shift,[events_ATLAS_new,events_LHCb_new,events_ALICE_new])
                        informations_new = np.append(slots_to_shift_new,[add_info[ii] for ii in range(len(add_info))])
                    else:
                        events_ATLAS_new = tbtb.events_in_IPN(other_B,B_new,'IP1')[1]
                        events_LHCb_new = tbtb.events_in_IPN(other_B,B_new,'IP8')[1]
                        events_ALICE_new = tbtb.events_in_IPN(other_B,B_new,'IP2')[1]
                        add_info = np.append(slots_to_shift_new,[events_ATLAS_new,events_LHCb_new,events_ALICE_new])
                        informations_new = np.append(other_slots_to_shift,[add_info[ii] for ii in range(len(add_info))])
                    list_informations[count-1] = [ii for ii in informations_new]
            
        # put all the informations, sectioned, that we want in the DataFrame 
        del list_informations[count:]
        df_fill_schemes2 = pd.DataFrame(index = np.arange(count))
        df_fill_schemes2['informations'] = list_informations
        df_fill_schemes = pd.DataFrame(np.unique(df_fill_schemes2), columns=df_fill_schemes2.columns)
        df_help = pd.DataFrame(df_fill_schemes2["informations"].values.tolist())
        df_fill_schemes = pd.DataFrame(index = np.arange(len(df_help.index)))
        vec_string_slot = df_help[df_help.columns[0:len(slots_to_shift1)]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1)
        help_vec = [list(ii.split(" ")) for ii in vec_string_slot]
        vec_string_slot2 = df_help[df_help.columns[len(slots_to_shift1):(len(slots_to_shift1)+len(slots_to_shift2))]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1)
        help_vec2 = [list(ii.split(" ")) for ii in vec_string_slot2]
        df_fill_schemes['slots_to_shift1'] = [[int(eval(ii)) for ii in i] for i in help_vec]
        df_fill_schemes['slots_to_shift2'] = [[int(eval(ii)) for ii in i] for i in help_vec2]
        #df_fill_schemes['distance'] = [LA.norm(ii-slots_to_shift_check[0]) for ii in df_fill_schemes['slots_to_shift']]
        df_fill_schemes['events_ATLAS/CMS'] = df_help.iloc[:,-3]
        df_fill_schemes['events_LHCb'] = df_help.iloc[:,-2]
        df_fill_schemes['events_ALICE'] = df_help.iloc[:,-1]

    return df_fill_schemes

def MC_shift_preserve_oneIP(slots_to_shift1, slots_to_shift2, shift_IP, n_sample, bunches, bunches_ns = 25, bunches_tune_shift = 12, bunches_tune_shift_ns = 25, empties = 7, shift_from_ABK = 0):
    '''
    a MC method to shift the trains in order to get a configuration with more collisions in LHCb or ALICE, the shift will done of 1 slot 
    singularly at every train, and the script will save all the configurations found
    Hypothesis:
        - considering that the wo beams are so similar, in "slots_to_shift" are represented the (first slot-1) and the (last slot +1) for every SPS_batch, only 
        for one of thetwo beams, which one is represented in "beam". The information of the position of the "twelve" bunches fot the other beam is represented in "slots_init"
        - for sake of semplicity we will call B1 as the beam that is represented by slots_to_shift
    Args:
        slots_to_shift: positions of the (first element -1) and (last element+1) of every SPS_batch for B1
        slots_init: initial position of the twelve bunches of the other beam (B2)
        beam: give the number of the beam connected with slots_to_shift (can be deleted)
        bunches: number of bunches for every PS batch (can be computed I think)
        slots_to_shift_check: the beam with whom compare the distance of MC simulation
        empties: number of empty slots for every empty gap between PS_batches, set as 7
        twelve: where  the "twelve" bunches of the detached beam, set as 0 (is this slots_to_shift-11?) control see if works
        ABK: free space before the Abort Gap that has the beam in case is not perfectly positioned in the Abort Gap Keeper
    Return: 
        pd.DataFrame: that contains the slots_to_shift, distance wrt slots_to_shift_check, slots_init, beam, events_in_ALICE, events_in_LHCb
        of all the MC simulation
    '''

    #dictionary that converts time in space
    Dict_ns_len = {'25ns': 1, '50ns':2, '75ns':3,'100ns':4}
    assert bunches_ns in [25,50,75,100], "bunches_ns accepted in LHC are 25 or 50 or 75 or 100"
    len_ns = Dict_ns_len[f'{bunches_ns}ns']
    assert bunches_tune_shift_ns in [25,50,75,100], "bunches_tune_shift_ns accepted in LHC are 25 or 50 or 75 or 100"
    len_ns_measure_tune = Dict_ns_len[f'{bunches_tune_shift_ns}ns']
    #bunches in ns for tune measurement
    if bunches_tune_shift != 0:
        bunch_in_ns_measure_tune = np.zeros(len_ns_measure_tune)
        bunch_in_ns_measure_tune[0] = 1
        bunches_in_ns_measure_tune = np.zeros((bunches_tune_shift-1)*(1+(len_ns_measure_tune-1))+1)
        for i in range(bunches_tune_shift-1):
            bunches_in_ns_measure_tune[(i*len_ns):((i+1)*len_ns)] = bunch_in_ns_measure_tune
        bunches_in_ns_measure_tune[-1] = 1


    #bunch in the ns 
    bunch_in_ns = np.zeros(len_ns)
    bunch_in_ns[0] = 1
    #PS batch in ns, that finish with one, because there is later the inj_space
    PS_batch_in_ns = np.zeros((bunches-1)*(1+(len_ns-1))+1)
    for i in range(bunches-1):
        PS_batch_in_ns[(i*len_ns):((i+1)*len_ns)] = bunch_in_ns
    PS_batch_in_ns[-1] = 1


    #vector that contains the number of remaining slots in empty gap that are more than 800ns 
    B1_more800ns = []
    index_INDIV1 = []

    B2_more800ns = []
    index_INDIV2 = []

    B1 = np.zeros(3564)
    B1_more800ns = np.zeros(len(slots_to_shift1))
    #if slots_init-1 !=slots_to_shift[0] and bunches_tune_shift!=0:
    B1_more800ns[0] = -(slots_to_shift1[0])-1
    # 3444 is the last bunch possible, after it there is the Abort Gap
    shift_from_ABK1 = 3444-slots_to_shift1[-1]
    B1_more800ns[-1] = shift_from_ABK1

    B2 = np.zeros(3564)
    B2_more800ns = np.zeros(len(slots_to_shift2))
    #if slots_init-1 !=slots_to_shift[0] and bunches_tune_shift!=0:
    B2_more800ns[0] = -(slots_to_shift2[0])-1
    shift_from_ABK2 = 3444-slots_to_shift2[-1]
    B2_more800ns[-1] = shift_from_ABK2

    #computation of the two beams and the B1_more800ns
    for i in np.arange(int((len(slots_to_shift1)-2)/2))+1:
        more800_ns1 = slots_to_shift1[2*i]-slots_to_shift1[2*i-1]-32+3
        more800_ns2 = slots_to_shift2[2*i]-slots_to_shift2[2*i-1]-32+3

        B1_more800ns[2*i-1] = more800_ns1
        B1_more800ns[2*i] = -more800_ns1

        B2_more800ns[2*i-1] = more800_ns2
        B2_more800ns[2*i] = -more800_ns2

        slot1 = np.mod(slots_to_shift1[2*i-2]+1,3564)
        slot2 = np.mod(slots_to_shift2[2*i-2]+1,3564)
        flag1 = 0
        flag2 = 0
        while flag1 == 0:
            if slots_to_shift1[2*i-1]-slots_to_shift1[2*i-2] == 3:
                index_INDIV1 = np.append(index_INDIV1, 2*i-2)
                B1[slot1] = 1
                flag1 = 1
            else:
                if bunches_tune_shift != 0:
                    if 2*i-2 ==0:
                        if slot1+len(bunches_in_ns_measure_tune)>3564:
                            B1[slot1:] = bunches_in_ns_measure_tune[:3564-slot1]
                            B1[:len(bunches_in_ns_measure_tune)-(3564-slot1)] = bunches_in_ns_measure_tune[3564-slot1:]
                        else:
                            B1[slot1:slot1+len(bunches_in_ns_measure_tune)] = bunches_in_ns_measure_tune
                        flag1 = 1
                    else:
                        if slot1+len(PS_batch_in_ns)>3564:
                            B1[slot1:] = PS_batch_in_ns[:3564-slot1]
                            B1[:len(PS_batch_in_ns)-(3564-slot1)] = PS_batch_in_ns[3564-slot1:]
                        else:
                            B1[slot1:slot1+len(PS_batch_in_ns)] = PS_batch_in_ns
                        slot1 += len(PS_batch_in_ns)+empties
                        if (slot1>=np.mod(slots_to_shift1[2*i-1],3564)-1+empties):
                            flag1 = 1
                else:
                    if slot1+len(PS_batch_in_ns)>3564:
                        B1[slot1:] = PS_batch_in_ns[:3564-slot1]
                        B1[:len(PS_batch_in_ns)-(3564-slot1)] = PS_batch_in_ns[3564-slot1:]
                    else:
                        B1[slot1:slot1+len(PS_batch_in_ns)] = PS_batch_in_ns
                    slot1 += len(PS_batch_in_ns)+empties
                    if (slot1>=np.mod(slots_to_shift1[2*i-1],3564)-1+empties):
                        flag1 = 1
        while flag2 == 0:
            if slots_to_shift2[2*i-1]-slots_to_shift2[2*i-2] == 3:
                index_INDIV2 = np.append(index_INDIV2, 2*i-2)
                B2[slot2] = 1
                flag2 = 1
            else:
                if bunches_tune_shift != 0:
                    if 2*i-2 ==0:
                        if slot2+len(bunches_in_ns_measure_tune)>3564:
                            B2[slot2:] = bunches_in_ns_measure_tune[:3564-slot2]
                            B2[:len(bunches_in_ns_measure_tune)-(3564-slot2)] = bunches_in_ns_measure_tune[3564-slot2:]
                        else:
                            B2[slot2:slot2+len(bunches_in_ns_measure_tune)] = bunches_in_ns_measure_tune
                        flag2 = 1
                    else:

                        if slot2+len(PS_batch_in_ns)>3564:
                            B2[slot2:] = PS_batch_in_ns[:3564-slot2]
                            B2[:len(PS_batch_in_ns)-(3564-slot2)] = PS_batch_in_ns[3564-slot2:]
                        else:
                            B2[slot2:slot2+len(PS_batch_in_ns)] = PS_batch_in_ns
                        slot2 += len(PS_batch_in_ns)+empties
                        if (slot2>=np.mod(slots_to_shift2[2*i-1],3564)-1+empties):
                            flag2 = 1
                else:
                    if slot2+len(PS_batch_in_ns)>3564:
                        B2[slot2:] = PS_batch_in_ns[:3564-slot2]
                        B2[:len(PS_batch_in_ns)-(3564-slot2)] = PS_batch_in_ns[3564-slot2:]
                    else:
                        B2[slot2:slot2+len(PS_batch_in_ns)] = PS_batch_in_ns
                    slot2 += len(PS_batch_in_ns)+empties
                    if (slot2>=np.mod(slots_to_shift2[2*i-1],3564)-1+empties):
                        flag2 = 1
    
    flag1 = 0
    slot1 = slots_to_shift1[-2]+1
    while flag1 == 0:
        if slot1+len(PS_batch_in_ns)>3564:
            B1[slot1:] = PS_batch_in_ns[:3564-slot1]
            B1[:len(PS_batch_in_ns)-(3564-slot1)] = PS_batch_in_ns[3564-slot1:]
        else:
            B1[slot1:slot1+len(PS_batch_in_ns)] = PS_batch_in_ns
        slot1 += len(PS_batch_in_ns)+empties
        if (slot1>=np.mod(slots_to_shift1[-1],3564)-1+empties):
            flag1 = 1

    flag2 = 0
    slot2 = slots_to_shift2[-2]+1
    while flag2 == 0:
        if slot2+len(PS_batch_in_ns)>3564:
            B2[slot2:] = PS_batch_in_ns[:3564-slot2]
            B2[:len(PS_batch_in_ns)-(3564-slot2)] = PS_batch_in_ns[3564-slot2:]
        else:
            B2[slot2:slot2+len(PS_batch_in_ns)] = PS_batch_in_ns
        slot2 += len(PS_batch_in_ns)+empties
        if (slot2>=np.mod(slots_to_shift2[-1],3564)-1+empties):
            flag2 = 1
    


    # MC method, you compute all the shifts that are possible, in order to have an improvement on the number of collisions
    vector = np.zeros(100*n_sample)
    # list that contains all the interesting information of the MonteCarlo, slots_to_shift moduled!!
    list_informations= [ii for ii in vector]
    slots_to_shift_dict = { 'beam1' :slots_to_shift1,
                        'beam2': slots_to_shift2
                    }
    B_more800ns_dict = {
                    'beam1':B1_more800ns,
                    'beam2':B2_more800ns
    }
    beams_dict = {
            'beam1':B1,
            'beam2':B2
    }

    index_INDIV_dict = {
                    'beam1': index_INDIV1,
                    'beam2': index_INDIV2
    }
    
    print(index_INDIV_dict)

    add_info = np.append(slots_to_shift2,[tbtb.events_in_IPN(np.roll(B1,0),B2,'IP1')[1],tbtb.events_in_IPN(np.roll(B1,0),B2,'IP8')[1],tbtb.events_in_IPN(np.roll(B1,0),B2,'IP2')[1]])
    informations = np.append(slots_to_shift1,[add_info[ii] for ii in range(len(add_info))])
    list_informations[0] = [ii for ii in informations]

    
    #take the informations of how many possible samples of the MonteCarlo can be considered
    count = 1
    
    if len(B1_more800ns>0) or len(B2_more800ns)>0:
        
        #repeat 100 times the same MC simulation with the same number of possible shifts
        for index in np.arange(100):
            # copy in order to not overwrite the initial data
            slots_to_shift1_new = copy.copy(slots_to_shift_dict['beam1'])
            slots_to_shift2_new = copy.copy(slots_to_shift_dict['beam2'])
            B1_new = copy.copy(beams_dict['beam1'])
            B2_new = copy.copy(beams_dict['beam2'])
            B1_more800ns_new = copy.copy(B_more800ns_dict['beam1'])
            B2_more800ns_new = copy.copy(B_more800ns_dict['beam2'])
            index_INDIV1_new= copy.copy(index_INDIV_dict['beam1'])
            index_INDIV2_new= copy.copy(index_INDIV_dict['beam2'])
            
            #domani check e cambia tutti pos_shiftB1
            for iter in np.arange(n_sample):
                pos_shiftB1 = np.random.randint(len(B1_more800ns))
                #
                pos_shiftB2 = []
                if pos_shiftB1%2==1:
                    slot1_init = np.mod(slots_to_shift1_new[pos_shiftB1-1]+shift_IP,3564)
                    slot1_fin = np.mod(slots_to_shift1_new[pos_shiftB1]+shift_IP,3564)
                    for i in range(len(slots_to_shift2_new)):
                        if i%2 == 1:
                            if slots_to_shift2_new[i-1]>=slot1_init and slots_to_shift2_new[i-1]<=slot1_fin:
                                pos_shiftB2 = np.append(pos_shiftB2,i)
                            if slots_to_shift2_new[i-1]<slot1_init and slots_to_shift2_new[i]>=slot1_init:
                                pos_shiftB2 = np.append(pos_shiftB2,i)
                else:
                    slot1_init = np.mod(slots_to_shift1_new[pos_shiftB1]+shift_IP,3564)
                    slot1_fin = np.mod(slots_to_shift1_new[pos_shiftB1+1]+shift_IP,3564)
                    for i in range(len(slots_to_shift2_new)):
                        if i%2 == 0:
                            if slots_to_shift2_new[i]>=slot1_init and slots_to_shift2_new[i]<=slot1_fin:
                                pos_shiftB2 = np.append(pos_shiftB2,i)
                            if slots_to_shift2_new[i]<slot1_init and slots_to_shift2_new[i+1]>=slot1_init:
                                pos_shiftB2 = np.append(pos_shiftB2,i)


                pos_shiftB2 = [int(ii) for ii in pos_shiftB2]
                #print(pos_shiftB1)
                #print(pos_shiftB2)
                if len(pos_shiftB2)>0:
                    if B1_more800ns[pos_shiftB1]!=0 and np.array([B2_more800ns[ii]!=0 for ii in pos_shiftB2]).all():
                        #check if all the positions are even or odd, and now, the check is done, so cheange both...careful with INDIV.. understand how to extend!!!
                        #print(B1_more800ns[pos_shiftB1])
                        #print([B2_more800ns[ii] for ii in pos_shiftB2])
                        count += 1
                        #for c in [1,2]:
                        flag_batch_divided = 0
                        reset_init1 = 0
                        reset_init2 = 0
                        #n_beam = c
                        #other_beam = [1,2][np.where([1,2]!=n_beam*np.ones(2))[0][0]]
                        
                        
                        
                        # random choice of the position where have the shift (could be optimized this, sth to do)
                        #pos_shift = np.random.randint(len(B_more800ns_new))
                        
                    
                        # flag that in cases you cannot move all the INDIVs it surpass this step, and didn't move anything
                        exit_flag_INDIV = 0
                    
                    ####
                    #ATTENTION: SOMTHING TO ADD, YOU ARE CONSIDERING THAT IF YOU PICK AN INDIV FOR B1 YOU ARE MOVING ALL TOGETHER AND SHIFT MORE TIMES B2, BUT....
                    # YOU HAVE TO CONSIDER ALSOT ANOTHER CASE!!! THAT SHIFTING B1 THEN CIN ORDER TO PICK THE POSITION OF B2 TO SHIFT THERE IS ALSO AN INDIV, IN THIS CASE YOU 
                    # HAVE TO MOVE ALL THE INDIVS OF B2 AT THE SAME TIME AND GO BACK IN B1 AND MOVE ALSO THE ONES THAT ARE CORRISPECTIVE WITH THE INDIV OF B2 THAT YOU MOVED!! IF 
                    # ALWAYS ALL THIS MOVEMENT IS POSSIBLE!!!!!
                    ####
                        
                        #shift value is the sign of the possible movement of the SPS batch
                        shift_value = int(np.sign(B1_more800ns_new[pos_shiftB1]))
                        # if the pos shift is odd it means that the shift of SPS batch will be on right
                        if pos_shiftB1%2 == 1:
                            #if we have a batch that is "divided" in the sense that begin at the end of the beam and finish at the beggining
                            # we shift in order to not have problem and then we restore at the initial configuration
                            if slots_to_shift1_new[pos_shiftB1-1]> slots_to_shift1_new[pos_shiftB1]:
                                set_to_zero1 = (3564-slots_to_shift1_new[pos_shiftB1-1])
                                slots_to_shift1_new = [int(ii) for ii in np.mod(slots_to_shift1_new + set_to_zero1*np.ones(len(slots_to_shift1_new)),3564)]
                                
                                B1_new = np.roll(B1_new,set_to_zero1)
                                flag_batch_divided = 1
                                reset_init1 += -set_to_zero1
                            for j in pos_shiftB2:
                                if slots_to_shift2_new[j-1]> slots_to_shift2_new[j]:
                                    set_to_zero2 = (3564-slots_to_shift2_new[j-1])
                                    slots_to_shift2_new = [int(ii) for ii in np.mod(slots_to_shift2_new + set_to_zero2*np.ones(len(slots_to_shift2_new)),3564)]
                                    
                                    B2_new = np.roll(B2_new,set_to_zero2)
                                    flag_batch_divided = 1
                                    reset_init2 += -set_to_zero2
                            if pos_shiftB1-1 in index_INDIV1_new:
                                # I'm moving all the INDIVs at the same time in order to not affect the number of collisions
                                # Pick all the INDIV except the pos_shift of the first INDIV
                                vec_INDIV = index_INDIV1_new[~(pos_shiftB1 == index_INDIV1_new+1)]+1
                                vec_INDIV = [int(i) for i in vec_INDIV]
                                #check if the movements are available
                                if np.array([B1_more800ns_new[i] !=0 for i in vec_INDIV]).all():#### HERE YOU SHOULD ALSO CHECK THE LENGTH OF POS_SHIFTB2_2 AND IF YOU CAN MOVE THOSE!!!
                                    # do the shift for every indiv with except of the pos_shift that we'll do later
                                    # and change all the information of the slots_to_shift_new
                                    for i in vec_INDIV:
                                        pos_shiftB1_2 = i
                                        if pos_shiftB1_2%2==1:
                                            slot1_init = np.mod(slots_to_shift1_new[pos_shiftB1_2-1]+shift_IP,3564)
                                            slot1_fin = np.mod(slots_to_shift1_new[pos_shiftB1_2]+shift_IP,3564)
                                            for j in range(len(slots_to_shift2_new)):
                                                if j%2 == 1:
                                                    if slots_to_shift2_new[j-1]>=slot1_init and slots_to_shift2_new[j-1]<=slot1_fin:
                                                        pos_shiftB2_2 = np.append(pos_shiftB2_2,j)
                                                    if slots_to_shift2_new[j-1]<slot1_init and slots_to_shift2_new[j]>=slot1_init:
                                                        pos_shiftB2_2 = np.append(pos_shiftB2_2,j)
                                        else:
                                            slot1_init = np.mod(slots_to_shift1_new[pos_shiftB1_2]+shift_IP,3564)
                                            slot1_fin = np.mod(slots_to_shift1_new[pos_shiftB1_2+1]+shift_IP,3564)
                                            for j in range(len(slots_to_shift2_new)):
                                                if j%2 == 0:
                                                    if slots_to_shift2_new[j]>=slot1_init and slots_to_shift2_new[j]<=slot1_fin:
                                                        pos_shiftB2_2 = np.append(pos_shiftB2_2,j)
                                                    if slots_to_shift2_new[j]<slot1_init and slots_to_shift2_new[j+1]>=slot1_init:
                                                        pos_shiftB2_2 = np.append(pos_shiftB2_2,j)

                                        pos_shiftB2_2 = [int(ii) for ii in pos_shiftB2_2]
                                        #if we have a batch that is "divided" in the sense that begin at the end of the beam and finish at the beggining
                                        # we shift in order to not have problem and then we restore at the initial configuration, the shift were summed in case we are moving
                                        # all the INDIV at the same time, we have to  be careful
                                        if slots_to_shift1_new[pos_shiftB1_2-1]> slots_to_shift1_new[pos_shiftB1_2]:
                                            set_to_zero1 = (3564-slots_to_shift1_new[pos_shiftB1_2-1])
                                            slots_to_shift1_new = [int(ii) for ii in np.mod(slots_to_shift1_new + set_to_zero1*np.ones(len(slots_to_shift1_new)),3564)]
                                            
                                            B1_new = np.roll(B1_new,set_to_zero1)
                                            flag_batch_divided = 1
                                            reset_init1 += -set_to_zero1
                                        for j in pos_shiftB2_2:
                                            if slots_to_shift2_new[j-1]> slots_to_shift2_new[j]:
                                                set_to_zero2 = (3564-slots_to_shift2_new[j-1])
                                                slots_to_shift2_new = [int(ii) for ii in np.mod(slots_to_shift2_new + set_to_zero2*np.ones(len(slots_to_shift2_new)),3564)]
                                            
                                                B2_new = np.roll(B2_new,set_to_zero2)
                                                flag_batch_divided = 1
                                                reset_init2 += -set_to_zero2
                                        
                                        B1_new[slots_to_shift1_new[pos_shiftB1_2-1]:slots_to_shift1_new[pos_shiftB1_2]] = np.roll(B1_new[slots_to_shift1_new[pos_shiftB1_2-1]:slots_to_shift1_new[pos_shiftB1_2]],shift_value)
                                        for j in pos_shiftB2_2:
                                            B2_new[slots_to_shift2_new[j-1]:slots_to_shift2_new[j]] = np.roll(B2_new[slots_to_shift2_new[j-1]:slots_to_shift2_new[j]],shift_value)
                                            slots_to_shift2_new[j-1] += shift_value
                                            B2_more800ns_new[j] -= shift_value
                                            B2_more800ns_new[j-1] -= shift_value
                                            if j!=len(B2_more800ns_new)-1:
                                                B2_more800ns_new[j+shift_value] += shift_value
                                            if j != 1:
                                                B2_more800ns_new[j-2] += shift_value
                                            slots_to_shift2_new[j] += shift_value
                        
                                        slots_to_shift1_new[pos_shiftB1_2-1] += shift_value
                                        B1_more800ns_new[pos_shiftB1_2] -= shift_value
                                        B1_more800ns_new[pos_shiftB1_2-1] -= shift_value
                                        if pos_shiftB1_2!=len(B1_more800ns_new)-1:
                                            B1_more800ns_new[pos_shiftB1_2+shift_value] += shift_value
                                        if pos_shiftB1_2 != 1:
                                            B1_more800ns_new[pos_shiftB1_2-2] += shift_value
                                        slots_to_shift1_new[pos_shiftB1_2] += shift_value
                        
                            
                                else:
                                    #if you cannot change the INDIV
                                    exit_flag_INDIV = 1

                            #if we haven't pick the INDIV in pos_shift and also if we picked it, we could move all the other INDIVs
                            if exit_flag_INDIV == 0:

                                #shift of B1
                                #print(shift_value)
                                #print([slots_to_shift_new])
                                B1_new[slots_to_shift1_new[pos_shiftB1-1]:slots_to_shift1_new[pos_shiftB1]] = np.roll(B1_new[slots_to_shift1_new[pos_shiftB1-1]:slots_to_shift1_new[pos_shiftB1]],shift_value)
                                for j in pos_shiftB2:
                                    B2_new[slots_to_shift2_new[j-1]:slots_to_shift2_new[j]] = np.roll(B2_new[slots_to_shift2_new[j-1]:slots_to_shift2_new[j]],shift_value)
                                    slots_to_shift2_new[j-1] += shift_value
                                    B2_more800ns_new[j] -= shift_value
                                    B2_more800ns_new[j-1] -= shift_value
                                    if j!=len(B2_more800ns_new)-1:
                                        B2_more800ns_new[j+shift_value] += shift_value
                                    if j != 1:
                                        B2_more800ns_new[j-2] += shift_value
                                    slots_to_shift2_new[j] += shift_value
                        
                                #change of all the parameters
                                slots_to_shift1_new[pos_shiftB1-1] += shift_value
                                B1_more800ns_new[pos_shiftB1] -= shift_value
                                B1_more800ns_new[pos_shiftB1-1] -= shift_value
                                if pos_shiftB1!= 1:
                                    B1_more800ns_new[pos_shiftB1-2] += shift_value
                                if pos_shiftB1!=len(B1_more800ns_new)-1:
                                    B1_more800ns_new[pos_shiftB1+shift_value] += shift_value
                                slots_to_shift1_new[pos_shiftB1] += shift_value
                        # shift of SPS batch will be on right, we will do the same passages but considering the opposite direction
                        else:
                            #if we have a batch that is "divided" in the sense that begin at the end of the beam and finish at the beggining
                            # we shift in order to not have problem and then we restore at the initial configuration
                            if slots_to_shift1_new[pos_shiftB1]> slots_to_shift1_new[pos_shiftB1+1]:
                                set_to_zero1 = (3564-slots_to_shift1_new[pos_shiftB1])
                                slots_to_shift1_new = [int(ii) for ii in np.mod(slots_to_shift1_new + set_to_zero1*np.ones(len(slots_to_shift1_new)),3564)]
                                
                                B1_new = np.roll(B1_new,set_to_zero1)
                                flag_batch_divided = 1
                                reset_init1 += -set_to_zero1
                            for j in pos_shiftB2:
                                if slots_to_shift2_new[j-1]> slots_to_shift2_new[j]:
                                    set_to_zero2 = (3564-slots_to_shift2_new[j-1])
                                    slots_to_shift2_new = [int(ii) for ii in np.mod(slots_to_shift2_new + set_to_zero2*np.ones(len(slots_to_shift2_new)),3564)]
                                    
                                    B2_new = np.roll(B2_new,set_to_zero2)
                                    flag_batch_divided = 1
                                    reset_init2 += -set_to_zero2
                            if pos_shiftB1 in index_INDIV1_new:
                                # I'm moving all the INDIVs at the same time in order to not affect the number of collisions
                                # Pick all the INDIV except the pos_shift of the first INDIV
                                vec_INDIV = index_INDIV1_new[~(pos_shiftB1 == index_INDIV1_new)]
                                vec_INDIV = [int(i) for i in vec_INDIV]
                                #check if the movements are available
                                if np.array([B1_more800ns_new[i] !=0 for i in vec_INDIV]).all():
                                    # do the shift for every indiv with except of the pos_shift that we'll do later
                                    # and change all the information of the slots_to_shift_new
                                    for i in vec_INDIV:
                                        pos_shiftB1_2 = i
                                        if pos_shiftB1_2%2==1:
                                            slot1_init = np.mod(slots_to_shift1_new[pos_shiftB1_2-1]+shift_IP,3564)
                                            slot1_fin = np.mod(slots_to_shift1_new[pos_shiftB1_2]+shift_IP,3564)
                                            for j in range(len(slots_to_shift2_new)):
                                                if j%2 == 1:
                                                    if slots_to_shift2_new[j-1]>=slot1_init and slots_to_shift2_new[j-1]<=slot1_fin:
                                                        pos_shiftB2_2 = np.append(pos_shiftB2_2,j)
                                                    if slots_to_shift2_new[j-1]<slot1_init and slots_to_shift2_new[j]>=slot1_init:
                                                        pos_shiftB2_2 = np.append(pos_shiftB2_2,j)
                                        else:
                                            slot1_init = np.mod(slots_to_shift1_new[pos_shiftB1_2]+shift_IP,3564)
                                            slot1_fin = np.mod(slots_to_shift1_new[pos_shiftB1_2+1]+shift_IP,3564)
                                            for j in range(len(slots_to_shift2_new)):
                                                if j%2 == 0:
                                                    if slots_to_shift2_new[j]>=slot1_init and slots_to_shift2_new[j]<=slot1_fin:
                                                        pos_shiftB2_2 = np.append(pos_shiftB2_2,j)
                                                    if slots_to_shift2_new[j]<slot1_init and slots_to_shift2_new[j+1]>=slot1_init:
                                                        pos_shiftB2_2 = np.append(pos_shiftB2_2,j)

                                        pos_shiftB2_2 = [int(ii) for ii in pos_shiftB2_2]
                                        #if we have a batch that is "divided" in the sense that begin at the end of the beam and finish at the beggining
                                        # we shift in order to not have problem and then we restore at the initial configuration, the shift were summed in case we are moving
                                        # all the INDIV at the same time, we have to  be careful
                                        if slots_to_shift1_new[pos_shiftB1_2]> slots_to_shift1_new[pos_shiftB1_2+1]:
                                            set_to_zero = (3564-slots_to_shift1_new[ppos_shiftB1_2])
                                            slots_to_shift1_new = [int(ii) for ii in np.mod(slots_to_shift1_new + set_to_zero*np.ones(len(slots_to_shift1_new)),3564)]
                                            
                                            B1_new = np.roll(B1_new,set_to_zero)
                                            flag_batch_divided = 1
                                            reset_init += -set_to_zero
                                        for j in pos_shiftB2_2:
                                            if slots_to_shift2_new[j-1]> slots_to_shift2_new[j]:
                                                set_to_zero2 = (3564-slots_to_shift2_new[j-1])
                                                slots_to_shift2_new = [int(ii) for ii in np.mod(slots_to_shift2_new + set_to_zero2*np.ones(len(slots_to_shift2_new)),3564)]
                                                
                                                B2_new = np.roll(B2_new,set_to_zero2)
                                                flag_batch_divided = 1
                                                reset_init2 += -set_to_zero2
                                        
                                        B1_new[slots_to_shift1_new[pos_shiftB1_2]:slots_to_shift1_new[pos_shiftB1_2+1]] = np.roll(B1_new[slots_to_shift1_new[pos_shiftB1_2]:slots_to_shift1_new[pos_shiftB1_2+1]],shift_value)
                                        for j in pos_shiftB2_2:
                                            B2_new[slots_to_shift2_new[j]:slots_to_shift2_new[j+1]] = np.roll(B2_new[slots_to_shift2_new[j]:slots_to_shift2_new[j+1]],shift_value)
                                            slots_to_shift2_new[j+1] += shift_value
                                            B2_more800ns_new[j+1] -= shift_value
                                            B2_more800ns_new[j] -= shift_value
                                            if j!=len(B2_more800ns_new)-2:
                                                B2_more800ns_new[j+2] += shift_value
                                            if j != 0:
                                                B2_more800ns_new[j+shift_value] += shift_value
                                            slots_to_shift2_new[j] += shift_value
                        
                                        slots_to_shift1_new[pos_shiftB1_2+1] += shift_value
                                        B1_more800ns_new[pos_shiftB1_2+1] -= shift_value
                                        B1_more800ns_new[pos_shiftB1_2] -= shift_value
                                        if pos_shiftB1_2!=len(B1_more800ns_new)-2:
                                            B1_more800ns_new[pos_shiftB1_2+2] += shift_value
                                        if pos_shiftB1_2 != 0:
                                            B1_more800ns_new[pos_shiftB1_2+shift_value] += shift_value
                                        slots_to_shift1_new[pos_shiftB1_2] += shift_value

                            
                                    
                                else:
                                    #if you cannot change the INDIV
                                    exit_flag_INDIV = 1
                            #if we haven't pick the INDIV in pos_shift and also if we picked it, we could move all the other INDIVs        
                            if exit_flag_INDIV == 0:
                                #shift of B1
                                B1_new[slots_to_shift1_new[pos_shiftB1]:slots_to_shift1_new[pos_shiftB1+1]] = np.roll(B1_new[slots_to_shift1_new[pos_shiftB1]:slots_to_shift1_new[pos_shiftB1+1]],shift_value)
                                for j in pos_shiftB2:
                                    B2_new[slots_to_shift2_new[j]:slots_to_shift2_new[j+1]] = np.roll(B2_new[slots_to_shift2_new[j]:slots_to_shift2_new[j+1]],shift_value)
                                    slots_to_shift2_new[j+1] += shift_value
                                    B2_more800ns_new[j+1] -= shift_value
                                    B2_more800ns_new[j] -= shift_value
                                    if j!=len(B2_more800ns_new)-2:
                                        B2_more800ns_new[j+2] += shift_value
                                    if j != 0:
                                        B2_more800ns_new[j+shift_value] += shift_value
                                    slots_to_shift2_new[j] += shift_value

                                #change of all the parameters
                                slots_to_shift1_new[pos_shiftB1+1] += shift_value
                                B1_more800ns_new[pos_shiftB1+1] -= shift_value
                                B1_more800ns_new[pos_shiftB1] -= shift_value
                                if pos_shiftB1!=len(B1_more800ns_new)-2:
                                    B1_more800ns_new[pos_shiftB1+2] += shift_value
                                #check in orrder to see if the there are bunches for tune measurement, in case affermative if they collide!
                                if pos_shiftB1 != 0:
                                    B1_more800ns_new[pos_shiftB1+shift_value] += shift_value
                                slots_to_shift1_new[pos_shiftB1] += shift_value
                                #print(shift_value)
                                
                                
                        #once done the shift save the information that you need in order to restart the MonteCarlo
                        if flag_batch_divided == 1:
                            slots_to_shift1_new = [int(ii) for ii in np.mod(slots_to_shift1_new + reset_init1*np.ones(len(slots_to_shift1_new)),3564)]
                            B1_new = np.roll(B1_new,reset_init1)
                            slots_to_shift2_new = [int(ii) for ii in np.mod(slots_to_shift2_new + reset_init2*np.ones(len(slots_to_shift2_new)),3564)]
                            B2_new = np.roll(B2_new,reset_init2)
                        
                        #B_more800ns_new_dict['beam1'] = B1_more800ns_new
                        #slots_to_shift_new_dict['beam1'] = slots_to_shift1_new
                        #B_more800ns_new_dict['beam2'] = B1_more800ns_new
                        #slots_to_shift_new_dict['beam2'] = slots_to_shift1_new
                        #beams_new_dict['beam1'] = B1_new
                        #beams_new_dict['beam2'] = B2_new
                        shift1 = copy.copy(slots_to_shift1_new)
                        shift2 = copy.copy(slots_to_shift2_new)
                        B___1 = copy.copy(B1_new)
                        B___2 = copy.copy(B2_new)
                            #np.roll(B___1,2670)
                        events_ATLAS_new = tbtb.events_in_IPN(B___1,B___2,'IP1')[1]
                        events_ALICE_new = tbtb.events_in_IPN(B___1,B___2,'IP2')[1]
                        events_LHCb_new = tbtb.events_in_IPN(B___1,B___2,'IP8')[1]
                        add_info = np.append(shift2,[events_ATLAS_new,events_LHCb_new,events_ALICE_new])
                        informations_new = np.append(shift1,[add_info[ii] for ii in range(len(add_info))])
                        list_informations[count-1] = [ii for ii in informations_new]
            
        # put all the informations, sectioned, that we want in the DataFrame 
        del list_informations[count:]
        df_fill_schemes2 = pd.DataFrame(index = np.arange(count))
        df_fill_schemes2['informations'] = list_informations
        c = np.where([df_fill_schemes2.iloc[0]['informations'][ii] != df_fill_schemes2.iloc[1]['informations'][ii] for ii in range(len(df_fill_schemes2.iloc[1]['informations']))])
        
        d = np.where([df_fill_schemes2.iloc[1]['informations'][ii] != df_fill_schemes2.iloc[2]['informations'][ii]for ii in range(len(df_fill_schemes2.iloc[1]['informations']))])
        
        df_fill_schemes = pd.DataFrame(np.unique(df_fill_schemes2), columns=df_fill_schemes2.columns)
        df_help = pd.DataFrame(df_fill_schemes2["informations"].values.tolist())
        df_fill_schemes = pd.DataFrame(index = np.arange(len(df_help.index)))
        vec_string_slot = df_help[df_help.columns[0:len(slots_to_shift1)]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1)
        help_vec = [list(ii.split(" ")) for ii in vec_string_slot]
        vec_string_slot2 = df_help[df_help.columns[len(slots_to_shift1):(len(slots_to_shift1)+len(slots_to_shift2))]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1)
        help_vec2 = [list(ii.split(" ")) for ii in vec_string_slot2]
        df_fill_schemes['slots_to_shift1'] = [[int(eval(ii)) for ii in i] for i in help_vec]
        df_fill_schemes['slots_to_shift2'] = [[int(eval(ii)) for ii in i] for i in help_vec2]
        #df_fill_schemes['distance'] = [LA.norm(ii-slots_to_shift_check[0]) for ii in df_fill_schemes['slots_to_shift']]
        df_fill_schemes['events_ATLAS/CMS'] = df_help.iloc[:,-3]
        df_fill_schemes['events_LHCb'] = df_help.iloc[:,-2]
        df_fill_schemes['events_ALICE'] = df_help.iloc[:,-1]

    return df_fill_schemes



def from_LPC_json_to_parquet(file_name, parquet_name = 'input', tune_shift_gap = 48):
    '''
    Having as input the name of the json file downloaded from LPC, this function returns a parquet file, with the name given as input, that contains:
        slots_to_shift: positions of the (first element -1) and (last element+1) of every SPS_batch for the deteached beam
        slots_init: initial position of the twelve bunches of the other beam 
        beam: give the number of the beam connected with slots_to_shift 
        events_LHCb: number of collisions in LHCb for the filling scheme given as input
        events_ALICE: number of collisions in ALICE for the filling scheme given as input
    Working Hypothesis:
    
    Args:
        file_name: name of the json file downloaded from LPC
        parquet_name: name of the parquet file that contains the information, set as 'input' considering to use that for the MonteCarlo simulation
    '''
    json_file = open(file_name)

    data_json = json.load(json_file)
    B1 = data_json['beam1']
    B2 = data_json['beam2']

    
    beam_detached = []

    json_dict = {'0':{}}

    # in order to understand which one of the two beams are detached from 0, in case both beams are detached we pick B1, because there are no prefence
    # considering that after we used this parquet for a MonteCalrlo completely casual
    
    if B1[0] == 0:
        beam_detached = np.append(beam_detached,1)
    if B2[0] == 0: 
        beam_detached = np.append(beam_detached,2)

    for j in [1,2]:
        B_emptyspaces = []
        B_fullspaces = []
        B_spaces = []
        slots_to_shift = []
        BEAM = data_json[f'beam{j}']
        zeros_beam = np.where(BEAM == np.zeros(3564))
        ones_beam =  np.where(BEAM == np.ones(3564))
            
        counter = 1
        # computaion all the length of the empty spaces, saving the length of the empty gap between PS batch
        for i in np.arange(len(zeros_beam[0])-1):
            flag_empty = 0
            flag_ns = 0
            if zeros_beam[0][i+1] == zeros_beam[0][i]+ 1:
                counter+=1
            else:
                B_emptyspaces = np.append(B_emptyspaces,[counter])
                counter = 1
        if i == len(zeros_beam[0])-2:
            B_emptyspaces = np.append(B_emptyspaces,[counter])
        
        counter = 1
        # computation  all the length of consecutive ones, saving the length of PS batch
        for i in np.arange(len(ones_beam[0])-1):
            if ones_beam[0][i+1] == ones_beam[0][i]+ 1:
                counter+=1
            else:

                B_fullspaces = np.append(B_fullspaces,[counter])
                counter = 1
        if i == len(ones_beam[0])-2:
            B_fullspaces = np.append(B_fullspaces,[counter])
        
        # merging the empty spaces and full spaces in a coherent ways
        counter = 0
        if not (j in beam_detached):
            print(j)
            for i in np.arange(len(B_fullspaces)):
                if B_emptyspaces[i] < 5:
                    counter += B_fullspaces[i]+B_emptyspaces[i]
                else:
                    B_spaces = np.append(B_spaces,[counter + B_fullspaces[i],B_emptyspaces[i]])
                    counter = 0
        else:
            for i in np.arange(len(B_fullspaces)):
                if B_emptyspaces[i+1] < 5:
                    counter += B_fullspaces[i]+B_emptyspaces[i+1]
                else:
                    B_spaces = np.append(B_spaces,[counter + B_fullspaces[i],B_emptyspaces[i+1]])
                    counter = 0
        
        # computation of positions of the (first element -1) and (last element+1) for the first SPS batch in case there is one in the tune_shift_gap
        if tune_shift_gap>0:
            slots_to_shift = [-1,B_spaces[0]+1]
        

        # computation of positions of the (first element -1) and (last element+1) of every SPS_batch from B_spaces
        for i in np.arange(len(B_spaces)):
            if tune_shift_gap == 0 or i > 1:
                if i%2 == 0 and B_spaces[i-1]>=31:
                    c = sum(B_spaces[:i])-1
                if i%2 == 0 and B_spaces[i+1]>=31:
                    d = sum(B_spaces[:i+1])+1
                    slots_to_shift = np.append(slots_to_shift,[c,d])
        
        slots_to_shift = [int(ii+int(j in beam_detached)*(B_emptyspaces[0])) for ii in slots_to_shift] 

        json_dict['0'][f'slots_to_shift{j}'] = slots_to_shift
    
    json_dict['0']['events_ATLAS/CMS'] = tbtb.events_in_IPN(B1,B2,'IP1')[1]
    json_dict['0']['events_LHCb'] = tbtb.events_in_IPN(B1,B2,'IP8')[1]
    json_dict['0']['events_ALICE'] = tbtb.events_in_IPN(B1,B2,'IP2')[1]
    json_dict['0']['beam_detached'] = beam_detached

    # creation of the parquet file, passing from a dictionary
    df = pd.DataFrame.from_dict(json_dict,orient = 'index')
    df.to_parquet(f'{parquet_name}.parquet')



def from_LPC_json_to_parquet_new(filename, parquet_name = 'input'):#, tune_shift_gap = 48):

    json_file = open(filename)

    data_json = json.load(json_file)
    B1 = data_json['beam1']
    B2 = data_json['beam2']

    no_beam_detached = 0
    if B1[0] == 0:
        beam = 1
        BEAM = B1
    elif B2[0] == 0: 
        beam = 2
        BEAM = B2
    else:
        beam = 1
        BEAM = B1
        no_beam_detached = 1

    counter = 0
    slots_to_shift = []
    for i in range(len(BEAM)):
        if BEAM[i] ==0:
            counter +=1
        elif BEAM[i] == 1:
            slot = i
            if len(slots_to_shift)==0:
                if no_beam_detached == 0:
                    slots_to_shift = np.append(slots_to_shift,i-1)
                else:
                    slots_to_shift = np.append(slots_to_shift,-1)
                counter = 0
            elif len(slots_to_shift)!=0 and counter >=31:
                slots_to_shift = np.append(slots_to_shift,[i-counter,i-1])
                counter = 0
            else: 
                counter = 0


    # computation of the position of the first twelve for the beam in case at least one is detached and if the one chosen is B1
    if beam == 1 and no_beam_detached == 0:
        flag = 0
        other_beam = [1,2][np.where([1,2]!=beam*np.ones(2))[0][0]]
        zeros = np.where(data_json[f'beam{other_beam}'] == np.zeros(3564))
        if zeros[0][0] == 0:
            for i in np.arange(len(zeros[0])-1):
                if zeros[0][i+1] == zeros[0][i]+ 1 and flag == 0:
                    counter+=1
                else:
                    slots_init = counter
                    flag = 1
        else: 
            slots_init = 0
    elif beam == 2 or no_beam_detached == 1:
        slots_init = 0

    slots_to_shift = np.append(slots_to_shift,[slot+1])
    json_dict = {'0':{
            'slots_to_shift':[int(ii) for ii in slots_to_shift],
            'events_LHCb':tbtb.events_in_IPN(B1,B2,'IP8')[1],
            'events_ALICE':tbtb.events_in_IPN(B1,B2,'IP2')[1],
            'slots_init': slots_init,
            'beam' :beam
            }}


    # creation of the parquet file, passing from a dictionary
    df = pd.DataFrame.from_dict(json_dict,orient = 'index')
    df.to_parquet(f'{parquet_name}.parquet')