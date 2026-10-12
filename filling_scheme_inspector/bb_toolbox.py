## Copied from https://gitlab.cern.ch/mrufolo/fillingstudies by Matteo Rufolo ##

import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
import filling_scheme_inspector.tool_box_to_bool as tbtb


def events_in_slots(filling_scheme_to_be_rolled, filling_scheme,LR):
    '''
    Create a dataframe that represent the particle of the filling_scheme that 
    collide with the particles of the filling_scheme_to_be_rolled in the position LR,
    moreover it returns a vector that give the information of how many collision 
    Args:
        filling_scheme_to_be_rolled: a boolean np.array, that represent the fillig scheme
        to be rolled in order to compute the collision
        filling_scheme: a boolean np.array, that represent the other filling scheme
        LR: a int that give the position where to compute the collision of the two beams
    Return:
        pd.DataFrame that give the information about the partner beam collimator "BB in LR"
        and the position of the LR 
        np.array that give the information about the posi( to verify)
    '''
    n_bunches = len(filling_scheme_to_be_rolled)
    filling_scheme_rolled = np.concatenate([filling_scheme_to_be_rolled[-(LR%n_bunches):],\
        filling_scheme_to_be_rolled[:-(LR%n_bunches)]])
    index_events = (filling_scheme_rolled*filling_scheme)
    return  pd.DataFrame({'BB_LR' : np.where(filling_scheme*index_events)[0].astype(int), 'pos_LR' : LR},\
    index = (np.where(filling_scheme_rolled*index_events)[0]+(-LR))%n_bunches), \
        np.concatenate([index_events[-((-LR)%n_bunches):],index_events[:-((-LR)%n_bunches)]])
 

def events_in_slots_vec(filling_scheme_to_be_rolled, filling_scheme, IPN_pos, n_LR):
    '''
    This function returns three vectors with the information, respectively, about 
    the partner bunch of filling_scheme in the collision with filling_scheme_to_be_rolled, 
    about the position of the collision saved in the previous vector, and finally about the
    total number of collision for that particle of the filling_scheme_to_be_rolled
    Args:
        filling_scheme_to_be_rolled: a boolean np.array, that represent the fillig scheme
        to be rolled in order to compute the collision
        filling_scheme: a boolean np.array, that represent the other filling scheme
        IPN_pos: int that give the position of the detector, around which we want to compute the LR
        n_LR: is a int that represent how many LR the user want to consider around the detector
    Return:
        np.array(v): information about the partner of the collision (filling_scheme)
        np.array(v_pos): information about the position around the detector where happen the collision
        np.array(tot_LR): how many collision for that particle in filling_scheme_to_be_rolled
    '''
    n_bunches = len(filling_scheme_to_be_rolled)
    v1 = np.empty([n_bunches,2*n_LR+1])
    v1[:] = np.nan
    v1_pos = np.empty([n_bunches,2*n_LR+1])
    v1_pos[:] = np.nan
    tot_LR = np.zeros(n_bunches)
    count = 0
    for i in (np.arange(n_LR*2+1) - n_LR + IPN_pos):
        if i !=IPN_pos:
            s = np.empty([n_bunches,1])
            s[:] = np.nan
            s_pos = np.empty([n_bunches,1])
            s_pos[:] = np.nan
            filling_scheme_rolled = np.concatenate([filling_scheme_to_be_rolled[-(i%n_bunches):],\
            filling_scheme_to_be_rolled[:-(i%n_bunches)]])
            index_events = (filling_scheme_rolled*filling_scheme)   
            v1[(np.where(index_events)[0]+(-i))%n_bunches,count] = np.concatenate([s[(np.where(index_events)[0]+(-n_LR))%n_bunches],\
                np.array(list(np.where(index_events))).T],axis = 1)[:,1]
            
            v1_pos[(np.where(index_events)[0]+(-i))%n_bunches,count] = np.concatenate([s_pos[(np.where(index_events)[0]+(-n_LR))%n_bunches],\
                i*np.ones([len(np.where(index_events)[0]),1])],axis = 1)[:,1]
            count +=1
            tot_LR +=np.concatenate([index_events[-((-i)%n_bunches):],index_events[:-((-i)%n_bunches)]])
    t = ~np.isnan(v1)
    v = [v1[ii][t[ii]].astype(int) for ii in np.arange(n_bunches)]

    t_pos = ~np.isnan(v1)
    v_pos = [v1_pos[ii][t_pos[ii]].astype(int) for ii in np.arange(n_bunches)]
    #v1 = [list(v1[ii]) for ii in np.arange(3564)]
    #print(t[26])
    #print(v[26].shape)
    #print([v[ii][t[ii]] for ii in np.arange(n_bunches)])
    return v ,v_pos ,tot_LR.astype(int)



def bbschedule(bool_slotsB1,bool_slotsB2, numberOfLRToConsider, Dict1 = {'ATLAS/CMS':0,'LHCB':2670, 'ALICE': 891}, n_slots = 3564, Dict_nLRs = {'Nan' : np.nan}):
    ''' 
    This function return two pd.DataFrame, associated to the two beams, with the information 
    about the purtner bunch for HO and LR, about the position of collision with respect 
    to the detector and also how many LR there are around that detector. The two filling schemes
    are loaded directly from the two boolean vectors as input.
    Args:
        bool_slotsB1: a boolean np.array that give the information about the position of the
        filling scheme of B1 
        bool_slotsB2: a boolean np.array that give the information about the position of the 
        filling scheme of B2
        numberofLRToConsider : a number of LR to consider in every detector of the accellerator
        Dic1 : a dictionary that give the position of the detectors, seen from the point of view of B1, 
        and also how many LR are considered in each detector, set with the ones of LHC
        n_slots : number of length of the accellerator, in this case is set as the number of harmonic RF of LHC divided by 10
        Dict_nLRs : a dictionary that give the number of LR to consider in every detector of the dictionary Dict1, 
        they must have the same length, is set as np.nan
    Returns:
        pd.DataFrame (df_B1): that give all the information described above about LR and HO, seen from
        a poinf of view of B1
        pd.DataFrame (df_B2): that give all the information described above about LR and HO, seen from
        a point of view of B2

    '''

    assert np.isnan([numberOfLRToConsider,Dict_nLRs[random.choice(list(Dict_nLRs.keys()))]]).any(), "one between numberOfLRToConsider and Dict_nLRs should be NaN"
    assert ~np.isnan([numberOfLRToConsider,Dict_nLRs[random.choice(list(Dict_nLRs.keys()))]]).all(), "one between numberOfLRToConsider and Dict_nLRs should be NaN"
    keys = list(Dict1.keys())#
    if np.isnan(Dict_nLRs[random.choice(list(Dict_nLRs.keys()))]):
        Dic1 = {}
        Dic2 = {}
        for i in range(len(keys)):
            Dic1[f'{keys[i]}'] = [Dict1[f'{keys[i]}'],numberOfLRToConsider]
            Dic2[f'{keys[i]}'] = [np.mod(n_slots-Dict1[f'{keys[i]}'],n_slots),numberOfLRToConsider]
    else:
        keys_LR = list(Dict_nLRs.keys())
        assert len(keys) == len(keys_LR), "Dict1 and Dict_nLRs must have the same lengths of keys"
        Dic1 = {}
        Dic2 = {}
        for i in range(len(keys)):
            Dic1[f'{keys[i]}'] = [Dict1[f'{keys[i]}'],Dict_nLRs[f'{keys_LR[i]}']]
            Dic2[f'{keys[i]}'] = [np.mod(n_slots-Dict1[f'{keys[i]}'],n_slots),Dict_nLRs[f'{keys_LR[i]}']]
    
    
    
    ones_B1 = np.where(bool_slotsB1)
    ones_B2 = np.where(bool_slotsB2)


    df_B1 = pd.DataFrame(index = ones_B1[0])

    df_B2 = pd.DataFrame(index = ones_B2[0])
    
    for j in np.arange(len(Dic1.keys())):
        
        IPN_B1 = list(Dic1.keys())[j]
        [v,v_pos,sum_v]\
         = events_in_slots_vec(bool_slotsB1,bool_slotsB2,Dic1[IPN_B1][0],Dic1[IPN_B1][1])
        df_B1[f'# of LR in {IPN_B1}'] = sum_v[ones_B1[0]]
        df_B1[f'HO partner in {IPN_B1}'] = events_in_slots(bool_slotsB1,bool_slotsB2,Dic1[IPN_B1][0])[0].iloc[:,0]
        
        df_B1[f'BB partners in {IPN_B1}'] = [list(v[ii]) for ii in ones_B1[0]]
        df_B1[f'Positions in {IPN_B1}'] = [list(v_pos[ii]-Dic1[IPN_B1][0]) for ii in ones_B1[0]]
    
    for j in np.arange(len(Dic1.keys())):
        
        IPN_B2 = list(Dic2.keys())[j]
        [v,v_pos,sum_v]\
         = events_in_slots_vec(bool_slotsB2,bool_slotsB1,Dic2[IPN_B2][0],Dic2[IPN_B2][1])
        df_B2[f'# of LR in {IPN_B2}'] = sum_v[ones_B2[0]]
        df_B2[f'HO partner in {IPN_B2}'] = events_in_slots(bool_slotsB2,bool_slotsB1,Dic2[IPN_B2][0])[0].iloc[:,0]
        df_B2[f'BB partners in {IPN_B2}'] = [list(v[ii]) for ii in ones_B2[0]]
        df_B2[f'Positions in {IPN_B2}'] = [list(v_pos[ii]-Dic1[IPN_B2][0]) for ii in ones_B2[0]]
    return df_B1,df_B2


def plots_filling_pattern(B1,B2):
    ''' 
    This function return two plots, one that represent the disposition of the two beams and the other the collisions in the different IPs and 
    the number of LR around ATLAS/CMS
    Args:
        B1: a boolean np.array that give the information about the position of the
        filling scheme of B1 
        B2: a boolean np.array that give the information about the position of the 
        filling scheme of B2
    '''
    
    fig, axes = plt.subplots(figsize= (15,3.4),ncols=1, nrows=2,gridspec_kw={'height_ratios': [1, 1]})
    for beam,ax,color in zip([B1,B2],axes,['C0','C3']):

        plt.sca(ax)
        plt.stem(beam,markerfmt='none',basefmt='none',linefmt=color)

        plt.xlim([0,len(B1)])
        plt.ylim([0,1])
    fig.suptitle('Filling Pattern')
    plt.xlabel('Bunch number')
    plt.tight_layout()

    fig, axes = plt.subplots(figsize= (15,5),ncols=1, nrows=3,gridspec_kw={'height_ratios': [1, 1,1]})
    for ip,ax,color in zip(['IP1','IP2','IP8'],axes,['C2','C1','C6']):

        plt.sca(ax)
        plt.stem(tbtb.events_in_IPN(B1,B2,ip)[0]*2,markerfmt='none',basefmt='none',linefmt=color)

        plt.xlim([0,len(B1)])
        plt.ylim([0,1])
        ax.set_title(f'Collisions in {ip}')
    plt.xlabel('Bunch number')
    plt.tight_layout()

    plt.tight_layout()

    [bb_scheduleB1, bb_scheduleB2] = bbschedule(B1,B2)
    fig, axes = plt.subplots(figsize= (15,5),ncols=1, nrows=2,gridspec_kw={'height_ratios': [1, 1]})
    for bb_df,ax,color in zip([bb_scheduleB1, bb_scheduleB2],axes,['C0','C3']):

        plt.sca(ax)
        plt.plot(bb_df['# LR in IP1'].values,'o',color=color)

        #plt.xticks(np.arange(len(bb_df))[::4],[f"T{_bunch['Train']}, B{_bunch['Tag']}" for _idx,_bunch in bb_df.iterrows()][::4],rotation=90)
    fig.suptitle('Number of LRs')
    plt.xlabel('Bunch number')
    plt.tight_layout()

    #\bb_scheduleB1['# LR in IP1']
