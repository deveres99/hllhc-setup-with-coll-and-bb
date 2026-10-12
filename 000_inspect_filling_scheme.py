import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from filling_scheme_inspector import tool_box_to_bool as tbtb
from filling_scheme_inspector import bb_toolbox as btb


def main():
    numberOfLRToConsider = 33
    [B1,B2] = tbtb.filling_scheme_from_scheme_editor("./example_data/25ns_2760b_2748_2492_2574_288bpi_13inj_800ns_bs200ns.json")
    [df_B1,df_B2] = btb.bbschedule(B1,B2,numberOfLRToConsider)

    '''
    Get all families (unique combinations of BBLR in each detector) and the 
    corresponding bunch counts
    '''
    summary_B1 = result = df_B1.groupby(
        ['# of LR in ATLAS/CMS', '# of LR in ALICE', '# of LR in LHCB']
    ).size().reset_index(name='count')
    summary_B2 = result = df_B2.groupby(
            ['# of LR in ATLAS/CMS', '# of LR in ALICE', '# of LR in LHCB']
        ).size().reset_index(name='count')


    '''
    Find most representative family per beam
    '''
    tot_B1 = np.sum(summary_B1["count"].to_numpy())
    mrf_B1 = summary_B1.index.to_numpy()[np.argmax(summary_B1["count"].to_numpy())]
    num_LR_ATLAS_CMS_B1 = summary_B1.loc[mrf_B1, "# of LR in ATLAS/CMS"]
    num_LR_ALICE_B1 = summary_B1.loc[mrf_B1, "# of LR in ALICE"]
    num_LR_LHCB_B1 = summary_B1.loc[mrf_B1, "# of LR in LHCB"]
    mask_B1 = (df_B1["# of LR in ATLAS/CMS"] == num_LR_ATLAS_CMS_B1) & (df_B1["# of LR in ALICE"] == num_LR_ALICE_B1) & (df_B1["# of LR in LHCB"] == num_LR_LHCB_B1)
    mrb_B1 = np.asarray(df_B1[mask_B1].index)

    tot_B2 = np.sum(summary_B2["count"].to_numpy())
    mrf_B2 = summary_B2.index.to_numpy()[np.argmax(summary_B2["count"].to_numpy())]
    num_LR_ATLAS_CMS_B2 = summary_B2.loc[mrf_B2, "# of LR in ATLAS/CMS"]
    num_LR_ALICE_B2 = summary_B2.loc[mrf_B2, "# of LR in ALICE"]
    num_LR_LHCB_B2 = summary_B2.loc[mrf_B2, "# of LR in LHCB"]
    mask_B2 = (df_B2["# of LR in ATLAS/CMS"] == num_LR_ATLAS_CMS_B2) & (df_B2["# of LR in ALICE"] == num_LR_ALICE_B2) & (df_B2["# of LR in LHCB"] == num_LR_LHCB_B2)
    mrb_B2 = np.asarray(df_B2[mask_B2].index)

    print(f"Most representative bunch family in B1 with {summary_B1.loc[mrf_B1, "count"]}/{tot_B1} bunches:")
    print(f"\t# of LR in ATLAS/CMS: {num_LR_ATLAS_CMS_B1}")
    print(f"\t# of LR in ALICE: {num_LR_ALICE_B1}")
    print(f"\t# of LR in LHCB: {num_LR_LHCB_B1}")
    print("-- example bunches:", mrb_B1[:10], "\n")

    print(f"Most representative bunch family in B2 with {summary_B2.loc[mrf_B2, "count"]}/{tot_B2} bunches:")
    print(f"\t# of LR in ATLAS/CMS: {num_LR_ATLAS_CMS_B2}")
    print(f"\t# of LR in ALICE: {num_LR_ALICE_B2}")
    print(f"\t# of LR in LHCB: {num_LR_LHCB_B2}")
    print("-- example bunches:", mrb_B2[:10], "\n")


    '''
    Plot histogram of bunch families
    '''
    fs = 18

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), layout="tight")
    ax1a = axes[0, 0]
    ax1b = axes[1, 0]
    ax1b_r1 = ax1b.twinx()
    ax1b_r2 = ax1b.twinx()
    ax1b_r2.spines["right"].set_position(("axes", 1.2))
    ax2a = axes[0, 1]
    ax2b = axes[1, 1]
    ax2b_r1 = ax2b.twinx()
    ax2b_r2 = ax2b.twinx()
    ax2b_r2.spines["right"].set_position(("axes", 1.2))

    ax1a.axvline(
        summary_B1.index.to_numpy()[np.argmax(summary_B1["count"].to_numpy())], 
        color='gray', 
        linestyle='--'
    )
    ax1a.step(
        summary_B1.index.to_numpy(), summary_B1["count"].to_numpy(), 
        where="mid", 
        color="blue"
    )
    ax1a.set_xlabel("bunch family", fontsize=fs)
    ax1a.set_ylabel("bunch count", fontsize=fs)
    ax1a.tick_params(axis="both", labelsize=fs - 2)
    ax1a.set_title("B1", fontsize=fs+2)

    ax2a.axvline(
        summary_B2.index.to_numpy()[np.argmax(summary_B2["count"].to_numpy())], 
        color='gray', 
        linestyle='--'
    )
    ax2a.step(
        summary_B2.index.to_numpy(), summary_B2["count"].to_numpy(), 
        where="mid", 
        color="red"
    )
    ax2a.set_xlabel("bunch family", fontsize=fs)
    ax2a.set_ylabel("bunch count", fontsize=fs)
    ax2a.tick_params(axis="both", labelsize=fs - 2)
    ax2a.set_title("B2", fontsize=fs+2)

    ax1b.axvline(
        summary_B1.index.to_numpy()[np.argmax(summary_B1["count"].to_numpy())], 
        color='gray', 
        linestyle='--'
    )
    ax1b.plot(
        summary_B1.index.to_numpy(), summary_B1["# of LR in ATLAS/CMS"].to_numpy(), 
        color='black', 
        marker='.', 
        linestyle='None'
    )
    ax1b.set_xlabel("bunch family", fontsize=fs)
    ax1b.set_ylabel("# of LR in ATLAS/CMS", fontsize=fs)
    ax1b.tick_params(axis="both", labelsize=fs - 2)
    ax1b_r1.plot(
        summary_B1.index.to_numpy(), summary_B1["# of LR in ALICE"].to_numpy(), 
        color='magenta', 
        marker='x', 
        linestyle='None'
    )
    ax1b_r1.set_ylabel("# of LR in ALICE", fontsize=fs, color='magenta')
    ax1b_r1.tick_params(axis="y", colors='magenta', labelsize=fs - 2)
    ax1b_r1.spines["right"].set_edgecolor('magenta')
    ax1b_r2.plot(
        summary_B1.index.to_numpy(), summary_B1["# of LR in LHCB"].to_numpy(), 
        color='red', 
        marker='+', 
        linestyle='None'
    )
    ax1b_r2.set_ylabel("# of LR in LHCb", fontsize=fs, color='red')
    ax1b_r2.tick_params(axis="y", colors='red', labelsize=fs - 2)
    ax1b_r2.spines["right"].set_edgecolor('red')

    
    ax2b.axvline(
        summary_B2.index.to_numpy()[np.argmax(summary_B2["count"].to_numpy())], 
        color='gray', 
        linestyle='--'
    )
    ax2b.plot(
        summary_B1.index.to_numpy(), summary_B1["# of LR in ATLAS/CMS"].to_numpy(), 
        color='black', 
        marker='.', 
        linestyle='None'
    )
    ax2b.set_xlabel("bunch family", fontsize=fs)
    ax2b.set_ylabel("# of LR in ATLAS/CMS", fontsize=fs)
    ax2b.tick_params(axis="both", labelsize=fs - 2)
    ax2b_r1.plot(
        summary_B2.index.to_numpy(), summary_B2["# of LR in ALICE"].to_numpy(), 
        color='magenta', 
        marker='x', 
        linestyle='None'
    )
    ax2b_r1.set_ylabel("# of LR in ALICE", fontsize=fs, color='magenta')
    ax2b_r1.tick_params(axis="y", colors='magenta', labelsize=fs - 2)
    ax2b_r1.spines["right"].set_edgecolor('magenta')
    ax2b_r2.plot(
        summary_B2.index.to_numpy(), summary_B2["# of LR in LHCB"].to_numpy(), 
        color='red', 
        marker='+', 
        linestyle='None'
    )
    ax2b_r2.set_ylabel("# of LR in LHCb", fontsize=fs, color='red')
    ax2b_r2.tick_params(axis="y", colors='red', labelsize=fs - 2)
    ax2b_r2.spines["right"].set_edgecolor('red')

    plt.show()


if __name__ == "__main__":
    main()