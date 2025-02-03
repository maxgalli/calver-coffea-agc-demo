from pathlib import Path

import awkward as ak
import dask
import dask_awkward as dak
import hist.dask
import coffea
import numpy as np
import uproot
from dask.distributed import Client

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import PackedSelection
from coffea import dataset_tools
import correctionlib

import warnings

import utils
utils.plotting.set_style()

warnings.filterwarnings("ignore")
NanoAODSchema.warn_missing_crossrefs = False # silences warnings about branches we will not use here


#client = Client("tls://localhost:8786")
from dask.distributed import LocalCluster


def calculate_trijet_mass(events):
    # pT > 30 GeV for leptons, > 25 GeV for jets
    selected_electrons = events.Electron[events.Electron.pt > 30 & (np.abs(events.Electron.eta) < 2.1)]
    selected_muons = events.Muon[events.Muon.pt > 30 & (np.abs(events.Muon.eta) < 2.1)]
    selected_jets = events.Jet[events.Jet.pt > 25 & (np.abs(events.Jet.eta) < 2.4)]

    # single lepton requirement
    event_filters = ((ak.count(selected_electrons.pt, axis=1) + ak.count(selected_muons.pt, axis=1)) == 1)
    # at least four jets
    event_filters = event_filters & (ak.count(selected_jets.pt, axis=1) >= 4)
    # at least two b-tagged jets ("tag" means score above threshold)
    B_TAG_THRESHOLD = 0.5
    event_filters = event_filters & (ak.sum(selected_jets.btagCSVV2 > B_TAG_THRESHOLD, axis=1) >= 2)

    # apply filters
    selected_jets = selected_jets[event_filters]

    trijet = ak.combinations(selected_jets, 3, fields=["j1", "j2", "j3"])  # trijet candidate
    trijet["p4"] = trijet.j1 + trijet.j2 + trijet.j3  # four-momentum of tri-jet system

    trijet["max_btag"] = np.maximum(trijet.j1.btagCSVV2, np.maximum(trijet.j2.btagCSVV2, trijet.j3.btagCSVV2))
    trijet = trijet[trijet.max_btag > B_TAG_THRESHOLD]  # at least one-btag in trijet candidates
    # pick trijet candidate with largest pT and calculate mass of system
    trijet_mass = trijet["p4"][ak.argmax(trijet.p4.pt, axis=1, keepdims=True)].mass
    return ak.flatten(trijet_mass)

B_TAG_THRESHOLD = 0.5

cset = correctionlib.CorrectionSet.from_file("corrections.json")

# perform object selection
def object_selection(events):
    elecs = events.Electron
    muons = events.Muon
    jets = events.Jet

    electron_reqs = (elecs.pt > 30) & (np.abs(elecs.eta) < 2.1) & (elecs.cutBased == 4) & (elecs.sip3d < 4)
    muon_reqs = ((muons.pt > 30) & (np.abs(muons.eta) < 2.1) & (muons.tightId) & (muons.sip3d < 4) &
                 (muons.pfRelIso04_all < 0.15))
    jet_reqs = (jets.pt > 30) & (np.abs(jets.eta) < 2.4) & (jets.isTightLeptonVeto)

    # Only keep objects that pass our requirements
    elecs = elecs[electron_reqs]
    muons = muons[muon_reqs]
    jets = jets[jet_reqs]

    return elecs, muons, jets


# event selection for 4j1b and 4j2b
def region_selection(elecs, muons, jets):
    ######### Store boolean masks with PackedSelection ##########
    selections = PackedSelection(dtype='uint64')
    # Basic selection criteria
    selections.add("exactly_1l", (ak.num(elecs) + ak.num(muons)) == 1)
    selections.add("atleast_4j", ak.num(jets) >= 4)
    selections.add("exactly_1b", ak.sum(jets.btagCSVV2 > B_TAG_THRESHOLD, axis=1) == 1)
    selections.add("atleast_2b", ak.sum(jets.btagCSVV2 > B_TAG_THRESHOLD, axis=1) >= 2)
    # Complex selection criteria
    selections.add("4j1b", selections.all("exactly_1l", "atleast_4j", "exactly_1b"))
    selections.add("4j2b", selections.all("exactly_1l", "atleast_4j", "atleast_2b"))

    return selections.all("4j1b"), selections.all("4j2b")


# observable calculation for 4j2b
def calculate_m_reco_top(jets):
    # reconstruct hadronic top as bjj system with largest pT
    trijet = ak.combinations(jets, 3, fields=["j1", "j2", "j3"])  # trijet candidates
    trijet["p4"] = trijet.j1 + trijet.j2 + trijet.j3  # four-momentum of tri-jet system
    trijet["max_btag"] = np.maximum(trijet.j1.btagCSVV2,
                                    np.maximum(trijet.j2.btagCSVV2, trijet.j3.btagCSVV2))
    trijet = trijet[trijet.max_btag > B_TAG_THRESHOLD]  # at least one-btag in candidates
    # pick trijet candidate with largest pT and calculate mass of system
    trijet_mass = trijet["p4"][ak.argmax(trijet.p4.pt, axis=1, keepdims=True)].mass
    observable = ak.flatten(trijet_mass)

    return observable


# create histograms with observables
def create_histograms(events):
    hist_4j1b = (
        hist.dask.Hist.new.Reg(25, 50, 550, name="HT", label=r"$H_T$ [GeV]")
        .StrCat([], name="process", label="Process", growth=True)
        .StrCat([], name="variation", label="Systematic variation", growth=True)
        .Weight()
    )

    hist_4j2b = (
        hist.dask.Hist.new.Reg(25, 50, 550, name="m_reco_top", label=r"$m_{bjj}$ [GeV]")
        .StrCat([], name="process", label="Process", growth=True)
        .StrCat([], name="variation", label="Systematic variation", growth=True)
        .Weight()
    )

    hist_dict = {"4j1b": hist_4j1b, "4j2b": hist_4j2b}

    process = events.metadata["process"]  # "ttbar" etc.
    variation = events.metadata["variation"]  # "nominal" etc.
    process_label = events.metadata["process_label"]  # nicer LaTeX labels

    # normalization for MC
    x_sec = events.metadata["xsec"]
    nevts_total = events.metadata["nevts"]
    lumi = 3378 # /pb
    if process != "data":
        xsec_weight = x_sec * lumi / nevts_total
    else:
        xsec_weight = 1

    elecs, muons, jets = object_selection(events)

    events["pt_scale_up"] = 1.03
    #events["pt_res_up"] = utils.systematics.jet_pt_resolution(jets.pt)
    events["pt_res_up"] = dak.map_partitions(utils.systematics.rand_gauss, jets.pt)

    syst_variations = ["nominal"]
    jet_kinematic_systs = ["pt_scale_up", "pt_res_up"]
    event_systs = []
    if process == "wjets":
        event_systs.append("scale_var")
    
    if variation == "nominal":
        syst_variations.extend(jet_kinematic_systs)
    
    for syst_var in syst_variations:
        if syst_var in jet_kinematic_systs:
            jets["pt"] = jets.pt * events[syst_var]

        # region selection
        selection_4j1b, selection_4j2b = region_selection(elecs, muons, jets)
        selections = {"4j1b": selection_4j1b, "4j2b": selection_4j2b}

        for region in selections:
            selection = selections[region]
            region_jets = jets[selection]
            #region_weights = ak.ones_like(region_jets.pt) * xsec_weight
            region_weights = dak.num(region_jets, axis=0) * xsec_weight
            if region == "4j1b":
                observable = ak.sum(region_jets.pt, axis=-1)
            elif region == "4j2b":
                observable = calculate_m_reco_top(region_jets)
            syst_var_name = syst_var
            if syst_var in event_systs:
                for i_dir, direction in enumerate(["up", "down"]):
                    if syst_var == "scale_var":
                        wgt_variation = cset["event_systematics"].evaluate("scale_var", direction, region_jets.pt[:, 0])
                    syst_var_name = f"{syst_var}_{direction}"
                    hist_dict[region].fill(
                        observable,
                        process=process,
                        variation=syst_var_name,
                        weight=region_weights * wgt_variation,
                    )
            else:
                if variation != "nominal":
                    syst_var_name = variation
                hist_dict[region].fill(
                    observable,
                    process=process,
                    variation=syst_var_name,
                    weight=region_weights,
                )
        ## 4j1b: HT
        #observable_4j1b = ak.sum(jets[selection_4j1b].pt, axis=-1)
        #hist_4j1b.fill(observable_4j1b, weight=xsec_weight, process=process_label, variation=variation)

        ## 4j2b: m_reco_top
        #observable_4j2b = calculate_m_reco_top(jets[selection_4j2b])
        #hist_4j2b.fill(observable_4j2b, weight=xsec_weight, process=process_label, variation=variation)

    return hist_dict

    
#### Main
if __name__ == "__main__":
    cluster = LocalCluster(n_workers=1, threads_per_worker=1)
    client = Client(cluster)

    print(f"awkward: {ak.__version__}")
    print(f"dask-awkward: {dak.__version__}")
    print(f"uproot: {uproot.__version__}")
    print(f"hist: {hist.__version__}")
    print(f"coffea: {coffea.__version__}")

    # fileset preparation
    N_FILES_MAX_PER_SAMPLE = 1
    # compared to coffea 0.7: list of file paths becomes list of dicts (path: trename)
    fileset = utils.file_input.construct_fileset(N_FILES_MAX_PER_SAMPLE)

    # fileset = {"ttbar__nominal": fileset["ttbar__nominal"]}  # to only process nominal ttbar
    # fileset

    # pre-process
    samples, _ = dataset_tools.preprocess(fileset, step_size=250_000)

    # workaround for https://github.com/CoffeaTeam/coffea/issues/1050 (metadata gets dropped, already fixed)
    for k, v in samples.items():
        v["metadata"] = fileset[k]["metadata"]

    ##################
    ##################
    #### DEBUG #######
    ##################
    ##################

    print("Entering debug area")

    events = NanoEventsFactory.from_root(
        {"cmsopendata2015_ttbar_19980_PU25nsData2015v1_76X_mcRun2_asymptotic_v12_ext3-v1_00000_0000.root": "Events"},
        metadata=fileset['ttbar__nominal']['metadata'],
        schemaclass=NanoAODSchema,
        #permit_dask=True
    )
    #events = events.events()

    out = create_histograms(events.events())