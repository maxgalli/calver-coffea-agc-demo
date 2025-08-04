from pathlib import Path
import time
import sys
import pickle
import matplotlib.pyplot as plt
import argparse
import os

import awkward as ak
import dask
import dask_awkward as dak
import hist.dask
import coffea
import numpy as np
import uproot
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.processor import ProcessorABC, Runner, IterativeExecutor, DaskExecutor
from coffea.analysis_tools import PackedSelection
from coffea import dataset_tools
import correctionlib

import warnings

import utils
#from utils.systematics import rand_gauss
utils.plotting.set_style()

warnings.filterwarnings("ignore")
NanoAODSchema.warn_missing_crossrefs = False # silences warnings about branches we will not use here


def rand_gauss(item):
    seeds = (
        ak.flatten(ak.typetracer.length_one_if_typetracer(item)).to_numpy().view("i4")
    )
    randomstate = np.random.Generator(np.random.PCG64(seeds))

    def getfunction(layout, depth, **kwargs):
        if isinstance(layout, ak.contents.NumpyArray) or not isinstance(
            layout, (ak.contents.Content,)
        ):
            return ak.contents.NumpyArray(
                randomstate.normal(loc=1, scale=0.05, size=len(layout)).astype(np.float32)
            )
        return None

    out = ak.transform(
        getfunction,
        ak.typetracer.length_zero_if_typetracer(item),
        behavior=item.behavior,
    )
    if ak.backend(item) == "typetracer":
        out = ak.Array(
            out.layout.to_typetracer(forget_length=True), behavior=out.behavior
        )

    assert out is not None
    return out


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
def object_selection(elecs, muons, jets):
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

    return selections


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


class create_histograms(ProcessorABC):
    # create histograms with observables
    def process(self, events):
        hist_4j1b = (
            hist.Hist.new.Reg(11, 110, 550, name="HT", label=r"$H_T$ [GeV]")
            .StrCat([], name="process", label="Process", growth=True)
            .StrCat([], name="variation", label="Systematic variation", growth=True)
            .Weight()
        )
    
        hist_4j2b = (
            hist.Hist.new.Reg(11, 110, 550, name="m_reco_top", label=r"$m_{bjj}$ [GeV]")
            .StrCat([], name="process", label="Process", growth=True)
            .StrCat([], name="variation", label="Systematic variation", growth=True)
            .Weight()
        )
    
        hist_dict = {"4j1b": hist_4j1b, "4j2b": hist_4j2b}
    
        process = events.metadata["process"]  # "ttbar" etc.
        variation = events.metadata["variation"]  # "nominal" etc.
        #process_label = events.metadata["process_label"]  # nicer LaTeX labels
    
        # normalization for MC
        x_sec = events.metadata["xsec"]
        nevts_total = events.metadata["nevts"]
        lumi = 3378 # /pb
        if process != "data":
            xsec_weight = x_sec * lumi / nevts_total
        else:
            xsec_weight = 1
    
        events["pt_scale_up"] = 1.03
        #events["pt_res_up"] = rand_gauss(events.Jet.pt)
        events["pt_res_up"] = utils.systematics.jet_pt_resolution(events.Jet.pt)
    
        syst_variations = ["nominal"]
        jet_kinematic_systs = ["pt_scale_up", "pt_res_up"]
        event_systs = [f"btag_var_{i}" for i in range(4)]
        if process == "wjets":
            event_systs.append("scale_var")
        
        if variation == "nominal":
            syst_variations.extend(jet_kinematic_systs)
            syst_variations.extend(event_systs)
        
        for syst_var in syst_variations:
            elecs = events.Electron
            muons = events.Muon
            jets = events.Jet
    
            if syst_var in jet_kinematic_systs:
                jets["pt"] = jets.pt * events[syst_var]
        
            elecs, muons, jets = object_selection(elecs, muons, jets)
    
            # region selection
            selections = region_selection(elecs, muons, jets)
    
            for region in hist_dict:
                selection = selections.all(region)
                region_jets = jets[selection]
                region_weights = ak.ones_like(ak.num(region_jets, axis=1)) * xsec_weight
                if region == "4j1b":
                    observable = ak.sum(region_jets.pt, axis=-1)
                elif region == "4j2b":
                    observable = calculate_m_reco_top(region_jets)
                syst_var_name = f"{syst_var}"
                if syst_var in event_systs:
                    for i_dir, direction in enumerate(["up", "down"]):
                        if syst_var == "scale_var":
                            wgt_variation = cset["event_systematics"].evaluate("scale_var", direction, region_jets.pt[:, 0])
                        elif syst_var.startswith("btag_var"):
                            i_jet = int(syst_var.rsplit("_",1)[-1])
                            wgt_variation = cset["event_systematics"].evaluate("btag_var", direction, region_jets.pt[:,i_jet])
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
    
        return {events.metadata["dataset"]: hist_dict}

    def postprocess(self, accumulator):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test AGC")
    parser.add_argument("--files-per-sample", "-n", type=int, default=1, help="Number of files per sample")
    parser.add_argument("--chunksize", "-c", type=int, default=200000, help="Chunksize")
    parser.add_argument("--n-workers", "-w", type=int, default=1, help="Number of workers")
    parser.add_argument("--force", "-f", action="store_true", help="Force recompute")
    parser.add_argument("--local", "-l", action="store_true", help="Use local files")
    args = parser.parse_args()

    cluster = LocalCluster(n_workers=args.n_workers, threads_per_worker=1)
    #cluster = SLURMCluster(
    #    queue="short",
    #    #queue="standard",
    #    #walltime="5:00:00",
    #    cores=1,
    #    processes=1,
    #    memory="6G",
    #    #log_directory="slurm_logs",
    #    #local_directory="slurm_logs",
    #)
    #cluster.adapt(minimum=args.n_workers, maximum=args.n_workers)
    client = Client(cluster)
    #print("Waiting for workers")
    #client.wait_for_workers(args.n_workers)

    N_FILES_MAX_PER_SAMPLE = args.files_per_sample
    chunksize = args.chunksize
    print(f"Using chunksize {chunksize}")
    force = args.force
    print(f"Using {N_FILES_MAX_PER_SAMPLE} files per sample")
    print(f"Using chunksize {chunksize}")
    if force:
        print("Forcing recompute")

    if os.path.exists(f"all_histograms_fps{N_FILES_MAX_PER_SAMPLE}.pkl") and not force:
        print("Loading histograms from disk")
        with open(f"all_histograms_fps{N_FILES_MAX_PER_SAMPLE}_virtualarrays.pkl", "rb") as f:
            out = pickle.load(f)
    else:
        # compared to coffea 0.7: list of file paths becomes list of dicts (path: trename)
        fileset = utils.file_input.construct_fileset(N_FILES_MAX_PER_SAMPLE, local=args.local)
        print(fileset.keys())
        print(fileset["ttbar__nominal"])
        #import subprocess
        #for key, value in fileset.items():
        #    print(key)
        #    counter = 0
        #    for f in value["files"]:
        #        print(f)
        #        if counter >= 5:
        #            break
        #        new_f_name = f.replace("https", "root")
        #        last_two_parts = new_f_name.rsplit("/", 2)[-2:]
        #        last_two_parts = "/".join(last_two_parts)
        #        print(f"Last two parts: {last_two_parts}")
        #        result = subprocess.run(['xrdcp', new_f_name, f"/scratch/gallim/240730_AGC/{last_two_parts}"], capture_output=True, text=True)
        #        print("Exit code:", result.returncode)
        #        print("Standard Output:\n", result.stdout)
        #        counter += 1

        t0 = time.monotonic()
        # Define Runner
        run = Runner(
            DaskExecutor(client=client, compression=None),
            chunksize=chunksize,
            skipbadfiles=True,
            schema=NanoAODSchema,
            savemetrics=True
        )
        # pre-process
        samples = run.preprocess(fileset, treename="Events") # treename not needed with coffea master branch
        proc_time = time.monotonic() - t0
        print(f"\npreprocessing took {proc_time:.2f} seconds")

        # workaround for https://github.com/CoffeaTeam/coffea/issues/1050 (metadata gets dropped, already fixed)
        #for k, v in samples.items():
        #    v["metadata"] = fileset[k]["metadata"]

        t0 = time.monotonic()
        # execute
        tmp, report = run(samples, processor_instance=create_histograms())
        # sort the key order to be the same as the initial fileset
        out = {key: tmp[key] for key in fileset}
        sorted(report["columns"])
        exec_time = time.monotonic() - t0
        print(f"\nexecution took {exec_time:.2f} seconds")
        with open(f"all_histograms_fps{N_FILES_MAX_PER_SAMPLE}_virtualarrays.pkl", "wb") as f:
            pickle.dump(out, f)

        # dump information into a csv file
        print("Dumping information into a csv file")
        import csv
        log_file = "report_virtualarrays.csv"
        if args.local:
            log_file = "report_virtualarrays_local.csv"
        #log_file = "report_distributed.csv"
        file_exists = os.path.isfile(log_file)
        from datetime import datetime
        timestamp = datetime.now().isoformat()
        with open(log_file, mode='a') as f:
            fieldnames = ['timestamp', 'n_files', 'n_workers', 'execution_time', 'chunksize']
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(fieldnames)
            writer.writerow([timestamp, N_FILES_MAX_PER_SAMPLE, args.n_workers, exec_time, chunksize])
            #writer.writerow([timestamp, N_FILES_MAX_PER_SAMPLE, args.n_workers, exec_time_tot, chunksize])

    # histograms
    full_histogram_4j1b = sum([v["4j1b"] for v in out.values()])
    full_histogram_4j2b = sum([v["4j2b"] for v in out.values()])

    # dump for stats inference with also pseudodata
    #print("Saving histograms to ROOT file with pseudodata")
    #hist_dct = {"4j1b": full_histogram_4j1b, "4j2b": full_histogram_4j2b}
    #utils.file_output.save_histograms(hist_dct, f"all_histograms_fps{N_FILES_MAX_PER_SAMPLE}.root")
    #for region, histogram in [("bin4j1b", full_histogram_4j1b), ("bin4j2b", full_histogram_4j2b)]:
    #    utils.file_output.save_histograms(histogram, f"all_histograms_fps{N_FILES_MAX_PER_SAMPLE}_{region}.root")

    fig_dir = Path.cwd() / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Plotting histograms")
    artists = full_histogram_4j1b[120j::hist.rebin(2), :, "nominal"].stack("process")[::-1].plot(
        stack=True, histtype="fill", linewidth=1,edgecolor="grey"
    )
    ax = artists[0].stairs.axes
    fig = ax.get_figure()
    ax.legend(frameon=False)
    ax.set_title(">= 4 jets, 1 b-tag")
    fig.savefig(fig_dir / f"coffea_4j_1b_{N_FILES_MAX_PER_SAMPLE}_virtualarrays.png", dpi=300)
    plt.close(fig)

    artists = full_histogram_4j2b[:, :, "nominal"].stack("process")[::-1].plot(
        stack=True, histtype="fill", linewidth=1,edgecolor="grey"
    )
    ax = artists[0].stairs.axes
    fig = ax.get_figure()
    ax.legend(frameon=False)
    ax.set_title(">= 4 jets, >= 2 b-tags")
    fig.savefig(fig_dir / f"coffea_4j_2b_{N_FILES_MAX_PER_SAMPLE}_virtualarrays.png", dpi=300)
    plt.close(fig)

    # b-tagging variations
    #ttbar_label = '$t\\bar{t}$'
    ttbar_label = "ttbar"
    fig, ax = plt.subplots()
    full_histogram_4j1b[120j::hist.rebin(2), ttbar_label, "nominal"].plot(label="nominal", linewidth=2)
    full_histogram_4j1b[120j::hist.rebin(2), ttbar_label, "btag_var_0_up"].plot(label="NP 1", linewidth=2)
    full_histogram_4j1b[120j::hist.rebin(2), ttbar_label, "btag_var_1_up"].plot(label="NP 2", linewidth=2)
    full_histogram_4j1b[120j::hist.rebin(2), ttbar_label, "btag_var_2_up"].plot(label="NP 3", linewidth=2)
    full_histogram_4j1b[120j::hist.rebin(2), ttbar_label, "btag_var_3_up"].plot(label="NP 4", linewidth=2)
    ax.legend(frameon=False)
    ax.set_xlabel("$H_T$ [GeV]")
    ax.set_title("b-tagging variations")
    fig.savefig(fig_dir / f"coffea_btag_variations_{N_FILES_MAX_PER_SAMPLE}_virtualarrays.png", dpi=300)
    plt.close(fig)

    # jet enrgy scale/resolution variations
    fig, ax = plt.subplots()
    full_histogram_4j2b[:, ttbar_label, "nominal"].plot(label="nominal", linewidth=2)
    full_histogram_4j2b[:, ttbar_label, "pt_scale_up"].plot(label="scale up", linewidth=2)
    full_histogram_4j2b[:, ttbar_label, "pt_res_up"].plot(label="resolution up", linewidth=2)
    ax.legend(frameon=False)
    ax.set_xlabel("$m_{bjj}$ [GeV]")
    ax.set_title("jet energy scale/resolution variations")
    fig.savefig(fig_dir / f"coffea_jet_kin_variations_{N_FILES_MAX_PER_SAMPLE}_virtualarrays.png", dpi=300)
    plt.close(fig)