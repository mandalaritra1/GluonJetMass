#### This file contains the processors for trijet hist selections. Plotting and resulting studies are in separate files.
#### LMH

#### import outer dependencies
import argparse
import awkward as ak
import numpy as np
import coffea
import os
import re
import pandas as pd
import hist
print(hist.__version__)
print(coffea.__version__)
from coffea import util, processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import Weights, PackedSelection
from collections import defaultdict
#### import our python packages
from .corrections import *
from .utils import *
from copy import deepcopy
parser = argparse.ArgumentParser()

parser.add_argument("year")
parser.add_argument("data")

### move to neatly logging instead of bare print statements
# class Log:
#     def __init__(self, mode="info"):
#         self.mode = mode
#     def info(self, msg):
#         if self.mode in ["info", "debug"]:
#             print("[INFO]", msg)
#     def debug(self, msg):
#         if self.mode == "debug":
#             print("[DEBUG]", msg)


#### Sal's code --> want to edit to do more than one jet at a time
def get_gen_sd_mass_jet( jet, subjets):
    combs = ak.cartesian( (jet, subjets), axis=1 )
    print("# of genjet and subjet combinations: ", len(combs))
    dr_jet_subjets = combs['0'].delta_r(combs['1'])
    combs = combs[dr_jet_subjets < 0.8]
    total = combs['1'].sum(axis=1)
    return total 

def get_dphi( coll0, coll1 ):
    #############
    #### Find dphi between 3rd jet and , returning none when the event does not have at least two jets
    #############
    combs = ak.cartesian( (coll0, coll1), axis=1 )
    dphi = np.abs(combs['0'].delta_phi(combs['1']))
    return ak.firsts((combs['1'])).ak.firsts(dphi)

def getJetFlavors(jet):
    genjet = jet.matched_gen
    jetflavs = {}
    jetflavs["Gluon"] = jet[np.abs(genjet.partonFlavour) == 21]
    jetflavs["UDS"]    = jet[np.abs(genjet.partonFlavour) < 4]
    jetflavs["Charm"]      = jet[np.abs(genjet.partonFlavour) == 4]
    jetflavs["Bottom"]      = jet[np.abs(genjet.partonFlavour) == 5]
    jetflavs["Other"]  = jet[(np.abs(genjet.partonFlavour) > 5) & (np.abs(genjet.partonFlavour) != 21)]
    return jetflavs
#bTag_options = ['bbloose', 'bloose', 'bbmed', 'bmed']
def applyBTag(events, btag):
    # print('btag input: ', btag, '\n')
    if (btag == 'bbloose'):
        sel = (events.FatJet[:,0].btagDeepB >= 0.2027) & (events.FatJet[:,1].btagDeepB >= 0.2027)
        events = events[sel]
        print('Loose WP CSV V2 B tag applied to leading two jets')
    elif (btag == 'bloose'):
        sel = (events.FatJet[:,0].btagDeepB >= 0.2027)
        events = events[sel]
        print('Loose WP CSV V2 B tag applied to leading jet only')
    elif (btag == 'bbmed'):
        sel = (events.FatJet[:,0].btagDeepB >= 0.6001) & (events.FatJet[:,1].btagDeepB >= 0.6001)
        events = events[sel]
        print('Medium WP CSV V2 B tag applied to first two jets')
    elif (btag == 'bmed'):
        sel = (events.FatJet[:,0].btagDeepB >= 0.6001)
        events = events[sel]
        print('Medium WP CSV V2 B tag applied to leading jet only')
    else:
        sel = np.ones(len(events), dtype = bool)
        print('no btag applied')
    return events, sel

##### TO DO #####
# do Rivet routine

#bcut options: b_loose (apply loose bTag threshold to only hardest jet), bb_loose (apply loose bTag to leading two jets),
#              b_med(apply medium bTag to only the hardest jet), bb_med (apply medium bTag to leading two jets)

class TrijetProcessor(processor.ProcessorABC):
    
    def __init__(self, ycut = 2.5, btag = 'None', data = False, jet_systematics = ['nominal'], jk=False, jk_range = None, do_minimal=True):
        
        self.ycut = ycut
        self.btag = btag
        self.do_gen = not data
        self.jk = jk
        self.jk_range = jk_range
        self.do_minimal = True
        if self.jk:
            # protect against doing unc and extra plots for jk --> only need nominal and memory intensive
            jet_systematics = ["nominal"]
            self.do_minimal = True
        self.jet_systematics = jet_systematics
        print("Data: ", data, " gen ", self.do_gen)
        
        #### Define axes for hists
        
        jet_cat = hist.axis.StrCategory([], growth=True, name="jetNumb", label="Jet")
        parton_cat = hist.axis.StrCategory([],growth=True,name="partonFlav", label="Parton Flavour")
        syst_cat = hist.axis.StrCategory([], growth=True, name='systematic', label="Systematic")
        dataset_axis = hist.axis.StrCategory([], growth=True, name="dataset", label="Primary dataset")
        fine_mass_bin = hist.axis.Regular(500, 0.0, 1000.0, name="mass", label=r"mass [GeV]")
        fine_pt_bin = hist.axis.Regular(400, 0.0, 8000.0, name="pt", label=r"$p_T$ [GeV]")
        mgen_bin_edges = np.array([0,5,10,20,40,60,80,100,120,140,160,180,200,300, 400, 500, 700, 900, 1100, 1300, 13000])
        mreco_bin_edges = np.sort(np.append(mgen_bin_edges,[(mgen_bin_edges[i]+mgen_bin_edges[i+1])/2 for i in 
                                                            range(len(mgen_bin_edges)-1)]))
        mass_gen_bin =  hist.axis.Variable(mgen_bin_edges, name="mgen", label=r"m_{GEN} (GeV)")                         
        mass_bin = hist.axis.Variable(mreco_bin_edges, name="mreco", label=r"m_{RECO} (GeV)")
        ptgen_edges = np.array([0,200,290,400,480,570,680,760,820,13000]) 
        pt_bin = hist.axis.Variable(ptgen_edges, name="ptreco", label=r"p_{T,RECO} (GeV)")  
        pt_gen_bin = hist.axis.Variable(ptgen_edges, name="ptgen", label=r"p_{T,GEN} (GeV)")
        #rho_gen_edges = np.array([-10, -8, -7, -6, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0])
        rho_gen_edges = np.array([-10, -8, -7, -6, -5, -4.4, -4, -3.6, -3.2, -2.8, -2.4, -2, -1.6, -1.2, -0.8, -0.4, 0])
        rho_edges = np.sort(np.append(rho_gen_edges,[(rho_gen_edges[i]+rho_gen_edges[i+1])/2 for i in range(len(rho_gen_edges)-1)]))
        rho_gen_bin = hist.axis.Variable(rho_gen_edges, name="mpt_gen", label=r"$-\log_10(\rho^2)_{GEN}$")
        rho_bin = hist.axis.Variable(rho_edges, name="mpt_reco", label=r"$-\log_10(\rho^2)$")
        y_bin = hist.axis.Regular(25, -4.0, 4.0, name="rapidity", label=r"$y$")
        eta_bin = hist.axis.Regular(25, -4., 4., name="eta", label=r"$\eta$")
        frac_axis = hist.axis.Regular(400, 0., 2.5, name="frac", label="Fraction")
        n_axis = hist.axis.Regular(5, 0, 5, name="n", label=r"Number")
        dr_axis = hist.axis.Regular(150, 0, 6.0, name="dr", label=r"$\Delta R$")
        dphi_axis = hist.axis.Regular(150, -np.pi, np.pi, name="dphi", label=r"$\Delta \phi$")
        phi_axis = hist.axis.Regular(25, -np.pi, np.pi, name="phi", label=r"$\phi$")
        weight_bin = hist.axis.Regular(100, 0, 5, name="corrWeight", label="Weight")
        jk_axis = hist.axis.IntCategory([], growth = True, name = 'jk', label = "Jackknife section" )
        
        cutflow = {}
        
        #### register histograms
        
        self._histos = {
                #### Plots to be unfolded/data only
                'ptjet_mjet_u_reco':              hist.Hist(dataset_axis,syst_cat, jk_axis, pt_bin, mass_bin, storage="weight", name="Events"),
                'ptjet_mjet_g_reco':              hist.Hist(dataset_axis,syst_cat, jk_axis, pt_bin, mass_bin, storage="weight", name="Events"),
                'ptjet_rhojet_u_reco':                  hist.Hist(dataset_axis, syst_cat, jk_axis, pt_bin, rho_bin, storage="weight", name="Events"),
                'ptjet_rhojet_g_reco':                  hist.Hist(dataset_axis, syst_cat, jk_axis, pt_bin, rho_bin, storage="weight", name="Events"),
        
                #### Plots for comparison
                'ptjet_mjet_u_gen':                hist.Hist(dataset_axis,syst_cat, jk_axis, pt_gen_bin, mass_gen_bin, storage="weight", label="Counts"),       
                'ptjet_mjet_g_gen':                hist.Hist(dataset_axis,syst_cat, jk_axis, pt_gen_bin, mass_gen_bin, storage="weight", label="Counts"),
                'ptjet_rhojet_u_gen':                  hist.Hist(dataset_axis, syst_cat, jk_axis, pt_gen_bin, rho_gen_bin, storage="weight", name="Events"),
                'ptjet_rhojet_g_gen':                  hist.Hist(dataset_axis, syst_cat, jk_axis, pt_gen_bin, rho_gen_bin, storage="weight", name="Events"),
                #### Plots for the analysis in the proper binning
                'response_matrix_rho_u':              hist.Hist(dataset_axis, syst_cat, jk_axis, pt_bin, pt_gen_bin, rho_bin,  rho_gen_bin, storage="weight", label="Counts"),
                'response_matrix_rho_g':              hist.Hist(dataset_axis, syst_cat, jk_axis, pt_bin, pt_gen_bin, rho_bin,  rho_gen_bin, storage="weight", label="Counts"),
                'response_matrix_u':           hist.Hist(dataset_axis,syst_cat, jk_axis, pt_bin, mass_bin, pt_gen_bin, mass_gen_bin, storage="weight",                                                         label="Counts"),
                'response_matrix_g':           hist.Hist(dataset_axis,syst_cat, jk_axis, pt_bin, mass_bin, pt_gen_bin, mass_gen_bin, storage="weight",                                                         label="Counts"),
                     
                #### misc.
                'cutflow':            cutflow,
                'jkflow':            processor.defaultdict_accumulator(int),
        }

        if not self.do_minimal:
            self._histos.update({ 
            ##### old fakes and misses
            'misses_u':                    hist.Hist(dataset_axis, syst_cat, jk_axis, pt_gen_bin, mass_gen_bin, storage="weight", name="Events"),
            'misses_g':                  hist.Hist(dataset_axis, syst_cat, jk_axis, pt_gen_bin, mass_gen_bin, storage="weight", name="Events"),
            'fakes_u':                     hist.Hist(dataset_axis, syst_cat, jk_axis, pt_bin, mass_bin, storage="weight", name="Events"),
            'fakes_g':                   hist.Hist(dataset_axis, syst_cat, jk_axis, pt_bin, mass_bin, storage="weight", name="Events"),
                
            #### gluon content/ btag study histos
            'alljet_ptreco_mreco':          hist.Hist(dataset_axis, jet_cat, parton_cat, mass_bin, pt_bin, storage="weight", name="Events"),
            'btag_eta':                     hist.Hist(dataset_axis, jet_cat, parton_cat, frac_axis, eta_bin, storage="weight", name="Events"),
            #### MET content histos
            'MET_over_sumET_pt_reco':       hist.Hist(dataset_axis, syst_cat, frac_axis, pt_bin, storage="weight", label="Events"),
            'MET_pt_reco':                  hist.Hist(dataset_axis, syst_cat, fine_pt_bin, pt_bin, storage="weight", label="Events"),
            #### Plots of things during the selection process / for debugging
            'HT_nocuts':                    hist.Hist(dataset_axis,syst_cat, fine_pt_bin, storage="weight", label="Events"),
            'HT_wXS':                       hist.Hist(dataset_axis,syst_cat,fine_pt_bin, storage="weight", label="Events"),
            'HT_aftercuts':                 hist.Hist(dataset_axis, syst_cat, fine_pt_bin, storage="weight", label="Events"),
            # 'njet_gen':                     hist.Hist(dataset_axis, syst_cat, n_axis, storage="weight", label="Events"),
            # 'njet_reco':                    hist.Hist(dataset_axis, syst_cat, n_axis, storage="weight", label="Events"),
            'dphimin_gen':                  hist.Hist(dataset_axis, syst_cat, dphi_axis, storage="weight", label="Events"),
            'dphimin_reco':                 hist.Hist(dataset_axis, syst_cat, dphi_axis, storage="weight", label="Events"),
            'asymm_reco':                   hist.Hist(dataset_axis, syst_cat, pt_bin, frac_axis, storage="weight", label="Events"),
            'asymm_gen':                    hist.Hist(dataset_axis, syst_cat, pt_gen_bin, frac_axis, storage="weight", label="Events"),
            'mass_orig':                    hist.Hist(dataset_axis, jk_axis, pt_bin, mass_bin, storage="weight", label="Events"),
            'sdmass_orig':                  hist.Hist(dataset_axis, jk_axis, pt_bin, mass_bin, storage="weight", label="Events"),
            'sdmass_ak8corr':               hist.Hist(dataset_axis, jk_axis, pt_bin, mass_bin, storage="weight", label="Events"),
            'sdmass_ak4corr':               hist.Hist(dataset_axis, jk_axis, pt_bin, mass_bin, storage="weight", label="Events"),
            # 'jet_rap_reco':                 hist.Hist(dataset_axis, syst_cat, y_bin, storage="weight", name="Events"),
            # 'jet_rap_gen':                  hist.Hist(dataset_axis, syst_cat, y_bin, storage="weight",name="Events"),
            # 'jet_phi_gen':                  hist.Hist(dataset_axis, syst_cat, phi_axis, storage="weight", label="Events"),
            # 'jet_phi_reco':                 hist.Hist(dataset_axis, syst_cat, phi_axis, storage="weight", label="Events"),
            'jet_eta_phi_precuts':          hist.Hist(dataset_axis, syst_cat, phi_axis, eta_bin, storage="weight", label="Counts"),
            'jet_eta_phi_preveto':          hist.Hist(dataset_axis, syst_cat, phi_axis, eta_bin, storage="weight", label="Counts"),
            'jet_pt_eta_phi':               hist.Hist(dataset_axis, syst_cat, pt_bin, phi_axis, eta_bin, storage="weight", label="Counts"),
            #### for investigation of removing fakes
            'fakes_eta_phi':                hist.Hist(dataset_axis, syst_cat, eta_bin, phi_axis, storage="weight", name="Events"),
            'fakes_asymm_dphi':             hist.Hist(dataset_axis, syst_cat, frac_axis, dphi_axis, storage="weight", name="Events"),
            'ptreco_mreco_fine_u':           hist.Hist(dataset_axis,syst_cat, jk_axis, fine_pt_bin, fine_mass_bin, storage="weight", label="Counts"),
            'ptreco_mreco_fine_g':           hist.Hist(dataset_axis,syst_cat, jk_axis, fine_pt_bin, fine_mass_bin, storage="weight",  label="Counts"), 
            #### Plots to get JMR and JMS in MC
            'm_u_jet_reco_over_gen':          hist.Hist(dataset_axis, pt_gen_bin, mass_gen_bin, frac_axis, storage="weight", label="Events"),
            'm_g_jet_reco_over_gen':          hist.Hist(dataset_axis, pt_gen_bin, mass_gen_bin, frac_axis, storage="weight", label="Events"),
        })
    
    @property
    def accumulator(self):
        return self._histos
        
    def process(self, events):
        out = self._histos
        dataset = events.metadata['dataset']
        filename = events.metadata['filename']
        if "madgraph" in dataset and "pythia" in dataset:
            mctype="pythiaMG"
        elif "herwig" in dataset:
            mctype='herwig'
        elif "_pythia" in dataset:
            mctype="pythia"
        else:
            mctype="data"  
        #####################################
        #### Find the IOV from the dataset name
        #####################################
        IOV = ('2016APV' if ( any(re.findall(r'APV',  dataset)) or any(re.findall(r'HIPM', dataset)))
               else '2018'    if ( any(re.findall(r'UL18', dataset)) or any(re.findall(r'UL2018',    dataset)))
               else '2017'    if ( any(re.findall(r'UL17', dataset)) or any(re.findall(r'UL2017',    dataset)))
               else '2016')
        #####################################
        #### Make loop for running 1/10 of dataset for jackknife
        #####################################
        if 'QCD' in dataset:
            datastr = 'QCD_'+mctype+IOV
        elif "WJets" in dataset:
            datastr = "WJets_"+mctype+IOV
        elif "ZJets" in dataset:
            datastr = 'ZJets_'+mctype+IOV
        elif "TTJets" in dataset:
            datastr = 'TTJets_'+mctype+IOV
        else: datastr = mctype+IOV
        datastr = mctype+IOV
        print("Filename: ", filename)
        print("Dataset: ", dataset)
        ####################################
        #### Inititalize cutflow table
        ###################################            
        
        out['cutflow'][datastr] = defaultdict(int)
        out['cutflow'][datastr]['nEvents initial'] += (len(events.FatJet))
        out['cutflow']['trigger_init'] = defaultdict(int)
        out['cutflow']['trigger_final'] = defaultdict(int)
        
        if (self.do_gen):
            firstidx = filename.find( "store/mc/" )
            fname2 = filename[firstidx:]
            fname_toks = fname2.split("/")
            # year = fname_toks[ fname_toks.index("mc") + 1]
            ht_bin = fname_toks[ fname_toks.index("mc") + 2]
            if "LHEWeight" in events.fields:
                weights = events["LHEWeight"].originalXWGTUP
            else:
                weights = events.genWeight
            out['cutflow'][datastr]['sumw for '+ht_bin] += np.sum(weights)
            ## Flag used for number of events

        
        index_list = np.arange(len(events))
        ###### Choose number of slices to break data into for jackknife method
        if self.jk:
            print("Self.jk ", self.jk)
            range_max = 10
        else: range_max=1
            
        if self.jk_range == None:
            jk_inds = range(0,range_max)
        else:
            jk_inds = range(self.jk_range[0], self.jk_range[1])
        for jk_index in jk_inds:
            print("Event indices ", index_list)
            if self.jk:
                print("Now doing jackknife {}".format(jk_index))
                print("Len of events before jk selection ", len(events))
            else:
                jk_index=-1
            print(index_list%range_max == jk_index)
            jk_sel = ak.where(index_list%range_max == jk_index, False, True)
            ######## Select portion for jackknife and ensure that all jets have a softdrop mass so sd mass correction does not fail
            events_jk = events[jk_sel]
            del jk_sel
            #### only consider pfmuons w/ similar selection to aritra for later jet isolation
            events_jk = ak.with_field(events_jk, 
                                      events_jk.Muon[(events_jk.Muon.mediumId > 0)
                                      &(np.abs(events_jk.Muon.eta) < 2.5)
                                      &(events_jk.Muon.pfIsoId > 1) ], 
                                      "Muon")
            FatJet=events_jk.FatJet
            FatJet["p4"] = ak.with_name(events_jk.FatJet[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
            print(FatJet)
            #### Make sure there is at least one jet to run over
            if ak.sum(ak.num(FatJet.pt)>0)<1:
                print("No fat jet pts at all")
                return out
            if self.do_gen:
                era = None
                GenJetAK8 = events_jk.GenJetAK8
                GenJetAK8['p4']= ak.with_name(events_jk.GenJetAK8[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
            else:
                firstidx = filename.find( "store/data/" )
                fname2 = filename[firstidx:]
                fname_toks = fname2.split("/")
                era = fname_toks[ fname_toks.index("data") + 1]
                print("IOV ", IOV, ", era ", era)
            print("starting jet corrections")
            #### Rewrite getjetcorrections to correct all jets and correct sd mass at the same time
            corrected_fatjets = GetJetCorrections(FatJet, events_jk, era, IOV, isData=not self.do_gen)
            corrected_fatjets = corrected_fatjets[corrected_fatjets.subJetIdx1 > -1]
            if ak.sum(ak.num(events_jk.SubJet.mass)>0)<1:
                print("No subjets")
                return out
            corrected_subjets = GetJetCorrections(events_jk.SubJet, events_jk, era, IOV, isData = not self.do_gen, mode = 'AK4')
            corrected_fatjets['msoftdrop'] =   (corrected_subjets[corrected_fatjets.subJetIdx1] + corrected_subjets[corrected_fatjets.subJetIdx2]).mass 

            # if not self.do_minimal:
            #     out["sdmass_orig"].fill(dataset=datastr, jk=jk_index, ptreco=events_jk[(ak.num(events_jk.FatJet) > 2)].FatJet[:,2].pt, mreco=events_jk[(ak.num(events_jk.FatJet) > 2)].FatJet[:,2].msoftdrop)
            #     out["sdmass_ak4corr"].fill(dataset=datastr, jk=jk_index, ptreco=corrected_fatjets[(ak.num(corrected_fatjets) > 2)][:,2].pt, mreco=corrected_fatjets[(ak.num(corrected_fatjets) > 2)][:,2].msoftdrop)
                # corrected_fatjets_ak8 = corrected_fatjets
                # corrected_fatjets_ak8 = GetCorrectedSDMass(corrected_fatjets, events_jk, era, IOV, isData=not self.do_gen, useSubjets = False)
                # out["sdmass_ak8corr"].fill(dataset=datastr, jk=jk_index, ptreco=corrected_fatjets_ak8[(ak.num(corrected_fatjets_ak8) > 2)][:,2].pt, mreco=corrected_fatjets_ak8[(ak.num(corrected_fatjets_ak8) > 2)][:,2].msoftdrop)

            for jetsyst in self.jet_systematics:
                #####################################
                #### For each jet correction, we need to add JMR and JMS corrections on top (except if we're doing data).
                #####################################
                print("Jet syst running over: ", jetsyst)
                # The HEM/JER/JMR/JMS/JES branches below are gated on self.do_gen;
                # for data only 'nominal' assigns corr_jets_final. Skip everything
                # else explicitly rather than falling through to an UnboundLocalError.
                if not self.do_gen and jetsyst != 'nominal':
                    continue
                if jetsyst == 'nominal':
                    if not self.do_gen:
                        print("Doing nominal data")
                        corr_jets_final = deepcopy(corrected_fatjets)
                    else:
                        corr_jets_final = applyjmrSF(IOV, applyjmsSF(IOV,corrected_fatjets))
                elif jetsyst=="HEM" and self.do_gen:
                   corr_jets_final = HEMCleaning(IOV,applyjmrSF(IOV, applyjmsSF(IOV,corrected_fatjets)))
                elif 'JER' in jetsyst and self.do_gen:
                    if "Up" in jetsyst:
                        corr_jets_final  = applyjmrSF(IOV, applyjmsSF(IOV,corrected_fatjets.JER.up))
                        corr_jets_final['msoftdrop'] = (corrected_subjets.JER.up[corrected_fatjets.subJetIdx1] + corrected_subjets.JER.up[corrected_fatjets.subJetIdx2]).mass
                    else: 
                        corr_jets_final  = applyjmrSF(IOV, applyjmsSF(IOV,corrected_fatjets.JER.down))
                        corr_jets_final['msoftdrop'] = (corrected_subjets.JER.down[corrected_fatjets.subJetIdx1] + corrected_subjets.JER.down[corrected_fatjets.subJetIdx2]).mass
                elif "JMR" in jetsyst and self.do_gen:
                    if "Up" in jetsyst:
                        corr_jets_final  = applyjmrSF(IOV, applyjmsSF(IOV,corrected_fatjets), var = "up")
                    else: 
                        corr_jets_final  =  applyjmrSF(IOV, applyjmsSF(IOV,corrected_fatjets), var = "down")
                elif "JMS" in jetsyst and self.do_gen:
                    if "Up" in jetsyst:
                        corr_jets_final  = applyjmrSF(IOV, applyjmsSF(IOV,corrected_fatjets, var = "up"))
                    else:
                        corr_jets_final =  applyjmrSF(IOV, applyjmsSF(IOV,corrected_fatjets, var = "down"))
                elif "JES" in jetsyst and self.do_gen:
                    if jetsyst[-2:]=="Up":
                        field = jetsyst[:-2]
                        corr_jets_final =  applyjmrSF(IOV, applyjmsSF(IOV,corrected_fatjets[field].up))
                        corr_jets_final['msoftdrop'] = (corrected_subjets[field].up[corrected_fatjets.subJetIdx1] + corrected_subjets[field].up[corrected_fatjets.subJetIdx2]).mass
                    elif jetsyst[-4:]=="Down":
                        field = jetsyst[:-4]
                        corr_jets_final =  applyjmrSF(IOV, applyjmsSF(IOV,corrected_fatjets[field].down))
                        corr_jets_final['msoftdrop'] = (corrected_subjets[field].down[corrected_fatjets.subJetIdx1] + corrected_subjets[field].down[corrected_fatjets.subJetIdx2]).mass
                #################################################################
                #### sort corrected jets by pt before being put into events object
                #################################################################

                sortJets_ind = ak.argsort(corr_jets_final.pt, ascending=False)
                corr_jets_sorted = corr_jets_final[sortJets_ind]
                events_corr = ak.with_field(events_jk, corr_jets_sorted, "FatJet")  
                del corr_jets_sorted, corr_jets_final
                ###################################
                ######### INITIALIZE WEIGHTS AND SELECTION
                ##################################
                sel = PackedSelection()
                if (jetsyst == "nominal"): out['cutflow'][datastr]['nEvents initial'] += (len(events.FatJet))
                print("mctype ", mctype, " gen? ", self.do_gen)
                if self.do_gen:
                    if "LHEWeight" in events_corr.fields: 
                        #print("Difference between weights calculated from xsdb and LHE :", (events_corr.LHEWeight.originalXWGTUP - getXSweight(dataset, IOV)))
                        weights = events_corr.LHEWeight.originalXWGTUP * getXSweight(dataset, IOV)
                    else:
                        weights = events_corr.genWeight * getXSweight(dataset, IOV)
                else:
                    ############
                    ### Doing data -- apply lumimask and require at least one jet to apply jet trigger prescales
                    ############
                    lumi_mask = getLumiMask(IOV)(events_corr.run, events_corr.luminosityBlock)
                    
                    events_corr = events_corr[lumi_mask]
                    weights = np.ones(len(events_corr))
                    if "ver2" in dataset:
                        trigsel, psweights, HLTflow_init, HLTflow_final = applyPrescales(events_corr, trigger= "PFJet", year = IOV)
                    else:
                        trigsel, psweights, HLTflow_init, HLTflow_final = applyPrescales(events_corr, year = IOV)
                    for path in HLTflow_init:
                        out['cutflow']['trigger_init'][path] += HLTflow_init[path]
                        out['cutflow']['trigger_final'][path] += HLTflow_final[path]
                    psweights=ak.where(ak.is_none(psweights), 1.0, psweights)
                    trigsel=ak.where(ak.is_none(trigsel), False, trigsel)
                    weights = ak.where(trigsel, psweights, weights)
                    sel.add("trigsel", trigsel)
                    if len(events_corr[trigsel])<1:
                        print("No events after golden json")
                        return out
                    if (jetsyst == "nominal"): 
                        out['cutflow'][datastr]['nEvents after trigger sel '] += (ak.sum(sel.all("trigsel")))
                        print("ADDED TRIGGER TO CUTFLOW FOR NOM FOR KEY ", dataset)
                if self.do_gen:
                    sel.add("npv", events_corr.PV.npvsGood > 0)
                else:
                    sel.add("npv", sel.all("trigsel") & (events_corr.PV.npvsGood > 0))
                ###################################
                ##### Intitialize and fill weights object
                ###################################
                
                weights_obj = Weights(len(weights))
                print("weights ", weights)
                weights_obj.add('initWeight', weight=weights)
                if self.do_gen:
                    #### Apply L1 prefiring weights
                    if "L1PreFiringWeight" in events_corr.fields:                
                        prefiringNom, prefiringUp, prefiringDown = GetL1PreFiringWeight(events_corr)
                        weights_obj.add("L1prefiring", weight=prefiringNom, weightUp=prefiringUp, weightDown=prefiringDown,)
                    #### Apply Pileup reweighting and get up and down uncertainties
                    puNom, puUp, puDown = GetPUSF(events_corr, IOV)
                    weights_obj.add("PUSF", weight=puNom, weightUp=puUp, weightDown=puDown,) 
                    #### Get luminosity uncertainties (nominal weight is 1.0)
                    lumiNom, lumiUp, lumiDown = GetLumiUnc(events_corr, IOV)
                    weights_obj.add("Luminosity", weight=lumiNom, weightUp=lumiUp, weightDown=lumiDown) 
                    #### Get q2 and pdf uncs (not available in pythia+pythia files)
                    if 'herwig' in dataset or 'madgraph' in dataset:
                        pdfNom, pdfUp, pdfDown = GetPDFWeights(events_corr)
                        weights_obj.add("PDF", weight=pdfNom, weightUp=pdfUp, weightDown=pdfDown) 
                        q2muFNom, q2muFUp, q2muFDown = GetQ2muF(events_corr)
                        weights_obj.add("Q2muF", weight=q2muFNom, weightUp=q2muFUp,weightDown=q2muFDown) 
                        q2muRNom, q2muRUp, q2muRDown = GetQ2muR(events_corr)
                        weights_obj.add("Q2muR", weight=q2muRNom, weightUp=q2muRUp, weightDown=q2muRDown) 
                                        #### Apply L1 prefiring weights
                    if "PSWeight" in events_corr.fields:                
                        ISRNom, ISRUp, ISRDown = GetPSWeights(events_corr, shower="ISR")
                        weights_obj.add("ISR", weight=ISRNom, weightUp=ISRUp, weightDown=ISRDown)
                        FSRNom, FSRUp, FSRDown = GetPSWeights(events_corr, shower="FSR")
                        weights_obj.add("FSR", weight=FSRNom, weightUp=FSRUp, weightDown=FSRDown)
                ###################################
                #### Apply MET filters
                ###################################
                METsel = np.array([events_corr.Flag[MET_filters[IOV][i]] for i in range(len(MET_filters[IOV])) if MET_filters[IOV][i] in events_corr.Flag.fields])
                METsel = np.logical_and.reduce(METsel, axis=0) ## a passing event should pass "ALL" the MET filters
                sel.add("METfilters", METsel)
                if ak.sum(METsel) < 1 :
                    print("No events passing MET filters.")
                    return out
                    
                #####################################
                #### Gen Jet Selection
                #################################### 
                if self.do_gen:
                    print("DOING GEN")
                    # if not self.do_minimal:
                    #     HT = ak.sum(events_corr.GenJetAK8.pt, axis=-1)
                        # out["HT_nocuts"].fill(dataset=datastr, systematic=jetsyst, pt=HT)
                        # out["HT_wXS"].fill(dataset=datastr, systematic=jetsyst, pt=HT, weight=weights)
                    # pt_cut_gen = genjet.pt > 200. #### removing
                    sel.add("triGenJet", (ak.num(events_corr.GenJetAK8) > 2)) # & pt_cut_gen ) ####removing to be consistent w/ aritra
                    GenJetAK8 = events_corr.GenJetAK8
                    GenJetAK8['p4']= ak.with_name(events_corr.GenJetAK8[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
                    genjet = ak.firsts(GenJetAK8[:,2:])  
                    rap_cut_gen = ak.where(sel.all("triGenJet"), np.abs(getRapidity(genjet.p4)) < self.ycut, False)
                    sel.add("rapGen", rap_cut_gen)
                    # if not self.do_minimal:
                    #     out["jet_rap_gen"].fill(dataset=datastr, syst = jetsyst, rapidity=getRapidity(GenJetAK8[sel.all("triGenJet")][:,2].p4), weight=weights[sel.all("triGenJet")])
                    #     out["jet_phi_gen"].fill(dataset=datastr, systematic=jetsyst, phi=GenJetAK8[sel.all("triGenJet")][:,2].phi, weight=weights[sel.all("triGenJet")])  
                    #### get dphi and pt asymm selections                     
                    genjet1 = ak.firsts(events_corr.GenJetAK8[:,0:])
                    genjet2 = ak.firsts(events_corr.GenJetAK8[:,1:])
                    ##### genjet3 is already defined genjet
                    #### calculate dphi_min
                    dphi12_gen = np.abs(genjet1.delta_phi(genjet2))
                    dphi13_gen = np.abs(genjet1.delta_phi(genjet))
                    dphi23_gen = np.abs(genjet2.delta_phi(genjet))
                    dphimin_gen = ak.min([dphi12_gen, dphi13_gen, dphi23_gen], axis = 0)
                    dphimin_gen_sel = ak.where(sel.all("triGenJet"), dphimin_gen > 1.0, False)
                    asymm_gen  = np.abs(genjet1.pt - genjet2.pt)/(genjet1.pt + genjet2.pt)
                    sel.add("dphiGen", dphimin_gen_sel)
                    print("Finding min dphi done")
                    # if not self.do_minimal and jetsyst=='nominal':
                    #     out["asymm_gen"].fill(dataset=datastr, systematic=jetsyst, ptgen=events_corr[sel.all("triGenJet")].GenJetAK8[:,2].pt, frac = asymm_gen[sel.all("triGenJet")], weight=weights[sel.all("triGenJet")])
                        # out["dphimin_gen"].fill(dataset=datastr, systematic=jetsyst, dphi = dphimin_gen[sel.all("triGenJet")], weight = weights[sel.all("triGenJet")])
                    gensubjets = events_corr.SubGenJetAK8
                    groomed_genjet = get_gen_sd_mass_jet(ak.firsts(GenJetAK8[:,2:]), gensubjets)
                    sel.add("genTot_seq", sel.all("triGenJet", "dphiGen", "rapGen") & ~ak.is_none(genjet.mass) & ~ak.is_none(groomed_genjet.mass))
                    if (len(events_corr[sel.all("genTot_seq")]) < 1): 
                        print("No gen jets selected")
                        return out        
        
                #####################################
                #### Reco Jet Selection
                ####################################
            
                # if not self.do_minimal and jetsyst=='nominal':
                #     out["njet_reco"].fill(dataset=datastr, syst = jetsyst, n=ak.to_numpy(ak.num(events_corr[sel.all("npv")].FatJet), allow_missing=True), 
                #                          weight = ak.to_numpy(weights[sel.all("npv")], allow_missing=True) )
                FatJet = events_corr.FatJet
                FatJet["p4"] = ak.with_name(events_corr.FatJet[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
                jet = ak.firsts(FatJet[:,2:])
                # pt_cut_reco = jet.pt > 200.    ### applying this just through binning of hist now
                sel.add("triRecoJet", (ak.num(events_corr.FatJet) > 2))
                sel.add("triRecoJet_seq", sel.all('npv', 'METfilters', 'triRecoJet'))
                rap_cut = np.abs(getRapidity(jet.p4)) < self.ycut
                rap_sel = ak.where(sel.all("triRecoJet_seq"), rap_cut, False)
                sel.add("recoRap2p5", rap_sel)
                sel.add("recoRap_seq", sel.all("triRecoJet_seq", "recoRap2p5")) 
                print("nevents after rap cut ", ak.sum(sel.all("recoRap_seq")))
                # if not self.do_minimal and jetsyst=='nominal':
                #     out["jet_rap_reco"].fill(dataset=datastr, syst = jetsyst, rapidity=ak.to_numpy(getRapidity(FatJet[sel.all("triRecoJet_seq")][:,2].p4), allow_missing=True), weight=weights[sel.all("triRecoJet_seq")])
                #     out["jet_phi_reco"].fill(dataset=datastr, systematic=jetsyst, phi=FatJet[sel.all("triRecoJet_seq")][:,2].phi, weight=weights[sel.all("triRecoJet_seq")]) 

                #### ak.first fills empty values with none --> ak.singletons 
                jet1 = ak.firsts(events_corr.FatJet[:,0:])
                jet2 = ak.firsts(events_corr.FatJet[:,1:])
                jet3 = ak.firsts(events_corr.FatJet[:,2:])
                dphi12 = np.abs(jet1.delta_phi(jet2))
                dphi13 = np.abs(jet1.delta_phi(jet3))
                dphi23 = np.abs(jet2.delta_phi(jet3))
                dphimin = ak.min([dphi12, dphi13, dphi23], axis = 0)
                dphi_sel = ak.where(sel.all("triRecoJet_seq"), (dphimin > 1.0), False)
                sel.add("recodphimin", dphi_sel)
                sel.add("recodphi_seq", sel.all("recodphimin", "recoRap_seq"))
                asymm = np.abs(jet1.pt - jet2.pt)/(jet1.pt + jet2.pt)
                if not self.do_minimal:
                    # out["dphimin_reco"].fill(dataset=datastr, systematic=jetsyst, dphi = dphimin[sel.all("triRecoJet_seq")], weight=weights[sel.all("triRecoJet_seq")])
                    out["asymm_reco"].fill(dataset=datastr, systematic=jetsyst, ptreco=events_corr[sel.all("triRecoJet_seq")].FatJet[:,2].pt, frac = asymm[sel.all("triRecoJet_seq")], weight=weights[sel.all("triRecoJet_seq")])
                #### Check that nearest pfmuon and is at least dR > 0.4 away
                # Get the nearest muon to that jet
                muon_sel =  ak.where(sel.all("triRecoJet_seq"), ak.all(jet3.delta_r(ak.singletons(jet3).nearest(events_corr.Muon))>0.4, axis=-1), False)
                sel.add("muonIso0p4", muon_sel)
                jetid_sel = ak.where(sel.all("triRecoJet_seq"), (jet3.jetId > 2), False)
                sel.add("jetId", jetid_sel)
                ####################################
                ### Apply HEM veto
                ####################################
                if IOV == '2018':
                    hemveto, nVetoed = HEMVeto(events_corr.FatJet, events_corr.run)
                    print("hem veto test", hemveto)
                    sel.add('hemveto', hemveto)
                else:
                    hemveto = ak.ones_like(weights, dtype=bool)
                    sel.add('hemveto', hemveto)
                ####  Get Final RECO selection
                sel.add("recoTot_seq", sel.all("recodphi_seq", "jetId", "muonIso0p4", "hemveto") & ~ak.is_none(jet3.mass) & ~ak.is_none(jet3.msoftdrop))
                #### Check eta phi map pre cuts
                if not self.do_minimal and jetsyst=='nominal': 
                        out['jet_eta_phi_precuts'].fill(dataset=datastr, systematic=jetsyst, phi=events_corr[sel.all("triRecoJet")].FatJet[:,2].phi, eta=events_corr[sel.all("triRecoJet")].FatJet[:,2].eta, weight=weights[sel.all("triRecoJet")])                
                if (len(events_corr[sel.all("recoTot_seq")]) < 1): 
                    print("no events passing reco sel")
                    return out 
                ################
                #### Find fakes, misses, and underflow and remove them to get final selection
                ###############
                
                if self.do_gen:
                    jet = ak.firsts(events_corr.FatJet[:,2:])
                    matches = genjet.delta_r(jet) < 0.4
                    sel.add("matched_gen", matches)
                    misses = ~matches | sel.require(genTot_seq=True, recoTot_seq=False)
                    sel.add("misses", misses )
                    print("Number of misses ", ak.sum(misses))
                    miss_sel = misses & sel.all("genTot_seq")
                    print("Number of misses w/ 3 jets ", ak.sum(miss_sel))
                    if ak.sum(miss_sel) > 0 and not self.do_minimal:
                        if jetsyst == "nominal": 
                            out['cutflow'][datastr]['misses'] += (len(events_corr[miss_sel].GenJetAK8))
                        print("Number of none missed jets ", ak.sum(ak.is_none(GenJetAK8[miss_sel][:,2])))
                        ###### Applying misses selection to gen jets and getting sd mass
                        miss_jets = events_corr[miss_sel].GenJetAK8[:,2]
                        groomed_missjet = get_gen_sd_mass_jet(miss_jets, events_corr[miss_sel].SubGenJetAK8)
                        miss_weights = weights[miss_sel]
                        out["misses_u"].fill(dataset=datastr, systematic=jetsyst, jk=jk_index, ptgen = miss_jets[~ak.is_none(miss_jets.mass)].pt, mgen = miss_jets[~ak.is_none(miss_jets.mass)].mass, weight = miss_weights[~ak.is_none(miss_jets.mass)])
                        out["misses_g"].fill(dataset=datastr, systematic=jetsyst, jk=jk_index, ptgen = miss_jets[~ak.is_none(groomed_missjet.mass)].pt, mgen = groomed_missjet[~ak.is_none(groomed_missjet.mass)].mass, weight = miss_weights[~ak.is_none(groomed_missjet.mass)])
                    if len(events_corr[sel.all("genTot_seq", "recoTot_seq", "matched_gen")])<1: 
                        print("No events after all selections and removing misses")
                        return out
                    #### Fakes include events missing a reco mass or sdmass value, events failing index dr matching, and events passing reco cut but failing the gen cut
                    matches = ~ak.is_none(jet.matched_gen)
                    sel.add("matched_reco", matches)
                    print("matched gen jets to reco ", matches)
                    fakes = ~matches | sel.require(genTot_seq=False, recoTot_seq=True)
                    sel.add("fakes", fakes)
                    fake_sel = sel.all("recoTot_seq") & fakes
                    if len(weights[fake_sel])>0 and not self.do_minimal:
                        fake_jets = events_corr[fake_sel].FatJet[:,2]
                        fake_weights = weights_obj.weight()[fake_sel]
                        out["fakes_u"].fill(dataset=datastr, systematic=jetsyst, jk = jk_index, ptreco = fake_jets[~ak.is_none(fake_jets.mass)].pt, mreco = fake_jets[~ak.is_none(fake_jets.mass)].mass, weight = fake_weights[~ak.is_none(fake_jets.mass)])
                        out["fakes_g"].fill(dataset=datastr, systematic=jetsyst, jk = jk_index, ptreco = fake_jets[~ak.is_none(fake_jets.msoftdrop)].pt, mreco = fake_jets[~ak.is_none(fake_jets.msoftdrop)].msoftdrop, weight = fake_weights[~ak.is_none(fake_jets.msoftdrop)])
                        if jetsyst=="nominal":
                            for syst in weights_obj.variations:
                                print("Weight variation: ", syst)
                                fake_weights = weights_obj.weight(syst)[fake_sel]
                                out["fakes_u"].fill(dataset=datastr, systematic=syst, jk = jk_index, ptreco = fake_jets[~ak.is_none(fake_jets.mass)].pt, mreco = fake_jets[~ak.is_none(fake_jets.mass)].mass, weight = fake_weights[~ak.is_none(fake_jets.mass)])
                                out["fakes_g"].fill(dataset=datastr, systematic=syst, jk = jk_index, ptreco = fake_jets[~ak.is_none(fake_jets.msoftdrop)].pt, mreco = fake_jets[~ak.is_none(fake_jets.msoftdrop)].msoftdrop, weight = fake_weights[~ak.is_none(fake_jets.msoftdrop)])
                    if (jetsyst == "nominal"): 
                        out['cutflow'][datastr]['fakes'] += (len(events_corr[fakes].FatJet))
                    if len(events_corr[sel.all("genTot_seq", "recoTot_seq", "matched_reco", "matched_gen")])<1: 
                        print("No events after all selections and removing fakes & misses")
                        return out
                    ##############
                    ### Make final selection and fill gen truth plots
                    ###############
                    sel.add("final_seq", sel.all("genTot_seq", "recoTot_seq", "matched_reco", "matched_gen"))

                    genjet = events_corr[sel.all("genTot_seq","matched_gen")].GenJetAK8[:,2]
                    groomed_genjet = get_gen_sd_mass_jet(genjet, events_corr[sel.all("genTot_seq","matched_gen")].SubGenJetAK8)
                    gen_weights = weights[sel.all("genTot_seq","matched_gen")]
                    out['ptjet_mjet_u_gen'].fill(dataset=datastr, systematic=jetsyst, jk=jk_index, ptgen=genjet.pt, mgen=genjet.mass, weight=gen_weights )
                    out['ptjet_mjet_g_gen'].fill(dataset=datastr, systematic=jetsyst, jk=jk_index, ptgen=genjet.pt, mgen=groomed_genjet.mass, weight=gen_weights )
                    out['ptjet_rhojet_u_gen'].fill(dataset=datastr, systematic=jetsyst, jk=jk_index, ptgen=genjet.pt, mpt_gen=2*np.log10(genjet.mass/genjet.pt), weight=gen_weights )
                    out['ptjet_rhojet_g_gen'].fill(dataset=datastr, systematic=jetsyst, jk=jk_index, ptgen=genjet.pt, mpt_gen=2*np.log10(groomed_genjet.mass/genjet.pt), weight=gen_weights )
                    #######################
                else:
                    sel.add("final_seq", sel.all("recoTot_seq"))

                #######################
                #### Apply final selections and jet veto map
                #######################
                if len(events_corr[sel.all("final_seq")])<1:
                        print("no more events after final sel")
                        return out
                
                #### Check eta phi map after cuts but before jet veto
                #### Make eta phi plot to check effects of cuts
                # if not self.do_minimal: out['jet_eta_phi_preveto'].fill(dataset=datastr, systematic=jetsyst, phi=events_corr.FatJet[:,2].phi, eta=events_corr.FatJet[:,2].eta, weight=weights)      
                
                #### Apply jet veto map
                # jet = events_corr.FatJet[:,2]
                # veto = ApplyVetoMap(IOV, jet, mapname='jetvetomap')
                # events_corr = events_corr[veto]
                # weights = weights[veto]
                # if len(events_corr)<1:
                #         print("no more events after jet veto")
                #         return out
 
                #######################
                #### Get final jets and weights and fill final plots
                #######################
                final_weights = weights_obj.weight()[sel.all("final_seq")]
                jet = events_corr[sel.all("final_seq")].FatJet[:,2]
                    
                ##################
                #### Apply final selections to GEN and fill any plots requiring gen, including resp. matrices
                ##################
                
                if self.do_gen:
                    ##### define reco-only jets to find ptjet_mjet_*_reco to get fakes
                    recojet = events_corr[sel.all("recoTot_seq", "matched_reco")].FatJet[:,2]
                    reco_weights = weights_obj.weight()[sel.all("recoTot_seq", "matched_reco")]
                    #### define gen jets with final selection for response matrix
                    genjet = events_corr[sel.all("final_seq")].FatJet[:,2].matched_gen
                    groomed_genjet = get_gen_sd_mass_jet(genjet, events_corr[sel.all("final_seq")].SubGenJetAK8)
                    weird_jets = events_corr[(events_corr[sel.all("final_seq")].GenJetAK8[:,2].mass < 20.) & (events_corr[sel.all("final_seq")].FatJet[:,2].mass >20.)]
                    if jetsyst == "nominal": out['cutflow'][datastr]['nEvents weird (mreco>20, mgen<20) ungroomed'] += len(weird_jets)
                    #### plots to check backgrounds and eta phi dists
                    if not self.do_minimal and jetsyst=='nominal':
                        #### plots for checking MET/sumET --> potentially need cut <0.3job
                        out["MET_over_sumET_pt_reco"].fill(dataset=datastr,systematic=jetsyst, frac=events_corr[sel.all("final_seq")].MET.pt/events_corr[sel.all("final_seq")].MET.sumEt, ptreco=jet.pt, weight=final_weights )
                        out["MET_pt_reco"].fill(dataset=datastr,systematic=jetsyst, pt=events_corr[sel.all("final_seq")].MET.pt, ptreco=jet.pt, weight=final_weights )
                        #### plots for checking whether jet veto map is needed
                        HT = ak.sum(events_corr[sel.all("final_seq")].GenJetAK8.pt, axis=-1)
                        out["HT_aftercuts"].fill(dataset=datastr, systematic=jetsyst, pt=HT, weight=final_weights)
                        # out["ptreco_mreco_fine_u"].fill(dataset=datastr,systematic=jetsyst, jk=jk_index, pt=jet.pt, mass=jet.mass, weight=reco_weights )
                        # out["ptreco_mreco_fine_g"].fill(dataset=datastr,systematic=jetsyst, jk=jk_index, pt=jet.pt, mass=jet.msoftdrop, weight=reco_weights )
                        
                    out["response_matrix_u"].fill(dataset=datastr, systematic=jetsyst, jk=jk_index, ptreco=jet.pt, ptgen=genjet.pt, 
                                                  mreco=jet.mass, mgen=genjet.mass, weight = final_weights)
                    out["response_matrix_g"].fill(dataset=datastr, systematic=jetsyst, jk=jk_index, ptreco=jet.pt, ptgen=genjet.pt, 
                                                  mreco=jet.msoftdrop, mgen=groomed_genjet.mass, weight = final_weights )
                    out["response_matrix_rho_u"].fill(dataset=datastr, systematic=jetsyst, jk=jk_index,  mpt_reco=2*np.log10(jet.mass/jet.pt), mpt_gen=2*np.log10(genjet.mass/genjet.pt), ptreco=jet.pt, ptgen=genjet.pt, weight=final_weights)
                    out["response_matrix_rho_g"].fill(dataset=datastr, systematic=jetsyst, jk=jk_index,  mpt_reco=2*np.log10(jet.msoftdrop/jet.pt), mpt_gen=2*np.log10(groomed_genjet.mass/genjet.pt), ptreco=jet.pt, ptgen=genjet.pt, weight=final_weights)
                    out["ptjet_mjet_u_reco"].fill(dataset=datastr, systematic=jetsyst, jk=jk_index, ptreco=recojet.pt, mreco=recojet.mass, weight=reco_weights)
                    out["ptjet_mjet_g_reco"].fill(dataset=datastr, systematic=jetsyst, jk=jk_index, ptreco=recojet.pt, mreco=recojet.msoftdrop, weight=reco_weights )
                    out["ptjet_rhojet_u_reco"].fill(dataset=datastr, systematic=jetsyst, jk=jk_index, ptreco=recojet.pt, mpt_reco=2*np.log10(recojet.mass/recojet.pt), weight=reco_weights)
                    out["ptjet_rhojet_g_reco"].fill(dataset=datastr, systematic=jetsyst, jk=jk_index, ptreco=recojet.pt,mpt_reco=2*np.log10(recojet.msoftdrop/recojet.pt), weight=reco_weights )
                    if not self.do_minimal:
                        out["jet_pt_eta_phi"].fill(dataset=datastr, systematic=jetsyst, ptreco=jet.pt, phi=jet.phi, eta=jet.eta, weight=final_weights)
                        out['m_u_jet_reco_over_gen'].fill(dataset=dataset, ptgen=genjet.pt, mgen=genjet.mass, frac = jet.mass/genjet.mass, 
                                                           weight = final_weights)
                        out['m_g_jet_reco_over_gen'].fill(dataset=dataset, ptgen=genjet.pt, mgen=groomed_genjet.mass, 
                                                           frac = jet.msoftdrop/groomed_genjet.mass, weight = final_weights)
                    if jetsyst=="nominal":
                        for syst in weights_obj.variations:
                            print("Weight variation: ", syst)
                            reco_weights = weights_obj.weight(syst)[sel.all("recoTot_seq", "matched_reco")]
                            final_weights = weights_obj.weight(syst)[sel.all("final_seq")]
                            #fill nominal, up, and down variations for each          
                            out["response_matrix_u"].fill(dataset=datastr, systematic=syst, jk=jk_index,ptreco=jet.pt, mreco=jet.mass, ptgen=genjet.pt, mgen=genjet.mass, weight=final_weights)
                            out["response_matrix_g"].fill(dataset=datastr, systematic=syst, jk=jk_index, ptreco=jet.pt, mreco=jet.msoftdrop, ptgen=genjet.pt, mgen=groomed_genjet.mass, weight=final_weights)
                            out["response_matrix_rho_u"].fill(dataset=datastr, systematic=syst, jk=jk_index,  mpt_reco=2*np.log10(jet.mass/jet.pt), mpt_gen=2*np.log10(genjet.mass/genjet.pt), ptreco=jet.pt, ptgen=genjet.pt, weight=final_weights)
                            out["response_matrix_rho_g"].fill(dataset=datastr, systematic=syst, jk=jk_index,  mpt_reco=2*np.log10(jet.msoftdrop/jet.pt),mpt_gen=2*np.log10(groomed_genjet.mass/genjet.pt), ptreco=jet.pt, ptgen=genjet.pt, weight=final_weights)
                            out["ptjet_mjet_u_reco"].fill(dataset=datastr, systematic=syst, jk=jk_index, ptreco=recojet.pt, 
                                                          mreco=recojet.mass, weight=reco_weights )
                            out["ptjet_mjet_g_reco"].fill(dataset=datastr, systematic=syst, jk=jk_index, ptreco=recojet.pt, 
                                                       mreco=recojet.msoftdrop, weight=reco_weights )
                            out["ptjet_rhojet_u_reco"].fill(dataset=datastr, systematic=syst, jk=jk_index, ptreco=recojet.pt, mpt_reco=2*np.log10(recojet.mass/recojet.pt), weight=reco_weights)
                            out["ptjet_rhojet_g_reco"].fill(dataset=datastr, systematic=syst, jk=jk_index, ptreco=recojet.pt,mpt_reco=2*np.log10(recojet.msoftdrop/recojet.pt), weight=reco_weights )
                            if not self.do_minimal:
                                out["jet_pt_eta_phi"].fill(dataset=datastr, systematic=syst, ptreco=jet.pt, phi=jet.phi, eta=jet.eta, weight=final_weights)
                        #### Gluon purity plots        
                        jet1flav = getJetFlavors(events_corr[sel.all("final_seq")].FatJet[:,0])
                        jet2flav = getJetFlavors(events_corr[sel.all("final_seq")].FatJet[:,1])
                        jet3flav = getJetFlavors(events_corr[sel.all("final_seq")].FatJet[:,2])
                        genjet1 = events_corr[sel.all("final_seq")].FatJet[:,0].matched_gen
                        genjet2 = events_corr[sel.all("final_seq")].FatJet[:,1].matched_gen
                        jet3 = events_corr[sel.all("final_seq")].FatJet[:,2]
                        jet3_bb = jet3[(np.abs(genjet1.partonFlavour) == 5) & (np.abs(genjet2.partonFlavour) == 5)]
                        jet3_b = jet3[(np.abs(genjet1.partonFlavour) == 5)]
                        jet3_jetbb_flav = getJetFlavors(jet3_bb)
                        jet3_jetb_flav = getJetFlavors(jet3_b)
                        
                        jets = {"jet1":jet1flav, "jet2":jet2flav,  "jet3":jet3flav, "jet3_bb":jet3_jetbb_flav, "jet3_b":jet3_jetb_flav}
                        if not self.do_minimal:
                            for flavor in jet1flav.keys():
                                for jetname, jetobj in jets.items():
                                    jetobj[flavor] = jetobj[flavor][~ak.is_none(jetobj[flavor])]
                                    out['alljet_ptreco_mreco'].fill(dataset=datastr, jetNumb = jetname, partonFlav = flavor, 
                                                                    mreco = jetobj[flavor].mass, 
                                                                    ptreco = jetobj[flavor].pt)
                                    out['btag_eta'].fill(dataset=datastr, jetNumb = jetname, partonFlav = flavor, 
                                                         frac = jetobj[flavor].btagDeepB, eta = jetobj[flavor].eta )
                        out['cutflow'][datastr]['nGluonJets'] += (len(jet3flav["Gluon"])+len(jet1flav["Gluon"])+len(jet2flav["Gluon"]))
                        out['cutflow'][datastr]['nJets'] += (len(events_corr[sel.all("final_seq")].FatJet[:,0])+len(events_corr[sel.all("final_seq")].FatJet[:,1])+len(events_corr[sel.all("final_seq")].FatJet[:,2]))
                        out['cutflow'][datastr]['nSoftestGluonJets'] += (len(jet3flav["Gluon"]))
                        out['cutflow'][datastr]['nSoftestGluonJets_b'] += (len(jet3_jetb_flav["Gluon"]))
                        out['cutflow'][datastr]['nSoftestGluonJets_bb'] += (len(jet3_jetbb_flav["Gluon"]))
                        out['cutflow'][datastr]['nSoftestJets_b'] += (len(jet3_b))
                        out['cutflow'][datastr]['nSoftestJets_bb'] += (len(jet3_bb))
                        out['cutflow'][datastr]['n3Jets'] += (len(events_corr[sel.all("final_seq")].FatJet[:,2].pt))
                ###############
                ##### If running over DATA fill only final reco plots
                ###############
                
                else:
                    out["ptjet_mjet_u_reco"].fill(dataset=datastr, systematic=jetsyst, jk=jk_index, ptreco=jet.pt, mreco=jet.mass, weight=final_weights  )
                    out["ptjet_mjet_g_reco"].fill(dataset=datastr, systematic=jetsyst, jk=jk_index, ptreco=jet.pt, mreco=jet.msoftdrop, weight=final_weights  )
                    out["ptjet_rhojet_u_reco"].fill(dataset=datastr, systematic=jetsyst, jk=jk_index, ptreco=jet.pt, mpt_reco=2*np.log10(jet.mass/jet.pt), weight=final_weights)
                    out["ptjet_rhojet_g_reco"].fill(dataset=datastr, systematic=jetsyst, jk=jk_index, ptreco=jet.pt,mpt_reco=2*np.log10(jet.msoftdrop/jet.pt), weight=final_weights )
                    if not self.do_minimal:
                        # out["ptreco_mreco_fine_u"].fill(dataset=datastr,systematic=jetsyst, jk=jk_index, pt=jet.pt, mass=jet.mass, weight=reco_weights  )
                        # out["ptreco_mreco_fine_g"].fill(dataset=datastr,systematic=jetsyst, jk=jk_index, pt=jet.pt, mass=jet.msoftdrop, weight=reco_weights  )
                        out["MET_over_sumET_pt_reco"].fill(dataset=datastr,systematic=jetsyst, frac=events_corr[sel.all("final_seq")].MET.pt/events_corr[sel.all("final_seq")].MET.sumEt, ptreco=jet.pt, weight=final_weights  )
                        out["MET_pt_reco"].fill(dataset=datastr,systematic=jetsyst, pt=events_corr[sel.all("final_seq")].MET.pt, ptreco=jet.pt, weight=final_weights )
                        out["jet_pt_eta_phi"].fill(dataset=datastr, systematic=jetsyst, ptreco=jet.pt, phi=jet.phi, eta=jet.eta, weight=final_weights )
                        HT = ak.sum(events_corr[sel.all("final_seq")].FatJet.pt, axis=-1)
                        out["HT_aftercuts"].fill(dataset=datastr, systematic=jetsyst, pt=HT, weight=final_weights )
                print("final jets ", jet)
                print("final jet pt ", jet.pt)
                print("final number of events added to hists " , len(events_corr))
                if (jetsyst == "nominal"):
                    for name in sel.names:
                        out["cutflow"][datastr][name] += sel.all(name).sum()
                        print("ADDED ", name, " TO CUTFLOW")
                negMSD = jet.msoftdrop<0.
                print("Number of negative softdrop values ", ak.sum(negMSD) )
                if (jetsyst == "nominal"): 
                    out['cutflow'][datastr]['nEvents failing softdrop condition'] += ak.sum(negMSD)
                    print("ADDED NEG SD EVENTS TO CUTFLOW")
                del events_corr, weights
            del events_jk
        out['cutflow'][datastr]['chunks'] += 1
        return out
    
    def postprocess(self, accumulator):
        return accumulator
    
# def main():
#     #### Next run processor with futures executor on all test files
#     from dask.distributed import Client
#     from plugins import runCoffeaJob
#     processor = makeTrijetHists()
#     result = runCoffeaJob(processor, jsonFile = "QCD_flat_files.json", winterfell = True, testing = True, data = False)
#     util.save(result, "coffeaOutput/trijet_pT" + str(processor.ptcut) + "_eta" + str(processor.etacut) + "_result_test.coffea")
    

if __name__ == '__main__':
    # Execute when the module is not initialized from an import statement: i.e. called from terminal command line
    main()

