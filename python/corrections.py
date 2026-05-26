import numpy as np
import awkward as ak
import correctionlib
from coffea.jetmet_tools import JetResolutionScaleFactor
from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
from coffea.lookup_tools import extractor
import copy
import os
from pathlib import Path

from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CORR = _PROJECT_ROOT / "correctionFiles"


def corr_path(*parts):
    return str(_CORR.joinpath(*parts))

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

#based heavily on https://github.com/b2g-nano/TTbarAllHadUproot/blob/optimize/python/corrections.py and https://gitlab.cern.ch/gagarwal/ttbardileptonic/-/blob/master/corrections.py?ref_type=heads and https://gitlab.cern.ch/gagarwal/ttbardileptonic/-/blob/master/jmeCorrections.py?ref_type=heads

def ApplyVetoMap(IOV, jets, mapname='jetvetomap'):
    if IOV=="2016APV":
        IOV="2016"
    fname = corr_path("jetvetomap", "jetvetomaps_UL"+IOV+".json.gz")
    hname = {
        "2016"   : "Summer19UL16_V1",
        "2017"   : "Summer19UL17_V1",
        "2018"   : "Summer19UL18_V1"
    }
    print("Len of jets before veto", len(jets))
    evaluator = correctionlib.CorrectionSet.from_file(fname)
    jetphi = np.where(jets.phi<3.141592, jets.phi, 3.141592)
    jetphi = np.where(jetphi>-3.141592, jetphi, -3.141592)
    vetoedjets = np.array(evaluator[hname[IOV]].evaluate(mapname, np.array(jets.eta), jetphi), dtype=bool)
    print("Sum of vetoed jets ", ak.sum(vetoedjets), " len of veto jets ", len(vetoedjets))
    print("Len of jets AFTER veto", len(jets[~vetoedjets]))
    return ~vetoedjets
    
def applyjmsSF(IOV, FatJet,  var = ''):
    jmsSF = {

        # "2016APV":{"sf": 1.00, "sfup": 1.0094, "sfdown": 0.9906}, 

        # "2016"   :{"sf": 1.00, "sfup": 1.0094, "sfdown": 0.9906}, 

        # "2017"   :{"sf": 0.982, "sfup": 0.986, "sfdown": 0.978},

        # "2018"   :{"sf": 0.999, "sfup": 1.001, "sfdown": 0.997}} 
    
            "2016APV":{"sf": 1.00, "sfup": 1.01, "sfdown": 0.99}, 

        "2016"   :{"sf": 1.00, "sfup": 1.01, "sfdown": 0.99}, 

        "2017"   :{"sf": 1.0, "sfup": 1.01, "sfdown": 0.99},

        "2018"   :{"sf": 1.0, "sfup": 1.01, "sfdown": 0.99}} 
    
    out = jmsSF[IOV]["sf"+var]
    

    FatJet = ak.with_field(FatJet, FatJet.mass * out, 'mass')
    FatJet = ak.with_field(FatJet, FatJet.msoftdrop * out, 'msoftdrop')
    return FatJet

def applyJMSbypt(IOV, FatJet, var = ''):

    ###### NEED JET FACTORY
    fname = corr_path("SFs", "ParticleNet_jmssf.json")
    iovKey = {
        "2016": "16preVFP",
        "2016APV": "16postVFP",
        "2017" : "17",
        "2018" : "18"
    }
    key = "jmssf_UL"+iovKey[IOV]
    evaluator = correctionlib.CorrectionSet.from_file(fname)
    if var == "Up":
        jms = evaluator[key].evaluate(np.array(FatJet.pt), "up")
    elif var == "Down":
        jms = evaluator[key].evaluate(np.array(FatJet.pt), "down")
    else:
        jms = evaluator[key].evaluate(np.array(FatJet.pt))
    print("PT is ", FatJet.pt, " and nom value is ", jmsNom)
    FatJet = ak.with_field(FatJet, FatJet.mass * jms, 'mass')
    FatJet = ak.with_field(FatJet, FatJet.msoftdrop * jms, 'msoftdrop')
    return FatJet

def applyjmrSF(IOV, FatJet, var = ''):
    jmrSF = {

       #  "2016APV":{"sf": 1.00, "sfup": 1.2, "sfdown": 0.8}, 
       #  "2016"   :{"sf": 1.00, "sfup": 1.2, "sfdown": 0.8}, 

       #  "2017"   :{"sf": 1.09, "sfup": 1.14, "sfdown": 1.04},

       #  "2018"   :{"sf": 1.108, "sfup": 1.142, "sfdown": 1.074}}   
    ##### new recmmendations is 2% uncertainty https://twiki.cern.ch/twiki/bin/viewauth/CMS/SoftDropJMSJMRULRun2

        "2016APV":{"sf": 1.0, "sfup": 1.02, "sfdown": 0.98}, 
        "2016"   :{"sf": 1.0, "sfup": 1.02, "sfdown": 0.98}, 

        "2017"   :{"sf": 1.0, "sfup": 1.02, "sfdown": 0.98},

        "2018"   :{"sf": 1.0, "sfup": 1.02, "sfdown": 0.98}}  
    
    jmrvalnom = jmrSF[IOV]["sf"+var]
    
    recomass = FatJet.mass
    genmass = FatJet.matched_gen.mass    
    
    deltamass = (recomass-genmass)*(jmrvalnom-1.0)
    condition = ((recomass+deltamass)/recomass) > 0
    jmrnom = ak.where( recomass <= 0.0, 0 , ak.where( condition , (recomass+deltamass)/recomass, 0 ))


    FatJet = ak.with_field(FatJet, FatJet.mass * jmrnom, 'mass')
    FatJet = ak.with_field(FatJet, FatJet.msoftdrop * jmrnom, 'msoftdrop')

    return FatJet
    
def GetPSWeights(df, shower = "ISR"):
    """ Return nominal, up, down weights for ISR or FSR.

    Some MC samples store PSWeight as a variable-length list with fewer than 4
    entries (e.g. QCD_Pt_* often has just [1]). Under awkward 2 (coffea 2025+)
    indexing past the end raises IndexError instead of silently returning None,
    so pad to length 4 with the neutral weight 1.0.
    """
    ps = ak.fill_none(ak.pad_none(df.PSWeight, 4, axis=1, clip=True), 1.0)
    ones = ak.ones_like(df.event)
    if shower == "ISR":
        return ones, ps[:,0], ps[:,2]
    elif shower == "FSR":
        return ones, ps[:,1], ps[:,3]
        
def GetL1PreFiringWeight(events):
    # original code https://gitlab.cern.ch/gagarwal/ttbardileptonic/-/blob/master/TTbarDileptonProcessor.py#L50
    ## Reference: https://twiki.cern.ch/twiki/bin/viewauth/CMS/L1PrefiringWeightRecipe
    # Need to check if weights (up/dn) produced are comparable or greater than JEC weights --> if yes apply, if no take as a systematic uncertainty
    ## var = "Nom", "Up", "Dn"
    L1PrefiringWeights = [events.L1PreFiringWeight.Nom, events.L1PreFiringWeight.Up, events.L1PreFiringWeight.Dn]
    return L1PrefiringWeights

def HEMCleaning(IOV, JetCollection):
    ## Reference: https://hypernews.cern.ch/HyperNews/CMS/get/JetMET/2000.html
    isHEM = ak.ones_like(JetCollection.pt)
    if (IOV == "2018"):
        detector_region1 = ((JetCollection.phi < -0.87) & (JetCollection.phi > -1.57) &
                           (JetCollection.eta < -1.3) & (JetCollection.eta > -2.5))
        detector_region2 = ((JetCollection.phi < -0.87) & (JetCollection.phi > -1.57) &
                           (JetCollection.eta < -2.5) & (JetCollection.eta > -3.0))
        jet_selection    = ((JetCollection.jetId > 1) & (JetCollection.pt > 15))

        isHEM            = ak.where(detector_region1 & jet_selection, 0.80, isHEM)
        isHEM            = ak.where(detector_region2 & jet_selection, 0.65, isHEM)
    JetCollection = ak.with_field(JetCollection, JetCollection.pt*isHEM, "pt" )
    return JetCollection
    
def HEMVeto(FatJets, runs):

    ## Reference: https://hypernews.cern.ch/HyperNews/CMS/get/JetMET/2000.html
    
    runid = (runs >= 319077)
    print(runid)
    detector_region1 = ((FatJets.phi < -0.87) & (FatJets.phi > -1.57) &
                       (FatJets.eta < -1.3) & (FatJets.eta > -2.5))
    detector_region2 = ((FatJets.phi < -0.87) & (FatJets.phi > -1.57) &
                       (FatJets.eta < -2.5) & (FatJets.eta > -3.0))
    jet_selection    = ((FatJets.jetId > 1) & (FatJets.pt > 15))

    vetoHEMFatJets = ak.any((detector_region1 & jet_selection & runid) ^ (detector_region2 & jet_selection & runid), axis=1)
    vetoHEM = ~(vetoHEMFatJets)
    
    return vetoHEM, ak.sum(vetoHEMFatJets)

def GetLumiUnc(events, IOV):
    lumi_unc = {"2016": 0.016,#0.012,
                "2016APV":  0.016,#0.012,
                "2017":  0.016,#0.023,
                "2018": 0.016,}#0.025}
    lumi_nom = ak.ones_like(events.L1PreFiringWeight.Nom)
    print("AK ones like event ", lumi_nom)
    print("Lumi unc val", lumi_unc[IOV] )
    lumi_up = (1.0+lumi_unc[IOV])*lumi_nom
    lumi_dn = (1.0-lumi_unc[IOV])*lumi_nom
    return lumi_nom, lumi_up, lumi_dn

def GetPUSF(events, IOV):
    # original code https://gitlab.cern.ch/gagarwal/ttbardileptonic/-/blob/master/TTbarDileptonProcessor.py#L38
    ## json files from: https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/LUM
        
    fname = corr_path("puWeights", "{0}_UL".format(IOV), "puWeights.json.gz")
    # print("PU SF filename: ", fname)
    hname = {
        "2016APV": "Collisions16_UltraLegacy_goldenJSON",
        "2016"   : "Collisions16_UltraLegacy_goldenJSON",
        "2017"   : "Collisions17_UltraLegacy_goldenJSON",
        "2018"   : "Collisions18_UltraLegacy_goldenJSON"
    }
    evaluator = correctionlib.CorrectionSet.from_file(fname)

    puUp = evaluator[hname[str(IOV)]].evaluate(np.array(events.Pileup.nTrueInt), "up")
    puDown = evaluator[hname[str(IOV)]].evaluate(np.array(events.Pileup.nTrueInt), "down")
    puNom = evaluator[hname[str(IOV)]].evaluate(np.array(events.Pileup.nTrueInt), "nominal")

    return puNom, puUp, puDown

def GetCorrectedSDMass(events, era, IOV, isData=False, uncertainties=None, useSubjets=True):
    # print("Nevents with negative subjet id ", ak.sum(events.FatJet.subJetIdx1 < 0), " out of ", len(events))
    if useSubjets:
        SubJets=events.SubJet
        SubJets["p4"] = ak.with_name(events.SubJet[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
        corr_subjets = GetJetCorrections(SubJets, events, era, IOV, isData=isData, uncertainties = uncertainties, mode='AK4')
        del SubJets
    else:
        FatJets = events.FatJet
        FatJets["p4"] = ak.with_name(FatJets[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
        FatJets["mass"] = FatJets.msoftdrop
        corr_subjets = GetJetCorrections(FatJets, events, era, IOV, isData=isData, uncertainties = uncertainties )
        corr_subjets =corr_subjets[corr_jets.subJetIdx1 > -1 ] 
        del FatJets
    corr_jets =corr_jets[(corr_jets.subJetIdx1 > -1)]
    fields = []
    # print("Avail corrected jet objs ", corr_jets.fields)
    fields.extend(field for field in corr_jets.fields if ("JES_" in field))
    if ("JER" in corr_jets.fields): fields.append("JER")
    if useSubjets:
        corr_jets["msoftdrop"] = (corr_subjets[events.FatJet.subJetIdx1]+corr_subjets[corr_jets.subJetIdx2]).mass
    else:
        corr_jets["msoftdrop"] = corr_subjets.mass
    for field in fields:
        if useSubjets:
            corr_jets[field]["up"]["msoftdrop"] = (corr_subjets[field]["up"][corr_jets.subJetIdx1]+corr_subjets[field]["up"][corr_jets.subJetIdx2]).mass
            corr_jets[field]["down"]["msoftdrop"] = (corr_subjets[field]["down"][corr_jets.subJetIdx1]+corr_subjets[field]["down"][corr_jets.subJetIdx2]).mass
        else:
            corr_jets[field]["up"]["msoftdrop"] = corr_subjets[field]["up"].mass
            corr_jets[field]["down"]["msoftdrop"] = corr_subjets[field]["down"].mass
    del corr_subjets
    # print("AK8 sdmass before corr ", events.FatJet.msoftdrop, " and after ", corr_jets.msoftdrop)
    return corr_jets

def GetJetCorrections(FatJets, events, era, IOV, isData=False, uncertainties = None, mode="AK8" ):
    AK_str = 'AK8PFPuppi'
    if mode=="AK4":
        AK_str = 'AK4PFPuppi'
    if uncertainties == None:
        uncertainty_sources = ["AbsoluteMPFBias","AbsoluteScale","AbsoluteStat","FlavorQCD","Fragmentation","PileUpDataMC","PileUpPtBB","PileUpPtEC1","PileUpPtEC2","PileUpPtHF",
"PileUpPtRef","RelativeFSR","RelativeJEREC1","RelativeJEREC2","RelativeJERHF","RelativePtBB","RelativePtEC1","RelativePtEC2","RelativePtHF","RelativeBal","RelativeSample", "RelativeStatEC","RelativeStatFSR","RelativeStatHF","SinglePionECAL","SinglePionHCAL","TimePtEta"]
    else:
        uncertainty_sources = uncertainties
    # original code https://gitlab.cern.ch/gagarwal/ttbardileptonic/-/blob/master/jmeCorrections.py
    jer_tag=None
    if (IOV=='2018'):
        jec_tag="Summer19UL18_V5_MC"
        jec_tag_data={
            "Run2018A": "Summer19UL18_RunA_V6_DATA",
            "Run2018B": "Summer19UL18_RunB_V6_DATA",
            "Run2018C": "Summer19UL18_RunC_V6_DATA",
            "Run2018D": "Summer19UL18_RunD_V6_DATA",
        }
        jer_tag = "Summer19UL18_JRV2_MC"
    elif (IOV=='2017'):
        # jec_tag="Summer19UL17_V5_MC"
        # jec_tag_data={
        #     "Run2017B": "Summer19UL17_RunB_V5_DATA",
        #     "Run2017C": "Summer19UL17_RunC_V5_DATA",
        #     "Run2017D": "Summer19UL17_RunD_V5_DATA",
        #     "Run2017E": "Summer19UL17_RunE_V5_DATA",
        #     "Run2017F": "Summer19UL17_RunF_V5_DATA",
        # }
        jec_tag="Summer19UL17_V6_MC"
        jec_tag_data={
            "Run2017B": "Summer19UL17_RunB_V6_DATA",
            "Run2017C": "Summer19UL17_RunC_V6_DATA",
            "Run2017D": "Summer19UL17_RunD_V6_DATA",
            "Run2017E": "Summer19UL17_RunE_V6_DATA",
            "Run2017F": "Summer19UL17_RunF_V6_DATA",
        }
        jer_tag = "Summer19UL17_JRV3_MC"
    elif (IOV=='2016'):
        jec_tag="Summer19UL16_V7_MC"
        jec_tag_data={
            "Run2016F": "Summer19UL16_RunFGH_V7_DATA",
            "Run2016G": "Summer19UL16_RunFGH_V7_DATA",
            "Run2016H": "Summer19UL16_RunFGH_V7_DATA",
        }
        jer_tag = "Summer20UL16_JRV3_MC"
    elif (IOV=='2016APV'):
        jec_tag="Summer19UL16_V7_MC"
        ## HIPM/APV     : B_ver1, B_ver2, C, D, E, F
        ## non HIPM/APV : F, G, H

        jec_tag_data={
            "Run2016B": "Summer19UL16APV_RunBCD_V7_DATA",
            "Run2016C": "Summer19UL16APV_RunBCD_V7_DATA",
            "Run2016D": "Summer19UL16APV_RunBCD_V7_DATA",
            "Run2016E": "Summer19UL16APV_RunEF_V7_DATA",
            "Run2016F": "Summer19UL16APV_RunEF_V7_DATA",
        }
        jer_tag = "Summer20UL16APV_JRV3_MC"
    else:
        print(f"Error: Unknown year \"{IOV}\".")


    ext = extractor()
    if not isData:
    #For MC
        # print("File "+'correctionFiles/JEC/{0}/{0}_L1FastJet_{1}.jec.txt'.format(jec_tag, AK_str)+" exists: ", os.path.exists('correctionFiles/JEC/{0}/{0}_L1FastJet_{1}.jec.txt'.format(jec_tag, AK_str)))
        ext.add_weight_sets([
            '* * '+corr_path("JEC", jec_tag, '{0}_L1FastJet_{1}.jec.txt'.format(jec_tag, AK_str)),
            '* * '+corr_path("JEC", jec_tag, '{0}_L2Relative_{1}.jec.txt'.format(jec_tag, AK_str)),
            '* * '+corr_path("JEC", jec_tag, '{0}_L3Absolute_{1}.jec.txt'.format(jec_tag, AK_str)),
            '* * '+corr_path("JEC", jec_tag, '{0}_UncertaintySources_{1}.junc.txt'.format(jec_tag, AK_str)),
            '* * '+corr_path("JEC", jec_tag, '{0}_Uncertainty_{1}.junc.txt'.format(jec_tag, AK_str)),
        ])
        print("done adding jec/junc files")
        #### Do AK8PUPPI jer files exist??
        if jer_tag:
            # print("JER tag: ", jer_tag)
            # print("File "+'correctionFiles/JER/{0}/{0}_PtResolution_AK8PFPuppi.jr.txt'.format(jer_tag)+" exists: ", os.path.exists('correctionFiles/JER/{0}/{0}_PtResolution_AK8PFPuppi.jr.txt'.format(jer_tag)))
            # print("File "+'correctionFiles/JER/{0}/{0}_SF_AK8PFPuppi.jersf.txt'.format(jer_tag)+" exists: ", os.path.exists('correctionFiles/JER/{0}/{0}_SF_AK8PFPuppi.jersf.txt'.format(jer_tag)))
            ext.add_weight_sets([
            '* * '+corr_path("JER", jer_tag, '{0}_PtResolution_{1}.jr.txt'.format(jer_tag, AK_str)),
            '* * '+corr_path("JER", jer_tag, '{0}_SF_{1}.jersf.txt'.format(jer_tag, AK_str))])
            # print("JER SF added")
    else:       
        #For data, make sure we don't duplicat
        tags_done = []
        for run, tag in jec_tag_data.items():
            if not (tag in tags_done):
                ext.add_weight_sets([
                '* * '+corr_path("JEC", tag, '{0}_L1FastJet_{1}.jec.txt'.format(tag, AK_str)),
                '* * '+corr_path("JEC", tag, '{0}_L2Relative_{1}.jec.txt'.format(tag, AK_str)),
                '* * '+corr_path("JEC", tag, '{0}_L3Absolute_{1}.jec.txt'.format(tag, AK_str)),
                '* * '+corr_path("JEC", tag, '{0}_L2L3Residual_{1}.jec.txt'.format(tag, AK_str)),
                ])
                #                 ext.add_weight_sets([
                # '* * '+'correctionFiles/JEC/{0}/{0}_L1FastJet_{1}.jec.txt'.format(tag, AK_str),
                # '* * '+'correctionFiles/JEC/{0}/{0}_L2Relative_{1}.jec.txt'.format(tag, AK_str),
                # '* * '+'correctionFiles/JEC/{0}/{0}_L3Absolute_{1}.jec.txt'.format(tag, AK_str),
                # '* * '+'correctionFiles/JEC/{0}/{0}_L2L3Residual_{1}.jec.txt'.format(tag, AK_str),
                # ])
                tags_done += [tag]
    ext.finalize()

    evaluator = ext.make_evaluator()

    if (not isData):
        jec_names = [
            '{0}_L1FastJet_{1}'.format(jec_tag, AK_str),
            '{0}_L2Relative_{1}'.format(jec_tag, AK_str),
            '{0}_L3Absolute_{1}'.format(jec_tag, AK_str)]
        #### if jes in arguments add total uncertainty values for comparison and easy plotting
        if 'jes' in uncertainty_sources:
            jec_names.extend(['{0}_Uncertainty_{1}'.format(jec_tag, AK_str)])
            uncertainty_sources.remove('jes')
        jec_names.extend(['{0}_UncertaintySources_{1}_{2}'.format(jec_tag, AK_str, unc_src) for unc_src in uncertainty_sources])

        if jer_tag: 
            jec_names.extend(['{0}_PtResolution_{1}'.format(jer_tag, AK_str),
                              '{0}_SF_{1}'.format(jer_tag, AK_str)])

    else:
        jec_names={}
        for run, tag in jec_tag_data.items():
            jec_names[run] = [
                '{0}_L1FastJet_{1}'.format(tag, AK_str),
                '{0}_L3Absolute_{1}'.format(tag, AK_str),
                '{0}_L2Relative_{1}'.format(tag, AK_str),
                '{0}_L2L3Residual_{1}'.format(tag, AK_str),]

    if not isData:
        jec_inputs = {name: evaluator[name] for name in jec_names}
    else:
        jec_inputs = {name: evaluator[name] for name in jec_names[era]}


    # print("jec_input", jec_inputs)
    jec_stack = JECStack(jec_inputs)
    if not isData:
        if mode == 'AK8':
            FatJets['pt_gen'] = ak.values_astype(ak.fill_none(FatJets.matched_gen.pt, 0), np.float32)
        if mode == 'AK4':        
            SubGenJetAK8 = events.SubGenJetAK8
            SubGenJetAK8['p4'] = ak.with_name(SubGenJetAK8[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
            FatJets["p4"] = ak.with_name(FatJets[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
            FatJets['pt_gen'] = ak.values_astype(ak.fill_none(FatJets.p4.nearest(SubGenJetAK8.p4, threshold=0.4).pt, 0), np.float32)
    if mode == 'AK4':          
        FatJets['area'] = ak.full_like( FatJets.pt, 0.503)
            
    FatJets['pt_raw'] = (1 - FatJets['rawFactor']) * FatJets['pt']
    FatJets['mass_raw'] = (1 - FatJets['rawFactor']) * FatJets['mass']
    # NB: do NOT name this 'rho' -- on a vector-mixin object that aliases .pt
    # to the rho value (see research-notes/bugs/jecstack_naming_issue.md).
    FatJets['jec_rho'] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, FatJets.pt)[0]

    name_map = jec_stack.blank_name_map
    # print("N events missing pt entry ", ak.sum(ak.num(FatJets.pt)<1))
    # print("N events w/ pt entry ", ak.sum(ak.num(FatJets.pt)>0))
    # print("Fatjet pt ", FatJets.pt)
    name_map['JetPt'] = 'pt'
    name_map['JetMass'] = 'mass'
    name_map['JetEta'] = 'eta'
    name_map['ptRaw'] = 'pt_raw'
    name_map['massRaw'] = 'mass_raw'
    name_map['JetA'] = 'area'
    name_map['ptGenJet'] = 'pt_gen'
    name_map['Rho'] = 'jec_rho'

    # coffea 2025+: CorrectedJetsFactory.build no longer accepts lazy_cache;
    # events also no longer expose a `.caches` attribute under the new NanoEvents.
    jet_factory = CorrectedJetsFactory(name_map, jec_stack)
    print("Fat jets for index 0 ", FatJets)
    corrected_jets = jet_factory.build(FatJets)
    # print("Available uncertainties: ", jet_factory.uncertainties())
    # print("Corrected jets object: ", corrected_jets.fields)
    return corrected_jets
def GetLHEWeight(events):
    from parton import mkPDF
    #### make sure to `pip3 install --user parton` in singularity shell
    # os.environ['LHAPDF_DATA_PATH'] = '/cvmfs/sft.cern.ch/lcg/releases/MCGenerators/lhapdf/6.2.1-7149a/x86_64-centos7-gcc8-opt/share/LHAPDF/'
    paths = ['/cvmfs/sft.cern.ch/lcg/external/lhapdfsets/current/']
    # pdfset = PDFSet('PDF4LHC21_mc_pdfas', pdfdir=path[0])
    # print("PDF set obj ", pdfset)
    # print("PDF set info ", pdfset.info)
    # pdfmembers = []
    print("Length of events ", len(events))
    with suppress_stdout_stderr():
        pdf = mkPDF('PDF4LHC21_mc', pdfdir=paths[0])
    # print("PDFSet info ", pdf.pdfset.info)
    # print("PDF grids ", pdf.pdfgrids)
    
    #### get pdf values from events
    q2 = events.Generator.scalePDF
    x1 = events.Generator.x1
    x2 = events.Generator.x2
    id1 = events.Generator.id1
    id2 = events.Generator.id2
    xfxq1 = pdf.xfxQ(id1, x1, q2, grid=False)
    xfxq2 = pdf.xfxQ(id2, x2, q2, grid=False)
    # print("x*f(x) = ", xfxq1, xfxq2)
    # print("Length of xfxq1 ", len(xfxq1))
    # print("type of pdf weight ", type(xfxq1))
    pdfNom = xfxq1*xfxq2
    pdfUp = pdf.xfxQ(id1, x1, 2*q2, grid=False)*pdf.xfxQ(id1, x1, 2*q2, grid=False)
    pdfDown = pdf.xfxQ(id1, x1, 0.5*q2, grid=False)*pdf.xfxQ(id1, x1, 0.5*q2, grid=False)
    #### Dont know what alphas do
        #     muRUp = self.lha.evalAlphas(2*q)**4
        # muRDown = self.lha.evalAlphas(0.5*q)**4
    return pdfNom, pdfUp, pdfDown

def GetQ2Weights(events):
# https://gitlab.cern.ch/gagarwal/ttbardileptonic/-/blob/master/corrections.py
    ## determines the envelope of the muR/muF up and down variations
    ## Case 1:
    ## LHEScaleWeight[0] -> (0.5, 0.5) # (muR, muF)
    ##               [1] -> (0.5, 1)
    ##               [2] -> (0.5, 2)
    ##               [3] -> (1, 0.5)
    ##               [4] -> (1, 1)
    ##               [5] -> (1, 2)
    ##               [6] -> (2, 0.5)
    ##               [7] -> (2, 1)
    ##               [8] -> (2, 2)
                  
    ## Case 2:
    ## LHEScaleWeight[0] -> (0.5, 0.5) # (muR, muF)
    ##               [1] -> (0.5, 1)
    ##               [2] -> (0.5, 2)
    ##               [3] -> (1, 0.5)
    ##               [4] -> (1, 2)
    ##               [5] -> (2, 0.5)
    ##               [6] -> (2, 1)
    ##               [7] -> (2, 2)

    q2Nom = np.ones(len(events))
    q2Up = np.ones(len(events))
    q2Down = np.ones(len(events))
    if ("LHEScaleWeight" in events.fields):
        if ak.all(ak.num(events.LHEScaleWeight, axis=1)==9):
            nom = events.LHEScaleWeight[:,4]
            scales = events.LHEScaleWeight[:,[0,1,3,5,7,8]]
            q2Up = ak.max(scales,axis=1)/nom
            q2Down = ak.min(scales,axis=1)/nom 
        elif ak.all(ak.num(events.LHEScaleWeight, axis=1)==8):
            scales = events.LHEScaleWeight[:,[0,1,3,4,6,7]]
            q2Up = ak.max(scales,axis=1)
            q2Down = ak.min(scales,axis=1)

    return q2Nom, q2Up, q2Down


def GetQ2muF(events):
    muF = np.ones(len(events))
    up = np.ones(len(events))
    down = np.ones(len(events))
    if ("LHEScaleWeight" in ak.fields(events)):
        if ak.all(ak.num(events.LHEScaleWeight, axis=1)==9):
            nom = events.LHEScaleWeight[:,4]
            up = events.LHEScaleWeight[:,5]/nom
            down = events.LHEScaleWeight[:,3]/nom
        elif ak.all(ak.num(events.LHEScaleWeight, axis=1)==8):
            up = events.LHEScaleWeight[:,4]
            down = events.LHEScaleWeight[:,3]
    return muF, up, down
    

def GetQ2muR(events):
    muR = np.ones(len(events))
    up = np.ones(len(events))
    down = np.ones(len(events))
    if ("LHEScaleWeight" in ak.fields(events)):
        if ak.all(ak.num(events.LHEScaleWeight, axis=1)==9):
            nom = events.LHEScaleWeight[:,7]
            up = events.LHEScaleWeight[:,5]/nom
            down = events.LHEScaleWeight[:,1]/nom
        elif ak.all(ak.num(events.LHEScaleWeight, axis=1)==8):
            up = events.LHEScaleWeight[:,6]
            down = events.LHEScaleWeight[:,1]
    return muR, up, down


def GetPDFWeights(events):
    
    # hessian pdf weights https://arxiv.org/pdf/1510.03865v1.pdf
    # https://github.com/nsmith-/boostedhiggs/blob/master/boostedhiggs/corrections.py#L60
    
    pdf_nom = np.ones(len(events))

    if "LHEPdfWeight" in events.fields:
        
        pdfUnc = ak.std(events.LHEPdfWeight,axis=1)/ak.mean(events.LHEPdfWeight,axis=1)
        pdfUnc = ak.fill_none(pdfUnc, 0.00)
        
        pdf_up = pdf_nom + pdfUnc
        pdf_down = pdf_nom - pdfUnc
        
        
#         arg = events.LHEPdfWeight[:, 1:-2] - np.ones((len(events), 100))
#         summed = ak.sum(np.square(arg), axis=1)
#         pdf_unc = np.sqrt((1. / 99.) * summed)
        
#         pdf_nom = np.ones(len(events))
#         pdf_up = pdf_nom + pdf_unc
#         pdf_down = np.ones(len(events))

    else:
        
        pdf_up = np.ones(len(events))
        pdf_down = np.ones(len(events))
        

    return pdf_nom, pdf_up, pdf_down



## --------------------------------- MET Filters ------------------------------#
## Reference: https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETOptionalFiltersRun2#2018_2017_data_and_MC_UL

MET_filters = {'2016APV':["goodVertices",
                          "globalSuperTightHalo2016Filter",
                          "HBHENoiseFilter",
                          "HBHENoiseIsoFilter",
                          "EcalDeadCellTriggerPrimitiveFilter",
                          "BadPFMuonFilter",
                          "BadPFMuonDzFilter",
                          "eeBadScFilter",
                          "hfNoisyHitsFilter"],
               '2016'   :["goodVertices",
                          "globalSuperTightHalo2016Filter",
                          "HBHENoiseFilter",
                          "HBHENoiseIsoFilter",
                          "EcalDeadCellTriggerPrimitiveFilter",
                          "BadPFMuonFilter",
                          "BadPFMuonDzFilter",
                          "eeBadScFilter",
                          "hfNoisyHitsFilter"],
               '2017'   :["goodVertices",
                          "globalSuperTightHalo2016Filter",
                          "HBHENoiseFilter",
                          "HBHENoiseIsoFilter",
                          "EcalDeadCellTriggerPrimitiveFilter",
                          "BadPFMuonFilter",
                          "BadPFMuonDzFilter",
                          "hfNoisyHitsFilter",
                          "eeBadScFilter",
                          "ecalBadCalibFilter"],
               '2018'   :["goodVertices",
                          "globalSuperTightHalo2016Filter",
                          "HBHENoiseFilter",
                          "HBHENoiseIsoFilter",
                          "EcalDeadCellTriggerPrimitiveFilter",
                          "BadPFMuonFilter",
                          "BadPFMuonDzFilter",
                          "hfNoisyHitsFilter",
                          "eeBadScFilter",
                          "ecalBadCalibFilter"]}

corrlib_namemap = {
    "2016APV":"2016preVFP_UL",
    "2016":"2016postVFP_UL",
    "2017":"2017_UL",
    "2018":"2018_UL"
}
