#### based on the trigger emulation method spelled out here https://cms.cern.ch/iCMS/jsp/db_notes/noteInfo.jsp?cmsnoteid=CMS%20AN-2015/154
import awkward as ak
import numpy as np
from coffea import processor, util
import hist
import pandas
import correctionlib
import sys
import os
from pathlib import Path
#### run.py stages python/ on dask workers so package-style imports resolve.
# sys.path.append(os.getcwd()+'/python/')
# from utils import getLumiMask
from python.utils import getLumiMask

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CORR = _PROJECT_ROOT / "correctionFiles"


def corr_path(*parts):
    return str(_CORR.joinpath(*parts))

class triggerProcessor(processor.ProcessorABC):
    def __init__(self, year = None, trigger = "", data = False):
        self.year = year
        self.trigger = trigger
        self.data = data
        dataset_cat = hist.axis.StrCategory([],growth=True,name="dataset", label="Dataset")
        HLT_cat = hist.axis.StrCategory([], growth=True, name="HLT_cat",label="")
        if data:
            pt_bin = hist.axis.Regular(200, 0, 2400.,name="pt", label="Jet pT (GeV)")
        else: pt_bin = hist.axis.Regular(200, 0, 3200.,name="pt", label="Jet pT (GeV)")
        self._histos = {
            'hist_trigEff': hist.Hist(dataset_cat, HLT_cat, pt_bin, storage="weight", name="Events"),
            'hist_trigEff_ptCut': hist.Hist(dataset_cat, HLT_cat, pt_bin, storage="weight", name="Events"),
            'hist_trigRef': hist.Hist(dataset_cat, HLT_cat, pt_bin, storage="weight", name="Events"),
            'hist_pt': hist.Hist(dataset_cat, pt_bin, storage="weight", name="Events"),
            'cutflow':      processor.defaultdict_accumulator(int),
            }
    
    @property
    def accumulator(self):
        return self._histos
    def process(self, events):
        out = self._histos
        #### choose events that have at least one fatjet
        events = events[ak.num(events.FatJet) >= 1]
        # print("Events metadata: ", events.metadata)
        
        #### Choose trigger objects that belong to jets (id == 1), after this selection remove events 
        #### that don't have corresponding jet trigger objects
        trigObj = events.TrigObj
        # print("Trigger object id:", trigObj.id)
        trigObj = trigObj[trigObj.id == 1]
        events = events[ak.num(trigObj, axis = -1) != 0]
        trigObj = trigObj[ak.num(trigObj, axis = -1) != 0]
        year = self.year
        trigger = self.trigger
        if self.data:
            datastring = "JetHT"
        else:
            datastring = "QCDsim"
        #### get HLT objects from events
        if year == '2016' or year == '2016APV':
            trigThresh = [40, 60, 80, 140, 200, 260, 320, 400, 450, 500]
            HLT_paths = [trigger + str(i) for i in trigThresh]
        elif year == '2017':
            trigThresh = [40, 60, 80, 140, 200, 260, 320, 400, 450, 500, 550]
            HLT_paths = [trigger + str(i) for i in trigThresh]
        elif year == '2018':
            trigThresh = [15, 25, 40, 60, 80, 140, 200, 260, 320, 400, 450, 500, 550]
            HLT_paths = [trigger + str(i) for i in trigThresh]
        else:
            print("Must provide a year to run trigger studies")
        
        #### Loop over HLT paths and calculate efficiencies for each
        efficiencies = {}
        for i in np.arange(len(HLT_paths)):
            path = HLT_paths[i]
            print("index: ", i, " HLT path: ", path)
            if (path in events.HLT.fields) and (i > 0):
                # print("Number of events w/ trigger ", ak.count_nonzero(events.HLT[HLT_paths[i]]))
                # print("HLT ref path: ", HLT_paths[i-1], " Number of Trues ", ak.count_nonzero(events.HLT[HLT_paths[i-1]]))
                #### choose jets belonging to ref trigger (i-1) and passing pt cut of trigger of trigger of interest (i)
                efficiencies[path] = events.FatJet[:,0].pt[(events.HLT[HLT_paths[i-1]]) & (trigObj.pt[:,0] > trigThresh[i]) & (trigObj.l1pt[:,0] >trigThresh[i-1])]
                #### trig eff
                out['hist_trigEff_ptCut'].fill(dataset = datastring + str(year), HLT_cat = path, pt = efficiencies[path])
                #### gives ratio of trigger paths (do not add to 1)
                out['hist_trigEff'].fill(dataset = datastring + str(year), HLT_cat = path, pt = events.FatJet[:,0].pt[(events.HLT[HLT_paths[i-1]]) & (events.HLT[HLT_paths[i]])])
                out['hist_trigRef'].fill(dataset = datastring +  str(year), HLT_cat = path, pt = events.FatJet[:,0].pt[(events.HLT[HLT_paths[i-1]])])
                out['hist_pt'].fill(dataset = datastring + str(year), pt = events.FatJet[:,0].pt)
        print("Final efficiencies:", efficiencies)
        if np.all([(len(eff)==0) for eff in efficiencies.values()]):
            print("No AK8PFJet triggers -- PFJet")
            print(events.metadata)
        else:
            print("Has AKPFJet triggers -- move to main JSON file")
            # print(events.metadata)
        return out
    
    def postprocess(self, accumulator):
        return accumulator
    
#### applyPrescales should only be run on data
class applyPrescales(processor.ProcessorABC):
    def __init__(self, trigger, year, turnOnPts, data = True):
        self.data = data
        self.trigger = trigger
        self.year = year
        self.turnOnPt = turnOnPts
        if data:
            pt_bin = hist.axis.Regular(1000, 0, 2400.,name="pt", label="Jet pT (GeV)")
        else: pt_bin = hist.axis.Regular(1000, 0, 3200.,name="pt", label="Jet pT (GeV)")
        dataset_cat = hist.axis.StrCategory([],growth=True,name="dataset", label="Dataset")
        HLT_cat = hist.axis.StrCategory([], growth=True, name="HLT_cat",label="")
        self._histos = {
            'hist_pt': hist.Hist(dataset_cat, HLT_cat, pt_bin, storage="weight", name="Events"),
            'hist_pt_byHLTpath': hist.Hist(dataset_cat, HLT_cat, pt_bin, storage="weight", name="Events"),
            'hist_pt_byHLTpath_wPS': hist.Hist(dataset_cat, HLT_cat, pt_bin, storage="weight", name="Events"),
            'hist_pt_wPS': hist.Hist(dataset_cat, HLT_cat, pt_bin, storage="weight", name="Events"),
            'cutflow':      processor.defaultdict_accumulator(int),
            }
    @property
    def accumulator(self):
        return self._histos
    def process(self, events):
        out = self._histos
        trigger = self.trigger
        turnOnPt = self.turnOnPt
        if self.data:
            datastring = "JetHT"
        else:
            datastring = "QCDsim"
        if self.year == '2016' or self.year == '2016APV':
            trigThresh = [40, 60, 80, 140, 200, 260, 320, 400, 450, 500]
            if trigger == "PFJet":
                pseval = correctionlib.CorrectionSet.from_file(corr_path("ps_weight_JSON_PFJet2016.json"))
            else:
                pseval = correctionlib.CorrectionSet.from_file(corr_path("ps_weight_JSON_2016.json"))
        elif self.year == '2017':
            trigThresh = [40, 60, 80, 140, 200, 260, 320, 400, 450, 500, 550]  
            if trigger == "PFJet":
                pseval = correctionlib.CorrectionSet.from_file(corr_path("ps_weight_JSON_PFJet"+self.year+".json"))
            else:
                pseval = correctionlib.CorrectionSet.from_file(corr_path("ps_weight_JSON_"+self.year+".json"))
        elif self.year == '2018':
            trigThresh = [15, 25, 40, 60, 80, 140, 200, 260, 320, 400, 450, 500, 550]
            if trigger == "PFJet":
                pseval = correctionlib.CorrectionSet.from_file(corr_path("ps_weight_JSON_PFJet"+self.year+".json"))
            else:
                pseval = correctionlib.CorrectionSet.from_file(corr_path("ps_weight_JSON_"+self.year+".json"))
        #### require at least one jet in each event
        HLT_paths = [trigger + str(i) for i in trigThresh]
        events = events[ak.num(events.FatJet) >= 1]                                           
        ### allRuns_AK8HLT.csv is the result csv of running 'brilcalc trg --prescale --hltpath "HLT_AK8PFJet*" --output-style                 csv' and is used to create the ps_weight_JSON files
        lumi_mask = getLumiMask(self.year)(events.run, events.luminosityBlock)
        # print("Length of events before lumi mask: ", len(events))
        events = events[lumi_mask]
        # print("Length of events after lumi mask: ", len(events))
        for i in np.arange(len(HLT_paths))[::-1]:
            path = HLT_paths[i]
            print("Index i: ", i, " for path: ", path)
            if path in events.HLT.fields:
                pt0 = events.FatJet[:,0].pt
                print(len(pt0), "leading pts: ", pt0, "for events", len(events))
                print("Events in path ", path, " ", len(events.HLT[path]))
                #### here we will use correctionlib to assign weights
                out['hist_pt'].fill(dataset = datastring, HLT_cat = path, pt = pt0[events.HLT[path]])
                out['hist_pt_wPS'].fill(dataset = datastring, HLT_cat = path, pt = pt0[events.HLT[path]], weight =                                                                      pseval['prescaleWeight'].evaluate(ak.to_numpy(events[events.HLT[path]].run), path, ak.to_numpy(ak.values_astype(events[events.HLT[path]].luminosityBlock, np.float32))))
                if i == (len(HLT_paths) - 1):
                    events_cut = events[((pt0 > turnOnPt[i]) & events.HLT[path])]
                    pt0 = events_cut.FatJet[:,0].pt
                    out['hist_pt_byHLTpath'].fill(dataset = datastring, HLT_cat = path, pt = pt0, )

                    out['hist_pt_byHLTpath_wPS'].fill(dataset = datastring, HLT_cat = path, pt = pt0, 
                                                      weight = pseval['prescaleWeight'].evaluate(ak.to_numpy(events_cut.run), path, ak.to_numpy(ak.values_astype(events_cut.luminosityBlock, np.float32))))
                else:
                    events_cut = events[((pt0 > turnOnPt[i]) & (pt0 <= turnOnPt[i+1]) & events.HLT[path])]
                    pt0 = events_cut.FatJet[:,0].pt
                    out['hist_pt_byHLTpath'].fill(dataset = datastring, HLT_cat = path, pt = pt0, )

                    out['hist_pt_byHLTpath_wPS'].fill(dataset = datastring, HLT_cat = path, pt = pt0, weight =                                                                         pseval['prescaleWeight'].evaluate(ak.to_numpy(events_cut.run), path, ak.to_numpy(ak.values_astype(events_cut.luminosityBlock, np.float32))))
                    print("# events passing pt cut ",  turnOnPt[i], ": ", len(pt0))
        return out
    def postprocess(self, accumulator):
        return accumulator
        
def main():
    #### Next run processor with futures executor on all test files
    from dask.distributed import Client
    from plugins import runCoffeaJob
    in_year = '2016'
    data_bool = False
    processor = triggerProcessor(year = in_year, trigger = 'AK8PFJet', data = data_bool)
    datastring = "JetHT" if processor.data == True else "QCDsim"
    filename = "datasets_UL_NANOAOD.json" if processor.data == True else "fileset_QCD.json"
    result = runCoffeaJob(processor, jsonFile = filename, casa = True, dask = True, testing = False, year = processor.year, data =     processor.data)
    util.save(result, 'coffeaOutput/triggerAssignment_{}_{}_{}_NewHist.coffea'.format(datastring, processor.year,                     processor.trigger))
    processor = applyPrescales(year = in_year, trigger = 'AK8PFJet', data = data_bool)
    dataname = "JetHT" if processor.data == True else "QCDsim"
    filename = "datasets_UL_NANOAOD.json" if processor.data == True else "fileset_QCD.json"
    result = runCoffeaJob(processor, jsonFile = filename, casa = True, dask = False, testing = True, year = processor.year, data =     processor.data)
    util.save(result, 'coffeaOutput/applyPrescales_{}_{}_{}_test_NewHist.coffea'.format(datastring, processor.year,                   processor.trigger))
    

if __name__ == '__main__':
    # Execute when the module is not initialized from an import statement: i.e. called from terminal command line
    main()
