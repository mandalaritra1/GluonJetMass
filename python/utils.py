from coffea.lumi_tools import LumiMask
import correctionlib
import awkward as ak
import numpy as np
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CORR = _PROJECT_ROOT / "correctionFiles"


def corr_path(*parts):
    return str(_CORR.joinpath(*parts))

turnOnPts_JetHT = {'2016': {'AK8PFJet40':0.,
                 'AK8PFJet60': 140., 
                 'AK8PFJet80': 210.,
                 'AK8PFJet140': 290., 
                 'AK8PFJet200': 380., 
                 'AK8PFJet260': 450., 
                 'AK8PFJet320': 550., 
                 'AK8PFJet400': 640., 
                 'AK8PFJet450': 690., 
                 'AK8PFJet500': 820.},
                '2016APV': {'AK8PFJet40':0., 
                     'AK8PFJet60': 140, 
                     'AK8PFJet80': 210., 
                     'AK8PFJet140': 290., 
                     'AK8PFJet200': 380., 
                     'AK8PFJet260': 450., 
                     'AK8PFJet320': 550., 
                     'AK8PFJet400': 640., 
                     'AK8PFJet450': 730., 
                     'AK8PFJet500': 820.},
                '2017': {'AK8PFJet40':0.,
                 'AK8PFJet60': 0., 
                 'AK8PFJet80': 160., 
                 'AK8PFJet140': 270., 
                 'AK8PFJet200': 310., 
                 'AK8PFJet260': 450., 
                 'AK8PFJet320': 560., 
                 'AK8PFJet400': 640., 
                 'AK8PFJet450': 700., 
                 'AK8PFJet500': 760., 
                 'AK8PFJet550': 810.},
                '2018': {'AK8PFJet15': 0.,
                 'AK8PFJet25': 0.,
                 'AK8PFJet40': 0.,
                 'AK8PFJet60': 0.,
                 'AK8PFJet80': 160., 
                 'AK8PFJet140': 270., 
                 'AK8PFJet200': 390., 
                 'AK8PFJet260': 470., 
                 'AK8PFJet320': 570., 
                 'AK8PFJet400': 650., 
                 'AK8PFJet450': 710., 
                 'AK8PFJet500': 760., 
                 'AK8PFJet550': 820.}
               }

turnOnPts_JetHT_old = {'2016': {'AK8PFJet40':0.,
                            'AK8PFJet60': 100., 
                            'AK8PFJet80': 200., 
                            'AK8PFJet140': 260., 
                            'AK8PFJet200': 350., 
                            'AK8PFJet260': 420., 
                            'AK8PFJet320': 530., 
                            'AK8PFJet400': 620., 
                            'AK8PFJet450': 690., 
                            'AK8PFJet500': 740.},
                   '2016APV': {'AK8PFJet40':0., 
                               'AK8PFJet60': 0, 
                               'AK8PFJet80': 220., 
                               'AK8PFJet140': 270., 
                               'AK8PFJet200': 360., 
                               'AK8PFJet260': 440., 
                               'AK8PFJet320': 530., 
                               'AK8PFJet400': 630., 
                               'AK8PFJet450': 690., 
                               'AK8PFJet500': 750.},
                   '2017' : {'AK8PFJet40':0.,
                             'AK8PFJet60': 0., 
                             'AK8PFJet80': 200., 
                             'AK8PFJet140': 260.,                 
                             'AK8PFJet200': 340., 
                             'AK8PFJet260': 420., 
                             'AK8PFJet320': 530., 
                             'AK8PFJet400': 620., 
                             'AK8PFJet450': 680., 
                             'AK8PFJet500': 745., 
                             'AK8PFJet550': 807.},
                   '2018' : {'AK8PFJet15': 0.,
                             'AK8PFJet25': 0.,
                             'AK8PFJet40': 0.,
                             'AK8PFJet60': 0.,
                             'AK8PFJet80': 0., 
                             'AK8PFJet140': 260., 
                             'AK8PFJet200': 360., 
                             'AK8PFJet260': 440., 
                             'AK8PFJet320': 550., 
                             'AK8PFJet400': 630., 
                             'AK8PFJet450': 700., 
                             'AK8PFJet500': 750., 
                             'AK8PFJet550': 810.}
            }

# in 
xsdb= { 'QCD_Pt_170to300_TuneCP5_13TeV_pythia8' : 104000.0,
        'QCD_Pt_300to470_TuneCP5_13TeV_pythia8' :  6806.0,
        'QCD_Pt_470to600_TuneCP5_13TeV_pythia8' :   552.0,
        'QCD_Pt_600to800_TuneCP5_13TeV_pythia8' :   154.6,
        'QCD_Pt_800to1000_TuneCP5_13TeV_pythia8' :   26.15,
        'QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8' :   0.03567,
        'QCD_Pt_1400to1800_TuneCP5_13TeV_pythia8' :   0.6419,
        'QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8' :   0.0877,
        'QCD_Pt_2400to3200_TuneCP5_13TeV_pythia8' :   0.005241,
        'QCD_Pt_3200toInf_TuneCP5_13TeV_pythia8' :    0.0001346,

        'QCD_HT100to200_TuneCH3_13TeV-madgraphMLM-herwig7': 23640000.0,
        'QCD_HT200to300_TuneCH3_13TeV-madgraphMLM-herwig7':  1546000.0,
        'QCD_HT300to500_TuneCH3_13TeV-madgraphMLM-herwig7':   321600.0,
        'QCD_HT500to700_TuneCH3_13TeV-madgraphMLM-herwig7':    30250.0,
        'QCD_HT700to1000_TuneCH3_13TeV-madgraphMLM-herwig7':    6364.0,
        'QCD_HT1000to1500_TuneCH3_13TeV-madgraphMLM-herwig7':   1117.0,
        'QCD_HT1500to2000_TuneCH3_13TeV-madgraphMLM-herwig7':    108.4,
        'QCD_HT2000toInf_TuneCH3_13TeV-madgraphMLM-herwig7':      22.36,
        
        # 'QCD_HT100to200_TuneCH3_13TeV-madgraphMLM-herwig7': 11370.0,
        # 'QCD_HT200to300_TuneCH3_13TeV-madgraphMLM-herwig7':   883.5,
        # 'QCD_HT300to500_TuneCH3_13TeV-madgraphMLM-herwig7':   259.6,
        # 'QCD_HT500to700_TuneCH3_13TeV-madgraphMLM-herwig7':    23.63,
        # 'QCD_HT700to1000_TuneCH3_13TeV-madgraphMLM-herwig7':    4.943,
        # 'QCD_HT1000to1500_TuneCH3_13TeV-madgraphMLM-herwig7':   0.8013,
        # 'QCD_HT1500to2000_TuneCH3_13TeV-madgraphMLM-herwig7':   0.06815,
        # 'QCD_HT2000toInf_TuneCH3_13TeV-madgraphMLM-herwig7':    0.01245,
       #### Values used to produce 2016/2017
       'QCD_HT100to200_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8':   23640000.0,
        'QCD_HT200to300_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8':   1546000.0,
        'QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8':    321600.0,
        'QCD_HT500to700_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8':      30250.0,
        'QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8':      6364.0,
        'QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8':     1117.0,
        'QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8':      108.4,
        'QCD_HT2000toInf_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8':        22.36,
       #### Values used to produce 2017/2018 mc
       'QCD_HT100to200_TuneCP5_13TeV-madgraphMLM-pythia8': 23700000.0,
       'QCD_HT200to300_TuneCP5_13TeV-madgraphMLM-pythia8' : 1552000.0,
       'QCD_HT300to500_TuneCP5_13TeV-madgraphMLM-pythia8' :  321100.0,
       'QCD_HT500to700_TuneCP5_13TeV-madgraphMLM-pythia8' :   30980.0,
       'QCD_HT700to1000_TuneCP5_13TeV-madgraphMLM-pythia8' :   6398.0,
       'QCD_HT1000to1500_TuneCP5_13TeV-madgraphMLM-pythia8' :  1122.0,
       'QCD_HT1500to2000_TuneCP5_13TeV-madgraphMLM-pythia8' :   109.4,
       'QCD_HT2000toInf_TuneCP5_13TeV-madgraphMLM-pythia8' :     21.74,
       
        'QCD_Pt-15to7000_TuneCH3_Flat_13TeV_herwig7': 1329000000.0,
        #### Backgrounds 
        'WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8': 53940.0,

       'ZJetsToNuNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8':271.3,
       'ZJetsToNuNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8': 72.69,
       'ZJetsToNuNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8': 9.961,
       'ZJetsToNuNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8': 2.425,
       'ZJetsToNuNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8': 1.076,
       'ZJetsToNuNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8': 0.2474,
       'ZJetsToNuNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8': 0.005609,

       'TTJets_TuneCP5_13TeV-madgraphMLM-pythia8': 471.7	,
       
 }

# in fb^-1 taken from https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVAnalysisSummaryTable
lumi = {'2018' : 59.74,
        '2017': 41.48,
        # '2016': 36.33 ####combined
        '2016APV':19.5,
        "2016":16.8
       }


#### Missing 2016APV XS's
sumw_qcd_mg = { '2016APV': {
        'QCD_HT200to300_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8': 101566726040.5,
        'QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8':  18372261702.0,
        'QCD_HT500to700_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8' :  2253983329.5,
        'QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8':   409397088.68359375,
        'QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8' :  20934451.69921875,
        'QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8':    1983266.7685546875,
        'QCD_HT2000toInf_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8':      196907.1982421875, 
},
    '2016': {
        'QCD_HT200to300_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8': 70210797200.0,
        'QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8': 14630450930.0,
        'QCD_HT500to700_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8' : 2254432227.65625,
        'QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8':  158406238.25,
        'QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8' :  8486996.25390625,
        'QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8':   1361777.333267212,
        'QCD_HT2000toInf_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8':      67843.4627532959,
    },
           '2017': {
               'QCD_HT200to300_TuneCP5_13TeV-madgraphMLM-pythia8' : 181026146494.0,
               'QCD_HT300to500_TuneCP5_13TeV-madgraphMLM-pythia8' :  36516294423.5,
               'QCD_HT500to700_TuneCP5_13TeV-madgraphMLM-pythia8' :   3486166624.203125,
               'QCD_HT700to1000_TuneCP5_13TeV-madgraphMLM-pythia8' :   563102688.9375,
               'QCD_HT1000to1500_TuneCP5_13TeV-madgraphMLM-pythia8' :   32239843.666503906,
               'QCD_HT1500to2000_TuneCP5_13TeV-madgraphMLM-pythia8' :    2451711.3540649414,
               'QCD_HT2000toInf_TuneCP5_13TeV-madgraphMLM-pythia8' :      248156.31952667236,
},
           '2018': {
               'QCD_HT200to300_TuneCP5_13TeV-madgraphMLM-pythia8' : 181673668336.0,
               'QCD_HT300to500_TuneCP5_13TeV-madgraphMLM-pythia8' :  35869948198.75,
               'QCD_HT500to700_TuneCP5_13TeV-madgraphMLM-pythia8' :   3529868167.8984375,
               'QCD_HT700to1000_TuneCP5_13TeV-madgraphMLM-pythia8' :  575521898.7402344,
               'QCD_HT1000to1500_TuneCP5_13TeV-madgraphMLM-pythia8' :   33323673.349121094,
               'QCD_HT1500to2000_TuneCP5_13TeV-madgraphMLM-pythia8' :    2520025.5220947266,
               'QCD_HT2000toInf_TuneCP5_13TeV-madgraphMLM-pythia8' :      260872.52236175537,
},
           }
 
num_gen_herwig = { "2016APV": {'QCD_HT100to200_TuneCH3_13TeV-madgraphMLM-herwig7': 3005498,
                               'QCD_HT200to300_TuneCH3_13TeV-madgraphMLM-herwig7': 1967563,
                               'QCD_HT300to500_TuneCH3_13TeV-madgraphMLM-herwig7': 1000485,
                               'QCD_HT500to700_TuneCH3_13TeV-madgraphMLM-herwig7': 496454,
                               'QCD_HT700to1000_TuneCH3_13TeV-madgraphMLM-herwig7':  488128,
                               'QCD_HT1000to1500_TuneCH3_13TeV-madgraphMLM-herwig7': 506141,
                               'QCD_HT1500to2000_TuneCH3_13TeV-madgraphMLM-herwig7': 492965,
                               'QCD_HT2000toInf_TuneCH3_13TeV-madgraphMLM-herwig7': 496469,
                              },
                  "2016": {'QCD_HT100to200_TuneCH3_13TeV-madgraphMLM-herwig7': 2942671,
                               'QCD_HT200to300_TuneCH3_13TeV-madgraphMLM-herwig7': 2006720,
                               'QCD_HT300to500_TuneCH3_13TeV-madgraphMLM-herwig7': 1006059,
                               'QCD_HT500to700_TuneCH3_13TeV-madgraphMLM-herwig7': 511149,
                               'QCD_HT700to1000_TuneCH3_13TeV-madgraphMLM-herwig7':  499379,
                               'QCD_HT1000to1500_TuneCH3_13TeV-madgraphMLM-herwig7': 493891,
                               'QCD_HT1500to2000_TuneCH3_13TeV-madgraphMLM-herwig7': 492975,
                               'QCD_HT2000toInf_TuneCH3_13TeV-madgraphMLM-herwig7':  488875,
                              },
                  "2017": {'QCD_HT100to200_TuneCH3_13TeV-madgraphMLM-herwig7': 2889515,
                               'QCD_HT200to300_TuneCH3_13TeV-madgraphMLM-herwig7': 1992969,
                               'QCD_HT300to500_TuneCH3_13TeV-madgraphMLM-herwig7': 1003410,
                               'QCD_HT500to700_TuneCH3_13TeV-madgraphMLM-herwig7': 516397,
                               'QCD_HT700to1000_TuneCH3_13TeV-madgraphMLM-herwig7':  501738,
                               'QCD_HT1000to1500_TuneCH3_13TeV-madgraphMLM-herwig7': 499775,
                               'QCD_HT1500to2000_TuneCH3_13TeV-madgraphMLM-herwig7': 507301,
                               'QCD_HT2000toInf_TuneCH3_13TeV-madgraphMLM-herwig7': 497985,
                              },
                  "2018": {'QCD_HT100to200_TuneCH3_13TeV-madgraphMLM-herwig7': 2934193,
                               'QCD_HT200to300_TuneCH3_13TeV-madgraphMLM-herwig7': 1990195,
                               'QCD_HT300to500_TuneCH3_13TeV-madgraphMLM-herwig7': 999343,
                               'QCD_HT500to700_TuneCH3_13TeV-madgraphMLM-herwig7': 497722,
                               'QCD_HT700to1000_TuneCH3_13TeV-madgraphMLM-herwig7':  495400,
                               'QCD_HT1000to1500_TuneCH3_13TeV-madgraphMLM-herwig7': 506420,
                               'QCD_HT1500to2000_TuneCH3_13TeV-madgraphMLM-herwig7': 507791,
                               'QCD_HT2000toInf_TuneCH3_13TeV-madgraphMLM-herwig7': 492537,
                              },
}

sumw_herwig = { "2016APV": {'QCD_HT100to200_TuneCH3_13TeV-madgraphMLM-herwig7':  141536808960.0,
                               'QCD_HT200to300_TuneCH3_13TeV-madgraphMLM-herwig7': 8881636224.0,
                               'QCD_HT300to500_TuneCH3_13TeV-madgraphMLM-herwig7': 1006554600.0,
                               'QCD_HT500to700_TuneCH3_13TeV-madgraphMLM-herwig7':   50850553.5,
                               'QCD_HT700to1000_TuneCH3_13TeV-madgraphMLM-herwig7':  11126359.375,
                               'QCD_HT1000to1500_TuneCH3_13TeV-madgraphMLM-herwig7':  1957313.9375,
                               'QCD_HT1500to2000_TuneCH3_13TeV-madgraphMLM-herwig7':   180116.1953125,
                               'QCD_HT2000toInf_TuneCH3_13TeV-madgraphMLM-herwig7':     36778.2041015625,
                              },
                  "2016": {'QCD_HT100to200_TuneCH3_13TeV-madgraphMLM-herwig7':   144014468096.0,
                               'QCD_HT200to300_TuneCH3_13TeV-madgraphMLM-herwig7': 8579037856.0,
                               'QCD_HT300to500_TuneCH3_13TeV-madgraphMLM-herwig7': 1083017536.0,
                               'QCD_HT500to700_TuneCH3_13TeV-madgraphMLM-herwig7':   56391885.375,
                               'QCD_HT700to1000_TuneCH3_13TeV-madgraphMLM-herwig7':  11584725.5,
                               'QCD_HT1000to1500_TuneCH3_13TeV-madgraphMLM-herwig7':  2099561.78125,
                               'QCD_HT1500to2000_TuneCH3_13TeV-madgraphMLM-herwig7':   181317.447265625,
                               'QCD_HT2000toInf_TuneCH3_13TeV-madgraphMLM-herwig7':     34087.36328125,
                              },
                  "2017": {'QCD_HT100to200_TuneCH3_13TeV-madgraphMLM-herwig7':   120172825088.0,
                               'QCD_HT200to300_TuneCH3_13TeV-madgraphMLM-herwig7': 5956648160.0,
                               'QCD_HT300to500_TuneCH3_13TeV-madgraphMLM-herwig7':  906730648.0,
                               'QCD_HT500to700_TuneCH3_13TeV-madgraphMLM-herwig7':   44980886.0,
                               'QCD_HT700to1000_TuneCH3_13TeV-madgraphMLM-herwig7':   9735645.125,
                               'QCD_HT1000to1500_TuneCH3_13TeV-madgraphMLM-herwig7':  1673187.5625,
                               'QCD_HT1500to2000_TuneCH3_13TeV-madgraphMLM-herwig7':   155704.818359375,
                               'QCD_HT2000toInf_TuneCH3_13TeV-madgraphMLM-herwig7':     29484.88671875,
                              },
                  "2018": {'QCD_HT100to200_TuneCH3_13TeV-madgraphMLM-herwig7':   117946849024.0,
                               'QCD_HT200to300_TuneCH3_13TeV-madgraphMLM-herwig7': 7728662208.0,
                               'QCD_HT300to500_TuneCH3_13TeV-madgraphMLM-herwig7':  754584448.0,
                               'QCD_HT500to700_TuneCH3_13TeV-madgraphMLM-herwig7':   37442241.0,
                               'QCD_HT700to1000_TuneCH3_13TeV-madgraphMLM-herwig7':   9712919.75,
                               'QCD_HT1000to1500_TuneCH3_13TeV-madgraphMLM-herwig7':  1673205.375,
                               'QCD_HT1500to2000_TuneCH3_13TeV-madgraphMLM-herwig7':   155008.26953125,
                               'QCD_HT2000toInf_TuneCH3_13TeV-madgraphMLM-herwig7':     29513.98876953125,
                              }, }

num_gen_herwig_flat = { 
    '2016': {'QCD_Pt-15to7000_TuneCH3_Flat_13TeV_herwig7' : 53923986},
    '2016APV': {'QCD_Pt-15to7000_TuneCH3_Flat_13TeV_herwig7': 45952213},
    '2017': {'QCD_Pt-15to7000_TuneCH3_Flat_13TeV_herwig7' : 99773000},
    '2018': {'QCD_Pt-15to7000_TuneCH3_Flat_13TeV_herwig7': 99775000}
}

num_gen_pythia = { '2016APV': {'QCD_Pt_170to300_TuneCP5_13TeV_pythia8' : 27885000,
'QCD_Pt_300to470_TuneCP5_13TeV_pythia8' : 54028000,
'QCD_Pt_470to600_TuneCP5_13TeV_pythia8' : 50782000,
'QCD_Pt_600to800_TuneCP5_13TeV_pythia8' : 61904000,
'QCD_Pt_800to1000_TuneCP5_13TeV_pythia8' : 35459000,
'QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8' : 19077000,
'QCD_Pt_1400to1800_TuneCP5_13TeV_pythia8' : 11000000,
'QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8' : 5262000,
'QCD_Pt_2400to3200_TuneCP5_13TeV_pythia8' : 2999000,
'QCD_Pt_3200toInf_TuneCP5_13TeV_pythia8' : 1000000},
    '2016': {'QCD_Pt_170to300_TuneCP5_13TeV_pythia8' : 29758000,
'QCD_Pt_300to470_TuneCP5_13TeV_pythia8' : 55264000,
'QCD_Pt_470to600_TuneCP5_13TeV_pythia8' : 52408000,
'QCD_Pt_600to800_TuneCP5_13TeV_pythia8' : 64584000,
'QCD_Pt_800to1000_TuneCP5_13TeV_pythia8' : 37698000,
'QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8' : 19892000,
'QCD_Pt_1400to1800_TuneCP5_13TeV_pythia8' : 10722000,
'QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8' : 5236000,
'QCD_Pt_2400to3200_TuneCP5_13TeV_pythia8' : 2848000,
'QCD_Pt_3200toInf_TuneCP5_13TeV_pythia8' : 996000},
           '2017': {'QCD_Pt_170to300_TuneCP5_13TeV_pythia8' : 29491000,
'QCD_Pt_300to470_TuneCP5_13TeV_pythia8' : 55358000,
'QCD_Pt_470to600_TuneCP5_13TeV_pythia8' : 50475000,
'QCD_Pt_600to800_TuneCP5_13TeV_pythia8' : 66419000,
'QCD_Pt_800to1000_TuneCP5_13TeV_pythia8' : 36890000,
'QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8' : 19461000,
'QCD_Pt_1400to1800_TuneCP5_13TeV_pythia8' : 10994000,
'QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8' : 5168000,
'QCD_Pt_2400to3200_TuneCP5_13TeV_pythia8' : 2997000,
'QCD_Pt_3200toInf_TuneCP5_13TeV_pythia8' : 1000000},
           '2018': {'QCD_Pt_170to300_TuneCP5_13TeV_pythia8' : 29478000,
'QCD_Pt_300to470_TuneCP5_13TeV_pythia8' : 57868000,
'QCD_Pt_470to600_TuneCP5_13TeV_pythia8' : 52448000,
'QCD_Pt_600to800_TuneCP5_13TeV_pythia8' : 66914000,
'QCD_Pt_800to1000_TuneCP5_13TeV_pythia8' : 36830000,
'QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8' : 19664000,
'QCD_Pt_1400to1800_TuneCP5_13TeV_pythia8' : 10982000,
'QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8' : 5491000,
'QCD_Pt_2400to3200_TuneCP5_13TeV_pythia8' : 2931000,
'QCD_Pt_3200toInf_TuneCP5_13TeV_pythia8' : 1000000}
}

num_gen_bg = {
    '2016APV': {
        'ZJetsToNuNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8': 7715405,
       'ZJetsToNuNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8':  7531529,
       'ZJetsToNuNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8':  6770574,
       'ZJetsToNuNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8':  2030858,
       'ZJetsToNuNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8':  703970,
       'ZJetsToNuNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8': 136393,
       'ZJetsToNuNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8':  111838,
    'TTJets_TuneCP5_13TeV-madgraphMLM-pythia8': 5047017 ,
    },
    '2016': {
        'ZJetsToNuNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8': 7083216,
       'ZJetsToNuNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8':  6814106,
       'ZJetsToNuNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8':  6114046,
       'ZJetsToNuNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8':  1881671,
       'ZJetsToNuNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8':  633500,
       'ZJetsToNuNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8': 115609,
       'ZJetsToNuNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8':  110461,
    'TTJets_TuneCP5_13TeV-madgraphMLM-pythia8': 5068919,
    },
    '2017': {
        'ZJetsToNuNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8': 18948271,
       'ZJetsToNuNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8':  17189820,
       'ZJetsToNuNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8':  13963690,
       'ZJetsToNuNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8':   4418971,
       'ZJetsToNuNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8':  1513585,
       'ZJetsToNuNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8':  267125,
       'ZJetsToNuNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8':   176201,
     'TTJets_TuneCP5_13TeV-madgraphMLM-pythia8': 10052151,
    },
    '2018': {
        'ZJetsToNuNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8': 28876062,
       'ZJetsToNuNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8':  22749608,
       'ZJetsToNuNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8':  19676607,
       'ZJetsToNuNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8':   5968910,
       'ZJetsToNuNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8':  2129122,
       'ZJetsToNuNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8':  381695,
       'ZJetsToNuNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8':   268224,
        'WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8':    79645994,
    },
    
}

sumw_bg = {
    '2016APV': {
        'ZJetsToNuNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8': 7715405,
       'ZJetsToNuNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8':  7531529,
       'ZJetsToNuNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8':  6770574,
       'ZJetsToNuNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8':  2030858,
       'ZJetsToNuNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8':  703970,
       'ZJetsToNuNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8': 136393,
       'ZJetsToNuNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8':  111838,
     'TTJets_TuneCP5_13TeV-madgraphMLM-pythia8': 9238796.376953125},
    '2016': {
        'ZJetsToNuNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8': 7083216,
       'ZJetsToNuNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8':  6814106,
       'ZJetsToNuNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8':  6114046,
       'ZJetsToNuNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8':  1881671,
       'ZJetsToNuNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8':  633500,
       'ZJetsToNuNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8': 115609,
       'ZJetsToNuNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8':  110461,
 'TTJets_TuneCP5_13TeV-madgraphMLM-pythia8': 10952247.5546875},
    '2017': {
        'ZJetsToNuNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8': 18948271,
       'ZJetsToNuNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8':  17189820,
       'ZJetsToNuNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8':  13963690,
       'ZJetsToNuNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8':   4418971,
       'ZJetsToNuNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8':  1513585,
       'ZJetsToNuNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8':  267125,
       'ZJetsToNuNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8':   176201,
        'WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8': 9566440293248.0,
   },
    '2018': {
        'ZJetsToNuNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8': 6299042.111694336,
       'ZJetsToNuNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8':  1676015.0802001953,
       'ZJetsToNuNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8':   233325.02221679688,
       'ZJetsToNuNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8':    13479.285751342773,
       'ZJetsToNuNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8':    2724.645004272461,
       'ZJetsToNuNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8':    136.23207068443298,
       'ZJetsToNuNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8':       2.5148895382881165,
        'TTJets_TuneCP5_13TeV-madgraphMLM-pythia8': 9437788.830200195,
        'WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8':    9537355436664.0,
    },
    
}

def getRapidity(p4):
    return 0.5 * np.log(( p4.energy + p4.pz ) / ( p4.energy - p4.pz ))
    
def getXSweight(dataset, IOV):
    print("Dataset: ", dataset)
    for year in np.array(list(lumi.keys())):
        if year in IOV:
            lum = lumi[year]
            print("Lumi ", lum, " for year ", year)
            for process in np.array(list(xsdb.keys())):
                if process in dataset:
                    xs = xsdb[process]
                    if 'herwig' in process:
                        if "madgraphMLM" in process:
                            print("Number of gen events for ", year, " ", process, ": ", sumw_herwig[year][process])
                            weight = xs * lum * 1000 / sumw_herwig[year][process]
                        else:
                            weight = xs * lum * 1000 / num_gen_herwig_flat[year][process]
                    else:
                        if "QCD" in process and process in sumw_qcd_mg[year].keys():
                            print("Sumw of events for ", year, " ", process, ": ", sumw_qcd_mg[year][process])
                            weight = xs * lum * 1000 / sumw_qcd_mg[year][process]
                        elif process in sumw_bg[year].keys():
                            print("Number of gen events for ", year, " ", process, ": ", sumw_bg[year][process])
                            weight = xs * lum * 1000 / sumw_bg[year][process]
                        else:
                            print("Don't have xs + sumw values for process", process)
                            weight = 1.
                    return weight


def getLumiMask(year):

    files = { '2016APV': corr_path("Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"),
              '2016': corr_path("Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"),
              '2017': corr_path("Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt"),
              '2018': corr_path("Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt")
            }

    mask = LumiMask(files[year])

    return mask

def get_gen_sd_mass_jet( jet, subjets):
    combs = ak.cartesian( (jet, subjets), axis=1 )
    dr_jet_subjets = combs['0'].delta_r(combs['1'])
    combs = combs[dr_jet_subjets < 0.8]
    total = combs['1'].sum(axis=1)
    return total 

def get_dphi( jet0, jet1 ):
    '''
    Find dphi between two jets, returning none when the event does not have at least two jets
    '''
    combs = ak.cartesian( (jet0, jet1), axis=1 )
    dphi = np.abs(combs['0'].delta_phi(combs['1']))
    return ak.firsts(dphi)

def update(events, collections):
    # https://github.com/nsmith-/boostedhiggs/blob/master/boostedhiggs/hbbprocessor.py
    """Return a shallow copy of events array with some collections swapped out"""
    out = events
#     logger.debug('update:%s:%s', time.time(), collections)
    
    for name, value in collections.items():
        out = ak.with_field(out, value, name)

    return out

### function to get apply prescale weights to 
def applyPrescales(events, year, trigger = "AK8PFJet", turnOnPts = turnOnPts_JetHT, data = True):
    print("Trigger year ", year)
    if year == '2016' or year == '2016APV':
        trigThresh = [40, 60, 80, 140, 200, 260, 320, 400, 450, 500]
        if trigger == "PFJet":
            pseval = correctionlib.CorrectionSet.from_file(corr_path("ps_weight_JSON_PFJet2016.json"))
        else:
            pseval = correctionlib.CorrectionSet.from_file(corr_path("ps_weight_JSON_2016.json"))
    elif year == '2017':
        trigThresh = [40, 60, 80, 140, 200, 260, 320, 400, 450, 500, 550]  
        if trigger == "PFJet":
            pseval = correctionlib.CorrectionSet.from_file(corr_path("ps_weight_JSON_PFJet"+year+".json"))
        else:
            pseval = correctionlib.CorrectionSet.from_file(corr_path("ps_weight_JSON_"+year+".json"))
    elif year == '2018':
        trigThresh = [15, 25, 40, 60, 80, 140, 200, 260, 320, 400, 450, 500, 550]
        if trigger == "PFJet":
            pseval = correctionlib.CorrectionSet.from_file(corr_path("ps_weight_JSON_PFJet"+year+".json"))
        else:
            pseval = correctionlib.CorrectionSet.from_file(corr_path("ps_weight_JSON_"+year+".json"))
    turnOnPts = np.array(list(turnOnPts[year].values()))
    HLT_paths = [trigger + str(i) for i in trigThresh]
    events_mask = np.full(len(events), False)
    weights = np.ones(len(events))
    HLT_cutflow_initial = {}
    HLT_cutflow_final = {}
            
    #### allRuns_AK8HLT.csv is the result csv of running 'brilcalc trg --prescale --hltpath "HLT_AK8PFJet*" --output-style                 csv' and is used to create the ps_weight_JSON files
    #### lumimask and requirement of one jet is already applied in jet processor

    for i in np.arange(len(HLT_paths))[::-1]:
        path = HLT_paths[i]
        if path in events.HLT.fields:
            print("events with HLT path ", HLT_paths[0], " ", events.HLT[path], " with sum ", ak.sum(events.HLT[path]))                  #### booking iniital HLT values
            HLT_cutflow_initial[path] = ak.sum(events.HLT[path])
            pt0 = ak.firsts(events.FatJet[:,0:].pt)
            psweights = pseval['prescaleWeight'].evaluate(ak.to_numpy(events.run), path, ak.to_numpy(ak.values_astype(events.luminosityBlock, np.float32)))
            #### here we will use correctionlib to assign weights
            if (i == (len(HLT_paths) - 1)):
                # events_cut = events[((pt0 > turnOnPts[i]) & events.HLT[path])]
                events_mask = np.where(((pt0 > turnOnPts[i]) & events.HLT[path]), True, events_mask)
#                 print("Number of ", path, "'s trues: ", sum(((pt0 > turnOnPts[i]) & events.HLT[path])), " number of total trues ", sum(events_mask))
                weights = np.where(((pt0 > turnOnPts[i]) & events.HLT[path]), psweights, weights)
                n_pass = ak.sum((pt0 > turnOnPts[i]) & events.HLT[path])
            else:
                # events_cut = events[((pt0 > turnOnPts[i]) & (pt0 <= turnOnPts[i+1]) & events.HLT[path])]
                events_mask = np.where(((pt0 > turnOnPts[i]) & (pt0 <= turnOnPts[i+1]) & events.HLT[path]), True, events_mask)
#                 print("Number of ", path, "'s path's trues: ", sum(((pt0 > turnOnPts[i]) & (pt0 <= turnOnPts[i+1]) & events.HLT[path])), " number of total trues ", sum(events_mask))
                weights = np.where(((pt0 > turnOnPts[i]) & (pt0 <= turnOnPts[i+1])), psweights, weights)
                n_pass = ak.sum((pt0 > turnOnPts[i]) & (pt0 <= turnOnPts[i+1]))
            HLT_cutflow_final[path] = n_pass
    return events_mask, weights, HLT_cutflow_initial, HLT_cutflow_final
