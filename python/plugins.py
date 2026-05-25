### This file houses helper functions for running processors
import pandas as pd
import time
from coffea import processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import pickle

#if using LPC dask or running locally use 'root://cmsxrootd.fnal.gov/'
#is using coffea casa use 'root://xcache/'

import os
import warnings
warnings.filterwarnings("ignore")
from distributed.diagnostics.plugin import UploadDirectory

def checkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def get_all_warnings( client ):
    logs = client.get_worker_logs()
    workers = list(logs.keys())
    for worker in workers:
        for log in logs[worker]:
            print("Log ", log)
            # if log[0] == 'WARNING' or log[0] == 'ERROR':
            #     print ()
            #     print (" ### Found warning for worker:", worker)
            #     print (log[1])

#reads in files and adds redirector, can specify year, default is all years
def handleData(jsonFile, redirector, year = '', testing = True, data = False, chunks = None):
    eras_data = {'2016':   '-UL2016',
                 '2016APV':'HIPM_UL2016', 
                 '2017':   'UL2017',
                 '2018':   'UL2018'
                    }
    eras_mc = {'2016APV':'UL16NanoAODAPV', 
               '2016':'UL16NanoAODv9', 
               '2017':'UL17NanoAODv9', 
               '2018':'UL18NanoAODv9'
              }
    dict = {}
    qualifiers = []
    if data:
        inputs = 'JetHT_data'
        if year == '2016' or year == '2016APV' or year == '2018' or year == '2017':
            qualifiers.append(eras_data[year])
        elif year == 2016 or year == 2018 or year == 2017:
            qualifiers.append(eras_data[str(year)]) 
        else:
            for era in list(eras_data.values()):
                print("Era: ", era)
                qualifiers.append(era)
    else:
        if "Wjet" in jsonFile:
            inputs = 'WJets'

        elif "Zjet" in jsonFile:
            inputs = 'ZJets_binned'
        elif "TTbar" in jsonFile:
            inputs = 'TTJets'
                        
        else:
            inputs = 'QCD_binned'
        if year == '2016' or year == '2017' or year == '2018' or year == '2016APV':
            qualifiers.append(eras_mc[year])
        elif year == 2016 or year == 2017 or year == 2018:
            qualifiers.append(eras_mc[str(year)])
        else:
            for era in list(eras_mc.values()):
                print("Era: ", era)
                qualifiers.append(era)
    df = pd.read_json(jsonFile) 
    for key in df[inputs].keys():
        for qualifier in qualifiers:
            if qualifier in key:
                if testing:
                    dict[key] = [redirector +  df[inputs][key][0]]
                else:
                    dict[key] = [redirector + df[inputs][key][i] for i in range(len(df[inputs][key]))]
    return dict

#initiate dask client and run coffea job
from dask.distributed import Client

def runCoffeaJob(processor_inst, jsonFile, dask = False, casa = False, testing = False, year = '', data = False, winterfell = False, verbose = True, datasetRange = None):
    #default is to run locally
    tstart = time.time()
    if casa:
        redirector = 'root://xcache/'
    elif winterfell:
        #### only data (not mc/sim) is on winterfell atm -- make sure using data json file and arguments
        redirector = '/mnt/data/cms'
    # elif data and not winterfell and not casa:
    #     redirector = 'root://cmsxrootd.fnal.gov/'
        #### default redirector
    else:
        redirector = ''
        # redirector = 'root://cmsxrootd.fnal.gov/'
        #### Nebraska redirector
        # redirector= 'root://xrootd-local.unl.edu/'
        #### MIT redirector
        # redirecotr='root://xrootd.cmsaf.mit.edu:1094/'
        #### DESY T2 redirector
        # redirector = 'root://dcache-cms-webdav-wan.desy.de/'
        #### CERN T2 redirector
        # redirector = 'root://eoscms.cern.ch:443/'
        # redirector='root://cmseos.fnal.gov:1094/'
        # redirector='root://cmseos.fnal.gov/'
        # redirector='root://cmsxrootd.hep.wisc.edu/'
    exe_args = {"schema": NanoAODSchema, 'skipbadfiles': False,}
    samples = handleData(jsonFile, redirector, year = year, testing = testing, data = data)
    if datasetRange!=None:
        print("Total avail datasets ", len(samples.keys()))
        samples = {key:samples[key] for key in list(samples.keys())[datasetRange[0]:datasetRange[1]]}
    #single files for testing
    # samples={'/QCD_HT200to300_TuneCH3_13TeV-madgraphMLM-herwig7/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM': ['root://cmsxrootd.fnal.gov//store/mc/RunIISummer20UL16NanoAODAPVv9/QCD_HT200to300_TuneCH3_13TeV-madgraphMLM-herwig7/NANOAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v1/60000/4503E31F-B62F-D446-82E4-288C0406F80F.root']}

    print("Running over datasets ", samples.keys())
    client = None
    cluster = None
    if casa and dask:
        print("Running on coffea casa")
        from coffea_casa import CoffeaCasaCluster
        cluster = CoffeaCasaCluster(job_extra = {'transfer_input_files':[ "correctionFiles", "python"] }, memory="10 GiB")
        cluster.adapt(minimum=4, maximum=70)
        client = Client(cluster)
        # client = Client("tls://lauren-2emeryl-2ehay-40cern-2ech.dask.cmsaf-prod.flatiron.hollandhpc.org:8786")
        # client.register_worker_plugin(UploadDirectory("/home/cms-jovyan/GluonJetMass", restart=True, update_path=True), nanny=True)
        # client.upload_file("python/plugins.py")
        # client.upload_file("python/utils.py")
        # client.upload_file("python/corrections.py")
        # client.upload_file("python/trijetProcessor.py") #upload additional files to the client                               
        # client.upload_file("python/dijetProcessor.py")
        # client.upload_file("python/triggerProcessor.py")
        # client.upload_file("correctionFiles/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt")
        # client.upload_file("correctionFiles/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt")
        # client.upload_file("correctionFiles/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt")
        # client.upload_file("correctionFiles/ps_weight_JSON_2016.json")
        # client.upload_file("correctionFiles/ps_weight_JSON_2017.json")
        # client.upload_file("correctionFiles/ps_weight_JSON_2018.json")
        # client.upload_file("correctionFiles/ps_weight_JSON_PFJet2016.json")
        # client.upload_file("correctionFiles/ps_weight_JSON_PFJet2017.json")
        # client.upload_file("correctionFiles/ps_weight_JSON_PFJet2018.json")
        # cluster = CoffeaCasaCluster(cores=11, memory="20 GiB", death_timeout = 60)
        # cluster.adapt(minimum=2, maximum=14)
        # client = Client(cluster)
        print("Client ", client)
        # coffea 2025+: run_uproot_job is removed; use processor.Runner with a
        # DaskExecutor instead. align_clusters is now a Runner kwarg.
        run_instance = processor.Runner(
            executor=processor.DaskExecutor(client=client, status=False, retries=10, treereduction=4),
            schema=NanoAODSchema,
            skipbadfiles=False,
            align_clusters=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_instance(samples, processor_inst, treename="Events")
    elif casa == False and dask:
        print("Running on LPC Condor")
        from lpcjobqueue import LPCCondorCluster
        #### make list of files and directories to upload to dask
        upload_to_dask = ['correctionFiles', 'python']
        cluster = LPCCondorCluster(memory='10 GiB', transfer_input_files=upload_to_dask, ship_env=False)
        ### memory is max memory per worker -- last I checked the max memory workers were using was ~2GiB
        #### minimum > 0: https://github.com/CoffeaTeam/coffea/issues/465
        cluster.adapt(minimum=1, maximum=100)
        print(cluster.dashboard_link)
        with Client(cluster) as client:
            print(client)
            print("Dashboard link ", client.dashboard_link)
            print(client.get_worker_logs())
            if verbose:
                run_instance = processor.Runner(
                                executor=processor.DaskExecutor(client=client, retries=10, treereduction=4,),#, status=False),
                                schema=NanoAODSchema,
                                savemetrics=True,
                                skipbadfiles=False,
                                chunksize=150000,
                    maxchunks = None,
                            )
            else:
                run_instance = processor.Runner(
                                executor=processor.DaskExecutor(client=client, retries=6, status=False, treereduction=4,),
                                schema=NanoAODSchema,
                                savemetrics=True,
                                skipbadfiles=False,
                                chunksize=150000,
                                maxchunks = None,
                            )
            # result, metrics = run_instance(samples,
            #                                "Events",
            #                                processor_instance = processor_inst,)
            with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # coffea 2025+: treename is keyword-only on Runner.__call__
                    result, metrics = run_instance(samples, processor_inst, treename="Events")
                    del metrics

#         print("Waiting for at least one worker...")
    else:
        #### iterative executor to print one file at a time
        print("Running locally")
        run_instance = processor.Runner(
            # executor = processor.FuturesExecutor(compression=None, workers=1),
            executor = processor.IterativeExecutor(workers=1),
            schema=NanoAODSchema,
            # chunksize=None,
            # maxchunks=None,
            skipbadfiles=True
        )
        with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # coffea 2025+: treename is keyword-only on Runner.__call__
                    result = run_instance(samples, processor_inst, treename="Events")
    elapsed = time.time() - tstart
    print(result)
    print("Time taken to run over samples ", elapsed)
    del cluster
    return result
def addFiles(files, RespOnly=False):
    respHists = ['response_matrix_u', 'response_matrix_g', 'ptreco_mreco_u', 'ptreco_mreco_g', 'ptgen_mgen_u', 'ptgen_mgen_g','fakes','fakes_u', 'misses', 'misses_u', 'fakes_g', "misses_g", 'jet_pt_eta_phi']
    ### load first file as base of results
    if RespOnly:
        results = pickle.load( open(files[0], "rb") )
        print(results.keys())
        results = {k: results[k] for k in respHists if k in results.keys()}
        if "misses" in results.keys():
            results["misses_u"] = results.pop("misses")
        if "fakes" in results.keys():
            results["fakes_u"] = results.pop("fakes")
    else:
        results = pickle.load( open(files[0], "rb") )
        if "misses" in results.keys():
            results["misses_u"] = results.pop("misses")
        if "fakes" in results.keys():
            results["fakes_u"] = results.pop("fakes")
    print("starting file ", files[0])
    print("new keys ", results.keys())
    for fname in files[1:]:
        print("doing file ", fname)
        with open(fname, "rb") as f:
            result = pickle.load( f )
            print(f)
            if RespOnly:
                for hist in [res for res in result if (res in respHists)]:
                    print("Starting ", hist)
                    if hist == "fakes" or hist == "fakes_u":
                        if hist in result.keys():
                            results["fakes_u"] += result[hist]
                            print("success for ", hist)
                    elif hist == "misses" or hist == "misses_u":
                        if hist in result.keys():
                            results["misses_u"] += result[hist]
                            print("success for ", hist)
                    else: results[hist] += result[hist]
                    # print("Success for ", hist)
            else:
                for hist in [res for res in result if (res in results) or res=="fakes" or res=="misses"]:
                    # print("Starting ", hist)
                    if hist == "cutflow":
                        for key in [key for key in result[hist]]:
                            print(key)
                            if key in results[hist].keys():
                                for k in [k for k in result[hist][key] if k in results[hist][key].keys()]:
                                    results[hist][key][k] += result[hist][key][k]
                                    print("success for ", key)
                            else:
                                results[hist][key] = {}
                                for k in [k for k in result[hist][key]]:
                                    results[hist][key][k] = result[hist][key][k]
                                    print("success for ", key)
                    else:
                        if hist == "fakes" or hist == "fakes_u":
                            if hist in result.keys():
                                results["fakes_u"] += result[hist]
                                print("success for ", hist)
                        elif hist == "misses" or hist == "misses_u":
                            if hist in result.keys():
                                results["misses_u"] += result[hist]
                                print("success for ", hist)
                        else: results[hist] += result[hist]
                        # print("Success for ", hist)
    print("Done")
    return(results)
    