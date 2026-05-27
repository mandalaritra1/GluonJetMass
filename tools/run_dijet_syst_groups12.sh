#!/usr/bin/env bash
set -euo pipefail

# Run MG dijet jet-systematic jobs on Coffea-Casa in larger groups.
# Nominal is run once by itself to avoid double-counting weight variations and
# cutflow when merging. HEM is one-sided and runs by itself. All remaining jobs
# run six Up/Down systematic sources at a time, i.e. twelve labels per job.

PROC="${PROC:-dijet}"
YEARS="${YEARS:-2018}"
MCTYPE="${MCTYPE:-MG}"
EXECUTOR="${EXECUTOR:-dask-casa}"
CHUNKSIZE="${CHUNKSIZE:-400000}"
DASK_MEMORY="${DASK_MEMORY:-5 GiB}"
MAX_WORKERS="${MAX_WORKERS:-200}"
REDIRECTOR="${REDIRECTOR:-root://xcache/}"
FILESET="${FILESET:-fileset_MG_pythia8.json}"
DATASTR="${DATASTR:-QCD_MG}"
TESTING="${TESTING:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

run_group() {
  local year="$1"
  local label="$2"
  shift 2

  local outdir="${OUTDIR:-coffeaOutput/dijet/jet_syst_groups12_${year}}"
  mkdir -p "${outdir}"

  local output="${outdir}/dijetHists_${DATASTR}_${year}_${label}"
  if [[ "${TESTING}" == "1" ]]; then
    output="${output}_testing"
  fi
  output="${output}.pkl"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${output}" ]]; then
    echo "Skipping ${year} ${label}; output exists: ${output}"
    return
  fi

  local cmd=(
    python run.py
    --proc "${PROC}"
    --mc
    --year "${year}"
    --mctype "${MCTYPE}"
    --executor "${EXECUTOR}"
    --chunksize "${CHUNKSIZE}"
    --dask-memory "${DASK_MEMORY}"
    --redirector "${REDIRECTOR}"
    --max-workers "${MAX_WORKERS}"
    --fileset "${FILESET}"
    --datastr "${DATASTR}"
  )

  if [[ "${TESTING}" == "1" ]]; then
    cmd+=(--testing)
  fi

  cmd+=(--jet-syst "$@" --output "${output}")

  echo
  echo "=== Running ${year} ${label}: $* ==="
  echo "Output: ${output}"
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  if [[ "${DRY_RUN}" == "0" ]]; then
    "${cmd[@]}"
  fi
}

for year in ${YEARS}; do
  run_group "${year}" "000_nominal" nominal
  run_group "${year}" "001_HEM" HEM
  run_group "${year}" "002_JER_to_Fragmentation" \
    JERUp JERDown \
    JES_AbsoluteMPFBiasUp JES_AbsoluteMPFBiasDown \
    JES_AbsoluteScaleUp JES_AbsoluteScaleDown \
    JES_AbsoluteStatUp JES_AbsoluteStatDown \
    JES_FlavorQCDUp JES_FlavorQCDDown \
    JES_FragmentationUp JES_FragmentationDown
  run_group "${year}" "003_PileUp" \
    JES_PileUpDataMCUp JES_PileUpDataMCDown \
    JES_PileUpPtBBUp JES_PileUpPtBBDown \
    JES_PileUpPtEC1Up JES_PileUpPtEC1Down \
    JES_PileUpPtEC2Up JES_PileUpPtEC2Down \
    JES_PileUpPtHFUp JES_PileUpPtHFDown \
    JES_PileUpPtRefUp JES_PileUpPtRefDown
  run_group "${year}" "004_RelativeFSR_to_RelativePtEC1" \
    JES_RelativeFSRUp JES_RelativeFSRDown \
    JES_RelativeJEREC1Up JES_RelativeJEREC1Down \
    JES_RelativeJEREC2Up JES_RelativeJEREC2Down \
    JES_RelativeJERHFUp JES_RelativeJERHFDown \
    JES_RelativePtBBUp JES_RelativePtBBDown \
    JES_RelativePtEC1Up JES_RelativePtEC1Down
  run_group "${year}" "005_RelativePtEC2_to_RelativeStatFSR" \
    JES_RelativePtEC2Up JES_RelativePtEC2Down \
    JES_RelativePtHFUp JES_RelativePtHFDown \
    JES_RelativeBalUp JES_RelativeBalDown \
    JES_RelativeSampleUp JES_RelativeSampleDown \
    JES_RelativeStatECUp JES_RelativeStatECDown \
    JES_RelativeStatFSRUp JES_RelativeStatFSRDown
  run_group "${year}" "006_RelativeStatHF_to_JMS" \
    JES_RelativeStatHFUp JES_RelativeStatHFDown \
    JES_SinglePionECALUp JES_SinglePionECALDown \
    JES_SinglePionHCALUp JES_SinglePionHCALDown \
    JES_TimePtEtaUp JES_TimePtEtaDown \
    JMRUp JMRDown \
    JMSUp JMSDown
done
