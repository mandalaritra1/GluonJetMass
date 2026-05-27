#!/usr/bin/env bash
set -euo pipefail

# Run the 2016 MG dijet processor on Coffea-Casa in jet-systematic groups.
# Nominal is run once by itself; HEM is one-sided and runs by itself; the
# remaining jobs run two Up/Down systematic sources at a time (four labels).

PROC="${PROC:-dijet}"
YEAR="${YEAR:-2016}"
MCTYPE="${MCTYPE:-MG}"
EXECUTOR="${EXECUTOR:-dask-casa}"
CHUNKSIZE="${CHUNKSIZE:-200000}"
MAX_WORKERS="${MAX_WORKERS:-200}"
REDIRECTOR="${REDIRECTOR:-root://xcache/}"
FILESET="${FILESET:-fileset_MG_pythia8.json}"
DATASTR="${DATASTR:-QCD_MG}"
OUTDIR="${OUTDIR:-coffeaOutput/dijet/jet_syst_groups4_${YEAR}}"
TESTING="${TESTING:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${OUTDIR}"

COMMON_ARGS=(
  --proc "${PROC}"
  --mc
  --year "${YEAR}"
  --mctype "${MCTYPE}"
  --executor "${EXECUTOR}"
  --chunksize "${CHUNKSIZE}"
  --redirector "${REDIRECTOR}"
  --max-workers "${MAX_WORKERS}"
  --fileset "${FILESET}"
  --datastr "${DATASTR}"
)

if [[ "${TESTING}" == "1" ]]; then
  COMMON_ARGS+=(--testing)
fi

run_group() {
  local label="$1"
  shift
  local output="${OUTDIR}/dijetHists_${DATASTR}_${YEAR}_${label}"
  if [[ "${TESTING}" == "1" ]]; then
    output="${output}_testing"
  fi
  output="${output}.pkl"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${output}" ]]; then
    echo "Skipping ${label}; output exists: ${output}"
    return
  fi

  echo
  echo "=== Running ${label}: $* ==="
  echo "Output: ${output}"
  local cmd=(python run.py "${COMMON_ARGS[@]}" --jet-syst "$@" --output "${output}")
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  if [[ "${DRY_RUN}" == "0" ]]; then
    "${cmd[@]}"
  fi
}

run_group "000_nominal" nominal
run_group "001_HEM" HEM
run_group "002_JER__JES_AbsoluteMPFBias" JERUp JERDown JES_AbsoluteMPFBiasUp JES_AbsoluteMPFBiasDown
run_group "003_JES_AbsoluteScale__JES_AbsoluteStat" JES_AbsoluteScaleUp JES_AbsoluteScaleDown JES_AbsoluteStatUp JES_AbsoluteStatDown
run_group "004_JES_FlavorQCD__JES_Fragmentation" JES_FlavorQCDUp JES_FlavorQCDDown JES_FragmentationUp JES_FragmentationDown
run_group "005_JES_PileUpDataMC__JES_PileUpPtBB" JES_PileUpDataMCUp JES_PileUpDataMCDown JES_PileUpPtBBUp JES_PileUpPtBBDown
run_group "006_JES_PileUpPtEC1__JES_PileUpPtEC2" JES_PileUpPtEC1Up JES_PileUpPtEC1Down JES_PileUpPtEC2Up JES_PileUpPtEC2Down
run_group "007_JES_PileUpPtHF__JES_PileUpPtRef" JES_PileUpPtHFUp JES_PileUpPtHFDown JES_PileUpPtRefUp JES_PileUpPtRefDown
run_group "008_JES_RelativeFSR__JES_RelativeJEREC1" JES_RelativeFSRUp JES_RelativeFSRDown JES_RelativeJEREC1Up JES_RelativeJEREC1Down
run_group "009_JES_RelativeJEREC2__JES_RelativeJERHF" JES_RelativeJEREC2Up JES_RelativeJEREC2Down JES_RelativeJERHFUp JES_RelativeJERHFDown
run_group "010_JES_RelativePtBB__JES_RelativePtEC1" JES_RelativePtBBUp JES_RelativePtBBDown JES_RelativePtEC1Up JES_RelativePtEC1Down
run_group "011_JES_RelativePtEC2__JES_RelativePtHF" JES_RelativePtEC2Up JES_RelativePtEC2Down JES_RelativePtHFUp JES_RelativePtHFDown
run_group "012_JES_RelativeBal__JES_RelativeSample" JES_RelativeBalUp JES_RelativeBalDown JES_RelativeSampleUp JES_RelativeSampleDown
run_group "013_JES_RelativeStatEC__JES_RelativeStatFSR" JES_RelativeStatECUp JES_RelativeStatECDown JES_RelativeStatFSRUp JES_RelativeStatFSRDown
run_group "014_JES_RelativeStatHF__JES_SinglePionECAL" JES_RelativeStatHFUp JES_RelativeStatHFDown JES_SinglePionECALUp JES_SinglePionECALDown
run_group "015_JES_SinglePionHCAL__JES_TimePtEta" JES_SinglePionHCALUp JES_SinglePionHCALDown JES_TimePtEtaUp JES_TimePtEtaDown
run_group "016_JMR__JMS" JMRUp JMRDown JMSUp JMSDown
