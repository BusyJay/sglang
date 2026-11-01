# GLM MLX DFlash2 verification optimization, 2026-11-02

Remote hardware: M5 Ultra256GiB, applegpu_g17d, MLX0.32.4, mlx-lm0.32.2.
Target mixed4bitcheckpoint revision fa04cbc3622b16aa7475404060b05e506d81bd4e;
originalBF16draft bf582e4eacc1810f76656d1811693ff6c6737d2a. Same repository
base185e108b3f2a59614003e616d674722160e5f2db, no nativeMTP loaded.
No changes to checkpoint quantization, equations or rounding boundaries.
Code committed locally and in the remote checkout; no Git hosting push.
Final source hashes:source_manifest_verify_final.json.

## Adopted mechanisms

1. Batch attention mHC input/output over the verification chain, preserve all
per-position snapshots. Fullgeneration gain small(<1%); retained.
2. Group routed gate/up picks across the wholechain, retain three-rowcompute
chunks and original scatter/rounding. Fullverify gain1.1-2.8%; retained.
3. KDA inputbank[24896,4096] fullchain4..8rowtile replaces max4rowtile, uses
BM4RPS2 for5..8rows andBM8RPS2for4. Exact same reduction and BF16casts.
This avoids rereading the largeweightbank and limits outputaccumulators.
4. KDA outputbank[4096,8192] fullchain5..8rowtileBM4RPS2. Additional complete
verification cost reduction1.5-2%,~.8-1.2ms, not claimed as mainoptimization.
5. Auxiliary prefix snapshot views collected across layers and submitted in
one async_eval. Later sourcewrites cannot overwrite snapshot storage.
6. DFlash proposalchain readback once perchain, replacingperrow scalaritem.
Normalgeneration effect withinnoise; no substantialthroughputclaim.

## Full-generation inputbank comparison

Baseline:wholechainexpertgrouping andbatchedmHC, oldmax4rowKDAinputtile.
Candidate:newinputbanktile, otherwiseidentical. Two cleansequentialprocesses,
24pairedcases128outputs,3prompts,32/2050context,AR/D3/D5/D7. Tokens and all
acceptancestatistics exact. Shortprompt context actually32. Throughput aggregate
is sumgeneratedtokens / sumdecodetime across3prompts; prefill excluded.

|Context|Arm|Baseline tok/s|Candidate tok/s|Gain|
|---|---|---|---|---|
|32|AR|37.526|37.429|-.26%|
|32|D3|70.476|71.426|1.35%|
|32|D5|69.403|76.203|9.80%|
|32|D7|69.728|75.267|7.94%|
|2050|AR|34.379|34.294|-.25%|
|2050|D3|61.872|62.477|.98%|
|2050|D5|53.877|59.515|10.47%|
|2050|D7|55.471|59.751|7.71%|

Same-resident11trialAB/BA fullverify,first2excluded, alloutputs/captures and
everypositioncache snapshot exact:

|Context|Rows|Baseline ms|Candidate ms|Costreduction|
|---|---|---|---|---|
|32|4|39.973|39.625|.87%|
|32|6|56.526|49.905|11.71%|
|32|8|65.597|59.237|9.69%|
|2050|4|44.554|44.215|.76%|
|2050|6|63.411|56.901|10.27%|
|2050|8|76.141|69.538|8.67%|

Raw:results/verify_dense_ab_baseline.json,verify_dense_ab_candidate.json,
verify_dense_paired.json. Completecompiledverify timing, not isolatedkernels.
No nativeMTPcheckpoint residence:187.0241GiB vs191.2883GiB at same-stage load,
4.2642GiB saved. The initialintegrationcomparekeptMTPresident allarms; do not
use that residence as productionDFlashmemory. CurrentinputbankABswap decreased.

## Follow-up outputbank and hostreadback

Sevenpairedtrials,same-targetfullverify,alloutputs/captures/snapshots exact:
short6row49.944->49.174ms,8row59.314->58.083;
long6row56.933->56.041,8row69.651->68.523. Fourrows unchangedwithinnoise.
The hostreadback comparison kept finaltargetkernels fixed,24pairedgeneration
cases exact. Candidate shortD3/D5/D7:71.114/78.601/76.956tok/s;
long:61.791/60.704/61.290. Hostreadbackeffects-.49..+.63%, withinnoise.
These generationnumbers precede finalauxiliaryasyncsnapshotHTTPvalidation.
Default remains D3 for robust long context. D5wins shortcontext; no universal5claim.

## Eval and shape investigation

See EVAL_AUDIT.md. Fixedcompileblocks andoperatorinputs, reducing12async
submissions to1 makes6rowfullverify49.484->56.029msshort,
56.228->64.514long. InstalledCPUstack sample shows nativeconditionwaitwithin
async_eval. Async does not mean absenceofbackpressure. Threecommandbuffer
knobdiagnosticsdidnotremoveit; no productionknobchange adopted.

SharedFFNgate/upwiderrowtile:actualsmall6bitsharedexperts show modestbenefit,
large8rowgate/up worsens .678->.889ms. Notaddedtoproductionbasedonlysmall
replaygain. NativeQMMhas tinyFP16differences; notdiscardedsolelybitdifference,
exactcustomwideKDApathwasfasterfirst. No KDAconvwindowdedupintroduced:
currentkernelalreadyloadswin[CK-1+TOK]onceperchannel; duplicatedsnapshotstorage
is a remainingmechanismoption, notprovenprimarycost.

GPUdispatchtimeline offinalverificationnotcaptured. Nativebackpressure cause
(taskcountvsmemory) remainsunresolved. ArchivedMLXsourcewaitlogic isreference,
not asserted as installedlibsourceidentity. No platformlimitationclaim.

## Validation

Inputbankdevice dense7+GLM63passed. Wholechainexpertgroupaffine10+mHC6+
GLM63+DFlash8passed. Outputbankdense7+auxiliary6+DFlash8passed. Following
asyncsnapshotchangeauxiliary6+DFlash8+GLM63passed. Testscoverrounding,mixedbits,
replacementweights,requestboundaries,prefixreuse,cachefork/rejection/abort.
Final five changed production/test files Ruff format/isort checks, selected
Ruff lint, registered-test admission and package checks, diff check all passed
after final HTTP. Remote manifest matched all 21 files (run-exa2i2u9).

PreviousgroupHTTP17cases+concurrent2exactvsarchivedAR,stream67.568tok/s;
its swapused1382.81->1871.06MiB(total2048->3072)grew,sharedhostattribution
unproven. Do notclaimswapstable. Final HTTP D3+D5 complete, distinct artifacts:
results/http_verify_final_d3.json, http_verify_final_d5.json.

## Reproduction

Allscriptsunderthisdirectory,remoteinvocationthrough existingtmux%0 only.
verify_dense_ab.sh reproducesinputbankgenerationAB;verify_dense_paired.py
fullverifyagainstbaseline_dense/mlx_dense_rows.py. verify_output_paired.py
isolatesoutputbankagainstbaseline_output. verify_host_ab.sh isolatestoken
readbackagainstbaseline_host. verify_submission_only.py fixescompileblocks;
verify_submission_paired.py changesbothcompileblocksandsubmissions.
async_checks_http.sh revalidatescurrentasyncsnapshotandservesD3/D5overHTTP.
Standalone kernel replay costs include explicitsyncandarenotadditive.


## Final HTTP serving acceptance

Stream numbers below measure client SSE arrival rate, not model decode runtime.
Historical JSON field `decode_tok_s` has that same client-arrival meaning.
Actual decoder-loop timings and precise aggregation are documented in
`llm-online/README.md`; HTTP and decoder prompts differ.

Final async snapshot + token vector readback + wide KDA input/output banks.
Each D3/D5 run has 17 exact semantic signatures and two concurrent exact
signatures compared with archived AR: raw tokens, logprobs/top3, seeded sample,
stop, prefix2050reuse2048, abort then continued request.

| Drafts | Stream aggregate tok/s | Two-request aggregate tok/s | Startup s |
|---|---|---|---|
|3|67.511|73.705|55.294|
|5|66.592|65.292|55.264|

TTFT first request1.504/1.572s; later .112-.145s. Same HTTP workload has D3
slightly faster, so default remains3. Standalone three-prompt short-context
D5 performance does not establish universal serving superiority.

System swap grew during final HTTP: D3 used1229.81->1467.69MiB, D5
1467.69->1624.06MiB and total2048->3072MiB. Cause unproven on shared host;
no claim of stable memory or fully solved service cost. Processes terminated
through fixture cleanup. No systemwired-memory tuning or extra tools installed.

Explicit remaining issue: native async_eval scheduling wait not fixed by
coalescing all submissions or command-buffer batching controls. Need GPU
command timeline/native instrumentation to distinguish task-count throttling
from allocator pressure and confirm dispatch behavior. Input-shape KDA bank
issue fixed with~10% generation gain for widechains; that measured gain is
separate from unresolved async scheduling wait. No platform limitation claim.
