# v44b_pre_normalization_frequency_response First Smoke Verdict

This file records the preregistered first-stage smoke result for `v44b_pre_normalization_frequency_response`.
It is a post-run verdict, not a modification of the preregistration.

## Run

Command:

```powershell
conda run -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v44b_pre_normalization_frequency_response --datasets "acm,dblp,flickr" --epochs 80 --device cuda --log-level WARNING
```

Output files:

```text
results/unified_aptc_9datasets_v44b_pre_normalization_frequency_response.csv
results/unified_aptc_9datasets_v44b_pre_normalization_frequency_response_diagnostics.jsonl
```

The diagnostics file may also contain a prior 1-epoch ACM connectivity check. The verdict below uses the latest record per dataset.

## Result Summary

| Dataset | ACC | NMI | ARI | Emb-KM ACC | Final ACC | Emb-Post Gap | Pre-HP Std | Corr | Response Gap | Band Mass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.7098 | 0.3451 | 0.3787 | 0.7098 | 0.7098 | 0.0000 | 0.0252 | -0.0737 | -0.0016 | 0.5009 |
| DBLP | 0.6623 | 0.3731 | 0.3144 | 0.6621 | 0.6623 | -0.0002 | 0.0109 | 0.3538 | 0.0045 | 0.6863 |
| Flickr | 0.3683 | 0.2106 | 0.1319 | 0.3641 | 0.3683 | -0.0042 | 0.0477 | 0.3017 | 0.0271 | 0.5033 |

## Gate Verdict

### Red-Line Gate

PASS:

- `legacy_head_used=false` on ACM/DBLP/Flickr.
- `v43b_enabled=false` on ACM/DBLP/Flickr.
- `v44_enabled=false` on ACM/DBLP/Flickr.
- `v44b_enabled=true` on ACM/DBLP/Flickr.
- The runner variant explicitly sets v43b, ideal embedding-pressure, and v44a post-normalized high-pass/topology weights to 0.

### Response Non-Degeneracy Gate

PARTIAL PASS:

Required 3/3:

```text
v44b_pre_hp_response_std > 1e-4
v44b_pre_hp_response_p90 > v44b_pre_hp_response_p10
```

This condition passed on all three datasets:

| Dataset | Pre-HP Std | P10 | P90 | Verdict |
| --- | ---: | ---: | ---: | --- |
| ACM | 0.0252 | 0.0018 | 0.0597 | PASS |
| DBLP | 0.0109 | 0.0027 | 0.0301 | PASS |
| Flickr | 0.0477 | 0.0003 | 0.0350 | PASS |

Required at least 2/3:

```text
v44b_response_gap > 0
v44b_conflict_response_corr >= 0.05
```

This condition passed on DBLP and Flickr, but failed on ACM:

| Dataset | Corr | Response Gap | Verdict |
| --- | ---: | ---: | --- |
| ACM | -0.0737 | -0.0016 | FAIL |
| DBLP | 0.3538 | 0.0045 | PASS |
| Flickr | 0.3017 | 0.0271 | PASS |

Interpretation: v44b successfully fixes the v44a measurement degeneracy, but the conflict-response coupling is not universally aligned, especially on ACM.

### Normalization-Degeneracy Safety Check

PASS as diagnostic evidence:

- `v44b_postnorm_hp_energy_mean=0.5` on all datasets.
- `v44b_postnorm_hp_energy_std≈0` on all datasets.
- `v44b_postnorm_energy_gap≈0` on all datasets.

This confirms that post-normalized high-pass energy is still degenerate and that v44b's useful signal comes from pre-normalization response diagnostics instead.

### Posterior/Readout Safety Gate

PASS:

| Dataset | Emb-Post Gap | Verdict |
| --- | ---: | --- |
| ACM | 0.0000 | PASS |
| DBLP | -0.0002 | PASS |
| Flickr | -0.0042 | PASS |

All satisfy:

```text
abs(embedding_posterior_gap) <= 0.02
```

Interpretation: failure is not caused by posterior/readout detachment.

### Performance Gate

FAIL:

| Dataset | Required ACC | Actual ACC | Verdict |
| --- | ---: | ---: | --- |
| ACM | >= 0.8000 | 0.7098 | FAIL |
| DBLP | >= 0.6450 | 0.6623 | PASS |
| Flickr | >= 0.4500 | 0.3683 | FAIL |

### Topology Safety Gate

FAIL:

| Dataset | Ceiling | Actual Band Mass | Verdict |
| --- | ---: | ---: | --- |
| ACM | <= 0.4991 | 0.5009 | FAIL |
| DBLP | <= 0.6877 | 0.6863 | PASS |
| Flickr | <= 0.5051 | 0.5033 | PASS |

ACM slightly violates the preregistered topology safety ceiling.

## Mechanism Interpretation

`v44b_pre_normalization_frequency_response` should stop after the first-stage smoke.

The mechanism achieved its diagnostic goal of escaping v44a's post-normalization constant-energy failure: pre-normalization high-pass response has clear variance on all three datasets, and DBLP/Flickr show positive conflict-response coupling. However, the run fails the preregistered performance gate on ACM and Flickr, fails conflict coupling on ACM, and slightly violates ACM topology safety.

This is a mechanistic partial pass but an overall preregistered gate failure.

Key interpretation:

- v44a's diagnosis was correct: post-normalized energy was the wrong target.
- v44b's pre-normalization response is measurable and non-constant.
- But using only a scalar node-level pre-HP correlation objective is not sufficient to improve ACM/Flickr performance or guarantee homophilic safety.
- The remaining issue is not readout detachment; it is how the frequency response is coupled back into topology/representation without harming homophilic graphs.

## Decision

Do not run:

- second-batch smoke on Wiki/BlogCatalog/Texas
- full 9-dataset smoke
- 260-epoch full run
- v44b weight sweep

The next mechanism should not simply increase `v44b_pre_hp_corr_weight`. A valid next attempt should preserve the pre-normalization response diagnostic but change the coupling target so it is not a global scalar correlation objective that can be positive on DBLP/Flickr while anti-aligned on ACM.

Candidate next directions:

1. Use pre-HP response only as a diagnostic, not as a loss, and combine it with topology band safety before optimizing.
2. Move from node-level global correlation to edge-local signed frequency response conditioned on topology masks.
3. Add a homophily-safety constraint that prevents high conflict response from increasing ambiguous band mass on ACM-like graphs, without using dataset-specific routing.
4. Revisit topology band resolution separately before coupling response to representation, because ACM band mass exceeded the safety ceiling.

## No-Fabrication Status

All v44b numbers in this verdict come from the completed 80-epoch smoke diagnostics and CSV output. No unrun datasets, second-batch results, full-run results, or SOTA claims are reported here.
