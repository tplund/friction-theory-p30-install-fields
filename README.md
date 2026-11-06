# Nature and Nurture in a Language Model — code, notebooks, and data

Release repository for the paper **"Nature and Nurture in a Language Model:
Installable Value Fields, Intrinsic Capacity, and the Forward-Consolidation
Boundary"** (Pødenphant Lund, 2026), part of the Behavioural Friction Theory
paper series.

- **Paper (Zenodo):** concept DOI [10.5281/zenodo.20732528](https://doi.org/10.5281/zenodo.20732528)
  (v1 record [10.5281/zenodo.20732529](https://doi.org/10.5281/zenodo.20732529))
- **Author:** Tomas Pødenphant Lund — ORCID [0009-0000-4724-2427](https://orcid.org/0009-0000-4724-2427)
- **License:** CC-BY-4.0 (see `LICENSE`)

## What this paper shows

The four Behavioural-Friction-Theory fields split two-and-two onto nature and
nurture. The **value** fields (Safety, Meaning) install into a base language
model by fine-tuning as *graded intrinsic* directions; the **capacity** fields
(Ability, Effort) are read off the substrate, and a value-disposition install
returns expression, not capability. An installed **self-efficacy** disposition
gates how much of a fixed capacity ceiling is *realised* (give-up / avoidance
threshold) while latent capability stays flat, capacity-relatively and
monotonically across model sizes. Forward consolidation is offered as the named
human/LLM boundary.

## Layout

```
paper/        the paper's Markdown master (source of the manuscript)
scripts/      fine-tuning data generators, notebook builders, and analyzers
              (build_install_fields_*  /  build_paper30_*  /  analyze_install_fields_*)
notebooks/    the Colab notebooks that run each install + readout
              (install_fields_*  /  paper30_*)
data/         per-run result files (one directory per run): per-token CR,
              raw top-k logprobs, orientation-cluster mass, attempt/forced-answer
              accuracy, preference-U bins, with the informativeness-gate thresholds
```

The notebook builders (`scripts/build_*_notebook.py`) are the source of the
`.ipynb` files — the notebooks are generated, never hand-edited. Each run writes
streamed JSONL with the raw per-token top-k preserved, so every reported number
can be recomputed from the data here.

## Reproducing a result

1. Open the relevant notebook in Colab (e.g.
   `notebooks/paper30_safety_replication_crossscale_v1_0_colab.ipynb` for the
   cross-ladder Safety install, or
   `notebooks/paper30_self_efficacy_forced_answer_control_v1_2_colab.ipynb` for
   the refusal-impossible competence control).
2. Run it (LoRA fine-tunes of Qwen2.5 / Llama-3.1 in 4-bit; free Colab GPU).
3. The corresponding `data/<run>/` directory holds the result files this release
   was analysed from; the `analyze_install_fields_*.py` scripts reproduce the
   reported contrasts.

## Citation

> Pødenphant Lund, T. (2026). *Nature and Nurture in a Language Model:
> Installable Value Fields, Intrinsic Capacity, and the Forward-Consolidation
> Boundary.* Zenodo. https://doi.org/10.5281/zenodo.20732528

## Version 4 additions (2026-11)

The experiments added in version 4 of the paper (§3.6, §4.1, §4.3, §5.4, §6.3, §7.2, §7.3) run on Modal GPUs rather than Colab. Each driver is a `scripts/modal_p30_*.py` file with a matching `scripts/analyze_p30_*.py` scorer. The data are in `data/results/`, at the paths the scripts read and write, so the analysis runs unchanged from the repository root.

| paper section | driver → scorer | data (`data/results/`) |
|---|---|---|
| §7.2 route B, base models (own experience consolidated; yoked control; description arm) | `modal_p30_route_b_own_experience.py` → `analyze_p30_route_b_own_experience.py` | `p30_route_b_own_experience_v1.*` (+ `.episodes.jsonl`, the models' own answers) |
| §7.2 route B, re-read of the saved adapters (unsaturated readouts; stake factorial) | `modal_p30_route_b_reread.py`, `modal_p30_route_b_fields.py` → `analyze_p30_route_b_reread.py`, `analyze_p30_route_b_fields.py` | `p30_route_b_reread_v1.*`, `p30_route_b_fields_v1.*` |
| §7.2 route B, Qwen2.5-7B-Instruct (danger × reward re-read) | `modal_p30_route_b_instruct.py` → `analyze_p30_route_b_instruct.py` | `p30_route_b_instruct_v1.*` (+ `.episodes.jsonl`) |
| §6.3 stake ladder (Qwen2.5 0.5B–32B, OLMo-2 7B stages and 13B; no training) | `modal_p30_stake_ladder.py` → `analyze_p30_stake_ladder.py` | `p30_stake_ladder_v1.*` (+ `.gen.jsonl`, own answers) |
| §5.4 prompt race (danger × task × stated record; four instruct models) | `modal_p30_prompt_race.py` → `analyze_p30_prompt_race.py` | `p30_prompt_race_v1.*` (+ `.chat.jsonl`, sampled choices) |
| §4.1 what the marker-CR readout measures (slot-vocabulary diagnostic; prompt-masked installs) | `modal_p30_marker_slot_diagnostic.py`, `modal_p30_initiation_rescue.py` → `analyze_p30_marker_slot_diagnostic.py`, `analyze_p30_initiation_rescue.py` | `p30_marker_slot_diagnostic_v1.*`, `p30_initiation_rescue_v1.*` |
| §4.3 Bradley–Terry decomposition; §7.3 hazard slopes (re-analysis of existing data) | `paper30_selective_influence_reanalysis.py` | `paper30_selective_influence_reanalysis_v1.*`; inputs in `data/n2_preference_u_v2_0_run_2026_06_20/`, `data/safety_readout_robustness_v1_1_run_2026_06_20/`, `data/initiation_probe_v1_2_results/`, `data/hazard_discount_v1_1_results/` (pass `--data-root` pointing at a folder that holds them under the script's expected names) |

Each result row records the generating script, model identifier, precision and sampling parameters, and each dataset has a `.manifest.json`. All models run in 4-bit NF4. The Modal drivers use `scripts/modal_budget.py` as a per-run cost cap. The design notes named in the script headers are the author's records and are not part of this release; the readings each experiment was read against are stated in the paper.
