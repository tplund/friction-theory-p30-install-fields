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
