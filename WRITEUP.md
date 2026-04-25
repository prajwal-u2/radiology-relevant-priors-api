# Relevant Priors — Experiments, Results & Next Steps

## Problem Statement

Given a patient's current radiology examination and a list of their prior examinations, predict for each prior whether it is relevant for the radiologist to review while reading the current study.

**Input signal:** Only the study description string and date (e.g. `"CT CHEST WITH CONTRAST"`, `"MRI thoracic spine wo con"`). No image data.

**Class distribution (public split):** 23.8% relevant, 76.2% not relevant.

---

## Experiment 1 — Keyword Overlap Baseline

**Approach:** Extract modality keywords (CT, MRI, MAM, US, XR, PET…) and body-part keywords (chest, brain, knee, spine…) from both descriptions. Predict relevant if any keywords overlap.

**Result:**
| Metric | Value |
|--------|-------|
| Accuracy | 83.3% |
| Precision | 0.623 |
| Recall | 0.755 |
| F1 | 0.683 |

**Failure modes observed:**
- **False negatives:** `MAM SCREEN 3D` vs `ULTRASOUND LT DIAG TARGET` — both breast-related but no keyword overlap (MAM ≠ ULTRASOUND at keyword level). Also `DIGITAL SCREENER W CAD` matched nothing.
- **False positives:** Same modality, very different body parts (CT ABD/PELVIS vs CT CHEST — both match "CT").

---

## Experiment 2 — Semantic Body-Part Groups (Primary Approach)

**Approach:** Instead of exact keyword matching, define 17 semantic groups where anatomically related terms are clustered:
- Group 0 (Breast): `mam`, `mammograph`, `breast`, `digital screener`, `mri breast`, `us breast`
- Group 1 (Chest): `chest`, `lung`, `pulmonary`, `thorac`, `rib`, `pleural`
- Group 2 (Cardiac): `cardiac`, `coron`, `myo perf`, `spect`, `pet/ct`
- Groups 3-16: Spine, Brain, Abdomen/Pelvis, Knee, Shoulder, Hip, Ankle/Foot, Wrist/Hand, Elbow, Thyroid, Vascular, Prostate, Bone Density, Sinus, Gynaecology

Two studies are relevant if they share a group. Additionally, certain cross-group pairs are treated as relevant (chest ↔ cardiac, abdomen ↔ gynaecology, abdomen ↔ prostate).

If one or both descriptions match **no group** (ambiguous), escalate to LLM.

**Result on public split:**
| Metric | Value |
|--------|-------|
| Accuracy | 92.0% |
| Precision | 0.855 |
| Recall | 0.800 |
| F1 | 0.827 |

**Remaining errors:**
- Studies with no useful description text (e.g. `"BREAST"` alone vs a mammogram — now handled by group 0)
- Edge cases where the same modality covers multiple regions (e.g. PET/CT whole body)

---

## Experiment 3 — LLM Fallback for Ambiguous Cases

**Approach:** For cases where heuristic returns `None` (description matches no semantic group), batch all ambiguous priors for a case into a single OpenAI call. The prompt asks the model to rate each prior as relevant/irrelevant given the current study description.

**Design decisions:**
- **Batching:** One LLM call per case (not per prior) prevents timeouts
- **Caching:** Results cached by `(current_study_id, prior_study_id)` — retries are free
- **Fallback:** If LLM call fails, return `False` (safe default — better to miss a prior than crash)

**Expected improvement:** The ~5-8% of ambiguous cases (short/vague descriptions like `"BREAST"`, `"GYN"`, `"Chest PA"`) should see meaningful accuracy gains.

---

## Final System Performance

| Stage | Cases handled | Accuracy |
|-------|--------------|----------|
| Heuristic only (public split) | ~95% | 92.0% |
| Heuristic + LLM fallback (estimated) | 100% | ~93-94% |
| Baseline (always predict false) | 100% | 76.2% |

**Latency:** Heuristic resolves in <1ms. LLM calls add ~2-5s per case with ambiguous priors, but affect only a small fraction of cases. Total p99 latency for a 100-case batch well under 60s.

---

## Next Steps & Improvements

### Short-term (high impact)
1. **TF-IDF / embedding similarity** — Encode descriptions with a pre-trained medical NLP model (e.g. ClinicalBERT, BiomedNLP-BiomedBERT) and predict relevance by cosine similarity. Would handle typos and abbreviation variations.
2. **Train a lightweight classifier** — Use the public labels to fine-tune a logistic regression or small transformer on description pairs. Should push accuracy above 95%.
3. **Date recency feature** — Priors more than 5-7 years old may be less relevant even if same body part. Add a time-decay factor.

### Medium-term
4. **Better abbreviation normalization** — Build a lookup table of common radiology abbreviations (`XR` = X-ray, `MAM` = mammography, `NM` = nuclear medicine, `WO` = without contrast, etc.) to normalize descriptions before matching.
5. **Modality-aware groups** — Within the same body part, some modality combinations are more relevant than others (CT vs MRI chest for oncology staging vs CT vs ultrasound).
6. **Side/laterality matching** — `MRI knee LEFT` should be more relevant to `CT knee LEFT` than `CT knee RIGHT`. Extract laterality and use it as a bonus signal.

### Long-term
7. **Full image-based relevance** — If DICOM metadata (series descriptions, body part examined tags) or thumbnail images become available, use a vision model to assess visual similarity.
8. **Radiologist feedback loop** — Collect radiologist accept/reject actions on presented priors to build labeled data and continually retrain.
9. **Structured DICOM tags** — In production, `Modality` (CT/MR/US) and `BodyPartExamined` DICOM tags would make this far more reliable than free-text parsing.

---

## Repository Structure

```
relevant-priors-api/
├── main.py            # FastAPI app — heuristic + LLM classifier
├── evaluate.py        # Local evaluation script
├── test_api.py        # API smoke test
├── requirements.txt
├── Dockerfile
├── render.yaml        # Render.com deployment config
└── README.md
```
