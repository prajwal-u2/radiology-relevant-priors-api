"""API to mark prior studies as relevant/not relevant."""

import json
import logging
import os
import time
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

# basic logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# app
app = FastAPI(title="Relevant Priors API", version="1.0")

# OpenAI client
client = OpenAI()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# in-memory cache
_cache: dict[tuple[str, str], bool] = {}


# heuristic rules

BREAST_GROUP   = {"mam", "mammograph", "breast", "digital screener",
                  "mri breast", "us breast", "ultrasound breast", "biopsy breast"}
CHEST_GROUP    = {"chest", "lung", "pulmonary", "thorax", "pleural", "mediastin"}
CARDIAC_GROUP  = {"cardiac", "heart", "coron", "myo", "echo", "cardio", "pet/ct"}
SPINE_GROUP    = {"spine", "cervic", "lumbar", "thoracic spine", "sacr"}
BRAIN_GROUP    = {"brain", "head", "cranial", "intracran", "neuro"}
ABD_PEL_GROUP  = {"abdomen", "abd", "pelvis", "liver", "renal", "kidney",
                  "pancrea", "bowel", "colon", "gallbladder", "spleen"}
MSK_GROUPS = {
    "knee":       {"knee", "patell"},
    "shoulder":   {"shoulder", "rotat"},
    "hip":        {"hip", "femur", "acetab"},
    "ankle_foot": {"ankle", "foot", "calcan", "achilles"},
    "wrist_hand": {"wrist", "hand", "finger", "carpal"},
    "elbow":      {"elbow"},
}

ALL_GROUPS: list[set[str]] = [
    BREAST_GROUP, CHEST_GROUP, CARDIAC_GROUP, SPINE_GROUP, BRAIN_GROUP, ABD_PEL_GROUP,
    *MSK_GROUPS.values(),
]


def _matched_groups(desc: str) -> set[int]:
    d = desc.lower()
    matched: set[int] = set()
    for i, grp in enumerate(ALL_GROUPS):
        for kw in grp:
            if kw in d:
                matched.add(i)
                break
    return matched


def heuristic_predict(current_desc: str, prior_desc: str) -> bool:
    cg = _matched_groups(current_desc)
    pg = _matched_groups(prior_desc)

    if cg & pg:
        return True

    # keep all breast variants together
    c, p = current_desc.lower(), prior_desc.lower()
    is_breast_c = "mam" in c or "breast" in c
    is_breast_p = "mam" in p or "breast" in p or "ultrasound" in p or "us breast" in p
    if is_breast_c and is_breast_p:
        return True
    if is_breast_p and ("mam" in c or "breast" in c or "ultrasound" in c):
        return True

    return False


def _is_ambiguous(current_desc: str, prior_desc: str) -> bool:
    """True if either description matches no semantic group (heuristic is unreliable)."""
    cg = _matched_groups(current_desc)
    pg = _matched_groups(prior_desc)
    return not cg or not pg


# LLM fallback for unsure pairs

LLM_SYSTEM = """You are a radiology informatics assistant.
Decide whether each prior radiology examination is relevant to show to a radiologist reading the current examination.

A prior is RELEVANT if it images the same or closely related anatomical region.

Rules:
- Same body part → relevant
- Breast imaging (mammogram, US breast, MRI breast) = relevant to each other
- Chest/lung CT <-> chest X-ray / PET-CT / lung cancer screening = relevant
- Cardiac studies relevant to other cardiac and to chest imaging
- Spine segments (cervical/thoracic/lumbar) relevant to each other
- Completely different body parts = NOT relevant
- Modality alone does NOT determine relevance

Respond ONLY with a JSON array in the same order as the priors given.
Each entry: {"study_id": "...", "relevant": true|false}
No markdown, no explanation."""


def _llm_batch_predict(current_desc: str, current_date: str, priors: list[dict]) -> dict[str, bool]:
    prior_lines = "\n".join(
        f'{i+1}. study_id={p["study_id"]} | date={p.get("study_date","")} | "{p["study_description"]}"'
        for i, p in enumerate(priors)
    )
    user_msg = f'Current (date={current_date}): "{current_desc}"\n\nPriors:\n{prior_lines}'
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": LLM_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            max_tokens=1000,
        )
        raw = (response.choices[0].message.content or "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        results = json.loads(raw)
        return {str(r["study_id"]): bool(r["relevant"]) for r in results}
    except Exception as e:
        log.error("LLM batch failed: %s", e)
        return {str(p["study_id"]): False for p in priors}


# request/response models

class Study(BaseModel):
    study_id: str
    study_description: str
    study_date: Optional[str] = ""


class Case(BaseModel):
    case_id: str
    current_study: Study
    prior_studies: list[Study]
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None


class PredictRequest(BaseModel):
    cases: list[Case]


class Prediction(BaseModel):
    case_id: str
    study_id: str
    predicted_is_relevant: bool


class PredictResponse(BaseModel):
    predictions: list[Prediction]


# endpoints

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    t0 = time.time()
    total_priors = sum(len(c.prior_studies) for c in req.cases)
    log.info("REQUEST | cases=%d priors=%d", len(req.cases), total_priors)

    predictions: list[Prediction] = []

    for case in req.cases:
        cur = case.current_study
        ambiguous_priors: list[Study] = []

        for prior in case.prior_studies:
            cache_key = (cur.study_id, prior.study_id)

            if cache_key in _cache:
                predictions.append(Prediction(
                    case_id=case.case_id,
                    study_id=prior.study_id,
                    predicted_is_relevant=_cache[cache_key],
                ))
                continue

            result = heuristic_predict(cur.study_description, prior.study_description)

            if _is_ambiguous(cur.study_description, prior.study_description):
                ambiguous_priors.append(prior)

            _cache[cache_key] = result
            predictions.append(Prediction(
                case_id=case.case_id,
                study_id=prior.study_id,
                predicted_is_relevant=result,
            ))

        # re-check ambiguous priors in one LLM call per case
        if ambiguous_priors:
            log.info("LLM | case=%s ambiguous=%d current='%s'",
                     case.case_id, len(ambiguous_priors), cur.study_description)
            llm_results = _llm_batch_predict(
                current_desc=cur.study_description,
                current_date=cur.study_date or "",
                priors=[p.model_dump() for p in ambiguous_priors],
            )
            for pred in predictions:
                if pred.case_id == case.case_id and pred.study_id in llm_results:
                    new_val = llm_results[pred.study_id]
                    _cache[(cur.study_id, pred.study_id)] = new_val
                    pred.predicted_is_relevant = new_val

    log.info("RESPONSE | predictions=%d elapsed=%.2fs", len(predictions), time.time() - t0)
    return PredictResponse(predictions=predictions)


@app.get("/health")
def health():
    return {"status": "ok", "cache_size": len(_cache)}


@app.get("/")
def root():
    return {"service": "relevant-priors-api", "version": "1.0"}
