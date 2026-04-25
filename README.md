# Relevant Priors API

Predicts whether prior radiology examinations are relevant to a radiologist reading a current examination.

## Architecture

**Two-stage classifier:**

1. **Heuristic (fast, ~92% accuracy)** — maps study descriptions to semantic body-part groups (breast, chest, cardiac, spine, brain, abdomen, MSK joints, etc.) and returns `true` if groups overlap, `false` if both studies match groups with no overlap.
2. **LLM fallback (OpenAI)** — only triggered for ambiguous cases where one or both study descriptions match zero keyword groups. All ambiguous priors for a case are batched into a single API call.

### Why this approach?
- Heuristic resolves ~95%+ of cases instantly with no API cost
- LLM is used surgically for genuinely ambiguous descriptions
- In-memory cache avoids repeat LLM calls for the same study pair

## Local Development

```bash
pip install -r requirements.txt
cp .env.example .env
# then edit .env and set OPENAI_API_KEY
uvicorn main:app --reload
```

Test it:
```bash
python test_api.py
```

Evaluate against public data:
```bash
cp /path/to/relevant_priors_public.json .
python evaluate.py
```

## Deployment (Render — free tier)

1. Push this folder to a GitHub repo
2. Go to [render.com](https://render.com) → New Web Service → connect your repo
3. Set environment variable: `OPENAI_API_KEY=sk-...`
4. Deploy — your URL will be `https://relevant-priors-api.onrender.com`

## API

### `POST /predict`

**Request:**
```json
{
  "cases": [
    {
      "case_id": "1467",
      "current_study": {
        "study_id": "1467",
        "study_description": "CT CHEST WITH CONTRAST",
        "study_date": "2025-08-26"
      },
      "prior_studies": [
        {
          "study_id": "1468",
          "study_description": "CT CHEST WITHOUT CNTRST",
          "study_date": "2024-04-13"
        }
      ]
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "case_id": "1467",
      "study_id": "1468",
      "predicted_is_relevant": true
    }
  ]
}
```

### `GET /health`
Returns `{"status": "ok", "cache_size": N}`
