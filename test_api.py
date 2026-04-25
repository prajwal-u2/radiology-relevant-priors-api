"""
Quick smoke test for the API.
Usage:
  python test_api.py
  python test_api.py https://your-url.com
"""

import sys
import json
import urllib.request
import urllib.error

BASE_URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8000"

SAMPLE_REQUEST = {
    "cases": [
        {
            "case_id": "1467",
            "current_study": {
                "study_id": "1467",
                "study_description": "CT CHEST WITH CONTRAST",
                "study_date": "2025-08-26"
            },
            "prior_studies": [
                {"study_id": "1468", "study_description": "CT CHEST WITHOUT CNTRST", "study_date": "2024-04-13"},
                {"study_id": "530032", "study_description": "MRI thoracic spine wo con", "study_date": "2024-09-25"},
                {"study_id": "1469", "study_description": "CT ANGIOGRAM, ABD/PELVIS", "study_date": "2024-09-14"},
                {"study_id": "2077140", "study_description": "XR chest 2V PA/lat", "study_date": "2025-08-20"},
                {"study_id": "1706965", "study_description": "MRI prostate wo/w con", "study_date": "2023-07-13"},
            ]
        }
    ]
}

# expected labels for the sample above

EXPECTED = {
    "1468": True,
    "530032": False,
    "1469": False,
    "2077140": True,
    "1706965": False,
}

def main():
    url = f"{BASE_URL}/predict"
    body = json.dumps(SAMPLE_REQUEST).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()}")
        return

    print("Response:")
    print(json.dumps(result, indent=2))

    print("\nAccuracy check:")
    correct = 0
    for pred in result["predictions"]:
        sid = pred["study_id"]
        got = pred["predicted_is_relevant"]
        exp = EXPECTED.get(sid, "?")
        ok = "✓" if got == exp else "✗"
        print(f"  {ok} study={sid} predicted={got} expected={exp}")
        if got == exp:
            correct += 1

    print(f"\n{correct}/{len(EXPECTED)} correct")

if __name__ == "__main__":
    main()
