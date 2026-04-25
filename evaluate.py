"""Quick local eval script. Run: python evaluate.py"""

import json
import sys
from main import heuristic_predict

def evaluate():
    with open("relevant_priors_public.json") as f:
        data = json.load(f)

    truth_dict = {
        (t["case_id"], t["study_id"]): t["is_relevant_to_current"]
        for t in data["truth"]
    }

    total = correct = 0
    tp = fp = tn = fn = 0
    ambiguous = 0

    for case in data["cases"]:
        cur = case["current_study"]
        for prior in case["prior_studies"]:
            key = (case["case_id"], prior["study_id"])
            if key not in truth_dict:
                continue

            label = truth_dict[key]
            result = heuristic_predict(cur["study_description"], prior["study_description"])

            if result is None:
                # if this ever happens, treat as False in offline eval
                result = False
                ambiguous += 1

            total += 1
            if result == label:
                correct += 1
            if result and label:
                tp += 1
            elif result and not label:
                fp += 1
            elif not result and label:
                fn += 1
            else:
                tn += 1

    acc = correct / total * 100
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    baseline = sum(1 for t in data["truth"] if not t["is_relevant_to_current"]) / len(data["truth"]) * 100

    print(f"Total predictions : {total}")
    print(f"Accuracy          : {acc:.2f}%  (baseline always-false: {baseline:.2f}%)")
    print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"Precision         : {precision:.3f}")
    print(f"Recall            : {recall:.3f}")
    print(f"F1                : {f1:.3f}")
    print(f"Ambiguous (→LLM)  : {ambiguous} ({ambiguous/total*100:.1f}%)")


if __name__ == "__main__":
    evaluate()
