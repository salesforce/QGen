from datetime import datetime
import json, os

def load_qd_annotations():
    annotations = []
    with open("quiz_design_data.jsonl", "r") as f:
        for line in f:
            annotations.append(json.loads(line))

    for d in annotations:
        d["timestamp"] = datetime.strptime(d["timestamp"], "%Y-%m-%d %H:%M:%S")
        d["doc_id"] = int(d["doc_id"])
        
    # Only keep the last annotation (as we store each step purposefully for timing)
    annotations = sorted(annotations, key=lambda a: a["timestamp"])
    M = {}
    for d in annotations:
        k = "%d||%s||%s" % (d["user_id"], d["doc_id"], d["answer_span"])
        M[k] = d
        
    unique_annotations = sorted(M.values(), key=lambda a: a["timestamp"])
    return unique_annotations

def build_qd_groups(annotations):
    with open("qd_content.json", "r") as f:
            evaluation_texts = json.load(f)

    groups = []
    for annot in annotations:
        answer_span = annot["answer_span"]
        document = evaluation_texts[annot["doc_id"]]["content"]
        paragraphs = document.split("<br />")
        relevant_paragraphs = [p for p in paragraphs if answer_span in p]
        relevant_paragraph = relevant_paragraphs[0]
        
        questions = []
        for q in annot["questions"]:
            label = 1 if "removed" not in q or q["removed"] is False else 0
            reason = q.get("reason", "No error")
            questions.append({"question": q["question"], "label": label, "reason": reason, "answer_span": answer_span, "model_name": q["model_name"]})

        d = {"doc_id": annot["doc_id"], "answer_span": answer_span, "context": relevant_paragraph, "questions": questions}
        groups.append(d)
    return groups