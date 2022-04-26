from flask import Flask, request, render_template, send_from_directory
from model_hf_generator import GeneratorHF
from datetime import datetime, timedelta
import os, random, json, flask

CACHE_FILE = "qd_cache.json"
ANNOT_FILE = "qd_annotations_running.jsonl"
CONTENT_FILE = "qd_content.json"

def load_question_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    else:
        return {}

def mark_paragraph_answer(paragraph, answer, model_card=""):
    if "prophetnet" in model_card:
        return "%s [SEP] %s" % (answer, paragraph)
    elif "mixqg" in model_card:
        return f"{answer} \\n {paragraph}"
    else:
        return "%s \n %s" % (answer, paragraph) # The default, used for our trained models

def save_question_cache():
    with open(CACHE_FILE, "w") as f:
        json.dump(cached_questions, f)

def deduplicate_questions(questions):
    M = {}
    for q in questions:
        if q["question"] not in M:
            M[q["question"]] = []
        M[q["question"]].append(q["model_name"])
    return [{"model_name": "|".join(v), "question": k} for k, v in M.items()]

def load_qgen_models():
    global QGEN_MODELS, scorer
    QGEN_MODELS = [
        # {"model_name": "dgpt2_sup", "model": GeneratorHF("distilgpt2", starter_file="qgen/dgpt2_squad_aaware_1.794.bin")},
        # {"model_name": "gpt2b_sup", "model": GeneratorHF("gpt2", starter_file="qgen/gpt2b_squad_aaware_1.575.bin")},
        # {"model_name": "bartb_sup", "model": GeneratorHF("facebook/bart-base", starter_file="qgen/bartb_nf_squad_aaware_1.492.bin")},
        # {"model_name": "bartl_sup", "model": GeneratorHF("facebook/bart-large", starter_file="qgen/bartL_nf_squad_aaware_1.290.bin")},
        # {"model_name": "gpt2m_sup", "model": GeneratorHF("gpt2-medium", starter_file="qgen/gpt2m_nf_squad_aaware_1.423.bin")},
        {"model_name": "mixqg-base", "model": GeneratorHF(model_card='Salesforce/mixqg-base')},
        {"model_name": "mixqg-large", "model": GeneratorHF(model_card='Salesforce/mixqg-large')},
        {"model_name": "prophetnet", "model": GeneratorHF(model_card='microsoft/prophetnet-large-uncased-squad-qg')}
    ]

    print("Qgen models loaded")

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

QGEN_MODELS = []
scorer = None

load_qgen_models()
cached_questions = load_question_cache()

@app.before_request
def before_request():
    user_id = -1
    if "user_id" in flask.request.cookies:
        try:
            user_id = int(flask.request.cookies["user_id"])
        except:
            pass
    
    if user_id < 0:
        max_user_id = 0
        if os.path.exists(ANNOT_FILE):
            with open(ANNOT_FILE, "r") as f:
                for line in f:
                    obj = json.loads(line)
                    max_user_id = max(max_user_id, obj.get("user_id", -1))
        user_id = max_user_id + 1    
    flask.request.user_id = user_id

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.set_cookie("user_id", value=str(flask.request.user_id), expires=datetime.now() + timedelta(days=365))
    return response

@app.route("/")
def api_home_page():
    return render_template("main_page.html")

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route("/api/load_documents")
def api_load_document():
    with open(CONTENT_FILE, "r") as f:
        data = json.load(f)
    return {"documents": data}

@app.route("/api/gen_questions", methods=["POST"])
def api_gen_questions():
    request_data = dict(request.form)

    doc_id = int(request_data["doc_id"])
    context = request_data["context"]
    answer_span = request_data["selection"]

    paragraphs = context.split("<br />")
    relevant_paragraphs = [p for p in paragraphs if answer_span in p]
    if len(relevant_paragraphs) == 0:
        return []
    else:
        question_key = "%d||%s" % (doc_id, answer_span)
        if question_key not in cached_questions:
            relevant_paragraph = relevant_paragraphs[0]
            response = []
            for model in QGEN_MODELS:
                marked_paragraph = mark_paragraph_answer(relevant_paragraph, answer_span, model_card=model["model"].model_card)

                questions = model["model"].generate([marked_paragraph], max_gen_length=30, num_beams=2)[0]
                question = questions[0]["output_text"]
                question = question[0].upper() + question[1:]
                response.append({"model_name": model["model_name"], "question": question})

            response = deduplicate_questions(response)
            cached_questions[question_key] = response
            save_question_cache()
        else:
            print("Reloaded from the cache")

        response = cached_questions[question_key]

        random.shuffle(response)
        return {"response": response}

@app.route("/api/annotate_questions", methods=["POST"])
def api_annotate_questions():
    request_data = dict(request.form)
    request_data["questions"] = json.loads(request_data["questions"].strip())

    ip_addr = request.remote_addr
    saved_object = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "ip_addr": ip_addr}
    saved_object["user_id"] = request.user_id
    saved_object["doc_id"] = request_data["doc_id"]
    saved_object["answer_span"] = request_data["answer_span"]
    saved_object["answer_span_idx"] = request_data["answer_span_idx"]
    saved_object["questions"] = request_data["questions"]
    saved_object["annotator_name"] = request_data["annotator_name"]

    with open(ANNOT_FILE, "a") as f:
        f.write(json.dumps(saved_object) + "\n")
    return {"response": 1}

@app.route("/api/cancel_selection", methods=["POST"])
def api_cancel_selection():
    request_data = dict(request.form)

    print("Delete request: ", request_data["doc_id"], request_data["answer_span"], request_data["annotator_name"])
    if os.path.exists(ANNOT_FILE):
        final_annotations = []
        num_deleted = 0
        with open(ANNOT_FILE, "r") as f:
            for line in f:
                obj = json.loads(line)
                if obj["doc_id"] == request_data["doc_id"] and obj["answer_span"] == request_data["answer_span"] and obj["annotator_name"] == request_data["annotator_name"]:
                    num_deleted += 1
                else:
                    final_annotations.append(obj)

        print("Num rows deleted: %d" % (num_deleted))
        with open(ANNOT_FILE, "w") as f:
            for obj in final_annotations:
                f.write(json.dumps(obj) + "\n")

    return {"response": 1}
