# MixQG: Neural Question Generation with Mixed Answer Types

This is the official code base for the following paper from Salesforce Research:

**Title**: Quiz Design Task: Helping Teachers Create Quizzes with Automated Question Generation

**Authors**: Philippe Laban, Chien-Sheng Wu, Lidiya Murakhovs'ka, Wenhao Liu, Caiming Xiong

## Dataset Release:

We release the dataset we collected during the study: `quiz_design_data.jsonl`. It can be opened in Python with the following:

```python
import utils_qd_data
annotations = utils_qd_data.load_qd_annotations()
qd_dataset = utils_qd_data.build_qd_groups(annotations)
```

Each line is a JSON object. The first entry looks like:
```json
{"doc_id": 0,
 "answer_span": "meets the needs of the present without compromising the ability of future generations to meet their own needs",
 "context": "Energy is sustainable if it "meets the needs of the present without compromising the ability of future generations to meet their own needs".  Most definitions of sustainable energy [...]",
 "questions": [{"question": "What does energy mean if it is sustainable?",
   "label": 0,
   "reason": "disfluent",
   "model_name": "dgpt2_sup"},
  {"question": "What does energy sustainability mean?",
   "label": 1,
   "reason": "No error",
   "model_name": "gpt2b_sup"},
  {"question": "How is energy sustainable?",
   "label": 0,
   "reason": "wrong_context",
   "model_name": "gpt2m_sup"},
  {"question": "What is sustainable energy?",
   "label": 0,
   "reason": "wrong_context",
   "model_name": "bartb_sup|prophetnet"},
  {"question": "What does it mean if energy is sustainable?",
   "label": 1,
   "reason": "No error",
   "model_name": "mixqg"},
  {"question": "What is the definition of sustainable energy?",
   "label": 1,
   "reason": "No error",
   "model_name": "bartl_sup"}]}
```

## Annotation Interface

We release the annotation interface used during the collection of the Quiz Design study.
The interface can instantiated with the following command:
```
FLASK_APP=run_flask_server flask run
```

The list of Question Generation models used to generate candidate questions can be modified in the first lines of `run_flask_server.py`.

## Cite the work

If you use the data or annotation interface, please cite the work:
```
@inproceedings{laban2022quiz,
  title={Quiz Design Task: Helping Teachers Create Quizzes with Automated Question Generation},
  author={Laban, Philippe and Wu, Chien-Sheng and Murakhovs'ka, Lidiya and Liu, Wenhao and Xiong, Caiming},
  booktitle={Findings of the North American Chapter of the Association for Computational Linguistics: NAACL 2022},
  year={2022}
}
```