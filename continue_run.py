"""Run only the remaining iterations (4 and 5) and append to run_50q.json."""
import json, os, random
import numpy as np
from src.models import create_model
from src.utils import load_beir_datasets, load_json, setup_seeds, clean_str, f1_score
from src.prompts import wrap_prompt
from src.attack import Attacker
import argparse

RESULT_FILE = "results/query_results/main/run_50q.json"
ADV_FILE    = "results/adv_targeted_results/nq.json"
BEIR_FILE   = "results/beir_results/nq-contriever.json"
MODEL_CFG   = "model_configs/openai_gpt4o_config.json"
M           = 10
TOP_K       = 5
ADV_PER_Q   = 5
SEED        = 12
START_ITER  = 3   # 0-indexed: we already have iters 0,1,2
TOTAL_ITERS = 5

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

setup_seeds(SEED)

corpus, queries, qrels = load_beir_datasets("nq", "test")
incorrect_answers = list(load_json(ADV_FILE).values())
with open(BEIR_FILE) as f:
    results = json.load(f)

with open(RESULT_FILE) as f:
    all_results = json.load(f)

args = argparse.Namespace(
    eval_model_code="contriever", eval_dataset="nq", split="test",
    attack_method="LM_targeted", adv_per_query=ADV_PER_Q,
    score_function="dot", M=M, top_k=TOP_K, seed=SEED,
    repeat_times=TOTAL_ITERS, name="run_50q", query_results_dir="main"
)
attacker = Attacker(args)
llm = create_model(MODEL_CFG)

asr_list = []

for iter in range(START_ITER, TOTAL_ITERS):
    print(f'\n######################## Iter: {iter+1}/{TOTAL_ITERS} #######################')
    target_queries_idx = range(iter * M, iter * M + M)

    target_queries = [incorrect_answers[idx]['question'] for idx in target_queries_idx]
    for i in target_queries_idx:
        top1_idx   = list(results[incorrect_answers[i]['id']].keys())[0]
        top1_score = results[incorrect_answers[i]['id']][top1_idx]
        target_queries[i - iter * M] = {
            'query': target_queries[i - iter * M],
            'top1_score': top1_score,
            'id': incorrect_answers[i]['id']
        }

    adv_text_groups = attacker.get_attack(target_queries)

    asr_cnt = 0
    iter_results = []

    for i in target_queries_idx:
        iter_idx = i - iter * M
        question = incorrect_answers[i]['question']
        incco_ans = incorrect_answers[i]['incorrect answer']
        print(f'############# Target Question: {iter_idx+1}/{M} #############')
        print(f'Question: {question}\n')

        topk_idx = list(results[incorrect_answers[i]['id']].keys())[:TOP_K]
        topk_results = [{'score': results[incorrect_answers[i]['id']][idx], 'context': corpus[idx]['text']} for idx in topk_idx]

        adv_text_set = set(adv_text_groups[iter_idx])
        adv_entries  = [{'score': 999.0, 'context': t} for t in adv_text_groups[iter_idx]]
        topk_results = adv_entries + topk_results

        topk_contents = [topk_results[j]["context"] for j in range(TOP_K)]
        query_prompt  = wrap_prompt(question, topk_contents, prompt_id=4)
        response      = llm.query(query_prompt)

        print(f'Output: {response}\n')
        if clean_str(incco_ans) in clean_str(response):
            asr_cnt += 1

        injected_adv = [t for t in topk_contents if t in adv_text_set]
        iter_results.append({
            "id": incorrect_answers[i]['id'],
            "question": question,
            "injected_adv": injected_adv,
            "input_prompt": query_prompt,
            "output_poison": response,
            "incorrect_answer": incco_ans,
            "answer": incorrect_answers[i]['correct answer']
        })

    asr_list.append(asr_cnt)
    all_results.append({f'iter_{iter}': iter_results})
    save_json(all_results, RESULT_FILE)
    print(f'Saved iter {iter+1} → {RESULT_FILE}  (ASR this iter: {asr_cnt}/{M})')

print("\n=== DONE — All 50 questions complete ===")
asr_arr = np.array([asr_cnt / M for asr_cnt in asr_list])
print(f"Iter 4-5 ASR: {[round(a,2) for a in asr_arr]}")
