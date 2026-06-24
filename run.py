import os
import subprocess
import sys

def run(test_params):

    log_file, log_name = get_log_name(test_params)

    cmd = [
        sys.executable, "-u", "main.py",
        "--eval_model_code", str(test_params['eval_model_code']),
        "--eval_dataset", str(test_params['eval_dataset']),
        "--split", str(test_params['split']),
        "--query_results_dir", str(test_params['query_results_dir']),
        "--model_name", str(test_params['model_name']),
        "--top_k", str(test_params['top_k']),
        "--use_truth", str(test_params['use_truth']),
        "--gpu_id", str(test_params['gpu_id']),
        "--attack_method", str(test_params['attack_method']),
        "--adv_per_query", str(test_params['adv_per_query']),
        "--score_function", str(test_params['score_function']),
        "--repeat_times", str(test_params['repeat_times']),
        "--M", str(test_params['M']),
        "--seed", str(test_params['seed']),
        "--name", log_name,
    ]

    with open(log_file, "w", encoding="utf-8") as file:
        subprocess.Popen(cmd, stdout=file, stderr=subprocess.STDOUT)


def get_log_name(test_params):
    # Generate a log file name
    os.makedirs(f"logs/{test_params['query_results_dir']}_logs", exist_ok=True)

    if test_params['use_truth']:
        log_name = f"{test_params['eval_dataset']}-{test_params['eval_model_code']}-{test_params['model_name']}-Truth--M{test_params['M']}x{test_params['repeat_times']}"
    else:
        log_name = f"{test_params['eval_dataset']}-{test_params['eval_model_code']}-{test_params['model_name']}-Top{test_params['top_k']}--M{test_params['M']}x{test_params['repeat_times']}"
    
    if test_params['attack_method'] != None:
        log_name += f"-adv-{test_params['attack_method']}-{test_params['score_function']}-{test_params['adv_per_query']}-{test_params['top_k']}"

    if test_params['note'] != None:
        log_name = test_params['note']
    
    return f"logs/{test_params['query_results_dir']}_logs/{log_name}.txt", log_name



test_params = {
    # beir_info
    'eval_model_code': "contriever",
    'eval_dataset': "nq",
    'split': "test",
    'query_results_dir': 'main',

    # LLM setting
    'model_name': 'openai_gpt4o',
    'use_truth': False,
    'top_k': 5,
    'gpu_id': 0,

    # attack
    'attack_method': 'LM_targeted',
    'adv_per_query': 5,
    'score_function': 'dot',
    'repeat_times': 5,
    'M': 10,
    'seed': 12,

    'note': 'run_50q'
}

for dataset in ['nq']:
    test_params['eval_dataset'] = dataset
    run(test_params)
