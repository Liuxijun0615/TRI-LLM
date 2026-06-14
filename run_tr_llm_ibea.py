# run_tr_llm_ibea.py
from Problems import RCBench
from Algorithms.TR_LLM_IBEA import TR_LLM_IBEA
import argparse
import json
import time
import pickle
import config
import os

parser = argparse.ArgumentParser(description="Run TRI-LLM diagnostics and ablations.")
parser.add_argument("--ablation-mode", choices=["full", "wo_ibea", "wo_str"], default=None)
parser.add_argument("--enable-diagnostics", dest="enable_diagnostics", action="store_true", default=None)
parser.add_argument("--disable-diagnostics", dest="enable_diagnostics", action="store_false")
parser.add_argument("--diag-save-plots", dest="diag_save_plots", action="store_true", default=None)
parser.add_argument("--no-diag-save-plots", dest="diag_save_plots", action="store_false")
parser.add_argument("--seed", type=int, default=None)
args = parser.parse_args()

if args.ablation_mode is not None:
    config.ABLATION_MODE = args.ablation_mode
if args.enable_diagnostics is not None:
    config.ENABLE_DIAGNOSTICS = args.enable_diagnostics
if args.diag_save_plots is not None:
    config.DIAG_SAVE_PLOTS = args.diag_save_plots
if args.seed is not None:
    config.SEED = args.seed

print(f"Start TR-LLM-IBEA | Mode={config.ABLATION_MODE}")
print("=" * 60)


time_record = {}

for setting in config.DATA_OBJECTIVES:
    dataset, objectives = setting[0], setting[1]

    print(f"\n current experiment: {dataset} - {objectives}")
    print("-" * 40)


    func = eval(f'RCBench.{objectives}')


    dataset_path = config.get_dataset_path(dataset, config.SEED, "train")
    try:
        with open(dataset_path, 'r', encoding='utf-8') as json_file:
            train_data = json.load(json_file)
        print(f"data loading successful: {dataset_path}")
    except Exception as e:
        print(f"data loading fail: {e}")
        continue


    bench = func(
        train_data,
        config.BATCH_SIZE,
        config.OPENAI_KEY,
        llm_model=config.LLM_MODEL
    )


    algorithm_label = f"TR-LLM-IBEA_{config.ABLATION_MODE}"
    detailed_dir, compatibility_path = config.get_save_path(
        algorithm_label, dataset, objectives, config.SEED
    )

    save_target_dir = detailed_dir

    print(f"save result: {save_target_dir}")

    # 进化优化
    print("start")
    start_time = time.time()

    try:
        Pop, Obj = TR_LLM_IBEA(
            problem=bench,
            max_iter=config.MAX_ITERATIONS,
            pop_size=config.POPULATION_SIZE,
            api_key=config.OPENAI_KEY,
            llm_model=config.LLM_MODEL,
            save_path=save_target_dir,
            re_evaluate_parents = False,
            test_batch_size = config.TEST_BATCH_SIZE
        )
        end_time = time.time()

        experiment_time = end_time - start_time
        time_record[f"{dataset} & {objectives}"] = experiment_time

        print(f"complete time: {experiment_time / 60:.2f}mins")

    except Exception as e:
        print(f"fail: {e}")
        import traceback
        traceback.print_exc()
        time_record[f"{dataset} & {objectives}"] = -1

try:
    time_file = os.path.join(save_target_dir, f"TimeConsumption.pkl")
    pickle.dump(time_record, open(time_file, "wb"))
except:
    pass

print("\n" + "=" * 60)
print("all complete")
