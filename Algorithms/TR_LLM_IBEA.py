from EA_Operators.LLM_EA import LLM_EA

import hashlib
import json
import os
import pickle
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import config
import Utils.hypervolume as hv_utils
from Utils.diagnostics import (
    DiagnosticsRecorder,
    compute_crowding_distance_all_maximize,
    compute_pareto_ranks_maximize,
    nsga2_environment_selection_maximize,
    select_parent_index,
    semantic_drift,
)


class TR_LLM_IBEA_Algorithm:
    def __init__(self, problem, pop_size, api_key, llm_model):
        self.problem = problem
        self.pop_size = pop_size
        self.llm_ea = LLM_EA(pop_size, llm_model, api_key)

        self.ablation_mode = getattr(config, "ABLATION_MODE", "full")
        if self.ablation_mode not in {"full", "wo_ibea", "wo_str"}:
            raise ValueError("ABLATION_MODE must be one of: full, wo_ibea, wo_str")

        # Adaptive semantic trust-region configuration.
        self.trust_radius_map = {}
        self.initial_radius = "L2"
        self.rho_threshold_high = 0.75
        self.rho_threshold_low = 0.0
        self.expected_gain_baseline = 0.01

        self.metrics_log = []
        self.uid_counter = 0
        self.diagnostics = None

    def _new_uid(self, prefix):
        self.uid_counter += 1
        return f"{prefix}_{self.uid_counter:06d}"

    @staticmethod
    def _prompt_hash(prompt):
        return hashlib.md5(str(prompt).encode("utf-8")).hexdigest()

    def _set_radius(self, uid, prompt, radius):
        self.trust_radius_map[uid] = radius
        self.trust_radius_map[prompt] = radius

    def _get_radius(self, uid, prompt):
        if self.ablation_mode == "wo_str":
            return "NoSTR"
        return self.trust_radius_map.get(uid, self.trust_radius_map.get(prompt, self.initial_radius))

    def get_trust_instruction(self, level):
        if level == "L1":
            return (
                "You are in a RESTRICTED mode. STRICTLY maintain the original structure. "
                "ONLY fine-tune adjectives, parameters, or synonyms. DO NOT alter the logic flow."
            )
        if level == "L2":
            return (
                "You are in a BALANCED mode. Rephrase the prompt to enhance clarity. "
                "You may reorganize sentences but MUST preserve the core semantic intent and strategy."
            )
        if level == "L3":
            return (
                "You are in an EXPANDED mode. Brainstorm a COMPLETELY NEW strategy. "
                "Ignore the original structure. Try novel perspectives."
            )
        return "You are in a BALANCED mode."

    def calculate_hv(self, population_objs):
        objs = np.asarray(population_objs, dtype=float)
        if objs.ndim == 1:
            objs = objs.reshape(1, -1)
        if len(objs) == 0:
            return 0.0
        objs = objs[np.all(np.isfinite(objs), axis=1)]
        if len(objs) == 0:
            return 0.0
        ref_point = np.full(objs.shape[1], -0.01)
        return float(hv_utils.calculate_hypervolume(objs, ref_point))

    def calculate_hv_contribution(self, population_objs, new_ind_obj):
        new_ind_obj = np.asarray(new_ind_obj, dtype=float)
        if new_ind_obj.ndim != 1 or not np.all(np.isfinite(new_ind_obj)):
            return np.nan
        current_hv = self.calculate_hv(population_objs)
        population_objs = np.asarray(population_objs, dtype=float)
        if population_objs.ndim == 1:
            population_objs = population_objs.reshape(1, -1)
        if len(population_objs) > 0:
            new_pop_objs = np.vstack([population_objs, new_ind_obj])
        else:
            new_pop_objs = np.array([new_ind_obj])
        return float(self.calculate_hv(new_pop_objs) - current_hv)

    def calc_ibea_fitness(self, objs):
        """Indicator fitness for maximization; larger fitness wins tournaments."""
        objs = np.asarray(objs, dtype=float)
        if objs.ndim == 1:
            objs = objs.reshape(1, -1)
        n = len(objs)
        if n == 0:
            return np.array([])

        objs = np.nan_to_num(objs, nan=-1e12, posinf=1e12, neginf=-1e12)
        kappa = getattr(config, "IBEA_KAPPA", 0.05)
        min_vals = np.min(objs, axis=0)
        max_vals = np.max(objs, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1e-6
        norm_objs = (objs - min_vals) / range_vals

        fitness = np.zeros(n)
        for i in range(n):
            sum_val = 0.0
            for j in range(n):
                if i != j:
                    indicator = np.max(norm_objs[i] - norm_objs[j])
                    sum_val += -np.exp(-indicator / kappa)
            fitness[i] = sum_val
        return fitness

    def ibea_environment_selection_maximize(self, pop, objs, pop_size):
        """IBEA environmental selection for maximization with selected-index tracking."""
        work_pop = list(pop)
        work_objs = np.asarray(objs, dtype=float).copy()
        work_indices = list(range(len(work_pop)))

        while len(work_pop) > pop_size:
            fitness = self.calc_ibea_fitness(work_objs)
            worst_idx = int(np.argmin(fitness))
            work_pop.pop(worst_idx)
            work_objs = np.delete(work_objs, worst_idx, axis=0)
            work_indices.pop(worst_idx)

        return work_pop, work_objs, work_indices

    def _make_diagnostics(self, save_dir):
        self.diagnostics = DiagnosticsRecorder(
            save_dir=save_dir,
            method=self.ablation_mode,
            enabled=getattr(config, "ENABLE_DIAGNOSTICS", True),
            save_raw=getattr(config, "DIAG_SAVE_RAW", True),
            save_plots=getattr(config, "DIAG_SAVE_PLOTS", True),
            semantic_backend=getattr(config, "DIAG_SEMANTIC_BACKEND", "token_jaccard"),
            fitness_tie_rel=getattr(config, "DIAG_FITNESS_TIE_REL", 0.01),
        )

    def save_generation_data(self, save_dir, gen_idx, pop, y_pop, pop_uids, pareto_indices, offspring_data=None):
        pop_data = []
        for i, (uid, prompt, objs) in enumerate(zip(pop_uids, pop, y_pop)):
            radius = self._get_radius(uid, prompt)
            item = {
                "ID": i,
                "UID": uid,
                "Prompt_Hash": self._prompt_hash(prompt),
                "Type": "Survivor",
                "Trust_Radius": radius,
                "Prompt": str(prompt).replace("\n", "\\n"),
            }
            for obj_i, obj_val in enumerate(objs):
                item[f"Objective_{obj_i + 1}"] = obj_val
            pop_data.append(item)

        df_pop = pd.DataFrame(pop_data)
        if not df_pop.empty:
            df_pop.to_csv(
                os.path.join(save_dir, f"population_iter_{gen_idx:02d}.csv"),
                index=False,
                encoding="utf-8-sig",
            )
        if len(pareto_indices) > 0 and not df_pop.empty:
            df_pop.iloc[pareto_indices].to_csv(
                os.path.join(save_dir, f"pareto_front_iter_{gen_idx:02d}.csv"),
                index=False,
                encoding="utf-8-sig",
            )
        if offspring_data:
            pd.DataFrame(offspring_data).to_csv(
                os.path.join(save_dir, f"offspring_iter_{gen_idx:02d}.csv"),
                index=False,
                encoding="utf-8-sig",
            )

        with open(os.path.join(save_dir, f"checkpoint_iter_{gen_idx:02d}.pkl"), "wb") as f:
            pickle.dump(
                {
                    "Iteration": gen_idx,
                    "Population": pop,
                    "Population_UIDs": pop_uids,
                    "Reward": y_pop,
                    "Trust_Radius_Map": self.trust_radius_map,
                    "Ablation_Mode": self.ablation_mode,
                },
                f,
            )

    def _record_metrics(self, gen_idx, current_hv, y_pop, pop, pop_uids, start_time_global):
        avg_objs = np.mean(y_pop, axis=0)
        max_objs = np.max(y_pop, axis=0)
        radii = [self._get_radius(uid, prompt) for uid, prompt in zip(pop_uids, pop)]
        log_entry = {
            "Generation": gen_idx,
            "Ablation_Mode": self.ablation_mode,
            "Hypervolume": current_hv,
            "Time_Elapsed": time.time() - start_time_global,
            "Total_Tokens": self.llm_ea.total_tokens,
            "Radius_L1": radii.count("L1"),
            "Radius_L2": radii.count("L2"),
            "Radius_L3": radii.count("L3"),
            "NoSTR": radii.count("NoSTR"),
        }
        for i, val in enumerate(avg_objs):
            log_entry[f"Avg_Obj_{i + 1}"] = float(val)
        for i, val in enumerate(max_objs):
            log_entry[f"Max_Obj_{i + 1}"] = float(val)
        self.metrics_log.append(log_entry)
        print(
            "   Stats: "
            f"HV={current_hv:.4f} | "
            f"Radius L1={radii.count('L1')}, L2={radii.count('L2')}, "
            f"L3={radii.count('L3')}, NoSTR={radii.count('NoSTR')}"
        )

    def save_summary_logs(self, save_dir):
        with open(os.path.join(save_dir, "metrics_history.json"), "w", encoding="utf-8") as f:
            json.dump(self.metrics_log, f, ensure_ascii=False, indent=2)
        pd.DataFrame(self.metrics_log).to_csv(
            os.path.join(save_dir, "metrics_summary.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    def run_final_test(self, save_dir, pop, test_batch_size):
        print(f"\n{'=' * 50}")
        print(f"STARTING FINAL TEST (Batch Size: {test_batch_size})")
        print(f"{'=' * 50}")

        test_batch_size = int(test_batch_size)
        if test_batch_size <= 0:
            summary = {
                "Test_Batch_Size": test_batch_size,
                "Final_Test_HV": 0.0,
                "Pareto_Solutions_Count": 0,
                "Skipped": True,
                "Reason": "TEST_BATCH_SIZE <= 0",
                "Test_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            summary_path = os.path.join(save_dir, "final_test_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print("Final test skipped because TEST_BATCH_SIZE <= 0.")
            return 0.0

        original_batch_num = self.problem.batch_num
        self.problem.batch_num = test_batch_size

        print(f"   Resampling {test_batch_size} samples from training data...")
        self.problem.Sample_Test_Data()

        print("   Evaluating final population on TEST batch...")
        test_y = self.problem.Evaluate(pop)
        _, test_fronts = compute_pareto_ranks_maximize(test_y)
        test_pareto_indices = test_fronts[0] if len(test_fronts) > 0 else []
        test_hv = self.calculate_hv(test_y)
        print(f"   Final Test HV: {test_hv:.4f}")

        test_data = []
        for i, (prompt, objs) in enumerate(zip(pop, test_y)):
            item = {
                "ID": i,
                "Is_Pareto": i in test_pareto_indices,
                "Prompt": str(prompt).replace("\n", "\\n"),
            }
            for obj_i, obj_val in enumerate(objs):
                item[f"Test_Objective_{obj_i + 1}"] = obj_val
            test_data.append(item)

        df_test = pd.DataFrame(test_data)
        df_test.to_csv(os.path.join(save_dir, "final_test_results.csv"), index=False, encoding="utf-8-sig")
        if not df_test.empty:
            df_test[df_test["Is_Pareto"] == True].to_csv(
                os.path.join(save_dir, "final_test_pareto.csv"),
                index=False,
                encoding="utf-8-sig",
            )

        final_summary = {
            "Test_Batch_Size": test_batch_size,
            "Final_Test_HV": float(test_hv),
            "Pareto_Solutions_Count": len(test_pareto_indices),
            "Test_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Average_Objectives": np.mean(test_y, axis=0).tolist(),
            "Max_Objectives": np.max(test_y, axis=0).tolist(),
        }

        summary_path = os.path.join(save_dir, "final_test_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(final_summary, f, ensure_ascii=False, indent=2)
        print(f"   Final test summary saved to: {summary_path}")

        self.problem.batch_num = original_batch_num
        return test_hv

    def _record_initial_diagnostics(self, y_pop):
        if self.diagnostics and self.diagnostics.enabled:
            ibea_fitness = self.calc_ibea_fitness(y_pop) if self.ablation_mode == "full" else None
            self.diagnostics.aggregate_selection(0, y_pop, ibea_fitness=ibea_fitness)
            self.diagnostics.aggregate_mutation(0)
            self.diagnostics.save()

    def _record_mutation_diagnostic(
        self,
        gen_idx,
        child_uid,
        parent_uid,
        parent_idx,
        parent_prompt,
        child_prompt,
        meta,
        radius_before,
        radius_after,
        y_pop,
        y_child,
        delta_hv,
        rho,
    ):
        if not (self.diagnostics and self.diagnostics.enabled):
            return

        prompt_parse_valid = bool(meta.get("parsed_ok")) and not bool(meta.get("fallback_used"))
        evaluation_valid = bool(np.all(np.isfinite(y_child)))
        valid_for_emr = prompt_parse_valid and evaluation_valid
        effective = bool(valid_for_emr and np.isfinite(delta_hv) and delta_hv > 0)
        operator = "Mutation(NoSTR)" if self.ablation_mode == "wo_str" else f"Mutation({radius_before})"

        item = {
            "Generation": gen_idx,
            "Method": self.ablation_mode,
            "Offspring_UID": child_uid,
            "Parent_UID": parent_uid,
            "Parent_Index": parent_idx,
            "Operator": operator,
            "Radius_Before": radius_before,
            "Radius_After": radius_after,
            "Parent_Prompt": str(parent_prompt).replace("\n", "\\n"),
            "Child_Prompt": str(child_prompt).replace("\n", "\\n"),
            "Prompt_Parse_Valid": prompt_parse_valid,
            "Evaluation_Valid": evaluation_valid,
            "Valid_For_EMR": valid_for_emr,
            "Delta_HV": delta_hv,
            "Rho": rho,
            "Effective_Mutation": effective,
            "Ineffective_Mutation": not effective,
            "Semantic_Drift": semantic_drift(
                parent_prompt,
                child_prompt,
                backend=getattr(config, "DIAG_SEMANTIC_BACKEND", "token_jaccard"),
            ),
            "Survived_To_Next_Generation": False,
        }
        for obj_i, obj_val in enumerate(y_child):
            item[f"Objective_{obj_i + 1}"] = obj_val

        self.diagnostics.record_mutation(item)

    def _mutation_child(self, gen_idx, parent_idx, parent_uid, parent_prompt, y_pop):
        if self.ablation_mode == "wo_str":
            radius_before = "NoSTR"
            constraint = getattr(config, "NO_STR_MUTATION_CONSTRAINT")
        else:
            radius_before = self._get_radius(parent_uid, parent_prompt)
            constraint = self.get_trust_instruction(radius_before)

        child_prompt, meta = self.llm_ea.trust_region_mutation(parent_prompt, constraint, return_meta=True)
        child_uid = self._new_uid("offspring")
        y_child = self.problem.Evaluate([child_prompt])[0]
        evaluation_valid = bool(np.all(np.isfinite(y_child)))
        delta_hv = self.calculate_hv_contribution(y_pop, y_child) if evaluation_valid else np.nan
        rho = delta_hv / self.expected_gain_baseline if np.isfinite(delta_hv) else np.nan

        if self.ablation_mode == "wo_str":
            radius_after = "NoSTR"
        else:
            radius_after = radius_before
            if np.isfinite(rho) and rho > self.rho_threshold_high:
                radius_after = "L2" if radius_before == "L1" else "L3"
            elif np.isfinite(rho) and rho <= self.rho_threshold_low:
                radius_after = "L2" if radius_before == "L3" else "L1"
            self._set_radius(child_uid, child_prompt, radius_after)

        self._record_mutation_diagnostic(
            gen_idx,
            child_uid,
            parent_uid,
            parent_idx,
            parent_prompt,
            child_prompt,
            meta,
            radius_before,
            radius_after,
            y_pop,
            y_child,
            delta_hv,
            rho,
        )

        print(
            f"   Mut({radius_before}) -> DeltaHV={delta_hv:.5f} | "
            f"Rho={rho:.2f} | Next={radius_after}"
        )
        return child_uid, child_prompt, y_child, {
            "Operator": "Mutation(NoSTR)" if self.ablation_mode == "wo_str" else f"Mutation({radius_before})",
            "Radius_Change": f"{radius_before}->{radius_after}",
            "Gain": delta_hv,
            "Rho": rho,
            "Prompt_Parse_Valid": bool(meta.get("parsed_ok")) and not bool(meta.get("fallback_used")),
        }

    def _crossover_child(self, parent1, parent2):
        child_prompt, meta = self.llm_ea.single_crossover(parent1, parent2, return_meta=True)
        child_uid = self._new_uid("offspring")
        if self.ablation_mode != "wo_str":
            self._set_radius(child_uid, child_prompt, self.initial_radius)
            radius_change = f"Reset->{self.initial_radius}"
        else:
            radius_change = "NoSTR"
        y_child = self.problem.Evaluate([child_prompt])[0]
        return child_uid, child_prompt, y_child, {
            "Operator": "Crossover",
            "Radius_Change": radius_change,
            "Gain": 0.0,
            "Rho": 0.0,
            "Prompt_Parse_Valid": bool(meta.get("parsed_ok")) and not bool(meta.get("fallback_used")),
        }

    def run(self, max_iter, save_path, re_evaluate_parents=False, test_batch_size=None):
        save_dir = save_path
        os.makedirs(save_dir, exist_ok=True)
        self._make_diagnostics(save_dir)

        if test_batch_size is None:
            test_batch_size = getattr(config, "TEST_BATCH_SIZE", self.problem.batch_num)

        print(
            f"TR-LLM-IBEA Started! Mode={self.ablation_mode}, "
            f"Pop={self.pop_size}, Iter={max_iter}, TestBS={test_batch_size}"
        )
        start_time_global = time.time()

        print("Initializing population...")
        pop = self.llm_ea.initialize(config.EXAMPLE_PROMPT)
        pop_uids = [self._new_uid("init") for _ in pop]
        for uid, prompt in zip(pop_uids, pop):
            if self.ablation_mode != "wo_str":
                self._set_radius(uid, prompt, self.initial_radius)

        print("Evaluating initial population...")
        self.problem.Sample_Test_Data()
        y_pop = self.problem.Evaluate(pop)

        current_hv = self.calculate_hv(y_pop)
        _, fronts = compute_pareto_ranks_maximize(y_pop)
        pareto_indices = fronts[0] if len(fronts) > 0 else []
        self.save_generation_data(save_dir, 0, pop, y_pop, pop_uids, pareto_indices)
        self._record_metrics(0, current_hv, y_pop, pop, pop_uids, start_time_global)
        self.save_summary_logs(save_dir)
        self._record_initial_diagnostics(y_pop)

        for iter_idx in range(max_iter):
            gen_idx = iter_idx + 1
            print(f"\n{'=' * 40}\nGeneration {gen_idx}/{max_iter}\n{'=' * 40}")

            self.problem.Sample_Test_Data()
            if re_evaluate_parents:
                y_pop = self.problem.Evaluate(pop)

            parent_y_pop = np.asarray(y_pop, dtype=float).copy()
            pareto_ranks, fronts = compute_pareto_ranks_maximize(parent_y_pop)
            crowding_distances = compute_crowding_distance_all_maximize(parent_y_pop, fronts)
            ibea_fitness = None if self.ablation_mode == "wo_ibea" else self.calc_ibea_fitness(parent_y_pop)

            offspring_pop = []
            offspring_uids = []
            y_offspring_pop = []
            offspring_details = []

            for _ in range(self.pop_size):
                p1_idx = select_parent_index(
                    parent_y_pop,
                    mode=self.ablation_mode,
                    ibea_fitness=ibea_fitness,
                    pareto_ranks=pareto_ranks,
                    crowding_distances=crowding_distances,
                    diagnostics_recorder=self.diagnostics,
                    generation=gen_idx,
                    role="parent1",
                )
                parent1 = pop[p1_idx]
                parent1_uid = pop_uids[p1_idx]

                detail_item = {
                    "Gen": gen_idx,
                    "Parent_Index": p1_idx,
                    "Parent_UID": parent1_uid,
                    "Parent_Prompt": parent1,
                }

                if np.random.rand() < 0.5:
                    p2_idx = select_parent_index(
                        parent_y_pop,
                        mode=self.ablation_mode,
                        ibea_fitness=ibea_fitness,
                        pareto_ranks=pareto_ranks,
                        crowding_distances=crowding_distances,
                        diagnostics_recorder=self.diagnostics,
                        generation=gen_idx,
                        role="parent2",
                    )
                    parent2 = pop[p2_idx]
                    child_uid, child_prompt, y_child, child_meta = self._crossover_child(parent1, parent2)
                    detail_item.update({"Parent2_Index": p2_idx, "Parent2_UID": pop_uids[p2_idx]})
                    detail_item.update(child_meta)
                else:
                    child_uid, child_prompt, y_child, child_meta = self._mutation_child(
                        gen_idx, p1_idx, parent1_uid, parent1, parent_y_pop
                    )
                    detail_item.update(child_meta)

                detail_item["Offspring_UID"] = child_uid
                detail_item["Prompt_Hash"] = self._prompt_hash(child_prompt)
                detail_item["Prompt"] = str(child_prompt).replace("\n", "\\n")
                for obj_i, obj_val in enumerate(y_child):
                    detail_item[f"Objective_{obj_i + 1}"] = obj_val
                offspring_details.append(detail_item)

                offspring_uids.append(child_uid)
                offspring_pop.append(child_prompt)
                y_offspring_pop.append(y_child)

            if self.diagnostics and self.diagnostics.enabled:
                self.diagnostics.aggregate_selection(gen_idx, parent_y_pop, ibea_fitness=ibea_fitness)

            y_offspring_array = np.asarray(y_offspring_pop, dtype=float)
            all_pop = pop + offspring_pop
            all_uids = pop_uids + offspring_uids
            all_y = np.concatenate((y_pop, y_offspring_array), axis=0)

            if self.ablation_mode == "wo_ibea":
                next_pop, next_y, selected_indices = nsga2_environment_selection_maximize(
                    all_pop, all_y, self.pop_size
                )
            else:
                next_pop, next_y, selected_indices = self.ibea_environment_selection_maximize(
                    all_pop, all_y, self.pop_size
                )
            next_uids = [all_uids[i] for i in selected_indices]

            pop = next_pop
            pop_uids = next_uids
            y_pop = next_y

            if self.diagnostics and self.diagnostics.enabled:
                self.diagnostics.update_mutation_survival(gen_idx, set(pop_uids))
                self.diagnostics.aggregate_mutation(gen_idx)
                self.diagnostics.save()

            current_hv = self.calculate_hv(y_pop)
            _, fronts = compute_pareto_ranks_maximize(y_pop)
            pareto_indices = fronts[0] if len(fronts) > 0 else []

            self.save_generation_data(
                save_dir,
                gen_idx,
                pop,
                y_pop,
                pop_uids,
                pareto_indices,
                offspring_data=offspring_details,
            )
            self._record_metrics(gen_idx, current_hv, y_pop, pop, pop_uids, start_time_global)
            self.save_summary_logs(save_dir)

        final_test_hv = self.run_final_test(save_dir, pop, test_batch_size)
        if self.diagnostics and self.diagnostics.enabled:
            self.diagnostics.save()

        print(f"All Done! Validation HV: {current_hv:.4f} | Final Test HV: {final_test_hv:.4f}")
        return pop, y_pop


def TR_LLM_IBEA(
    problem,
    max_iter,
    pop_size,
    api_key,
    llm_model,
    save_path,
    re_evaluate_parents=False,
    test_batch_size=None,
):
    algorithm = TR_LLM_IBEA_Algorithm(problem, pop_size, api_key, llm_model)
    return algorithm.run(max_iter, save_path, re_evaluate_parents, test_batch_size)
