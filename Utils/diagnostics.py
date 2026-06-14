import json
import math
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd


def _as_2d_array(objs):
    arr = np.asarray(objs, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def dominates_maximize(a, b):
    """Return True when objective vector a Pareto-dominates b in a maximization task."""
    return np.all(a >= b) and np.any(a > b)


def fast_non_dominated_sort_maximize(objs):
    """NSGA-II fast non-dominated sorting for maximization objectives."""
    objs = _as_2d_array(objs)
    n = len(objs)
    if n == 0:
        return []

    dominated_sets = [[] for _ in range(n)]
    domination_counts = np.zeros(n, dtype=int)
    fronts = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            if dominates_maximize(objs[i], objs[j]):
                dominated_sets[i].append(j)
                domination_counts[j] += 1
            elif dominates_maximize(objs[j], objs[i]):
                dominated_sets[j].append(i)
                domination_counts[i] += 1

    for i in range(n):
        if domination_counts[i] == 0:
            fronts[0].append(i)

    current = 0
    while current < len(fronts) and fronts[current]:
        next_front = []
        for p in fronts[current]:
            for q in dominated_sets[p]:
                domination_counts[q] -= 1
                if domination_counts[q] == 0:
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        current += 1

    return fronts


def compute_pareto_ranks_maximize(objs):
    """Return zero-based Pareto ranks and fronts for maximization objectives."""
    objs = _as_2d_array(objs)
    ranks = np.full(len(objs), np.inf)
    fronts = fast_non_dominated_sort_maximize(objs)
    for rank, front in enumerate(fronts):
        for idx in front:
            ranks[idx] = rank
    return ranks.astype(int), fronts


def compute_ndr(objs):
    """NDR: fraction of individuals on the first non-dominated front."""
    objs = _as_2d_array(objs)
    if len(objs) == 0:
        return np.nan, 0
    fronts = fast_non_dominated_sort_maximize(objs)
    nd_count = len(fronts[0]) if fronts else 0
    return nd_count / len(objs), nd_count


def compute_crowding_distance_maximize(objs, front):
    """Crowding distance for one front; larger distance is preferred."""
    objs = _as_2d_array(objs)
    front = list(front)
    n = len(front)
    distances = np.zeros(n, dtype=float)
    if n == 0:
        return distances
    if n <= 2:
        distances[:] = np.inf
        return distances

    front_arr = np.asarray(front, dtype=int)
    n_obj = objs.shape[1]
    for obj_idx in range(n_obj):
        values = objs[front_arr, obj_idx]
        order = np.argsort(values)
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf
        min_val = values[order[0]]
        max_val = values[order[-1]]
        denom = max_val - min_val
        if abs(denom) < 1e-12:
            continue
        for pos in range(1, n - 1):
            prev_val = values[order[pos - 1]]
            next_val = values[order[pos + 1]]
            distances[order[pos]] += (next_val - prev_val) / denom
    return distances


def compute_crowding_distance_all_maximize(objs, fronts=None):
    """Return a crowding-distance vector aligned with the whole population."""
    objs = _as_2d_array(objs)
    if fronts is None:
        _, fronts = compute_pareto_ranks_maximize(objs)
    distances = np.zeros(len(objs), dtype=float)
    for front in fronts:
        front_distances = compute_crowding_distance_maximize(objs, front)
        for idx, distance in zip(front, front_distances):
            distances[idx] = distance
    return distances


def nsga2_environment_selection_maximize(pop, objs, pop_size):
    """NSGA-II environmental selection for maximization objectives."""
    objs = _as_2d_array(objs)
    ranks, fronts = compute_pareto_ranks_maximize(objs)
    selected_indices = []

    for front in fronts:
        if len(selected_indices) + len(front) <= pop_size:
            selected_indices.extend(front)
            continue

        remaining = pop_size - len(selected_indices)
        if remaining <= 0:
            break
        front_distances = compute_crowding_distance_maximize(objs, front)
        order = np.argsort(-front_distances)
        selected_indices.extend([front[i] for i in order[:remaining]])
        break

    if len(selected_indices) < pop_size and len(pop) > 0:
        ranked = np.lexsort((-compute_crowding_distance_all_maximize(objs, fronts), ranks))
        for idx in ranked:
            if idx not in selected_indices:
                selected_indices.append(int(idx))
            if len(selected_indices) == pop_size:
                break

    selected_pop = [pop[i] for i in selected_indices]
    selected_objs = objs[selected_indices]
    return selected_pop, selected_objs, selected_indices


def _safe_float(value):
    if value is None:
        return np.nan
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def select_parent_index(
    y_pop,
    mode,
    ibea_fitness=None,
    pareto_ranks=None,
    crowding_distances=None,
    diagnostics_recorder=None,
    generation=None,
    role=None,
):
    """
    Binary tournament selection shared by full, wo_ibea, and wo_str.

    Diagnostics always record Pareto-rank ties, even when the tournament winner is
    chosen by indicator fitness.
    """
    n = len(y_pop)
    if n == 0:
        raise ValueError("Cannot select a parent from an empty population.")
    if n == 1:
        return 0

    candidates = np.random.choice(n, 2, replace=False)
    a_idx, b_idx = int(candidates[0]), int(candidates[1])

    if pareto_ranks is None:
        pareto_ranks, fronts = compute_pareto_ranks_maximize(y_pop)
        crowding_distances = compute_crowding_distance_all_maximize(y_pop, fronts)
    if crowding_distances is None:
        _, fronts = compute_pareto_ranks_maximize(y_pop)
        crowding_distances = compute_crowding_distance_all_maximize(y_pop, fronts)

    rank_a = int(pareto_ranks[a_idx])
    rank_b = int(pareto_ranks[b_idx])
    rank_tie = rank_a == rank_b
    selection_criterion = "pareto_rank_crowding" if mode == "wo_ibea" else "indicator_fitness"

    if mode == "wo_ibea":
        if rank_a < rank_b:
            selected_idx = a_idx
        elif rank_b < rank_a:
            selected_idx = b_idx
        else:
            crowd_a = crowding_distances[a_idx]
            crowd_b = crowding_distances[b_idx]
            if crowd_a > crowd_b:
                selected_idx = a_idx
            elif crowd_b > crowd_a:
                selected_idx = b_idx
            else:
                selected_idx = int(np.random.choice([a_idx, b_idx]))
        fit_a = np.nan
        fit_b = np.nan
    else:
        if ibea_fitness is None:
            raise ValueError("IBEA fitness is required outside wo_ibea mode.")
        fit_a = _safe_float(ibea_fitness[a_idx])
        fit_b = _safe_float(ibea_fitness[b_idx])
        if fit_a > fit_b:
            selected_idx = a_idx
        elif fit_b > fit_a:
            selected_idx = b_idx
        else:
            selected_idx = int(np.random.choice([a_idx, b_idx]))

    if diagnostics_recorder is not None and diagnostics_recorder.enabled:
        diagnostics_recorder.record_tournament(
            {
                "Generation": generation,
                "Method": mode,
                "Tournament_ID": diagnostics_recorder.next_tournament_id(generation),
                "Tournament_Role": role,
                "Candidate_A_Index": a_idx,
                "Candidate_B_Index": b_idx,
                "Candidate_A_Rank": rank_a,
                "Candidate_B_Rank": rank_b,
                "Rank_Tie": bool(rank_tie),
                "Candidate_A_Fitness": fit_a,
                "Candidate_B_Fitness": fit_b,
                "Selected_Index": selected_idx,
                "Selection_Criterion": selection_criterion,
            }
        )

    return selected_idx


def _tokenize(text):
    tokens = re.findall(r"[A-Za-z0-9_]+", str(text).lower())
    return set(tokens)


def semantic_drift(parent_prompt, child_prompt, backend="token_jaccard"):
    """
    Avg.Semantic Drift support.

    Default backend is token-level Jaccard distance, so diagnostics do not depend
    on an external embedding model or network downloads.
    """
    if backend == "sentence_transformer":
        try:
            from sentence_transformers import SentenceTransformer
            from numpy.linalg import norm

            model_name = os.environ.get("DIAG_SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
            try:
                model = SentenceTransformer(model_name, local_files_only=True)
            except TypeError:
                return semantic_drift(parent_prompt, child_prompt, backend="token_jaccard")
            emb = model.encode([str(parent_prompt), str(child_prompt)], show_progress_bar=False)
            denom = norm(emb[0]) * norm(emb[1])
            if denom > 0:
                similarity = float(np.dot(emb[0], emb[1]) / denom)
                return float(max(0.0, min(2.0, 1.0 - similarity)))
        except Exception:
            pass

    parent_tokens = _tokenize(parent_prompt)
    child_tokens = _tokenize(child_prompt)
    if not parent_tokens and not child_tokens:
        return 0.0
    union = parent_tokens | child_tokens
    if not union:
        return 0.0
    intersection = parent_tokens & child_tokens
    return 1.0 - (len(intersection) / len(union))


def aggregate_selection_pressure(records, objs, method, generation, ibea_fitness=None, tie_rel=0.01):
    """Aggregate NDR, PRTR, and ITR for one generation."""
    objs = _as_2d_array(objs)
    ndr, nd_count = compute_ndr(objs)
    gen_records = [r for r in records if r.get("Generation") == generation]
    tournament_count = len(gen_records)
    rank_tie_count = sum(1 for r in gen_records if bool(r.get("Rank_Tie")))
    prtr = rank_tie_count / tournament_count if tournament_count > 0 else np.nan

    fitness_range = np.nan
    threshold = np.nan
    indicator_resolved_count = 0
    itr = np.nan
    if method == "full" and ibea_fitness is not None and len(ibea_fitness) > 0:
        finite_fitness = np.asarray(ibea_fitness, dtype=float)
        finite_fitness = finite_fitness[np.isfinite(finite_fitness)]
        if len(finite_fitness) > 0:
            fitness_range = float(np.max(finite_fitness) - np.min(finite_fitness))
            threshold = float(tie_rel * fitness_range)
        if rank_tie_count > 0 and np.isfinite(threshold):
            for rec in gen_records:
                if bool(rec.get("Rank_Tie")):
                    fa = _safe_float(rec.get("Candidate_A_Fitness"))
                    fb = _safe_float(rec.get("Candidate_B_Fitness"))
                    if np.isfinite(fa) and np.isfinite(fb) and abs(fa - fb) >= threshold:
                        indicator_resolved_count += 1
            itr = indicator_resolved_count / rank_tie_count

    return {
        "Generation": generation,
        "Method": method,
        "Population_Size": int(len(objs)),
        "NDR": ndr,
        "NonDominated_Count": int(nd_count),
        "PRTR": prtr,
        "Rank_Tie_Count": int(rank_tie_count),
        "Tournament_Count": int(tournament_count),
        "Fitness_Range": fitness_range,
        "Fitness_Tie_Threshold": threshold,
        "ITR": itr,
        "Indicator_Resolved_Count": int(indicator_resolved_count),
    }


def aggregate_mutation_effectiveness(records, method, generation):
    """Aggregate EMR, IMR, SR, and Avg.Semantic Drift for one generation."""
    gen_records = [r for r in records if r.get("Generation") == generation]
    count = len(gen_records)
    if count == 0:
        return {
            "Generation": generation,
            "Method": method,
            "Mutation_Count": 0,
            "EMR": np.nan,
            "IMR": np.nan,
            "SR": np.nan,
            "Avg_Semantic_Drift": np.nan,
            "Avg_Delta_HV": np.nan,
            "Median_Delta_HV": np.nan,
            "Prompt_Parse_Invalid_Count": 0,
            "Evaluation_Invalid_Count": 0,
            "Effective_Count": 0,
            "Ineffective_Count": 0,
            "Survived_Count": 0,
            "Radius_L1_Count": 0,
            "Radius_L2_Count": 0,
            "Radius_L3_Count": 0,
            "NoSTR_Count": 0,
        }

    effective_count = sum(1 for r in gen_records if bool(r.get("Effective_Mutation")))
    ineffective_count = sum(1 for r in gen_records if bool(r.get("Ineffective_Mutation")))
    survived_count = sum(1 for r in gen_records if bool(r.get("Survived_To_Next_Generation")))
    drift_values = [_safe_float(r.get("Semantic_Drift")) for r in gen_records]
    delta_values = [_safe_float(r.get("Delta_HV")) for r in gen_records]
    drift_values = [v for v in drift_values if np.isfinite(v)]
    delta_values = [v for v in delta_values if np.isfinite(v)]

    return {
        "Generation": generation,
        "Method": method,
        "Mutation_Count": int(count),
        "EMR": effective_count / count,
        "IMR": ineffective_count / count,
        "SR": survived_count / count,
        "Avg_Semantic_Drift": float(np.mean(drift_values)) if drift_values else np.nan,
        "Avg_Delta_HV": float(np.mean(delta_values)) if delta_values else np.nan,
        "Median_Delta_HV": float(np.median(delta_values)) if delta_values else np.nan,
        "Prompt_Parse_Invalid_Count": sum(1 for r in gen_records if not bool(r.get("Prompt_Parse_Valid"))),
        "Evaluation_Invalid_Count": sum(1 for r in gen_records if not bool(r.get("Evaluation_Valid"))),
        "Effective_Count": int(effective_count),
        "Ineffective_Count": int(ineffective_count),
        "Survived_Count": int(survived_count),
        "Radius_L1_Count": sum(1 for r in gen_records if r.get("Radius_Before") == "L1"),
        "Radius_L2_Count": sum(1 for r in gen_records if r.get("Radius_Before") == "L2"),
        "Radius_L3_Count": sum(1 for r in gen_records if r.get("Radius_Before") == "L3"),
        "NoSTR_Count": sum(1 for r in gen_records if r.get("Radius_Before") == "NoSTR"),
    }


class DiagnosticsRecorder:
    """Collect and persist mechanism-diagnostic data for one experiment run."""

    SELECTION_RAW_COLUMNS = [
        "Generation",
        "Method",
        "Tournament_ID",
        "Tournament_Role",
        "Candidate_A_Index",
        "Candidate_B_Index",
        "Candidate_A_Rank",
        "Candidate_B_Rank",
        "Rank_Tie",
        "Candidate_A_Fitness",
        "Candidate_B_Fitness",
        "Selected_Index",
        "Selection_Criterion",
    ]
    SELECTION_AGG_COLUMNS = [
        "Generation",
        "Method",
        "Population_Size",
        "NDR",
        "NonDominated_Count",
        "PRTR",
        "Rank_Tie_Count",
        "Tournament_Count",
        "Fitness_Range",
        "Fitness_Tie_Threshold",
        "ITR",
        "Indicator_Resolved_Count",
    ]
    MUTATION_RAW_BASE_COLUMNS = [
        "Generation",
        "Method",
        "Offspring_UID",
        "Parent_UID",
        "Parent_Index",
        "Operator",
        "Radius_Before",
        "Radius_After",
        "Parent_Prompt",
        "Child_Prompt",
        "Prompt_Parse_Valid",
        "Evaluation_Valid",
        "Valid_For_EMR",
        "Delta_HV",
        "Rho",
        "Effective_Mutation",
        "Ineffective_Mutation",
        "Semantic_Drift",
        "Survived_To_Next_Generation",
    ]
    MUTATION_AGG_COLUMNS = [
        "Generation",
        "Method",
        "Mutation_Count",
        "EMR",
        "IMR",
        "SR",
        "Avg_Semantic_Drift",
        "Avg_Delta_HV",
        "Median_Delta_HV",
        "Prompt_Parse_Invalid_Count",
        "Evaluation_Invalid_Count",
        "Effective_Count",
        "Ineffective_Count",
        "Survived_Count",
        "Radius_L1_Count",
        "Radius_L2_Count",
        "Radius_L3_Count",
        "NoSTR_Count",
    ]

    def __init__(
        self,
        save_dir,
        method,
        enabled=True,
        save_raw=True,
        save_plots=True,
        semantic_backend="token_jaccard",
        fitness_tie_rel=0.01,
    ):
        self.save_dir = save_dir
        self.method = method
        self.enabled = bool(enabled)
        self.save_raw = bool(save_raw)
        self.save_plots = bool(save_plots)
        self.semantic_backend = semantic_backend
        self.fitness_tie_rel = fitness_tie_rel
        self.diagnostics_dir = os.path.join(save_dir, "Diagnostics")
        self.plots_dir = os.path.join(self.diagnostics_dir, "plots")
        self.selection_tournaments = []
        self.selection_pressure = []
        self.mutation_offspring = []
        self.mutation_effectiveness = []
        self._tournament_counters = {}

        if self.enabled:
            os.makedirs(self.diagnostics_dir, exist_ok=True)
            if self.save_plots:
                os.makedirs(self.plots_dir, exist_ok=True)

    def next_tournament_id(self, generation):
        key = int(generation) if generation is not None else -1
        current = self._tournament_counters.get(key, 0) + 1
        self._tournament_counters[key] = current
        return current

    def record_tournament(self, record):
        if self.enabled:
            self.selection_tournaments.append(record)

    def record_mutation(self, record):
        if self.enabled:
            self.mutation_offspring.append(record)

    def update_mutation_survival(self, generation, survivor_uids):
        if not self.enabled:
            return
        survivor_uids = set(survivor_uids)
        for rec in self.mutation_offspring:
            if rec.get("Generation") == generation:
                rec["Survived_To_Next_Generation"] = rec.get("Offspring_UID") in survivor_uids

    def aggregate_selection(self, generation, objs, ibea_fitness=None):
        if not self.enabled:
            return None
        row = aggregate_selection_pressure(
            self.selection_tournaments,
            objs,
            self.method,
            generation,
            ibea_fitness=ibea_fitness,
            tie_rel=self.fitness_tie_rel,
        )
        self.selection_pressure = [r for r in self.selection_pressure if r.get("Generation") != generation]
        self.selection_pressure.append(row)
        self.selection_pressure.sort(key=lambda x: x.get("Generation", 0))
        return row

    def aggregate_mutation(self, generation):
        if not self.enabled:
            return None
        row = aggregate_mutation_effectiveness(self.mutation_offspring, self.method, generation)
        self.mutation_effectiveness = [
            r for r in self.mutation_effectiveness if r.get("Generation") != generation
        ]
        self.mutation_effectiveness.append(row)
        self.mutation_effectiveness.sort(key=lambda x: x.get("Generation", 0))
        return row

    def save(self):
        if not self.enabled:
            return

        if self.save_raw:
            selection_raw_df = pd.DataFrame(self.selection_tournaments)
            for col in self.SELECTION_RAW_COLUMNS:
                if col not in selection_raw_df.columns:
                    selection_raw_df[col] = np.nan
            selection_raw_df.to_csv(
                os.path.join(self.diagnostics_dir, "selection_tournaments_raw.csv"),
                index=False,
                encoding="utf-8-sig",
                columns=self.SELECTION_RAW_COLUMNS,
            )

            mutation_raw_df = pd.DataFrame(self.mutation_offspring)
            objective_cols = sorted(
                [c for c in mutation_raw_df.columns if re.match(r"Objective_\d+$", str(c))],
                key=lambda x: int(str(x).split("_")[-1]),
            )
            mutation_columns = self.MUTATION_RAW_BASE_COLUMNS + objective_cols
            for col in mutation_columns:
                if col not in mutation_raw_df.columns:
                    mutation_raw_df[col] = np.nan
            mutation_raw_df.to_csv(
                os.path.join(self.diagnostics_dir, "mutation_offspring_raw.csv"),
                index=False,
                encoding="utf-8-sig",
                columns=mutation_columns,
            )

        selection_agg_df = pd.DataFrame(self.selection_pressure)
        for col in self.SELECTION_AGG_COLUMNS:
            if col not in selection_agg_df.columns:
                selection_agg_df[col] = np.nan
        selection_agg_df.to_csv(
            os.path.join(self.diagnostics_dir, "selection_pressure_by_generation.csv"),
            index=False,
            encoding="utf-8-sig",
            columns=self.SELECTION_AGG_COLUMNS,
        )

        mutation_agg_df = pd.DataFrame(self.mutation_effectiveness)
        for col in self.MUTATION_AGG_COLUMNS:
            if col not in mutation_agg_df.columns:
                mutation_agg_df[col] = np.nan
        mutation_agg_df.to_csv(
            os.path.join(self.diagnostics_dir, "mutation_effectiveness_by_generation.csv"),
            index=False,
            encoding="utf-8-sig",
            columns=self.MUTATION_AGG_COLUMNS,
        )

        summary = {
            "Method": self.method,
            "Created_At": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Selection_Tournament_Rows": len(self.selection_tournaments),
            "Selection_Generation_Rows": len(self.selection_pressure),
            "Mutation_Offspring_Rows": len(self.mutation_offspring),
            "Mutation_Generation_Rows": len(self.mutation_effectiveness),
            "Semantic_Backend": self.semantic_backend,
            "Fitness_Tie_Rel": self.fitness_tie_rel,
            "Definitions": {
                "NDR": "|first non-dominated front| / |population|",
                "PRTR": "rank-tie tournaments / all parent-selection tournaments",
                "ITR": "rank-tie tournaments resolved by indicator fitness / rank-tie tournaments",
                "EMR": "valid mutation offspring with positive Delta_HV / mutation offspring",
                "IMR": "invalid or non-positive Delta_HV mutation offspring / mutation offspring",
                "SR": "mutation offspring that survive environmental selection / mutation offspring",
                "Avg_Semantic_Drift": "mean token-Jaccard semantic distance by default",
            },
        }
        with open(os.path.join(self.diagnostics_dir, "diagnostics_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        if self.save_plots:
            self.save_experiment_plots()

    def save_experiment_plots(self):
        if not self.enabled:
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return

        sel = pd.DataFrame(self.selection_pressure)
        mut = pd.DataFrame(self.mutation_effectiveness)

        def save_line(df, y_col, filename, ylabel=None):
            if df.empty or y_col not in df.columns:
                return
            plot_df = df.dropna(subset=[y_col])
            if plot_df.empty:
                return
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(plot_df["Generation"], plot_df[y_col], marker="o", label=self.method)
            ax.set_xlabel("Generation")
            ax.set_ylabel(ylabel or y_col)
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(self.plots_dir, filename))
            plt.close(fig)

        save_line(sel, "NDR", "selection_ndr.pdf", "NDR")
        save_line(sel, "PRTR", "selection_prtr.pdf", "PRTR")
        save_line(sel, "ITR", "selection_itr.pdf", "ITR")
        save_line(mut, "SR", "mutation_sr.pdf", "SR")
        save_line(mut, "Avg_Semantic_Drift", "mutation_semantic_drift.pdf", "Avg. Semantic Drift")

        if not mut.empty and {"EMR", "IMR"}.issubset(mut.columns):
            plot_df = mut.dropna(subset=["EMR", "IMR"], how="all")
            if not plot_df.empty:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(plot_df["Generation"], plot_df["EMR"], marker="o", label="EMR")
                ax.plot(plot_df["Generation"], plot_df["IMR"], marker="s", label="IMR")
                ax.set_xlabel("Generation")
                ax.set_ylabel("Rate")
                ax.grid(True, alpha=0.3)
                ax.legend()
                fig.tight_layout()
                fig.savefig(os.path.join(self.plots_dir, "mutation_emr_imr.pdf"))
                plt.close(fig)
