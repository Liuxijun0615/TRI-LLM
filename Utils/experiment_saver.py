import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
from Utils import nondomination
from Utils.hypervolume import hypervolume


class ExperimentSaver:
    def __init__(self, experiment_name, algorithm_name, dataset_name, objectives, seed):
        self.experiment_name = experiment_name
        self.algorithm_name = algorithm_name
        self.dataset_name = dataset_name
        self.objectives = objectives
        self.seed = seed


        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = f"Results/{dataset_name}/{algorithm_name}_{objectives}_Seed_{seed}_{timestamp}"
        os.makedirs(self.save_dir, exist_ok=True)


        self.config = {
            'experiment_name': experiment_name,
            'algorithm': algorithm_name,
            'dataset': dataset_name,
            'objectives': objectives,
            'seed': seed,
            'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'save_dir': self.save_dir
        }

        self._save_config()

    def _save_config(self):

        config_file = f"{self.save_dir}/config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        print(f"{config_file}")

    def save_population(self, population, objectives, iteration, additional_info=None):

        pop_data = {
            'iteration': iteration,
            'population': population,
            'objectives': objectives.tolist() if isinstance(objectives, np.ndarray) else objectives,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        if additional_info:
            pop_data.update(additional_info)

        pickle_file = f"{self.save_dir}/population_iter_{iteration:02d}.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(pop_data, f)


        if len(objectives) > 0:
            csv_file = f"{self.save_dir}/objectives_iter_{iteration:02d}.csv"
            df = pd.DataFrame(objectives, columns=[f'Objective_{i + 1}' for i in range(objectives.shape[1])])
            df['Prompt_Index'] = range(len(population))
            df['Prompt'] = population
            df.to_csv(csv_file, index=False, encoding='utf-8')

        return pop_data

    def save_pareto_front(self, population, objectives, iteration):


        if len(objectives) > 0:
            try:
                fronts = nondomination.fast_non_dominated_sort(objectives)
                if fronts and len(fronts) > 0 and len(fronts[0]) > 0:
                    pareto_front = fronts[0]

                    pareto_population = [population[i] for i in pareto_front]
                    pareto_objectives = objectives[pareto_front]

                    pareto_data = {
                        'iteration': iteration,
                        'pareto_front_indices': pareto_front.tolist(),
                        'pareto_population': pareto_population,
                        'pareto_objectives': pareto_objectives.tolist(),
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }


                    pareto_file = f"{self.save_dir}/pareto_front_iter_{iteration:02d}.pkl"
                    with open(pareto_file, 'wb') as f:
                        pickle.dump(pareto_data, f)


                    if len(pareto_objectives) > 0:
                        pareto_csv = f"{self.save_dir}/pareto_front_iter_{iteration:02d}.csv"
                        df = pd.DataFrame(pareto_objectives,
                                          columns=[f'Objective_{i + 1}' for i in range(pareto_objectives.shape[1])])
                        df['Prompt_Index'] = pareto_front
                        df['Prompt'] = pareto_population
                        df.to_csv(pareto_csv, index=False, encoding='utf-8')

                    print(f"{len(pareto_population)}")
                    return pareto_data
                else:
                    print(f"fail")
                    return None
            except Exception as e:
                print(f"fail")
                return None
        else:
            print(f"fail")
            return None

    def save_metrics(self, metrics_dict, iteration=None):

        metrics_file = f"{self.save_dir}/metrics.json"


        if os.path.exists(metrics_file):
            with open(metrics_file, 'r', encoding='utf-8') as f:
                all_metrics = json.load(f)
        else:
            all_metrics = {}

        metrics_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if iteration is not None:
            if isinstance(iteration, str):
                key = iteration
            else:
                key = f'iteration_{iteration:02d}'
            all_metrics[key] = metrics_dict
        else:
            all_metrics['final'] = metrics_dict

        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)


        self._save_metrics_to_csv(all_metrics)

        return all_metrics

    def _save_metrics_to_csv(self, all_metrics):

        rows = []
        for key, metrics in all_metrics.items():
            row = {'iteration': key}
            row.update(metrics)
            rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            csv_file = f"{self.save_dir}/metrics.csv"
            df.to_csv(csv_file, index=False, encoding='utf-8')

    def save_convergence_analysis(self, all_populations, all_objectives):

        convergence_data = {
            'iterations': list(range(len(all_populations))),
            'hypervolume': [],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }


        for i, objectives in enumerate(all_objectives):
            if len(objectives) > 0:

                try:
                    hv = hypervolume(objectives, None)
                    convergence_data['hypervolume'].append(float(hv))
                    print(f"✅  {i} hv: {hv:.4f}")
                except Exception as e:
                    print(f"fail")
                    convergence_data['hypervolume'].append(0.0)
            else:
                convergence_data['hypervolume'].append(0.0)


        convergence_file = f"{self.save_dir}/convergence_analysis.pkl"
        with open(convergence_file, 'wb') as f:
            pickle.dump(convergence_data, f)


        convergence_csv = f"{self.save_dir}/convergence_analysis.csv"
        df = pd.DataFrame({
            'iteration': convergence_data['iterations'],
            'hypervolume': convergence_data['hypervolume']
        })
        df.to_csv(convergence_csv, index=False)

        print(f"save: {convergence_csv}")
        return convergence_data

    def save_final_summary(self, final_population, final_objectives, total_time, api_stats):

        if len(final_objectives) > 0:
            fronts = nondomination.fast_non_dominated_sort(final_objectives)
            if fronts and len(fronts) > 0:
                final_pareto_front = fronts[0]
                final_pareto_population = [final_population[i] for i in final_pareto_front]
                final_pareto_objectives = final_objectives[final_pareto_front]
            else:
                final_pareto_population = []
                final_pareto_objectives = np.array([])
        else:
            final_pareto_population = []
            final_pareto_objectives = np.array([])

        summary = {
            'experiment_name': self.experiment_name,
            'algorithm': self.algorithm_name,
            'dataset': self.dataset_name,
            'objectives': self.objectives,
            'seed': self.seed,
            'start_time': self.config['start_time'],
            'end_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_duration_seconds': total_time,
            'total_duration_hours': total_time / 3600,
            'final_population_size': len(final_population),
            'final_pareto_front_size': len(final_pareto_population),
            'api_statistics': api_stats,
            'final_pareto_front': {
                'population': final_pareto_population,
                'objectives': final_pareto_objectives.tolist() if len(final_pareto_objectives) > 0 else []
            }
        }


        summary_file = f"{self.save_dir}/experiment_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


        if len(final_pareto_population) > 0:
            final_pareto_file = f"{self.save_dir}/final_pareto_front.csv"
            df = pd.DataFrame(final_pareto_objectives,
                              columns=[f'Objective_{i + 1}' for i in range(final_pareto_objectives.shape[1])])
            df['Prompt'] = final_pareto_population
            df.to_csv(final_pareto_file, index=False, encoding='utf-8')

        print(f"save: {summary_file}")
        return summary

