
# Problems/RCBench.py
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config

import numpy as np
import random
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import re
import pickle
import time
import ast
from fuzzywuzzy import fuzz
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
import threading


MAX_WORKERS = getattr(config, "EVAL_MAX_WORKERS", 16)

print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)


def build_chat_openai(**kwargs):
    request_timeout = getattr(config, "LLM_REQUEST_TIMEOUT", None)
    client_max_retries = getattr(config, "LLM_CLIENT_MAX_RETRIES", None)
    if request_timeout is not None:
        kwargs["timeout"] = request_timeout
    if client_max_retries is not None:
        kwargs["max_retries"] = client_max_retries
    return ChatOpenAI(**kwargs)


def extract_item_list(response, target):
    try:
        response = response.replace(" ", " ")
        target = target.replace(" ", " ").replace("&amp;", "&").replace("&reg;", "®")
        index = response.rfind(target)
        if index != -1:
            preceding_text = response[:index].strip()
            numbers = re.findall(r'\d+', preceding_text)
            if numbers:
                result_list = numbers
            else:
                result_list = []
        else:
            result_list = []
    except:
        result_list = []
    return result_list


def detect_error(response, target, mode='improve'):
    try:
        if not response or not isinstance(response, list) or len(response) == 0:
            return False, None
        if not target:
            return False, None
        target_str = str(target).strip().lower()
        normalized_response = []
        for item in response:
            try:
                item_str = str(item).strip().lower()
                if item_str:
                    normalized_response.append(item_str)
            except:
                continue
        if len(normalized_response) == 0:
            return False, None
        for idx, item in enumerate(normalized_response):
            if (item == target_str or
                    target_str in item or
                    item in target_str or
                    fuzz.ratio(item, target_str) > 80):
                return True, idx
        return False, None
    except Exception as e:
        safe_print(f"fail: {e}")
        return False, None


def diversity_calculate(list_recommond, sample_data):
    if not list_recommond or len(list_recommond) == 0:
        return 0.0
    record_category = []
    for product in list_recommond:
        try:
            if "candidate_set" in sample_data and "category_list" in sample_data:
                index = sample_data["candidate_set"].index(product)
                category = sample_data["category_list"][index]
                record_category.extend(category)
        except (ValueError, IndexError, TypeError):
            continue
    if len(record_category) == 0:
        return 0.0
    unique_category = list(set(record_category))
    diversity = len(unique_category) / len(record_category)
    return min(diversity, 1.0)


def APT(list_recommond, original_data):
    if not list_recommond or len(list_recommond) == 0:
        return 0.0
    record_set_label = []
    for product in list_recommond:
        try:
            idx = original_data["candidate_set"].index(product)
            record_set_label.append(original_data["popular_list"][idx])
        except (ValueError, IndexError, TypeError):
            continue
    if len(record_set_label) == 0:
        return 0.0
    vec_set_label = np.array(record_set_label)
    fairness = np.sum(vec_set_label) / len(list_recommond)
    return min(fairness, 1.0)



class BaseEvaluator:
    def __init__(self, train_data, batch_num, api_key, llm_model, obj_num):
        self.train_data = train_data
        self.batch_num = batch_num
        self.api_key = api_key
        self.llm_model = llm_model
        self.obj_num = obj_num
        self.sample_data = []

        # Initialize LLM for Recommendation
        if llm_model == 'glm':
            os.environ["ZHIPUAI_API_KEY"] = api_key
            self.llm_recommond = ChatZhipuAI(model="glm-4")
        elif llm_model == 'gpt':
            self.llm_recommond = build_chat_openai(api_key=self.api_key)
        elif llm_model == 'deepseek-chat':
            self.llm_recommond = build_chat_openai(
                api_key=self.api_key,
                base_url="https://api.deepseek.com/v1",
                model="deepseek-chat",
                temperature=config.LLM_TEMPERATURE
            )
        self.output_parser_recommond = StrOutputParser()


        prompt_translate = ChatPromptTemplate.from_messages([
            ("user", '''Please transfer a set of product names into a Python list that can be directly executed.
            Format: ["item1", "item2", "item3", ...]
            The set of product names is: {input}''')
        ])


        if llm_model == 'deepseek-chat':
            self.llm_translate = build_chat_openai(
                api_key=self.api_key,
                base_url="https://api.deepseek.com/v1",
                model="deepseek-chat"
            )
        else:
            self.llm_translate = self.llm_recommond

        self.chain_translate = prompt_translate | self.llm_translate | StrOutputParser()

    def Sample_Test_Data(self):
        self.sample_data = random.sample(self.train_data, self.batch_num)

    def Translate(self, input_text):

        cleaned = input_text.strip()


        try:

            start = cleaned.find('[')
            end = cleaned.rfind(']')
            if start != -1 and end != -1:
                json_str = cleaned[start:end + 1]
                return ast.literal_eval(json_str)
        except:
            pass


        if "```python" in cleaned:
            matches = re.findall(r"```python\s*(.*?)\s*```", cleaned, re.DOTALL)
            if matches:
                try:
                    return ast.literal_eval(matches[0].strip())
                except:
                    pass

        try:
            resp = self.chain_translate.invoke({"input": input_text})
            cleaned_resp = resp.strip()
            start = cleaned_resp.find('[')
            end = cleaned_resp.rfind(']')
            if start != -1 and end != -1:
                return ast.literal_eval(cleaned_resp[start:end + 1])
        except:
            pass

        return []

    def process_single_sample(self, data, chain_recommond, prompt):

        max_retries = max(1, int(getattr(config, "MAX_RETRIES", 3)))
        retry_delay = getattr(config, "RETRY_DELAY", 2)

        for attempt in range(max_retries):
            try:
                response = chain_recommond.invoke({
                    "optimized_prompt": prompt,
                    "samples": data["input"],
                })

                parsed_response = self.Translate(response)

                if not parsed_response or not isinstance(parsed_response, list):

                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return None

                parsed_response = [str(item).strip() for item in parsed_response]


                metrics = {}

                # Accuracy
                flag_error, target_index = detect_error(parsed_response, data['target'], mode='select')
                if flag_error and target_index is not None and target_index >= 0:
                    metrics['Acc'] = 1.0 / (target_index + 1)
                else:
                    metrics['Acc'] = 0.0

                # Diversity
                metrics['Div'] = diversity_calculate(parsed_response, data)

                # Fairness
                metrics['Fair'] = APT(parsed_response, data)

                return metrics

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    safe_print(f"❌ Sample failed after retries: {e}")
                    return None
        return None

    def Evaluate_(self, prompt):

        prompt_recommond = ChatPromptTemplate.from_messages([
            ("system", "You are a recommender for the shopping"),
            ("user",
             "{optimized_prompt}\n"
             "Note that, you should make the recommendation for only the candidate set \n"
             "The samples are listed as follows: \n"
             "{samples}")
        ])
        chain = prompt_recommond | self.llm_recommond | self.output_parser_recommond

        results_acc = []
        results_div = []
        results_fair = []

        if not self.sample_data:
            return 0.0, 0.0, 0.0

        executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        future_to_data = {}
        pending = set()
        try:
            future_to_data = {
                executor.submit(self.process_single_sample, data, chain, prompt): data
                for data in self.sample_data
            }
            pending = set(future_to_data)

            sample_timeout = getattr(config, "EVAL_SAMPLE_TIMEOUT", None)
            for future in as_completed(future_to_data, timeout=sample_timeout):
                pending.discard(future)
                res = future.result()
                if res:
                    results_acc.append(res['Acc'])
                    results_div.append(res['Div'])
                    results_fair.append(res['Fair'])
                else:

                    results_acc.append(0)
                    results_div.append(0)
                    results_fair.append(0)
        except TimeoutError:
            safe_print(f"⚠️ Batch evaluation timed out; marking {len(pending)} pending samples as failed.")
            for future in pending:
                future.cancel()
                results_acc.append(0)
                results_div.append(0)
                results_fair.append(0)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)


        avg_acc = np.mean(results_acc) if results_acc else 0
        avg_div = np.mean(results_div) if results_div else 0
        avg_fair = np.mean(results_fair) if results_fair else 0

        safe_print(
            f"   ⚡ Batch Eval Done ({len(results_acc)} samples). Acc: {avg_acc:.3f}, Div: {avg_div:.3f}, Fair: {avg_fair:.3f}")
        return avg_acc, avg_div, avg_fair

    def Evaluate(self, pop):

        safe_print(f"\n🚀 Starting Evaluation for {len(pop)} prompts (Parallel Batch Mode)...")

        final_results = []

        for i, prompt in enumerate(pop):

            r1, r2, r3 = self.Evaluate_(prompt)

            if self.obj_num == 3:
                final_results.append([r1, r2, r3])
            elif self.obj_num == 2:

                if isinstance(self, Acc_Div):
                    final_results.append([r1, r2])
                elif isinstance(self, Acc_Fair):
                    final_results.append([r1, r3])
                else:
                    final_results.append([r1, r2])  # Default Acc_Div
            else:
                final_results.append([r1])

        return np.array(final_results)



class Acc_Div(BaseEvaluator):
    def __init__(self, train_data, batch_num, api_key, llm_model):
        super().__init__(train_data, batch_num, api_key, llm_model, obj_num=2)


class Acc_Fair(BaseEvaluator):
    def __init__(self, train_data, batch_num, api_key, llm_model):
        super().__init__(train_data, batch_num, api_key, llm_model, obj_num=2)


class Acc_Div_Fair(BaseEvaluator):
    def __init__(self, train_data, batch_num, api_key, llm_model):
        super().__init__(train_data, batch_num, api_key, llm_model, obj_num=3)


class RCBench:
    def __init__(self, dataset_name, objectives):
        self.dataset_name = dataset_name
        self.objectives = objectives

        data_path = config.get_dataset_path(dataset_name, config.SEED, "train")
        safe_print(f"📂 Loading dataset from: {data_path}")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}")

        with open(data_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)

        batch_num = config.BATCH_SIZE
        api_key = config.OPENAI_KEY
        llm_model = config.LLM_MODEL

        safe_print(f"🔧 RCBench Init: {dataset_name}, Objectives: {objectives}, Parallel Workers: {MAX_WORKERS}")

        if 'Acc' in objectives and 'Div' in objectives and 'Fair' in objectives:
            self.problem_instance = Acc_Div_Fair(train_data, batch_num, api_key, llm_model)
        elif 'Acc' in objectives and 'Fair' in objectives:
            self.problem_instance = Acc_Fair(train_data, batch_num, api_key, llm_model)
        elif 'Acc' in objectives and 'Div' in objectives:
            self.problem_instance = Acc_Div(train_data, batch_num, api_key, llm_model)
        else:
            safe_print("⚠️ Defaulting to Acc_Div.")
            self.problem_instance = Acc_Div(train_data, batch_num, api_key, llm_model)

    def Sample_Test_Data(self):
        self.problem_instance.Sample_Test_Data()

    def Evaluate(self, pop):
        return self.problem_instance.Evaluate(pop)

    @property
    def obj_num(self):
        return self.problem_instance.obj_num

    @property
    def batch_num(self):
        return self.problem_instance.batch_num

    @batch_num.setter
    def batch_num(self, value):
        self.problem_instance.batch_num = value
