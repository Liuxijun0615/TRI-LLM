from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks import get_openai_callback
from .utils import extract_edit_prompt, IBEA_Selection
import numpy as np
import os
import time
import config


def build_chat_openai(**kwargs):
    request_timeout = getattr(config, "LLM_REQUEST_TIMEOUT", None)
    client_max_retries = getattr(config, "LLM_CLIENT_MAX_RETRIES", None)
    if request_timeout is not None:
        kwargs["timeout"] = request_timeout
    if client_max_retries is not None:
        kwargs["max_retries"] = client_max_retries
    return ChatOpenAI(**kwargs)


class LLM_EA():
    def __init__(self, pop_size, llm_model, api_key):

        self.pop_size = pop_size
        self.total_api_calls = 0
        self.failed_api_calls = 0
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.start_time = None


        if llm_model == 'glm':
            os.environ["ZHIPUAI_API_KEY"] = api_key
            llm_base = ChatZhipuAI(model="glm-4", api_key=api_key)
        elif llm_model == 'deepseek-chat':
            llm_base = build_chat_openai(
                api_key=api_key,
                base_url=config.LLM_BASE_URL,
                model=config.LLM_MODEL,
                temperature=config.LLM_TEMPERATURE
            )
        else:  # GPT Default
            llm_base = build_chat_openai(api_key=api_key)

        # 1. Initialization Chain
        prompt_init = ChatPromptTemplate.from_messages([
            ("system", "You are an initializer to provide a set of initial prompts according to user's requirement"),
            ("user", config.INITIAL_PROMPT)]
        )
        self.chain_initialize = prompt_init | llm_base | StrOutputParser()

        prompt_cross = ChatPromptTemplate.from_messages([
            ("system", "You are an expert prompt optimizer for Recommender Systems."),
            ("user", config.CROSSOVER_PROMPT)
        ])
        self.chain_operator = prompt_cross | llm_base | StrOutputParser()


        prompt_mut = ChatPromptTemplate.from_messages([
            ("system", "You are an expert prompt optimizer for Recommender Systems."),
            ("user", config.MUTATION_PROMPT)
        ])
        self.chain_tr_mutation = prompt_mut | llm_base | StrOutputParser()

    def _update_token_usage(self, cb):
        if cb:
            self.total_tokens += cb.total_tokens
            self.prompt_tokens += cb.prompt_tokens
            self.completion_tokens += cb.completion_tokens

    def _invoke_chain_with_retries(self, chain, input_vars, action_name):
        max_retries = max(1, int(getattr(config, "LLM_OPERATOR_MAX_RETRIES", getattr(config, "MAX_RETRIES", 3))))
        retry_delay = getattr(config, "RETRY_DELAY", 2)
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                with get_openai_callback() as cb:
                    output = chain.invoke(input_vars)
                    self._update_token_usage(cb)
                self.total_api_calls += 1
                return output
            except Exception as e:
                self.failed_api_calls += 1
                last_error = e
                if attempt < max_retries:
                    print(f"fail ({attempt}/{max_retries})... {e}")
                    time.sleep(retry_delay)

        print(f"fail: {last_error}")
        return None

    def _build_operator_meta(self, raw_output, extracted_prompts, fallback_used):
        return {
            "raw_output": raw_output if getattr(config, "DIAG_SAVE_LLM_RAW_OUTPUT", False) else None,
            "parsed_ok": len(extracted_prompts) > 0,
            "fallback_used": bool(fallback_used),
            "num_extracted_prompts": len(extracted_prompts),
        }

    def initialize(self, example):
        pop = []
        print(f"start init")

        for i in range(self.pop_size):
            input_vars = {}
            if "{example}" in config.INITIAL_PROMPT:
                input_vars["example"] = example

            output = self._invoke_chain_with_retries(self.chain_initialize, input_vars, "初始化")
            if output is None:
                pop.append(example)
                continue

            individual = extract_edit_prompt(output)
            if len(individual) == 0:
                individual = [example]
            pop.extend(individual)


        return pop[:self.pop_size]

    def single_crossover(self, p1, p2, return_meta=False):

        output = self._invoke_chain_with_retries(
            self.chain_operator,
            {"prompt1": p1, "prompt2": p2},
            "crossover",
        )
        if output is None:
            if return_meta:
                return p1, self._build_operator_meta("", [], True)
            return p1

        res = extract_edit_prompt(output)
        fallback_used = len(res) == 0
        child = res[0] if len(res) > 0 else p1
        if return_meta:
            return child, self._build_operator_meta(output, res, fallback_used)
        return child  # Fallback return parent if parsing failed.

    def crossover(self, pop):

        offspring = []
        print(f"(Pop Size: {len(pop)})...")
        for _ in range(self.pop_size):
            # 随机选择两个父代
            p1, p2 = np.random.choice(pop, 2, replace=False)
            child = self.single_crossover(p1, p2)
            offspring.append(child)
        return offspring

    def trust_region_mutation(self, parent, constraint, return_meta=False):

        output = self._invoke_chain_with_retries(
            self.chain_tr_mutation,
            {"prompt": parent, "constraint": constraint},
            "mutation",
        )
        if output is None:
            if return_meta:
                return parent, self._build_operator_meta("", [], True)
            return parent

        res = extract_edit_prompt(output)
        fallback_used = len(res) == 0
        child = res[0] if len(res) > 0 else parent
        if return_meta:
            return child, self._build_operator_meta(output, res, fallback_used)
        return child

    def IBEA_selection(self, pop, y_pop, offspring, y_offspring):

        pop_merge = pop + offspring
        y_merge = np.concatenate((y_pop, y_offspring), axis=0)

        selected_pop, selected_y = IBEA_Selection(pop_merge, y_merge, self.pop_size, config.IBEA_KAPPA)
        return selected_pop, selected_y
