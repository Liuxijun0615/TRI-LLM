


OPENAI_KEY = 'key'  # 替换为你的实际API密钥

# LLM 模型配置
LLM_MODEL = 'model name'
LLM_BASE_URL = "https://"

# LLM 参数
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 1000
LLM_REQUEST_TIMEOUT = 60
LLM_CLIENT_MAX_RETRIES = 2
LLM_OPERATOR_MAX_RETRIES = 5


SEED = 625


MAX_ITERATIONS = 20
POPULATION_SIZE = 20
BATCH_SIZE = 10
TEST_BATCH_SIZE = 0
EVAL_MAX_WORKERS = 16
EVAL_SAMPLE_TIMEOUT = 240


DEBUG_MODE = True
LOG_LEVEL = 'DEBUG'

ENABLE_DIAGNOSTICS = True
ABLATION_MODE = "full"

DIAG_FITNESS_TIE_REL = 0.01
DIAG_SEMANTIC_BACKEND = "token_jaccard"
DIAG_SAVE_RAW = True
DIAG_SAVE_PLOTS = True


DIAG_SAVE_LLM_RAW_OUTPUT = False

NO_STR_MUTATION_CONSTRAINT = (
    "Freely rewrite the prompt to improve recommendation performance. "
    "You may change the structure, reasoning strategy, and wording without using any semantic trust-region constraint."
)


MAX_RETRIES = 3
RETRY_DELAY = 5

DATA_OBJECTIVES = [
    ['Movie', 'Acc_Div'],
    ['Game', 'Acc_Div'],
    ['Bundle', 'Acc_Div'],
    ['Movie', 'Acc_Fair'],
    ['Game', 'Acc_Fair'],
    ['Bundle', 'Acc_Fair'],
    ['Movie', 'Acc_Div_Fair'],
    ['Game', 'Acc_Div_Fair'],
    ['Bundle', 'Acc_Div_Fair']
]

DATASET_PATHS = {
    'Movie': 'DataSet/Movie',
    'Game': 'DataSet/Game',
    'Bundle': 'DataSet/Bundle'
}


RESULTS_BASE_DIR = 'Results'

IBEA_KAPPA = 0.05


INITIAL_PROMPT = (
    "Now, I have a prompt for my task. I want to modify this prompt to better achieve my task. \n"
    "I will give an example of my current prompt. Please randomly generate a prompt based on my example. \n"
    "My example is as follows: \n"
    "{example} \n"
    "Note that the final prompt should be bracketed with <START> and <END>."
)

EXAMPLE_PROMPT = (
    "Based on the user's current session interactions, you need to answer the following subtasks step by step:\n"
    "1. Discover combinations of items within the session, where the number of combinations can be one or more.\n"
    "2. Based on the items within each combination, infer the user's interactive intent within each combination.\n"
    "3. Select the intent from the inferred ones that best represents the user's current preferences.\n"
    "4. Based on the selected intent, please rerank the 20 items in the candidate set according to the possibility of potential user interactions and show me your ranking results with item index.\n"
    "Note that the order of all items in the candidate set must be given, and the items for ranking must be within the candidate set.\n"
)

CROSSOVER_PROMPT_IBEA_NSGA2 = (
    "Please follow the instruction step-by-step to generate a better prompt. \n"
    "1. Cross over the following prompts and generate two new prompts: \n"
    "Prompt 1: {prompt1} \n"
    "Prompt 2: {prompt2}. \n"
    "2. Mutate the prompt generated in Step 1 and generate "
    "a final prompt bracketed with <START> and <END>."
)

CROSSOVER_PROMPT_MOEAD = (
    "Please follow the instruction step-by-step to generate a better prompt. \n"
    "1. Cross over the following prompts and generate a new prompt: \n"
    "Prompt 1: {prompt1} \n"
    "Prompt 2: {prompt2}. \n"
    "2. Mutate the prompt generated in Step 1 and generate "
    "a final prompt bracketed with <START> and <END>."
)
CROSSOVER_PROMPT = (
    "You are an expert prompt optimizer for Recommender Systems. "
    "Please generate a new prompt by crossing over the following two parent prompts:\n\n"
    "Prompt 1: {prompt1}\n\n"
    "Prompt 2: {prompt2}\n\n"
    "Output ONLY the final merged prompt bracketed with <START> and <END>."
)

MUTATION_PROMPT = (
    "You are an expert prompt optimizer for Recommender Systems. "
    "Please mutate the following prompt to improve its performance:\n\n"
    "Prompt: {prompt}\n\n"
    "SYSTEM CONSTRAINT (CRITICAL): You must strictly follow the instruction below to guide your mutation:\n"
    "{{\n" 
    "    {constraint}\n"
    "}}\n\n"
    "Output ONLY the final mutated prompt bracketed with <START> and <END>."
)

ZEROSHOT_PROMPT = (
    "You are a recommender system. Based on the user's interaction history, "
    "recommend the top-20 most likely items the user will interact with next.\n"
    "Output the results as a ranked list of item indices."
)

FEWSHOT_PROMPT = (
    "You are a recommender system. Here are some examples of user history and their preferred next items:\n"
    "{examples}\n\n"
    "Now, based on the current user's session interactions, recommend the top-20 items.\n"
    "Session: {session}\n"
    "Recommendation:"
)


APE_GENERATION_PROMPT = (
    "I need to write a prompt for a recommender system AI. \n"
    "The task is to recommend items based on user history.\n"
    "Please generate {num_prompts} distinct and effective instruction prompts that I can feed to the AI.\n"
    "Each prompt should be wrapped with <START> and <END>."
)


OPRO_META_PROMPT = (
    "You are an optimizer. Your goal is to generate a new instruction that improves the recommendation accuracy.\n"
    "Here are some previous instructions and their corresponding performance scores (Accuracy):\n"
    "{history}\n\n"
    "Based on the insights from the history, generate a new, different, and potentially better instruction.\n"
    "Wrap the new instruction with <START> and <END>."
)

def get_save_path(algorithm, dataset, objectives, seed=SEED):
    import os
    from datetime import datetime


    base_dir = f"{RESULTS_BASE_DIR}/{dataset}"
    os.makedirs(base_dir, exist_ok=True)


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detailed_dir = f"{base_dir}/{algorithm}_{objectives}_Seed_{seed}_{timestamp}"
    os.makedirs(detailed_dir, exist_ok=True)


    compatibility_path = f"{base_dir}/{algorithm}_{objectives}_Seed_{seed}"

    return detailed_dir, compatibility_path


def get_dataset_path(dataset_name, seed=SEED, file_type="train"):

    import os

    base_path = DATASET_PATHS.get(dataset_name, f"Dataset/{dataset_name}")
    if not os.path.exists(base_path):
        alt_base_path = base_path.replace("Dataset", "DataSet", 1)
        if os.path.exists(alt_base_path):
            base_path = alt_base_path

    if file_type == "train":
        return f"{base_path}/train_seed_{seed}.json"
    elif file_type == "validation":
        return f"{base_path}/valid.json"
    else:
        return f"{base_path}/{file_type}"
