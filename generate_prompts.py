import argparse
import sys

import yaml
from pprint import pprint
from prompt_builders import *
from tasks import *
import nltk


nltk.download('punkt')
nltk.download('punkt_tab')


def get_full_output_path(output_path: str) -> str:
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    full_output_path = os.path.join(curr_dir, output_path)
    if full_output_path[-1] == '/':
        full_output_path = full_output_path[:-1]
    return full_output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='text_simplification/config_text_simplification.yaml')
    parser.add_argument('--output_path', '-o', default='text_simplification/generated_prompts/')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    args.output_path = get_full_output_path(args.output_path)
    config['args'] = {'config': args.config, 'output_path': args.output_path}

    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_path, 'logs')).mkdir(parents=True, exist_ok=True)

    pprint(config)
    task = load_task(config)

    prompt_builder_zero_shot = ZeroShotPromptBuilder(config, task)
    prompt_builder_few_shot = FewShotPromptBuilder(config, task)
    prompt_builder_instruction_induction = InstructionInductionPromptBuilder(config, task)
    prompt_builder_mixed = MixedInstructionInductionPromptBuilder(config, task)
    prompt_builder_optimized = InstructionOptimizationPromptBuilder(config, task)

    time_now_str = datetime.datetime.now().strftime('%m_%d_%H-%M-%S')
    num_shot = config['main']['few_shot_num_examples']

    # In this block, prompts are generated one time
    prompt_builder_zero_shot.generate_and_save_prompt(json_filename=f'{args.output_path}/zero-shot.json')
    # In this block, better prompts are found using random search on validation dataset
    prompt_builder_few_shot.generate_prompts_random_search(json_filename=f'{args.output_path}/{time_now_str}_few-shot_{num_shot}.json')
    instruction_induction_prompts_data_pool = prompt_builder_instruction_induction.generate_prompts_random_search(json_filename=f'{args.output_path}/{time_now_str}_instruction_induction_{num_shot}.json', num_to_return=4)
    prompt_builder_optimized.generate_and_save_prompt(json_filename=f'{args.output_path}/{time_now_str}_optimized_{num_shot}.json',
                                                      prompt_data=instruction_induction_prompts_data_pool)

    # Store instruction-inducted prompts for debug purposes
    with open(f'{args.output_path}/{time_now_str}_instruction_induction_{num_shot}_prompts_data_pool.jsonl', 'w') as f:
        json.dump(instruction_induction_prompts_data_pool, f, indent=4)
