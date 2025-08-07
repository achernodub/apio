import datetime
import random
import sys
from itertools import permutations
from tasks import *
from utils import *
from llm_utils import *


class PromptBuilder:
    def __init__(self, config: dict, task: Task):
        self.config = config
        self.output_path = config['args']['output_path']
        self.task = task
        self.type = 'not set'
        self.use_llm_proxy = config['main']['use_llm_proxy']
        self.task_name = self.config['main']['task_name']
        self.task_description = self.config['main']['task_description']
        self.few_shot_num_examples = self.config['main']['few_shot_num_examples']
        self.prompt_builder_llm_model_name = config['prompt_builder_llm']['model_name']
        self.prompt_builder_llm_temperature = config['prompt_builder_llm']['temperature']
        self.prompt_builder_llm_max_output_tokens = config['prompt_builder_llm']['max_output_tokens']
        self.prompt_builder_llm_num_threads = config['prompt_builder_llm']['num_threads']
        self.prompt_builder_llm_top_p = config['prompt_builder_llm']['top_p']
        self.backend_llm_model_name = config['backend_llm']['model_name']
        self.backend_llm_temperature = config['backend_llm']['temperature']
        self.backend_llm_max_output_tokens = config['backend_llm']['max_output_tokens']
        self.backend_llm_num_threads = config['backend_llm']['num_threads']
        self.backend_llm_top_p = config['backend_llm']['top_p']
        self.random_search_num_trials = 1

    def run_prompt_builder_llm(self, prompt: str, model_name: bool = None) -> str:
        if model_name is None:
            model_name = self.prompt_builder_llm_model_name
        output_text = llm_single(self.use_llm_proxy,
                                 prompt,
                                 model_name=model_name,
                                 temperature=self.prompt_builder_llm_temperature,
                                 max_output_tokens=self.prompt_builder_llm_max_output_tokens,
                                 top_p=self.prompt_builder_llm_top_p)
        return output_text

    def run_llm_prompt_builder_batch(self, prompts: [str], model_name: bool = None) -> str:
        if model_name is None:
            model_name = self.prompt_builder_llm_model_name
        output_texts = llm_batch(self.use_llm_proxy,
                                 prompts,
                                 model_name=model_name,
                                 temperature=self.prompt_builder_llm_temperature,
                                 max_output_tokens=self.prompt_builder_llm_max_output_tokens,
                                 batch_size=self.prompt_builder_llm_num_threads,
                                 top_p=self.prompt_builder_llm_top_p)
        return output_texts

    def run_backend_llm(self, prompt: str, model_name: bool = None) -> str:
        if model_name is None:
            model_name = self.backend_llm_model_name
        output_text = llm_single(self.use_llm_proxy,
                                 prompt,
                                 model_name=model_name,
                                 temperature=self.backend_llm_temperature,
                                 max_output_tokens=self.backend_llm_max_output_tokens,
                                 top_p=self.backend_llm_top_p)
        return output_text

    def run_llm_backend_batch(self, prompts: [str], model_name: bool = None) -> str:
        if model_name is None:
            model_name = self.backend_llm_model_name
        output_texts = llm_batch(self.use_llm_proxy,
                                 prompts,
                                 model_name=model_name,
                                 temperature=self.backend_llm_temperature,
                                 max_output_tokens=self.backend_llm_max_output_tokens,
                                 batch_size=self.backend_llm_num_threads,
                                 top_p=self.backend_llm_top_p)
        return output_texts

    @abstractmethod
    def postprocess_text(self, text: str) -> str:
        pass

    def postprocess_texts(self, texts: [str]) -> [str]:
        return [self.postprocess_text(text) for text in texts]

    def get_prompt_tempate_from_instructions(self, instructions: list[str]) -> str:
        prompt_template = self.prompt_header
        for instruction in instructions:
            prompt_template += f'* {instruction}\n'
        prompt_template += self.prompt_footer
        return prompt_template

    def evaluate_instructions(self, instructions: str, data: dict, return_outputs: bool = False):
        prompt_template = self.get_prompt_tempate_from_instructions(instructions)
        return self.task.evaluate_prompt_template(prompt_template, data, return_outputs)

    def get_few_shot_examples(self, prompt_data: dict = None) -> (list[str], list[str]):
        if prompt_data is not None:
            few_shot_input_examples = prompt_data['few_shot_input_examples']
            few_shot_output_examples = prompt_data['few_shot_output_examples']
        else:
            num_shot = self.config['main']['few_shot_num_examples']
            few_shot_input_examples, few_shot_output_examples = self.task.sample_random_train_data_inputs_outputs(num_samples=num_shot)
        return few_shot_input_examples, few_shot_output_examples

    @abstractmethod
    def generate_prompt(self, prompt_data = None) -> dict:
        pass

    def save_prompt(self, json_filename: str, prompt_data: dict):
        if self.config['main']['save_' + self.type]:
            with open(json_filename, 'w') as f:
                json.dump(prompt_data, f, indent=4)
            prompt_template_filename = json_filename.replace('.json', '.txt')
            Path(prompt_template_filename).write_text(prompt_data['PROMPT_TEMPLATE'])

    #def evaluate_prompt_templates_valid(self, prompt_templates: [str]) -> [float]:
    #    pass

    def generate_prompts_random_search(self, json_filename: str, num_to_return: int = 1):
        prompts_data_pool = [self.generate_prompt() for _ in range(self.random_search_num_trials)]
        prompt_templates = [prompts_data['PROMPT_TEMPLATE'] for prompts_data in prompts_data_pool]
        valid_scores = self.task.evaluate_prompt_templates_valid(prompt_templates, return_outputs = False)
        sorted_prompts_data_pool, sorted_valid_scores = (zip(*sorted(zip(prompts_data_pool, valid_scores),
                                                                     key=lambda x: x[1], reverse=True)))
        self.save_prompt(json_filename=json_filename, prompt_data=sorted_prompts_data_pool[0])
        self.logger.info(f'sorted_valid_scores = {sorted_valid_scores}')
        self.logger.info(f'best_prompt:\n\n {json.dumps(sorted_prompts_data_pool[0], indent=4)}')
        num_to_return = min(num_to_return, len(sorted_prompts_data_pool))
        return sorted_prompts_data_pool[:num_to_return]

    def generate_and_save_prompt(self, json_filename: str, prompt_data: dict = None):
        self.json_filename = json_filename
        prompt_data = self.generate_prompt(prompt_data=prompt_data)
        self.save_prompt(json_filename=json_filename, prompt_data=prompt_data)
        return prompt_data

class ZeroShotPromptBuilder(PromptBuilder):
    def __init__(self, config: dict, task: Task):
        super().__init__(config, task)
        self.type = 'prompt_builder_zero_shot'
        #self.logger = get_logger(filename=f'{self.output_path}/logs/zero_shot.log')
        self.prompt_header = config['prompt_builder_zero_shot']['prompt_header']
        self.prompt_footer = config['prompt_builder_zero_shot']['prompt_footer']

    def generate_prompt(self, prompt_data: dict = None) -> dict:
        prompt_template = self.prompt_header + self.prompt_footer
        prompt_template = prompt_template.strip()
        if prompt_data is None:
            prompt_data = dict()
        prompt_data['config'] = self.config
        prompt_data['PROMPT_TEMPLATE'] = prompt_template
        return prompt_data


class FewShotPromptBuilder(PromptBuilder):
    def __init__(self, config: dict, task: Task):
        super().__init__(config, task)
        self.type = 'prompt_builder_few_shot'
        self.logger = get_logger(filename=f'{self.output_path}/logs/few_shot.log')
        self.prompt_header = config['prompt_builder_few_shot']['prompt_header']
        self.input_example_label = config['prompt_builder_few_shot']['input_example_label']
        self.output_example_label = config['prompt_builder_few_shot']['output_example_label']
        self.prompt_footer = config['prompt_builder_few_shot']['prompt_footer']
        self.random_search_num_trials = config['prompt_builder_few_shot']['random_search_num_trials']

    def generate_prompt(self, prompt_data: dict = None) -> dict:
        few_shot_input_examples, few_shot_output_examples = self.get_few_shot_examples(prompt_data=prompt_data)

        prompt_template = self.prompt_header
        for input_text, output_text in zip(few_shot_input_examples, few_shot_output_examples):
            few_shot_example = f'\n{self.input_example_label} {input_text}\n{self.output_example_label} {output_text}\n'
            prompt_template += few_shot_example

        prompt_template += '\n' + self.prompt_footer
        prompt_template = prompt_template.strip()

        if prompt_data is None:
            prompt_data = dict()
        prompt_data['config'] = self.config
        prompt_data['PROMPT_TEMPLATE'] = prompt_template
        prompt_data['few_shot_input_examples'] = few_shot_input_examples
        prompt_data['few_shot_output_examples'] = few_shot_output_examples
        return prompt_data


class InstructionInductionPromptBuilder(PromptBuilder):
    def __init__(self, config: dict, task: Task):
        super().__init__(config, task)
        self.type = 'prompt_builder_instruction_induction'
        self.logger = get_logger(filename=f'{self.output_path}/logs/instruction_induction.log')
        self.prompt_header = config['prompt_builder_instruction_induction']['prompt_header']
        self.input_example_label = config['prompt_builder_instruction_induction']['input_example_label']
        self.output_example_label = config['prompt_builder_instruction_induction']['output_example_label']
        self.prompt_footer = config['prompt_builder_instruction_induction']['prompt_footer']
        self.random_search_num_trials = config['prompt_builder_instruction_induction']['random_search_num_trials']
        self.meta_prompt_induce_instruction_template = (
            f'Below is an example of an input-output pair for the {self.task_description} task.\n'
            f'{self.input_example_label}: <input_text> {self.output_example_label}: <output_example>\n'
            'You are the prompt engineer. Could you give an instruction for this example? Do not mention any part of the considered texts.')

    def postprocess_text(self, text: str) -> str:
        text = text.replace('Instruction:', '')
        text = text.strip()
        return text

    def induce_instruction(self, input_text: str, output_text: str) -> str:
        meta_prompt_induce_instruction = self.meta_prompt_induce_instruction_template.replace('<input_text>', input_text).replace('<output_example>', output_text)
        instruction_induced = self.run_prompt_builder_llm(meta_prompt_induce_instruction)
        instruction_induced = self.postprocess_text(instruction_induced)
        return instruction_induced

    def induce_instructions(self, input_texts: [str], output_texts: [str]) -> [str]:
        meta_prompts = []
        for input_text, output_text in zip(input_texts, output_texts):
            meta_prompt = self.meta_prompt_induce_instruction_template
            meta_prompt = meta_prompt.replace('<input_text>', input_text)
            meta_prompt = meta_prompt.replace('<output_example>', output_text)
            meta_prompts.append(meta_prompt)
        instructions = self.run_llm_prompt_builder_batch(meta_prompts)
        instructions = self.postprocess_texts(instructions)
        return instructions

    def generate_prompt(self, prompt_data: dict = None) -> dict:
        few_shot_input_examples, few_shot_output_examples = self.get_few_shot_examples(prompt_data=prompt_data)
        instructions = self.induce_instructions(few_shot_input_examples, few_shot_output_examples)
        prompt_template = self.get_prompt_tempate_from_instructions(instructions=instructions)

        if prompt_data is None:
            prompt_data = dict()
        prompt_data['config'] = self.config
        prompt_data['PROMPT_TEMPLATE'] = prompt_template
        prompt_data['few_shot_input_examples'] = few_shot_input_examples
        prompt_data['few_shot_output_examples'] = few_shot_output_examples
        prompt_data['instructions'] = instructions
        return prompt_data


class MixedInstructionInductionPromptBuilder(PromptBuilder):
    def __init__(self, config: dict, task: Task):
        super().__init__(config, task)
        self.type = 'prompt_builder_mixed'
        #self.logger = get_logger(filename=f'{self.output_path}/logs/mixed.log')
        self.prompt_header = config['prompt_builder_mixed']['prompt_header']
        self.input_example_label = config['prompt_builder_mixed']['input_example_label']
        self.output_example_label = config['prompt_builder_mixed']['output_example_label']
        self.prompt_footer = config['prompt_builder_mixed']['prompt_footer']

    def generate_prompt(self, prompt_data: dict = None) -> dict:
        if 'few_shot_input_examples' not in prompt_data:
            raise ValueError('few_shot_input_examples not in prompt_data')
        if 'few_shot_output_examples' not in prompt_data:
            raise ValueError('few_shot_output_examples not in prompt_data')
        if 'instructions' not in prompt_data:
            raise ValueError('instructions not in prompt_data')

        few_shot_input_examples = prompt_data['few_shot_input_examples']
        few_shot_output_examples = prompt_data['few_shot_output_examples']
        instructions = prompt_data['instructions']

        prompt_template = self.prompt_header
        for input_text, output_text, instruction in zip(few_shot_input_examples, few_shot_output_examples, instructions):
            few_shot_example = f'\n{self.input_example_label} {input_text}\n{self.output_example_label} {output_text}\n'
            prompt_template += f'* {instruction} For example:{few_shot_example}\n'
        prompt_template += self.prompt_footer
        prompt_template = prompt_template.strip()

        if prompt_data is None:
            prompt_data = dict()
        prompt_data['config'] = self.config
        prompt_data['PROMPT_TEMPLATE'] = prompt_template
        prompt_data['few_shot_input_examples'] = few_shot_input_examples
        prompt_data['few_shot_output_examples'] = few_shot_output_examples
        prompt_data['instructions'] = instructions
        return prompt_data


class InstructionOptimizationPromptBuilder(PromptBuilder):
    def __init__(self, config: dict, task: Task):
        super().__init__(config, task)
        self.type = 'prompt_builder_instruction_optimization' #save_prompt_builder_instruction_optimization
        self.logger = get_logger(filename=f'{self.output_path}/logs/instruction_optimization.log')
        self.logger_feedbacks = get_logger(filename=f'{self.output_path}/logs/instruction_optimization_feedbacks.log')
        self.prompt_header = config['prompt_builder_instruction_optimization']['prompt_header']
        self.input_example_label = config['prompt_builder_instruction_optimization']['input_example_label']
        self.output_example_label = config['prompt_builder_instruction_optimization']['output_example_label']
        self.prompt_footer = config['prompt_builder_instruction_optimization']['prompt_footer']
        self.beam_size = config['prompt_builder_instruction_optimization']['beam_size']
        self.num_optimization_iterations = config['prompt_builder_instruction_optimization']['num_optimization_iterations']
        self.train_batch_size = config['prompt_builder_instruction_optimization']['train_batch_size']
        self.rephrase_random_instructions_num_samples = config['prompt_builder_instruction_optimization']['rephrase_random_instructions_num_samples']
        self.rephrase_random_instructions_scale_factor = config['prompt_builder_instruction_optimization']['rephrase_random_instructions_scale_factor']
        self.generate_permuted_instructions_scale_factor = config['prompt_builder_instruction_optimization']['generate_permuted_instructions_scale_factor']
        self.improve_by_feedback_scale_factor = config['prompt_builder_instruction_optimization']['improve_by_feedback_scale_factor']
        self.max_instructions_num = config['prompt_builder_instruction_optimization']['max_instructions_num']
        self.instructions_feedback_num_trials = config['prompt_builder_instruction_optimization']['instructions_feedback_num_trials']
        self.meta_prompt_induce_instruction_template = (
            f'You are a super-talented prompt engineer. You are working on improvement of the {self.task_description} System.'
            f'{self.task_description} System has built-in instructions, but they lead to discrepancies between {self.task_description} System\'s Output and Gold Output:\n'
            f'{self.task_description} System\'s Input: <input_text>\n\n'
            f'{self.task_description} System\'s Output: <output_text>\n\n'
            f'Gold Output: <output_example>\n\n'
            f'Suggest improved instruction to force the {self.task_description} System\'s Output exactly the same as Gold Output for the given {self.task_description} System\'s Input.'
            f' Put the improved instruction between <improved_instruction> and </improved_instruction> tags. Do not use no more than two sentences. Do not mention Gold Output. Do not mention directly any part of the considered texts.')

    def postprocess_text(self, text):
        text = text.replace('Instruction:', '')
        text = text.strip()
        return text

    def evaluate_instructions_pool_valid(self, instructions_pool: [str]) -> [float]:
        print('\n\nEvaluating instructions on validation set...')
        scores = []
        for instructions in tqdm(instructions_pool, total=len(instructions_pool), desc='*** Evaluating instructions valid'):
            prompt_template = self.get_prompt_tempate_from_instructions(instructions)
            score = self.task.evaluate_prompt_template_valid(prompt_template=prompt_template, num_samples=-1)
            scores.append(score)
        return scores

    def evaluate_instructions_pool_test(self, instructions_pool: [str]) -> [float]:
        print('\n\nEvaluating instructions on test set...')
        scores = []
        for instructions in tqdm(instructions_pool, total=len(instructions_pool), desc='*** Evaluating instructions test'):
            prompt_template = self.get_prompt_tempate_from_instructions(instructions)
            score = self.task.evaluate_prompt_template_test(prompt_template=prompt_template, num_samples=-1)
            scores.append(score)
        return scores

    def generate_permuted_instructions(self, instructions: [str], scale_factor: int) -> [str]:
        all_permuted_instructions = list(list(curr_instructions) for curr_instructions in permutations(instructions))
        random.shuffle(all_permuted_instructions)
        selected_permuted_instructions = []
        for permuted_instructions in all_permuted_instructions:
            if permuted_instructions == instructions:
                continue
            selected_permuted_instructions.append(permuted_instructions)
        return selected_permuted_instructions[:min(scale_factor, len(all_permuted_instructions))]

    def generate_permuted_instructions_from_instructions_pool(self, instructions_pool: [[str]]) -> [[str]]:
        scale_factor = self.generate_permuted_instructions_scale_factor
        permuted_instructions_pool = []
        for instructions in tqdm(instructions_pool, total=len(instructions_pool), desc='Generating permuted instructions'):
            curr_permuted_instructions = self.generate_permuted_instructions(instructions=instructions, scale_factor=scale_factor)
            permuted_instructions_pool.extend(curr_permuted_instructions)
        #permuted_instructions_pool = list(set(permuted_instructions_pool))
        return permuted_instructions_pool

    def rephrase_single_instruction(self, instruction: str) -> str: # To be deprecated
        meta_prompt = (f'Generate a variation of the following instruction while keeping the semantic meaning, updated instruction must be no more than two sentences:\n\n'                       
                       f'Instruction:{instruction}\n\n'
                       f'Updated instruction:\n\n:')
        rephrased_instruction = self.run_prompt_builder_llm(meta_prompt)
        return rephrased_instruction

    def rephrase_random_instructions(self, instructions: [str], num_samples: int) -> [str]:
        idxs = [i for i in range(len(instructions))]
        random.shuffle(idxs)
        idxs = idxs[:min(num_samples, len(idxs))]
        rephrased_instructions = instructions.copy()
        for idx in idxs:
            rephrased_instructions[idx] = self.rephrase_single_instruction(instruction=instructions[idx])
        return rephrased_instructions

    def rephrase_random_instructions_from_instructions_pool(self, instructions_pool: [[str]]) -> [[str]]:
        num_samples = self.rephrase_random_instructions_num_samples
        scale_factor = self.rephrase_random_instructions_scale_factor
        rephrased_instructions_pool = []
        for _ in tqdm(range(scale_factor), total=scale_factor, desc='Rephrase random instructions'):
            for instructions in instructions_pool:
                rephrased_instructions = self.rephrase_random_instructions(instructions=instructions, num_samples=num_samples)
                rephrased_instructions_pool.append(rephrased_instructions)
        return rephrased_instructions_pool

    def induce_instructions(self, input_texts: [str], output_texts: [str], gold_output_texts: [str]) -> [str]:
        meta_prompts = []
        for input_text, output_text, gold_output_text in zip(input_texts, output_texts, gold_output_texts):
            meta_prompt = self.meta_prompt_induce_instruction_template
            meta_prompt = meta_prompt.replace('<input_text>', input_text)
            meta_prompt = meta_prompt.replace('<output_text>', output_text)
            meta_prompt = meta_prompt.replace('<output_example>', gold_output_text)
            meta_prompts.append(meta_prompt)
        instructions = self.run_llm_prompt_builder_batch(meta_prompts)
        instructions = self.postprocess_texts(instructions)
        return instructions

    def run_evaluate_train_batch(self, instructions: [str], train_batch: dict):
        train_inputs, train_gold_outputs = self.task.extract_inputs_outputs_train(data=train_batch)
        prompt_template = self.get_prompt_tempate_from_instructions(instructions=instructions)
        prompts = [self.task.fill_prompt_template(prompt_template, train_input) for train_input in train_inputs]
        train_outputs = self.task.run_backend_llm_batch(prompts=prompts)

        train_errors = [levenstein_distance(output_text, gold_output_text) for output_text, gold_output_text in
                        zip(train_outputs, train_gold_outputs)]
        sorted_train_inputs, sorted_train_outputs, sorted_train_gold_outputs, sorted_train_errors = (
            zip(*sorted(zip(train_inputs, train_outputs, train_gold_outputs, train_errors),
                        key=lambda x: x[3], reverse=True)))

        mean_train_error = sum(train_errors) / len(train_errors)
        return sorted_train_inputs, sorted_train_outputs, sorted_train_gold_outputs, sorted_train_errors, mean_train_error

    def get_train_log_str(self, train_inputs: [str], train_outputs: [str], train_gold_outputs: [str],
                          train_errors: [float], train_error: float, start_no: int = 1):
        log_str = ''
        for n, (train_input, train_output, train_gold_output, train_error) in enumerate(zip(train_inputs, train_outputs, train_gold_outputs, train_errors)):
            log_str += f'Input {n + start_no}: \"{train_input}\"\n'
            log_str += f'System\'s Output {n + start_no}: \"{train_output}\"\n'
            log_str += f'Gold Output {n + start_no}: \"{train_gold_output}\"\n'
            log_str += f'Error {n + start_no} between System\'s Output {n + start_no} and Gold Output {n + start_no} for given Input {n + start_no}: {train_error} different words.\n\n'

        log_str += f'Mean error for examples {start_no}-{start_no + len(train_inputs) - 1}: {train_error:.2f} words.'
        return log_str

    def get_feedbacks_meta_prompt(self, train_log_str: str, instructions: [str]):
        meta_prompt = f'You are a super-talented prompt engineer. You are working on improvement of the {self.task_description} System.\n'
        meta_prompt += f'The System has these Instructions:\n'
        for instruction in instructions:
            meta_prompt += f'* {instruction}\n'
        meta_prompt += '\nBelow are the examples of System\'s work:\n'
        meta_prompt += train_log_str
        meta_prompt += f'\n\nSuggest new instruction to augment existing instructions forcing the System\'s Outputs to be exactly the same as Gold Outputs for the given System\'s Inputs. You need to minimize Errors between System\'s Outputs and Gold Outputs.'
        meta_prompt += f' Put new instruction between <new_instruction> and </new_instruction> tags. Do not use no more than two sentences. Do not mention Gold Output. Do not use "newline" symbols in your answer. Prioritize fixing cases which have larger error (which have more different words).' #  Do not mention directly any part of the considered texts.
        return meta_prompt

    def get_updated_instructions_feedback(self, instructions: [str]):
        train_batch = self.task.sample_random_train_data(num_samples=self.train_batch_size)
        best_instructions = instructions.copy()

        train_inputs, train_outputs, train_gold_outputs, train_errors, train_error_before = (
            self.run_evaluate_train_batch(instructions=instructions, train_batch=train_batch))

        best_train_error = train_error_before

        train_log_str = self.get_train_log_str(train_inputs=train_inputs,
                                               train_outputs=train_outputs,
                                               train_gold_outputs=train_gold_outputs,
                                               train_errors=train_errors,
                                               train_error=train_error_before)

        meta_prompt = self.get_feedbacks_meta_prompt(train_log_str, instructions=instructions)
        self.logger_feedbacks.debug(meta_prompt)

        for i in range(self.instructions_feedback_num_trials):
            new_instruction_output = self.run_prompt_builder_llm(prompt=meta_prompt)
            # new_instruction_output = self.run_prompt_builder_llm(prompt=meta_prompt, model_name='deepseek-r1-distill-qwen-7b')
            #new_instruction_output = self.run_prompt_builder_llm(prompt=meta_prompt, model_name='o1-mini')

            new_instruction = extract_tagged_text(text=new_instruction_output,
                                                  begin_tag='<new_instruction>',
                                                  end_tag='</new_instruction>')

            candidate_instructions = instructions.copy()
            candidate_instructions.insert(random.randint(0, len(candidate_instructions)), new_instruction)

            _, _, _, _, train_error_after = (
                self.run_evaluate_train_batch(instructions=candidate_instructions, train_batch=train_batch))

            msg = ''
            if train_error_after < best_train_error:
                best_train_error = train_error_after
                best_instructions = candidate_instructions
                #msg = '[BEST]'
            #print(f'\ni={i} of {self.instructions_feedback_num_trials - 1} len(best_instructions) = {len(best_instructions)} | {train_error_before:.2f} -> {train_error_after:.2f} {msg}')

        return best_instructions

    def improve_instructions_feedback_from_instructions_pool(self, instructions_pool: [[str]]) -> [[str]]:
        instructions_pool = instructions_pool * self.improve_by_feedback_scale_factor
        improved_instructions_pool = []
        for instructions in tqdm(instructions_pool, total=len(instructions_pool), desc='!!! Improving feedbacks instructions pool'):
            updated_instructions = self.get_updated_instructions_feedback(instructions=instructions)
            if len(updated_instructions) > len(instructions):
                improved_instructions_pool.append(updated_instructions)
        return improved_instructions_pool

    def generate_prompt(self, prompt_data: [dict]) -> dict:
        instructions_pool = [d['instructions'] for d in prompt_data]
        initial_valid_score = -1000000
        initial_instructions = instructions_pool[0]
        best_valid_score = -1000000
        best_instructions = initial_instructions
        time_now_str = datetime.datetime.now().strftime('%m_%d_%H-%M-%S')

        for iter_no in range(self.num_optimization_iterations + 1):
            curr_prompt_templates = [self.get_prompt_tempate_from_instructions(curr_instructions) for curr_instructions in instructions_pool]
            curr_valid_scores = self.task.evaluate_prompt_templates_valid(prompt_templates=curr_prompt_templates, return_outputs=False)

            instructions_pool, curr_valid_scores =  map(list, (zip(*sorted(zip(instructions_pool, curr_valid_scores),
                                                                key=lambda x: x[1], reverse=True))))

            self.logger.info(f'\niter_no = {iter_no:02}, valid_scores = {curr_valid_scores}')
            self.logger.info('-'*80)

            if iter_no == 0:
                initial_valid_score = curr_valid_scores[0]

            self.logger.info(f'init_valid_score = {initial_valid_score:.2f}')

            if len(instructions_pool) >= self.beam_size:
                instructions_pool = instructions_pool[:self.beam_size]
                curr_valid_scores = curr_valid_scores[:self.beam_size]

            best_msg = ''
            if curr_valid_scores[0] > best_valid_score:
                best_valid_score = curr_valid_scores[0]
                best_instructions = instructions_pool[0]
                best_msg = '[ *** BEST *** ]'

            self.logger.info(f'best_valid_score = {best_valid_score:.2f} {best_msg}')

            self.logger.info('-' * 80)
            self.logger.info(f'best_instructions = {json.dumps(best_instructions, indent=4)}')
            self.logger.info('-' * 80)

            improved_instructions_pool = self.improve_instructions_feedback_from_instructions_pool(instructions_pool=instructions_pool)
            rephrased_instructions_pool = self.rephrase_random_instructions_from_instructions_pool(instructions_pool=instructions_pool)
            permuted_instructions_pool = self.generate_permuted_instructions_from_instructions_pool(instructions_pool=instructions_pool)

            instructions_pool.extend(improved_instructions_pool)
            instructions_pool.extend(rephrased_instructions_pool)
            instructions_pool.extend(permuted_instructions_pool)

            # Save current iteration
            prompt_template = self.get_prompt_tempate_from_instructions(instructions=best_instructions)

            prompt_data = dict()
            prompt_data['config'] = self.config
            prompt_data['PROMPT_TEMPLATE'] = prompt_template
            prompt_data['initial_instructions'] = initial_instructions
            prompt_data['initial_valid_score'] = initial_valid_score
            prompt_data['instructions'] = best_instructions
            prompt_data['best_valid_score'] = best_valid_score

            debug_dir_name = os.path.join(os.path.dirname(self.json_filename), 'optimized_prompts_debug/')
            Path(debug_dir_name).mkdir(parents=True, exist_ok=True)
            with open(f'{debug_dir_name}/{time_now_str}_iter_{iter_no:02}_valid_{best_valid_score:.2f}.json', 'w') as f:
                json.dump(prompt_data, f, indent=4)

        return prompt_data
