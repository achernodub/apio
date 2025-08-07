import json
import os.path
import random
from abc import abstractmethod
from easse.sari import corpus_sari
from pathlib import Path
from utils import *
from llm_utils import *

nltk.download('punkt')


class Task:
    def __init__(self, config: dict):
        self.config = config
        self.task_name = config['main']['task_name']
        self.use_llm_proxy = config['main']['use_llm_proxy']
        self.sample_num_train = config['main']['sample_num_train']
        self.sample_num_valid = config['main']['sample_num_valid']
        self.sample_num_test = config['main']['sample_num_test']
        self.train_reference_no = 0

    @abstractmethod
    def get_train_data(self, index_from=0, index_to=-1) -> dict:
        pass

    @abstractmethod
    def get_valid_data(self, index_from=0, index_to=-1) -> dict:
        pass

    @abstractmethod
    def get_test_data(self, index_from=0, index_to=-1) -> dict:
        pass

    def sample_random_train_data(self, num_samples: int) -> dict:
        train_data = self.get_train_data()
        num_data = len(train_data['orig_sents'])
        idxs = [i for i in range(num_data)]
        random.shuffle(idxs)
        idxs = idxs[:num_samples]
        train_data['orig_sents'] = [train_data['orig_sents'][i] for i in idxs]
        for ref_no, curr_refs_sents in enumerate(train_data['refs_sents']):
            train_data['refs_sents'][ref_no] = [curr_refs_sents[i] for i  in idxs]

        if self.task_name in ['gec_jfleg', 'gec_bea']:  ########## Quick hack, to fix
            train_data['orig_sents_tok'] = [train_data['orig_sents_tok'][i] for i in idxs]
            for ref_no, curr_refs_sents in enumerate(train_data['refs_sents_detok']):
                train_data['refs_sents_detok'][ref_no] = [curr_refs_sents[i] for i  in idxs]

        return train_data

    def extract_inputs_outputs_train(self, data: dict):
        input_texts = data['orig_sents']
        if self.task_name in ['gec_jfleg', 'gec_bea']: ########## Quick hack, to fix
            output_texts = data['refs_sents_detok'][self.train_reference_no]
        else:
            output_texts = data['refs_sents'][self.train_reference_no]
        return input_texts, output_texts

    def sample_random_train_data_inputs_outputs(self, num_samples: int) -> (list[str], list[str]):
        train_data = self.sample_random_train_data(num_samples)
        input_texts, output_texts = self.extract_inputs_outputs_train(train_data)
        return input_texts, output_texts

    def fill_prompt_template(self, prompt_template, text) -> str:
        prompt = prompt_template.replace('<input_text>', text)
        return prompt

    @abstractmethod
    def postprocess_text(self, text: str) -> str:
        pass

    def postprocess_texts(self, texts: [str]) -> [str]:
        return [self.postprocess_text(text) for text in texts]

    def run_backend_llm_batch(self, prompts: [str]) -> [str]:
        outputs = llm_batch(self.use_llm_proxy,
                            prompts,
                            model_name=self.config['backend_llm']['model_name'],
                            temperature=self.config['backend_llm']['temperature'],
                            max_output_tokens=self.config['backend_llm']['max_output_tokens'],
                            batch_size=self.config['backend_llm']['num_threads'],
                            top_p=self.config['backend_llm']['top_p'],
                            verbose=True)
        return self.postprocess_texts(outputs)

    def run_prompt_builder_llm_batch(self, prompts: [str]) -> [str]:
        outputs = llm_batch(self.use_llm_proxy,
                            prompts,
                            model_name=self.config['backend_llm']['model_name'],
                            temperature=self.config['backend_llm']['temperature'],
                            max_output_tokens=self.config['backend_llm']['max_output_tokens'],
                            batch_size=self.config['backend_llm']['num_threads'],
                            top_p=self.config['backend_llm']['top_p'],
                            verbose=True)
        return self.postprocess_texts(outputs)

    def evaluate_prompt_template(self, prompt_template: str, data: dict, return_outputs: bool = False):
        prompts = [self.fill_prompt_template(prompt_template, sent) for sent in data['orig_sents']]
        outputs = self.run_backend_llm_batch(prompts)
        score = self.evaluate_outputs(outputs=outputs, data=data)
        if return_outputs:
            return score, outputs
        return score

    def evaluate_prompt_templates(self, prompt_templates: [str], data: dict, return_outputs: bool = False):
        all_prompts = []
        for prompt_template in prompt_templates:
            for sent in data['orig_sents']:
                all_prompts.append(self.fill_prompt_template(prompt_template, sent))
        all_outputs = self.run_backend_llm_batch(all_prompts)
        outputs_list = []
        scores = []
        outputs_size = len(data['orig_sents'])
        for prompt_no in range(len(prompt_templates)):
            i = prompt_no * outputs_size
            j = (prompt_no + 1) * outputs_size
            outputs = all_outputs[i:j]
            score = self.evaluate_outputs(outputs=outputs, data=data)
            outputs_list.append(outputs)
            scores.append(score)
        if return_outputs:
            return scores, outputs_list
        return scores

    def evaluate_prompt_templates_valid(self, prompt_templates: [str], return_outputs: bool = False) -> [float]:
        data = self.get_valid_data()
        return self.evaluate_prompt_templates(prompt_templates, data, return_outputs)

    @abstractmethod
    def evaluate_outputs(self, outputs: [str], data: dict) -> float:
        pass

    def evaluate_prompt_template_train(self, prompt_template: str, num_samples: int = -1, return_outputs: bool = False) -> float:
        data = self.get_train_data(index_to=num_samples)
        return self.evaluate_prompt_template(prompt_template, data, return_outputs)

    def evaluate_prompt_template_valid(self, prompt_template: str, num_samples: int = -1, return_outputs: bool = False) -> float:
        data = self.get_valid_data(index_to=num_samples)
        return self.evaluate_prompt_template(prompt_template, data, return_outputs)

    def evaluate_prompt_template_test(self, prompt_template: str, num_samples: int = -1, return_outputs: bool = False) -> float:
        data = self.get_test_data(index_to=num_samples)
        return self.evaluate_prompt_template(prompt_template, data, return_outputs)


class TaskTextSimplification(Task):
    def __init__(self, config: dict):
        super().__init__(config)
        self.load_data()
        self.train_reference_no = 7 #### the best performing on SARI reference in Asset-2K (valid)

    def load_data(self):
        with open('text_simplification/data/valid_asset_2K.json') as f:
            self.train_data = json.load(f)[75:]
            if self.sample_num_train > 0:
                self.train_data = self.train_data[:self.sample_num_train]

        with open('text_simplification/data/valid_asset_2K.json') as f:
            self.valid_data = json.load(f)[:75]
            if self.sample_num_valid > 0:
                self.valid_data = self.train_data[:self.sample_num_valid]

        with open('text_simplification/data/test_asset_359.json') as f:
            self.test_data = json.load(f)
            if self.sample_num_test > 0:
                self.test_data = self.test_data[:self.sample_num_test]

        self.train_orig_sents, self.train_refs_sents = self._extract_data_asset(self.train_data, ref_num=10)
        self.valid_orig_sents, self.valid_refs_sents = self._extract_data_asset(self.valid_data, ref_num=10)
        self.test_orig_sents, self.test_refs_sents = self._extract_data_asset(self.test_data, ref_num=10)

    def get_train_data(self, index_from=0, index_to=-1) -> dict:
        if index_to < 0:
            index_to = len(self.train_orig_sents)
        orig_sents = self.train_orig_sents[index_from:index_to]
        refs_sents = self._bound_refs_sents_asset(self.train_refs_sents, index_from, index_to)
        return {'orig_sents': orig_sents, 'refs_sents': refs_sents}

    def get_valid_data(self, index_from=0, index_to=-1) -> dict:
        if index_to < 0:
            index_to = len(self.valid_orig_sents)
        orig_sents = self.valid_orig_sents[index_from:index_to]
        refs_sents = self._bound_refs_sents_asset(self.valid_refs_sents, index_from, index_to)
        return {'orig_sents': orig_sents, 'refs_sents': refs_sents}

    def get_test_data(self, index_from=0, index_to=-1) -> dict:
        if index_to < 0:
            index_to = len(self.test_orig_sents)
        orig_sents = self.test_orig_sents[index_from:index_to]
        refs_sents = self._bound_refs_sents_asset(self.test_refs_sents, index_from, index_to)
        return {'orig_sents': orig_sents, 'refs_sents': refs_sents}

    def _extract_data_asset(self, text_simplification_data: [dict], ref_num: int) -> [dict]:
        orig_sents = []
        refs_sents = [[] for _ in range(ref_num)]
        for d in text_simplification_data:
            orig_sents.append(d['text'])
            for i, curr_refs_sents in enumerate(refs_sents):
                curr_refs_sents.append(d[f'ref{i}'])
        return orig_sents, refs_sents

    def _bound_refs_sents_asset(self, refs_sents_input, index_from, index_to):
        refs_sents = []
        for curr_refs_sents in refs_sents_input:
            refs_sents.append(curr_refs_sents[index_from:index_to])
        return refs_sents

    def postprocess_text(self, text: str) -> str:
        text = text.replace('\n\n', '')
        text = text.strip()
        text = text.strip('\"')

        extra_strings = ['Answer:',
                         'A simplified version of the input sentence is:',
                         'A simplified version of the input sentence is:',
                         'Simplified sentence:',
                         'Simplified answer:',
                         'Simple sentence:',
                         'The simplified sentence is:',
                         'Simplified input sentence:',
                         'Simplified:',
                         'Simplification:',
                         'Simplified input:',
                         "The answer simplifies the given sentence by removing unnecessary details and focusing on the main point. Here's a simplified version:Simplified",
                         'Simplified Sentence:',
                         'Simpler Sentence:',
                         'Simple answer:',
                         'Simplified sentence:',
                         'Simplified answer:',
                         'Simple sentence:',
                         'The simplified sentence is:',
                         'Simplified input sentence:',
                         'Simplified:',
                         'Simplification:',
                         'Simplified input:',
                         "The answer simplifies the given sentence by removing unnecessary details and focusing on the main point. Here's a simplified version:Simplified",
                         'Simplified Sentence:',
                         'Simpler Sentence:',
                         'Simple answer:',
                         'The sentence simplifies to:',
                         'Here\'s a simplified version of the sentence:',
                         'The instructions you\'ve provided are for simplifying text, but they\'re not clear about the specific steps or rules to follow. Here\'s a simplified version of your sentence:',
                         'Simplify by stating the action directly:',
                         'The simplified answer would be:',
                         'Input sentence simplified:',
                         'Certainly! Here\'s a simplified version of the sentence:',
                         'Output simplified sentence:',
                         'Simplify the sentence by removing unnecessary words and keeping only essential components:',
                         'The sentence could be simplified as follows:',
                         'To simplify the sentence, you can break it down into smaller parts and make each part easier to understand. Here\'s a simplified version:Original sentence:',
                         'The sentence simplifies to:',
                         'The complex sentence can be simplified as follows:',
                         'Simple Sentence:',
                         'The sentence you provided is already quite simple and clear. However, if the goal was to simplify it further while maintaining clarity, you might consider:',
                         'The sentence you provided is already quite simple and clear. However, if the goal is to simplify it further while maintaining clarity, you could consider:',
                         'The input is a simple statement, not a complex sentence. However, if you\'re asking for simplification, the answer could be:',
                         'The sentence you provided is already quite simple and doesn\'t require simplification. However, if the goal is to make it even more straightforward while maintaining clarity, you could consider rephrasing it as follows:',
                         'The simple sentence is:',
                         'The sentence in simpler terms could be:',
                         'Output:',
                         'The answer is:',
                         'The input sentence simplified is:',
                         'Simplify the input sentence:',
                         'The simplified sentence could be:',
                         'Here is the simplified version of the input sentence:',
                         'The input sentence simplifies to:',
                         ]
        for s in extra_strings:
            text = text.replace(s, '')

        text = text.strip()
        return text

    def evaluate_outputs(self, outputs: [str], data: dict) -> float:
        score = corpus_sari(sys_sents=outputs, orig_sents=data['orig_sents'], refs_sents=data['refs_sents'])
        return score


class TaskGecJfleg(Task):
    def __init__(self, config: dict):
        super().__init__(config)
        self.load_data()
        self.train_reference_no = 0

    def _read_lines(self, short_filename: str) -> str:
        return Path(f'gec_jfleg/data/{short_filename}').read_text().splitlines()

    def load_data(self):
        dev_ref0 = self._read_lines('dev.ref0')
        dev_ref1 = self._read_lines('dev.ref1')
        dev_ref2 = self._read_lines('dev.ref2')
        dev_ref3 = self._read_lines('dev.ref3')
        dev_src = self._read_lines('dev.src')
        dev_ref0_detok  = self._read_lines('dev.ref0.detok')
        dev_ref1_detok  = self._read_lines('dev.ref1.detok')
        dev_ref2_detok  = self._read_lines('dev.ref2.detok')
        dev_ref3_detok  = self._read_lines('dev.ref3.detok')
        dev_src_detok = self._read_lines('dev.src.detok')

        num_split_train_valid = 554
        #num_split_train_valid = 500 # good but expensive
        train_ref0 = dev_ref0[:num_split_train_valid]
        train_ref1 = dev_ref1[:num_split_train_valid]
        train_ref2 = dev_ref2[:num_split_train_valid]
        train_ref3 = dev_ref3[:num_split_train_valid]
        train_ref0_detok = dev_ref0_detok[:num_split_train_valid]
        train_ref1_detok = dev_ref1_detok[:num_split_train_valid]
        train_ref2_detok = dev_ref2_detok[:num_split_train_valid]
        train_ref3_detok = dev_ref3_detok[:num_split_train_valid]
        train_src = dev_src[:num_split_train_valid]
        train_src_detok = dev_src_detok[:num_split_train_valid]

        valid_ref0_detok = dev_ref0_detok[num_split_train_valid:]
        valid_ref1_detok = dev_ref1_detok[num_split_train_valid:]
        valid_ref2_detok = dev_ref2_detok[num_split_train_valid:]
        valid_ref3_detok = dev_ref3_detok[num_split_train_valid:]
        valid_ref0 = dev_ref0[num_split_train_valid:]
        valid_ref1 = dev_ref1[num_split_train_valid:]
        valid_ref2 = dev_ref2[num_split_train_valid:]
        valid_ref3 = dev_ref3[num_split_train_valid:]
        valid_src = dev_src[num_split_train_valid:]
        valid_src_detok = dev_src_detok[num_split_train_valid:]

        test_ref0_detok = self._read_lines('test.ref0.detok')
        test_ref1_detok = self._read_lines('test.ref1.detok')
        test_ref2_detok = self._read_lines('test.ref2.detok')
        test_ref3_detok = self._read_lines('test.ref3.detok')
        test_ref0 = self._read_lines('test.ref0')
        test_ref1 = self._read_lines('test.ref1')
        test_ref2 = self._read_lines('test.ref2')
        test_ref3 = self._read_lines('test.ref3')
        test_src = self._read_lines('test.src')
        test_src_detok = self._read_lines('test.src.detok')

        if self.sample_num_train > 0:
            train_ref0 = valid_ref0[:self.sample_num_train]
            train_ref1 = valid_ref1[:self.sample_num_train]
            train_ref2 = valid_ref2[:self.sample_num_train]
            train_ref3 = valid_ref3[:self.sample_num_train]
            train_ref0_detok = valid_ref0_detok[:self.sample_num_train]
            train_ref1_detok = valid_ref1_detok[:self.sample_num_train]
            train_ref2_detok = valid_ref2_detok[:self.sample_num_train]
            train_ref3_detok = valid_ref3_detok[:self.sample_num_train]
            train_src = valid_src[:self.sample_num_train]
            train_src_detok = valid_src_detok[:self.sample_num_train]

        if self.sample_num_valid > 0:
            valid_ref0_detok = valid_ref0_detok[:self.sample_num_valid]
            valid_ref1_detok = valid_ref1_detok[:self.sample_num_valid]
            valid_ref2_detok = valid_ref2_detok[:self.sample_num_valid]
            valid_ref3_detok = valid_ref3_detok[:self.sample_num_valid]
            valid_ref0 = valid_ref0[:self.sample_num_valid]
            valid_ref1 = valid_ref1[:self.sample_num_valid]
            valid_ref2 = valid_ref2[:self.sample_num_valid]
            valid_ref3 = valid_ref3[:self.sample_num_valid]
            valid_src = valid_src[:self.sample_num_valid]
            valid_src_detok = valid_src_detok[:self.sample_num_valid]

        if self.sample_num_valid > 0:
            test_ref0_detok = test_ref0_detok[:self.sample_num_test]
            test_ref1_detok = test_ref1_detok[:self.sample_num_test]
            test_ref2_detok = test_ref2_detok[:self.sample_num_test]
            test_ref3_detok = test_ref3_detok[:self.sample_num_test]
            test_ref0 = test_ref0[:self.sample_num_test]
            test_ref1 = test_ref1[:self.sample_num_test]
            test_ref2 = test_ref2[:self.sample_num_test]
            test_ref3 = test_ref3[:self.sample_num_test]
            test_src = test_src[:self.sample_num_test]
            test_src_detok = test_src_detok[:self.sample_num_test]

        self.train_orig_sents = train_src_detok
        self.train_orig_sents_tok = train_src
        self.train_refs_sents = [train_ref0, train_ref1, train_ref2, train_ref3]
        self.train_refs_sents_detok = [train_ref0_detok, train_ref1_detok, train_ref2_detok, train_ref3_detok]

        self.valid_orig_sents = valid_src_detok
        self.valid_orig_sents_tok = valid_src
        self.valid_refs_sents = [valid_ref0, valid_ref1, valid_ref2, valid_ref3]
        self.valid_refs_sents_detok = [valid_ref0_detok, valid_ref1_detok, valid_ref2_detok, valid_ref3_detok]

        self.test_orig_sents = test_src_detok
        self.test_orig_sents_tok = test_src
        self.test_refs_sents = [test_ref0, test_ref1, test_ref2, test_ref3]
        self.test_refs_sents_detok = [test_ref0_detok, test_ref1_detok, test_ref2_detok, test_ref3_detok]

    def _bound_refs_sents_asset(self, refs_sents_input, index_from, index_to):
        refs_sents = []
        for curr_refs_sents in refs_sents_input:
            refs_sents.append(curr_refs_sents[index_from:index_to])
        return refs_sents

    def get_train_data(self, index_from=0, index_to=-1) -> dict:
        if index_to < 0:
            index_to = len(self.train_orig_sents)
        orig_sents = self.train_orig_sents[index_from:index_to]
        orig_sents_tok = self.train_orig_sents_tok[index_from:index_to]
        refs_sents = self._bound_refs_sents_asset(self.train_refs_sents, index_from, index_to)
        refs_sents_detok = self._bound_refs_sents_asset(self.train_refs_sents_detok, index_from, index_to)
        return {'orig_sents': orig_sents, 'orig_sents_tok': orig_sents_tok, 'refs_sents': refs_sents, 'refs_sents_detok': refs_sents_detok}

    def get_valid_data(self, index_from=0, index_to=-1) -> dict:
        if index_to < 0:
            index_to = len(self.valid_orig_sents)
        orig_sents = self.valid_orig_sents[index_from:index_to]
        orig_sents_tok = self.valid_orig_sents_tok[index_from:index_to]
        refs_sents = self._bound_refs_sents_asset(self.valid_refs_sents, index_from, index_to)
        refs_sents_detok = self._bound_refs_sents_asset(self.valid_refs_sents_detok, index_from, index_to)
        return {'orig_sents': orig_sents, 'orig_sents_tok': orig_sents_tok, 'refs_sents': refs_sents, 'refs_sents_detok': refs_sents_detok}

    def get_test_data(self, index_from=0, index_to=-1) -> dict:
        if index_to < 0:
            index_to = len(self.test_orig_sents)
        orig_sents = self.test_orig_sents[index_from:index_to]
        orig_sents_tok = self.test_orig_sents_tok[index_from:index_to]
        refs_sents = self._bound_refs_sents_asset(self.test_refs_sents, index_from, index_to)
        refs_sents_detok = self._bound_refs_sents_asset(self.test_refs_sents_detok, index_from, index_to)
        return {'orig_sents': orig_sents, 'orig_sents_tok': orig_sents_tok, 'refs_sents': refs_sents, 'refs_sents_detok': refs_sents_detok}

    def sample_train_data_random(self, num_samples: int) -> list:
        train_data = self.get_train_data()
        input_texts = train_data['orig_sents']
        output_texts = train_data['refs_sents_detok'][3] #### the best reference in jfleg-dev
        idxs = [i for i in range(len(input_texts))]
        random.shuffle(idxs)
        idxs = idxs[:num_samples]
        input_texts = [input_texts[i] for i in idxs]
        output_texts = [output_texts[i] for i in idxs]
        return input_texts, output_texts

    def postprocess_text(self, text: str) -> str:
        text = text.replace('\n\n', '')
        text = text.strip()
        text = text.strip('\"')

        extra_strings = ['Answer:',
                         'A simplified version of the input sentence is:',
                         'Sure! Here\u2019s a general instruction for the task:',
                         'Task:',
                         ]
        for s in extra_strings:
            text = text.replace(s, '')

        #text = tokenize_text(text) ################################
        text = text.strip()
        return text

    #def evaluate_jfleg_test(self, filename_out, top100=False):
    #    time.sleep(1.0)
    #    # cmd = 'python3 jfleg/eval/gleu.py --src jfleg/test/test.src --ref jfleg/test/test.ref[0-3] --hyp %s' % filename_out
    #    if not top100:
    #        cmd = 'python3 gleu_augmented.py --src test.src --ref test.ref[0-3] --hyp %s' % filename_out
    #    else:
    #        cmd = 'python3 gleu_augmented.py --src test.top100.src --ref test.top100.ref[0-3] --hyp %s' % filename_out
    #    results_str = os.popen(cmd).read()
    #    gleu_score = float(results_str.split('\'')[1]) * 100.0
    #    # python3 jfleg/eval/gleu.py --src jfleg/test/test.src --ref jfleg/test/test.ref[0-3] --hyp jfleg.test.top100.out.2024-05-14-11-58-Rsp.txt
    #    # [['0.613868', '0.023800', '(0.567,0.661)']]
    #    return gleu_score

    def evaluate_outputs(self, outputs: [str], data: dict) -> float:
        curr_dirname = os.path.dirname(os.path.abspath(__file__))
        rand_value = random.randint(0, 100500)
        temp_output_filename = f'{curr_dirname}/gec_jfleg/data/{rand_value}_outputs.txt'

        outputs = [tokenize_text(o) for o in outputs]
        Path(temp_output_filename).write_text('\n'.join(outputs))

        temp_source_filename = f'{curr_dirname}/gec_jfleg/data/{rand_value}_src.txt'
        Path(temp_source_filename).write_text('\n'.join(data['orig_sents_tok']))

        refs_sents_filenames = []
        refs_sents_str = ''
        for i, curr_refs_sents in enumerate(data['refs_sents']):
            curr_curr_refs_sents_filename = f'{curr_dirname}/gec_jfleg/data/{rand_value}_ref_{i}.txt'
            Path(curr_curr_refs_sents_filename).write_text('\n'.join(curr_refs_sents))
            refs_sents_filenames.append(curr_curr_refs_sents_filename)
            refs_sents_str += f'{curr_curr_refs_sents_filename} '

        time.sleep(2.0)
        cmd = f'python gec_jfleg/data/gleu.py --src {temp_source_filename} --ref {refs_sents_str} --hyp {temp_output_filename}'
        results_str = os.popen(cmd).read()
        gleu_score = float(results_str.split('\'')[1]) * 100.0

        os.remove(temp_output_filename)
        os.remove(temp_source_filename)
        for refs_sents_filename in refs_sents_filenames:
            os.remove(refs_sents_filename)

        return gleu_score


class TaskGecBea(Task):
    def __init__(self, config: dict):
        super().__init__(config)
        self.load_data()
        self.train_reference_no = 0

    def _read_lines(self, short_filename: str) -> str:
        return Path(f'gec_bea/data/{short_filename}').read_text().splitlines()

    def load_data(self):
        dev_src = self._read_lines('bea-dev.src')
        dev_src_detok = self._read_lines('bea-dev.src.detok')
        dev_ref0 = self._read_lines('bea-dev.ref0')
        dev_ref0_detok  = self._read_lines('bea-dev.ref0.detok')

        num_split_train_valid = 100
        train_ref0 = dev_ref0[num_split_train_valid:]
        train_ref0_detok = dev_ref0_detok[num_split_train_valid:]
        train_src = dev_src[num_split_train_valid:]
        train_src_detok = dev_src_detok[num_split_train_valid:]

        valid_ref0_detok = dev_ref0_detok[:num_split_train_valid]
        valid_ref0 = dev_ref0[:num_split_train_valid]
        valid_src = dev_src[:num_split_train_valid]
        valid_src_detok = dev_src_detok[:num_split_train_valid]

        test_src = self._read_lines('bea-test.src')
        test_src_detok = self._read_lines('bea-test.src.detok')

        if self.sample_num_train > 0:
            train_ref0 = valid_ref0[:self.sample_num_train]
            train_ref0_detok = valid_ref0_detok[:self.sample_num_train]
            train_src = valid_src[:self.sample_num_train]
            train_src_detok = valid_src_detok[:self.sample_num_train]

        if self.sample_num_valid > 0:
            valid_ref0_detok = valid_ref0_detok[:self.sample_num_valid]
            valid_ref0 = valid_ref0[:self.sample_num_valid]
            valid_src = valid_src[:self.sample_num_valid]
            valid_src_detok = valid_src_detok[:self.sample_num_valid]

        if self.sample_num_test > 0:
            test_src = test_src[:self.sample_num_test]
            test_src_detok = test_src_detok[:self.sample_num_test]

        self.train_orig_sents = train_src_detok
        self.train_orig_sents_tok = train_src
        self.train_refs_sents = [train_ref0]
        self.train_refs_sents_detok = [train_ref0_detok]

        self.valid_orig_sents = valid_src_detok
        self.valid_orig_sents_tok = valid_src
        self.valid_refs_sents = [valid_ref0]
        self.valid_refs_sents_detok = [valid_ref0_detok]

        self.test_orig_sents = test_src_detok
        self.test_orig_sents_tok = test_src
        self.test_refs_sents = [test_src] # test data is not available for BEA-2019
        self.test_refs_sents_detok = [test_src_detok] # test data is not available for BEA-2019

    def _bound_refs_sents_asset(self, refs_sents_input, index_from, index_to):
        refs_sents = []
        for curr_refs_sents in refs_sents_input:
            refs_sents.append(curr_refs_sents[index_from:index_to])
        return refs_sents

    def get_train_data(self, index_from=0, index_to=-1) -> dict:
        if index_to < 0:
            index_to = len(self.train_orig_sents)
        orig_sents = self.train_orig_sents[index_from:index_to]
        orig_sents_tok = self.train_orig_sents_tok[index_from:index_to]
        refs_sents = self._bound_refs_sents_asset(self.train_refs_sents, index_from, index_to)
        refs_sents_detok = self._bound_refs_sents_asset(self.train_refs_sents_detok, index_from, index_to)
        return {'orig_sents': orig_sents, 'orig_sents_tok': orig_sents_tok, 'refs_sents': refs_sents, 'refs_sents_detok': refs_sents_detok}

    def get_valid_data(self, index_from=0, index_to=-1) -> dict:
        if index_to < 0:
            index_to = len(self.valid_orig_sents)
        orig_sents = self.valid_orig_sents[index_from:index_to]
        orig_sents_tok = self.valid_orig_sents_tok[index_from:index_to]
        refs_sents = self._bound_refs_sents_asset(self.valid_refs_sents, index_from, index_to)
        refs_sents_detok = self._bound_refs_sents_asset(self.valid_refs_sents_detok, index_from, index_to)
        return {'orig_sents': orig_sents, 'orig_sents_tok': orig_sents_tok, 'refs_sents': refs_sents, 'refs_sents_detok': refs_sents_detok}

    def get_test_data(self, index_from=0, index_to=-1) -> dict:
        if index_to < 0:
            index_to = len(self.test_orig_sents)
        orig_sents = self.test_orig_sents[index_from:index_to]
        orig_sents_tok = self.test_orig_sents_tok[index_from:index_to]
        refs_sents = self._bound_refs_sents_asset(self.test_refs_sents, index_from, index_to)
        refs_sents_detok = self._bound_refs_sents_asset(self.test_refs_sents_detok, index_from, index_to)
        return {'orig_sents': orig_sents, 'orig_sents_tok': orig_sents_tok, 'refs_sents': refs_sents, 'refs_sents_detok': refs_sents_detok}

    def sample_train_data_random(self, num_samples: int) -> list:
        train_data = self.get_train_data()
        input_texts = train_data['orig_sents']
        output_texts = train_data['refs_sents_detok'][0]
        idxs = [i for i in range(100, len(input_texts))] # hack: we don't sample first 100 samples for training, to keep the ability use them for independent test
        random.shuffle(idxs)
        idxs = idxs[:num_samples]
        input_texts = [input_texts[i] for i in idxs]
        output_texts = [output_texts[i] for i in idxs]
        return input_texts, output_texts

    def postprocess_text(self, text: str) -> str:
        text = text.replace('\n\n', '')
        text = text.strip()
        text = text.strip('\"')

        extra_strings = ['Answer:',
                         'Another version of the input sentence is:',
                         ]
        for s in extra_strings:
            text = text.replace(s, '')

        #text = tokenize_text(text)
        text = text.strip()
        return text


    def evaluate_outputs(self, outputs: [str], data: dict) -> float:
        curr_dirname = os.path.dirname(os.path.abspath(__file__))
        rand_value = random.randint(0, 100500)

        temp_source_filename = f'{curr_dirname}/gec_bea/data/{rand_value}_src.txt'
        Path(temp_source_filename).write_text('\n'.join(data['orig_sents_tok']))

        temp_output_filename = f'{curr_dirname}/gec_bea/data/{rand_value}_outputs.txt'
        outputs = [tokenize_text(o) for o in outputs]
        Path(temp_output_filename).write_text('\n'.join(outputs))

        temp_output_m2_filename = f'{curr_dirname}/gec_bea/data/{rand_value}_outputs_m2.txt'

        cmd = f'errant_parallel -orig {temp_source_filename} -cor {temp_output_filename} -out {temp_output_m2_filename}'
        os.popen(cmd).read()
        time.sleep(1.0)

        temp_ref_filename = f'{curr_dirname}/gec_bea/data/{rand_value}_ref0.txt'
        Path(temp_ref_filename).write_text('\n'.join(data['refs_sents'][0]))

        temp_ref_m2_filename = f'{curr_dirname}/gec_bea/data/{rand_value}_ref0_m2.txt'

        cmd = f'errant_parallel -orig {temp_source_filename} -cor {temp_ref_filename} -out {temp_ref_m2_filename}'
        os.popen(cmd).read()
        time.sleep(1.0)

        cmd = f'errant_compare -hyp {temp_output_m2_filename} -ref {temp_ref_m2_filename}'
        result_str = os.popen(cmd).read()

        f05_value = float(result_str.split('	')[-1].split('\n')[0])*100.0

        os.remove(temp_source_filename)
        os.remove(temp_ref_filename)
        os.remove(temp_ref_m2_filename)
        os.remove(temp_output_m2_filename)
        os.remove(temp_output_filename)
        return f05_value

def load_task(config: dict) -> Task:
    task_name = config['main']['task_name']
    if task_name == 'text_simplification':
         return TaskTextSimplification(config)
    if task_name == 'gec_jfleg':
         return TaskGecJfleg(config)
    if task_name == 'gec_bea':
         return TaskGecBea(config)
    else:
        raise ValueError('Invalid task name')
