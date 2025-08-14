# APIO: Automatic Prompt Induction and Optimization for Grammatical Error Correction and Text Simplification
## Introduction
This page contains the official implementation for the paper ["APIO: Automatic Prompt Induction and Optimization 
for Grammatical Error Correction and Text Simplification](https://arxiv.org/pdf/2508.09378), which has been accepted for publication at 
[RANLP 2025](https://ranlp.org/ranlp2025/). The repository includes prompts, data, code, and outputs used in the paper.

<p align="center">
  <a href="https://en.wikipedia.org/wiki/Apio_(appetizer)" title="Click to find out what APIO is">
    <img src="apio_pic.png" alt="Click to find out what APIO is" width="400px" height="400px">
  </a>
</p>

## Citation
Please cite our [paper](https://arxiv.org/pdf/2508.09378) if you use it in your research.

```
@article{chernodub2025apio,
  title={APIO: Automatic Prompt Induction and Optimization for Grammatical Error Correction and Text Simplification},
  author={Chernodub, Artem and Saini, Aman and Huh, Yejin and Kulkarni, Vivek and Raheja, Vipul},
  journal={arXiv preprint arXiv:2508.09378},
  year={2025},
  note={Accepted for publication at Recent Advances in Natural Language Processing conference (RANLP 2025)},
  url={https://arxiv.org/abs/2508.09378}
}
```

## Requirements

- Python 3.9 or higher

## Installation

```
pip install -r requirements.txt

python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt')"
python -c "import nltk; nltk.download('punkt_tab')"
```

## Prompts generation and evaluation

To generate the prompts, you need to run the script `generate.py` with task-specific parameters:
* `--config` - the path to YAML config file with predefined parameters;
* `--output_path` - the path to the output folders where generated prompts will be stored; if it doesn't exists, it 
will be created.

An example of generated folder after the work of `generate.py` script may look as follows:

```
|__ logs/ --> logs related to generation of prompts;
|__ optimized_prompts_debug/ --> intermediate prompts obtained during the prompt optimization;  

zero-shot.json --> generated zero-shot with meta-data;
zero-shot.txt --> pure zero-shot template;

02_13_12-38-55_few-shot_3.json --> generated few-shot prompt template with meta-data; 
02_13_12-38-55_few-shot_3.txt --> pure few-shot prompt template;

02_13_12-38-55_instruction_induction_3.json --> generated inducted from data prompt template with meta-data; 
02_13_12-38-55_instruction_induction_3.txt --> pure inducted from data prompt template;

02_13_12-38-55_optimized_3.json --> generated optimized inducted from data prompt template with meta-data;
02_13_12-38-55_optimized_3.txt --> pure optimized from data prompt template;
```

To evaluate generated prompt templates, please run the `evaluate.py` script with the following task-specific parameters: 
* `--config` - the path to YAML config file with predefined parameters;
* `--output_path` - the path with generated prompts: the script will search JSON files in the directory and evaluate them one by one.
* `--num_samples` (optional) - number of samples to evaluate. If `-1` is set, then evaluation is performed on all the data (by default);
* `--valid` (optional) - perform evaluation on the validation dataset (not on the test);
* `--train` (optional) - perform evaluation on the train dataset (not on the test).

After running the evaluation script, it will create the subfolders inside it: `evaluated_prompts_test`, 
`evaluated_prompts_train` or `evaluated_prompts_valid`, depending on which flags were used for generation.

An example of generated folder after the work of `evaluate.py` script may look as follows:
```
|__ logs/ --> logs related to generation of prompts;
|__ optimized_prompts_debug/ --> intermediate prompts obtained during the prompt optimization;
|__ evaluated_prompts_test/ --> files with evaluation results, each JSON filename now contains the evaluation score;
    |__ zero-shot_gpt-4o-mini_48.03.json --> prompt's JSON data with added evaluation results;
    |__ zero-shot_gpt-4o-mini_48.03.txt --> LLM outputs which were used for evaluation;
    |__ 02_13_12-38-55_few-shot_3_gpt-4o-mini_47.16.json --> prompt's JSON data with added evaluation results;
    |__ 02_13_12-38-55_few-shot_3_gpt-4o-mini_47.16.txt --> LLM outputs which were used for evaluation;
    |__ 02_13_12-38-55_instruction_induction_3_gpt-4o-mini_48.79.json --> prompt's JSON data with added evaluation results;
    |__ 02_13_12-38-55_instruction_induction_3_gpt-4o-mini_48.79.txt --> LLM outputs which were used for evaluation;
    |__ 02_13_12-38-55_optimized_3_gpt-4o-mini_49.27.json --> prompt's JSON data with added evaluation results;
    |__ 02_13_12-38-55_optimized_3_gpt-4o-mini_49.27.txt --> LLM outputs which were used for evaluation;
    |__ evaluation.log --> evaluation log.

zero-shot.json --> generated zero-shot with meta-data;
zero-shot.txt --> pure zero-shot template;

02_13_12-38-55_few-shot_3.json --> generated few-shot prompt template with meta-data; 
02_13_12-38-55_few-shot_3.txt --> pure few-shot prompt template;

02_13_12-38-55_instruction_induction_3.json --> generated inducted from data prompt template with meta-data; 
02_13_12-38-55_instruction_induction_3.txt --> pure inducted from data prompt template;

02_13_12-38-55_optimized_3.json --> generated optimized inducted from data prompt template with meta-data;
02_13_12-38-55_optimized_3.txt --> pure optimized from data prompt template;
```

## Tasks

### Grammatical Error Correction
The Grammatical Error Correction task focuses on correcting grammatical errors with minimal edits.

#### Dataset
We use BEA-2019-dev dataset (4384 samples) for train and validation and BEA-test (4477 samples) for test.
BEA-test is a hidden benchmark, so the code below only generates the tokenized outputs which should be submitted to 
BEA-2019 benchmark [Codalab website](https://codalab.lisn.upsaclay.fr/competitions/4057).

#### Evaluation

* validation: subset of BEA-2019-dev (100 samples); not used for training; used for model selection;
* mini-test: subset of BEA-2019-dev (99 samples); not used for training; not used for model selection;
* test dataset: BEA-2019-test (4479 samples); evaluation is performed on the official BEA-2019 CodaLab page; 
* prompts, optimization logs, and raw outputs: see in the folders `gpt-4o-mini_gec_bea` and `gpt-4o-mini_gec_bea`.

[[papers with code page](https://paperswithcode.com/sota/grammatical-error-correction-on-bea-2019-test)]
[[nlp progress page](https://nlpprogress.com/english/grammatical_error_correction.html)]
[[CodaLab page](https://codalab.lisn.upsaclay.fr/competitions/4057#participate)]


#### Running scripts
Running prompt generation
```
python generate_prompts.py --config gec_bea/config_gec_bea.yaml --output_path gec_bea/generated_prompts
```

Running evaluation on test set (BEA-test, 4477 samples), only generate output data
```
python evaluate_prompts.py --config gec_bea/config_gec_bea.yaml
```

Running evaluation on validation set (subset of BEA-dev, 100 samples)
```
python evaluate_prompts.py --config gec_bea/config_gec_bea.yaml --valid
```

To perform evaluation on 99 samples from the BEA-dev dataset which were not used for training or validation, run this code:
```
python evaluate_prompts.py -c gec_bea/config_gec_bea.yaml --train --num_samples 99
```

### Text Simplification

Text Simplification task is to transform complex text into a simpler, more readable, and accessible form while retaining 
its meaning.

[[papers with code page](https://paperswithcode.com/sota/text-simplification-on-asset?metric=SARI%20(EASSE%3E%3D0.2.1))]
[[nlp progress page](https://nlpprogress.com/english/simplification.html)]

#### Dataset
We use ASSET-valid dataset (2000 samples) for train and validation and ASSET-test (359 samples) for test.

#### Evaluation

* validation dataset: subset of ASSET-dev (100 samples); not used for training; used for model selection;
* test dataset: ASSET-test (359 samples);
* prompts, optimization logs, and raw outputs: see in the folders `outputs\gpt-4o-mini_text_simplification` and `outputs\gpt-4o_text_simplification`.

We use [SARI](https://aclanthology.org/Q16-1029/) metric from the [easse package](https://github.com/feralvam/easse). 
Also, see [ASSET papers with code page](https://paperswithcode.com/sota/text-simplification-on-asset?metric=SARI%20(EASSE%3E%3D0.2.1)).

#### Running scripts
Running prompt generation
```
python generate_prompts.py --config text_simplification/config_text_simplification.yaml --output_path text_simplification/generated_prompts
```

Running evaluation on test set (ASSET-test, 359 samples)
```
python evaluate_prompts.py --config text_simplification/config_text_simplification.yaml
```

Running evaluation on validation set (subset of ASSET-train, 100 samples)
```
python evaluate_prompts.py --config text_simplification/config_text_simplification.yaml -v
```

# LLM parameters
LLM parameters for prompt-generation:
```yaml
backend_llm: 
  model_name: gpt-4o-mini or gpt-4o
  temperature: 0.0
  max_output_tokens: 256
  num_threads: 40
  top_p: 0.1
 ```

LLM parameters for prompt-optimization:
```yaml
prompt_builder_llm:
  model_name: gpt-4o-mini or gpt-4o
  temperature: 1.0
  max_output_tokens: 4096
  num_threads: 40
  top_p: 1.0
```

## Results
| **Task**           | **Approach**                                                                                                                                               | **LLM**         | **Test Score** |
|--------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------|:--------------:|
| **GEC**            | Copy                                                                                                                                                       | –               | 0.00           |
|                    | SFT ([Omelianchuk et al., 2024](https://aclanthology.org/2024.bea-1.3/))                                                                                   | Multiple        | **72.80**      |
|                    |                                                                                                                                                            |                 |                |
|                    | Zero-shot ([Loem et al., 2023](https://aclanthology.org/2023.bea-1.18/))                                                                                   | GPT-3           | 53.07          |
|                    | Few-shot (16 examples) ([Loem et al., 2023](https://aclanthology.org/2023.bea-1.18/))                                                                      | GPT-3           | **57.41**      |
|                    | Few-shot (4 examples) ([Tang et al., 2024](https://aclanthology.org/2024.naacl-long.99/))                                                                  | GPT-3.5-Turbo   | 53.20          |
|                    |                                                                                                                                                            |                 |                |
|                    | [Zero-shot](outputs/gpt-4o-mini_gec_bea/zero-shot.txt) (adapted from [Loem et al., 2023](https://aclanthology.org/2023.bea-1.18/))                         | GPT-4o-mini     | 49.90          |
|                    | [Few-shot (3 randomly sampled examples)](outputs/gpt-4o-mini_gec_bea/02_13_15-35-19_few-shot_3.txt)                                                        | GPT-4o-mini     | 53.01          |
|                    | [APIO-Induction-Only (3 instructions)](outputs/gpt-4o-mini_gec_bea/02_13_15-35-19_instruction_induction_3.txt)                                             | GPT-4o-mini     | 38.72          |
|                    | [APIO (7 instructions)](outputs/gpt-4o-mini_gec_bea/02_13_15-35-19_optimized_3.txt)                                                                        | GPT-4o-mini     | **57.07**      |
|                    |                                                                                                                                                            |                 |                |
|                    | [Zero-shot](outputs/gpt-4o_gec_bea/zero-shot.txt) (adapted from [Loem et al., 2023](https://aclanthology.org/2023.bea-1.18/))                              | GPT-4o          | 54.66          |
|                    | [Few-shot (3 examples, randomly sampled)](outputs/gpt-4o_gec_bea/03_26_09-33-12_few-shot_3.txt)                                                            | GPT-4o          | 44.50          |
|                    | [APIO-Induction-Only (3 instructions)](outputs/gpt-4o_gec_bea/03_26_09-33-12_instruction_induction_3.txt)                                                  | GPT-4o          | 43.37          |
|                    | [APIO (10 instructions)](outputs/gpt-4o_gec_bea/03_26_09-33-12_optimized_3.txt)                                                                            | GPT-4o          | **59.40**      |
|                    |                                                                                                                                                            |                 |                |
| **Text Simplification**| Copy                                                                                                                                                       | –               | 20.70          |
|                    | SFT ([Sheang and Saggion, 2021](https://aclanthology.org/2021.inlg-1.38/))                                                                                 | T5-base         | 45.04          |
|                    | Best reference (ref-0)                                                                                                                                     | –               | **52.62**      |
|                    |                                                                                                                                                            |                 |                |
|                    | Few-shot (15 SARI-selected examples, random ordering) ([Vadlamannati & Şahin, 2023](https://aclanthology.org/2023.inlg-main.18/))                          | GPT-3-175B      | 47.94          |
|                    |                                                                                                                                                            |                 |                |
|                    | [Zero-shot](outputs/gpt-4o-mini_text_simplification/zero-shot.txt) (adapted from [Raheja et al., 2023](https://aclanthology.org/2023.findings-emnlp.350/)) | GPT-4o-mini     | 48.03          |
|                    | [Few-shot](outputs/gpt-4o-mini_text_simplification/02_13_12-38-55_few-shot_3.txt) (3 randomly sampled examples)                                            | GPT-4o-mini     | 47.16          |
|                    | [APIO-Induction-Only](outputs/gpt-4o-mini_text_simplification/02_13_12-38-55_instruction_induction_3.txt) (3 instructions)                                 | GPT-4o-mini     | 48.79          |
|                    | [APIO](outputs/gpt-4o-mini_text_simplification/02_13_12-38-55_optimized_3.txt) (6 instructions)                                                            | GPT-4o-mini     | **49.27**      |
|                    |                                                                                                                                                            |                 |                |
|                    | [Zero-shot](outputs/gpt-4o_text_simplification/zero-shot.txt) (adapted from [Raheja et al., 2023](https://aclanthology.org/2023.findings-emnlp.350/))                  | GPT-4o          | 47.73          |
|                    | [Few-shot](outputs/gpt-4o_text_simplification/03_24_16-15-44_few-shot_3.txt) (3 examples, randomly sampled)                                                                                                                | GPT-4o          | 47.87          |
|                    | [APIO-Induction-Only](outputs/gpt-4o_text_simplification/03_24_16-15-44_instruction_induction_3.txt) (3 instructions)                                                                                                                   | GPT-4o          | 48.93          |
|                    | [APIO](outputs/gpt-4o_text_simplification/03_24_16-15-44_optimized_3.txt) (10 instructions)                                                                                                                                 | GPT-4o          | **49.47**      |

Metrics: GEC (BEA-2019-Test | $F_{0.5}$) and Text Simplification results (ASSET-Test | SARI).  Results are grouped by 
baselines (Copy, Best-reference, and SFT), and by other prompt-based methods from different models. Best reference 
baseline is unavailable for the GEC task because the BEA-2019-Test dataset has not been published.
