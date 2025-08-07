import argparse
import glob
import yaml
from pprint import pprint

from tasks import *
from utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='text_simplification/config_text_simplification.yaml')
    #parser.add_argument('--config', default='gec_jfleg/config_gec_jfleg.yaml')
    parser.add_argument('--input_path', required=True, help='Path to the generated prompts directory')
    parser.add_argument('--num_samples', '-n',default=-1, type=int)
    parser.add_argument('--valid', '-v', action='store_true', help='Test on validation dataset')
    parser.add_argument('--train', '-t', action='store_true', help='Test on train dataset')

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        pprint(config)

    #args.input_path = f'{os.path.dirname(args.config)}/generated_prompts/'
    evalset_name = 'test'
    if args.valid:
        evalset_name = 'valid'
    elif args.train:
        evalset_name = 'train'

    args.output_path = f'{args.input_path}/evaluated_prompts_{evalset_name}/'
    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    print(args.input_path)
    print(args.output_path)
    print(evalset_name)

    logger = get_logger(filename=f'{args.output_path}/_evaluation.log')
    logger.info('*' * 80)
    logger.info(json.dumps(config, indent=4))
    logger.info('*' * 80)

    task = load_task(config)
    llm_model_name = config['backend_llm']['model_name']

    test_data = task.get_test_data(index_from=0, index_to=args.num_samples)
    input_texts = test_data['orig_sents']
    references = test_data['refs_sents']
    references_repacked = [['N/A' for _ in range(len(references))] for _ in range(len(input_texts))]
    for i, ref in enumerate(references):
        for j, curr_ref in enumerate(ref):
            references_repacked[j][i] = curr_ref

    input_filenames = sorted(glob.glob(f'{args.input_path}/*.json'), reverse=True) # zero-shot starts first
    prompt_data_list = [json.load(open(f)) for f in input_filenames]
    for n, (prompt_data, input_filename) in enumerate(zip(prompt_data_list, input_filenames)):
        prompt_template = prompt_data['PROMPT_TEMPLATE']

        if evalset_name == 'test':
            score, outputs = task.evaluate_prompt_template_test(prompt_template=prompt_template,
                                                                num_samples=args.num_samples,
                                                                return_outputs=True)
        elif evalset_name == 'valid':
            score, outputs = task.evaluate_prompt_template_valid(prompt_template=prompt_template,
                                                                 num_samples=args.num_samples,
                                                                 return_outputs=True)
        elif evalset_name == 'train':
            score, outputs = task.evaluate_prompt_template_train(prompt_template=prompt_template,
                                                                 num_samples=args.num_samples,
                                                                 return_outputs=True)
        else:
            raise NotImplementedError

        short_filename = '%40s' % os.path.basename(input_filename).replace('.json', '')
        logger.info(f'[{n+1}/{len(prompt_data_list)}] {short_filename} | Test score: {score:.2f}')

        output_data = prompt_data
        output_data['TEST_SCORE'] = score
        output_data['llm_outputs'] = []

        for input_text, output, curr_references in zip(input_texts, outputs, references_repacked):
            output_data['llm_outputs'].append({'input': input_text, 'references': curr_references, 'output': output})

        output_json_filename = f'{os.path.join(args.output_path, short_filename.strip())}_{llm_model_name}_{score:.2f}.json'
        with open(output_json_filename, 'w') as f:
            json.dump(output_data, f, indent=4)
            Path(output_json_filename.replace('.json', '.txt')).write_text('\n'.join(outputs))
            outputs = [tokenize_text(o) for o in outputs]
            Path(output_json_filename.replace('.json', '.tok.txt')).write_text('\n'.join(outputs))

    logger.info('*' * 80)
