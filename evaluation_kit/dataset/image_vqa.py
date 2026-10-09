# Adapted from VLMEvalKit (https://github.com/open-compass/VLMEvalKit), Apache License 2.0; modified by the OpenVLHarness authors.
import os
from functools import partial

from .image_base import ImageBaseDataset
from .utils import build_judge
from .utils.judge_cache import guard_cache
from ..smp import *
from ..smp.file import get_intermediate_file_path


class ImageVQADataset(ImageBaseDataset):
    TYPE = 'VQA'

    DATASET_URL = {
        'OCRVQA_TEST':
        'https://opencompass.openxlab.space/utils/VLMEval/OCRVQA_TEST.tsv',
        'OCRVQA_TESTCORE':
        'https://opencompass.openxlab.space/utils/VLMEval/OCRVQA_TESTCORE.tsv',
        'TextVQA_VAL':
        'https://opencompass.openxlab.space/utils/VLMEval/TextVQA_VAL.tsv',
        'DocVQA_VAL':
        'https://opencompass.openxlab.space/utils/VLMEval/DocVQA_VAL.tsv',
        'DocVQA_TEST':
        'https://opencompass.openxlab.space/utils/VLMEval/DocVQA_TEST.tsv',
        'InfoVQA_VAL':
        'https://opencompass.openxlab.space/utils/VLMEval/InfoVQA_VAL.tsv',
        'InfoVQA_TEST':
        'https://opencompass.openxlab.space/utils/VLMEval/InfoVQA_TEST.tsv',
        'ChartQA_TEST':
        'https://opencompass.openxlab.space/utils/VLMEval/ChartQA_TEST.tsv',
        'GQA_TestDev_Balanced':
        'https://opencompass.openxlab.space/utils/VLMEval/GQA_TestDev_Balanced.tsv',
    }

    DATASET_MD5 = {
        'OCRVQA_TEST': 'ca46a6d74b403e9d6c0b670f6fc00db9',
        'OCRVQA_TESTCORE': 'c5239fe77db8bdc1f2ad8e55e0d1fe97',
        'TextVQA_VAL': 'b233b31f551bbf4056f2f955da3a92cd',
        'DocVQA_VAL': 'd5ee77e1926ff10690d469c56b73eabf',
        'DocVQA_TEST': '6a2f28cac26ef2d3447374e8c6f6c8e9',
        'InfoVQA_VAL': '2342e9c225222f0ef4dec545ebb126fe',
        'InfoVQA_TEST': 'df535bf51b88dc9718252c34131a6227',
        'ChartQA_TEST': 'c902e0aa9be5582a7aad6dcf52734b42',
        'GQA_TestDev_Balanced': '99b62f22e224d9b2f32dcbe41359d1c9',
    }

    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        assert msgs[-1]['type'] == 'text'
        msgs[-1][
            'value'] += '\nAnswer the question using a single word or phrase.'
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        return self.evaluate_heuristic(eval_file, **judge_kwargs)

    def _get_eval_method(self):
        """Return the dataset-specific VQA evaluation method name."""
        dataset = self.dataset_name
        if listinstr(['TextVQA'], dataset):
            return 'vqa_score'
        elif listinstr(['ChartQA'], dataset):
            return 'relaxed_accuracy'
        elif listinstr(['OCRVQA', 'GQA'], dataset):
            return 'accuracy'
        elif listinstr(['DocVQA', 'InfoVQA'], dataset):
            return 'anls'
        else:
            return 'vqa_score'

    # It returns a DataFrame
    def evaluate_heuristic(self, eval_file, **judge_kwargs):
        from .utils.vqa_eval import hit_calculate, process_line

        data = load(eval_file)
        dataset = self.dataset_name
        assert 'answer' in data and 'prediction' in data
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]
        method = self._get_eval_method()

        model_name = judge_kwargs.get('model', None)
        if model_name and model_name != 'exact_matching':
            data, res = self._extract_with_judge(eval_file, data, method, **judge_kwargs)
        else:
            lt = len(data)
            pool = mp.Pool(16)
            lines = [data.iloc[i] for i in range(lt)]
            res = pool.map(partial(process_line, method=method), lines)

            data['eval_gt'] = [r['gt'] for r in res]
            data['eval_pred'] = [r['pred'] for r in res]
            data['eval_match'] = [r['match'] for r in res]
            data['eval_score'] = [np.mean(r['match']) for r in res]

        detailed_result_file = get_intermediate_file_path(eval_file, '_results')
        dump(data, detailed_result_file)

        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]
        hit = hit_calculate(res, dataset)
        ret = dict()
        if 'split' in data:
            splits = set(data['split'])
            for sp in splits:
                sub = [r for l, r in zip(lines, res) if l['split'] == sp]
                # [np.mean(x['match']) >= full_score_weight for x in sub]
                hit = hit_calculate(sub, dataset)
                ret[sp] = np.mean(hit) * 100
            sub = [r for l, r in zip(lines, res)]
            hit = hit_calculate(sub, dataset)
            ret['Overall'] = np.mean(hit) * 100
        else:
            ret['Overall'] = np.mean(hit) * 100
            if 'category' in data:
                cates = list(set(data['category']))
                cates.sort()
                for c in cates:
                    sub = [r for l, r in zip(lines, res) if l['category'] == c]
                    # [np.mean(x['match']) >= full_score_weight for x in sub]
                    hit = hit_calculate(sub, dataset)
                    ret[c] = np.mean(hit) * 100
        ret = d2df(ret)
        ret.round(2)

        result_file = get_intermediate_file_path(eval_file, '_acc')
        dump(ret, result_file)
        return ret

    def _extract_with_judge(self, eval_file, data, method, **judge_kwargs):
        """Use LLM judge to extract answers, then apply metric. Returns (data, res)."""
        from .utils.vqa_judge import vqa_vanilla_eval

        model = build_judge(**judge_kwargs)
        nproc = judge_kwargs.get('nproc', 4)
        dataset = self.dataset_name

        name_str = judge_kwargs.get('model', 'judge')
        if '/' in name_str:
            name_str = name_str.split('/')[-1]

        # Use pkl for track_progress_rich cache (dict results with list values)
        result_file = get_intermediate_file_path(eval_file, f'_{name_str}_result', 'pkl')
        guard_cache(result_file, data, judge_kwargs.get('model'))
        data = vqa_vanilla_eval(model, data, nproc, result_file, dataset, method=method)

        # Dump as xlsx for visualization, then reload
        eval_record = get_intermediate_file_path(eval_file, f'_{name_str}_result')
        dump(data, eval_record)
        data = load(eval_record)

        # Build res list in the same format as process_line output
        res = [dict(match=data.iloc[i]['eval_match']) for i in range(len(data))]
        return data, res


class CustomVQADataset(ImageBaseDataset):
    TYPE = 'VQA'

    def load_data(self, dataset):
        data_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')

        if file_size(data_path, 'GB') > 1:
            local_path = data_path.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get(
                    'FORCE_LOCAL', None):
                from ..tools import LOCALIZE

                LOCALIZE(data_path, local_path)
            data_path = local_path
        return load(data_path)

    def evaluate(self, eval_file, **judge_kwargs):
        raise NotImplementedError


class FSC147Dataset(ImageVQADataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'FSC147_test': '', 'FSC147_val': '', 'FSC147_train': '',
        # FSCD-LVIS counting variants (one row per (image, class), answer = int count).
        # TSVs produced by `prepare_fscdlvis_count.py`; reuses this evaluator since
        # the eval logic (parse_count → MAE/RMSE/EM) is identical.
        'FSCD_LVIS_count_train': '', 'FSCD_LVIS_count_val': '', 'FSCD_LVIS_count_test': '',
        'FSCD_LVIS_unseen_count_train': '', 'FSCD_LVIS_unseen_count_test': '',
    }
    DATASET_MD5 = {}

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        tgt_path = self.dump_image(line)
        msgs = []
        msgs.extend([dict(type='image', value=p) for p in tgt_path])
        msgs.append(dict(type='text', value=line['question']))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        import numpy as np
        from .utils.fsc147 import parse_count

        data = load(eval_file)
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]

        # Counts are read with the strict rule-based parse alone; no language model
        # extracts answers, whatever judge is configured. Unparseable answers are
        # wrong and enter MAE/RMSE as 0.
        parsed = data['prediction'].apply(parse_count)
        valid = parsed.notna()
        predictions = parsed.fillna(0).astype(int)

        answers = data['answer'].astype(int)

        diff = (predictions - answers)
        mae = diff.abs().mean()
        rmse = np.sqrt((diff ** 2).mean())
        exact = valid & (predictions == answers)
        acc = exact.mean() * 100

        result = {'mae': round(float(mae), 2), 'rmse': round(float(rmse), 2), 'acc': round(float(acc), 2)}

        result_file = get_intermediate_file_path(eval_file, '_acc')
        dump(d2df(result), result_file)

        from .eval_bridge import emit_counting_per_row
        # Graded sort score: 1.0 on exact, else clipped relative accuracy.
        # Rows with smaller absolute miss rank above larger misses; gt<=0 with
        # any nonzero pred is bottom (score = 0).
        gt_safe = np.maximum(answers.to_numpy(), 1)
        graded = 1.0 - diff.abs().to_numpy() / gt_safe
        score_per_row = np.where(
            exact.to_numpy(), 1.0, np.clip(graded, 0.0, None)
        )
        emit_counting_per_row(
            eval_file, data['index'], data['image_path'],
            extra_cols={
                'gt': answers.astype(int).tolist(),
                'pred': predictions.astype(int).tolist(),
                'exact_match': exact.astype(int).tolist(),
                'abs_error': diff.abs().astype(int).tolist(),
                'sq_error': (diff ** 2).astype(int).tolist(),
            },
            sort_score=score_per_row.tolist(),
        )
        return result


