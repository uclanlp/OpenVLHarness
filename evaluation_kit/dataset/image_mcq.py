# Adapted from VLMEvalKit (https://github.com/open-compass/VLMEvalKit), Apache License 2.0; modified by the OpenVLHarness authors.
import warnings

from .hf_release import hf_tsv_url
from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from .utils.judge_cache import guard_cache
from ..smp import *
import pandas as pd

MMMB_URLS = {
    'MMMB_ar': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmmb/mmmb_ar.tsv',
    'MMMB_cn': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmmb/mmmb_cn.tsv',
    'MMMB_en': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmmb/mmmb_en.tsv',
    'MMMB_pt': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmmb/mmmb_pt.tsv',
    'MMMB_ru': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmmb/mmmb_ru.tsv',
    'MMMB_tr': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmmb/mmmb_tr.tsv',
}

MTL_MMBench_URLS = {
    'MMBench_dev_ar': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmbench/mmbench_dev_ar.tsv',
    'MMBench_dev_cn': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmbench/mmbench_dev_cn.tsv',
    'MMBench_dev_en': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmbench/mmbench_dev_en.tsv',
    'MMBench_dev_pt': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmbench/mmbench_dev_pt.tsv',
    'MMBench_dev_tr': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmbench/mmbench_dev_tr.tsv',
    'MMBench_dev_ru': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmbench/mmbench_dev_ru.tsv',
}

MMMB_MD5 = {
    'MMMB_ar': 'f3a18b6385f1d9701840aa42de27aead', 'MMMB_cn': '13ed82fa89730037292fcaa27f08f430',
    'MMMB_en': '1cd781a71ec5a2983c090b84105d6a01', 'MMMB_pt': '548ea2b3bb2da991790386f0015d30d1',
    'MMMB_ru': 'ce1cc8a0533425ab0d86b326ebfc2984', 'MMMB_tr': '0733739d43090327975294292bc5cd67'
}

MTL_MMBench_MD5 = {
    'MMBench_dev_ar': '4271b4a0d0200e1a86380a878e0d64a4', 'MMBench_dev_cn': '2ed5135326fed02c8e51ea50dda8222f',
    'MMBench_dev_en': 'd9ab776fc018b3d45785e9a5c23431c2', 'MMBench_dev_pt': '4ddfbcd27ef12444b908c03831cd0295',
    'MMBench_dev_tr': '4fab39d501389d3d6cc90264bb708f11', 'MMBench_dev_ru': '5ba1171ff2e68f80637bf78349e402a5'
}


class ImageMCQDataset(ImageBaseDataset):

    TYPE = 'MCQ'

    DATASET_URL = {
        # MMBench v1.0
        'MMBench_DEV_EN': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_DEV_EN.tsv',
        'MMBench_TEST_EN': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_TEST_EN.tsv',
        'MMBench_DEV_CN': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_DEV_CN.tsv',
        'MMBench_TEST_CN': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_TEST_CN.tsv',
        'MMBench_DEV_KO': 'https://huggingface.co/datasets/NCSOFT/K-MMBench/resolve/main/MMBench_DEV_KO.tsv',
        'MMBench': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench.tsv',  # Internal
        'MMBench_CN': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_CN.tsv',  # Internal
        # MMBench v1.1
        'MMBench_DEV_EN_V11': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_DEV_EN_V11.tsv',
        'MMBench_TEST_EN_V11': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_TEST_EN_V11.tsv',
        'MMBench_DEV_CN_V11': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_DEV_CN_V11.tsv',
        'MMBench_TEST_CN_V11': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_TEST_CN_V11.tsv',
        'MMBench_V11': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_V11.tsv',  # Internal
        'MMBench_CN_V11': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_CN_V11.tsv',  # Internal
        # SEEDBench Series
        'SEEDBench_IMG': 'https://opencompass.openxlab.space/utils/benchmarks/SEEDBench/SEEDBench_IMG.tsv',
        'SEEDBench_IMG_KO': 'https://huggingface.co/datasets/NCSOFT/K-SEED/resolve/main/SEEDBench_IMG_KO.tsv',
        'SEEDBench2': 'https://huggingface.co/datasets/VLMEval/SEEDBench2/resolve/main/SEEDBench2.tsv',
        'SEEDBench2_Plus': 'https://opencompass.openxlab.space/utils/benchmarks/SEEDBench/SEEDBench2_Plus.tsv',
        # ScienceQA Series
        'ScienceQA_VAL': 'https://opencompass.openxlab.space/utils/benchmarks/ScienceQA/ScienceQA_VAL.tsv',
        'ScienceQA_TEST': 'https://opencompass.openxlab.space/utils/benchmarks/ScienceQA/ScienceQA_TEST.tsv',
        # MMT-Bench
        'MMT-Bench_ALL_MI': 'https://opencompass.openxlab.space/utils/benchmarks/MMT-Bench/MMT-Bench_ALL_MI.tsv',
        'MMT-Bench_ALL': 'https://opencompass.openxlab.space/utils/benchmarks/MMT-Bench/MMT-Bench_ALL.tsv',
        'MMT-Bench_VAL_MI': 'https://opencompass.openxlab.space/utils/benchmarks/MMT-Bench/MMT-Bench_VAL_MI.tsv',
        'MMT-Bench_VAL': 'https://opencompass.openxlab.space/utils/benchmarks/MMT-Bench/MMT-Bench_VAL.tsv',
        # AesBench
        'AesBench_VAL': 'https://huggingface.co/datasets/VLMEval/AesBench/resolve/main/AesBench_VAL.tsv',
        'AesBench_TEST': 'https://huggingface.co/datasets/VLMEval/AesBench/resolve/main/AesBench_TEST.tsv',
        # Q-Bench1
        'Q-Bench1_VAL': 'https://huggingface.co/datasets/zhangzicheng/qbench_tsv/resolve/main/Q-Bench1_VAL.tsv',
        'Q-Bench1_TEST': 'https://huggingface.co/datasets/zhangzicheng/qbench_tsv/resolve/main/Q-Bench1_TEST.tsv',
        # A-Bench
        'A-Bench_VAL': 'https://huggingface.co/datasets/zhangzicheng/abench_tsv/resolve/main/A-bench_VAL.tsv',
        'A-Bench_TEST': 'https://huggingface.co/datasets/zhangzicheng/abench_tsv/resolve/main/A-bench_TEST.tsv',
        # R-Bench
        'R-Bench-Dis': 'https://huggingface.co/datasets/lcysyzxdxc/R-Bench/resolve/main/R-bench-dis.tsv',
        'R-Bench-Ref': 'https://huggingface.co/datasets/lcysyzxdxc/R-Bench/resolve/main/R-bench-ref.tsv',
        # Other Benchmarks
        'CCBench': 'https://opencompass.openxlab.space/utils/VLMEval/CCBench.tsv',
        'AI2D_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/AI2D_TEST.tsv',
        'AI2D_TEST_NO_MASK': 'https://opencompass.openxlab.space/utils/VLMEval/AI2D_TEST_NO_MASK.tsv',
        'MMStar': 'https://opencompass.openxlab.space/utils/VLMEval/MMStar.tsv',
        'MMStar_KO': 'https://huggingface.co/datasets/NCSOFT/K-MMStar/resolve/main/MMStar_KO.tsv',
        'MMStar_TR': 'https://huggingface.co/datasets/kesimeg/MMStar_tr/resolve/main/MMStar_TR.tsv',
        'RealWorldQA': hf_tsv_url('RealWorldQA'),
        'MLLMGuard_DS': 'https://opencompass.openxlab.space/utils/VLMEval/MLLMGuard_DS.tsv',
        'BLINK': 'https://opencompass.openxlab.space/utils/VLMEval/BLINK.tsv',
        'BLINK_circular': 'https://opencompass.openxlab.space/utils/VLMEval/BLINK_circular.tsv',
        'TaskMeAnything_v1_imageqa_random': (
            'https://huggingface.co/datasets/weikaih/TaskMeAnything-v1-imageqa-random/'
            'resolve/main/TaskMeAnything-v1-imageqa-random.tsv'
        ),
        'A-OKVQA': 'https://huggingface.co/datasets/Allen8/A-OKVQA/resolve/main/a-okvqa.tsv',
        'WorldMedQA-V': 'https://opencompass.openxlab.space/utils/VLMEval/WorldMedQA-V.tsv',
        'VisOnlyQA-VLMEvalKit': (
            'https://huggingface.co/datasets/ryokamoi/VisOnlyQA_Eval_Real/'
            'resolve/main/visonlyqa_vlmevalkit.tsv'
        ),
        'MMCR': 'http://opencompass.openxlab.space/utils/VLMEval/MMCR.tsv',
        'MMSci_DEV_MCQ': 'https://opencompass.openxlab.space/utils/VLMEval/MMSci_DEV_MCQ.tsv',
        "MMVP": "http://opencompass.openxlab.space/utils/VLMEval/MMVP.tsv",
        "K-DTCBench": "https://huggingface.co/datasets/NCSOFT/K-DTCBench/resolve/main/K-DTCBench.tsv",
        # For Internal Use Only
        'MMBench_V11_MINI': 'https://opencompass.openxlab.space/utils/TEST/MMBench_V11_MINI.tsv',
        'MMStar_MINI': 'https://opencompass.openxlab.space/utils/TEST/MMStar_MINI.tsv',
        'AI2D_MINI': 'https://opencompass.openxlab.space/utils/TEST/AI2D_MINI.tsv',
        "VStarBench": hf_tsv_url("VStarBench"),
        'PathMMU_VAL': 'https://huggingface.co/datasets/Pfei111/PathMMU/resolve/main/PathMMU_VAL.tsv',
        'PathMMU_TEST': 'https://huggingface.co/datasets/Pfei111/PathMMU/resolve/main/PathMMU_TEST.tsv',
        'CMMU_MCQ': 'https://huggingface.co/datasets/Pfei111/CMMU_VAL_MCQ/resolve/main/CMMU_VAL_MCQ.tsv',
        'MicroVQA': 'https://opencompass.openxlab.space/utils/VLMEval/MicroVQA.tsv',
        'MMSIBench_circular': 'https://opencompass.openxlab.space/utils/VLMEval/MMSIBench_circular.tsv',
        'SpatialEval': 'https://opencompass.openxlab.space/utils/VLMEval/SpatialEval.tsv',
        "StaticEmbodiedBench": "https://huggingface.co/datasets/xiaojiahao/StaticEmbodiedBench/resolve/main/StaticEmbodiedBench.tsv",  # noqa
        "StaticEmbodiedBench_circular": "https://huggingface.co/datasets/xiaojiahao/StaticEmbodiedBench/resolve/main/StaticEmbodiedBench_circular.tsv"  # noqa
    }

    DATASET_MD5 = {
        # MMBench v1.0
        'MMBench_DEV_EN': 'b6caf1133a01c6bb705cf753bb527ed8',
        'MMBench_TEST_EN': '6939fadb0ce626fefc0bdc9c64efc528',
        'MMBench_DEV_CN': '08b8fc3324a5ed74155350f57be69fbd',
        'MMBench_TEST_CN': '7e1239baf0ee4c8b513e19705a0f317e',
        'MMBench_DEV_KO': '72e1cde9124b5015be6d0dd5c9b5500d',
        'MMBench': '4115aea3383f3dd0083be6a633e0f820',  # Internal Only
        'MMBench_CN': '2e053ffc90ea598b1feae13c36dc13ee',    # Internal Only
        # MMBench v1.1
        'MMBench_DEV_EN_V11': '30c05be8f2f347a50be25aa067248184',
        'MMBench_TEST_EN_V11': '26f0f15381a21720255091d3e0316ce6',
        'MMBench_DEV_CN_V11': '593f9b5f6bea453d870a798b34ae4f37',
        'MMBench_TEST_CN_V11': '74bbe4556dac745613c7cbe5ad787050',
        'MMBench_V11': 'b9276414f57af1308dcc4d0cd9b42e7c',  # Internal Only
        'MMBench_CN_V11': '95f6980dd1b4de38e3cbffe0305a3f25',    # Internal Only
        # SEEDBench
        'SEEDBench_IMG': '68017231464752261a2526d6ca3a10c0',
        'SEEDBench_IMG_KO': 'b354a9ac3493f3ccf294e69b216bfab3',
        'SEEDBench2': '4ec15cf864c4f16274112284f531813e',
        'SEEDBench2_Plus': 'e32d3216dc4f452b0fe497a52015d1fd',
        # ScienceQA
        'ScienceQA_VAL': '96320d05e142e585e7204e72affd29f3',
        'ScienceQA_TEST': 'e42e9e00f9c59a80d8a5db35bc32b71f',
        # MMT-Bench
        'MMT-Bench_ALL_MI': '5272157097e19cdd7cb41e412ab3b7c7',
        'MMT-Bench_ALL': 'b273a2f4c596fe4f2605de0494cd632f',
        'MMT-Bench_VAL_MI': 'c7d7b998eb5cd9aa36c7d4f721472462',
        'MMT-Bench_VAL': '8dd4b730f53dbf9c3aed90ca31c928e0',
        # AesBench
        'AesBench_VAL': '3edb0c319e9187aa0b97fe7a11700a8c',
        'AesBench_TEST': '58b1f7ba2cc32e1d68896d6ee716bbf8',
        # Q-Bench1
        'Q-Bench1_VAL': '837bdb6cd2da571713543462815187b7',
        'Q-Bench1_TEST': '15e759bfd58c9d5f30b23a317d347153',
        # A-Bench
        'A-Bench_VAL': '218563ec50d34bb336c814143a5bb9c1',
        'A-Bench_TEST': '567013fb033a20cf23f51d8e865bd16c',
        # R-Bench
        'R-Bench-Dis': 'd6e961dbfc43350688af2560226830b4',
        'R-Bench-Ref': '270c1cb555acb523f3fdb178ed57021d',
        # Other Benchmarks
        'CCBench': 'f5dde47f24dc5a6fb6e595b409b466ac',
        'AI2D_TEST': '0f593e0d1c7df9a3d69bf1f947e71975',
        'AI2D_TEST_NO_MASK': 'fd8f463634d4fe9fbd23b876e8eea5be',
        'MMStar': 'e1ecd2140806c1b1bbf54b43372efb9e',
        'MMStar_KO': 'cc6049c7314bb54b9ac5e247a2bfb357',
        'RealWorldQA': '4de008f55dc4fd008ca9e15321dc44b7',
        'MLLMGuard_DS': '975fc0dd7119386e198c37d71e274b3f',
        'BLINK': '3b6649b6a662184ea046908e5506260e',
        'BLINK_circular': '75aee2332420c7654dc51b1442fafc7b',
        'TaskMeAnything_v1_imageqa_random': '023fef69e2ca21827afb77c5ec3bc889',
        'WorldMedQA-V': '441e63875e30c87f5750528b57b41285',
        "VisOnlyQA-VLMEvalKit": 'cf460a31d2acb8d3a7cecd0e69298bfa',
        'MMCR': '9052635f2c3835bdb87755ef73564f5e',
        'MMSci_DEV_MCQ': '865144aa866e29b251bdc7d63a735b6b',
        "MMVP": "8cb732b141a0cba5b42159df2839e557",
        "K-DTCBench": "fe72a85b010513d3840b5f3be2de6ed3",
        "VStarBench": "b18854d7075574be06b631cd5f7d2d6a",
        'MicroVQA': 'd7506438701a2076ec277f8bb3586c1a',
        'MMSIBench_circular': '7be2b9e8a280863272e89fab5ba40807',
        'SpatialEval': '4c8eb33142b26be2916fb9164287b72b',
        "StaticEmbodiedBench": "5c50611650ca966970180a80d49429f0",
        "StaticEmbodiedBench_circular": "034cf398a3c7d848d966e1081e4baf68"
    }

    DATASET_URL.update(MMMB_URLS)
    DATASET_URL.update(MTL_MMBench_URLS)
    DATASET_MD5.update(MMMB_MD5)
    DATASET_MD5.update(MTL_MMBench_MD5)
    DEFAULT_JUDGE = ['chatgpt-0125', 'gpt-4-0125']

    @classmethod
    def supported_datasets(cls):
        # Register ``MMStar_v2`` as a stratified sub-sample variant of
        # MMStar (300 rows, seed=42 defaults). Implementation lives in
        # ``load_data``; no separate TSV cached.
        return list(cls.DATASET_URL) + ['MMStar_v2']

    def load_data(self, dataset):
        """Default MCQ load path, plus a special branch for the
        ``MMStar_v2`` stratified sub-sample variant.

        ``MMStar_v2`` is NOT a separate upstream dataset — it loads the
        same MMStar TSV and then applies an opt-in stratified sub-sample
        on the ``l2_category`` column (which nests inside ``category``,
        so both layers' distributions are preserved). Defaults: 300
        samples, seed 42. Override via ``MVT_MMSTAR_SUBSET_N`` and
        ``MVT_MMSTAR_SUBSET_SEED`` env vars when you need a different
        size or seed.
        """
        if dataset == 'MMStar_v2':
            import os as _os
            df = super().load_data('MMStar')
            from .utils.mmstar_subset import apply_mmstar_stratified_subset
            env_n = _os.environ.get('MVT_MMSTAR_SUBSET_N', '').strip()
            n = int(env_n) if env_n else 300
            env_seed = _os.environ.get('MVT_MMSTAR_SUBSET_SEED', '').strip()
            seed = int(env_seed) if env_seed else 42
            return apply_mmstar_stratified_subset(df, n=n, seed=seed)
        return super().load_data(dataset)

    def build_prompt(self, line):

        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += 'Please select the correct answer from the options above. \n'

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        return self.evaluate_heuristic(eval_file, **judge_kwargs)

    def evaluate_heuristic(self, eval_file, **judge_kwargs):
        from .utils.multiple_choice import (
            report_acc, report_acc_MMT, report_acc_MMSci, mcq_circular_eval, mcq_vanilla_eval, report_acc_MMVP
        )
        # assert dataset is not None
        dataset_map = {
            'MMBench_TEST_EN': 'MMBench', 'MMBench_TEST_EN_V11': 'MMBench_V11',
            'MMBench_TEST_CN': 'MMBench_CN', 'MMBench_TEST_CN_V11': 'MMBench_CN_V11'
        }
        dataset = self.dataset_name
        if dataset in dataset_map:
            dataset = dataset_map[dataset]
        nproc = judge_kwargs.pop('nproc', 4)

        circular = False
        if listinstr(['mmbench', 'ccbench', 'circular', 'mmcr'], dataset.lower()):
            data = load(eval_file)
            data['index'] = [int(x) for x in data['index']]
            dump(data, eval_file)
            circular = True

        model = (judge_kwargs.get('model') or 'exact_matching')
        judge_model = model
        name_str_map = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4'}
        name_str = name_str_map[model] if model in name_str_map else model
        if "/" in name_str:
            name_str = name_str.split("/")[-1]

        if model in (None, 'exact_matching'):  # a null judge model means heuristic matching only
            model = None
        else:
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                warnings.warn(DEBUG_MESSAGE)
                model = None

        result_file = get_intermediate_file_path(eval_file, f'_{name_str}_result', 'pkl')

        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]
        # If not choice label, then use lower case
        for k in data.keys():
            data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

        meta = self.data
        meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
        data_map = {x: y for x, y in zip(data['index'], data['question'])}
        for k in data_map:
            assert k in meta_q_map, (
                f'eval_file should be the same as or a subset of dataset {self.dataset_name}'
            )

        # Circular results are keyed by question group, so any change drops the file.
        guard_cache(result_file, data, judge_model, keyed=not circular)
        if circular:
            data = mcq_circular_eval(model, data, meta, nproc, result_file, self.dataset_name)
        else:
            data = mcq_vanilla_eval(model, data, meta, nproc, result_file, self.dataset_name)

        # load split
        eval_record = get_intermediate_file_path(eval_file, f'_{name_str}_result')
        dump(data, eval_record)
        data = load(eval_record)

        # May have different report acc functions for different datasets
        if 'MMT' in dataset:
            acc = report_acc_MMT(data)
        elif 'MMSci' in dataset:
            acc = report_acc_MMSci(data)
        elif dataset == 'MMVP':
            acc = report_acc_MMVP(data)
        else:
            acc = report_acc(data)

        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(acc, score_file)

        # The piece of code is for internal use, to check vanilla acc (circ0 & all) for circular datasets
        if circular and os.environ.get('PRINT_VANILLA', None) == '1':
            acc_map = {}
            acc_map['circular'] = acc
            # Vanilla Circ0 Acc
            data = load(eval_file)
            data['index'] = [int(x) for x in data['index']]
            if 'g_index' in data:
                data['g_index'] = [int(x) for x in data['g_index']]
                circ0 = data[data['g_index'] == data['index']]
            else:
                offset = 1e6
                circ0 = data[data['index'] <= offset]
            result_file = get_intermediate_file_path(eval_file, f'_{name_str}_vanilla_result', 'pkl')
            guard_cache(result_file, circ0, judge_model)
            data0 = mcq_vanilla_eval(model, circ0, meta, nproc, result_file, self.dataset_name)
            dump(data0, get_intermediate_file_path(eval_file, f'_{name_str}_vanilla_circ0_result'))
            data = load(get_intermediate_file_path(eval_file, f'_{name_str}_vanilla_circ0_result'))
            acc_map['vanilla_0'] = report_acc(data)
            # Vanilla ALL Acc
            data = load(eval_file)
            guard_cache(result_file, data, judge_model)
            dataall = mcq_vanilla_eval(model, data, meta, nproc, result_file, self.dataset_name)
            dump(dataall, get_intermediate_file_path(eval_file, f'_{name_str}_vanilla_all_result'))
            data = load(get_intermediate_file_path(eval_file, f'_{name_str}_vanilla_all_result'))
            acc_map['vanilla_all'] = report_acc(data)
            # Merge & Print the Evaluation Results
            for k, v in acc_map.items():
                if 'split' not in v:
                    v['split'] = [None] * len(v)
                if len(v) == 1 and pd.isna(v['split'][0]):
                    v['split'] = [k]
                else:
                    assert not pd.isna(v['split'][0])
                    v['split'] = [k + '_' + sp for sp in v['split']]
            score_all = [acc_map['vanilla_0'], acc_map['vanilla_all'], acc_map['circular']]
            score_all = pd.concat(score_all)
            print(score_all)
            score_file = get_intermediate_file_path(eval_file, '_acc_all', 'csv')
            dump(score_all, score_file)

        if dataset == 'AesBench_VAL':
            warnings.warn('Note that AesBench VAL is just a toy version of AesBench TEST. For full results, \
                           please evaluate on AesBench TEST. The AesBench TEST dataset is more than 20 times \
                           larger than the VAL dataset and the leaderboard results are based on AesBench TEST.')
        if dataset == 'VisOnlyQA-VLMEvalKit':
            warnings.warn('Note that the results on VisOnlyQA-VLMEvalKit are different from the results on \
                           the original VisOnlyQA. VisOnlyQA-VLMEvalKit does not include the \
                           chemistry__shape_multi split and uses a different evaluation prompt. Please \
                           explicitly specify the version of the dataset when you report results.')

        return acc


class CustomMCQDataset(ImageMCQDataset):

    def load_data(self, dataset):
        data_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')

        if file_size(data_path, 'GB') > 1:
            local_path = data_path.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL', None):
                from ..tools import LOCALIZE
                LOCALIZE(data_path, local_path)
            data_path = local_path
        return load(data_path)


