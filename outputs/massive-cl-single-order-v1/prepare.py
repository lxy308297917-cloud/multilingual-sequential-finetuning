"""Freeze the new language order using verified V1 examples and the same replay algorithm."""
from importlib.util import module_from_spec, spec_from_file_location

from common import CFG, DATA, EVAL, LANGUAGES, PROTOCOL, ROOT, atomic_json, load, read, sha

OLD = ROOT.parent / 'massive-cl-v1'


def main():
    old = read(OLD / 'DATA_MANIFEST.json')
    ranking = read(OLD / 'RESULTS.json')['final']['dev']['single']
    order = tuple(sorted(ranking, key=lambda loc: -ranking[loc]['slot_micro_f1']))
    assert order == LANGUAGES and tuple(CFG['evaluation_order']) == EVAL
    source = DATA.parent / 'massive-cl-v1-data'
    assert sha(source / 'MANIFEST.json') == sha(OLD / 'DATA_MANIFEST.json')
    replay_module = OLD / 'prepare.py'
    spec = spec_from_file_location('original_massive_schedule', replay_module)
    algorithm = module_from_spec(spec)
    spec.loader.exec_module(algorithm)
    assert algorithm.DATA == DATA and algorithm.LANGUAGES == LANGUAGES
    DATA.mkdir()
    names = ['joint.jsonl', *[f'train_{loc}.jsonl' for loc in LANGUAGES],
             *[f'{split}_{loc}.jsonl' for split in ('dev', 'test') for loc in EVAL]]
    for name in names:
        prior_file = source / name
        assert sha(prior_file) == old['files'][name], name
        (DATA / name).symlink_to(prior_file)
    frozen = {loc: load(f'train_{loc}.jsonl', old) for loc in LANGUAGES}
    replay = algorithm.replay_stages(frozen)
    assert all(replay[loc]['budget'] == {key: old['train'][loc][key] for key in replay[loc]['budget']}
               for loc in LANGUAGES)
    receipt = dict(old, protocol_sha256=sha(PROTOCOL), prior_order_manifest_sha256=sha(source / 'MANIFEST.json'),
                   prior_order_results_sha256=sha(OLD / 'RESULTS.json'), replay_algorithm_sha256=sha(replay_module),
                   ranking_split='dev', ranking_metric='slot_micro_f1',
                   ranking={loc: ranking[loc]['slot_micro_f1'] for loc in LANGUAGES},
                   train={loc: old['train'][loc] for loc in LANGUAGES}, replay=replay,
                   files={path.name: sha(path) for path in sorted(DATA.glob('*.jsonl'))})
    atomic_json(DATA / 'MANIFEST.json', receipt)
    atomic_json(ROOT / 'DATA_MANIFEST.json', receipt)
    print('Order:', ' -> '.join(LANGUAGES))
    print('Replay fraction:', {loc: round(replay[loc]['input_fraction'], 6) for loc in LANGUAGES})


if __name__ == '__main__':
    main()
