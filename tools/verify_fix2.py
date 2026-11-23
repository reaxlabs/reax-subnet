"""Reproduce the critic's exact-float flag-off differential comparisons."""
import json
import os
from pathlib import Path
import subprocess
import sys

base, head = (str(Path(p).resolve()) for p in sys.argv[1:3])
script = str(Path(__file__).with_name('diff_fuzz.py').resolve())
for need in (0, 1):
    for mode in ('unset', 'env_empty', 'env_prestart'):
        outputs = []
        for root in (base, head):
            env = dict(os.environ, PYTHONPATH=root + '/src' + os.pathsep + os.environ.get('PYTHONPATH', ''))
            out = subprocess.check_output([sys.executable, script, mode, '3000', '92', root, str(need)], env=env)
            outputs.append(json.loads(out))
        assert outputs[0] == outputs[1], (need, mode)
        exceptions = sum(bool(x) and x[0] == 'EXC' for x in outputs[0])
        assert not exceptions, (need, mode, exceptions)
        print(f'NEED_RULE={need} {mode}: 3000 comparisons IDENTICAL; 0 exceptions', flush=True)
print('18000 comparisons IDENTICAL', flush=True)
