"""Check the installed environment offline; never resolve or install packages."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from packaging.requirements import Requirement

NETWORK_GUARD = '''import importlib.machinery
import importlib.util
import os
import socket
import sys

def _skillmap_no_network(*args, **kwargs):
    raise RuntimeError("SkillMap offline check blocks Python network access")

def _skillmap_block_network():
    socket.socket.connect = _skillmap_no_network
    socket.socket.connect_ex = _skillmap_no_network
    socket.create_connection = _skillmap_no_network
    socket.getaddrinfo = _skillmap_no_network
    socket.gethostbyname = _skillmap_no_network
    socket.gethostbyname_ex = _skillmap_no_network

_skillmap_block_network()
# Chain the interpreter's existing startup hook rather than shadowing it.
# Homebrew uses this hook to retain its global site-packages location.
_guard_directory = os.path.realpath(os.path.dirname(__file__))
_original_paths = [p for p in sys.path if os.path.realpath(p or os.getcwd()) != _guard_directory]
_original_spec = importlib.machinery.PathFinder.find_spec("sitecustomize", _original_paths)
if _original_spec is not None and _original_spec.loader is not None:
    _original_hook = importlib.util.module_from_spec(_original_spec)
    _original_spec.loader.exec_module(_original_hook)
_skillmap_block_network()
'''



def direct_requirements(path, seen=None):
    """Read repository requirement files recursively; no resolution or network access."""
    path = Path(path).resolve()
    seen = set() if seen is None else seen
    if path in seen:
        return []
    seen.add(path)
    requirements = []
    for line in path.read_text().splitlines():
        line = line.split(' #', 1)[0].strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('-r '):
            requirements += direct_requirements(path.parent / line[3:].strip(), seen)
            continue
        if line.startswith('-'):
            raise ValueError(f'Unsupported requirement option: {line}')
        requirement = Requirement(line)
        if requirement.url:
            raise ValueError('URL requirements cannot be checked as an installed-version constraint')
        requirements.append(requirement)
    return requirements


def installed_compatibility(requirements, version_lookup=importlib.metadata.version):
    """Compare installed metadata to direct constraints; do not import heavy libraries."""
    rows = []
    for requirement in requirements:
        applicable = requirement.marker is None or requirement.marker.evaluate()
        if not applicable:
            rows.append({'requirement':str(requirement), 'applicable':False, 'compatible':True})
            continue
        try:
            installed = version_lookup(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        rows.append({'requirement':str(requirement), 'applicable':True, 'installed_version':installed,
                     'compatible':installed is not None and requirement.specifier.contains(installed, prereleases=True)})
    return rows


def offline_environment(root, out):
    """Create an inherited Python socket guard and isolate caches/fixtures externally."""
    guard = out/'network-guard'
    guard.mkdir()
    (guard/'sitecustomize.py').write_text(NETWORK_GUARD)
    env = os.environ.copy()
    env.update({'PYTHONDONTWRITEBYTECODE':'1', 'MPLBACKEND':'Agg', 'MPLCONFIGDIR':str(out/'mpl'),
                'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1',
                'NUMEXPR_NUM_THREADS':'1', 'VECLIB_MAXIMUM_THREADS':'1',
                'PIP_NO_INDEX':'1', 'PIP_DISABLE_PIP_VERSION_CHECK':'1', 'PIP_NO_CACHE_DIR':'1',
                'PYTHONPATH':os.pathsep.join((str(guard),str(root)))})
    return env


def run_child(command, root, env, timeout=120):
    """Capture an offline read-only subprocess result without shell execution."""
    try:
        completed = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=timeout)
        return {'command':command,'returncode':completed.returncode,
                'stdout':completed.stdout,'stderr':completed.stderr,'timed_out':False}
    except subprocess.TimeoutExpired as exc:
        def decode(value):
            return value.decode(errors='replace') if isinstance(value, bytes) else (value or '')
        return {'command':command,'returncode':None,'stdout':decode(exc.stdout),'stderr':decode(exc.stderr),
                'timed_out':True}


def check(root, out, run_pytest=False):
    root, out = Path(root).resolve(strict=True), Path(out).resolve()
    if out == root or root in out.parents or out.exists():
        raise ValueError('Use a fresh output directory outside the source root')
    requirements = direct_requirements(root/'requirements-test.txt')
    compatibility = installed_compatibility(requirements)
    out.mkdir(parents=True)
    env = offline_environment(root,out)
    pip_check = run_child([sys.executable,'-B','-m','pip','check'],root,env)
    guard_check = run_child([sys.executable,'-B','-c',
                            'import socket; socket.create_connection(("127.0.0.1", 9))'],root,env)
    guard_verified = guard_check['returncode'] != 0 and 'SkillMap offline check blocks Python network access' in guard_check['stderr']
    report = {'scope':'Installed-version compatibility and offline source execution only; no dependency resolution, installation, clean-environment creation or model training.',
              'python_version':sys.version,'python_executable':sys.executable,
              'python_minimum_declared':'3.12','python_minimum_compatible':sys.version_info >= (3,12),
              'ci_python_target':'3.12','running_ci_python_target':sys.version_info[:2] == (3,12),
              'requirements':compatibility, 'all_direct_requirements_compatible':all(x['compatible'] for x in compatibility),
              'pip_check':pip_check, 'python_network_guard_verified':guard_verified,
              'network_guard_scope':'Python socket connect/DNS guard inherited via sitecustomize/PYTHONPATH; not an OS network sandbox and does not restrict non-Python tools.',
              'source_root':str(root),'output_directory':str(out),
              'limitations':['Requirements are lower bounds, not a lockfile or an independently tested minimum-version matrix.',
                             'Installed environment may contain additional packages; passing this check does not prove a fresh offline installation.',
                             'Python3.12 hosted CI is declarative until actually executed.',
                             'No wheels downloaded, dependency installations attempted, or external API requests made.']}
    if run_pytest:
        report['pytest'] = run_child([sys.executable,'-B','-m','pytest','-p','no:cacheprovider',
                                      '--basetemp='+str(out/'pytest'),'tests','-q'],root,env,timeout=300)
    report['passed'] = (report['python_minimum_compatible'] and report['all_direct_requirements_compatible'] and
                        pip_check['returncode'] == 0 and guard_verified and
                        ('pytest' not in report or report['pytest']['returncode'] == 0))
    (out/'environment.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--pytest',action='store_true',help='run source tests with inherited Python network guard')
    args=parser.parse_args(argv)
    report=check(args.root,args.out_dir,args.pytest)
    print(json.dumps({'passed':report['passed'],
                      'all_direct_requirements_compatible':report['all_direct_requirements_compatible'],
                      'pip_check_returncode':report['pip_check']['returncode'],
                      'network_guard_verified':report['python_network_guard_verified'],
                      'report':str(args.out_dir/'environment.json')}))
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
