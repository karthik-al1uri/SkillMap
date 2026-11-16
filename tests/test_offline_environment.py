"""Offline checks assess installed metadata without installations or API access."""
import importlib.metadata
import subprocess
import sys
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from scripts.check_offline_environment import (check, direct_requirements, installed_compatibility,
                                               offline_environment, run_child)


def test_recursive_direct_constraints_and_markers(tmp_path):
    (tmp_path/'base.txt').write_text('plotly>=5.19\nscikit-learn>=1.5\n')
    (tmp_path/'test.txt').write_text('-r base.txt\npytest>=8 # test runner\n')
    requirements=direct_requirements(tmp_path/'test.txt')
    assert [r.name for r in requirements]==['plotly','scikit-learn','pytest']
    def installed(name):
        versions={'plotly':'5.18.0','scikit-learn':'1.5.0','pytest':'9.0.2'}
        return versions[name]
    rows=installed_compatibility(requirements,installed)
    assert [r['compatible'] for r in rows]==[False,True,True]
    missing=installed_compatibility([Requirement('absent>=1')],lambda name: (_ for _ in ()).throw(importlib.metadata.PackageNotFoundError(name)))
    assert missing[0]['installed_version'] is None and not missing[0]['compatible']
    inactive=installed_compatibility([Requirement('absent; python_version<"2"')],installed)
    assert inactive[0]['compatible'] and not inactive[0]['applicable']


def test_requirement_options_and_urls_are_not_resolved(tmp_path):
    path=tmp_path/'requirements.txt'
    for text in ('--index-url https://example.invalid\n','thing @ https://example.invalid/x.whl\n'):
        path.write_text(text)
        with pytest.raises(ValueError):
            direct_requirements(path)


def test_guard_blocks_child_socket_before_connection(tmp_path):
    root=tmp_path/'source';root.mkdir()
    out=tmp_path/'external';out.mkdir()
    env=offline_environment(root,out)
    result=run_child([sys.executable,'-B','-c','import socket; socket.create_connection(("127.0.0.1",9))'],root,env)
    assert result['returncode']!=0
    assert 'SkillMap offline check blocks Python network access' in result['stderr']
    assert env['MPLBACKEND']=='Agg' and env['OMP_NUM_THREADS']=='1' and env['PIP_NO_INDEX']=='1'
    assert not list(root.iterdir())


def test_output_guard_prevents_source_or_existing_writes(tmp_path):
    root=tmp_path/'source';root.mkdir()
    existing=tmp_path/'existing';existing.mkdir()
    for destination in (root,root/'new',existing):
        with pytest.raises(ValueError):
            check(root,destination)
    assert not list(root.iterdir())


def test_guard_preserves_original_interpreter_startup_hook(tmp_path):
    root=tmp_path/'source';root.mkdir()
    (root/'sitecustomize.py').write_text('import os\nos.environ["SKILLMAP_TEST_STARTUP"] = "preserved"\n')
    out=tmp_path/'external';out.mkdir()
    env=offline_environment(root,out)
    result=run_child([sys.executable,'-B','-c','import os,socket; print(os.environ.get("SKILLMAP_TEST_STARTUP")); socket.create_connection(("127.0.0.1",9))'],root,env)
    assert result['stdout'].strip()=='preserved'
    assert result['returncode']!=0
    assert 'SkillMap offline check blocks Python network access' in result['stderr']
