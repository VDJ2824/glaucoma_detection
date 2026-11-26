"""Platform/guard/requirement-marker checks; these do not emulate a full Python runtime."""
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/'notebooks/03_individual_model_training.ipynb'


def code_cells():
    return [''.join(c['source']) for c in json.loads(NOTEBOOK.read_text())['cells'] if c['cell_type']=='code']


@pytest.mark.parametrize('kaggle,colab,expected',[(True,True,'kaggle'),(True,False,'kaggle'),(False,True,'colab'),(False,False,'local')])
def test_platform_detection_prioritizes_kaggle(kaggle,colab,expected,monkeypatch):
    for name in ('KAGGLE_KERNEL_RUN_TYPE','KAGGLE_URL_BASE','COLAB_RELEASE_TAG'):
        monkeypatch.delenv(name,raising=False)
    class PlatformPath(type(Path())):
        def is_dir(self):
            if str(self)=='/kaggle/working':return kaggle
            if str(self)=='/content':return colab
            return super().is_dir()
    source=code_cells()[0].replace('from pathlib import Path','')
    namespace={'Path':PlatformPath}
    exec(source,namespace)
    assert namespace['PLATFORM']==expected
    if expected=='kaggle':
        assert str(namespace['REPO_ROOT']).startswith('/kaggle/working')
        assert '/content/drive' not in str(namespace['SESSION_FILE'])
    elif expected=='colab':
        assert str(namespace['SESSION_FILE']).startswith('/content/drive')
    else:
        assert str(namespace['SESSION_FILE']).startswith(str(namespace['REPO_ROOT']))
    for flag in ('RUN_CLOUD_SETUP','CLONE_REPOSITORY','INSTALL_DEPENDENCIES','RELOCATE_FROZEN_SESSION',
                 'LOAD_METADATA','BUILD_MODEL','RUN_FORWARD_SMOKE','RUN_STAGE1','RUN_STAGE2',
                 'RESUME_STAGE1','RESUME_STAGE2','SHOW_HISTORY','SAVE_HISTORY_PLOT','ALLOW_IMAGENET_DOWNLOAD'):
        assert namespace[flag] is False
    assert namespace['WEIGHTS'] is None


@pytest.mark.parametrize('version,valid',[((3,9),False),((3,10),True),((3,12),True),((3,13),True),((3,14),False)])
def test_setup_version_guard_without_mount_or_clone(version,valid):
    namespace={'RUN_CLOUD_SETUP':True,'CLONE_REPOSITORY':False,'INSTALL_DEPENDENCIES':False,
               'PLATFORM':'local','REPO_ROOT':ROOT,'sys':SimpleNamespace(version_info=version,path=[])}
    if valid:exec(code_cells()[1],namespace)
    else:
        with pytest.raises(RuntimeError,match='3.10–3.13'):exec(code_cells()[1],namespace)


@pytest.mark.parametrize('version,numpy_version,tf_version',[('3.12','2.0.2','2.20.0'),('3.13','2.1.0','2.21.0')])
def test_installation_markers_support_target_python(version,numpy_version,tf_version):
    metadata=(ROOT/'pyproject.toml').read_text()
    python_range=re.search(r'requires-python = "([^"]+)"',metadata).group(1)
    assert version in SpecifierSet(python_range)
    for filename in ('requirements.txt','requirements-data.txt','requirements-training.txt'):
        active=[]
        for line in (ROOT/filename).read_text().splitlines():
            if not line or line.startswith(('#','-')):continue
            req=Requirement(line)
            if req.marker is None or req.marker.evaluate({'python_version':version}):active.append(req)
        for name,selected in [('numpy',numpy_version),('tensorflow',tf_version)]:
            candidates=[req for req in active if req.name==name]
            if candidates:
                assert len(candidates)==1 and selected in candidates[0].specifier
