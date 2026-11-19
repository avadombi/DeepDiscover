"""Generate source-based API pages without importing the library or TensorFlow."""
from __future__ import annotations
import ast
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'src'/'deepdiscover'  # src/deepdiscover is the source of truth when present.
DEST=ROOT/'docs'/'reference'/'generated'


def generate():
    if not SRC.is_dir():
        print('Package source absent: using the bundled API overview for standalone preview.')
        return
    DEST.mkdir(parents=True,exist_ok=True)
    modules=[]
    for file in sorted(SRC.glob('*.py')):
        if file.stem.startswith('_'):continue
        tree=ast.parse(file.read_text(encoding='utf-8'),filename=str(file))
        parts=[f'# `deepdiscover.{file.stem}`','', 'Public classes and functions of this module.','']
        entries=0
        for n in tree.body:
            if isinstance(n,ast.ClassDef) and not n.name.startswith('_'):
                parts.extend([f'## `{n.name}`',''])
                entries+=1
                init=next((m for m in n.body if isinstance(m,(ast.FunctionDef,ast.AsyncFunctionDef)) and m.name=='__init__'),None)
                if init:
                    # Display signature without method decorators or source body.
                    parts.extend(['```python',n.name+'('+ast.unparse(init.args).removeprefix('self, ').removeprefix('self')+')','```',''])
                for m in n.body:
                    if isinstance(m,(ast.FunctionDef,ast.AsyncFunctionDef)) and not m.name.startswith('_'):
                        sig=ast.unparse(m.args).removeprefix('self, ').removeprefix('self')
                        parts.extend([f'### `{m.name}`','', '```python',f'{m.name}({sig})','```',''])
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and not n.name.startswith('_'):
                parts.extend([f'## `{n.name}`','', '```python',f'{n.name}({ast.unparse(n.args)})','```',''])
                entries+=1
        if entries:
            (DEST/f'{file.stem}.md').write_text('\n'.join(parts).rstrip()+'\n',encoding='utf-8')
            modules.append(file.stem)
    toc=['# Detailed API pages','','Generated from the public package source.','','```{toctree}',':maxdepth: 2','',*modules,'```','']
    (DEST/'index.md').write_text('\n'.join(toc),encoding='utf-8')
    print(f'API documentation generated for {len(modules)} modules.')


if __name__=='__main__':generate()
