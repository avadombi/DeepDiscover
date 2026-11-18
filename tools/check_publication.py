"""Validate figure files and documentation paths before deploying the site."""
from __future__ import annotations
from pathlib import Path
from urllib.parse import urlparse
from xml.etree import ElementTree as ET
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/'docs'/'_static'/'figures'
DOCS=ROOT/'docs'
BAD=[]


def validate_figures():
    for i in range(1,6):
        f=FIG/f'Fig{i}.svg'
        if not f.is_file():
            BAD.append(f'Missing {f.relative_to(ROOT)}')
            continue
        text=f.read_text(encoding='utf-8-sig')
        if 'DEEPDISCOVER_FIGURE_PLACEHOLDER' in text:
            BAD.append(f'{f.name} still contains the preview placeholder')
        try:
            tree=ET.fromstring(text)
            if tree.tag.split('}')[-1].lower()!='svg':
                BAD.append(f'{f.name} is not an SVG')
            for el in tree.iter():
                tag=el.tag.split('}')[-1].lower()
                if tag in ('script','foreignobject'):
                    BAD.append(f'{f.name} contains an active element: {tag}')
                for k,v in el.attrib.items():
                    attr=k.split('}')[-1].lower()
                    if attr.startswith('on') and len(attr)>2:
                        BAD.append(f'{f.name} contains an event handler')
                    if attr in ('href','src') and v.strip().lower().startswith(('http:','https:','file:','javascript:')):
                        BAD.append(f'{f.name} contains an external link or resource')
        except ET.ParseError as ex:
            BAD.append(f'Invalid XML in {f.name}: {ex}')


def validate_links():
    for file in DOCS.rglob('*.md'):
        source=file.read_text(encoding='utf-8')
        for target in re.findall(r'(?<!!)[^\]]+\]\(([^)]+)\)',source):
            target=target.split('#',1)[0].strip()
            if not target or urlparse(target).scheme or target.startswith('/'):
                continue
            if not (file.parent/target).exists():
                BAD.append(f'{file.relative_to(ROOT)}: missing link {target}')
        for target in re.findall(r'(?m)^```\{figure\}\s+(\S+)',source):
            if not (file.parent/target).exists():
                BAD.append(f'{file.relative_to(ROOT)}: missing figure {target}')


if __name__=='__main__':
    validate_figures()
    validate_links()
    if BAD:
        print('Publication checks need attention:')
        for b in BAD:print(' -',b)
        sys.exit(1)
    print('Publication checks passed. Five user SVG figures are present, and all local links resolve.')
