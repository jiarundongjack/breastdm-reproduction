"""Offline, hash-checked setup for user-supplied source; see THIRD_PARTY_CODE.md.

This project-authored utility contains no model, training or metric implementation.
It does not download source, grant permissions, or execute supplied Python files.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def records():
    return json.loads((ROOT / 'provenance/source_inventory.json').read_text(encoding='utf-8'))['files']


def normalized(data):
    return data.decode('utf-8-sig').replace('\r\n', '\n').replace('\r', '\n')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def contained(root, relative):
    root = root.resolve()
    path = root / relative
    if not path.resolve().is_relative_to(root) or path.is_symlink():
        raise ValueError('Linked or escaping path: ' + str(path))
    for parent in path.parents:
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError('Linked parent: ' + str(parent))
    return path


def valid(data, row):
    return digest(normalized(data).encode('utf-8')) == row['normalized_sha256']


def missing_sources(root=None):
    root = ROOT if root is None else root
    missing = []
    for row in records():
        if row['action'] != 'remove':
            continue
        path = contained(root, row['path'])
        if not path.is_file() or not valid(path.read_bytes(), row):
            missing.append(row['path'])
    return missing


def require_sources(root=None):
    missing = missing_sources(root)
    if missing:
        raise RuntimeError(
            f'{len(missing)} source files are absent or do not match the audited version. '
            'Run python code/setup_third_party.py --check and follow THIRD_PARTY_CODE.md. '
            'Original BreastDM sources alone cannot supply unresolved reproduction adaptations.')


def restore(private_source=None, breastdm_source=None, paths=None):
    selected = [r for r in records() if r['action'] == 'remove'
                and (not paths or r['path'] in paths)]
    if paths and set(paths) != {r['path'] for r in selected}:
        raise ValueError('--path must name a removed file in the inventory.')
    pending, unavailable = [], []
    for row in selected:
        target = contained(ROOT, row['path'])
        if target.exists():
            if not target.is_file() or not valid(target.read_bytes(), row):
                raise ValueError('Refusing to overwrite a different file: ' + row['path'])
            continue
        data = None
        if private_source:
            source = contained(private_source, row['path'])
            if source.is_file():
                data = source.read_bytes()
                if not valid(data, row):
                    raise ValueError('Private source hash mismatch: ' + row['path'])
        recipe = row.get('restoration_recipe')
        if data is None and breastdm_source and recipe:
            source = contained(breastdm_source, recipe['source_path'])
            if source.is_file():
                text = normalized(source.read_bytes())
                if digest(text.encode()) != recipe['source_normalized_sha256']:
                    raise ValueError('Upstream source hash mismatch: ' + recipe['source_path'])
                if recipe.get('insert_line') is not None:
                    lines = text.splitlines(keepends=True)
                    lines.insert(recipe['insert_index'], recipe['insert_line'] + '\n')
                    text = ''.join(lines)
                data = text.encode('utf-8')
        if data is None:
            unavailable.append(row['path'])
            continue
        if not valid(data, row):
            raise ValueError('Restored source does not match the audited file: ' + row['path'])
        # Reproduce the audited Git blob exactly, including its line endings/BOM.
        text = normalized(data)
        endings = [ending for ending, count in row['newline_runs'] for _ in range(count)]
        lines = text.split('\n')
        if len(lines) != len(endings) + 1:
            raise ValueError('Line-ending metadata mismatch: ' + row['path'])
        exact = ''.join(line + ending for line, ending in zip(lines[:-1], endings)) + lines[-1]
        data = (b'\xef\xbb\xbf' if row.get('utf8_bom') else b'') + exact.encode('utf-8')
        if digest(data) != row['sha256']:
            raise ValueError('Byte identity check failed: ' + row['path'])
        pending.append((target, data))
    # Validate the complete requested set before making any writes.
    if unavailable:
        raise RuntimeError('No verified source supplied for:\n' + '\n'.join(unavailable)
                           + '\nProvide your own private source tree, or select known upstream '
                             'files individually using --path. No files were written.')
    for target, data in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)
    return len(pending)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--private-source', type=Path,
                        help='your own previously acquired reproduction source tree; not downloaded')
    parser.add_argument('--breastdm-source', type=Path,
                        help='your own original-author source tree at the documented revision')
    parser.add_argument('--path', action='append', help='restore only this inventory path (repeatable)')
    args = parser.parse_args()
    try:
        if args.check or not (args.private_source or args.breastdm_source):
            missing = missing_sources()
            print('\n'.join(missing) if missing else 'All required restored source hashes match.')
            return 2 if missing else 0
        print(f'Restored {restore(args.private_source, args.breastdm_source, args.path)} files locally.')
        return 0
    except (ValueError, RuntimeError, OSError, UnicodeError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
