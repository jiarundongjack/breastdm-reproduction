"""Verify and restore downloaded ZIP assets beside code/data/result.
Usage: python "pretrained and trained weights/restore_assets.py" --asset-dir path/to/downloaded/assets [--verify-only]
"""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,zipfile,shutil

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-dir',type=Path,required=True)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    manifest_dir=Path(__file__).resolve().parent
    root=manifest_dir.parent
    manifest=json.loads((manifest_dir/'ASSETS.json').read_text(encoding='utf-8'))
    with (manifest_dir/'FILES.json').open('r',encoding='utf-8') as f:records=json.load(f)
    for asset in manifest['assets']:
        archive=args.asset_dir/asset['name']
        if not archive.is_file():raise FileNotFoundError(f'Missing release asset: {archive}')
        if archive.stat().st_size!=asset['bytes'] or digest(archive)!=asset['sha256']:
            raise ValueError(f'Archive checksum mismatch: {archive}')
        expected={r['path']:r for r in records if r['storage']==asset['name']}
        with zipfile.ZipFile(archive) as z:
            if set(z.namelist())!=set(expected) or len(z.namelist())!=len(expected):
                raise ValueError(f'Unexpected ZIP contents: {archive}')
            for member in z.infolist():
                rel=PurePosixPath(member.filename)
                dest=(root/rel).resolve()
                if rel.is_absolute() or '..' in rel.parts or root not in dest.parents:
                    raise ValueError(f'Unsafe ZIP member: {member.filename}')
                if args.verify_only:
                    h=hashlib.sha256()
                    with z.open(member) as f:
                        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
                    if h.hexdigest()!=expected[member.filename]['sha256']:raise ValueError(member.filename)
                    continue
                if dest.exists():
                    if digest(dest)!=expected[member.filename]['sha256']:
                        raise FileExistsError(f'Existing different file; refusing overwrite: {dest}')
                    continue
                dest.parent.mkdir(parents=True,exist_ok=True)
                with z.open(member) as src,dest.open('xb') as out:shutil.copyfileobj(src,out)
                if digest(dest)!=expected[member.filename]['sha256']:raise ValueError(f'Extracted checksum mismatch: {dest}')
        print('Verified' if args.verify_only else 'Restored',asset['name'],flush=True)
    print('All assets passed.')

if __name__=='__main__':main()
