from pathlib import Path
import shutil
from tqdm import tqdm
from fire import Fire
NG_PATH='/private/home/namangoyal/dataset/data-bin/bookwiki_CC-NEWS_openwebtext_stories_cc100-mmap2-bin'
def shard_cc(path=NG_PATH):
    CC = Path(path)
    for p in tqdm(CC.ls()):
        fname = p.name
        if 'train' in fname:
            splat = fname.split('train')[-1]
            splat, suffix = splat[:-4], splat[-4:]
            if not splat: splat = '0'
            dest = f'sharded_cc/shard{splat}/train{suffix}'
            Path(dest).parent.mkdir(exist_ok=True, parents=True)
            shutil.copyfile(p, dest)
        elif 'valid' in fname:
            splat = fname.split('valid')[-1]
            splat, suffix = splat[:-4], splat[-4:]
            if not splat: splat = '0'
            dest = f'sharded_cc/shard{splat}/valid{suffix}'
            Path(dest).parent.mkdir(exist_ok=True, parents=True)
            shutil.copyfile(p, dest)
        elif fname == 'dict.txt':
            dest = 'sharded_cc/dict.txt'
            shutil.copyfile(p, dest)

if __name__ == '__main__':
    Fire(shard_cc)
