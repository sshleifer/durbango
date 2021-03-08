from pathlib import Path
import shutil
from tqdm import tqdm
from fire import Fire
NG_PATH ='/private/home/namangoyal/dataset/data-bin/bookwiki_CC-NEWS_openwebtext_stories_cc100-mmap2-bin'

def shard_cc(path=NG_PATH, dest_dir='sharded_cc'):
    CC = Path(path)
    Path(dest_dir).mkdir(exist_ok=True, parents=True)
    for p in tqdm(CC.ls()):
        fname = p.name # eg train11.bin or train11.idx
        if 'train' in fname:
            number_and_ext = fname.split('train')[-1]
            number, ext = number_and_ext[:-4], number_and_ext[-4:] # TODO: os.path.splittext
            if not number: number = '0'
            dest = f'{dest_dir}/shard{number}/train{ext}'
            Path(dest).parent.mkdir(exist_ok=True)
            shutil.copyfile(p, dest)
        elif 'valid' in fname:
            # Dont shard valid
            dest = f'{dest_dir}sharded_cc/{fname}'
            shutil.copyfile(p, dest)
        elif fname == 'dict.txt':
            dest = f'{dest_dir}/dict.txt'
            shutil.copyfile(p, dest)

    # copy dict.txt to each shard
    for p in Path(dest_dir).ls():
        if p.is_dir():
            shutil.copy(f'{dest_dir}/dict.txt', p / 'dict.txt')


if __name__ == '__main__':
    Fire(shard_cc)
