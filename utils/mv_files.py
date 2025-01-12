import glob
import os
import shutil

import fire


def mv_files(src_dir, tgt_dir, pattern):
    for file in glob.glob(os.path.join(src_dir, pattern)):
        shutil.move(file, os.path.join(tgt_dir, os.path.basename(file)))


if __name__ == "__main__":
    fire.Fire(mv_files)
