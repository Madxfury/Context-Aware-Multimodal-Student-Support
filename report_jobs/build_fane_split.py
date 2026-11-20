"""Build the FANE split used to retrain Swin-FANE for the report.

Test  = the 406 test files the original model never saw in training
        (model_predictions_on_project_test_split.csv, image_also_in_train == False).
Train/val = every other distinct image (by MD5) that
        - is not one of the test images,
        - has no contradictory label (label_conflict == False),
        - does not share a 64-bit difference hash with any test image (near-duplicate).
        Split 85/15 within each emotion, random seed 42.
Usage: python build_fane_split.py <Samvya_Dataset/1_facial_emotion_FANE dir> <out.csv>
"""
import sys, numpy as np, pandas as pd
from PIL import Image

SEED = 42
root, out = sys.argv[1], sys.argv[2]
man = pd.read_csv(f"{root}/image_manifest.csv")
sp = pd.read_csv(f"{root}/split_used_in_project.csv")
pr = pd.read_csv(f"{root}/model_predictions_on_project_test_split.csv")

def dhash(path):
    im = Image.open(path).convert("L").resize((9, 8), Image.LANCZOS)
    a = np.asarray(im, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)

unseen = pr[~pr.image_also_in_train].copy()
t = sp[sp.split == "test"][["file_in_split", "md5", "same_image_in_images_folder"]]
test = unseen.merge(t, on="file_in_split", how="left", validate="one_to_one")
assert len(test) == 406 and test.md5.notna().all()
test["file"] = test.same_image_in_images_folder
test["emotion"] = test.true_emotion
test["split"] = "test"
test["dhash"] = [dhash(f"{root}/{f}") for f in test.file]

pool = man[~man.label_conflict & ~man.md5.isin(set(test.md5))].drop_duplicates("md5").copy()
pool["dhash"] = [dhash(f"{root}/{f}") for f in pool.file]
n_before = len(pool)
pool = pool[~pool.dhash.isin(set(test.dhash))]
print(f"pool distinct images {n_before}, removed as near-duplicates of test {n_before-len(pool)}, kept {len(pool)}")

rng = np.random.default_rng(SEED)
parts = []
for emo, g in pool.groupby("emotion"):
    idx = rng.permutation(len(g)); n_val = int(round(0.15 * len(g)))
    g = g.iloc[idx].copy(); g["split"] = ["val"] * n_val + ["train"] * (len(g) - n_val)
    parts.append(g)
tv = pd.concat(parts)
res = pd.concat([tv[["file", "emotion", "split", "md5", "dhash"]], test[["file", "emotion", "split", "md5", "dhash", "file_in_split"]]])
res.to_csv(out, index=False)
print(res.groupby(["split", "emotion"]).size().unstack(fill_value=0))
print("test distinct md5:", test.md5.nunique(), "| overlap train/val vs test md5:", len(set(tv.md5) & set(test.md5)), "| dhash overlap:", len(set(tv.dhash) & set(test.dhash)))
