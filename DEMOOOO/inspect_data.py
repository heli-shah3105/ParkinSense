"""
inspect_data.py - READ-ONLY diagnostic (changes nothing, writes nothing)

Answers, in one run:
  1. Does Folder 2 actually have .wav files on disk, and do the CSV paths resolve?
  2. How are Folder 3 files named, and which (speaker, session, utterance) keys do they have?
  3. How many Folder 3 files can be matched to a CORRECT prompt from the Folder 2 CSV
     using (speaker, session, utterance)?  (The current step1 uses utterance only - wrong.)
  4. What does Folder 1 look like (file names / word codes)?

Usage:  python inspect_data.py
Then paste the whole output back.
"""
import os
import re
import glob
from collections import Counter, defaultdict

import pandas as pd

from config import FOLDER1_ROOT, FOLDER2_ROOT, FOLDER2_CSV, FOLDER3_ROOT

KAGGLE_PREFIX = "/kaggle/input/dysarthria-and-nondysarthria-speech-dataset/"
CSV_RE = re.compile(r"Dataset/(?P<cat>[^/]+)/(?P<spk>[^/]+)/Session(?P<sess>\d+)/Wav/(?P<utt>\d+)\.wav", re.I)
F3_DIR_RE = re.compile(r"wav_(?P<mic>arrayMic|headMic)_(?P<spk>[A-Za-z]+\d+)(?:S(?P<sess>\d+))?$")


def hr(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def last_int(stem):
    nums = re.findall(r"\d+", stem)
    return int(nums[-1]) if nums else None


# ------------------------------------------------------------------ Folder 2
hr("FOLDER 2: CSV + audio on disk")
csv_df = pd.read_csv(FOLDER2_CSV)
print(f"CSV rows: {len(csv_df)}  columns: {list(csv_df.columns)}")

wavs_f2 = glob.glob(os.path.join(FOLDER2_ROOT, "**", "*.wav"), recursive=True)
print(f".wav files found under {FOLDER2_ROOT}: {len(wavs_f2)}")
for p in wavs_f2[:3]:
    print("   e.g.", p)

# Does the remapped CSV path exist?
exists = 0
for p in csv_df["Wav_path"].head(200):
    rel = p[len(KAGGLE_PREFIX):] if str(p).startswith(KAGGLE_PREFIX) else str(p).lstrip("/")
    if os.path.exists(os.path.join(FOLDER2_ROOT, rel)):
        exists += 1
print(f"First 200 CSV rows whose remapped path exists locally: {exists}/200")

csv_keys = {}
csv_cat = {}
bad_regex = 0
for _, r in csv_df.iterrows():
    m = CSV_RE.search(str(r["Wav_path"]).replace("\\", "/"))
    if not m:
        bad_regex += 1
        continue
    key = (m["spk"].upper(), int(m["sess"]), int(m["utt"]))
    csv_keys[key] = str(r["Prompts"])
    csv_cat[m["spk"].upper()] = m["cat"]
print(f"CSV rows parsed into (speaker, session, utt) keys: {len(csv_keys)}  (unparsed: {bad_regex})")
print("CSV speakers -> category:")
spk_sessions = defaultdict(set)
for (s, se, u) in csv_keys:
    spk_sessions[s].add(se)
for s in sorted(spk_sessions):
    print(f"   {s:6s} {csv_cat.get(s, ''):28s} sessions={sorted(spk_sessions[s])}")

n_instr = sum(1 for v in csv_keys.values() if v.strip().startswith("["))
n_img = sum(1 for v in csv_keys.values() if v.strip().startswith("input/images"))
print(f"Prompts that are instructions '[...]': {n_instr}   image prompts: {n_img}   real text: {len(csv_keys)-n_instr-n_img}")

# ------------------------------------------------------------------ Folder 3
hr("FOLDER 3: structure and file names")
f3_files = []   # (category, mic, spk, sess, utt, path)
for cat in sorted(os.listdir(FOLDER3_ROOT)) if os.path.isdir(FOLDER3_ROOT) else []:
    cat_dir = os.path.join(FOLDER3_ROOT, cat)
    if not os.path.isdir(cat_dir):
        continue
    for d in sorted(os.listdir(cat_dir)):
        dp = os.path.join(cat_dir, d)
        if not os.path.isdir(dp):
            continue
        m = F3_DIR_RE.match(d)
        files = glob.glob(os.path.join(dp, "**", "*.wav"), recursive=True)
        print(f"{cat:6s} {d:26s} wavs={len(files):5d}  e.g. {os.path.basename(files[0]) if files else '-'}")
        if not m:
            continue
        for f in files:
            f3_files.append((cat, m["mic"], m["spk"].upper(),
                             int(m["sess"]) if m["sess"] else None,
                             last_int(os.path.splitext(os.path.basename(f))[0]), f))
print(f"\nTotal Folder 3 wavs parsed: {len(f3_files)}")

# ------------------------------------------------------------------ Join test
hr("JOIN TEST: Folder 3 (speaker, session, utt) -> Folder 2 CSV prompt")
by_spk = defaultdict(lambda: Counter())
examples = defaultdict(list)
for cat, mic, spk, sess, utt, path in f3_files:
    if mic != "headMic":
        continue
    if sess is not None:
        hit = (spk, sess, utt) in csv_keys
        how = "spk+session+utt"
    else:
        # no session in the folder name: try every session the CSV has for this speaker
        cands = [(spk, s, utt) for s in spk_sessions.get(spk, []) if (spk, s, utt) in csv_keys]
        hit = len(cands) > 0
        how = f"spk+utt (no session in folder name; {len(cands)} candidate sessions)"
    by_spk[spk]["matched" if hit else "unmatched"] += 1
    if len(examples[spk]) < 2:
        examples[spk].append((os.path.basename(path), sess, utt, hit, how))

for spk in sorted(by_spk):
    c = by_spk[spk]
    tot = c["matched"] + c["unmatched"]
    in_csv = "yes" if spk in spk_sessions else "NO (speaker not in CSV)"
    print(f"{spk:6s} headMic wavs={tot:5d} matched={c['matched']:5d} unmatched={c['unmatched']:5d}  in_csv={in_csv}")
    for ex in examples[spk]:
        print(f"        {ex}")

# ------------------------------------------------------------------ Folder 1
hr("FOLDER 1: UASpeech file names")
f1 = glob.glob(os.path.join(FOLDER1_ROOT, "*", "*.wav"))
print(f"wavs: {len(f1)}")
codes = Counter()
for p in f1:
    m = re.match(r"^[A-Z]+\d+_B\d+_([A-Z]+\d+)_M\d+$", os.path.splitext(os.path.basename(p))[0])
    if m:
        codes[re.match(r"[A-Z]+", m.group(1)).group(0)] += 1
print("word-code prefixes and counts:", dict(codes))
print("examples:", [os.path.basename(p) for p in f1[:4]])
print("\nDONE - paste everything above.")