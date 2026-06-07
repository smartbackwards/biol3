# L3 — Setup & Run Checklist

## 1. Environment

```
pip install deepface opencv-python numpy matplotlib tqdm tf-keras
```

If the machine has a GPU:
```
pip install tensorflow[and-cuda]
```

Python 3.10+ required.

---

## 2. One-time fix — swap RetinaFace → MTCNN as primary detector

Edit `verify.py` line with `BACKENDS`:

```python
# change this:
BACKENDS = ("retinaface", "mtcnn", "ssd", "opencv")

# to this:
BACKENDS = ("mtcnn", "ssd", "opencv", "retinaface")
```

This drops exp1 from ~4 h to ~20-30 min on CPU. On GPU it's ~5 min.

---

## 3. Generate data splits

```
python prepare_data.py
```

Reads CelebA images from `../L1/facial_auth/celeba_subset/`.
Writes `data/enroll.csv`, `data/test_clean.csv`, `data/test_lowres.csv`, `data/test_occluded.csv`.

---

## 4. Enroll CelebA identities (100 users)

```
python enroll.py
```

Writes `database/embeddings.pkl`. Takes ~10-20 min.

---

## 5. Enroll group members (REQUIRED by assignment)

Take 5-10 photos per person (phone is fine). Save each person's photos in a separate folder.

```
python enroll_member.py --name "Name 1" --photos_dir my_photos/person1/
python enroll_member.py --name "Name 2" --photos_dir my_photos/person2/
python enroll_member.py --name "Name 3" --photos_dir my_photos/person3/
```

Appends to the existing `database/embeddings.pkl`.
python enroll_member.py --name "bartek m" --photos_dir my_photos/bm
python enroll_member.py --name "bartek t" --photos_dir my_photos/bt
python enroll_member.py --name "krzysiek" --photos_dir my_photos/ks
---

## 6. Run experiments

Run in order — exp2 and exp3 load the baseline EER from exp1.

```
python experiments/exp1_clean.py
python experiments/exp2_lowres.py
python experiments/exp3_occlusion.py
```

Expected runtimes (CPU, with MTCNN fix):

| Experiment | Trials | Estimated time |
|---|---|---|
| exp1 clean | 1 000 | 20-30 min |
| exp2 low-res | 5 000 | 1.5-2.5 h |
| exp3 occlusion | 5 000 | 1.5-2.5 h |

To re-run after a partial failure add `--rerun`:
```
python experiments/exp1_clean.py --rerun
```

---

## 7. Check outputs

```
results/
  exp1/  scores.csv  metrics.txt  roc.png  scores_clean.pkl
  exp2/  scores.csv  metrics.txt  roc.png
  exp3/  scores.csv  metrics.txt  roc.png
```

`metrics.txt` in each folder contains EER, FAR, FRR and the Δ vs clean baseline.

---

## 8. Demo (for the lab presentation)

```
python demo.py <path_to_image.jpg> <identity_id>
python demo.py <path_to_image.jpg> <identity_id> --simulate lowres
python demo.py <path_to_image.jpg> <identity_id> --simulate mask
python demo.py <path_to_image.jpg> <identity_id> --simulate glasses
```

Use one of your own enrolled photos + your identity id for the live demo.


conda env create -f bio64_env.yml