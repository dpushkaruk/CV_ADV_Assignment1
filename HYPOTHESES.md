# Hypotheses log

Rule: every block gets its hypothesis + predicted outcome **before** the runs, and its result + explanation
after. Wrong predictions stay in the log. Numbers are LOSO EMD over 24 samples (mean ± std across folds) unless
"8 folds" is stated. Lower is better.

> **About the timeline.** Block 0 (retrospective) and Block 1 (H1.1–H1.4) were committed to this repository on
> 2026-10-26, before the tiled runs (commits `26fb175`, `c371eaf`). The final experiments ran in one Kaggle
> session on 2026-10-31 (run ids carry timestamps, see `results/cv_results.csv`). H1–H7 were written in
> `notebooks/cv-v3.ipynb` before their blocks, in the order shown here, but they were committed to git only
> on 2026-11-01 together with the results, so for them the Kaggle notebook version history is the evidence
> for the order. The CNN runs of the final notebook (frozen CNNs, ConvNeXt fine-tuning, ResNet-18 ablations)
> are covered by Block 1 (H1.2–H1.4). The ResNet-18 ablation details had no separate written prediction and
> are marked *post-hoc*.

---

## Block 0: whole-photo baseline (retrospective, written after the runs, committed 2026-10-26)

Setup: whole photo resized to 224×224, frozen backbone + MLP head, EMD loss, LOSO over 23 samples (`H374` was
lost by a file-name parsing bug, fixed in Block 1), 30 epochs, AdamW 1e-3, head bias = mean training curve.

| run | LOSO EMD | Public LB |
|---|---|---|
| uniform curve | 98.6 | 125.63 (submitted later) |
| mean training curve | 88.0 | 102.37 (submitted later) |
| EfficientNet-B0 (frozen) | 43.5 | 73 |
| ConvNeXtV2-Base (frozen) | 35.9 | 92 |
| mean of the two | 37.6 | 81 |

Observation: CV and LB rank the models in opposite order.

## Block 1: physically calibrated tiling (committed 2026-10-26, before the tiled runs)

**H1.1 (the CV/LB disagreement is a scale/device shift, not just LB noise).** Whole-photo resizing lets the model
learn device-specific appearance instead of grain size, and the test set is a different device at a different
px/mm. *Prediction:* with tiles at one common px/mm, CV and LB rank the models the same way, and the LB of the best
tile model is clearly better than the whole-photo models (< 70).
**Result: refuted.** The tiled models scored 76.55 (MAE ridge), 83.05 (DINOv2 FT) and 86.73 (EfficientNet-B0
frozen) on the public LB, none of them below 70, and tiled EfficientNet-B0 (86.7) is worse than whole-photo
EfficientNet-B0 (73). The CV order (EffB0 39.26 ≈ DINOv2 FT 39.30 < MAE 37.81 within 1.5) does not predict the LB
order either. **Why:** calibration removed the scale part of the shift, but not the rest. The test photos
are from unseen devices with different sharpness/noise/colour after resampling (H5, notebook C11: test→train
feature distance 3.3× the LOSO distance), and the mean-curve baseline itself is 14 EMD worse on the LB than in
CV (102.4 vs 88.3), so the test *soils* differ as well. Whole-photo models may have scored better on the LB by
using the whole field of view (more context per prediction), which tiling throws away.

**H1.2 (4 px/mm, 224-px tiles = 56 mm is a sensible default).** *Prediction:* LOSO EMD of frozen ConvNeXt with
tiles ≤ the whole-photo result (35.9). The error moves to the fine end of the curve, and the coarse end improves
most.
**Result: refuted on the number, supported on where the error is.** Tiled frozen ConvNeXt-T 41.29 and
ConvNeXtV2-B 41.12 are worse than whole-photo ConvNeXtV2-B 35.9 (note: 23 vs 24 samples, and the head and epochs
differ). The error is concentrated at the fine end: fine soils 55–57 vs gravel 30 for every tiled model. 2 and
3 px/mm (112 / 75 mm tiles) give the same CV (39.3–41.2 vs 39.1–40.8), so 4 px/mm is kept as the resolution that
does not upsample any training photo.

**H1.3 (384-px tiles = 96 mm help coarse gravel).** *Prediction:* a small overall gain and a large gain on the
coarsest samples (H038 / H516), with the fine end unchanged.
**Result: refuted.** ConvNeXt-T frozen 41.14 (96 mm) vs 41.29 (56 mm), ResNet-18 FT on 8 folds 36.49 vs 34.74, and
ViT-S ridge 41.19 vs 39.87. H038 stays the worst sample for every model (mean EMD 118). **Why:** a larger tile
gives 4× fewer tiles per photo (3 per Motorola photo at 96 mm), so the training set shrinks. The coarse end is
not limited by the field of view but by the few coarse samples (12 gravels, only a few with > 20 mm grains).

**H1.4 (tile-level training makes fine-tuning viable).** *Prediction:* partial fine-tuning (last stage, frozen BN
statistics) beats the frozen backbone on CV.
**Result: refuted.** Full ConvNeXt-T fine-tuning with frozen BN 41.71 vs frozen 41.29 (MLP) / 39.26 (ridge).
Fine-tuning ViT-S / DINOv2-S is equal to frozen within noise (H3 below). **Why:** ~2.1k tiles are still only 23
independent samples. Fine-tuning distorts the features on the training domain (CKA, H3) without adding
information that the frozen texture statistics do not already carry.

Reference points: uniform curve 98.61, mean training curve (LOSO) **88.32**. The per-sample std is about 30,
so with n = 24 the standard error of a CV mean is about 6 EMD. Paired differences between runs are more
precise, but in the paired bootstrap (notebook C10) none of the runs is significantly better than
EfficientNet-B0 frozen.

---

## H1: frozen features (before C3)

**Hypothesis.** DINOv2/DINOv3 frozen features beat the supervised ViT-S and the ImageNet CNNs
(LOSO EMD < 39), because SSL keeps the texture / low-level cues that supervised ImageNet pretraining discards.
MAE frozen is the worst (it is known for weak linear-probe features). CLS + mean-patch ("avgcat") ≥ CLS only.
Differences are inside the noise (std ≈ 30), so rankings will show in the median and per-class EMD rather than
as a clear separation.

**Result: not supported.**

| run | CV | fine | sand | gravel |
|---|---|---|---|---|
| maeb_frozen_avgcat | **39.45 ± 30.38** | 47.61 | 49.69 | 30.42 |
| vits_sup_frozen_cls | 40.52 ± 31.97 | 51.38 | 47.32 | 31.35 |
| vits_sup_frozen_avgcat | 40.88 ± 31.91 | 53.26 | 48.58 | 30.46 |
| dinov2s_frozen_cls | 42.54 ± 31.20 | 57.56 | 51.94 | 29.87 |
| dinov2s_frozen_avgcat | 42.66 ± 30.93 | 59.35 | 52.54 | 28.80 |
| dinov3s_frozen_avgcat | 42.94 ± 32.00 | 58.14 | 52.62 | 30.04 |
| dinov2b_frozen_avgcat | 45.54 ± 33.90 | 64.72 | 55.87 | 30.05 |

DINO features are the *worst* frozen features here, and MAE-B is the best ViT (≈ EfficientNet-B0, 39.26).
CLS vs CLS + mean patch makes no difference. Gravel is identical for every model (29–31), and the whole spread
comes from fine and sand soils.
**Why.** Fines (< 0.063 mm) are sub-pixel at 4 px/mm, so the fine end of the curve can only be inferred from
colour and low-level texture. MAE is trained to reconstruct pixels and keeps those cues, while DINO is trained
to be *invariant* to colour jitter and blur, which removes them. DINOv2-B < DINOv2-S is consistent with a
1536-d feature overfitting 23 training samples. (Tested in H6 and the Part 4 controls.)

## H2: heads on frozen features (before C5)

**Hypothesis.** Ridge / weighted kNN on the same frozen features are within ±3 EMD of the MLP-EMD head, and kNN
is worse on fine and sand soils.

**Result: partly not supported.** Ridge / kNN are within ±3 EMD of the MLP head for 6 of 7 backbones
(exception: DINOv2-B, kNN 41.36 vs MLP 45.54). kNN is *not* worse on fine/sand: for the DINO backbones it is
better on fine (51–54 vs 57–62 for ridge). The best single sklearn run is MAE-B + ridge, 37.81 ± 27.05, but
all 16 runs lie in 37.8–44.0.
**Why.** kNN predicts a convex combination of training curves, which is a strong shrinkage prior when only 23
samples are available. Choosing the best of ~30 configurations by CV on 24 samples is optimistic (winner's
curse), so 37.8 is a biased estimate.

## H3: fine-tuning vs frozen (before C8)

**Hypothesis.** Fine-tuning a ViT on 24 samples does not beat its frozen features (as with ConvNeXt: FT 41.7 vs
frozen 39–41), because of overfitting.

**Result: supported within noise.**

| encoder | frozen MLP | frozen ridge | fine-tuned (LLRD 0.75) |
|---|---|---|---|
| ViT-S (sup.) | 40.52 | 39.87 | 39.84 ± 33.90 |
| DINOv2-S | 42.54 | 42.15 | 39.30 ± 35.52 |
| ConvNeXt-T | 41.29 | 39.26 | 41.71 ± 33.35 |

FT is equal to the best frozen head for ViT-S and ConvNeXt, and about 3 EMD better for DINOv2-S. That gain is
not significant (the paired CI vs EfficientNet frozen is −4.1 … +4.7), and the std grows after fine-tuning
(35.5 vs 30–31), which means FT helps easy samples and hurts hard ones. CKA (C15) shows how much fine-tuning
changes the encoder: on train tiles the final-layer CKA to the pretrained model drops to 0.58–0.59, but on
test tiles it stays at 0.80–0.91. The network re-wired itself for the training domain only.

## H4: ViT from scratch (before C13)

**Hypothesis.** A ViT-S with random initialisation trained for 10 epochs on ~2.1k tiles from 23 samples learns
nothing beyond the mean curve: CV ≈ 80–88 and much worse than any pretrained model (37–45). It is refuted if
CV < 60.

**Result: refuted.** timm ViT-S/16 from scratch 37.73 ± 32.01. Our own ViT-S/16 (Lecture-1 implementation,
verified equal to timm with the same weights) reached **37.40 ± 29.28**, which is better than every pretrained
ViT and inside the noise of the best model overall (36.57). Scratch ViTs are best on sand (36–37 vs 42–52) and
slightly worse on gravel (32–33 vs 27–31).
**Why the prediction was wrong.** The prediction assumed the model must learn grain *shapes*, which needs either
a lot of data or pretrained features. The controls (H6, Part 4) show that the signal is global colour/texture
statistics. A randomly initialised ViT with a curve head initialised at the mean curve, EMD loss, weight decay
0.05 and drop-path 0.1 can fit a few global statistics in 10 epochs, and that is all the task rewards on 24
samples. ImageNet pretraining mostly adds object-level features that this task does not use. So "pretraining
matters more for ViTs than for CNNs" holds for tasks that need spatial structure, and this data set barely does.

## H5: colour normalisation and domain shift (before C12)

**Hypothesis.** The train/test difference is mainly colour (white balance, contrast), not sharpness. Gray-world
or Reinhard normalisation per photo lowers the test→train nearest-neighbour distance in feature space, and CV
does not get worse by more than the noise (~2 points). It is refuted if the distance does not change or CV drops
clearly (colour carries signal).

**Result: refuted.** The test/train NN-distance ratio (DINOv2-S features) goes from 3.28 to 2.91 with gray-world
and stays at 3.28 with Reinhard, so the shift is not mainly colour. CV gets worse: ViT-S ridge 39.87 → 41.83 (gw) /
42.58 (Reinhard), and DINOv2-S ridge is unchanged (42.15 → 41.71 / 41.41). Colour carries label signal (consistent
with the grayscale control in Part 4: 39.87 → 42.78). Test photos come from two iPhones that do not appear in
train, at 14–20 px/mm vs 4.6 px/mm for most train photos. After resampling to 4 px/mm they differ in
sharpness/noise (Laplacian variance 3.5–4.2k vs 2.2k/7.2k for train devices), so the shift is in the
acquisition, not the colour.

## H6: shortcut control, handcrafted colour / texture (before C14)

**Hypothesis.** Ridge and kNN on ~22 handcrafted features (RGB percentiles, HSV, gradient energy, Laplacian
variance at 3 scales) reach CV ≤ 45, close to the pretrained backbones. That would mean most of the signal is
global colour/texture statistics and not learned grain shape. It is refuted if CV > 60.

**Result: supported.** Handcrafted ridge **38.82 ± 22.81**, kNN 39.61 ± 26.93. This is better than 29 of the 32
deep-feature runs compared in C10, and it has the lowest std of all 24-fold runs. The Part 4 controls on frozen ViT-S + ridge (baseline 39.87) agree:

| control | CV | conclusion |
|---|---|---|
| shuffle 32-px patches (layout destroyed) | 39.73 | spatial layout above 8 mm is not used |
| blur r = 2 | 42.12 | fine detail helps a little (fine soils 51 → 58) |
| grayscale | 42.78 | colour helps (+2.9, mostly fine and sand) |
| shuffle all pixels (colour histogram only) | 43.41 | the colour histogram alone gives most of the signal |
| metadata only (scale, device, tile position) | 86.69 / 88.52 | = mean curve: no scale/device shortcut |

## H7: continued DINO on tiles, scale-safe vs resize (before C16/C17)

**Hypothesis.** DINO adaptation of DINOv2-S on unlabelled train tiles with a *scale-safe* multi-crop (crop
without resize, mm/px preserved) gives ridge/kNN CV ≤ the original (42.15 / 40.08) and lowers the test/train
NN-distance ratio (3.28). Resize crops (standard RandomResizedCrop, which changes mm/px) do not help or hurt.
It is refuted if scale-safe ≥ original or there is no difference between safe and resize.
*Limitation stated in advance:* SSL sees the unlabelled tiles of all 24 train samples, including the held-out
sample of each LOSO fold. This is transductive (no labels) but a mild leak.

**Result: mostly refuted.**

| encoder | ridge | kNN | test/train NN ratio | final teacher entropy (max 7.62) |
|---|---|---|---|---|
| DINOv2-S original | 42.15 | 40.08 | 3.28 | |
| + continued DINO, scale-safe | 41.76 | 40.76 | 5.24 | 2.65 (eff. prototypes 1541 → 73) |
| + continued DINO, resize | 43.57 | 42.73 | 5.23 | 2.38 (→ 55) |
| + scale-safe + anchor to original | 41.95 | 40.83 | 5.73 | 6.52 (→ 1352) |

Scale-safe adaptation does not help (±0.7, noise), and the resize variant is worse than scale-safe by ~2 in
both heads, which is the only part that matches the prediction. The NN-distance ratio *increased* to 5.2–5.7,
so adaptation pulled the encoder towards the training devices and away from the test photos. The teacher
entropy and the effective number of prototypes show partial collapse in both unanchored runs (546 tiles,
60 epochs). The anchored run did not collapse, but it also learned almost nothing (CKA to the original 0.93–0.99).
Fine-tuning the adapted encoders on the 8 ablation folds (sub_dinov2s_cdino_safe_ft 37.93, _lora 37.42) equals
fine-tuning the original (37.77 / 37.56). Cost: 9–10 GPU-min per adaptation run on a T4.

---

## Part 1 (CNN): observations, *post-hoc* (no hypothesis was written before these runs)

ResNet-18 ablations, 8 ablation folds, baseline AdamW + frozen BN = 34.74:

* BN statistics frozen vs updated: 34.74 vs 34.64. No difference with random 64-tile batches.
* Batches built from only 2 physical samples (`group_batches=2`): 32.19 (frozen BN) / 33.61 (updated BN).
  This was expected to hurt, because BN statistics then describe 2 samples. It did not hurt, and freezing BN
  kept it 1.4 better than updating.
* GroupNorm instead of BN: 35.10 (no gain, 1.8× memory).
* SGD (lr 1e-2 / 1e-3): 33.03; no warmup: 35.35; no augmentation: 34.61; 384-px tiles (96 mm): 36.49.

All differences are 0.1–2.6 EMD on 8 folds with std ≈ 22, so none of them is distinguishable from noise.
