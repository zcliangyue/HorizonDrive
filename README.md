# HorizonDrive: Self-Corrective Autoregressive World Model for Long-horizon Driving Simulation

**[Conglang Zhang](https://github.com/zcliangyue)<sup>1,\*</sup>, [Yifan Zhan](https://github.com/Yifever20002)<sup>2,\*</sup>,** Qingjie Wang<sup>3</sup>, Zhanpeng Ouyang<sup>3</sup>, Yu Li<sup>4</sup>, Zihao Yang<sup>5</sup>, Xiaoyang Guo<sup>6</sup>, Weiqiang Ren<sup>3</sup>, Qian Zhang<sup>3</sup>, Zhen Dong<sup>1</sup>, Yinqiang Zheng<sup>2</sup>, Wei Yin<sup>3,‡</sup>, Zhengqing Chen<sup>3,†</sup>

<sup>1</sup> [Wuhan University](https://www.whu.edu.cn/) &nbsp; <sup>2</sup> [The University of Tokyo](https://www.u-tokyo.ac.jp/en/) &nbsp; <sup>3</sup> [Horizon Robotics](https://en.horizon.auto/) &nbsp; <sup>4</sup> [Tsinghua University](https://www.tsinghua.edu.cn/en/) &nbsp; <sup>5</sup> [University of Science and Technology of China](https://en.ustc.edu.cn/) &nbsp; <sup>6</sup> [The Chinese University of Hong Kong](https://www.cuhk.edu.hk/english/)

\* Equal contribution &nbsp; ‡ Project lead &nbsp; † Corresponding author

<div align="center">
  <img src="assets/logo_WHU.png" height="60">&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/Horizon_Robotics.svg" height="60">
</div>

<br>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2605.11596-b31b1b.svg)](https://arxiv.org/abs/2605.11596)
[![Project Page](https://img.shields.io/badge/🌐_Project_Page-1a73e8)](https://zcliangyue.github.io/HorizonDrive/)
[![GitHub](https://img.shields.io/badge/GitHub-Code-181717?logo=github)](https://github.com/zcliangyue/HorizonDrive)

</div>

<!-- Project page: https://zcliangyue.github.io/HorizonDrive/ -->

---

## 📌 TODO

- [ ] Nuscenes Inference code
- [ ] DMD checkpoints

> Code coming soon — stay tuned!

---

## 🌍 Overview

**HorizonDrive** is an anti-drifting training-and-distillation framework for **real-time, long-horizon autoregressive driving simulation**. The goal is to make the **teacher rollout-capable** so that its own autoregressive rollouts remain stable and provide reliable long-horizon supervision at bounded memory.

---

## ✨ Key Features

* **Controllable autonomous video generation**
* **Long-horizon AR rollout**
* **Few-step inference by distillation**
* **Close-loop simulation**

---

## 🧪 Abstract

Closed-loop driving simulation requires real-time interaction beyond short offline clips, pushing current driving world models toward autoregressive (AR) rollout. Existing AR distillation approaches typically rely on frame sinks or student-side degradation training. The former transfers poorly to driving due to fast ego-motion and rapid scene changes, while the latter remains bounded by the teacher’s single-pass output length and thus provides only a limited supervision horizon. A natural question is: can the teacher itself be extended via AR rollout to provide unbounded-horizon supervision at bounded memory cost? The key difficulty is that a standard teacher drifts under its own predictions, contaminating the supervision it provides. **Our key insight is to make the teacher rollout-capable, ensuring reliable supervision from its own AR rollouts.** This is instantiated as HorizonDrive, an anti-drifting training-and-distillation framework for AR driving simulation. First, scheduled rollout recovery (SRR) trains the base model to reconstruct ground-truth future clips from prediction-corrupted histories, yielding a teacher that remains stable across long AR rollouts. Second, the rollout-capable teacher is extended via AR rollout, providing long-horizon distribution-matching supervision under bounded memory, while a short-window student aligns to it with teacher rollout DMD (TRD) for efficient real-time deployment. HorizonDrive natively supports minute-scale AR rollout under bounded memory; on nuScenes, HorizonDrive reduces FID by 52% and FVD by 37%, and lowers ARE and DTW by 21% and 9% relative to the strongest long-horizon streaming baselines, while remaining competitive with single-pass driving video generators.

---


## 📚 Citation

If you find our work useful, please cite it as

```bibtex
@misc{zhang2026horizondriveselfcorrectiveautoregressiveworld,
  title={HorizonDrive: Self-Corrective Autoregressive World Model for Long-horizon Driving Simulation},
  author={Zhang, Conglang and Zhan, Yifan and Wang, Qingjie and Ouyang, Zhanpeng and Li, Yu and Yang, Zihao and Guo, Xiaoyang and Ren, Weiqiang and Zhang, Qian and Dong, Zhen and Zheng, Yinqiang and Yin, Wei and Chen, Zhengqing},
  year={2026},
  eprint={2605.11596},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2605.11596},
}
```

---

## Acknowledgments

We gratefully acknowledge the open-source works [**CompoSIA**](https://github.com/Yifever20002/CompoSIA), [**Self-Forcing**](https://github.com/guandeh17/Self-Forcing) and [**LongLive**](https://github.com/NVlabs/LongLive).
