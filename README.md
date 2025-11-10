# Interleave-VLA: Enhancing Robot Manipulation with Interleaved Image-Text Instructions

Official repository for Interleave‑VLA, the first vision‑language‑action (VLA) framework that understands interleaved image–text instructions and directly produces continuous actions in real‑world scenarios.

## Overview 🧭
![Overview](/assets/overview.png)

**Quick links**: 📄 [Paper](https://arxiv.org/abs/2505.02152) · 🌐 [Project Website](https://interleave-vla.github.io/Interleave-VLA-Anonymous/) · 📦 [Dataset](https://huggingface.co/collections/Interleave-VLA/interleave-vla-dataset-6866a10654b16d02032db7a1) · 💻 [Code](https://github.com/orgs/Interleave-VLA/repositories)

Interleave‑VLA is a flexible, model‑agnostic upgrade that extends state‑of‑the‑art VLA models with minimal changes and strong zero‑shot generalization, achieving up to 2× better out‑of‑domain generalization to unseen objects compared with text‑only VLA baselines.

## Get Started 🚀

Interleave‑VLA is built upon state‑of‑the‑art VLA models. We provide two implementations:
### Interleave‑π0

Train and evaluate:

 ✅ Documentation: [Interleave‑π0](/open-pi-zero/doc/interleave_pi0.md) — complete and ready to use.

 🤗 Checkpoint on HuggingFace: [Interleave‑π0 Checkpoint](https://huggingface.co/Interleave-VLA/interleave-pi0-bridge).

### Interleave‑OpenVLA
Guide coming soon:

 🛠️ Documentation: [Interleave‑OpenVLA](/openvla/doc/interleave_openvla.md) — in progress ⏳.

 📥 Checkpoint — coming soon ⏳.

## Roadmap 🗺️
- [x] Release Interleave‑π0 code
- [x] Release Interleave‑π0 documentation
- [x] Release Interleave‑π0 checkpoint
- [x] Release Interleave‑OpenVLA code
- [ ] Release Interleave‑OpenVLA documentation
- [ ] Release Interleave‑OpenVLA checkpoint

## Acknowledgements 🙏
This project builds upon the following works ❤️:

- [open-pi-zero](https://github.com/allenzren/open-pi-zero.git)
- [OpenVLA](https://github.com/openvla/openvla)
- [InternVL](https://github.com/OpenGVLab/InternVL)

## Support 💬
📧 Questions or feedback? Contact Cunxin Fan at <alfayoung2004@gmail.com> or open an issue.

## Citation 📚
If you find our work helpful, please consider citing our paper:
```
@misc{fan2025interleavevlaenhancingrobotmanipulation,
      title={Interleave-VLA: Enhancing Robot Manipulation with Interleaved Image-Text Instructions},
      author={Cunxin Fan and Xiaosong Jia and Yihang Sun and Yixiao Wang and Jianglan Wei and Ziyang Gong and Xiangyu Zhao and Masayoshi Tomizuka and Xue Yang and Junchi Yan and Mingyu Ding},
      year={2025},
      eprint={2505.02152},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2505.02152},
}
```

If you find this repository helpful, please give it a ⭐ — thanks! 🙌