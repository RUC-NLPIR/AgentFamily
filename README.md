<div align="center">

# ✨ RUC-NLPIR Agent Family

<h4>Building General, Scalable, Powerful, and Safe Super Agents</h4>

[![GitHub Stars](https://img.shields.io/github/stars/RUC-NLPIR/ARPO?style=social)](https://github.com/RUC-NLPIR/ARPO)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2507.19849)
[![HuggingFace](https://img.shields.io/badge/🤗-Models%20%26%20Datasets-yellow)](https://huggingface.co/collections/dongguanting/arpo-688229ff8a6143fe5b4ad8ae)

<p align="center">
  <a href="#-latest-news">News</a> •
  <a href="#-agent-family">Agent Family</a> •
  <a href="#-dataset">Dataset</a> •
  <a href="#-model-zoo">Model Zoo</a> •
  <a href="#-citation">Citation</a>
</p>

</div>

---

## 🎯 Overview

Welcome to the **RUC-NLPIR Agent Family**! Our mission is to develop general-purpose, scalable, powerful, and secure intelligent agents. This repository encompasses:

- 🔍 **Deep Search & Research Agents**: Advanced information retrieval and synthesis
- 🛠️ **Multi-Tool Reasoning Agents**: Autonomous tool discovery and execution
- 🚀 **Agentic Reinforcement Learning**: State-of-the-art RL algorithms for agent training
- 📊 **Comprehensive Benchmarks**: Evaluation datasets and protocols

> [!TIP]
> ⭐ **Star us on GitHub** to stay updated with the latest releases and improvements!

---

## 📣 Latest News

- **[Oct 14, 2025]** 🚀 **AEPO Released!** Entropy-balanced agentic RL algorithm with superior performance on GAIA, HLE, and AIME. [[Code]](https://github.com/RUC-NLPIR/ARPO/tree/main/AEPO) [[Models]](https://huggingface.co/collections/dongguanting/aepo-68ef6832c99697ee03d5e1c7)

- **[Aug 11, 2025]** 📢 ARPO featured on multiple platforms: [X](https://x.com/kakakbibibi/status/1950211490943832393) | [WeChat](https://mp.weixin.qq.com/s/mFNRs-bHCAAe3x4QZHF8aA) | [Zhihu](https://zhuanlan.zhihu.com/p/1938022709545141501) | [YouTube](https://www.youtube.com/watch?v=FOK2tRtq7TE) | [Xiaohongshu](https://www.xiaohongshu.com/explore/68885b6b000000002501bb5e)

- **[July 29, 2025]** 🔥 ARPO honored as 🤗 HuggingFace **Daily Paper #1** and **Weekly Paper #1**! [[Paper]](https://huggingface.co/papers/2507.19849)

- **[July 29, 2025]** 📄 ARPO paper now available on [arXiv](https://arxiv.org/abs/2507.19849) and [Hugging Face](https://huggingface.co/papers/2507.19849)

- **[July 25, 2025]** 🎉 Full release: ARPO model checkpoints (3B~14B), datasets (SFT, RL, Evaluation), and complete codebase! [[🤗 Collection]](https://huggingface.co/collections/dongguanting/arpo-688229ff8a6143fe5b4ad8ae)

- **[July 25, 2025]** ⚡ Major optimization: Qwen3-14B training with batch size 128 takes only **10 minutes/step** with dynamic cache mechanism!

---

## 🔥 Agent Family

### 🤖 Agentic RL Series

<table>
<tr>
<td width="50%">

**[AEPO: Agentic Entropy-Balanced Policy Optimization](https://arxiv.org/abs/2510.14545)**

An advanced agentic RL algorithm that balances entropy in both rollout and policy update phases, achieving superior stability and performance.

[![GitHub](https://img.shields.io/badge/-Code-black?logo=github)](https://github.com/RUC-NLPIR/ARPO)
[![arXiv](https://img.shields.io/badge/arXiv-2510.14545-b31b1b.svg)](https://arxiv.org/abs/2510.14545)
[![Stars](https://img.shields.io/github/stars/RUC-NLPIR/ARPO?style=social)](https://github.com/RUC-NLPIR/ARPO)

</td>
<td width="50%">

**[ARPO: Agentic Reinforced Policy Optimization](https://arxiv.org/abs/2507.19849)**

Pioneering agentic RL with entropy-driven adaptive branching during high-entropy tool-call rounds for enhanced exploration.

[![GitHub](https://img.shields.io/badge/-Code-black?logo=github)](https://github.com/RUC-NLPIR/ARPO)
[![arXiv](https://img.shields.io/badge/arXiv-2507.19849-b31b1b.svg)](https://arxiv.org/abs/2507.19849)
[![Stars](https://img.shields.io/github/stars/RUC-NLPIR/ARPO?style=social)](https://github.com/RUC-NLPIR/ARPO)

</td>
</tr>
</table>

### 🔍 Deep Research Agent Series

<table>
<tr>
<td width="50%">

**[DeepAgent: General Reasoning with Scalable Toolsets](https://arxiv.org/abs/2510.21618)**

End-to-end deep reasoning agent featuring autonomous thinking, tool discovery, and brain-inspired memory folding mechanism.

[![GitHub](https://img.shields.io/badge/-Code-black?logo=github)](https://github.com/RUC-NLPIR/DeepAgent)
[![arXiv](https://img.shields.io/badge/arXiv-2510.21618-b31b1b.svg)](https://arxiv.org/abs/2510.21618)
[![Stars](https://img.shields.io/github/stars/RUC-NLPIR/DeepAgent?style=social)](https://github.com/RUC-NLPIR/DeepAgent)

</td>
<td width="50%">

**[HiRA: Hierarchical Reasoning Framework](https://arxiv.org/abs/2507.02652)**

Decoupled planning and execution framework with strategic planning and domain-specific execution by specialized agents.

[![GitHub](https://img.shields.io/badge/-Code-black?logo=github)](https://github.com/RUC-NLPIR/HiRA)
[![arXiv](https://img.shields.io/badge/arXiv-2507.02652-b31b1b.svg)](https://arxiv.org/abs/2507.02652)
[![Stars](https://img.shields.io/github/stars/RUC-NLPIR/HiRA?style=social)](https://github.com/RUC-NLPIR/HiRA)

</td>
</tr>
<tr>
<td width="50%">

**[WebThinker: Deep Research Capability](https://arxiv.org/abs/2504.21776)** 
*NeurIPS 2025*

Deep research agent empowering LRMs with autonomous search, web browsing, and research report drafting.

[![GitHub](https://img.shields.io/badge/-Code-black?logo=github)](https://github.com/RUC-NLPIR/WebThinker)
[![arXiv](https://img.shields.io/badge/arXiv-2504.21776-b31b1b.svg)](https://arxiv.org/abs/2504.21776)
[![Stars](https://img.shields.io/github/stars/RUC-NLPIR/WebThinker?style=social)](https://github.com/RUC-NLPIR/WebThinker)

</td>
<td width="50%">

**[Search-o1: Agentic Search-Enhanced LRMs](https://arxiv.org/abs/2501.05366)**
*EMNLP 2025*

Integrates autonomous knowledge retrieval with large reasoning models through Agentic RAG and reasoning-in-documents.

[![GitHub](https://img.shields.io/badge/-Code-black?logo=github)](https://github.com/RUC-NLPIR/Search-o1)
[![arXiv](https://img.shields.io/badge/arXiv-2501.05366-b31b1b.svg)](https://arxiv.org/abs/2501.05366)
[![Stars](https://img.shields.io/github/stars/RUC-NLPIR/Search-o1?style=social)](https://github.com/RUC-NLPIR/Search-o1)

</td>
</tr>
</table>

### 🛠️ Multi-Tool Reasoning

<table>
<tr>
<td>

**[Tool-Star: Multi-Tool Reasoner via RL](https://arxiv.org/abs/2505.16410)**

End-to-end framework empowering LLMs to autonomously interact with multi-tool environments through Self-Critic RL.

[![GitHub](https://img.shields.io/badge/-Code-black?logo=github)](https://github.com/RUC-NLPIR/Tool-Star)
[![arXiv](https://img.shields.io/badge/arXiv-2505.16410-b31b1b.svg)](https://arxiv.org/abs/2505.16410)
[![Stars](https://img.shields.io/github/stars/RUC-NLPIR/Tool-Star?style=social)](https://github.com/RUC-NLPIR/Tool-Star)

</td>
</tr>
</table>

---

## 📦 Dataset

High-quality datasets for agentic training and evaluation:

| Dataset | Size | Type | Download |
|---------|------|------|----------|
| Open-AgentRL-SFT | 3K | Supervised Fine-tuning | [🤗 HuggingFace](https://huggingface.co/datasets/Gen-Verse/Open-AgentRL-SFT-3K) |
| Open-AgentRL | 30K | Reinforcement Learning | [🤗 HuggingFace](https://huggingface.co/datasets/Gen-Verse/Open-AgentRL-30K) |

---

## 🤖 Model Zoo

Pre-trained and fine-tuned models ready for deployment:

| Model | Base | Type | Download |
|-------|------|------|----------|
| Qwen2.5-7B-RA-SFT | Qwen2.5-7B | SFT | [🤗 HuggingFace](https://huggingface.co/Gen-Verse/Qwen2.5-7B-RA-SFT) |
| Qwen3-4B-RA-SFT | Qwen3-4B | SFT | [🤗 HuggingFace](https://huggingface.co/Gen-Verse/Qwen3-4B-RA-SFT) |
| DemyAgent-4B | Qwen3-4B | Agent | [🤗 HuggingFace](https://huggingface.co/Gen-Verse/DemyAgent-4B) |

> [!NOTE]
> More model checkpoints (3B~14B) available in our [🤗 ARPO Collection](https://huggingface.co/collections/dongguanting/arpo-688229ff8a6143fe5b4ad8ae)

---

## 📄 Citation

If you find our work helpful, please cite:

```bibtex
@article{dong2025arpo,
  title     = {Agentic Reinforced Policy Optimization},
  author    = {Dong, Guanting and Mao, Hangyu and Ma, Kai and others},
  journal   = {arXiv preprint arXiv:2507.19849},
  year      = {2025}
}

@article{dong2025toolstar,
  title     = {Tool-Star: Empowering LLM-Brained Multi-Tool Reasoner via Reinforcement Learning},
  author    = {Dong, Guanting and Chen, Yifei and Li, Xiaoxi and others},
  journal   = {arXiv preprint arXiv:2505.16410},
  year      = {2025}
}
```

<details>
<summary>More citations</summary>

```bibtex
@article{dong2025aepo,
  title     = {Agentic Entropy-Balanced Policy Optimization},
  author    = {Dong, Guanting and others},
  journal   = {arXiv preprint arXiv:2510.14545},
  year      = {2025}
}
```

</details>

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

## 📞 Contact

For questions or collaboration:
- 📧 Email: [dou@ruc.edu.cn](mailto:dou@ruc.edu.cn)
- 🐦 Twitter: Follow our updates on [X](https://x.com/kakakbibibi)

---

## ⭐ Star History

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=RUC-NLPIR/ARPO&type=Date)](https://star-history.com/#RUC-NLPIR/ARPO&Date)

</div>

---

<div align="center">

**[⬆ Back to Top](#-ruc-nlpir-agent-family)**

Made with ❤️ by RUC-NLPIR Lab

</div>
