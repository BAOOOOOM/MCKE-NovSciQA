# Learn to Understand: Knowledge Exemplification via Multi-Agent Cooperation for Science Question Answering

This repository provides the **code and dataset** for the paper:

> **Learn to Understanding: Knowledge Exemplification via Multi-Agent Cooperation for Science Question Answering**  

The paper proposes **MCKE (Multi-Agent Cooperation-based Knowledge Exemplification)**, a schema-inspired framework that constructs and adapts *exemplified knowledge* to guide large language models in science question answering.

---

## ⭐ Repository Highlights

- **Reference implementation of MCKE**  
  This repository implements the main components of the MCKE framework, including exemplified knowledge construction (Knowledge Exemplification Module), schema-aware adaptation (Exemplified Knowledge Adaptation Module), and question answering (Question Answering Module).  
  For ease of implementation and reproduction, the pipeline is decomposed into **six sequential steps (01–06)**. The correspondence between each step and its functionality is documented within the code.

- **NovSciQA dataset**  
  We release **NovSciQA**, a science question answering dataset built upon *novel, non-existent scientific knowledge*, designed to evaluate genuine knowledge understanding rather than memorization.

- **Reproducible experimental pipeline**  
  The code is organized as a step-by-step pipeline (01–06), closely following the methodology described in the paper.

---

## ⚠️ Notes on Usage

- Due to data licensing and size constraints, **some datasets need to be downloaded separately** and placed into the corresponding directories before running the code.
- After preparing the required data, the scripts in `MCKE/ScienceQA/` can be executed **sequentially (01 → 06)** to reproduce the main results.
- **Environment setup instructions and detailed configuration will be updated in future releases**.

---

## 📌 Citation

If you find this repository or the **NovSciQA dataset** useful for your research, please consider citing our paper:

```bibtex
@ARTICLE{11365952,
  author={Bao, Meikai and Zhang, Kai and Liu, Xukai and Liu, Qi and Zhao, Hongke and Chen, Enhong},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={Learn to Understand: Knowledge Exemplification via Multi-Agent Cooperation for Science Question Answering}, 
  year={2026},
  volume={},
  number={},
  pages={1-13},
  doi={10.1109/TKDE.2026.3658068}}
