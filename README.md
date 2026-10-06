<h1 align="center">Yaroslav Sergaev</h1>
<p align="center">
<strong>ML Engineer / Data Scientist</strong> · LLM &amp; VLM · RL · VLA · AI Agents · Speech · CV
</p>

<p align="center">
<a href="mailto:yaroslav.sergaev@gmail.com"><img src="https://img.shields.io/badge/Email-yaroslav.sergaev-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Email"></a>
<a href="https://t.me/tohubohoo"><img src="https://img.shields.io/badge/Telegram-tohubohoo-26A5E4?style=for-the-badge&logo=telegram&logoColor=white" alt="Telegram"></a>
<a href="https://github.com/adelardw"><img src="https://img.shields.io/badge/GitHub-adelardw-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a>
</p>

---

### About

I build LLM agents and search systems in production, train speech and vision models end to end, and do research on speculative decoding and multimodal models. **Interested in RL, MLLM (LLM / VLM / VLA) and AI agents.**

Currently at **Sber Business Soft**, previously at **MTS Exolve** and **YADRO**.  
PhD student at **ISP RAS** (multimodal language models) · MSc in Machine Learning & Data Analysis, **HSE University** · BSc in Theoretical Physics, **UNN**

---

### Experience

| Company | Role | Key results |
|---------|------|-------------|
| **Sber Business Soft** | ML Engineer (NLP, LLM) | LLM agent in the SberBusiness Online web and mobile apps: ~3K WAU, 9 major releases, latency SLA under 30 s<br>Agent harness: API tools, Schema-Guided Reasoning, skills and a memory sub-agent; new tools plug in via YAML/JSON configs<br>Hybrid search with LLM ranking: hit@1 0.84 → 0.94, MRR 0.88 → 0.96<br>Agent evaluation with LLM-as-a-Judge and Agent-as-a-Judge benchmarks |
| **MTS Exolve** | ML Engineer (DL, NLP, Audio) | Whisper-based ASR for phone calls, trained end to end with QAT (FP8): WER/CER ~10-20%, replaced an external vendor, saving ~5M RUB/month<br>LLM inference throughput 4× (1 → 4 RPS under a 100 RPS load) with DeepSpeed<br>Llama-3-8B fine-tuned with LoRA (SFT and rejection sampling) as a website assistant, plus RAG on Qdrant |
| **YADRO** | ML Engineer (DL, CV) | Text detection (TextSnake, DBNet) for edge devices: ResNet-50 backbone replaced with MobileNetV3-Small, ~25× fewer backbone parameters<br>2D barcode detection, image classification on MobileNetV3, zero-shot classifier on MobileCLIP<br>ONNX / TFLite conversion with 90%+ of the original quality retained |

---

### Research

**[FlowDraft](https://github.com/adelardw/FlowDraft)** · paper accepted to SMILES School Projects Proceedings (SSPP) 2026, Skoltech Applied AI Center  
Lossless speculative decoding: a diffusion (flow-map) drafter embedded in a frozen LLM and trained on its own refinement chain. On Qwen3-0.6B it beats a reproduced Orthrus baseline: 2.58 vs 2.20 accepted tokens per cycle, 1.44× vs 1.35× speedup.

**[mdfr-rppgfau](https://github.com/adelardw/mdfr-rppgfau)** · co-author of a paper that passed Phase 1 review at AAAI 2027  
Multimodal deepfake detection: FAU (Swin-T + ME-GraphAU) and rPPG (DeepFakesON-Phys) branches fused via a Q-Former, trained on a mix of FF++, Celeb-DF and VCDF-X. Found that near-perfect test metrics were inflated (34% of test clips were augmented copies of training clips, and the model could memorize actors), then added MTCNN face crops, contrastive learning with a memory bank and identity-disjoint splits. On 53 held-out videos from unseen generators: 0.815 accuracy, 0.844 AUROC.

**[AudioDenoisingNet](https://github.com/adelardw/AudioDenoisingNet)** · MSc thesis, HSE University  
Compact speech denoising: a U-Net (1.85M parameters) on STFT spectrograms with a phase-correction head; processes 8 s of audio in ~1.2 s on a laptop CPU. Fixed an ISTFT edge artifact without retraining, raising SI-SDR on VoiceBank+DEMAND from 5.2 to 9.5 dB.

---

### Projects

| Project | Description |
|---------|-------------|
| [SelfExtensionAgent](https://github.com/adelardw/SelfExtensionAgent) | Self-extending agent harness on LangGraph: when a tool is missing, the agent writes it in Python, checks it (static analysis, LLM review, sandboxed smoke test) and registers it. 70+ tools, long-term memory, MCP, human-in-the-loop approvals; works with any OpenAI-compatible model, including local ones via Ollama |
| [MobileClipClassifier](https://github.com/adelardw/MobileClipClassifier-0-shot) | Zero-shot image tagging with MobileCLIP on TFLite: tag embeddings are stored in a JSON codebook and each image goes to the tag with the highest cosine similarity, so new tags need no retraining |

---

### Tech Stack

**LLM & Agents**

![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![MCP](https://img.shields.io/badge/MCP-000000?style=for-the-badge&logo=modelcontextprotocol&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white)

**Deep Learning**

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Lightning](https://img.shields.io/badge/Lightning-792EE5?style=for-the-badge&logo=pytorchlightning&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Transformers](https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![PEFT](https://img.shields.io/badge/PEFT-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![TRL](https://img.shields.io/badge/TRL-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)


**Inference & Optimization**

![DeepSpeed](https://img.shields.io/badge/DeepSpeed-0078D4?style=for-the-badge&logo=microsoft&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white)
![TFLite](https://img.shields.io/badge/TFLite-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![AutoAWQ](https://img.shields.io/badge/AutoAWQ-2C2C2C?style=for-the-badge)
![TorchAO](https://img.shields.io/badge/TorchAO-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

**Backend & MLOps**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Airflow](https://img.shields.io/badge/Airflow-017CEE?style=for-the-badge&logo=apacheairflow&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-0C4B8E?style=for-the-badge)
![Jenkins](https://img.shields.io/badge/Jenkins-D24939?style=for-the-badge&logo=jenkins&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![Bash](https://img.shields.io/badge/Bash-4EAA25?style=for-the-badge&logo=gnubash&logoColor=white)

**Data & Search**

![PySpark](https://img.shields.io/badge/Spark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-DC244C?style=for-the-badge)
![FAISS](https://img.shields.io/badge/FAISS-0467DF?style=for-the-badge&logo=meta&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=redis&logoColor=white)

---

### Achievements

- **Papers:** FlowDraft accepted to SSPP 2026; co-author of a paper that passed Phase 1 review at AAAI 2027
- **Top-3** at the Gazprom ML Hackathon (digital-twin case)
- **SMILES 2026** research program, Skoltech Applied AI Center
- **YSDA** (Yandex School of Data Analysis): NLP, CV, ML
