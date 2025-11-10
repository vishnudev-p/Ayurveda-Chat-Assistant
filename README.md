#  Ayurveda Chat Assistant – Pure Knowledge Meets AI Wisdom 🌿

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green)
![MongoDB](https://img.shields.io/badge/MongoDB-Vector_Storage-brightgreen)
![RAG](https://img.shields.io/badge/RAG-Powered-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> A **Retrieval-Augmented Generation (RAG)** powered **Ayurveda Chat Assistant** that answers only **Ayurveda-related questions**, sourced exclusively from verified Ayurvedic texts.  
> No random data. No hallucinations. Just **authentic Ayurvedic wisdom**, intelligently delivered.

---

## 🌿 Overview

The **Ayurveda Chat Assistant** bridges the gap between **ancient Ayurvedic wisdom** and **modern AI technology**.  
It uses **RAG architecture** to retrieve relevant information, rerank responses, and generate insightful answers — all derived solely from an internal Ayurveda dataset stored in **MongoDB Vector Storage**.

🧠 Built with:
- **Sentence Transformers** for embeddings  
- **CrossEncoder** for reranking  
- **Gemma** and **Mistral** for contextual generation  
- **FastAPI** for backend  
- **MongoDB** for secure and efficient vector storage  

---

## 🎥 Demo Interface

Here’s a quick look at the Ayurveda Chat Assistant in action 👇  

<video src="assets/ayurveda-chat-demo.mp4" controls="controls" width="100%" height="auto"></video>

*(The video above will play directly in GitHub UI. If it doesn’t, ensure the `.mp4` file is committed to `/assets/` folder in your repository.)*


---

## 🌟 Key Features

✅ **Domain-Specific Knowledge** — Answers only Ayurveda questions  
✅ **RAG-Based Retrieval** — Uses hybrid (BM25 + Embedding) search  
✅ **MongoDB Vector Storage** — Efficient storage and retrieval of embeddings  
✅ **CrossEncoder Reranking** — Improves relevance and accuracy  
✅ **Gemma/Mistral for Generation** — Produces well-structured Ayurvedic responses  
✅ **FastAPI + Ngrok** — Lightweight API-based architecture  
✅ **Zero Hallucination** — Uses only verified internal Ayurvedic content  

---

## ⚙️ System Architecture

```text
 ┌──────────────────────────┐
 │        User Query        │
 └─────────────┬────────────┘
               │
               ▼
 ┌──────────────────────────┐
 │   SentenceTransformer    │  → Embedding model
 └─────────────┬────────────┘
               │
               ▼
 ┌──────────────────────────┐
 │   BM25 + Vector Search   │  → Hybrid retrieval
 └─────────────┬────────────┘
               │
               ▼
 ┌──────────────────────────┐
 │   CrossEncoder Reranker  │  → Rank top chunks
 └─────────────┬────────────┘
               │
               ▼
 ┌──────────────────────────┐
 │   Generator (Gemma/Mistral) │ → Generate Ayurvedic answer
 └──────────────────────────┘
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
