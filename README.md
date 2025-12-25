# My Self-Learning Journey of LLMs  
## 🔍 Retrieval-Augmented Generation (RAG) with Custom Documents

This project is a self-learning initiative focused on building an AI system that can **read real documents and answer questions grounded in factual sources**.  
It combines **Large Language Models (LLMs)** with a **retrieval system** to reduce hallucination and improve answer accuracy.

---

## ✨ Project Objectives

- Understand the architecture of **Retrieval-Augmented Generation (RAG)**
- Learn why **LLMs require retrieval and embeddings** for reliable answers
- Practice processing **real-world documents** (PDF / TXT)
- Develop a **production-oriented retrieval pipeline**
- Gain hands-on experience in **LLM fine-tuning and dataset preparation**

---

## 🧠 Learning Modules

The project is structured into progressive modules for step-by-step learning.

---

### 📌 Module 1 — Introduction to LLMs

- Load and experiment with LLMs from **Hugging Face** (e.g., Gemma, Flan-T5)
- Perform summarization, explanation, and question answering
- Understand key LLM limitations such as **hallucination**

---

### 📌 Module 2 — Embeddings & Vector Search

- Learn the concept of **text embeddings**
- Use **Sentence-Transformers** (`all-MiniLM-L6-v2`)
- Convert documents into vector representations
- Implement **semantic search** using **FAISS**
- Retrieve content based on meaning rather than keywords

---

### 📌 Module 3 — Full RAG Pipeline

Build a complete pipeline that retrieves relevant documents **before** generating answers.

#### 3.1 Input Normalization
- Clean and normalize user queries
- Improve retrieval quality through better query formulation

#### 3.2 Retrieval Layer Improvements

Enhance retrieval accuracy using production-grade techniques:

- **Auto Top-K**: Dynamically select the optimal number of retrieved chunks
- **Similarity Filtering**: Remove low-relevance results
- **Multi-Query Retrieval**: Generate multiple query variants to capture missing context
- **MMR Reranking**: Balance relevance and diversity to avoid redundant chunks

These techniques significantly reduce hallucination and improve response grounding.

#### 3.3 Prompt Builder & Context Citation *(In Progress)*

- Select the best document chunks as context
- Construct prompts that enforce **source-based answers**
- Enable citation-aware responses from the LLM

---

## 🧪 Sample Data

- **Sherlock Holmes** (English) — `.txt`
- **Thai documents** — `.pdf` (e.g., Wikipedia PDFs)

Documents can be easily replaced with custom datasets.

---

## 🧩 LLM Fine-Tuning & Dataset Preparation

This project also covers the fundamentals of **fine-tuning large language models**, including:

- Cleaning and preprocessing noisy text data
- Tokenizing and structuring datasets for training
- Splitting datasets into training and evaluation sets
- Fine-tuning pretrained models with appropriate hyperparameters
- Evaluating model performance and preparing for deployment

A structured workflow enables LLMs to be adapted for **task-specific NLP applications**.

---

## 📦 Dependencies

Install required libraries (e.g., on Google Colab):

```bash
pip install transformers accelerate sentencepiece huggingface_hub \
    sentence-transformers faiss-cpu PyPDF2 nltk

# 🧪 PEFT / LoRA / QLoRA / DLoRA Lab

This repository documents hands-on experiments with **Parameter-Efficient Fine-Tuning (PEFT)** techniques on Large Language Models (LLMs) under limited GPU resources (~15GB VRAM).

The focus is on **practical setup, real errors encountered, and lessons learned**, rather than idealized benchmarks.

---

## 🎯 Objectives

- Understand trade-offs between LoRA, QLoRA, and DLoRA
- Experiment with multi-adapter fine-tuning (DLoRA)
- Identify common pitfalls when training LLMs on Colab-scale GPUs
- Build a reusable lab setup for PEFT experiments

---

## 🧠 Techniques Explored

- **LoRA** – Low-Rank Adaptation
- **QLoRA** – 4-bit Quantization + LoRA
- **DLoRA** – Multiple task-specific LoRA adapters on a single base model

---

## 🤖 Models

- Base models:  
  - `microsoft/phi-2`  
  - `mistralai/Mistral-7B-v0.1`
- Frameworks & tools:
  - HuggingFace Transformers
  - PEFT
  - BitsAndBytes
  - Datasets

---

## 🔬 Experiments & Findings

### 1. Base Model Loading

- Loading 7B models can trigger **OOM during checkpoint shard loading**
- Peak VRAM usage often occurs **before training begins**

**Lesson learned**  
> Peak memory usage during model loading is as important as training-time memory.

---

### 2. QLoRA (4-bit Quantization)

Configuration example:
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

