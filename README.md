# TurkishMedQA-LLM: Specialized Medical Question-Answering Model

## Overview

TurkishMedQA-LLM is a specialized medical question-answering system built for the Turkish language. This project addresses the critical gap in Turkish clinical NLP resources by developing a domain-adapted large language model capable of providing accurate and contextually appropriate medical responses.

## 🎯 Key Features

- **Large-scale Turkish Medical Dataset**: 47,000+ high-quality doctor-patient Q&A pairs across 25 medical specialties
- **Multi-agent Quality Control**: AI-powered filtering pipeline ensuring clinical accuracy and linguistic quality
- **Domain-adapted Fine-tuning**: Instruction-tuned model using parameter-efficient techniques (LoRA)
- **Multi-specialty Coverage**: Cardiology, neurology, dermatology, endocrinology, otolaryngology, and more
- **Context-aware Responses**: Chat-style evaluation with doctor-specific conditioning

## 📊 Dataset Statistics

| Metric | Value |
|--------|-------|
| Total QA Pairs | ~47,000 |
| Medical Specialties | 25 |
| Average Question Length | 45.4 tokens |
| Average Answer Length | 64.1 tokens |
| Train/Val/Test Split | 70%/15%/15% |

## 🏗️ Architecture

### Data Collection & Processing
1. **Web Scraping**: Ethical data collection from Turkish medical Q&A platforms
2. **Multi-agent Filtering**: Three-stage LLM-based quality control pipeline
   - Agent 1: Relevance filtering
   - Agent 2: Quality scoring (0-10 scale)
   - Agent 3: Fine-tuning appropriateness assessment
3. **Preprocessing**: HTML cleaning, deduplication, metadata extraction

### Model Development
- **Base Model**: `malhajar/Mistral-7B-v0.2-meditron-turkish`
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Training**: 8 epochs on NVIDIA H100 GPU
- **Prompt Format**: Structured "Soru/Yanıt" format

## 🚀 Getting Started

### Prerequisites
```bash
pip install transformers
pip install peft
pip install torch
pip install datasets
```

### Quick Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    "malhajar/Mistral-7B-v0.2-meditron-turkish"
)
tokenizer = AutoTokenizer.from_pretrained(
    "malhajar/Mistral-7B-v0.2-meditron-turkish"
)

# Load fine-tuned adapter (replace with your model path)
model = PeftModel.from_pretrained(base_model, "path/to/fine-tuned-model")

# Generate response
prompt = "Soru: Baş ağrısı neden olur?\nYanıt:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, temperature=0.7, max_length=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## 📈 Performance

### Multiple-Choice QA Benchmark
- **Base Model**: 56.32% accuracy
- **Fine-tuned Model**: 62.96% accuracy
- **Improvement**: +6.64 percentage points

### Evaluation Metrics
- Multiple-choice accuracy on 500-question benchmark
- BLEU and ROUGE scores (limited utility observed)
- Qualitative chat-style evaluation with doctor conditioning

## 🔧 Training Details

### Hyperparameters
- **Epochs**: 8
- **Temperature**: 0.7 (inference)
- **Sequence Length**: 512 tokens
- **GPU**: NVIDIA H100 (80GB)
- **Method**: LoRA with targeted projection layers

### Data Filtering Criteria
- Relevance score ≥ 7/10
- Clinical accuracy validation
- Linguistic fluency assessment
- Fine-tuning appropriateness check

## 🎯 Use Cases

- **Healthcare Professionals**: Quick reference and decision support
- **Medical Students**: Learning and study assistance
- **General Public**: Basic medical information and guidance
- **Telemedicine**: Automated first-line response systems

## ⚠️ Important Disclaimers

- This model is for **educational and research purposes only**
- **Not a substitute for professional medical advice**
- Always consult qualified healthcare providers for medical decisions
- Responses should be verified by medical professionals

---

**⭐ If you find this project useful, please consider giving it a star!**
