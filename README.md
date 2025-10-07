# PicoDAC & PicoDAC-IT – Whitepaper

[![HuggingFace](https://img.shields.io/badge/HuggingFace-Mattimax-brightgreen)](https://huggingface.co/Mattimax)
[![M.INC](https://img.shields.io/badge/M.INC-Official-blue)](https://huggingface.co/MINC01)

Autore: [Mattia / Mattimax](https://huggingface.co/Mattimax)  
Sito aziendale: [M.INC](https://sites.google.com/view/mattimax-site/home-page)

---

## 1. Panoramica

PicoDAC è un **modello GPT-like ultraleggero** ottimizzato per dialoghi in italiano, progettato per essere:

* **Performante**: inferenza rapida anche su GPU consumer o CPU.
* **Compatto**: <10M parametri, memoria ridotta, sequenze corte (64 token).
* **Customizable**: tokenizer e pipeline addestrabile da zero, facile da fine-tunare.

La linea include:

* **PicoDAC**: modello base addestrato su [`Mattimax/Little_ITA_60k`](https://huggingface.co/Mattimax/Little_ITA_60k).
* **PicoDAC-IT**: versione *instruction-tuned* sul dataset [`ruslanmv/italian-dataset-mini`](https://huggingface.co/ruslanmv/italian-dataset-mini).

---

## 2. Architettura

### 2.1 Parametri globali

| Parametro                             | Valore     |
| ------------------------------------- | ---------- |
| Vocabolario                           | 1920 token |
| Dimensione embedding (`d_model`)      | 240        |
| Numero di layer (`n_layers`)          | 6          |
| Numero di attention heads (`n_heads`) | 6          |
| Head dimension (`d_head`)             | 40         |
| FFN hidden (`d_ff`)                   | 960        |
| Lunghezza contesto (`n_ctx`)          | 64         |
| Dropout                               | 0.0        |

**Parametri totali stimati**: ~4,6M FP32 (embedding, layer Transformer, RMSNorm finale)

---

### 2.2 Tokenizer & special tokens

* **BPE tokenizer** custom (1920 vocab)  
* **Special tokens**: `<PAD>, <BOS>, <EOS>, <SEP>, <UNK>`  
* **Template post-processing**:

```

single: <BOS> {testo} <EOS>
pair:   <BOS> {prompt} <SEP> {response} <EOS>

```

---

### 2.3 Blocco Transformer

Ogni layer:

```

Input -> RMSNorm -> CausalSelfAttention -> Residual
-> RMSNorm -> FeedForward -> Residual -> Output

```

* Multi-head (`n_heads=6`, `d_head=40`)  
* Causal mask (`N_CTX x N_CTX`)  
* FeedForward: `Linear(d_model,d_ff) → SiLU → Dropout → Linear(d_ff,d_model)`  
* RMSNorm: `x_norm = x * 1 / sqrt(mean(x^2)+eps) * weight`

---

### 2.4 Pipeline TinyGPT

```

Input tokens (B,T)
|
v
Token embedding + Positional embedding
|
v
Stack di 6 TransformerBlock
|
v
RMSNorm finale
|
v
Linear(d_model -> vocab_size)
|
v
Logits per token

```

* LM head weight sharing: `lm_head.weight = tok_emb.weight`  
* Output logits → next-token prediction (Causal LM)  
* Padding gestito tramite -100 label mask

---

### 2.5 Quantizzazione

* Matrici principali → int8  
* Embedding & RMSNorm → precisione ridotta (emulazione)  
* Salvataggio: `safetensors` + `scales.json`  
* Ottimizzato per inferenza CPU/GPU low-memory

---

### 2.6 Loss & Ottimizzazione

* CrossEntropy LM (`ignore_index=-100` per padding)  
* AdamW: decay su pesi normali, no decay su embeddings e RMSNorm  
* Cosine LR scheduler con warmup 500 step  
* Gradient clipping 1.0  
* Batch size 128, max_steps 30k

---

## 3. Pipeline di training

1. Costruzione tokenizer da dataset JSONL  
2. Creazione dataset e DataLoader  
3. Istanziazione modello TinyGPT  
4. Ottimizzatore + scheduler  
5. Loop di training: forward → loss → backward → optimizer step  
6. Log / checkpoint / validation ogni `save_interval`  
7. Salvataggio artifact:

```

tokenizer.json
config.json
best/model.safetensors
best/scales.json
train_stats.json

```

---

## 4. Diagramma dei dati (ASCII)

```

JSONL {"prompt": ..., "response": ...}
|
v
Tokenizer BPE + special tokens
|
v
Tensor shape (B, N_CTX)
|
v
TinyGPT (6 layer Transformer)
|
v
Logits (B, N_CTX, vocab_size)
|
v
CrossEntropyLoss → Backprop

````

---

## 5. Esempio di inferenza

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Mattimax/PicoDAC-IT")
model = AutoModelForCausalLM.from_pretrained("Mattimax/PicoDAC-IT", trust_remote_code=True)

prompt = "Ciao, come stai oggi?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
````

> `trust_remote_code=True` è necessario perché TinyGPT è modello custom non standard Transformers.

---

## 6. Sezione avanzata per sviluppatori

* Parametri RMSNorm custom e attivazioni SiLU
* Early-stop / checkpointing con copia del best quantizzato
* Gradient accumulation possibile
* Compatibilità CPU/GPU garantita

### Layer & Parametri Dettagliati

**Config generale**

| Parametro   | Valore |
| ----------- | ------ |
| Vocabolario | 1920   |
| d_model     | 240    |
| n_layers    | 6      |
| n_heads     | 6      |
| d_head      | 40     |
| d_ff        | 960    |
| n_ctx       | 64     |
| Dropout     | 0.0    |

**Parametri TransformerBlock (per blocco)**

| Componente  | Parametri |
| ----------- | --------- |
| QKV         | 172,800   |
| Output attn | 57,600    |
| FFN         | 460,800   |
| RMSNorm     | 480       |
| **Totale**  | 691,680   |

*6 blocchi → 4,150,080*
*RMSNorm finale: 240*
**Totale FP32 ≈ 4,626,480 (~4.6M)**

---

**Quantizzazione & FLOPS**

* Matrici principali → int8
* Embedding & RMSNorm → precisione ridotta (~50-75% risparmio)
* FLOPS per token ≈ 4.24M
* Risultato: inferenza rapida su GPU consumer, notebook o CPU

---

## 7. Risorse e link

* [PicoDAC](https://huggingface.co/Mattimax/PicoDAC)
* [PicoDAC-IT](https://huggingface.co/Mattimax/PicoDAC-IT)
* [M.INC HuggingFace](https://huggingface.co/MINC01)
* [Sito aziendale](https://sites.google.com/view/mattimax-site/home-page)
