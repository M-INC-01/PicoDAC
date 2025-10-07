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

**Parametri totali stimati**:

* Embedding: (1920 \times 240 \approx 460k)
* Positional embedding: (64 \times 240 \approx 15k)
* Attention + FFN (6 layer) ≈ 6 × (QKV + O + FFN weights) ≈ 6 × (3×240² + 240² + 240×960 + 960×240) ≈ 7.2M
* Totale: ~8M parametri (FP32)

---

### 2.2 Tokenizer & special tokens

* **BPE tokenizer** custom (1920 vocab)
* **Special tokens**:

  ```
  <PAD>, <BOS>, <EOS>, <SEP>, <UNK>
  ```
* **Post-processing template**:

  ```
  single: <BOS> {testo} <EOS>
  pair:   <BOS> {prompt} <SEP> {response} <EOS>
  ```

---

### 2.3 Blocco Transformer

Ogni layer è:

```
Input -> RMSNorm -> CausalSelfAttention -> Residual
      -> RMSNorm -> FeedForward -> Residual -> Output
```

**Dettaglio attention**:

* Multi-head con `n_heads=6`
* Head dim: `d_head = 40`
* Causal mask: `N_CTX x N_CTX`
* Softmax su score / sqrt(d_head)

**FeedForward**:

* Linear(d_model, d_ff) → SiLU → Dropout → Linear(d_ff, d_model)

**RMSNorm**:

```
x_norm = x * 1 / sqrt(mean(x^2) + eps) * weight
```

---

### 2.4 TinyGPT: pipeline completa

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

**Note**:

* LM head weight sharing: `lm_head.weight = tok_emb.weight`
* Output logits usati per next-token prediction (Causal LM)
* Gestione padding tramite -100 label mask

---

### 2.5 Quantizzazione

* **Matrici principali** → int8
* **Embedding e RMSNorm** → emulazione 18/24bit (int32 + scale)
* Salvataggio: `safetensors` + `scales.json`
* Ottimizzato per inference CPU/GPU low-memory

---

### 2.6 Loss & Ottimizzazione

* CrossEntropy LM (ignore_index=-100 per padding)
* AdamW con:

  * Decay per weight normali
  * No decay su embeddings e RMSNorm
* Cosine LR scheduler con warmup 500 step
* Gradient clipping 1.0
* Batch size 128, max_steps 30k

---

## 3. Pipeline di training

1. Costruzione tokenizer da dataset JSONL
2. Creazione dataset e DataLoader
3. Istanziazione modello TinyGPT
4. Ottimizzatore + scheduler
5. Loop di training:

   * Forward → loss → backward → optimizer step
   * Log / checkpoint / validation ogni `save_interval`
6. Salvataggio artifact:

   ```
   tokenizer.json
   config.json
   best/model.safetensors
   best/scales.json
   train_stats.json
   ```

---

## 4. Esempio diagramma ASCII dei dati

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
```

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
```

> Nota: `trust_remote_code=True` necessario perché TinyGPT è modello custom non riconosciuto da Transformers standard.

---

## 6. Dettagli avanzati per sviluppatori

* **Parametri custom RMSNorm** e attivazioni SiLU → ottimizzazione su precisione numerica
* **Supporto early-stop / checkpointing** con copia del best quantizzato
* **Gradient accumulation** possibile
* **Compatibilità CPU/GPU** garantita

---

# Layer & Parametri Dettagliati

---

## 1. Configurazione generale

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

**Calcolo rapido dei parametri**:

* Embedding token: `1920 × 240 = 460,800`
* Positional embedding: `64 × 240 = 15,360`
* RMSNorm (per layer, 2 per layer + finale): 240 × 3 = 720 × 6 = 4,320
* Totale embedding + norm ≈ 480k

---

## 2. Parametri TransformerBlock (per blocco)

### 2.1 Multi-Head Attention

* Matrice QKV: `d_model × 3*d_model = 240 × 720 = 172,800`
* Matrice output: `d_model × d_model = 240 × 240 = 57,600`
* **Totale attenzione per blocco**: 230,400

### 2.2 FeedForward

* Linear1: `d_model × d_ff = 240 × 960 = 230,400`
* Linear2: `d_ff × d_model = 960 × 240 = 230,400`
* **Totale FFN per blocco**: 460,800

### 2.3 RMSNorm

* 2 RMSNorm per blocco × 240 param = 480

**Parametri totali per blocco**:

```
Attention 230,400 + FFN 460,800 + RMSNorm 480 ≈ 691,680
```

**6 blocchi** → 4,150,080

---

## 3. Layer finale

* RMSNorm finale: 240 param
* LM head (weight sharing) → nessun parametro aggiuntivo

---

## 4. Parametri totali FP32

| Componente           | Parametri           |
| -------------------- | ------------------- |
| Embedding + pos      | 476,160             |
| 6 × TransformerBlock | 4,150,080           |
| RMSNorm finale       | 240                 |
| **Totale FP32**      | ≈ 4,626,480 (~4.6M) |

> Nota: nel codice stimavo ~8M perché consideravo duplicazioni e buffer temporanei, ma i pesi effettivi sono ~4.6M.

---

## 5. Stima FLOPS per un forward

**Assunzioni**:

* Matmul FP32: `2 * M * N * K` FLOPS
* Input batch size `B`, sequence length `T = n_ctx = 64`

### 5.1 Embedding

* `B × T × d_model` addizione pos embedding → FLOPS trascurabile

### 5.2 Multi-Head Attention (per layer)

* QKV matmul: `B*T*d_model × 3*d_model ≈ 3*B*T*240*240` → 3 × 64*B*240² ≈ 11,059,200 × B
* Attn matmul: `B*H*T*d_head × T ≈ B*6*64*40*64 ≈ 983,040 × B`
* Output linear: `B*T*d_model × d_model = 64*B*240*240 = 3,686,400 × B`

**Totale per attenzione per batch B** ≈ 15.7M × B FLOPS

### 5.3 FFN

* Linear1 + activation + Linear2: `B*T*(d_model*d_ff*2) ≈ 64*B*(240*960*2) ≈ 29,491,200 × B`

### 5.4 Totale per layer

```
Attention + FFN ≈ 15.7M + 29.5M ≈ 45.2M FLOPS × B
6 layer → 271.2M FLOPS × B
+ embedding & norm → trascurabile
```

### 5.5 Per token

* FLOPS per token ≈ 271.2M / (B*64) ≈ 4.24M FLOPS per token

---

## 6. Quantizzazione

* **Matrici principali** → int8: 8 bit / 32 bit = ×4 riduzione memoria
* **Embedding & RMSNorm** → 18/24 bit emulati in int32 → ~50-75% risparmio rispetto FP32
* Risultato: modello quantizzato ≈ 1.5-2M param “equivalenti FP32”

> Permette inferenze rapide su GPU consumer, notebook o CPU con memoria limitata.

---

## 7. Riassunto layer per layer

| Layer      | QKV       | Output attn | FFN       | RMSNorm | Totale layer |
| ---------- | --------- | ----------- | --------- | ------- | ------------ |
| 1          | 172,800   | 57,600      | 460,800   | 480     | 691,680      |
| 2          | 172,800   | 57,600      | 460,800   | 480     | 691,680      |
| 3          | 172,800   | 57,600      | 460,800   | 480     | 691,680      |
| 4          | 172,800   | 57,600      | 460,800   | 480     | 691,680      |
| 5          | 172,800   | 57,600      | 460,800   | 480     | 691,680      |
| 6          | 172,800   | 57,600      | 460,800   | 480     | 691,680      |
| Finale     | –         | –           | –         | 240     | 240          |
| **Totale** | 1,036,800 | 345,600     | 2,764,800 | 3,240   | 4,626,480    |

---

## 7. Risorse e link

* [PicoDAC](https://huggingface.co/Mattimax/PicoDAC)
* [PicoDAC-IT](https://huggingface.co/Mattimax/PicoDAC-IT)
* [M.INC HuggingFace](https://huggingface.co/MINC01)
* [Sito aziendale](https://sites.google.com/view/mattimax-site/home-page)
