# 774 Million Parameter Large Language Model from Scratch

## Stage 1: **Designing and Training Custom GPT Architecture**
- Designed and implemented core components of the **GPT architecture**, including:
  - **Multi-head attention**
  - **Dropout**
  - **Causal attention**
  - **Feedforward neural networks** with **GELU activation**
  - **Layer normalization** and **shortcut connections**
- Configured **model parameters** to optimize training and inference performance:
  - Incorporated **attention mechanisms** with techniques like **softmax normalization** and scaling by `sqrt(dimension)` to stabilize variance and improve gradient properties.
- Integrated **Byte Pair Encoding (BPE)** for **tokenization** and **text generation**:
  - Utilized **token embeddings** combined with **positional embeddings** for input representation.
- Calculated **text generation loss** using **cross-entropy** and **perplexity metrics** to evaluate and refine model performance.
- Developed a **PyTorch-based training framework**:
  - Included **dataset dataloaders** for efficient training and validation loss computation.
  - Implemented **temperature scaling** and **top-k sampling** to control randomness in **text generation**.
- Managed **model persistence**:
  - Enabled **loading and saving weights** in **PyTorch** to facilitate **reproducibility** and **deployment**.

## Stage 2: **Foundation Model**
- Loaded **pretrained weights** of the **GPT-2 774M parameter model** from **OpenAI**, originally saved via **TensorFlow**.
- Ensured seamless integration of **weights** from the **OpenAI-provided parameter dictionary** into the custom `GPTModel` instance.
- Preserved the integrity of the **foundational model architecture** for downstream tasks.

## Stage 3: **Classification Finetuning**
- Developed a **spam classification model** using a **dataset** of 1,494 **SMS messages** (spam and ham) from the **UC Irvine Repository**.
- Preprocessed **data** by padding **messages** to the longest sequence in each batch using the `<endoftext>` **token** for consistency.
- Fine-tuned a **pretrained LLM**:
  - Replaced the original **50,257-class output layer** with a custom **2-class output layer** (spam: 1, not spam: 0).
  - Configured the final **transformer block** and **LayerNorm module** for **trainability**, enabling the **model** to adapt to the **classification task**.

## Stage 4: **Instruction Finetuning & Evaluating LLM**
- Used a **dataset** of 1,100 **instruction-response pairs**, converted into **Alpaca format**.
- Implemented a custom **collate function** to dynamically pad sequences within each batch, minimizing unnecessary padding and improving computational efficiency.
- Automated **conversational benchmarking**:
  - Used the **Llama 3.1 8 Billion Parameter Model** to evaluate fine-tuned **model responses**.
- Deployed the **model** using the open-source **Ollama application**:
  - Utilized **REST API** to streamline **request automation** within **Python scripts**.
- Achieved an average **accuracy score** of **62.7%**.
