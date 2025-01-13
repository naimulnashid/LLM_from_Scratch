Stage 1: Designing and Training Custom GPT Architecture
Designed and implemented core components of the GPT architecture, including multi-head attention, dropout, causal attention, feedforward neural networks with GELU activation, layer normalization, and shortcut connections, ensuring scalability and efficiency. 
Configured model parameters to optimize training and inference performance, incorporating attention mechanisms with key techniques like softmax normalization and scaling by sqrt(dimension) to stabilize variance and improve gradient properties. 
Integrated Byte Pair Encoding (BPE) for tokenization and text generation, leveraging token embeddings combined with positional embeddings for input representation. 
Calculated text generation loss using cross-entropy and perplexity metrics to evaluate and refine model performance. 
Developed a PyTorch-based training framework, including dataset dataloaders for efficient training and validation loss computation, and implemented temperature scaling and top-k sampling to control randomness in text generation. 
Managed model persistence by loading and saving weights in PyTorch to facilitate reproducibility and deployment.

Stage 2: Foundation Model
Loaded pretrained weights of the large GPT-2 774MM parameter model from OpenAI, that was originally saved via TensorFlow, ensuring seamless alignment with a custom PyTorch-based GPTModel framework. 
Ensured seamless integration of weights from the OpenAI-provided parameter dictionary into the custom GPTModel instance, ensuring compatibility and preserving the integrity of the foundational model architecture  for downstream tasks, establishing a solid base for further fine-tuning and customization. 
Loaded pretrained weights from OpenAI's foundational models to initialize a robust language model framework.
Ensured seamless integration of foundational model architecture for downstream tasks, establishing a solid base for further fine-tuning and customization.

Stage 3: Classification Finetuning
Developed a spam classification model using a dataset of 1,494 SMS messages (spam and ham) from the UC Irvine Repository. 
Preprocessed data by padding messages to the longest sequence in each batch using the <endoftext> token for consistency. 
Designed and implemented dataloaders to streamline the data ingestion process for the model.
Fine-tuned a pretrained LLM by replacing its original 50,257-class output layer with a custom 2-class output layer (spam: 1, not spam: 0). 
Configured the final transformer block and LayerNorm module for trainability, enabling the model to adapt effectively to the classification task. 
Calculated classification loss and accuracy metrics to evaluate model performance during training and testing. 
Successfully fine-tuned the LLM to classify text, achieving enhanced accuracy in identifying spam versus non-spam content.

Stage 4.1: Instruction Finetuning
Conducted fine-tuning of a pretrained LLM using a dataset of 1,100 instruction-response pairs.
Prepared the dataset by converting instructions into Alpaca format, splitting into train-validation-test subsets, and organizing data into training batches. 
Implemented a custom collate function to dynamically pad sequences within each batch to minimize unnecessary padding, improving computational efficiency. 
Fine-tuned the model on prepared instruction data, improving its response accuracy and alignment with user queries.
Extracted and saved responses to evaluate and optimize model performance post-finetuning.

Stage 4.2: Evaluating LLM
Automated Conversational Benchmark with Llama 3.1 8 Billion Parameter Model to Evaluate the Fine-tuned Model Responses.
The model was deployed using the open-source Ollama application, with REST API utilized to streamline request automation within Python scripts.
Achieved average accuracy score of 62.7
