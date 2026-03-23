"""Training loop for fine-tuning LLMs with Unsloth."""
from typing import Dict, Any
from pathlib import Path

try:
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    # Placeholder types for type checking when unsloth is not installed
    FastLanguageModel = None
    load_dataset = None
    SFTTrainer = None
    TrainingArguments = None


def load_jsonl_dataset(dataset_path: str) -> Any:
    """Load dataset from JSONL file.

    Args:
        dataset_path: Path to the JSONL file.

    Returns:
        Dataset object from the datasets library.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """
    if not UNSLOTH_AVAILABLE:
        raise ImportError("unsloth and datasets are required for loading datasets")

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    dataset = load_dataset('json', data_files=str(dataset_path))
    return dataset


def format_instruction(example: Dict[str, str]) -> str:
    """Format example into instruction-following format.

    Args:
        example: Dictionary containing 'instruction', 'output', and optionally 'thinking'.

    Returns:
        Formatted string with special tokens.

    Raises:
        KeyError: If required fields 'instruction' or 'output' are missing.
    """
    instruction = example['instruction']
    output = example['output']
    thinking = example.get('thinking')

    if thinking:
        formatted = (
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"<thinking>{thinking}</thinking>\n"
            f"{output}<|im_end|>"
        )
    else:
        formatted = (
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"{output}<|im_end|>"
        )

    return formatted


def train_with_unsloth(
    train_dataset_path: str,
    eval_dataset_path: str,
    output_dir: str,
    base_model: str,
    params: Dict[str, Any]
) -> Dict[str, float]:
    """Train a model with Unsloth and PEFT/LoRA.

    Args:
        train_dataset_path: Path to the training dataset JSONL file.
        eval_dataset_path: Path to the evaluation dataset JSONL file.
        output_dir: Directory to save the trained model.
        base_model: Name or path of the base model to fine-tune.
        params: Dictionary containing hyperparameters:
            - learning_rate: float
            - lora_rank: int (8, 16, 32, 64)
            - lora_alpha: int (16, 32, 64, 128)
            - lora_dropout: float (0.0-0.1)
            - batch_size: int (1, 2, 4)
            - num_epochs: int (1-5)

    Returns:
        Dictionary with 'eval_loss' and 'train_loss' keys.

    Raises:
        ImportError: If unsloth or required dependencies are not installed.
        KeyError: If required parameters are missing from params.
        FileNotFoundError: If dataset files do not exist.
    """
    if not UNSLOTH_AVAILABLE:
        raise ImportError(
            "unsloth, datasets, trl, and transformers are required for training. "
            "Install with: pip install unsloth datasets trl transformers"
        )

    # Validate required parameters
    required_params = ['learning_rate', 'lora_rank', 'lora_alpha', 'lora_dropout', 'batch_size', 'num_epochs']
    for param in required_params:
        if param not in params:
            raise KeyError(f"Missing required parameter: {param}")

    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=2048,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )

    # Setup LoRA
    model, tokenizer = FastLanguageModel.get_peft_model(
        model,
        r=params['lora_rank'],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=params['lora_alpha'],
        lora_dropout=params['lora_dropout'],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
    )

    # Load datasets
    train_dataset = load_dataset('json', data_files=train_dataset_path)['train']
    eval_dataset = load_dataset('json', data_files=eval_dataset_path)['train']

    # Format datasets
    def formatting_prompts_func(examples):
        instructions = examples['instruction']
        outputs = examples['output']
        thinkings = examples.get('thinking', [None] * len(instructions))

        texts = []
        for instruction, output, thinking in zip(instructions, outputs, thinkings):
            example = {
                'instruction': instruction,
                'output': output,
                'thinking': thinking
            }
            text = format_instruction(example) + tokenizer.eos_token
            texts.append(text)

        return {'text': texts}

    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=params['num_epochs'],
        per_device_train_batch_size=params['batch_size'],
        per_device_eval_batch_size=params['batch_size'],
        gradient_accumulation_steps=4,
        warmup_steps=5,
        learning_rate=params['learning_rate'],
        fp16=not FastLanguageModel.is_bfloat16_supported(),
        bf16=FastLanguageModel.is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    # Train
    train_result = trainer.train()

    # Get final metrics
    train_loss = train_result.training_loss if hasattr(train_result, 'training_loss') else 0.0
    eval_metrics = trainer.evaluate()
    eval_loss = eval_metrics.get('eval_loss', 0.0)

    # Save final model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path / "final_model"))
    tokenizer.save_pretrained(str(output_path / "final_model"))

    return {
        'eval_loss': eval_loss,
        'train_loss': train_loss
    }
