import torch
import logging
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from accelerate import Accelerator

# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('zephyr_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration with logging
logger.info("Starting Zephyr fine-tuning...")
logger.info(f"Current timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Using open-source Zephyr model
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
DATASET_PATH = "./input_dataset.json"
OUTPUT_DIR = "./zephyr_output"

# Sample dataset - replace with your actual data
sample_data = [
    {
        "prompt": "",
        "response": ""
    }
]

# Save sample dataset
try:
    logger.info(f"Creating sample dataset at {DATASET_PATH}")
    import json
    with open(DATASET_PATH, 'w') as f:
        json.dump(sample_data, f)
    logger.info("Sample dataset created successfully")
except Exception as e:
    logger.error(f"Failed to create dataset: {str(e)}")
    raise

# Initialize accelerator for CPU
try:
    logger.info("Initializing accelerator for CPU")
    accelerator = Accelerator(cpu=True)
    DEVICE = accelerator.device
    logger.info(f"Using device: {DEVICE}")
except Exception as e:
    logger.error(f"Accelerator initialization failed: {str(e)}")
    raise

# Quantization config for CPU
logger.info("Setting up quantization configuration")
try:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    logger.info("Quantization config created successfully")
except Exception as e:
    logger.error(f"Quantization config failed: {str(e)}")
    raise

# Load model and tokenizer
logger.info(f"Loading model: {MODEL_NAME}")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer loaded successfully")

    logger.info("Now loading model (this may take several minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": accelerator.device},
        low_cpu_mem_usage=True
    )
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    raise

# LoRA configuration
logger.info("Configuring LoRA adapter")
try:
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    logger.info("LoRA adapter configured successfully")
    
    # Log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable_params:,} | Total params: {total_params:,} | Percentage: {100 * trainable_params / total_params:.2f}%")
except Exception as e:
    logger.error(f"LoRA configuration failed: {str(e)}")
    raise

# Dataset preparation
logger.info("Preparing dataset")
try:
    def preprocess_function(examples):
        texts = [f"<|user|>\n{p}</s>\n<|assistant|>\n{r}</s>" for p, r in zip(examples['prompt'], examples['response'])]
        tokenized = tokenizer(texts, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

    dataset = Dataset.from_json(DATASET_PATH)
    logger.info(f"Dataset loaded with {len(dataset)} examples")
    
    dataset = dataset.map(preprocess_function, batched=True)
    logger.info("Dataset preprocessing completed")
except Exception as e:
    logger.error(f"Dataset preparation failed: {str(e)}")
    raise

# Training arguments
logger.info("Setting up training arguments")
try:
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        save_steps=500,
        logging_steps=10,  # More frequent logging
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        num_train_epochs=3,  # Increased epochs for small dataset
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        disable_tqdm=False,
        log_level="info",
        save_total_limit=2
    )
    logger.info("Training arguments configured")
except Exception as e:
    logger.error(f"Training arguments setup failed: {str(e)}")
    raise

# Training function with logging
def train():
    try:
        logger.info("Starting training process")
        logger.info(f"Training configuration:\n{training_args}")
        
        from transformers import Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer
        )
        
        logger.info("Trainer initialized. Starting training...")
        train_result = trainer.train()
        logger.info("Training completed successfully")
        
        # Save metrics
        metrics = train_result.metrics
        logger.info(f"Training metrics: {metrics}")
        
        # Save model
        logger.info(f"Saving model to {OUTPUT_DIR}")
        trainer.save_model(OUTPUT_DIR)
        logger.info("Model saved successfully")
        
        return trainer
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

# Run training
try:
    trainer = train()
except Exception as e:
    logger.error(f"Training execution failed: {str(e)}")
    raise

# Generation function with logging
def generate_sexting_message(prompt):
    try:
        logger.info(f"Generating response for prompt: '{prompt}'")
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=DEVICE
        )
        
        formatted_prompt = f"<|user|>\n{prompt}</s>\n<|assistant|>\n"
        logger.debug(f"Formatted prompt: {formatted_prompt}")
        
        output = pipe(
            formatted_prompt,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1
        )
        
        response = output[0]['generated_text'].split("<|assistant|>")[-1].strip()
        response = response.replace("</s>", "").strip()
        logger.info(f"Generated response: '{response}'")
        
        return response
    except Exception as e:
        logger.error(f"Generation failed for prompt '{prompt}': {str(e)}")
        raise

# Test the model
try:
    test_prompts = [
        "this is a test prompt",
        "this is a test prompt",
        "this is a test prompt"
    ]
    
    logger.info("Running test generations...")
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        test_output = generate_sexting_message(prompt)
        print(f"Response: {test_output}")
        print("-"*50)
        
except Exception as e:
    logger.error(f"Test generation failed: {str(e)}")

logger.info("Fine-tuning process completed")