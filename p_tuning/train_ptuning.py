#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para realizar P-Tuning del modelo Mistral-7B-Instruct-v0.2.

P-Tuning es una técnica de Parameter-Efficient Fine-Tuning (PEFT) que añade
tokens virtuales entrenables al prompt mientras mantiene los parámetros
del modelo base congelados.
"""

import os
import sys
import logging
import argparse
from datetime import datetime
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    default_data_collator,
    set_seed,
)
from peft import (
    get_peft_model,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
    PeftModel,
    PeftConfig,
)
from huggingface_hub import login
import numpy as np
from tqdm import tqdm

# Configuración del logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"p_tuning_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Script para P-Tuning de Mistral-7B-Instruct-v0.2")
    
    # Argumentos para el modelo y datos
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2",
                        help="Nombre o ruta del modelo base")
    parser.add_argument("--data_path", type=str, default="../processed_data",
                        help="Ruta al dataset procesado")
    parser.add_argument("--output_dir", type=str, default="../models/p_tuning",
                        help="Directorio para guardar el modelo fine-tuned")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Token de HuggingFace para descargar modelos")
    
    # Argumentos específicos de P-Tuning
    parser.add_argument("--num_virtual_tokens", type=int, default=20,
                        help="Número de tokens virtuales a añadir")
    parser.add_argument("--prompt_tuning_init", type=str, default="TEXT",
                        choices=["RANDOM", "TEXT"],
                        help="Método de inicialización para los tokens virtuales")
    parser.add_argument("--prompt_tuning_init_text", type=str, 
                        default="Responde a esta consulta médica de manera precisa y profesional:",
                        help="Texto para inicializar los tokens virtuales (si prompt_tuning_init=TEXT)")
    
    # Argumentos de entrenamiento
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Tamaño del batch por dispositivo para entrenamiento")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="Tamaño del batch por dispositivo para evaluación")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Número de pasos para acumulación de gradientes")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Tasa de aprendizaje inicial")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        help="Tipo de scheduler para la tasa de aprendizaje")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Ratio de calentamiento para el scheduler")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Peso de decaimiento para la optimización")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Número de épocas de entrenamiento")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Número máximo de pasos de entrenamiento (-1 para usar num_train_epochs)")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Número de pasos entre logs")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Número de pasos entre evaluaciones")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Número de pasos entre guardados del modelo")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla aleatoria para reproducibilidad")
    parser.add_argument("--fp16", action="store_true",
                        help="Usar precisión mixta FP16")
    parser.add_argument("--bf16", action="store_true",
                        help="Usar precisión mixta BF16 (requiere GPU compatible)")
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                        help="Cargar modelo en cuantización de 4 bits para reducir memoria")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Longitud máxima de secuencia para entrenamiento")
    
    return parser.parse_args()

def load_and_prepare_dataset(data_path, tokenizer, max_seq_length):
    """Carga y prepara el dataset para el entrenamiento."""
    logger.info(f"Cargando dataset desde {data_path}")
    try:
        dataset = load_from_disk(data_path)
        logger.info(f"Dataset cargado: {dataset}")
        
        # Verificar si el dataset ya está dividido en train/validation
        if "train" not in dataset.keys() or "validation" not in dataset.keys():
            logger.info("Dividiendo dataset en train/validation")
            # Dividir el dataset si no está ya dividido
            dataset = dataset.train_test_split(test_size=0.1, seed=42)
            dataset = {
                "train": dataset["train"],
                "validation": dataset["test"]
            }
        
        # Verificar si los datos ya están tokenizados
        sample = dataset["train"][0]
        if "input_ids" not in sample or "attention_mask" not in sample:
            logger.info("Tokenizando dataset")
            
            def tokenize_function(examples):
                return tokenizer(
                    examples["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=max_seq_length,
                    return_tensors="pt"
                )
            
            tokenized_datasets = {}
            for split, ds in dataset.items():
                tokenized_datasets[split] = ds.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=["text"] if "text" in ds.column_names else None
                )
            
            dataset = tokenized_datasets
        
        return dataset
    
    except Exception as e:
        logger.error(f"Error al cargar o preparar el dataset: {e}")
        raise

def compute_metrics(eval_preds):
    """Calcula métricas de evaluación."""
    logits, labels = eval_preds
    
    # Calcular pérdida
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    # Calcular perplejidad
    perplexity = torch.exp(loss)
    
    return {
        "loss": loss.item(),
        "perplexity": perplexity.item()
    }

def main():
    """Función principal para el entrenamiento con P-Tuning."""
    args = parse_args()
    
    # Configurar semilla para reproducibilidad
    set_seed(args.seed)
    
    # Crear directorio de salida si no existe
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Iniciar sesión en HuggingFace Hub si se proporciona token
    if args.hf_token:
        login(token=args.hf_token)
    else:
        # Intentar obtener el token desde variable de entorno
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
            logger.info("Iniciando sesión en HuggingFace Hub con token de variable de entorno")
        else:
            logger.warning("No se proporcionó token de HuggingFace. Algunos modelos pueden no ser accesibles.")
    
    try:
        # Cargar tokenizer
        logger.info(f"Cargando tokenizer para {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            use_fast=True,
            padding_side="right",
            trust_remote_code=True
        )
        
        # Asegurar que el tokenizer tenga token de padding
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Estableciendo pad_token = eos_token ({tokenizer.eos_token})")
        
        # Cargar modelo base
        logger.info(f"Cargando modelo base {args.model_name}")
        model_kwargs = {}
        if args.load_in_4bit:
            model_kwargs.update({
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
                "bnb_4bit_use_double_quant": True,
            })
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            **model_kwargs
        )
        
        # Configurar P-Tuning
        logger.info(f"Configurando P-Tuning con {args.num_virtual_tokens} tokens virtuales")
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT if args.prompt_tuning_init == "TEXT" else PromptTuningInit.RANDOM,
            num_virtual_tokens=args.num_virtual_tokens,
            prompt_tuning_init_text=args.prompt_tuning_init_text if args.prompt_tuning_init == "TEXT" else None,
            tokenizer_name_or_path=args.model_name,
        )
        
        # Crear modelo PEFT
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        # Cargar y preparar dataset
        dataset = load_and_prepare_dataset(args.data_path, tokenizer, args.max_seq_length)
        
        # Configurar argumentos de entrenamiento
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps,
            logging_steps=args.logging_steps,
            evaluation_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            fp16=args.fp16,
            bf16=args.bf16,
            report_to="tensorboard",
            remove_unused_columns=False,
            push_to_hub=False,
            seed=args.seed,
        )
        
        # Configurar data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Inicializar Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        
        # Entrenar modelo
        logger.info("Iniciando entrenamiento")
        train_result = trainer.train()
        
        # Guardar estadísticas de entrenamiento
        trainer.save_state()
        
        # Guardar modelo final
        logger.info(f"Guardando modelo final en {args.output_dir}")
        trainer.save_model(args.output_dir)
        
        # Guardar tokenizer
        tokenizer.save_pretrained(args.output_dir)
        
        # Guardar configuración PEFT
        model.save_pretrained(args.output_dir)
        
        # Evaluar modelo
        logger.info("Evaluando modelo final")
        eval_results = trainer.evaluate()
        
        # Imprimir y guardar resultados de evaluación
        logger.info(f"Resultados de evaluación: {eval_results}")
        with open(os.path.join(args.output_dir, "eval_results.txt"), "w") as f:
            for key, value in eval_results.items():
                f.write(f"{key} = {value}\n")
        
        logger.info("Entrenamiento completado con éxito")
        
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
