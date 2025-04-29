#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para realizar QLoRA fine-tuning del modelo Mistral-7B-Instruct-v0.2
utilizando un dataset médico procesado.

Este script implementa Parameter-Efficient Fine-Tuning (PEFT) mediante QLoRA
(Quantized Low-Rank Adaptation) para adaptar el modelo Mistral a un dominio médico
con recursos computacionales limitados.
"""

import os
import sys
import logging
import argparse
import torch
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
import bitsandbytes as bnb
from datetime import datetime

# Configuración del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"qlora_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="QLoRA Fine-tuning para Mistral-7B-Instruct-v0.2")
    
    # Parámetros del modelo
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2",
                        help="Nombre o ruta del modelo base a utilizar")
    parser.add_argument("--tokenizer_name", type=str, default=None,
                        help="Nombre o ruta del tokenizer (si es diferente del modelo)")
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                        help="Cargar modelo en precisión de 4 bits")
    parser.add_argument("--load_in_8bit", action="store_true", default=False,
                        help="Cargar modelo en precisión de 8 bits")
    
    # Parámetros de QLoRA
    parser.add_argument("--lora_r", type=int, default=16,
                        help="Rango de las matrices LoRA")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="Valor de alpha para LoRA")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="Dropout para LoRA")
    parser.add_argument("--target_modules", type=str, nargs="+", 
                        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                        help="Módulos objetivo para aplicar LoRA")
    
    # Parámetros de datos
    parser.add_argument("--data_path", type=str, default="../processed_data",
                        help="Ruta al dataset procesado")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Longitud máxima de secuencia para el entrenamiento")
    
    # Parámetros de entrenamiento
    parser.add_argument("--output_dir", type=str, default="../models/qlora",
                        help="Directorio donde guardar el modelo fine-tuned")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Tamaño de batch por dispositivo para entrenamiento")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="Tamaño de batch por dispositivo para evaluación")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Número de pasos para acumulación de gradientes")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Tasa de aprendizaje inicial")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Peso de decaimiento para regularización")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Número de épocas de entrenamiento")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Ratio de calentamiento para el learning rate scheduler")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        help="Tipo de scheduler para learning rate")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Número de pasos entre logs")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Número de pasos entre evaluaciones")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Número de pasos entre guardados del modelo")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla para reproducibilidad")
    
    # Parámetros adicionales
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="Usar precisión mixta FP16")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Usar precisión mixta BF16 (si está disponible)")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Usar gradient checkpointing para ahorrar memoria")
    
    return parser.parse_args()

def verify_hf_token():
    """Verifica que el token de HuggingFace esté configurado."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.warning("No se ha encontrado el token de HuggingFace en las variables de entorno.")
        logger.warning("Algunas funcionalidades pueden no estar disponibles.")
        logger.warning("Configure la variable de entorno HF_TOKEN con su token de HuggingFace.")
        return False
    return True

def load_tokenized_dataset(data_path):
    """Carga el dataset tokenizado desde el disco."""
    try:
        logger.info(f"Cargando dataset desde {data_path}")
        dataset = load_from_disk(data_path)
        logger.info(f"Dataset cargado con éxito. Información: {dataset}")
        return dataset
    except Exception as e:
        logger.error(f"Error al cargar el dataset: {e}")
        raise

def load_model_and_tokenizer(args):
    """Carga el modelo y el tokenizer con la configuración especificada."""
    logger.info(f"Cargando modelo {args.model_name}")
    
    # Verificar si se debe cargar en 4-bit o 8-bit
    if args.load_in_4bit and args.load_in_8bit:
        logger.warning("Se especificaron tanto load_in_4bit como load_in_8bit. Usando 4-bit por defecto.")
        args.load_in_8bit = False
    
    # Configuración de cuantización
    quantization_config = None
    if args.load_in_4bit:
        logger.info("Cargando modelo en precisión de 4 bits")
        quantization_config = bnb.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif args.load_in_8bit:
        logger.info("Cargando modelo en precisión de 8 bits")
        quantization_config = bnb.BitsAndBytesConfig(
            load_in_8bit=True,
        )
    
    try:
        # Cargar el modelo
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Preparar el modelo para entrenamiento en k-bits
        if args.load_in_4bit or args.load_in_8bit:
            model = prepare_model_for_kbit_training(model)
        
        # Cargar el tokenizer
        tokenizer_name = args.tokenizer_name if args.tokenizer_name else args.model_name
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
        )
        
        # Asegurarse de que el tokenizer tenga token de padding
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                logger.warning("El tokenizer no tiene token de padding ni EOS. Usando un token especial.")
                tokenizer.pad_token = "[PAD]"
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Error al cargar el modelo o tokenizer: {e}")
        raise

def setup_lora(model, args):
    """Configura LoRA para el modelo."""
    logger.info("Configurando LoRA")
    
    try:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        logger.info(f"Configuración LoRA: {lora_config}")
        
        # Obtener modelo con PEFT
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model
    
    except Exception as e:
        logger.error(f"Error al configurar LoRA: {e}")
        raise

def train(model, tokenizer, dataset, args):
    """Entrena el modelo con los parámetros especificados."""
    logger.info("Iniciando entrenamiento")
    
    try:
        # Crear directorio de salida si no existe
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Configurar argumentos de entrenamiento
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_train_epochs=args.num_train_epochs,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type=args.lr_scheduler_type,
            logging_steps=args.logging_steps,
            evaluation_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            fp16=args.fp16,
            bf16=args.bf16,
            gradient_checkpointing=args.gradient_checkpointing,
            report_to="tensorboard",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            seed=args.seed,
            data_seed=args.seed,
        )
        
        # Crear data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Crear trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=data_collator,
        )
        
        # Entrenar modelo
        logger.info("Comenzando entrenamiento")
        trainer.train()
        
        # Guardar modelo
        logger.info(f"Guardando modelo en {args.output_dir}")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        # Evaluar modelo
        logger.info("Evaluando modelo en conjunto de validación")
        eval_results = trainer.evaluate()
        logger.info(f"Resultados de evaluación: {eval_results}")
        
        # Guardar métricas de evaluación
        with open(os.path.join(args.output_dir, "eval_results.txt"), "w") as f:
            f.write(f"Eval Loss: {eval_results['eval_loss']}\n")
        
        return eval_results
    
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {e}")
        raise

def main():
    """Función principal del script."""
    # Parsear argumentos
    args = parse_args()
    
    # Configurar semilla para reproducibilidad
    set_seed(args.seed)
    
    # Verificar token de HuggingFace
    verify_hf_token()
    
    try:
        # Cargar dataset
        dataset = load_tokenized_dataset(args.data_path)
        
        # Cargar modelo y tokenizer
        model, tokenizer = load_model_and_tokenizer(args)
        
        # Configurar LoRA
        model = setup_lora(model, args)
        
        # Entrenar modelo
        eval_results = train(model, tokenizer, dataset, args)
        
        logger.info("Entrenamiento completado con éxito")
        logger.info(f"Pérdida final en validación: {eval_results['eval_loss']}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error en el proceso de fine-tuning: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
