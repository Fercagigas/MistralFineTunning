"""
Script para realizar Adapter Tuning en el modelo Mistral-7B-Instruct-v0.2
utilizando un dataset médico procesado.

Este script implementa el método de Parameter-Efficient Fine-Tuning (PEFT)
conocido como Adapter Tuning, que agrega pequeñas capas adaptadoras entre
las capas del modelo pre-entrenado, manteniendo la mayoría de los parámetros
originales congelados.
"""

import os
import sys
import logging
import argparse
import torch
import numpy as np
from datetime import datetime
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed
)
from peft import (
    get_peft_model,
    AdapterConfig,
    TaskType,
    PeftModel,
    PeftConfig
)
from transformers.trainer_utils import get_last_checkpoint

# Configuración del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"adapter_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Script para Adapter Tuning de Mistral-7B-Instruct-v0.2")
    
    # Argumentos del modelo
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2",
                        help="Nombre o ruta del modelo base a utilizar")
    parser.add_argument("--tokenizer_name", type=str, default=None,
                        help="Nombre o ruta del tokenizer (si es diferente al modelo)")
    
    # Argumentos de datos
    parser.add_argument("--dataset_path", type=str, default="../processed_data/medical_dataset",
                        help="Ruta al dataset procesado")
    
    # Argumentos de PEFT (Adapter Tuning)
    parser.add_argument("--adapter_dim", type=int, default=64,
                        help="Dimensión de las capas adaptadoras")
    parser.add_argument("--adapter_dropout", type=float, default=0.1,
                        help="Tasa de dropout para las capas adaptadoras")
    parser.add_argument("--adapter_init_scale", type=float, default=1e-3,
                        help="Escala de inicialización para las capas adaptadoras")
    
    # Argumentos de entrenamiento
    parser.add_argument("--output_dir", type=str, default="../models/adapter_tuning",
                        help="Directorio donde se guardarán los checkpoints y el modelo final")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Número de épocas de entrenamiento")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Tamaño del batch por dispositivo para entrenamiento")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="Tamaño del batch por dispositivo para evaluación")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Número de pasos para acumulación de gradientes")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Tasa de aprendizaje inicial")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Peso de decaimiento para regularización")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Proporción de pasos de calentamiento")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Número de pasos entre logs")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Número de pasos entre evaluaciones")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Número de pasos entre guardados de checkpoints")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Norma máxima del gradiente para clipping")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla para reproducibilidad")
    parser.add_argument("--fp16", action="store_true",
                        help="Si se debe usar precisión mixta (FP16)")
    parser.add_argument("--bf16", action="store_true",
                        help="Si se debe usar precisión mixta (BF16)")
    parser.add_argument("--resume_from_checkpoint", action="store_true",
                        help="Si se debe continuar desde el último checkpoint")
    
    return parser.parse_args()

def check_hf_token():
    """Verifica que el token de HuggingFace esté configurado."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.warning("No se encontró el token de HuggingFace en las variables de entorno.")
        logger.warning("Si el modelo requiere autenticación, el script podría fallar.")
        logger.warning("Configure la variable de entorno HF_TOKEN con su token de HuggingFace.")
    return token

def load_dataset(dataset_path):
    """Carga el dataset procesado desde el disco."""
    try:
        logger.info(f"Cargando dataset desde {dataset_path}")
        dataset = load_from_disk(dataset_path)
        logger.info(f"Dataset cargado con éxito. Información: {dataset}")
        
        # Verificar que el dataset tenga las divisiones necesarias
        required_splits = ["train", "validation"]
        for split in required_splits:
            if split not in dataset:
                raise ValueError(f"El dataset no contiene la división '{split}'")
        
        return dataset
    except Exception as e:
        logger.error(f"Error al cargar el dataset: {e}")
        raise

def load_model_and_tokenizer(args, hf_token):
    """Carga el modelo base y el tokenizer."""
    try:
        logger.info(f"Cargando tokenizer: {args.tokenizer_name or args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name or args.model_name,
            token=hf_token,
            trust_remote_code=True
        )
        
        # Asegurar que el tokenizer tenga un token de padding
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                logger.warning("No se encontró token de padding ni EOS. Usando un token arbitrario.")
                tokenizer.pad_token = tokenizer.eos_token = "</s>"
        
        logger.info(f"Cargando modelo base: {args.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            token=hf_token,
            torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
            device_map="auto",
            trust_remote_code=True
        )
        
        logger.info(f"Modelo cargado con éxito. Número de parámetros: {model.num_parameters():,}")
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Error al cargar el modelo o tokenizer: {e}")
        raise

def prepare_adapter_config(args):
    """Prepara la configuración del adaptador para PEFT."""
    try:
        logger.info("Configurando Adapter Tuning")
        adapter_config = AdapterConfig(
            r=args.adapter_dim,  # Dimensión de reducción del adaptador
            lora_alpha=32,       # Escala para LoRA (usado en algunos adaptadores)
            lora_dropout=args.adapter_dropout,
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        
        logger.info(f"Configuración del adaptador: {adapter_config}")
        return adapter_config
    
    except Exception as e:
        logger.error(f"Error al configurar el adaptador: {e}")
        raise

def apply_adapter_to_model(model, adapter_config):
    """Aplica la configuración del adaptador al modelo base."""
    try:
        logger.info("Aplicando configuración del adaptador al modelo")
        model = get_peft_model(model, adapter_config)
        
        # Imprimir información sobre parámetros entrenables vs congelados
        model.print_trainable_parameters()
        
        return model
    
    except Exception as e:
        logger.error(f"Error al aplicar el adaptador al modelo: {e}")
        raise

def prepare_training_args(args):
    """Prepara los argumentos de entrenamiento para el Trainer."""
    try:
        # Crear directorio de salida si no existe
        os.makedirs(args.output_dir, exist_ok=True)
        
        logger.info("Configurando argumentos de entrenamiento")
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            logging_dir=os.path.join(args.output_dir, "logs"),
            logging_steps=args.logging_steps,
            evaluation_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=args.fp16,
            bf16=args.bf16,
            max_grad_norm=args.max_grad_norm,
            report_to="tensorboard",
            remove_unused_columns=False,
            label_names=["labels"],
            dataloader_drop_last=True,
            seed=args.seed,
        )
        
        logger.info(f"Argumentos de entrenamiento configurados: {training_args}")
        return training_args
    
    except Exception as e:
        logger.error(f"Error al configurar los argumentos de entrenamiento: {e}")
        raise

def train_model(model, tokenizer, dataset, training_args, args):
    """Entrena el modelo con los adaptadores."""
    try:
        logger.info("Preparando data collator")
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        logger.info("Inicializando Trainer")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        
        # Verificar si se debe continuar desde un checkpoint
        checkpoint = None
        if args.resume_from_checkpoint:
            checkpoint = get_last_checkpoint(args.output_dir)
            if checkpoint is None:
                logger.warning("No se encontró ningún checkpoint para continuar el entrenamiento.")
            else:
                logger.info(f"Continuando entrenamiento desde checkpoint: {checkpoint}")
        
        logger.info("Iniciando entrenamiento")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        logger.info("Entrenamiento completado")
        logger.info(f"Métricas de entrenamiento: {train_result.metrics}")
        
        # Guardar métricas de entrenamiento
        trainer.save_metrics("train", train_result.metrics)
        
        # Guardar modelo y tokenizer
        logger.info(f"Guardando modelo en {args.output_dir}")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        # Guardar argumentos de entrenamiento
        trainer.save_state()
        
        return trainer
    
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {e}")
        raise

def evaluate_model(trainer, dataset):
    """Evalúa el modelo entrenado."""
    try:
        logger.info("Evaluando modelo en conjunto de validación")
        eval_results = trainer.evaluate(eval_dataset=dataset["validation"])
        
        logger.info(f"Resultados de evaluación: {eval_results}")
        trainer.save_metrics("eval", eval_results)
        
        return eval_results
    
    except Exception as e:
        logger.error(f"Error durante la evaluación: {e}")
        raise

def main():
    """Función principal que ejecuta el flujo completo de Adapter Tuning."""
    # Parsear argumentos
    args = parse_args()
    
    # Configurar semilla para reproducibilidad
    set_seed(args.seed)
    
    # Verificar token de HuggingFace
    hf_token = check_hf_token()
    
    try:
        # Cargar dataset
        dataset = load_dataset(args.dataset_path)
        
        # Cargar modelo y tokenizer
        model, tokenizer = load_model_and_tokenizer(args, hf_token)
        
        # Preparar configuración del adaptador
        adapter_config = prepare_adapter_config(args)
        
        # Aplicar adaptador al modelo
        model = apply_adapter_to_model(model, adapter_config)
        
        # Preparar argumentos de entrenamiento
        training_args = prepare_training_args(args)
        
        # Entrenar modelo
        trainer = train_model(model, tokenizer, dataset, training_args, args)
        
        # Evaluar modelo
        eval_results = evaluate_model(trainer, dataset)
        
        logger.info("Proceso de Adapter Tuning completado con éxito")
        logger.info(f"Pérdida final en validación: {eval_results['eval_loss']:.4f}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error en el proceso de Adapter Tuning: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
