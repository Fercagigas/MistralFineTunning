#!/usr/bin/env python
"""
Script para ejecutar todo el pipeline de fine-tuning PEFT:
1. Preparación de datos
2. Fine-tuning con QLoRA
3. Fine-tuning con Adapter Tuning
4. Fine-tuning con P-Tuning
5. Evaluación de modelos
"""

import os
import sys
import argparse
import subprocess
import json
from datetime import datetime
from pathlib import Path

# Importar utilidades comunes
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.common import get_logger, exception_handler, get_timestamp, login_huggingface


@exception_handler
def run_command(cmd, description):
    """Ejecuta un comando y registra su salida"""
    logger.info(f"Ejecutando: {description}")
    logger.info(f"Comando: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Mostrar salida en tiempo real
    for line in iter(process.stdout.readline, ''):
        if line:
            logger.info(line.rstrip())
    
    process.stdout.close()
    return_code = process.wait()
    
    if return_code != 0:
        logger.error(f"Error al ejecutar {description}. Código de salida: {return_code}")
        return False
    
    logger.info(f"{description} completado con éxito.")
    return True


def parse_args():
    """Parsea los argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description="Pipeline completo de fine-tuning PEFT")
    
    # Argumentos generales
    parser.add_argument("--input_data", type=str, required=True,
                        help="Ruta al archivo de datos de entrada")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2",
                        help="Nombre del modelo base a ajustar")
    parser.add_argument("--output_dir", type=str, default="./models",
                        help="Directorio base para guardar los modelos")
    parser.add_argument("--processed_data_dir", type=str, default="./processed_data",
                        help="Directorio para datos procesados")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Token de Hugging Face (opcional, también puede usar HF_TOKEN env var)")
    
    # Argumentos de entrenamiento
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Número de épocas para entrenamiento")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Tamaño de batch para entrenamiento")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Tasa de aprendizaje para entrenamiento")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Longitud máxima de secuencia")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla para reproducibilidad")
    
    # Argumentos específicos de métodos PEFT
    parser.add_argument("--lora_r", type=int, default=16,
                        help="Rango de las matrices LoRA (para QLoRA)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="Factor de escalado LoRA (para QLoRA)")
    parser.add_argument("--adapter_size", type=int, default=64,
                        help="Dimensión de las capas adaptadoras (para Adapter Tuning)")
    parser.add_argument("--num_virtual_tokens", type=int, default=20,
                        help="Número de tokens virtuales (para P-Tuning)")
    
    # Flags para controlar qué pasos ejecutar
    parser.add_argument("--skip_data_prep", action="store_true",
                        help="Omitir la preparación de datos")
    parser.add_argument("--skip_qlora", action="store_true",
                        help="Omitir el entrenamiento con QLoRA")
    parser.add_argument("--skip_adapter", action="store_true",
                        help="Omitir el entrenamiento con Adapter Tuning")
    parser.add_argument("--skip_ptuning", action="store_true",
                        help="Omitir el entrenamiento con P-Tuning")
    parser.add_argument("--skip_evaluation", action="store_true",
                        help="Omitir la evaluación de modelos")
    
    return parser.parse_args()


@exception_handler
def prepare_data(args):
    """Ejecuta el script de preparación de datos"""
    cmd = [
        "python", "utils/prepare_dataset.py",
        "--input_data", args.input_data,
        "--output_dir", args.processed_data_dir,
        "--model_name", args.model_name,
        "--max_length", str(args.max_length),
        "--seed", str(args.seed)
    ]
    
    return run_command(cmd, "Preparación de datos")


@exception_handler
def train_qlora(args):
    """Ejecuta el entrenamiento con QLoRA"""
    timestamp = get_timestamp()
    output_dir = os.path.join(args.output_dir, f"qlora_{timestamp}")
    
    cmd = [
        "python", "qlora/train_qlora.py",
        "--model_name", args.model_name,
        "--data_path", args.processed_data_dir,
        "--output_dir", output_dir,
        "--num_epochs", str(args.num_epochs),
        "--learning_rate", str(args.learning_rate),
        "--batch_size", str(args.batch_size),
        "--lora_r", str(args.lora_r),
        "--lora_alpha", str(args.lora_alpha),
        "--max_length", str(args.max_length),
        "--seed", str(args.seed)
    ]
    
    success = run_command(cmd, "Entrenamiento con QLoRA")
    if success:
        return output_dir
    return None


@exception_handler
def train_adapter(args):
    """Ejecuta el entrenamiento con Adapter Tuning"""
    timestamp = get_timestamp()
    output_dir = os.path.join(args.output_dir, f"adapter_{timestamp}")
    
    cmd = [
        "python", "adapter_tuning/train_adapter.py",
        "--model_name", args.model_name,
        "--data_path", args.processed_data_dir,
        "--output_dir", output_dir,
        "--num_epochs", str(args.num_epochs),
        "--learning_rate", str(args.learning_rate),
        "--batch_size", str(args.batch_size),
        "--adapter_size", str(args.adapter_size),
        "--max_length", str(args.max_length),
        "--seed", str(args.seed)
    ]
    
    success = run_command(cmd, "Entrenamiento con Adapter Tuning")
    if success:
        return output_dir
    return None


@exception_handler
def train_ptuning(args):
    """Ejecuta el entrenamiento con P-Tuning"""
    timestamp = get_timestamp()
    output_dir = os.path.join(args.output_dir, f"ptuning_{timestamp}")
    
    cmd = [
        "python", "p_tuning/train_ptuning.py",
        "--model_name", args.model_name,
        "--data_path", args.processed_data_dir,
        "--output_dir", output_dir,
        "--num_epochs", str(args.num_epochs),
        "--learning_rate", str(args.learning_rate),
        "--batch_size", str(args.batch_size),
        "--num_virtual_tokens", str(args.num_virtual_tokens),
        "--max_length", str(args.max_length),
        "--seed", str(args.seed)
    ]
    
    success = run_command(cmd, "Entrenamiento con P-Tuning")
    if success:
        return output_dir
    return None


@exception_handler
def evaluate_models(args, qlora_model=None, adapter_model=None, ptuning_model=None):
    """Ejecuta la evaluación de modelos"""
    # Construir la lista de modelos a evaluar
    models_to_evaluate = []
    
    if qlora_model:
        models_to_evaluate.extend(["--qlora_model", qlora_model])
    
    if adapter_model:
        models_to_evaluate.extend(["--adapter_model", adapter_model])
    
    if ptuning_model:
        models_to_evaluate.extend(["--ptuning_model", ptuning_model])
    
    if not models_to_evaluate:
        logger.warning("No hay modelos para evaluar. Omitiendo evaluación.")
        return False
    
    # Construir el comando de evaluación
    timestamp = get_timestamp()
    output_file = f"evaluation_results_{timestamp}.json"
    
    cmd = [
        "python", "utils/evaluate_models.py",
        "--base_model", args.model_name,
        "--test_data", os.path.join(args.processed_data_dir, "test.json"),
        "--output_file", output_file,
        "--seed", str(args.seed)
    ] + models_to_evaluate
    
    return run_command(cmd, "Evaluación de modelos")


@exception_handler
def main():
    """Función principal que ejecuta todo el pipeline"""
    args = parse_args()
    
    # Crear directorios de salida
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.processed_data_dir, exist_ok=True)
    
    # Iniciar sesión en Hugging Face
    login_huggingface(args.hf_token)
    
    # Guardar configuración
    config_path = os.path.join(args.output_dir, f"config_{get_timestamp()}.json")
    with open(config_path, 'w') as f:
        # Convertir args a diccionario, excluyendo hf_token por seguridad
        config = {k: v for k, v in vars(args).items() if k != 'hf_token'}
        json.dump(config, f, indent=2)
    
    logger.info(f"Configuración guardada en {config_path}")
    
    # Ejecutar pipeline
    qlora_model = adapter_model = ptuning_model = None
    
    # 1. Preparación de datos
    if not args.skip_data_prep:
        if not prepare_data(args):
            logger.error("Error en la preparación de datos. Abortando pipeline.")
            return
    else:
        logger.info("Omitiendo preparación de datos.")
    
    # 2. Fine-tuning con QLoRA
    if not args.skip_qlora:
        qlora_model = train_qlora(args)
        if not qlora_model:
            logger.error("Error en el entrenamiento con QLoRA.")
    else:
        logger.info("Omitiendo entrenamiento con QLoRA.")
    
    # 3. Fine-tuning con Adapter Tuning
    if not args.skip_adapter:
        adapter_model = train_adapter(args)
        if not adapter_model:
            logger.error("Error en el entrenamiento con Adapter Tuning.")
    else:
        logger.info("Omitiendo entrenamiento con Adapter Tuning.")
    
    # 4. Fine-tuning con P-Tuning
    if not args.skip_ptuning:
        ptuning_model = train_ptuning(args)
        if not ptuning_model:
            logger.error("Error en el entrenamiento con P-Tuning.")
    else:
        logger.info("Omitiendo entrenamiento con P-Tuning.")
    
    # 5. Evaluación de modelos
    if not args.skip_evaluation:
        if not evaluate_models(args, qlora_model, adapter_model, ptuning_model):
            logger.error("Error en la evaluación de modelos.")
    else:
        logger.info("Omitiendo evaluación de modelos.")
    
    logger.info("Pipeline completado.")


if __name__ == "__main__":
    # Configurar logger
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"pipeline_{get_timestamp()}.log")
    
    logger = get_logger("pipeline", log_file)
    logger.info("Iniciando pipeline de fine-tuning PEFT")
    
    try:
        main()
    except Exception as e:
        logger.error(f"Error en el pipeline: {str(e)}", exc_info=True)
        sys.exit(1)
