#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de evaluación para comparar modelos fine-tuned con diferentes métodos PEFT.
Este script permite evaluar y comparar el rendimiento de modelos entrenados con
QLoRA, Adapter Tuning y P-Tuning.
"""

import os
import json
import math
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    GenerationConfig
)
from peft import (
    PeftModel,
    PeftConfig,
    LoraConfig,
    AdapterConfig,
    PrefixTuningConfig,
    get_peft_model
)

# Configuración del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"eval_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Constantes
PEFT_METHODS = ["qlora", "adapter", "ptuning"]
DEFAULT_BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_MAX_LENGTH = 512
DEFAULT_METRICS = ["loss", "perplexity"]

def load_model(
    method: str,
    model_path: str,
    base_model_name: str = DEFAULT_BASE_MODEL,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    token: Optional[str] = None
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Carga un modelo fine-tuned con un método PEFT específico.
    
    Args:
        method: Método PEFT utilizado ('qlora', 'adapter', 'ptuning')
        model_path: Ruta al modelo fine-tuned
        base_model_name: Nombre del modelo base
        device: Dispositivo para cargar el modelo ('cuda', 'cpu')
        token: Token de HuggingFace para modelos privados
        
    Returns:
        Tuple con el modelo y el tokenizador
    """
    logger.info(f"Cargando modelo con método {method} desde {model_path}")
    
    try:
        # Cargar el tokenizador
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            token=token,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        # Cargar el modelo base
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            token=token,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Cargar la configuración PEFT según el método
        if method == "qlora":
            peft_config = LoraConfig.from_pretrained(model_path)
        elif method == "adapter":
            peft_config = AdapterConfig.from_pretrained(model_path)
        elif method == "ptuning":
            peft_config = PrefixTuningConfig.from_pretrained(model_path)
        else:
            raise ValueError(f"Método PEFT no soportado: {method}")
        
        # Cargar el modelo PEFT
        model = PeftModel.from_pretrained(model, model_path)
        
        # Mover el modelo al dispositivo especificado
        if device == "cpu":
            model = model.to(device)
        
        # Configurar el modelo para evaluación
        model.eval()
        
        logger.info(f"Modelo {method} cargado exitosamente")
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Error al cargar el modelo {method}: {str(e)}")
        raise

def load_eval_dataset(
    dataset_path: str,
    split: str = "test",
    max_samples: Optional[int] = None,
    format: str = "auto"
) -> Dataset:
    """
    Carga el conjunto de datos de evaluación.
    
    Args:
        dataset_path: Ruta al conjunto de datos
        split: Split del dataset a utilizar
        max_samples: Número máximo de muestras a cargar
        format: Formato del dataset ('json', 'csv', 'jsonl', 'auto')
        
    Returns:
        Dataset de evaluación
    """
    logger.info(f"Cargando dataset de evaluación desde {dataset_path}")
    
    try:
        # Determinar el formato si es 'auto'
        if format == "auto":
            if dataset_path.endswith('.json'):
                format = 'json'
            elif dataset_path.endswith('.jsonl'):
                format = 'json'
            elif dataset_path.endswith('.csv'):
                format = 'csv'
            else:
                raise ValueError(f"No se pudo determinar el formato del dataset: {dataset_path}")
        
        # Cargar el dataset
        dataset = load_dataset(format, data_files=dataset_path, split=split)
        
        # Limitar el número de muestras si es necesario
        if max_samples is not None and max_samples > 0:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        logger.info(f"Dataset cargado con {len(dataset)} muestras")
        return dataset
    
    except Exception as e:
        logger.error(f"Error al cargar el dataset: {str(e)}")
        raise

def prepare_inputs(
    tokenizer: PreTrainedTokenizer,
    text: str,
    max_length: int = DEFAULT_MAX_LENGTH,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Prepara las entradas para el modelo.
    
    Args:
        tokenizer: Tokenizador del modelo
        text: Texto a tokenizar
        max_length: Longitud máxima de la secuencia
        device: Dispositivo para los tensores
        
    Returns:
        Diccionario con los tensores de entrada
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    )
    
    # Mover los tensores al dispositivo
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    return inputs

def compute_metrics(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    batch_size: int = 8,
    max_length: int = DEFAULT_MAX_LENGTH,
    metrics: List[str] = DEFAULT_METRICS,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, float]:
    """
    Calcula las métricas de evaluación para un modelo.
    
    Args:
        model: Modelo a evaluar
        tokenizer: Tokenizador del modelo
        dataset: Dataset de evaluación
        batch_size: Tamaño del batch
        max_length: Longitud máxima de la secuencia
        metrics: Lista de métricas a calcular
        device: Dispositivo para la evaluación
        
    Returns:
        Diccionario con las métricas calculadas
    """
    logger.info(f"Calculando métricas: {', '.join(metrics)}")
    
    results = {metric: 0.0 for metric in metrics}
    total_samples = 0
    
    # Verificar si el dataset tiene las columnas necesarias
    required_columns = ["input", "output"] if "accuracy" in metrics else ["input"]
    missing_columns = [col for col in required_columns if col not in dataset.column_names]
    
    if missing_columns:
        logger.warning(f"El dataset no tiene las columnas requeridas: {missing_columns}")
        if "accuracy" in metrics:
            metrics.remove("accuracy")
            logger.warning("Se ha eliminado 'accuracy' de las métricas a calcular")
    
    # Procesar el dataset en batches
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluando"):
        batch = dataset[i:i+batch_size]
        batch_size_actual = len(batch["input"])
        total_samples += batch_size_actual
        
        # Calcular métricas para cada ejemplo en el batch
        for j in range(batch_size_actual):
            input_text = batch["input"][j]
            
            # Tokenizar la entrada
            inputs = prepare_inputs(tokenizer, input_text, max_length, device)
            input_ids = inputs["input_ids"]
            
            # Calcular la pérdida
            with torch.no_grad():
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss.item()
            
            # Actualizar métricas
            if "loss" in metrics:
                results["loss"] += loss
            
            if "perplexity" in metrics:
                results["perplexity"] += math.exp(loss)
            
            # Calcular exactitud si está disponible
            if "accuracy" in metrics and "output" in batch:
                output_text = batch["output"][j]
                
                # Generar la respuesta del modelo
                with torch.no_grad():
                    generated_ids = model.generate(
                        input_ids=input_ids,
                        max_length=max_length,
                        num_return_sequences=1,
                        do_sample=False
                    )
                
                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                # Comparar con la respuesta esperada
                exact_match = generated_text.strip() == output_text.strip()
                results["accuracy"] += float(exact_match)
    
    # Calcular promedios
    for metric in results:
        results[metric] /= total_samples
    
    logger.info(f"Métricas calculadas: {results}")
    return results

def evaluate_model(
    method: str,
    model_path: str,
    dataset: Dataset,
    base_model_name: str = DEFAULT_BASE_MODEL,
    batch_size: int = 8,
    max_length: int = DEFAULT_MAX_LENGTH,
    metrics: List[str] = DEFAULT_METRICS,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    token: Optional[str] = None
) -> Dict[str, float]:
    """
    Evalúa un modelo con un método PEFT específico.
    
    Args:
        method: Método PEFT utilizado
        model_path: Ruta al modelo
        dataset: Dataset de evaluación
        base_model_name: Nombre del modelo base
        batch_size: Tamaño del batch
        max_length: Longitud máxima de la secuencia
        metrics: Lista de métricas a calcular
        device: Dispositivo para la evaluación
        token: Token de HuggingFace
        
    Returns:
        Diccionario con las métricas calculadas
    """
    logger.info(f"Iniciando evaluación del modelo {method} en {model_path}")
    
    try:
        # Cargar el modelo y el tokenizador
        model, tokenizer = load_model(method, model_path, base_model_name, device, token)
        
        # Calcular métricas
        results = compute_metrics(
            model, tokenizer, dataset, batch_size, max_length, metrics, device
        )
        
        # Liberar memoria
        del model
        torch.cuda.empty_cache() if device == "cuda" else None
        
        return results
    
    except Exception as e:
        logger.error(f"Error al evaluar el modelo {method}: {str(e)}")
        return {metric: float('nan') for metric in metrics}

def run_inference(
    method: str,
    model_path: str,
    prompts: List[str],
    base_model_name: str = DEFAULT_BASE_MODEL,
    max_length: int = DEFAULT_MAX_LENGTH,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    token: Optional[str] = None,
    generation_kwargs: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Realiza inferencia con un modelo fine-tuned.
    
    Args:
        method: Método PEFT utilizado
        model_path: Ruta al modelo
        prompts: Lista de prompts para inferencia
        base_model_name: Nombre del modelo base
        max_length: Longitud máxima de la secuencia
        device: Dispositivo para la inferencia
        token: Token de HuggingFace
        generation_kwargs: Parámetros adicionales para la generación
        
    Returns:
        Lista de respuestas generadas
    """
    logger.info(f"Realizando inferencia con el modelo {method}")
    
    try:
        # Cargar el modelo y el tokenizador
        model, tokenizer = load_model(method, model_path, base_model_name, device, token)
        
        # Configurar parámetros de generación
        gen_kwargs = {
            "max_length": max_length,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "num_return_sequences": 1
        }
        
        # Actualizar con parámetros personalizados
        if generation_kwargs:
            gen_kwargs.update(generation_kwargs)
        
        # Generar respuestas para cada prompt
        responses = []
        for prompt in tqdm(prompts, desc="Generando respuestas"):
            inputs = prepare_inputs(tokenizer, prompt, max_length, device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **gen_kwargs
                )
            
            response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            responses.append(response)
        
        # Liberar memoria
        del model
        torch.cuda.empty_cache() if device == "cuda" else None
        
        return responses
    
    except Exception as e:
        logger.error(f"Error al realizar inferencia con el modelo {method}: {str(e)}")
        return ["Error: " + str(e)] * len(prompts)

def compare_results(
    results: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compara los resultados de evaluación de diferentes modelos.
    
    Args:
        results: Diccionario con los resultados de cada modelo
        output_path: Ruta para guardar el informe
        
    Returns:
        Diccionario con los resultados comparativos
    """
    logger.info("Comparando resultados de los modelos")
    
    # Crear DataFrame para la comparación
    df = pd.DataFrame(results).T
    
    # Encontrar el mejor modelo para cada métrica
    best_models = {}
    for metric in df.columns:
        if metric == "loss" or metric == "perplexity":
            best_model = df[metric].idxmin()
            best_value = df[metric].min()
        else:
            best_model = df[metric].idxmax()
            best_value = df[metric].max()
        
        best_models[metric] = {"model": best_model, "value": best_value}
    
    # Calcular mejoras relativas
    comparative_results = {
        "metrics": df.to_dict(),
        "best_models": best_models
    }
    
    # Imprimir resultados
    print("\n" + "="*50)
    print("RESULTADOS DE LA EVALUACIÓN")
    print("="*50)
    print(df.round(4))
    print("\nMejores modelos por métrica:")
    for metric, info in best_models.items():
        print(f"  - {metric}: {info['model']} ({info['value']:.4f})")
    print("="*50 + "\n")
    
    # Guardar resultados si se especifica una ruta
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(comparative_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Resultados guardados en {output_path}")
    
    return comparative_results

def parse_arguments():
    """
    Parsea los argumentos de línea de comandos.
    
    Returns:
        Argumentos parseados
    """
    parser = argparse.ArgumentParser(
        description="Evaluación de modelos fine-tuned con diferentes métodos PEFT"
    )
    
    # Argumentos para los modelos
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f"Modelo base (default: {DEFAULT_BASE_MODEL})"
    )
    parser.add_argument(
        "--qlora-path",
        type=str,
        help="Ruta al modelo QLoRA"
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Ruta al modelo Adapter"
    )
    parser.add_argument(
        "--ptuning-path",
        type=str,
        help="Ruta al modelo P-Tuning"
    )
    
    # Argumentos para el dataset
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Ruta al dataset de evaluación"
    )
    parser.add_argument(
        "--dataset-format",
        type=str,
        default="auto",
        choices=["auto", "json", "jsonl", "csv"],
        help="Formato del dataset (default: auto)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split del dataset a utilizar (default: test)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Número máximo de muestras a evaluar (default: todas)"
    )
    
    # Argumentos para la evaluación
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=DEFAULT_METRICS,
        help=f"Métricas a calcular (default: {' '.join(DEFAULT_METRICS)})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Tamaño del batch (default: 8)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help=f"Longitud máxima de la secuencia (default: {DEFAULT_MAX_LENGTH})"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Dispositivo para la evaluación (default: cuda si está disponible, sino cpu)"
    )
    
    # Argumentos para la inferencia
    parser.add_argument(
        "--run-inference",
        action="store_true",
        help="Ejecutar inferencia con los modelos"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        help="Prompts para inferencia"
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        help="Archivo con prompts para inferencia (un prompt por línea)"
    )
    
    # Argumentos para la salida
    parser.add_argument(
        "--output",
        type=str,
        default=f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        help="Ruta para guardar el informe de evaluación"
    )
    
    # Token de HuggingFace
    parser.add_argument(
        "--hf-token",
        type=str,
        help="Token de HuggingFace para modelos privados"
    )
    
    return parser.parse_args()

def main():
    """Función principal del script."""
    # Parsear argumentos
    args = parse_arguments()
    
    # Verificar que al menos un modelo está especificado
    model_paths = {
        "qlora": args.qlora_path,
        "adapter": args.adapter_path,
        "ptuning": args.ptuning_path
    }
    
    specified_models = [m for m, p in model_paths.items() if p is not None]
    if not specified_models:
        logger.error("Debe especificar al menos un modelo para evaluar")
        return
    
    logger.info(f"Modelos a evaluar: {', '.join(specified_models)}")
    
    # Cargar el dataset de evaluación
    dataset = load_eval_dataset(
        args.dataset,
        args.split,
        args.max_samples,
        args.dataset_format
    )
    
    # Evaluar cada modelo especificado
    results = {}
    for method, path in model_paths.items():
        if path is not None:
            logger.info(f"Evaluando modelo {method} en {path}")
            results[method] = evaluate_model(
                method=method,
                model_path=path,
                dataset=dataset,
                base_model_name=args.base_model,
                batch_size=args.batch_size,
                max_length=args.max_length,
                metrics=args.metrics,
                device=args.device,
                token=args.hf_token
            )
    
    # Comparar resultados
    output_path = os.path.join(os.path.dirname(__file__), "..", args.output)
    compare_results(results, output_path)
    
    # Realizar inferencia si se solicita
    if args.run_inference:
        prompts = []
        
        # Cargar prompts desde argumentos o archivo
        if args.prompts:
            prompts.extend(args.prompts)
        
        if args.prompts_file:
            try:
                with open(args.prompts_file, 'r', encoding='utf-8') as f:
                    file_prompts = [line.strip() for line in f if line.strip()]
                    prompts.extend(file_prompts)
            except Exception as e:
                logger.error(f"Error al cargar prompts desde archivo: {str(e)}")
        
        if not prompts:
            logger.warning("No se especificaron prompts para inferencia")
            return
        
        # Realizar inferencia con cada modelo
        inference_results = {}
        for method, path in model_paths.items():
            if path is not None:
                logger.info(f"Realizando inferencia con modelo {method}")
                responses = run_inference(
                    method=method,
                    model_path=path,
                    prompts=prompts,
                    base_model_name=args.base_model,
                    max_length=args.max_length,
                    device=args.device,
                    token=args.hf_token
                )
                
                inference_results[method] = {
                    f"prompt_{i}": {
                        "prompt": prompt,
                        "response": response
                    }
                    for i, (prompt, response) in enumerate(zip(prompts, responses))
                }
        
        # Guardar resultados de inferencia
        inference_output = os.path.join(
            os.path.dirname(__file__),
            "..",
            f"inference_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(inference_output, 'w', encoding='utf-8') as f:
            json.dump(inference_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Resultados de inferencia guardados en {inference_output}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    logger.info(f"Evaluación completada en {elapsed_time:.2f} segundos")
