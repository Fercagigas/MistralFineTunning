#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para preparar y procesar el dataset médico para fine-tuning con PEFT.

Este script:
1. Carga el dataset "ruslanmv/ai-medical-chatbot"
2. Procesa las columnas 'question' y 'answer' para formatearlas adecuadamente
3. Divide el dataset en conjuntos de entrenamiento y evaluación
4. Implementa una función para tokenizar los datos
5. Guarda el dataset procesado para su uso en los scripts de fine-tuning
"""

import os
import argparse
import logging
from typing import Dict, Optional, Union

import datasets
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizer

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_dataset_from_hub(dataset_name: str, token: Optional[str] = None) -> DatasetDict:
    """
    Carga un dataset desde Hugging Face Hub.
    
    Args:
        dataset_name: Nombre del dataset en Hugging Face Hub
        token: Token de autenticación de Hugging Face (opcional)
    
    Returns:
        Dataset cargado
    """
    logger.info(f"Cargando dataset: {dataset_name}")
    
    # Si se proporciona un token, usarlo para la autenticación
    if token:
        os.environ["HUGGINGFACE_TOKEN"] = token
    
    try:
        # Cargar el dataset
        dataset = datasets.load_dataset(dataset_name)
        logger.info(f"Dataset cargado exitosamente: {dataset}")
        return dataset
    except Exception as e:
        logger.error(f"Error al cargar el dataset: {e}")
        raise

def format_medical_dataset(dataset: Union[Dataset, DatasetDict]) -> DatasetDict:
    """
    Formatea el dataset médico para el fine-tuning.
    
    Args:
        dataset: Dataset original con columnas 'question' y 'answer'
    
    Returns:
        Dataset formateado con columnas 'input_text' y 'target_text'
    """
    logger.info("Formateando dataset médico")
    
    # Si el dataset es un DatasetDict, procesamos cada split
    if isinstance(dataset, DatasetDict):
        formatted_dataset = DatasetDict()
        for split, data in dataset.items():
            formatted_dataset[split] = format_medical_dataset(data)
        return formatted_dataset
    
    # Verificar que las columnas necesarias existen
    required_columns = ['question', 'answer']
    for col in required_columns:
        if col not in dataset.column_names:
            raise ValueError(f"El dataset debe contener la columna '{col}'")
    
    # Formatear el dataset para el fine-tuning
    def format_example(example):
        # Formatear la entrada como una instrucción para el modelo
        input_text = f"<s>[INST] {example['question']} [/INST]"
        # La respuesta esperada
        target_text = f"{example['answer']}</s>"
        
        return {
            "input_text": input_text,
            "target_text": target_text,
            # Combinamos para tener el formato completo de instrucción-respuesta
            "full_text": f"{input_text} {target_text}"
        }
    
    # Aplicar la función de formato a todo el dataset
    formatted_dataset = dataset.map(format_example)
    logger.info("Dataset formateado exitosamente")
    
    return formatted_dataset

def split_dataset(dataset: Union[Dataset, DatasetDict], 
                 test_size: float = 0.1, 
                 seed: int = 42) -> DatasetDict:
    """
    Divide el dataset en conjuntos de entrenamiento y evaluación.
    
    Args:
        dataset: Dataset a dividir
        test_size: Proporción del dataset para evaluación (0.0-1.0)
        seed: Semilla para reproducibilidad
    
    Returns:
        DatasetDict con splits 'train' y 'validation'
    """
    logger.info(f"Dividiendo dataset (test_size={test_size}, seed={seed})")
    
    # Si ya es un DatasetDict, verificamos si ya tiene los splits necesarios
    if isinstance(dataset, DatasetDict):
        if 'train' in dataset and 'validation' not in dataset:
            # Si solo tiene 'train', dividimos ese
            train_val = dataset['train'].train_test_split(
                test_size=test_size, seed=seed
            )
            return DatasetDict({
                'train': train_val['train'],
                'validation': train_val['test']
            })
        elif 'train' in dataset and 'validation' in dataset:
            # Ya tiene los splits necesarios
            logger.info("El dataset ya contiene splits 'train' y 'validation'")
            return dataset
        else:
            # Caso especial: tiene otros splits, intentamos adaptarlos
            logger.warning(f"El dataset tiene splits inusuales: {list(dataset.keys())}")
            # Tomamos el primer split disponible y lo dividimos
            first_split = list(dataset.keys())[0]
            train_val = dataset[first_split].train_test_split(
                test_size=test_size, seed=seed
            )
            return DatasetDict({
                'train': train_val['train'],
                'validation': train_val['test']
            })
    
    # Si es un Dataset simple, lo dividimos
    train_val = dataset.train_test_split(test_size=test_size, seed=seed)
    return DatasetDict({
        'train': train_val['train'],
        'validation': train_val['test']
    })

def tokenize_dataset(dataset: DatasetDict, 
                    tokenizer: PreTrainedTokenizer,
                    max_length: int = 512,
                    text_column: str = "full_text") -> DatasetDict:
    """
    Tokeniza el dataset para el fine-tuning.
    
    Args:
        dataset: Dataset a tokenizar
        tokenizer: Tokenizador a utilizar
        max_length: Longitud máxima de secuencia
        text_column: Nombre de la columna que contiene el texto a tokenizar
    
    Returns:
        Dataset tokenizado
    """
    logger.info(f"Tokenizando dataset (max_length={max_length})")
    
    def tokenize_function(examples):
        # Tokenizamos los textos
        tokenized = tokenizer(
            examples[text_column],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors=None  # Para que devuelva listas en lugar de tensores
        )
        
        # Para el entrenamiento de modelos de lenguaje, los labels son los mismos input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # Aplicar la función de tokenización a todo el dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[col for col in dataset["train"].column_names if col != text_column]
    )
    
    logger.info("Dataset tokenizado exitosamente")
    return tokenized_dataset

def save_processed_dataset(dataset: DatasetDict, output_dir: str) -> None:
    """
    Guarda el dataset procesado en disco.
    
    Args:
        dataset: Dataset procesado
        output_dir: Directorio donde guardar el dataset
    """
    logger.info(f"Guardando dataset procesado en: {output_dir}")
    
    # Crear el directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar cada split por separado
    for split, data in dataset.items():
        split_dir = os.path.join(output_dir, split)
        data.save_to_disk(split_dir)
        logger.info(f"Split '{split}' guardado en: {split_dir}")

def prepare_dataset(
    dataset_name: str,
    model_name: str,
    output_dir: str,
    hf_token: Optional[str] = None,
    max_length: int = 512,
    test_size: float = 0.1
) -> None:
    """
    Función principal que prepara el dataset para fine-tuning.
    
    Args:
        dataset_name: Nombre del dataset en Hugging Face Hub
        model_name: Nombre del modelo para cargar el tokenizador
        output_dir: Directorio donde guardar el dataset procesado
        hf_token: Token de autenticación de Hugging Face
        max_length: Longitud máxima de secuencia para tokenización
        test_size: Proporción del dataset para evaluación
    """
    # 1. Cargar el dataset
    dataset = load_dataset_from_hub(dataset_name, token=hf_token)
    
    # 2. Formatear el dataset
    formatted_dataset = format_medical_dataset(dataset)
    
    # 3. Dividir el dataset
    split_dataset_dict = split_dataset(formatted_dataset, test_size=test_size)
    
    # 4. Cargar el tokenizador
    logger.info(f"Cargando tokenizador para el modelo: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        token=hf_token,
        trust_remote_code=True
    )
    
    # Asegurarse de que el tokenizador tenga los tokens especiales necesarios
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 5. Tokenizar el dataset
    tokenized_dataset = tokenize_dataset(
        split_dataset_dict, 
        tokenizer, 
        max_length=max_length
    )
    
    # 6. Guardar el dataset procesado
    save_processed_dataset(tokenized_dataset, output_dir)
    
    # También guardar el dataset formateado pero no tokenizado (útil para análisis)
    formatted_output_dir = os.path.join(os.path.dirname(output_dir), "formatted_data")
    save_processed_dataset(split_dataset_dict, formatted_output_dir)
    
    logger.info("Preparación del dataset completada exitosamente")

def main():
    """Punto de entrada principal cuando se ejecuta como script."""
    parser = argparse.ArgumentParser(
        description="Prepara y procesa un dataset médico para fine-tuning con PEFT"
    )
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ruslanmv/ai-medical-chatbot",
        help="Nombre del dataset en Hugging Face Hub"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Nombre del modelo para cargar el tokenizador"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./processed_data",
        help="Directorio donde guardar el dataset procesado"
    )
    
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Token de autenticación de Hugging Face"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Longitud máxima de secuencia para tokenización"
    )
    
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Proporción del dataset para evaluación (0.0-1.0)"
    )
    
    args = parser.parse_args()
    
    # Si no se proporciona un token, intentar obtenerlo de la variable de entorno
    if args.hf_token is None:
        args.hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    
    # Preparar el dataset
    prepare_dataset(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        output_dir=args.output_dir,
        hf_token=args.hf_token,
        max_length=args.max_length,
        test_size=args.test_size
    )

if __name__ == "__main__":
    main()
