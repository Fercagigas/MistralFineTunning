# PEFT Fine-tuning para Mistral-7B-Instruct-v0.2

Este proyecto implementa diferentes técnicas de Parameter-Efficient Fine-Tuning (PEFT) para adaptar el modelo Mistral-7B-Instruct-v0.2 a un dominio médico específico. Las técnicas implementadas son QLoRA, Adapter Tuning y P-Tuning, que permiten ajustar modelos de gran tamaño con recursos computacionales limitados.

## Descripción General

El fine-tuning de modelos de lenguaje de gran tamaño (LLMs) como Mistral-7B-Instruct-v0.2 tradicionalmente requiere recursos computacionales significativos. Las técnicas PEFT abordan este desafío al ajustar solo un pequeño subconjunto de parámetros, manteniendo el rendimiento mientras se reduce drásticamente la huella de memoria y los requisitos computacionales.

Este proyecto proporciona:

- Scripts para preparar y procesar datos médicos para fine-tuning
- Implementaciones de tres técnicas PEFT populares: QLoRA, Adapter Tuning y P-Tuning
- Herramientas de evaluación para comparar el rendimiento de los modelos ajustados
- Un pipeline completo que puede ejecutarse con un solo comando

## Estructura del Proyecto

```
peft-finetune/
├── qlora/               # Carpeta para QLoRA
│   └── train_qlora.py    # Script para QLoRA fine-tuning
├── adapter_tuning/      # Carpeta para Adapter Tuning
│   └── train_adapter.py  # Script para Adapter Tuning
├── p_tuning/            # Carpeta para P-Tuning
│   └── train_ptuning.py  # Script para P-Tuning
├── models/              # Carpeta para guardar los modelos
├── utils/               # Carpeta para utilidades comunes
│   ├── common.py          # Funciones de utilidad comunes
│   ├── prepare_dataset.py # Script de preparación de datos
│   └── evaluate_models.py # Script de evaluación
├── processed_data/      # Carpeta donde se guardarán los datos procesados
├── run_all.py           # Script para ejecutar todo el pipeline
└── requirements.txt     # Dependencias del proyecto
```

## Instalación y Configuración

### Requisitos Previos

- Python 3.8 o superior
- CUDA compatible con PyTorch (para aceleración GPU)
- Al menos 16GB de RAM (se recomienda 32GB)
- Al menos 8GB de VRAM para GPU (se recomienda 16GB+)

### Instalación

1. Clone el repositorio:
   ```bash
   git clone https://github.com/usuario/peft-finetune.git
   cd peft-finetune
   ```

2. Cree un entorno virtual y actívelo:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. Instale las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure su token de Hugging Face:
   ```bash
   export HF_TOKEN="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"
   ```
   
   O añádalo a su archivo `.bashrc` o `.zshrc`:
   ```bash
   echo 'export HF_TOKEN="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"' >> ~/.bashrc
   source ~/.bashrc
   ```

## Uso

### Preparación de Datos

El script `utils/prepare_dataset.py` procesa y prepara los datos médicos para el fine-tuning:

```bash
python utils/prepare_dataset.py \
    --input_data path/to/medical/data.csv \
    --output_dir processed_data \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --max_length 512 \
    --train_size 0.8
```

Argumentos:
- `--input_data`: Ruta al archivo de datos médicos (CSV, JSON, etc.)
- `--output_dir`: Directorio donde se guardarán los datos procesados
- `--model_name`: Nombre del modelo base para tokenización
- `--max_length`: Longitud máxima de secuencia para tokenización
- `--train_size`: Proporción de datos para entrenamiento (el resto para validación)

### Fine-tuning con QLoRA

QLoRA (Quantized Low-Rank Adaptation) combina cuantización y adaptadores de bajo rango para un fine-tuning eficiente:

```bash
python qlora/train_qlora.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --data_path processed_data \
    --output_dir models/qlora \
    --num_epochs 3 \
    --learning_rate 2e-4 \
    --batch_size 8 \
    --lora_r 16 \
    --lora_alpha 32
```

Argumentos principales:
- `--model_name`: Modelo base a ajustar
- `--data_path`: Ruta a los datos procesados
- `--output_dir`: Directorio para guardar el modelo ajustado
- `--lora_r`: Rango de las matrices de adaptación LoRA
- `--lora_alpha`: Factor de escalado LoRA

### Fine-tuning con Adapter Tuning

Adapter Tuning inserta pequeñas capas adaptadoras entre las capas existentes del modelo:

```bash
python adapter_tuning/train_adapter.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --data_path processed_data \
    --output_dir models/adapter \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --batch_size 8 \
    --adapter_size 64
```

Argumentos principales:
- `--adapter_size`: Dimensión de las capas adaptadoras

### Fine-tuning con P-Tuning

P-Tuning optimiza embeddings continuos de prompt en lugar de los parámetros del modelo:

```bash
python p_tuning/train_ptuning.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --data_path processed_data \
    --output_dir models/ptuning \
    --num_epochs 3 \
    --learning_rate 5e-5 \
    --batch_size 8 \
    --num_virtual_tokens 20
```

Argumentos principales:
- `--num_virtual_tokens`: Número de tokens virtuales para P-Tuning

### Evaluación de Modelos

Para evaluar y comparar los modelos ajustados:

```bash
python utils/evaluate_models.py \
    --base_model mistralai/Mistral-7B-Instruct-v0.2 \
    --qlora_model models/qlora \
    --adapter_model models/adapter \
    --ptuning_model models/ptuning \
    --test_data processed_data/test.json \
    --output_file evaluation_results.json
```

### Ejecutar Todo el Pipeline

Para ejecutar todo el proceso (preparación de datos, fine-tuning con los tres métodos y evaluación):

```bash
python run_all.py \
    --input_data path/to/medical/data.csv \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --num_epochs 3 \
    --batch_size 8
```

## Métodos PEFT Implementados

### QLoRA (Quantized Low-Rank Adaptation)

QLoRA combina la cuantización de 4 bits con adaptadores de bajo rango para reducir significativamente los requisitos de memoria. Características principales:

- Cuantiza el modelo base a 4 bits
- Añade matrices de bajo rango (LoRA) que se entrenan en precisión completa
- Utiliza doble cuantización para reducir aún más el uso de memoria
- Implementa descomposición de valor singular (SVD) para inicialización eficiente

Ventajas:
- Reduce drásticamente los requisitos de VRAM (hasta 70%)
- Mantiene la calidad del modelo original
- Permite fine-tuning en GPUs de consumo (8GB VRAM)

### Adapter Tuning

Adapter Tuning inserta pequeñas capas adaptadoras entrenables entre las capas del modelo pre-entrenado:

- Mantiene los pesos originales del modelo congelados
- Añade pequeñas capas adaptadoras (bottleneck) después de ciertas capas
- Solo entrena estos adaptadores, que típicamente contienen <1% de los parámetros originales

Ventajas:
- Muy eficiente en memoria
- Modular: diferentes adaptadores pueden especializarse en diferentes tareas
- Fácil de intercambiar adaptadores sin recargar el modelo base

### P-Tuning

P-Tuning optimiza embeddings continuos de prompt en lugar de ajustar los parámetros del modelo:

- Introduce "tokens virtuales" continuos que se optimizan durante el entrenamiento
- Mantiene el modelo base completamente congelado
- Solo entrena los embeddings de estos tokens virtuales

Ventajas:
- Extremadamente eficiente en parámetros (solo se entrenan los embeddings)
- Efectivo para tareas específicas donde el contexto es importante
- Requiere mínima memoria para entrenamiento

## Comparación y Selección de Método

La elección del método PEFT depende de varios factores:

1. **Recursos disponibles**:
   - QLoRA: Requiere más VRAM que otros métodos, pero menos que el fine-tuning completo
   - Adapter Tuning: Muy eficiente en memoria
   - P-Tuning: El más eficiente en memoria

2. **Tipo de tarea**:
   - QLoRA: Mejor para adaptación general a dominios específicos
   - Adapter Tuning: Bueno para múltiples tareas relacionadas
   - P-Tuning: Excelente para tareas de clasificación y generación guiada

3. **Rendimiento**:
   - QLoRA: Generalmente proporciona el mejor rendimiento entre los métodos PEFT
   - Adapter Tuning: Buen equilibrio entre eficiencia y rendimiento
   - P-Tuning: Puede tener limitaciones en tareas complejas

El script de evaluación `utils/evaluate_models.py` proporciona métricas comparativas para ayudar a seleccionar el método más adecuado para su caso de uso específico.

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abra un issue para discutir cambios importantes antes de enviar un pull request.

## Licencia

Este proyecto está licenciado bajo la licencia MIT - vea el archivo LICENSE para más detalles.
