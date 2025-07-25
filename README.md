# Clasificador Inteligente de Imágenes de Animales (TensorFlow 2) 

---

1.  **Configura el entorno virtual e instala las dependencias:**
    Puedes usar `pip` con `requirements.txt` o `conda` con `environment.yml`. Se recomienda usar entornos virtuales para evitar conflictos de dependencias.

    **Opción 1: Usando `requirements.txt` (recomendado con `venv`)**
    ```bash
    py -3.10 -m venv tf-env
    .\tf-env\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

    **Opción 2: Usando `environment.yml` (uso exclusivo con Anaconda)**
    ```bash
    conda env create -f environment.yml
    conda activate clasificador-imagenes # 'clasificador-imagenes' es el nombre del entorno definido en environment.yml
    ```
---
