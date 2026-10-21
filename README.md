# Detector Automático de Grooming con Naive Bayes 🛡️🤖

Script en Python que implementa un modelo de Aprendizaje Automático supervisado utilizando **scikit-learn** para la detección automática de patrones de *grooming* (acoso a menores en línea) a partir de mensajes de texto en plataformas de comunicación y videojuegos.

## Descripción del Proyecto

El *grooming* en entornos digitales (videojuegos, redes sociales y mensajería instantánea) representa un riesgo significativo para la seguridad infantil. Ante el volumen masivo de interacciones diarias, la supervisión manual es inviable.

Este proyecto implementa un pipeline de Procesamiento de Lenguaje Natural (PLN) capaz de analizar el contenido textual e identificar mensajes sospechosos en tiempo real mediante clasificación supervisada.

### Flujo de Trabajo (Pipeline):

- **Vectorización y Extracción de Características (Bag of Words):** Conversión del texto a matriz numérica de frecuencias de palabras utilizando `CountVectorizer`.
- **División del Dataset (Train/Test Split):** Separación de los datos etiquetados en un 70% para entrenamiento y un 30% para pruebas (`random_state=42`).
- **Entrenamiento del Modelo:** Clasificador probabilístico **Multinomial Naive Bayes** (`MultinomialNB`).
- **Evaluación de Desempeño:** Medición del modelo con *Accuracy*, *Precision*, *Recall*, *F1-Score* y Matriz de Confusión.
- **Inferencia en Tiempo Real:** Clasificación de nuevas frases no vistas durante el entrenamiento para validar la capacidad de detección.

## Requisitos Técnicos

Para ejecutar este proyecto de manera local, asegúrate de tener instalado **Python 3.x** y las siguientes librerías de procesamiento de datos y aprendizaje automático:

- `pandas`
- `scikit-learn`

## Instalación de Dependencias

Puedes instalar los paquetes requeridos ejecutando el siguiente comando en tu terminal:

```bash
pip install pandas scikit-learn
```

## Cómo Ejecutar el Script

1. Clona este repositorio o descarga el archivo principal del proyecto.
2. Abre tu terminal o entorno de desarrollo en la carpeta raíz del archivo.
3. Ejecuta el script con el siguiente comando:

```bash
python "tp_grupal_clasificador_bayesiano.py"
```

## Estructura del Código

El script está estructurado modularmente en las siguientes secciones lógicas:

- **1. Definición del Dataset:** Carga el conjunto de datos estructurado con mensajes etiquetados como grooming o no grooming.
- **2. Preprocesamiento y Vectorización:** Transforma los datos de texto utilizando la técnica *Bag of Words*.
- **3. División del Dataset:** Aplica `train_test_split` garantizando reproducibilidad.
- **4. Entrenamiento del Modelo:** Se entrena el algoritmo `MultinomialNB`.
- **5. Evaluación del Rendimiento:** Genera las métricas formales de desempeño en consola.
- **6. Inferencia:** Evalúa nuevos textos inéditos para verificar su detección.
