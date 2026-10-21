from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd

# ---------------------------
# 1. Definición del Dataset
# ---------------------------

data = {
    "mensajes escritos": [
        "¿Cuántos años tenés? No le digas a nadie que hablamos.",
        "Hola, ¿cómo estás? ¿Querés jugar algo online?",
        "Sos muy linda, no le muestres esto a tus papás.",
        "¿Terminaste la tarea de matemáticas?",
        "Podés mandarme una foto tuya, pero que sea solo para mí.",
        "¿Querés que te ayude con el juego que no podías pasar?",
        "No le contés a nadie que hablamos por acá, ¿sí?",
        "¿Te gustó la película de ayer?",
        "¿Estás solo en casa ahora?",
        "¿Querés hacer una videollamada para estudiar juntos?",
        "No le digas a tus padres que te escribí, es nuestro secreto.",
        "¿Cómo te fue en el examen de historia?",
        "Mandame una foto tuya, pero que nadie más la vea.",
        "¿Jugamos Minecraft esta tarde?",
        "Sos muy especial para mí, no le cuentes a nadie lo que hablamos."
        "¿Querés que te pase los apuntes de biología?",
        "¿Podés mostrarme cómo estás vestida ahora?",
        "¿Tenés ganas de salir a andar en bici mañana?",
        "No hace falta que le digas a nadie que hablamos tanto.",
        "¿Querés que estudiemos juntos para el parcial?"
    ],
    'label': [
        "grooming", 
        "no grooming", 
        "grooming", 
        "no grooming", 
        "grooming", 
        "no grooming", 
        "grooming", 
        "no grooming",
        "grooming",
        "no grooming",
        "grooming",
        "no grooming",
        "grooming",
        "no grooming",
        "grooming"
        "no grooming",
        "grooming",
        "no grooming",
        "grooming",
        "no grooming"
        ]
}

# Creación del DataFrame de trabajo
df = pd.DataFrame(data)

# ------------------------------------
# 2. Preprocesamiento y Vectorización 
# ------------------------------------
X = df['mensajes escritos']
y = df['label']


# Conversión del texto a matriz numérica de frecuencias
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# -----------------------------------------------
# 3. División del Dataset (Entrenamiento y Prueba 
# -----------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=42)

# ---------------------------
# 4. Entrenamiento del Modelo
# ---------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# -----------------------------
# 5. Evaluación del Rendimiento
# -----------------------------
y_pred = model.predict(X_test)

print("=" * 50)
print("Reporte de clasificación:")
print(" ")
print(classification_report(y_test, y_pred))

print("=" * 50)
print("Matriz de confusión:")
print(" ")
print(confusion_matrix(y_test, y_pred))


# -----------------------------------
# 6. Clasificación de Nuevos Mensajes
# -----------------------------------
nuevas_mensajes = [
    "¿Podés mandarme una foto tuya? No se la muestres a nadie.",
    "¿Jugamos Minecraft esta tarde?",
    "No le digas a tus papás que hablamos, ¿sí?",
    "¿Terminaste el trabajo práctico de historia?"
]

# Vectorización de los nuevos mensajes y predicción
nuevas_vectorizadas = vectorizer.transform(nuevas_mensajes)
predicciones = model.predict(nuevas_vectorizadas)

print("=" * 50)
print("Clasificación de nuevos mensajes:")
print(" ")

for mensaje, prediccion in zip(nuevas_mensajes, predicciones):
    print(f'Mensaje: "{mensaje}" → Clasificación: {prediccion}')

