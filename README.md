# Proyecto: Reconocimiento de comandos de audio (repo: Cinthia-hub/audio)

Este repositorio contiene scripts para:
- Grabar audio (interactivo o batch),
- Extraer features acústicas (MFCC, RMS, chroma, etc.),
- Entrenar un modelo simple con TensorFlow/Keras,
- Evaluar y visualizar resultados,
- Hacer inferencia en tiempo real desde el micrófono.

Contenido principal
- create.py: grabador (interactivo y batch) + extracción y guardado de features en CSV.
- get_feactures_no_silent.py: extrae features desde una estructura de carpetas (genera features_test.csv).
- quick_train.py: entrenamiento rápido (prueba).
- model.py: entrenamiento "completo" y generación de reportes y figura.
- try_model.py: prueba de un WAV individual (muestra predicciones y, si existe, métricas globales).
- try_model2.py: evaluación global sobre features_test.csv (matriz de confusión y figura de reporte).
- save_results.py: evaluación y guardado de reporte en .png y .txt.
- real_time.py: inferencia en vivo (pulsa 'g' para grabar y predecir).
- segmentar_audio.py: corta un WAV de ruido en segmentos aumentados (augmentación).
- audio.py: script de visualización de MFCC/RMS para un WAV de ejemplo.
- test_env.py, test_microfo.py: pruebas rápidas del entorno y del micrófono.
- requirements.txt: dependencias (instalar con pip).
  
Instalación rápida (con virtualenv)
1. Crear y activar virtualenv:
   - python -m venv .venv
   - Windows: .venv\Scripts\activate
2. Instalar dependencias:
   - pip install -r requirements.txt

### Flujo recomendado (paso a paso) — comandos listos
1. Verificar TensorFlow / GPU:
   - python test_env.py

2. Probar grabación de micrófono:
   - python test_microfo.py

3. Crear datos (carpetas y grabaciones)
   - Crear carpetas por defecto (no graba): 
     - python create.py
   - Modo interactivo (graba y añade features al CSV):
     - python create.py --interactive --duration 2 --sr 16000
   - Modo batch (graba N muestras por cada clase):
     - python create.py --count 3 --duration 2
   - Nota: create.py por defecto usa `pruebas_audio/` como carpeta raíz y `features_test.csv` como CSV de salida.

4. Alternativa: extraer features desde una estructura de carpetas ya existente:
   - python get_feactures_no_silent.py
   - Resultado: features_test.csv

5. Entrenamiento rápido (prueba):
   - Asegúrate de tener `features.csv` en el directorio.
   - python quick_train.py
   - Guarda: `scaler.pkl`, `label_classes.npy`, `modelo_audio.keras`.

6. Entrenamiento completo y reportes:
   - python model.py
   - Entrenamiento por defecto en ese script: epochs=100. Guarda modelos y reportes (classification_report_model.txt, single_report_model.png...).

7. Evaluación y generación de métricas gráficas:
   - python try_model2.py
   - (o) python save_results.py
   - Ambos requieren: `features_test.csv`, `scaler.pkl`, `modelo_audio.keras`. `label_classes.npy` es opcional (si existe se usa).

8. Probar un WAV concreto:
   - Edita `ARCHIVO_AUDIO` en try_model.py o coloca un archivo en la ruta por defecto.
   - python try_model.py

9. Inferencia en tiempo real (desde micrófono):
   - python real_time.py
   - Pulsa 'g' para grabar 2s y obtener predicción; ESC para salir.
   - Requiere entorno con GUI (OpenCV) para mostrar ventanas.

Explicación breve de artefactos esperados
- features.csv: CSV con features usado para entrenamiento (columnas feat_0..feat_N y columna label).
- features_test.csv: CSV para evaluación (generado por get_feactures_no_silent.py o create.py).
- scaler.pkl: StandardScaler guardado durante el entrenamiento (obligatorio para inferencia correcta).
- modelo_audio.keras: archivo del modelo Keras entrenado.
- label_classes.npy: arreglo con orden de clases (útil para mapear índices a etiquetas).

Descripción rápida de cada script (uso)
- test_env.py: comprobar TF y GPUs -> python test_env.py
- test_microfo.py: grabar y reproducir audio de prueba -> python test_microfo.py
- create.py: grabar y extraer features por clase -> python create.py --interactive
- segmentar_audio.py: crear segmentos de ruido aumentados -> python segmentar_audio.py
- get_feactures_no_silent.py: extracción masiva de features -> python get_feactures_no_silent.py
- quick_train.py: entrenamiento rápido desde features.csv -> python quick_train.py
- model.py: entrenamiento completo y reportes -> python model.py
- try_model.py: prueba de un WAV y (opcional) métricas globales -> python try_model.py
- try_model2.py: evaluación completa sobre features_test.csv -> python try_model2.py
- save_results.py: evaluación y guardado de reportes -> python save_results.py
- real_time.py: inferencia en vivo desde micrófono -> python real_time.py
- audio.py: visualización MFCC/RMS de un WAV -> python audio.py
- check_model.py: listar archivos y verificar existencia de modelos -> python check_model.py
