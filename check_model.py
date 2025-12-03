import os
print("cwd:", os.getcwd())
print("Archivos en el directorio actual:")
for f in os.listdir('.'):
    print(" -", f)
print("\nExiste 'modelo_audio.keras'?:", os.path.exists("modelo_audio.keras"))
print("Existe 'modelo_audio' (SavedModel carpeta)?:", os.path.exists("modelo_audio"))
print("Existe 'modelo_audio.h5'?:", os.path.exists("modelo_audio.h5"))