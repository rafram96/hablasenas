"""
process_clip.py (refactorizado)

Reemplaza el procesamiento batch de clips por un flujo interactivo de captura de muestras.
"""
from src.interactive_data_collector import collect_samples

if __name__ == '__main__':
    project_root = __import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__file__), '..'))
    output_root = __import__('os').path.join(project_root, 'data', 'features')
    while True:
        label = input("Ingrese etiqueta (letra/palabra) o ENTER para salir: ")
        if not label:
            print("Saliendo del pipeline interactivo.")
            break
        try:
            ms = int(input("Número de muestras (por defecto 50): ") or 50)
        except ValueError:
            ms = 50
        try:
            th = float(input("Umbral ratio (por defecto 0.2): ") or 0.2)
        except ValueError:
            th = 0.2
        collect_samples(label, output_root, ms, th)
