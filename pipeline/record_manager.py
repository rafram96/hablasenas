"""
Script de utilidad para listar y eliminar grabaciones (.npy) y sus archivos de resumen,
además de actualizar automáticamente labels.json.
"""
import os
import json


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    features_dir = os.path.join(project_root, 'data', 'features')
    labels_file = os.path.join(features_dir, 'labels.json')

    while True:
        if not os.path.isfile(labels_file):
            print(f"No se encontró labels.json en: {labels_file}")
            break
        with open(labels_file, 'r', encoding='utf-8') as f:
            labels = json.load(f)

        if not labels:
            print("No hay grabaciones registradas en labels.json.")
            break

        print("\nGrabaciones registradas:")
        for idx, entry in enumerate(labels):
            print(f"[{idx}] {entry['filename']} -> etiqueta: {entry['label']}")

        selection = input("Ingrese índices a eliminar (separados por coma) o ENTER para cancelar: ")
        if not selection.strip():
            print("Operación cancelada.")
            break

        try:
            indices = sorted({int(i) for i in selection.split(',') if i.strip().isdigit()}, reverse=True)
        except ValueError:
            print("Entrada inválida.")
            break

        for idx in indices:
            if idx < 0 or idx >= len(labels):
                print(f"Índice fuera de rango: {idx}")
                continue
            entry = labels.pop(idx)
            rel_path = os.path.normpath(entry['filename'])
            npy_path = os.path.join(project_root, rel_path)
            summary_path = os.path.splitext(npy_path)[0] + '_summary.json'
            for path in (npy_path, summary_path):
                if os.path.isfile(path):
                    os.remove(path)
                    print(f"Eliminado: {path}")
                else:
                    print(f"No encontrado (omitido): {path}")
            base = os.path.splitext(os.path.basename(npy_path))[0]
            reports_dir = os.path.join(project_root, 'data', 'reports', entry['label'])
            report_path = os.path.join(reports_dir, f"{base}_report.json")
            if os.path.isfile(report_path):
                os.remove(report_path)
                print(f"Eliminado reporte: {report_path}")
            else:
                print(f"No se encontró reporte (omitido): {report_path}")

        with open(labels_file, 'w', encoding='utf-8') as f:
            json.dump(labels, f, indent=2, ensure_ascii=False)
        print("labels.json actualizado.")

        cont = input("¿Deseas gestionar más grabaciones? Presiona ENTER para salir o cualquier otra tecla para continuar: ")
        if not cont.strip():
            print("Finalizando gestor de grabaciones.")
            break

if __name__ == '__main__':
    main()
