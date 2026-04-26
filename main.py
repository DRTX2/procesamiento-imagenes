import sys
from PySide6.QtWidgets import QApplication
from interfaz import VentanaPrincipal
from estado import EstadoPreprocesamiento
import modelo

def main():
    app = QApplication(sys.argv)
    
    # 1. Crear el Estado (ViewModel)
    estado = EstadoPreprocesamiento()
    
    # 2. Buscar imagen inicial y cargarla en el estado
    ruta_inicial = modelo.buscar_imagen_inicial()
    if ruta_inicial:
        estado.establecer_imagen(ruta_inicial)
    
    # 3. Crear la Vista (View) inyectando el estado
    ventana = VentanaPrincipal(estado)
    ventana.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
