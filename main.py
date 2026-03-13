import sys
from PyQt5.QtWidgets import QApplication
from src.gui import DiabetesGUI

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    gui = DiabetesGUI()
    gui.resize(500, 900)
    gui.show()
    
    sys.exit(app.exec_())