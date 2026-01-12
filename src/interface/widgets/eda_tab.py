import ast
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QComboBox, QPushButton, QScrollArea, QLineEdit, 
                             QFormLayout, QTextEdit, QFrame, QTabWidget, QListWidget)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from src.core.registry import Registry
from src.utils.imports import import_class

class EDATab(QTabWidget): # Ahora el Tab mismo es un contenedor de pestañas
    def __init__(self, session, style_kwargs=None):
        super().__init__()
        self.session = session
        self.style_kwargs = style_kwargs or {}
        
        # Estilos de plot (oscuro)
        self.c_bg = self.style_kwargs.get("figure.facecolor", "#2b2d30")
        self.c_ax = self.style_kwargs.get("axes.facecolor", "#1e1f22")
        self.c_text = self.style_kwargs.get("text.color", "#cfd8dc")
        self.c_spine = self.style_kwargs.get("axes.edgecolor", "#4e5154")

        self.init_ui()

    def init_ui(self):
        # --- PESTAÑA 1: CONFIGURACIÓN Y SELECCIÓN ---
        self.tab_config = QWidget()
        config_layout = QHBoxLayout(self.tab_config)
        
        # Columna Izquierda: Historial de EDA
        hist_panel = QFrame()
        hist_panel.setFixedWidth(250)
        hist_vbox = QVBoxLayout(hist_panel)
        hist_vbox.addWidget(QLabel("<b>Historial de Análisis:</b>"))
        self.list_history = QListWidget()
        self.list_history.itemClicked.connect(self.load_from_history)
        hist_vbox.addWidget(self.list_history)
        config_layout.addWidget(hist_panel)
        
        # Columna Derecha: Selección y Parámetros (Estilo Preprocessing)
        params_panel = QWidget()
        params_layout = QVBoxLayout(params_panel)
        
        # Selector de función
        f_select = QFormLayout()
        self.combo_analyzer = QComboBox()
        self.combo_analyzer.addItem("FalseMIAnalyzer", userData="src.eda.analyzers.falseMI_analyzer")
        self.combo_analyzer.currentIndexChanged.connect(self.on_analyzer_changed)
        f_select.addRow("Analizador:", self.combo_analyzer)
        params_layout.addLayout(f_select)
        
        # Formulario de parámetros (Estilo Preprocessing)
        self.scroll_config = QScrollArea()
        self.scroll_config.setWidgetResizable(True)
        self.scroll_config.setStyleSheet("background-color: #f8f9fa; border: 1px solid #dee2e6;")
        
        self.form_widget = QWidget()
        self.form_widget.setStyleSheet("background-color: #f8f9fa;")
        self.form_layout = QFormLayout(self.form_widget)
        self.scroll_config.setWidget(self.form_widget)
        params_layout.addWidget(self.scroll_config)
        
        self.btn_run = QPushButton("Ejecutar Análisis y Ver Resultados")
        self.btn_run.setFixedHeight(45)
        self.btn_run.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.btn_run.clicked.connect(self.run_analysis)
        params_layout.addWidget(self.btn_run)
        
        config_layout.addWidget(params_panel)

        # --- PESTAÑA 2: RESULTADOS Y VISUALIZACIÓN ---
        self.tab_results = QWidget()
        results_layout = QVBoxLayout(self.tab_results)
        
        # Sub-pestañas para Reporte vs Plots
        self.inner_results = QTabWidget()
        
        # Reporte
        self.txt_report = QTextEdit()
        self.txt_report.setReadOnly(True)
        self.txt_report.setStyleSheet(f"background-color: {self.c_ax}; color: {self.c_text}; font-family: monospace;")
        
        # Plots
        plot_container = QWidget()
        p_vbox = QVBoxLayout(plot_container)
        h_ctrl = QHBoxLayout()
        self.cb_plot_type = QComboBox()
        self.cb_plot_type.currentIndexChanged.connect(self.update_canvas)
        h_ctrl.addWidget(QLabel("Ver Distribución:"))
        h_ctrl.addWidget(self.cb_plot_type)
        h_ctrl.addStretch()
        p_vbox.addLayout(h_ctrl)
        
        self.figure = Figure(facecolor=self.c_bg, tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        p_vbox.addWidget(self.canvas)
        
        self.inner_results.addTab(self.txt_report, "Resumen Texto")
        self.inner_results.addTab(plot_container, "Gráficos")
        
        results_layout.addWidget(self.inner_results)

        # Añadir al Tab principal
        self.addTab(self.tab_config, "Configuración EDA")
        self.addTab(self.tab_results, "Resultados")

        self.inputs = {}
        self.history_data = [] # Guardará los reportes realizados
        self.on_analyzer_changed()

    def on_analyzer_changed(self):
        """Genera los campos basándose en el analizados seleccionado."""
        while self.form_layout.count():
            item = self.form_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.inputs = {}

        name = self.combo_analyzer.currentText()
        path = self.combo_analyzer.currentData()
        params = Registry.get_analyzer_params(name, path)

        for arg, val in params.items():
            lbl = QLabel(arg)
            lbl.setStyleSheet("color: #333; font-weight: bold;")
            le = QLineEdit(str(val))
            le.setStyleSheet("background-color: white; color: black; border: 1px solid #ccc; padding: 4px;")
            self.form_layout.addRow(lbl, le)
            self.inputs[arg] = le

    def run_analysis(self):
        if not self.session.current_data: return

        # 1. Parsear Kwargs
        kwargs = {}
        for arg, le in self.inputs.items():
            try: kwargs[arg] = ast.literal_eval(le.text())
            except: kwargs[arg] = le.text()

        # 2. Ejecutar
        try:
            name = self.combo_analyzer.currentText()
            path = self.combo_analyzer.currentData()
            Cls = import_class(name, path)
            analyzer = Cls(**kwargs)
            report = analyzer.analyze(self.session.current_data)
            
            # 3. Guardar en Historial
            run_name = f"{name} - {len(self.history_data) + 1}"
            self.history_data.append({'name': run_name, 'report': report})
            self.list_history.addItem(run_name)
            
            # 4. Mostrar y saltar a pestaña de resultados
            self.show_results(report)
            self.setCurrentIndex(1) 
        except Exception as e:
            self.txt_report.setText(f"Error: {e}")
            self.setCurrentIndex(1)
            
        try:
            # (Ejecución del analizados...)
            report = analyzer.analyze(self.session.current_data)
            
            # Guardamos en la sesión
            self.session.eda_report = report
            
            # ACTUALIZAMOS LA UI
            self.show_results(report)
            self.setCurrentIndex(1) # Ir a resultados
            
            # --- TRUCO: Forzar refresco de otras pestañas ---
            # Si el widget padre tiene el método refresh_gui, lo llamamos
            # Generalmente en tu BCIWorkstationGUI.py tienes esta función
            if hasattr(self.parentWidget(), 'refresh_gui'):
                self.parentWidget().refresh_gui()
        except Exception as e:
            self.txt_report.setText(f"Error durante el análisis: {e}")
            self.setCurrentIndex(1)
            
    def show_results(self, report):
        self.last_report = report
        self.session.eda_report = report 
        self.txt_report.setText(report.get('summary', ""))
        
        self.cb_plot_type.blockSignals(True)
        self.cb_plot_type.clear()
        visuals = report.get('visuals', {})
        self.cb_plot_type.addItems(list(visuals.keys()))
        self.cb_plot_type.blockSignals(False)
        self.update_canvas()

    def update_canvas(self):
        if not hasattr(self, 'last_report'): return
        key = self.cb_plot_type.currentText()
        data = self.last_report.get('visuals', {}).get(key, [])
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(self.c_ax)
        ax.hist(data, bins=30, color='#589df6', alpha=0.8)
        ax.set_title(key, color=self.c_text)
        ax.tick_params(colors=self.c_text)
        self.canvas.draw()

    def load_from_history(self, item):
        """Permite volver a ver un análisis previo clicando en el historial."""
        idx = self.list_history.row(item)
        data = self.history_data[idx]
        self.show_results(data['report'])
        self.setCurrentIndex(1)