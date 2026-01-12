import ast
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QComboBox, QPushButton, QScrollArea, QLineEdit, 
                             QFormLayout, QTextEdit, QFrame, QTabWidget, QListWidget, QApplication)
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
        self.available_analyzers = Registry.discover_eda_analyzers()
        
        for name, path in self.available_analyzers.items():
            self.combo_analyzer.addItem(name, userData=path)
        # ---------------------------------

        self.combo_analyzer.currentIndexChanged.connect(self.on_analyzer_changed)
        f_select.addRow("<b>Seleccionar Función EDA:</b>", self.combo_analyzer)
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
        self.cb_metric_selector = QComboBox()
        self.cb_channel_selector = QComboBox() 
        self.cb_metric_selector.currentIndexChanged.connect(self.update_plot_selections)
        self.cb_channel_selector.currentIndexChanged.connect(self.update_canvas)

        h_ctrl.addWidget(QLabel("Métrica:"))
        h_ctrl.addWidget(self.cb_metric_selector)
        h_ctrl.addWidget(QLabel("Canal:"))
        h_ctrl.addWidget(self.cb_channel_selector)
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

    def update_plot_selections(self):
        """Cuando cambia la métrica, actualizamos la lista de canales disponibles."""
        if not hasattr(self, 'last_report'): return
        metric = self.cb_metric_selector.currentText()
        channels = list(self.last_report['visuals'][metric].keys())
        
        self.cb_channel_selector.blockSignals(True)
        self.cb_channel_selector.clear()
        self.cb_channel_selector.addItem("Todos los canales") # Opción para superponer
        self.cb_channel_selector.addItems(channels)
        self.cb_channel_selector.blockSignals(False)
        self.update_canvas()

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
        if not self.session.current_data:
            self.txt_report.setText("Error: No hay datos cargados en la sesión.")
            return

        # 1. Mostrar cursor de carga (Feedback al usuario)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        
        try:
            # 2. Parsear Argumentos desde la interfaz
            kwargs = {}
            for arg, le in self.inputs.items():
                try:
                    # ast.literal_eval convierte "[1, 2]" en una lista real de Python
                    kwargs[arg] = ast.literal_eval(le.text())
                except:
                    kwargs[arg] = le.text()

            # 3. Importar e Instanciar el Analizador
            name = self.combo_analyzer.currentText()
            path = self.combo_analyzer.currentData()
            Cls = import_class(name, path)
            
            # Limpiamos kwargs de 'picks' por si acaso y pasamos los canales de la sesión
            kwargs.pop('picks', None)
            analyzer = Cls(picks=self.session.selected_channels, **kwargs)
            
            # 4. EJECUTAR EL ANÁLISIS
            report = analyzer.analyze(self.session.current_data)
            
            # 5. GUARDAR RESULTADOS
            # En la sesión global (para que el TimeDomainTab lo vea)
            self.session.eda_report = report
            
            # En el historial local de esta pestaña
            run_name = f"{name} - {self.list_history.count() + 1}"
            self.history_data.append({'name': run_name, 'report': report})
            self.list_history.addItem(run_name)
            
            # 6. ACTUALIZAR INTERFAZ
            self.show_results(report)
            self.setCurrentIndex(1) # Cambiar a la pestaña de resultados
            
            # 7. FORZAR REFRESCO GLOBAL (Sincronización de pestañas)
            # Buscamos la ventana principal para llamar a su refresh_gui
            main_win = self.window() # Obtiene la QMainWindow directamente
            if hasattr(main_win, 'refresh_gui'):
                main_win.refresh_gui()
                
        except Exception as e:
            # En caso de error, mostramos el mensaje en el reporte
            error_msg = f"ERROR DURANTE EL ANÁLISIS ({name}):\n{str(e)}"
            self.txt_report.setText(error_msg)
            self.setCurrentIndex(1)
            print(f"DEBUG: {error_msg}") # También por consola para desarrollo
            
        finally:
            # Restaurar el cursor normal pase lo que pase
            QApplication.restoreOverrideCursor()
                
    def show_results(self, report):
        """Puebla los datos en las pestañas correspondientes."""
        self.last_report = report
        self.session.eda_report = report 

        # 1. Pestaña Resumen
        self.txt_report.setText(report.get('summary', "Análisis completado."))

        # 2. Pestaña Plots: Llenamos las MÉTRICAS disponibles
        self.cb_metric_selector.blockSignals(True)
        self.cb_metric_selector.clear()
        
        visuals = report.get('visuals', {})
        if visuals:
            # Llenamos con "Variance Ratio", "Peak Amplitude", etc.
            self.cb_metric_selector.addItems(list(visuals.keys()))
        
        self.cb_metric_selector.blockSignals(False)

        # Disparamos la actualización de canales para la primera métrica
        self.update_plot_selections()

    def update_canvas(self):
        # ... lógica de dibujo ...
        metric = self.cb_metric_selector.currentText()
        target_ch = self.cb_channel_selector.currentText()
        visuals = self.last_report['visuals'][metric]

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(self.c_ax)

        if target_ch == "Todos los canales":
            # Superponemos histogramas con transparencia
            for ch, data in visuals.items():
                ax.hist(data, bins=30, alpha=0.5, label=ch)
            ax.legend(fontsize=8, facecolor=self.c_ax, labelcolor=self.c_text)
        else:
            # Mostramos solo el canal seleccionado
            data = visuals[target_ch]
            ax.hist(data, bins=30, color='#589df6', edgecolor='white')
            ax.set_title(f"{metric} - Canal: {target_ch}", color=self.c_text)

        self.canvas.draw()

    def load_from_history(self, item):
        """Permite volver a ver un análisis previo clicando en el historial."""
        idx = self.list_history.row(item)
        data = self.history_data[idx]
        self.show_results(data['report'])
        self.setCurrentIndex(1)