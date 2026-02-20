import traceback
from PyQt6.QtWidgets import QTabWidget, QMessageBox, QApplication
from PyQt6.QtCore import Qt

from src.core.registry import Registry
from src.utils.imports import import_class

# Import our new decoupled widgets
from src.interface.widgets.eda_config_widget import EDAConfigWidget
from src.interface.widgets.eda_results_widget import EDAResultsWidget


class EDATab(QTabWidget):
    """
    Master Orchestrator for Exploratory Data Analysis.
    
    This class acts purely as a Controller. It listens for configuration 
    signals, executes the mathematical Analyzers safely, and passes the 
    resulting EDAReport to the Results viewer.
    """
    def __init__(self, session_state, parent=None):
        super().__init__(parent)
        self.session = session_state
        
        # Try to inherit the theme from the main BCIWorkstationGUI
        theme_kwargs = getattr(parent, 'style_kwargs', {}) if parent else {}

        # 1. Instantiate the Decoupled Components
        self.config_widget = EDAConfigWidget(session_state)
        self.results_widget = EDAResultsWidget(theme_kwargs=theme_kwargs)

        # 2. Add them as main tabs
        self.addTab(self.config_widget, "EDA Configuration")
        self.addTab(self.results_widget, "Diagnostic Results")

        # 3. THE MAGIC BRIDGE: Connect User Inputs to Math Execution
        self.config_widget.analysis_requested.connect(self.run_analysis)

    def refresh(self):
        """Called by the main GUI (bciworkstation.py) when new datasets are loaded."""
        self.config_widget.refresh_contexts()

    def run_analysis(self, analyzer_name: str, payload: dict, context_names: list):
        if not context_names:
            QMessageBox.warning(self, "No Context", "Please select at least one context to analyze.")
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        
        try:
            analyzers_dict = Registry.discover_eda_analyzers()
            path = analyzers_dict[analyzer_name]
            AnalyzerClass = import_class(analyzer_name, path)
            
            # 1. Initialize Analyzer cleanly
            init_kwargs = payload.get("init_kwargs", {})
            analyzer = AnalyzerClass(**init_kwargs)

            # 2. Convert string names into real MNE objects from the workspace
            call_kwargs_strs = payload.get("call_kwargs", {})
            call_kwargs_objs = {}
            for param_name, ctx_string in call_kwargs_strs.items():
                if ctx_string is not None:
                    # Fetch the actual Data object (like the Reference EMG)
                    obj = self.session.workspace.get(ctx_string)
                    if obj is not None:
                        call_kwargs_objs[param_name] = obj

            # 3. Loop through all selected contexts
            generated_reports = []
            for ctx_name in context_names:
                data = self.session.workspace.get(ctx_name)
                if data is None: continue
                
                # We unpack **call_kwargs_objs so if it contains {'reference': <mne_obj>}, 
                # it safely fulfills the analyzer's positional requirement!
                try:
                    report = analyzer.analyze(data, **call_kwargs_objs)
                except TypeError as e:
                    # Fallback in case an analyzer signature is unexpectedly restrictive
                    report = analyzer.analyze(data)
                
                # Tag the report with the context name so the Viewer can group it!
                report.metadata["context"] = ctx_name
                generated_reports.append(report)

            if not generated_reports:
                raise ValueError("No valid data found for the selected contexts.")

            # Pass the LIST of reports to the smart viewer
            self.results_widget.display_report(generated_reports)
            
            self.config_widget.add_history_entry(analyzer_name)
            self.setCurrentIndex(1)

        except Exception as e:
            err_msg = traceback.format_exc()
            QMessageBox.critical(self, "Analysis Failed", f"An error occurred in {analyzer_name}:\n\n{str(e)}\n\n{err_msg}")
            
        finally:
            QApplication.restoreOverrideCursor()
        
        