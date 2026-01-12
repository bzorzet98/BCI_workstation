from src.utils.imports import import_class

class EDAProcessor:
    def __init__(self, pipeline_config):
        """
        Initializes all analyzers defined in the configuration.
        """
        self.analyzers = []
        for config in pipeline_config:
            analyzer_class = import_class(config['class_name'], config['module_name'])
            # Instantiate analyzer with its specific parameters
            instance = analyzer_class(**config.get('kwargs', {}))
            self.analyzers.append(instance)

    def run(self, data):
        """
        Applies all analyzers assigned to the current stage.
        """
        report = {}
        for analyzer in self.analyzers:
            report[analyzer.__class__.__name__] = analyzer.analyze(data)
        return report