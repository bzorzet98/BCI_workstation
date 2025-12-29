import matplotlib
import matplotlib.pyplot as plt

# Try to force Qt for interactive windows
try:
    matplotlib.use('Qt5Agg')
except:
    pass

class Plotter:
    """
    Handles all MNE visualization calls.
    Now supports direct channel picking via arguments.
    """
    
    @staticmethod
    def plot_interactive(mne_object, picks=None):
        """
        Opens the native MNE interactive browse window.
        """
        if mne_object is None:
            print("No data to plot.")
            return

        # block=True ensures the terminal waits/GUI handles the window
        # n_channels gets overridden if picks provides fewer channels
        try:
            mne_object.plot(
                block=True, 
                title="Interactive View",
                picks=picks,  # <--- Pass selections here directly
                show=True
            )
        except Exception as e:
            print(f"Plotting failed: {e}")

    @staticmethod
    def plot_psd(mne_object, fmax=60, picks=None):
        """
        Plots Power Spectral Density.
        """
        if mne_object is None: return

        # New MNE 1.0+ syntax: object.compute_psd().plot()
        # compute_psd accepts 'picks'
        try:
            spectrum = mne_object.compute_psd(fmax=fmax, picks=picks)
            fig = spectrum.plot(show=False, picks=picks) # Plot needs picks too sometimes for layout
            plt.show(block=True)
        except Exception as e:
            print(f"PSD plotting failed: {e}")

    @staticmethod
    def plot_sensors(mne_object):
        """Plots sensor locations (picks usually not needed here, shows montage)."""
        if mne_object is None: return
        try:
            mne_object.plot_sensors(show_names=True, block=True)
        except Exception as e:
            print(f"Sensor plot failed: {e}")