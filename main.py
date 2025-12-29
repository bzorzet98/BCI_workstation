import sys
import ast # Safe parsing of python literals (lists, dicts)
import init_paths
from src.core.data_manager import DataManager
from src.core.session import SessionState
from src.core.registry import Registry
from src.interface.plotter import Plotter

def parse_user_input(user_str, default_value):
    """
    Smartly converts string input into Python types (float, int, list, bool).
    """
    if user_str.strip() == "":
        return default_value
    
    # Handle Booleans explicitly
    if user_str.lower() == 'true': return True
    if user_str.lower() == 'false': return False
    
    # Try to parse literals (lists, dicts, numbers)
    try:
        return ast.literal_eval(user_str)
    except:
        # Fallback: keep as string
        return user_str

def main():
    # --- INITIALIZATION ---
    print("=========================================")
    print("      BCI WORKSTATION (CLI MODE)         ")
    print("=========================================")
    
    data_manager = DataManager()
    session = SessionState()
    
    # --- PHASE 1: LOAD DATA ---
    while True:
        print("\n[1] Select Dataset")
        datasets = data_manager.get_available_datasets()
        for i, ds in enumerate(datasets):
            print(f"    {i}: {ds}")
            
        try:
            choice = int(input("Choice ID: "))
            ds_name = datasets[choice]
            data_manager.load_dataset_class(ds_name)
            break
        except (ValueError, IndexError):
            print("Invalid selection.")

    # Get Metadata to help user choose
    meta = data_manager.get_metadata()
    print(f"\nLoaded {ds_name}.")
    print(f"Available Subjects: {meta['subjects'][0]} ... {meta['subjects'][-1]}")
    print(f"Available Data available: {meta['data_available']}")
    
    while True:
        try:
            sub_id = int(input("Subject ID: "))
            data_to_load = input("Data to Load: ").strip()
            
            raw_data = data_manager.load_data(sub_id, data_to_load)
            if raw_data:
                session.load_data(raw_data, ds_name, sub_id, data_to_load)
                break
            else:
                print("Could not load data. Try again.")
        except ValueError:
            print("Invalid input.")

    # --- PHASE 2: INTERACTIVE SESSION ---
    while True:
        print("\n------------------------------------------------")
        print(session.get_info_string())
        print("ACTIONS:")
        print("  [p] Plot Data (Interactive)")
        print("  [s] Plot Sensors")
        print("  [f] Plot PSD (Frequency)")
        print("  [a] Apply Processing Step (Filter, Epoch, etc.)")
        print("  [e] Export Pipeline Config")
        print("  [q] Quit")
        
        cmd = input("Command: ").strip().lower()
        
        if cmd == 'q':
            break
            
        elif cmd == 'p':
            print("Opening interactive plot... (Close window to continue)")
            Plotter.plot_interactive(session.current_data)
            
        elif cmd == 's':
            Plotter.plot_sensors(session.current_data)
            
        elif cmd == 'f':
            Plotter.plot_psd(session.current_data)
            
        elif cmd == 'e':
            fname = input("Filename [pipeline.json]: ") or "pipeline.json"
            session.export_pipeline(fname)
            
        elif cmd == 'a':
            # --- DYNAMIC REGISTRY LOOKUP ---
            # 1. Ask Registry what is possible for this data type
            steps = Registry.get_available_steps(session.current_data)
            
            print("\nAvailable Processing Steps:")
            for i, step in enumerate(steps):
                print(f"  {i}: {step['name']} ({step['type']})")
                
            try:
                step_idx = int(input("Select Step ID: "))
                selected_step = steps[step_idx]
                name = selected_step['name']
                sType = selected_step['type']
                
                # 2. Ask Registry for arguments
                defaults = Registry.get_defaults(name, sType)
                
                print(f"\nConfiguring '{name}'. Press Enter to keep default.")
                final_kwargs = {}
                
                for arg, default_val in defaults.items():
                    prompt = f"  {arg} [default: {default_val}]: "
                    user_val = input(prompt)
                    final_kwargs[arg] = parse_user_input(user_val, default_val)
                
                # 3. Construct Config
                step_config = {
                    "type": sType,
                    "name": name,
                    "module_name": "src.preprocessing.preprocessing_functions" if sType == 'function' else None,
                    "kwargs": final_kwargs
                }
                
                # 4. Execute via Session
                session.run_transform(step_config)
                
            except (ValueError, IndexError) as e:
                print(f"Selection Error: {e}")
            except Exception as e:
                print(f"Execution Error: {e}")

if __name__ == "__main__":
    main()