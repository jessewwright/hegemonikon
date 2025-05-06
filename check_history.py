import pyabc
from datetime import datetime

def check_history_attributes():
    # Create a temporary database
    import tempfile
    temp_dir = tempfile.gettempdir()
    db_path = f"sqlite:///{temp_dir}/abc_nes.db"
    
    # Create a dummy ABCSMC object and history
    abc = pyabc.ABCSMC(
        models=lambda x: x,
        parameter_priors=pyabc.Distribution(x=pyabc.RV('uniform', 0, 1))
    )
    
    # Initialize history with dummy data
    abc.new(db_path, {'x': 0.5})
    history = abc.history
    
    # Get all populations
    populations = history.get_all_populations()
    
    # Print available columns
    print("\nAvailable columns in populations:")
    print(populations.columns.tolist())
    
    # Print first row
    print("\nFirst row:")
    print(populations.iloc[0])
    
    # Print last row
    print("\nLast row:")
    print(populations.iloc[-1])

if __name__ == '__main__':
    check_history_attributes()
