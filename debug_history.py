import pyabc
import tempfile

def debug_history():
    # Create a temporary database
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
    
    # Get populations DataFrame
    populations = history.get_all_populations()
    
    # Print detailed information about the DataFrame
    print("\nDataFrame Info:")
    print("Columns:", populations.columns.tolist())
    print("\nFirst row data:")
    print(populations.iloc[0])
    print("\nLast row data:")
    print(populations.iloc[-1])
    
    # Print types of each column
    print("\nColumn types:")
    for col in populations.columns:
        print(f"{col}: {type(populations[col].iloc[0])}")

if __name__ == '__main__':
    debug_history()
