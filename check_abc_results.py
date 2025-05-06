import sqlite3

def check_database():
    # Connect to the database
    conn = sqlite3.connect('C:/Users/jesse/AppData/Local/Temp/abc_nes.db')
    cursor = conn.cursor()
    
    # Get table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print("\nDatabase Tables:")
    for table in tables:
        print(table[0])
    
    # Get population information
    cursor.execute("SELECT t, population_size, epsilon FROM populations ORDER BY t")
    populations = cursor.fetchall()
    print("\nPopulation Information:")
    print("t\tpopulation_size\tepsilon")
    print("-"*40)
    for t, size, eps in populations:
        print(f"{t}\t{size}\t{eps:.4f}")
    
    # Get some sample particles
    cursor.execute("SELECT * FROM particles LIMIT 5")
    print("\nSample Particles:")
    print(cursor.fetchall())
    
    # Get parameter names
    cursor.execute("SELECT DISTINCT parameter_name FROM parameters")
    params = cursor.fetchall()
    print("\nParameters:")
    print([p[0] for p in params])
    
    conn.close()

if __name__ == '__main__':
    check_database()
