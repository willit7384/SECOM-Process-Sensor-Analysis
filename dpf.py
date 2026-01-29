import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine, text, inspect
import getpass

"""Data Pipeline Functions (dpf.py)"""

def find_constant_features(df: pd.DataFrame, threshold: int = 1):
    """
    Returns a list of columns that have 'threshold' or fewer unique values.
    Useful for dropping constant or near-constant features.
    
    Parameters:
    - df: pandas DataFrame
    - threshold: max number of unique values to consider a column constant
    
    Returns:
    - List of column names
    """
    return [col for col in df.columns if df[col].nunique() <= threshold]


def Check(df):
    """
    Provides a comprehensive overview of a pandas DataFrame:
    - Shape of the DataFrame
    - Data types of each column
    - Number of missing values per column
    - Percentage of missing values per column
    - Number of unique values per column
    - Sample of the first few rows
    
    Parameters:
    df (pd.DataFrame): The DataFrame to inspect
    """
    print("Initating Data Checking Process...")
    print('Shape of the DataFrame:')
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    
    # Create summary DataFrame
    summary = pd.DataFrame({
        'Dtype': df.dtypes,
        'Missing': df.isnull().sum(),
        'Missing %': (df.isnull().sum() / len(df) * 100).round(2),
        'Unique': df.nunique()
    })
    
    print(summary)
    print("\n" + "="*50)
    print("First 5 rows:")
    print(df.head())
    
    return summary

def interactive_cleanse(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interactively cleans a DataFrame:
    - Drops columns
    - Handles missing values (drop rows, fill with mean/median/mode/custom)
    - Converts data types
    - Drops duplicates
    Returns the cleaned DataFrame.
    """
    print("\n=== Interactive Data Cleansing ===")
    df_clean = df.copy()
    
    while True:
        print("\nCurrent shape:", df_clean.shape)
        action = input(
            "\nWhat do you want to do? \n"
            "1: Drop columns\n"
            "2: Handle missing values\n"
            "3: Change data types\n"
            "4: Drop duplicate rows\n"
            "5: Show missing value summary\n"
            "6: Finish cleansing\n"
            "Choice (1-6): "
        ).strip()

        if action == "1":
            print("\nCurrent columns:", list(df_clean.columns))
            cols_to_drop = input("Enter columns to drop (comma-separated): ").strip()
            if cols_to_drop:
                cols = [c.strip() for c in cols_to_drop.split(",")]
                cols = [c for c in cols if c in df_clean.columns]
                if cols:
                    df_clean = df_clean.drop(columns=cols)
                    print(f"Dropped columns: {cols}")
                else:
                    print("No valid columns to drop.")

        elif action == "2":
            print("\nMissing values per column:")
            print(df_clean.isnull().sum()[df_clean.isnull().sum() > 0])
            col = input("Enter column name to handle missing values (or 'all'): ").strip()
            if col.lower() == 'all':
                columns = df_clean.columns[df_clean.isnull().any()]
            elif col in df_clean.columns:
                columns = [col]
            else:
                print("Column not found or has no missing values.")
                continue

            for c in columns:
                print(f"\nHandling missing values in '{c}' ({df_clean[c].dtype})")
                strategy = input(
                    "Strategy?\n"
                    "1: Drop rows\n"
                    "2: Fill with mean (numeric)\n"
                    "3: Fill with median (numeric)\n"
                    "4: Fill with mode\n"
                    "5: Fill with custom value\n"
                    "Choice (1-5): "
                ).strip()

                if strategy == "1":
                    df_clean = df_clean.dropna(subset=[c])
                elif strategy == "2" and np.issubdtype(df_clean[c].dtype, np.number):
                    df_clean[c] = df_clean[c].fillna(df_clean[c].mean())
                elif strategy == "3" and np.issubdtype(df_clean[c].dtype, np.number):
                    df_clean[c] = df_clean[c].fillna(df_clean[c].median())
                elif strategy == "4":
                    df_clean[c] = df_clean[c].fillna(df_clean[c].mode()[0] if not df_clean[c].mode().empty else np.nan)
                elif strategy == "5":
                    value = input("Enter custom value: ").strip()
                    if value:
                        if np.issubdtype(df_clean[c].dtype, np.number):
                            try:
                                value = float(value) if '.' in value else int(value)
                            except:
                                pass
                        df_clean[c] = df_clean[c].fillna(value)

        elif action == "3":
            print("\nCurrent dtypes:")
            print(df_clean.dtypes)
            col = input("Enter column to change type: ").strip()
            if col in df_clean.columns:
                new_type = input("New type (int, float, str, category, datetime): ").strip().lower()
                try:
                    if new_type == "int":
                        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').astype('Int64')
                    elif new_type == "float":
                        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    elif new_type == "str":
                        df_clean[col] = df_clean[col].astype(str)
                    elif new_type == "category":
                        df_clean[col] = df_clean[col].astype('category')
                    elif new_type == "datetime":
                        df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                    print(f"Converted '{col}' to {new_type}")
                except Exception as e:
                    print(f"Conversion failed: {e}")

        elif action == "4":
            before = df_clean.shape[0]
            df_clean = df_clean.drop_duplicates()
            after = df_clean.shape[0]
            print(f"Dropped {before - after} duplicate rows.")

        elif action == "5":
            print(df_clean.isnull().sum())

        elif action == "6":
            print("\nCleansing complete!")
            break

        else:
            print("Invalid choice.")

    return df_clean


def interactive_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interactively transforms a DataFrame:
    - Create new columns
    - Rename columns
    - Apply simple functions
    - Basic feature engineering
    """
    print("\n=== Interactive Data Transformation ===")
    df_trans = df.copy()

    while True:
        print("\nCurrent columns:", list(df_trans.columns))
        action = input(
            "\nWhat do you want to do?\n"
            "1: Create new column\n"
            "2: Rename column\n"
            "3: Apply function to column\n"
            "4: Finish transformation\n"
            "Choice (1-4): "
        ).strip()

        if action == "1":
            new_col = input("New column name: ").strip()
            expr = input(
                "Expression (use existing column names, e.g., 'Age + 5' or 'Name.str.upper()'): "
            ).strip()
            try:
                df_trans[new_col] = df_trans.eval(expr) if ' ' in expr or '+' in expr else eval(f"df_trans['{expr.split()[0]}'].{expr}")
                print(f"Created '{new_col}'")
            except Exception as e:
                print(f"Failed: {e}")

        elif action == "2":
            old = input("Old column name: ").strip()
            new = input("New column name: ").strip()
            if old in df_trans.columns and new:
                df_trans = df_trans.rename(columns={old: new})
                print(f"Renamed '{old}' → '{new}'")

        elif action == "3":
            col = input("Column to transform: ").strip()
            func = input("Function (e.g., '.str.lower()', '.abs()', '*100'): ").strip()
            try:
                df_trans[col] = eval(f"df_trans['{col}']{func}")
                print(f"Applied {func} to '{col}'")
            except Exception as e:
                print(f"Failed: {e}")

        elif action == "4":
            print("\nTransformation complete!")
            break

    return df_trans


def create_sqlite_database(df: pd.DataFrame, db_path: str = None, table_name: str = None) -> str:
    """
    Saves the DataFrame to a SQLite database.
    Interactively asks for database path and table name if not provided.
    """
    if db_path is None:
        default_db = "data.db"
        db_path = input(f"Enter SQLite database path (default: {default_db}): ").strip() or default_db

    if table_name is None:
        default_table = "main_table"
        table_name = input(f"Enter table name (default: {default_table}): ").strip() or default_table

    engine = create_engine(f"sqlite:///{db_path}", echo=False)

    print(f"\nWriting DataFrame ({df.shape[0]} rows) to '{table_name}' in '{db_path}'...")
    df.to_sql(table_name, engine, if_exists='replace', index=False)
    
    abs_path = os.path.abspath(db_path)
    print(f"Success! Database saved at: {abs_path}")
    return abs_path


def interactive_create_mysql_database(df, host='localhost', user='root', password=''):
    """
    Interactive version: asks for database name, table name, and password if needed.
    """
    print("\n=== Save to MySQL Database ===")
    
    db_name = input(f"Database name (default: main_db): ").strip() or 'main_db'
    table_name = input(f"Table name (default: drug_overdose_data): ").strip() or 'drug_overdose_data'
    
    if not password:
        import getpass
        password = getpass.getpass("MySQL password: ")
    
    # Auto-create DB if needed
    from sqlalchemy import create_engine, text
    
    engine_no_db = create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}")
    with engine_no_db.connect() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {db_name}"))
        conn.commit()
    
    engine = create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}/{db_name}")
    
    print(f"Loading {df.shape[0]} rows into {db_name}.{table_name}...")
    df.to_sql(table_name, engine, if_exists='replace', index=False)
    print("Done! Data saved to MySQL.")

    from sqlalchemy import create_engine, text, inspect
import getpass

def interactive_add_to_mysql(df):
    """
    Interactively adds a pandas DataFrame to an existing MySQL database.
    Guides you through selecting database, table, and write mode.
    """
    print("\n=== Interactive: Add DataFrame to MySQL Database ===\n")
    
    # Get credentials
    host = input("MySQL host (default: localhost): ").strip() or 'localhost'
    user = input("MySQL user (default: root): ").strip() or 'root'
    password = getpass.getpass("MySQL password: ")
    port = input("MySQL port (default: 3306): ").strip() or '3306'
    
    # Connect without database to list available ones
    try:
        engine_no_db = create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}:{port}")
        with engine_no_db.connect() as conn:
            result = conn.execute(text("SHOW DATABASES"))
            databases = [row[0] for row in result if row[0] not in ('information_schema', 'performance_schema', 'mysql', 'sys')]
    except Exception as e:
        print(f"Connection failed: {e}")
        return
    
    print("\nAvailable databases:")
    for i, db in enumerate(databases, 1):
        print(f"  {i}: {db}")
    
    # Choose or create database
    choice = input(f"\nEnter number (1-{len(databases)}) to select, or type a new database name: ").strip()
    
    if choice.isdigit() and 1 <= int(choice) <= len(databases):
        db_name = databases[int(choice) - 1]
    else:
        db_name = choice.strip() or 'my_data_db'
        print(f"Will use/create database: {db_name}")
    
    # Auto-create database if it doesn't exist
    try:
        with engine_no_db.connect() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{db_name}`"))
            conn.commit()
        print(f"Database `{db_name}` is ready.")
    except Exception as e:
        print(f"Failed to create/access database: {e}")
        return
    
    # Now connect to the selected database
    engine = create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{db_name}")
    
    # List existing tables
    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()
    except Exception as e:
        print(f"Could not list tables: {e}")
        return
    
    if tables:
        print("\nExisting tables in this database:")
        for i, table in enumerate(tables, 1):
            print(f"  {i}: {table}")
    else:
        print("\nNo existing tables in this database.")
    
    # Get table name
    default_table = "my_table"
    table_name = input(f"\nEnter table name to write to (default: {default_table}): ").strip() or default_table
    
    # Choose write mode
    print("\nWhat to do if table already exists?")
    print("1: Replace (drop and recreate)")
    print("2: Append (add rows, must match columns)")
    print("3: Fail (error if exists)")
    mode_choice = input("Choice (1-3, default 2): ").strip() or "2"
    
    if mode_choice == "1":
        if_exists = 'replace'
        mode_text = "replace"
    elif mode_choice == "3":
        if_exists = 'fail'
        mode_text = "fail if exists"
    else:
        if_exists = 'append'
        mode_text = "append"
    
    # Final confirmation
    print(f"\nSummary:")
    print(f"   Database: {db_name}")
    print(f"   Table: {table_name}")
    print(f"   Rows to add: {df.shape[0]}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Mode: {mode_text}")
    
    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    # Write the data
    try:
        print(f"\nWriting data to {db_name}.{table_name}...")
        df.to_sql(table_name, engine, if_exists=if_exists, index=False)
        print(f"Success! Data added to MySQL table `{table_name}`.")
        
        # Quick verification
        with engine.connect() as conn:
            count = conn.execute(text(f"SELECT COUNT(*) FROM `{table_name}`")).scalar()
            print(f"Total rows in table now: {count}")
    except Exception as e:
        print(f"Failed to write data: {e}")


def create_postgres_db(user=None, password=None, host=None, port=None, db_name=None):
    """
    Interactively create a PostgreSQL database if it doesn't exist.
    Returns the SQLAlchemy engine connected to the new database.
    """
    print("\n=== PostgreSQL Database Setup ===")
    
    host = host or input("Host (default 'localhost'): ").strip() or 'localhost'
    port = port or input("Port (default 5432): ").strip() or '5432'
    user = user or input("User (default 'postgres'): ").strip() or 'postgres'
    password = password or getpass.getpass("Password: ")
    
    db_name = db_name or input("Database name (default 'nonprofit_990'): ").strip() or 'nonprofit_990'
    
    # Connect to default DB to create new database if needed
    default_engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/postgres")
    
    try:
        with default_engine.connect() as conn:
            conn.execute(text(f"COMMIT"))  # Needed to run CREATE DATABASE
            conn.execute(text(f"CREATE DATABASE {db_name}"))
            print(f"Database '{db_name}' created successfully.")
    except Exception as e:
        print(f"Database creation skipped (might already exist): {e}")
    
    # Connect to the target database
    engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}")
    print(f"Connected to PostgreSQL database '{db_name}'")
    return engine

def add_dataframe_to_postgres(df: pd.DataFrame, engine, table_name: str = None, if_exists: str = 'replace'):
    """
    Adds a DataFrame to a PostgreSQL database.
    
    Parameters:
    - df: pandas DataFrame
    - engine: SQLAlchemy engine connected to the target database
    - table_name: Name of the table to create or append to
    - if_exists: 'replace', 'append', or 'fail'
    """
    table_name = table_name or input("Table name to write to (default 'irs_990_data'): ").strip() or 'irs_990_data'
    
    try:
        df.to_sql(table_name, engine, if_exists=if_exists, index=False)
        print(f"Data successfully written to '{table_name}' ({df.shape[0]} rows).")
        
        # Verification
        with engine.connect() as conn:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
            print(f"Total rows in table now: {count}")
    except Exception as e:
        print(f"Failed to write data: {e}")

# ================= Example usage =================

# Suppose 'foundations', 'grants', 'officers', etc. are your DataFrames from the IRS 990 parser
# You can create the PostgreSQL database and load one DataFrame as follows:

if __name__ == "__main__":
    # Create/connect to database
    engine = create_postgres_db()
    
    # Example: add foundations table
    import os
    csv_path = os.path.join("IRS990CSV_Folder", "foundations.csv")
    if os.path.exists(csv_path):
        foundations_df = pd.read_csv(csv_path)
        add_dataframe_to_postgres(foundations_df, engine, table_name="foundations", if_exists='replace')
