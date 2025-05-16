"""
Initialize the medical practice database with schema and sample data
"""

import sqlite3
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database settings
db_url = os.getenv("DATABASE_URL", "sqlite:///medical_practice.db")
db_file = db_url.replace("sqlite:///", "")

def init_database():
    """Initialize the SQLite database with tables and sample data"""
    print(f"Initializing database: {db_file}")
    
    if os.path.exists(db_file):
        os.remove(db_file)
        print(f"Removed existing database: {db_file}")
    
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    
    # Bank Statements
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS bank_statements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date DATE,
        description VARCHAR(255),
        withdrawal REAL,
        deposit REAL,
        balance REAL
    );
    """)
    
    cursor.execute("""
    INSERT INTO bank_statements (date, description, withdrawal, deposit, balance)
    VALUES 
    ('2025-01-02', 'Insurance Reimbursement (Aetna)', NULL, 145000.00, 145000.00),
    ('2025-01-05', 'Vendor Payment - Medline', 35820.00, NULL, 109180.00),
    ('2025-01-10', 'Payroll', 72100.00, NULL, 37080.00),
    ('2025-01-15', 'Patient Payment (POS)', NULL, 8700.00, 45780.00);
    """)

    # Profit & Loss Reports
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS profit_loss_reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        period_start DATE,
        period_end DATE,
        total_revenue REAL,
        total_expense REAL,
        net_profit REAL
    );
    """)
    
    cursor.execute("""
    INSERT INTO profit_loss_reports (period_start, period_end, total_revenue, total_expense, net_profit)
    VALUES
    ('2024-10-01', '2024-12-31', 474500.00, 362500.00, 112000.00),
    ('2024-07-01', '2024-09-30', 500000.00, 380000.00, 120000.00);
    """)

    # Purchase Orders
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS purchase_orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        po_number VARCHAR(50),
        date DATE,
        vendor VARCHAR(100),
        total_amount REAL
    );
    """)
    
    cursor.execute("""
    INSERT INTO purchase_orders (po_number, date, vendor, total_amount)
    VALUES
    ('MS-PO-2025-011', '2025-01-12', 'Medline Industries', 18565.00),
    ('MS-PO-2025-012', '2025-01-15', 'Surgical Supplies Co.', 12250.00);
    """)

    # Purchase Order Items
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS purchase_order_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        purchase_order_id INTEGER,
        item_description VARCHAR(255),
        quantity INTEGER,
        unit_price REAL,
        total_price REAL
    );
    """)
    
    cursor.execute("""
    INSERT INTO purchase_order_items (purchase_order_id, item_description, quantity, unit_price, total_price)
    VALUES 
    (1, 'Ortho Implant Kit', 5, 2400.00, 12000.00),
    (1, 'Surgical Drapes Set', 100, 22.50, 2250.00),
    (2, 'Hip Prosthesis', 10, 800.00, 8000.00);
    """)

    # Supply Catalog
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS supply_catalog (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        item_name VARCHAR(255),
        sku VARCHAR(50),
        unit_price REAL,
        vendor VARCHAR(100),
        notes TEXT
    );
    """)
    
    cursor.execute("""
    INSERT INTO supply_catalog (item_name, sku, unit_price, vendor, notes)
    VALUES 
    ('Ortho Knee Implant (Standard)', 'OT-KI-STD', 2450.00, 'OrthoTech Supplies', 'FDA approved'),
    ('Hip Replacement Stem', 'OT-HR-STEM', 1980.00, 'OrthoTech Supplies', 'Titanium coated');
    """)

    # Equity Ownership
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS equity_ownership (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name VARCHAR(100),
        role VARCHAR(100),
        ownership_percent REAL,
        type VARCHAR(50)
    );
    """)
    
    cursor.execute("""
    INSERT INTO equity_ownership (name, role, ownership_percent, type)
    VALUES
    ('Dr. Alicia Mendez', 'Medical Director', 35.00, 'Voting Equity'),
    ('Dr. Rajiv Kapoor', 'Surgeon', 25.00, 'Voting Equity'),
    ('MedSure Holdings', 'Investment Partner', 40.00, 'Preferred Equity');
    """)

    # Payor Contracts
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS payor_contracts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        payor_name VARCHAR(100),
        effective_from DATE,
        effective_to DATE,
        payment_terms TEXT
    );
    """)
    
    cursor.execute("""
    INSERT INTO payor_contracts (payor_name, effective_from, effective_to, payment_terms)
    VALUES
    ('Aetna', '2023-01-01', '2025-12-31', 'Claims due in 30 days, 45-day payout'),
    ('Blue Cross', '2023-06-01', '2025-06-01', 'Monthly claims, 60-day payout');
    """)

    # Contract Procedures
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS contract_procedures (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        payor_contract_id INTEGER,
        cpt_code VARCHAR(10),
        procedure_name VARCHAR(255),
        fee_schedule_rate REAL,
        notes TEXT
    );
    """)
    
    cursor.execute("""
    INSERT INTO contract_procedures (payor_contract_id, cpt_code, procedure_name, fee_schedule_rate, notes)
    VALUES
    (1, '29881', 'Knee Arthroscopy (Meniscectomy)', 1250.00, 'Ambulatory surgery only'),
    (1, '27447', 'Total Knee Arthroplasty', 7800.00, 'Includes implant');
    """)
    
    # Commit changes
    conn.commit()

    # Print table information
    print("\nTables created and their row counts:")
    tables = [
        "bank_statements", 
        "profit_loss_reports", 
        "purchase_orders",
        "purchase_order_items", 
        "supply_catalog", 
        "equity_ownership", 
        "payor_contracts", 
        "contract_procedures"
    ]
    
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        row_count = cursor.fetchone()[0]
        print(f"{table}: {row_count} rows")
        
        cursor.execute(f"PRAGMA table_info({table});")
        columns = cursor.fetchall()
        print(f"  Columns:")
        for column in columns:
            print(f"  - {column[1]} ({column[2]})")  # column name and type
    
    # Close connection
    conn.close()

if __name__ == "__main__":
    init_database()
    print("\nDatabase initialization complete. You can now run the application.")