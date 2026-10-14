from pathlib import Path
import pandas as pd 
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
import json
import numpy as np

# Find the main project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Location of the Excel dataset
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "online_retail_II.xlsx"


# def inspect_dataset():
#     # Load both sheets from the Excel file
#     sheets = pd.read_excel(DATA_PATH, sheet_name=None)
#     # Display the row count for each sheet
#     print("Sheets found:")
#     for sheet_name, sheet_df in sheets.items():
#         print(f"{sheet_name}: {len(sheet_df):,} rows")
#     # Combine both sheets
#     df = pd.concat(sheets.values(), ignore_index=True)
#     print(f"\nTotal rows: {len(df):,}")
#     print("\nColumn names:")
#     print(df.columns.tolist())
#     print("\nFirst five rows:")
#     print(df.head())
#     print("\nMissing values:")
#     print(df.isnull().sum())


# def inspect_data_quality():
#     sheets = pd.read_excel(DATA_PATH, sheet_name=None)
#     df = pd.concat(sheets.values(), ignore_index=True)
#     print("\n--- Data Quality Inspection ---")
#     print("\nInvoice examples:")
#     print(df["Invoice"].astype(str).head(10).tolist())
#     print("\nInvoices starting with C:")
#     print(df["Invoice"].astype(str).str.startswith("C").sum())
#     print("\nQuantity statistics:")
#     print(df["Quantity"].describe())
#     print("\nNon-positive quantities:")
#     print((df["Quantity"] <= 0).sum())
#     print("\nPrice statistics:")
#     print(df["Price"].describe())
#     print("\nNon-positive prices:")
#     print((df["Price"] <= 0).sum())
#     print("\nUnique stock codes:")
#     print(df["StockCode"].nunique())
#     print("\nMost common stock codes:")
#     print(df["StockCode"].value_counts().head(15))
#     print("\nEarliest transaction:", df["InvoiceDate"].min())
#     print("Latest transaction:", df["InvoiceDate"].max())


# def inspect_stock_codes():
#     sheets = pd.read_excel(DATA_PATH, sheet_name=None)
#     df = pd.concat(sheets.values(), ignore_index=True)
#     # Find stock codes that contain letters or symbols
#     stock_codes = df["StockCode"].astype(str).str.strip()
#     unusual = df[
#         stock_codes.str.contains(r"[^0-9]", regex=True, na=False)
#     ]
#     print("\n--- Unusual Stock Codes ---")
#     print("\nNumber of rows with non-numeric stock codes:")
#     print(len(unusual))
#     print("\nMost common unusual stock codes:")
#     print(
#         unusual.groupby("StockCode")
#         .agg(
#             Rows=("StockCode", "size"),
#             Description=("Description", "first")
#         )
#         .sort_values("Rows", ascending=False)
#         .head(30)
#         .to_string()
#     )


# def inspect_non_product_candidates():
#     sheets = pd.read_excel(DATA_PATH, sheet_name=None)
#     df = pd.concat(sheets.values(), ignore_index=True)
#     stock_codes = df["StockCode"].astype(str).str.strip()
#     # Look for codes that are not purely numeric
#     unusual = df[
#         stock_codes.str.contains(r"[^0-9]", regex=True, na=False)
#     ]
#     # Summarize each code and its descriptions
#     summary = (
#         unusual.groupby("StockCode")
#         .agg(
#             Rows=("StockCode", "size"),
#             Descriptions=(
#                 "Description",
#                 lambda x: " | ".join(
#                     x.dropna().astype(str).unique()[:3]
#                 )
#             )
#         )
#         .sort_values("Rows", ascending=False)
#     )
#     print("\n--- Non-Product Candidates ---")
#     print(summary.to_string())
#     print(f"\nUnique unusual stock codes: {len(summary)}")


REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
# 2. Known non-product stock codes
# These are known service, adjustment, voucher, or test
# entries. We will exclude only explicit codes, not all
# codes containing letters.

NON_PRODUCT_CODES = {
    "POST",          # Postage
    "DOT",           # Dotcom postage
    "M",             # Manual entry
    "D",             # Discount
    "BANK CHARGES",  # Bank charges
    "ADJUST2",       # Adjustment
    "B",             # Bad-debt adjustment
    "TEST002",       # Test product
    "GIFT",          # Gift entry
}



# 3. Load both Excel sheets


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )
    print("Loading Online Retail II dataset...")
    sheets = pd.read_excel(
        DATA_PATH,
        sheet_name=None,
        engine="openpyxl"
    )
    # Store sheet-level and column-level profiles
    sheet_profiles = []
    column_profiles = []
    for sheet_name, sheet_df in sheets.items():
        print(f"{sheet_name}: {len(sheet_df):,} rows")
        # Record row and column counts for each sheet
        sheet_profiles.append({
            "Sheet": sheet_name,
            "Rows": len(sheet_df),
            "Columns": len(sheet_df.columns)
        })
        # Record details for every column
        for column in sheet_df.columns:
            column_profiles.append({
                "Sheet": sheet_name,
                "Column": column,
                "Data Type": str(sheet_df[column].dtype),
                "Missing Values": int(
                    sheet_df[column].isna().sum()
                ),
                "Missing Percentage": round(
                    sheet_df[column].isna().mean() * 100,
                    2
                )
            })

    # Save DE1 reports
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    sheet_profile_df = pd.DataFrame(sheet_profiles)
    data_profile_df = pd.DataFrame(column_profiles)
    sheet_profile_df.to_csv(
        REPORT_DIR / "sheet_row_counts.csv",
        index=False
    )
    data_profile_df.to_csv(
        REPORT_DIR / "data_profile.csv",
        index=False
    )
    print("\n--- DE1: Sheet Row Counts ---")
    print(sheet_profile_df.to_string(index=False))
    print("\n--- DE1: Column Data Profile ---")
    print(data_profile_df.to_string(index=False))

    # Combine sheets for the existing cleaning pipeline
    df = pd.concat(
        sheets.values(),
        ignore_index=True
    )
    print(f"\nTotal original rows: {len(df):,}")
    print("DE1 reports saved in:", REPORT_DIR)

    return df



# 4. Clean the transaction data

def clean_dataset(df):
    df = df.copy()

    # Standardize stock codes and descriptions
    df["StockCode"] = (
        df["StockCode"]
        .astype("string")
        .str.strip()
        .str.upper()
    )
    df["Description"] = (
        df["Description"]
        .astype("string")
        .str.strip()
    )
    # Ensure dates and numeric columns have correct types
    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce"
    )
    df["Quantity"] = pd.to_numeric(
        df["Quantity"],
        errors="coerce"
    )
    df["Price"] = pd.to_numeric(
        df["Price"],
        errors="coerce"
    )
    # Prepare a sequential cleaning log
    cleaning_log = []
    def apply_rule(data, rule_name, mask):
        """Remove rows matching a rule and log the count."""
        before = len(data)
        data = data.loc[~mask].copy()
        removed = before - len(data)
        cleaning_log.append({
            "Rule": rule_name,
            "Rows before": before,
            "Rows removed": removed,
            "Rows remaining": len(data),
        })
        print(
            f"{rule_name}: removed {removed:,}; "
            f"{len(data):,} rows remain"
        )
        return data
    print("\n--- Cleaning Transactions ---")

    # Rule 1: Remove rows without a customer ID
    df = apply_rule(
        df,
        "Missing customer ID",
        df["Customer ID"].isna()
    )
    # Rule 2: Remove rows without a product description
    df = apply_rule(
        df,
        "Missing description",
        df["Description"].isna()
        | df["Description"].eq("")
    )
    # Rule 3: Remove cancellation invoices
    invoice_text = df["Invoice"].astype("string").str.strip()

    df = apply_rule(
        df,
        "Cancellation invoices",
        invoice_text.str.upper().str.startswith(
            "C", na=False
        )
    )
    # Rule 4: Keep only positive quantities
    df = apply_rule(
        df,
        "Non-positive or invalid quantity",
        df["Quantity"].isna()
        | (df["Quantity"] <= 0)
    )
    # Rule 5: Keep only positive prices
    df = apply_rule(
        df,
        "Non-positive or invalid price",
        df["Price"].isna()
        | (df["Price"] <= 0)
    )
    # Rule 6: Exclude known non-product entries.
    # Voucher codes such as GIFT_0001_40 are excluded
    # by their prefix, without excluding DCGS products.
    stock_code = df["StockCode"]

    non_product_mask = (
        stock_code.isin(NON_PRODUCT_CODES)
        | stock_code.str.startswith(
            "GIFT_0001_", na=False
        )
    )
    df = apply_rule(
        df,
        "Known non-product codes",
        non_product_mask
    )
    # Save the cleaning log
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    log_df = pd.DataFrame(cleaning_log)
    log_path = REPORT_DIR / "cleaning_log.csv"
    log_df.to_csv(log_path, index=False)
    print("\n--- Cleaning Summary ---")
    print(log_df.to_string(index=False))
    print(f"\nOriginal rows: {sum(log_df['Rows removed']) + len(df):,}")
    print(f"Clean rows: {len(df):,}")
    print(f"Cleaning log saved to: {log_path}")
    return df, log_df


def split_by_time(df):
    """
    Split cleaned transactions chronologically.

    Test period: final two calendar months of the dataset.
    R2 targets: test products that each customer never bought in training.
    """

    print("\n--- Time-Based Split (DE3) ---")

    # Use the latest transaction date to determine the test period.
    latest_date = df["InvoiceDate"].max()
    test_start = latest_date - pd.DateOffset(months=2)
    train_df = df[
        df["InvoiceDate"] < test_start
    ].copy()
    test_df = df[
        df["InvoiceDate"] >= test_start
    ].copy()
    print(f"Latest transaction date: {latest_date}")
    print(f"Test period starts: {test_start}")
    print(f"Training rows: {len(train_df):,}")
    print(f"Testing rows: {len(test_df):,}")
    print(
        "\nTraining date range:",
        train_df["InvoiceDate"].min(),
        "to",
        train_df["InvoiceDate"].max()
    )
    print(
        "Testing date range:",
        test_df["InvoiceDate"].min(),
        "to",
        test_df["InvoiceDate"].max()
    )
    # Normalize IDs consistently for matching.
    train_df["CustomerKey"] = (
        pd.to_numeric(train_df["Customer ID"])
        .astype("int64")
        .astype(str)
    )
    test_df["CustomerKey"] = (
        pd.to_numeric(test_df["Customer ID"])
        .astype("int64")
        .astype(str)
    )
    train_df["ProductKey"] = (
        train_df["StockCode"].astype(str).str.strip().str.upper()
    )
    test_df["ProductKey"] = (
        test_df["StockCode"].astype(str).str.strip().str.upper()
    )
    # Unique customer-product pairs purchased during training.
    train_pairs = train_df[
        ["CustomerKey", "ProductKey"]
    ].drop_duplicates()
    # Unique customer-product pairs purchased during testing.
    test_pairs = test_df[
        ["CustomerKey", "ProductKey"]
    ].drop_duplicates()
    # R2: keep test products absent from that customer's training history.
    test_targets = (
        test_pairs.merge(
            train_pairs,
            on=["CustomerKey", "ProductKey"],
            how="left",
            indicator=True
        )
        .query("_merge == 'left_only'")
        [["CustomerKey", "ProductKey"]]
        .reset_index(drop=True)
    )
    # Save the train and test transactions without helper columns.
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.drop(
        columns=["CustomerKey", "ProductKey"]
    ).to_csv(
        output_dir / "train_transactions.csv",
        index=False
    )
    test_df.drop(
        columns=["CustomerKey", "ProductKey"]
    ).to_csv(
        output_dir / "test_transactions.csv",
        index=False
    )
    # Save the evaluation targets.
    test_targets.to_csv(
        output_dir / "test_targets.csv",
        index=False
    )
    print("\n--- R2 Evaluation Targets ---")
    print(f"Test customer-product pairs: {len(test_pairs):,}")
    print(f"New-to-customer test pairs: {len(test_targets):,}")
    print(
        "Customers with new test products:",
        test_targets["CustomerKey"].nunique()
    )
    print("\nSaved train_transactions.csv")
    print("Saved test_transactions.csv")
    print("Saved test_targets.csv")

    return train_df, test_df, test_targets





def build_customer_product_matrix():
    """
    Build a sparse customer-product matrix using training data only.

    Rows    = customers
    Columns = products
    Values  = total quantity purchased
    """

    print("\n--- Building Customer-Product Matrix ---")

    # Load training data only
    train_path = (
        PROJECT_ROOT
        / "outputs"
        / "train_transactions.csv"
    )
    train_df = pd.read_csv(train_path)
    print(f"Training transactions loaded: {len(train_df):,}")
    # Ensure consistent customer and product identifiers
    train_df["Customer ID"] = (
        train_df["Customer ID"].astype(int).astype(str)
    )
    train_df["StockCode"] = train_df["StockCode"].astype(str)
    # Aggregate quantities for each customer-product pair
    purchase_data = (
        train_df.groupby(
            ["Customer ID", "StockCode"],
            as_index=False
        )["Quantity"]
        .sum()
    )
    # Create consistent row and column mappings
    customers = sorted(
        purchase_data["Customer ID"].unique()
    )
    products = sorted(
        purchase_data["StockCode"].unique()
    )
    customer_to_index = {
        customer: index
        for index, customer in enumerate(customers)
    }
    product_to_index = {
        product: index
        for index, product in enumerate(products)
    }
    # Convert identifiers into matrix coordinates
    row_indices = (
        purchase_data["Customer ID"]
        .map(customer_to_index)
        .to_numpy()
    )
    col_indices = (
        purchase_data["StockCode"]
        .map(product_to_index)
        .to_numpy()
    )
    values = purchase_data["Quantity"].to_numpy(
        dtype=np.float32
    )
    # Build the CSR sparse matrix
    matrix = csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(len(customers), len(products)),
        dtype=np.float32
    )
    # Save matrix and index mappings
    output_dir = PROJECT_ROOT / "outputs" / "matrix"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_npz(
        output_dir / "customer_product_matrix.npz",
        matrix
    )
    with open(
        output_dir / "customers.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(customers, file)
    with open(
        output_dir / "products.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(products, file)
    # DE5: Calculate matrix statistics
    rows, cols = matrix.shape
    total_cells = rows * cols
    nonzero_entries = matrix.nnz
    density = (
        nonzero_entries / total_cells * 100
        if total_cells > 0
        else 0
    )
    sparsity = 100 - density
    # Memory required by a dense float32 matrix
    dense_bytes = (
        total_cells * np.dtype(np.float32).itemsize
    )
    # Actual storage used by the CSR sparse matrix
    sparse_bytes = (
        matrix.data.nbytes
        + matrix.indices.nbytes
        + matrix.indptr.nbytes
    )
    # Create the DE5 report
    matrix_report = pd.DataFrame([{
        "Customers": rows,
        "Products": cols,
        "Matrix Shape": str(matrix.shape),
        "Total Cells": total_cells,
        "Nonzero Entries": nonzero_entries,
        "Density Percentage": round(density, 4),
        "Sparsity Percentage": round(sparsity, 4),
        "Dense Memory MB": round(
            dense_bytes / (1024 ** 2), 2
        ),
        "Sparse Memory MB": round(
            sparse_bytes / (1024 ** 2), 2
        ),
        "Dense to Sparse Ratio": round(
            dense_bytes / sparse_bytes, 2
        ) if sparse_bytes else None
    }])
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    matrix_report.to_csv(
        REPORT_DIR / "matrix_report.csv",
        index=False
    )
    # Print the results
    print("\n--- DE5: Matrix Statistics ---")
    print(f"Customers: {rows:,}")
    print(f"Products: {cols:,}")
    print(f"Matrix shape: {matrix.shape}")
    print(f"Total cells: {total_cells:,}")
    print(f"Non-zero entries: {nonzero_entries:,}")
    print(f"Density: {density:.4f}%")
    print(f"Sparsity: {sparsity:.4f}%")
    print(f"Dense memory: {dense_bytes / (1024 ** 2):.2f} MB")
    print(f"Sparse memory: {sparse_bytes / (1024 ** 2):.2f} MB")

    if sparse_bytes:
        print(
            f"Dense-to-sparse ratio: "
            f"{dense_bytes / sparse_bytes:.2f}x"
        )
    print(
        f"Matrix saved to: "
        f"{output_dir / 'customer_product_matrix.npz'}"
    )
    print("Customer and product mappings saved successfully.")
    print(
        f"DE5 report saved to: "
        f"{REPORT_DIR / 'matrix_report.csv'}"
    )

    return matrix, customers, products


def main():
    df = load_dataset()
    cleaned_df, cleaning_log = clean_dataset(df)

    print("\n--- Final Validation ---")
    print(
        "Missing customer IDs:",
        cleaned_df["Customer ID"].isna().sum()
    )
    print(
        "Missing descriptions:",
        cleaned_df["Description"].isna().sum()
    )
    print(
        "Non-positive quantities:",
        (cleaned_df["Quantity"] <= 0).sum()
    )
    print(
        "Non-positive prices:",
        (cleaned_df["Price"] <= 0).sum()
    )
    invoice_text = cleaned_df["Invoice"].astype(str).str.strip()
    print(
        "Cancellation invoices:",
        invoice_text.str.upper().str.startswith("C").sum()
    )
    print(
        "Unique customers:",
        cleaned_df["Customer ID"].nunique()
    )
    print(
        "Unique products:",
        cleaned_df["StockCode"].nunique()
    )
    # Save the cleaned transactions
    output_path = (
        PROJECT_ROOT
        / "outputs"
        / "cleaned_transactions.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    print(f"\nCleaned data saved to: {output_path}")
    print(f"Final cleaned rows: {len(cleaned_df):,}")
    # Split the cleaned data chronologically
    train_df, test_df, test_targets = split_by_time(cleaned_df)
    matrix, customers, products = build_customer_product_matrix()

if __name__ == "__main__":
    main()