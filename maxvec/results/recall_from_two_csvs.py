import pandas as pd
import sys
import re

def clean_dataframe(df):
    """
    Cleans the dataframe by:
    1. Stripping leading/trailing spaces from column names.
    2. Removing data type annotations in parentheses (e.g., '(int64)') from column names.
    3. Stripping leading/trailing spaces from any string values in the dataframe.
    """
    df.columns = df.columns.str.replace(r'\([^)]*\)', '', regex=True).str.strip()
    
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].str.strip()
        
    return df

def calculate_recall(ground_truth_file, retrieved_file, query_col, nn_col, recall_at_k=None):
    """
    Calculates the recall for each query given a ground truth CSV and a retrieved CSV.
    """
    df_truth = pd.read_csv(ground_truth_file)
    df_retrieved = pd.read_csv(retrieved_file)

    df_truth = clean_dataframe(df_truth)
    df_retrieved = clean_dataframe(df_retrieved)

    # Optional: If you only want Recall@K (e.g., Top 10), you can sort by distance and limit here
    if recall_at_k is not None:
        df_truth = df_truth.groupby(query_col).head(recall_at_k)
        df_retrieved = df_retrieved.groupby(query_col).head(recall_at_k)

    truth_grouped = df_truth.groupby(query_col)[nn_col].apply(set).reset_index()
    retrieved_grouped = df_retrieved.groupby(query_col)[nn_col].apply(set).reset_index()

    truth_grouped.rename(columns={nn_col: 'truth_nn'}, inplace=True)
    retrieved_grouped.rename(columns={nn_col: 'retrieved_nn'}, inplace=True)

    merged = pd.merge(truth_grouped, retrieved_grouped, on=query_col, how='left')
    merged['retrieved_nn'] = merged['retrieved_nn'].apply(lambda x: x if isinstance(x, set) else set())

    def compute_recall(row):
        truth_set = row['truth_nn']
        retrieved_set = row['retrieved_nn']
        
        if len(truth_set) == 0:
            return 0.0
            
        intersection = truth_set.intersection(retrieved_set)
        return len(intersection) / len(truth_set)

    merged['recall'] = merged.apply(compute_recall, axis=1)
    mean_recall = merged['recall'].mean()

    return merged[[query_col, 'recall']], mean_recall


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python recall_from_two_csvs.py <ground_truth_csv> <retrieved_csv>")
        sys.exit(1)
        
    GROUND_TRUTH_CSV = sys.argv[1]
    RETRIEVED_CSV = sys.argv[2]
    
    # --- Auto-Detect K from filename ---
    # Looks for the pattern "-k_100_" in the retrieved filename
    match = re.search(r'-k_(\d+)_', RETRIEVED_CSV)
    if match:
        RECALL_AT_K = int(match.group(1))
    else:
        print(f"Warning: Could not extract 'k' from filename {RETRIEVED_CSV}. Defaulting to k=100.")
        RECALL_AT_K = 100
    
    # --- Auto-Detect Column Names ---
    try:
        header_df = pd.read_csv(GROUND_TRUTH_CSV, nrows=0)
        clean_columns = header_df.columns.str.replace(r'\([^)]*\)', '', regex=True).str.strip()
        
        if 'rv_reviewkey_queries' in clean_columns:
            QUERY_ID_COLUMN = 'rv_reviewkey_queries'
            NEIGHBOR_ID_COLUMN = 'rv_reviewkey'
        elif 'i_imagekey_queries' in clean_columns:
            QUERY_ID_COLUMN = 'i_imagekey_queries'
            NEIGHBOR_ID_COLUMN = 'i_imagekey'
        else:
            if 'images' in GROUND_TRUTH_CSV.lower():
                QUERY_ID_COLUMN = 'i_imagekey_queries'
                NEIGHBOR_ID_COLUMN = 'i_imagekey'
            elif 'reviews' in GROUND_TRUTH_CSV.lower():
                QUERY_ID_COLUMN = 'rv_reviewkey_queries'
                NEIGHBOR_ID_COLUMN = 'rv_reviewkey'
            else:
                raise ValueError("Could not auto-detect if the dataset is 'reviews' or 'images'.")
                
    except Exception as e:
        print(f"Error determining dataset columns: {e}")
        sys.exit(1)
    
    try:
        recall_df, avg_recall = calculate_recall(
            ground_truth_file=GROUND_TRUTH_CSV, 
            retrieved_file=RETRIEVED_CSV, 
            query_col=QUERY_ID_COLUMN, 
            nn_col=NEIGHBOR_ID_COLUMN,
            recall_at_k=RECALL_AT_K
        )
        
        print(f"\nTarget Dataset Config: Query={QUERY_ID_COLUMN} | Target={NEIGHBOR_ID_COLUMN} | K={RECALL_AT_K}")
        print("Recall per query - Best:")
        print(recall_df.sort_values(by='recall', ascending=False).head(10)) 
        print("\nRecall per query - Worst:")
        print(recall_df.sort_values(by='recall', ascending=True).head(10))  
        print("\nRecall of the first query:")
        print(recall_df.iloc[0])
        
        print(f"\nOverall Average Recall@{RECALL_AT_K}: {avg_recall:.4f}\n")
        
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the files exist in the same directory.")