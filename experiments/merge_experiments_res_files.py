import glob
import pandas as pd
import os
import argparse

def concatenate_nested_csvs(target_folder, nodes):
    """
    Recursively finds all CSV results within the specified target_folder,
    merges them into a master DataFrame, and saves it.
    """
    # Build the pattern relative to the target_folder provided
    search_path = os.path.join(target_folder, "**", f"*{nodes}nodes_*.csv")
    
    print(f"🔍 Searching in: {os.path.abspath(target_folder)}")
    
    files = glob.glob(search_path, recursive=True)

    if not files:
        print(f"❌ Error: No files found in '{target_folder}' matching pattern '{nodes}nodes_*.csv'")
        return

    print(f"✅ Found {len(files)} files to combine!")
        
    try:
        dataframes = []
        for f in files:
            temp_df = pd.read_csv(f)
            
            # Keep track of the source and density for data traceability
            parent_folder = os.path.basename(os.path.dirname(f))
            temp_df['density_folder'] = parent_folder
            temp_df['source_file'] = f 
            
            dataframes.append(temp_df)
            
        combined_df = pd.concat(dataframes, ignore_index=True)
        combined_df.drop_duplicates(inplace=True)
        
        # Save the master file inside the target folder
        output_filename = os.path.join(target_folder, f"MASTER_grid_results_{nodes}nodes_ALL.csv")
        combined_df.to_csv(output_filename, index=False)
        
        print(f"\n🎉 Success! All files concatenated.")
        print(f"💾 Saved as: '{output_filename}'")
        print(f"📊 Total rows in master file: {len(combined_df)}")
        
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge grid result CSVs from a specific folder.")
    parser.add_argument("--folder", required=True, help="The target directory to search for CSVs.")
    parser.add_argument("--nodes", type=int, default=15, help="Number of nodes to filter by (default: 15).")
    
    args = parser.parse_args()
    
    concatenate_nested_csvs(args.folder, args.nodes)