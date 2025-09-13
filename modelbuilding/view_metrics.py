import pickle
import os

MODEL_DIR = 'processed_data/saved_models'
METRICS_FILE = os.path.join(MODEL_DIR, 'trained_models.pkl')

def load_and_print_metrics(file_path):
    if not os.path.exists(file_path):
        print(f"Error: Metrics file not found at {file_path}")
        return

    try:
        with open(file_path, 'rb') as f:
            trained_models_metrics = pickle.load(f)
        
        print("="*50)
        print("SAVED MODEL PERFORMANCE METRICS")
        print("="*50)
        
        for horizon_name, metrics in trained_models_metrics.items():
            print(f"\n--- {horizon_name.upper()} ---")
            # Ensure metrics are not None and are float before formatting
            mae_overall = metrics.get('mae')
            rmse_overall = metrics.get('rmse')
            r2_overall = metrics.get('r2_score')

            print(f"  Overall MAE: {mae_overall:.2f} km/h" if mae_overall is not None else "  Overall MAE: N/A")
            print(f"  Overall RMSE: {rmse_overall:.2f} km/h" if rmse_overall is not None else "  Overall RMSE: N/A")
            print(f"  Overall R2 Score: {r2_overall:.3f}" if r2_overall is not None else "  Overall R2 Score: N/A")
            
            print("\n  Step-wise Performance:")
            step_maes = metrics.get('step_maes', [])
            step_rmses = metrics.get('step_rmses', [])
            step_r2_scores = metrics.get('step_r2_scores', [])
            
            if not step_maes: # Check if lists are empty
                print("    No step-wise metrics found.")
                continue

            # Determine the number of steps, ensure it's consistent or take min length
            num_steps_mae = len(step_maes)
            num_steps_rmse = len(step_rmses)
            num_steps_r2 = len(step_r2_scores)
            num_steps = min(num_steps_mae, num_steps_rmse, num_steps_r2) # Use the smallest to avoid index errors if lists differ in length

            if num_steps == 0 and (num_steps_mae > 0 or num_steps_rmse > 0 or num_steps_r2 > 0) : # if one list had data but others not, leading to num_steps = 0
                 print("    Step-wise metrics lists have inconsistent lengths or some are empty.")


            print(f"    {'Step':<5} | {'MAE (km/h)':<12} | {'RMSE (km/h)':<13} | {'R2 Score':<10}")
            print(f"    {'-'*5} | {'-'*12} | {'-'*13} | {'-'*10}")
            for i in range(num_steps):
                mae_val = f"{step_maes[i]:.2f}" if i < num_steps_mae and step_maes[i] is not None else "N/A"
                rmse_val = f"{step_rmses[i]:.2f}" if i < num_steps_rmse and step_rmses[i] is not None else "N/A"
                r2_val = f"{step_r2_scores[i]:.3f}" if i < num_steps_r2 and step_r2_scores[i] is not None else "N/A"
                print(f"    {i+1:<5} | {mae_val:<12} | {rmse_val:<13} | {r2_val:<10}")
        
    except Exception as e:
        print(f"Error loading or printing metrics: {e}")

if __name__ == '__main__':
    load_and_print_metrics(METRICS_FILE) 