import pandas as pd

def compare_metrics(run_a_metrics, run_b_metrics, preferred_order, run_ids):
    # Create comparison DataFrame for metrics
    metrics_comparison = []

    # Get all unique keys from both metrics
    all_keys = set()
    def extract_keys(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                extract_keys(v, prefix + k + ".")
            else:
                all_keys.add(prefix + k)
    
    extract_keys(run_a_metrics)
    extract_keys(run_b_metrics)
    
    # Function to get nested value
    def get_nested_value(d, key):
        keys = key.split('.')
        value = d
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return "N/A"
    
    def format_value(value):
        # Format values (round floats to 2 decimals)
        if isinstance(value, float):
            return round(value, 2)
        return value
    
    # Create ordered list: first preferred order (if available), then remaining keys
    ordered_keys = []
    for key in preferred_order:
        if key in all_keys:
            ordered_keys.append(key)
    
    # Add remaining keys not in preferred order
    remaining_keys = sorted(all_keys - set(ordered_keys))
    ordered_keys.extend(remaining_keys)
    
    # Compare metrics
    for key in ordered_keys:
        value_a = get_nested_value(run_a_metrics, key)
        value_b = get_nested_value(run_b_metrics, key)
        
        # Format values
        formatted_value_a = format_value(value_a)
        formatted_value_b = format_value(value_b)
        
        # Determine if values are different
        is_different = str(formatted_value_a) != str(formatted_value_b)
        
        metrics_comparison.append({
            'Metric': key,
            run_ids[0]: str(formatted_value_a),
            run_ids[1]: str(formatted_value_b),
        })
    
    return pd.DataFrame(metrics_comparison)