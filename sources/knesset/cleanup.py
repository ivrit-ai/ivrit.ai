import numpy as np

def fix_remaining_negative_diffs(time_series):
    """
    Fix remaining negative differences by redistributing them to positive differences.
    Uses a greedy approach that starts with the largest positive difference first.

    Args:
        time_series: Time series array from previous phase

    Returns:
        fixed_time_series: Time series with all negative differences fixed
    """
    fixed_diffs = np.diff(time_series)
    next_to_fix = np.argmin(fixed_diffs)

    while fixed_diffs[next_to_fix] < 0:
        search_radius = 1
        last_idx = len(fixed_diffs) - 1
        search_aborted = False
        
        # Find the smallest window that can offset the negative
        while True:
            window_slice = slice(
                max(0, next_to_fix - search_radius), min(last_idx, next_to_fix + search_radius + 1)
            )
            diffs_window = fixed_diffs[window_slice]
            # If enough positive can offset the negative - window is good
            if diffs_window.sum() >= 0:
                break
            search_radius += 1
            
            if next_to_fix - search_radius < 0 and next_to_fix + search_radius + 1 > last_idx:
                search_aborted = True
                break
        
        if search_aborted:
            # Last resort - we cancel the negative diff
            # and hope the positive shift won't be too bad
            fixed_diffs[next_to_fix] = 0
            break
        
        # Get the amount we need to offset
        amount_to_offset = abs(fixed_diffs[next_to_fix])
        
        # Create array to track the contributions
        positive_offset_contributions = np.zeros_like(diffs_window)
        
        # Get indices of positive differences sorted by size (largest first)
        positive_indices = np.where(diffs_window > 0)[0]
        sorted_indices = positive_indices[np.argsort(-diffs_window[positive_indices])]
        
        # Distribute the negative amount to offset starting with largest positive values
        remaining_to_offset = amount_to_offset
        for idx in sorted_indices:
            # How much we can take from this positive difference
            contribution = min(remaining_to_offset, diffs_window[idx])
            positive_offset_contributions[idx] = -contribution  # Negative because we're reducing the positive
            remaining_to_offset -= contribution
            
            # If we've offset the entire negative amount, we're done
            if remaining_to_offset <= 0:
                break
        
        # Set the negative difference to zero
        fixed_diffs[next_to_fix] = 0
        
        # Apply the contributions to the window
        fixed_diffs[window_slice] += positive_offset_contributions
        
        # Find the next negative difference to fix
        next_to_fix = np.argmin(fixed_diffs)

    # Reconstruct fixed time series from fixed diffs
    fixed_time_series = np.cumsum(np.concatenate([time_series[:1], fixed_diffs]))
    return fixed_time_series


def cleanup_time_index(time_pointer_array):
    """
    Clean up a time series array by redistributing negative differences.

    Args:
        time_pointer_array: Input time series array to clean up

    Returns:
        cleaned_time_series: Fully cleaned time series
    """
    # Make a copy of the input array
    pre = time_pointer_array.copy()

    # Replace initial zero with minimal non-zero value
    pre[0] = np.min(np.where(pre > 0, pre, np.inf))

    return fix_remaining_negative_diffs(pre)
