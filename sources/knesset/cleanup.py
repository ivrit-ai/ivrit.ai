import numpy as np
from scipy.stats import rv_histogram


def calc_diff_stats(input, confidence=0.98):
    """
    Calculate statistics for differences between consecutive values in the input array.

    Args:
        input: Input time series array
        confidence: Confidence interval for determining suspect values (default: 0.98)

    Returns:
        diffs: Array of differences between consecutive values
        positive_diffs: Array of positive differences only
        suspect_above: Upper threshold for suspect differences
        suspect_below: Lower threshold for suspect differences
        hist_dist: Histogram distribution of differences
    """
    diffs = np.diff(input)
    positive_diffs = np.clip(diffs, 0, np.inf)
    diffs_std = np.std(positive_diffs)
    values_within_one_std = np.array([v for v in positive_diffs if v < diffs_std])
    hist = np.histogram(values_within_one_std, bins=100)
    hist_dist = rv_histogram(hist, density=False)
    _, suspect_above = hist_dist.interval(confidence)
    return diffs, suspect_above, hist_dist


def fix_abnormal_diffs(
    pre, diffs, suspect_above, hist_dist, magnitude_ratio_similarity_threshold=1.3
):
    """
    Fix abnormal differences by looking for counter-differences to cancel them out.

    Args:
        pre: Original time series array
        diffs: Array of differences between consecutive values
        suspect_above: Upper threshold for suspect differences
        hist_dist: Histogram distribution of differences
        magnitude_ratio_similarity_threshold: Maximum magnitude ratio to consider two diffs as offsetting each other

    Returns:
        post: Time series with abnormal differences fixed
    """
    # Fix loop
    current_diff_idx = 0
    while current_diff_idx < len(diffs):
        is_abnormal_diff = (
            diffs[current_diff_idx] > suspect_above or diffs[current_diff_idx] < 0
        )
        if is_abnormal_diff:
            abnormal_diff_amount = diffs[current_diff_idx]  # negative or positive
            abnormal_diff_amount_magnitude = abs(abnormal_diff_amount)
            search_counter_abnormal_diff_idx = current_diff_idx + 1
            max_search_distance = np.ceil(
                abnormal_diff_amount_magnitude / hist_dist.median()
            )
            search_until_idx = current_diff_idx + max_search_distance
            # Search forward for an offsetting candidate
            while search_counter_abnormal_diff_idx < min(len(diffs), search_until_idx):
                diff_amount_at_current_search_idx = diffs[
                    search_counter_abnormal_diff_idx
                ]

                # Only opposite signs will cancel each other
                if diff_amount_at_current_search_idx != 0 and np.sign(
                    diff_amount_at_current_search_idx
                ) != np.sign(abnormal_diff_amount):
                    amount_after_canceling_each_other = (
                        abnormal_diff_amount + diff_amount_at_current_search_idx
                    )

                    # How similar are the diffs? (in relative terms)
                    magnitude_ratio = np.exp(
                        abs(
                            np.log(
                                abs(
                                    abnormal_diff_amount
                                    / diff_amount_at_current_search_idx
                                    + 1e-6
                                )
                            )
                        )
                    )

                    if magnitude_ratio < magnitude_ratio_similarity_threshold:
                        replacement_value_for_each_diff = (
                            amount_after_canceling_each_other / 2
                        )
                        diffs[current_diff_idx] = np.ceil(
                            replacement_value_for_each_diff
                        )
                        diffs[search_counter_abnormal_diff_idx] = np.floor(
                            replacement_value_for_each_diff
                        )
                        break
                search_counter_abnormal_diff_idx += 1
        current_diff_idx += 1

    post = np.cumsum(np.concatenate([pre[:1], diffs]))
    return post


def fix_with_sliding_window_sorting(time_series, hist_dist, estimated_window_time_duration=40):
    """
    Fix time series using sliding window approach to sort time indices within windows.

    Args:
        time_series: Time series array from previous phase
        hist_dist: Histogram distribution of differences

    Returns:
        fixed_time_series: Time series with sliding window fixes applied
    """
    fixed_time_series = time_series.copy()

    # Calculate window parameters based on histogram statistics
    sort_window_size = int(np.ceil(estimated_window_time_duration / hist_dist.mean()))
    sort_window_hop_size = max(1, sort_window_size // 3)

    for window_start_idx in range(
        0, len(fixed_time_series) - sort_window_size, sort_window_hop_size
    ):
        window_slice = slice(window_start_idx, window_start_idx + sort_window_size)
        window_diffs = np.diff(fixed_time_series[window_slice])

        # If nothing to fix - move along
        if np.all(window_diffs >= 0):
            continue

        sorted_window = np.sort(fixed_time_series[window_slice])
        sorted_window_diffs = np.diff(sorted_window)

        if np.all(sorted_window_diffs >= 0):
            # Commit to it
            fixed_time_series[window_slice] = sorted_window

    return fixed_time_series


def fix_remaining_negative_diffs(time_series):
    """
    Fix remaining negative differences by redistributing them to positive differences.

    Args:
        time_series: Time series array from previous phase

    Returns:
        fixed_time_series: Time series with all negative differences fixed
    """
    fixed_diffs = np.diff(time_series)
    next_to_fix = np.argmin(fixed_diffs)

    while fixed_diffs[next_to_fix] < 0:
        search_radius = 1

        # Find the smallest window that can offset the negative
        while True:
            window_slice = slice(
                next_to_fix - search_radius, next_to_fix + search_radius + 1
            )
            diffs_window = fixed_diffs[window_slice]
            # If enough positive can offset the negative - window is good
            if diffs_window.sum() >= 0:
                break
            search_radius += 1

        positive_diffs = np.where(diffs_window > 0, diffs_window, 0)
        positive_offset_contribution_weights = positive_diffs / positive_diffs.sum()
        amount_to_offset = fixed_diffs[next_to_fix]

        # Assign contribution amount to each positive offset ensuring only integers are used
        left_to_distribute = amount_to_offset  # track how much is left to distribute
        positive_offset_contributions = np.zeros_like(
            positive_offset_contribution_weights, dtype=np.int64
        )
        for contribution_idx in range(len(positive_offset_contributions) - 1):
            weight = positive_offset_contribution_weights[contribution_idx]
            contribution_rounded = np.round(weight * amount_to_offset)
            left_to_distribute -= contribution_rounded
            positive_offset_contributions[contribution_idx] = int(contribution_rounded)

        positive_offset_contributions[-1] = left_to_distribute
        fixed_diffs[next_to_fix] = 0  # offset
        # Collect contributions (add, since it's the opposite)
        fixed_diffs[window_slice] += positive_offset_contributions
        next_to_fix = np.argmin(fixed_diffs)

    # Reconstruct fixed time series from fixed diffs
    fixed_time_series = np.cumsum(np.concatenate([time_series[:1], fixed_diffs]))
    return fixed_time_series


def cleanup_time_index(time_pointer_array, stats_confidence=0.95):
    """
    Clean up a time series array by fixing abnormal differences, applying sliding window fixes,
    and redistributing remaining negative differences.

    Args:
        time_pointer_array: Input time series array to clean up

    Returns:
        cleaned_time_series: Fully cleaned time series
    """
    # Make a copy of the input array
    pre = time_pointer_array.copy()

    # Replace initial zero with minimal non-zero value
    pre[0] = np.min(np.where(pre > 0, pre, np.inf))

    # Calculate difference statistics
    diffs, suspect_above, hist_dist = calc_diff_stats(pre, stats_confidence)

    # Phase 1: Fix abnormal differences
    phase_1_fixed = fix_abnormal_diffs(pre, diffs, suspect_above, hist_dist)

    # Phase 2: Apply sliding window fixes
    phase_2_fixed = fix_with_sliding_window_sorting(phase_1_fixed, hist_dist)

    # Phase 3: Fix remaining negative differences
    phase_3_fixed = fix_remaining_negative_diffs(phase_2_fixed)

    return phase_3_fixed
