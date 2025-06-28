def binary_search(arr: list, target: int | float) -> int:
    """
    Performs binary search on a sorted list.

    Args:
        arr (list): The sorted list of elements.
        target (int | float): The value to search for.

    Returns:
        int: The index of the target if found, otherwise -1.
    """
    left = 0
    right = len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid  # Found
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1  # Not found


# Example usage:
if __name__ == "__main__":
    data = [1, 3, 5, 7, 9, 11, 13, 15]
    target = 7

    result = binary_search(data, target)
    if result != -1:
        print(f"Found {target} at index {result}")
    else:
        print(f"{target} not found in the array.")
