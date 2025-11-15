"""
General shared helpers (placeholder).
"""
def chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]
