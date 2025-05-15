from difflib import SequenceMatcher
def calculate_similarity( a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()