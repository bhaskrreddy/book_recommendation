import re
from datetime import datetime

def extract_year(date_str: str) -> int:
    try:
        # Attempt to parse the date using common formats
        formats = ["%Y-%m-%d", "%Y-%m", "%Y"]
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt).year
            except ValueError:
                continue
        
        # If all else fails, extract the first four digits as the year
        year_match = re.search(r"\d{4}", date_str)
        if year_match:
            return int(year_match.group(0))
        
        # Handle cases where year is not found
        raise ValueError(f"Year not found in the string: {date_str}")

    except Exception as e:
        print(f"Error extracting year from {date_str}: {e}")
        raise
