import json
import re
import ast

def parse_search_results(results):
    """
    Parse search results that might be in string format but contain valid JSON data.
    
    Args:
        results: The search results, which might be a string, list, or other format
        
    Returns:
        The parsed search results as a list of dictionaries if successful, otherwise the original results
    """
    # If already a list, no need to parse
    if isinstance(results, list):
        return results
        
    # If not a string, we can't parse it
    if not isinstance(results, str):
        return results
        
    # Only try to parse if it looks like it might contain JSON
    if '[' not in results or ']' not in results:
        return results
    
    try:
        # First try direct JSON parsing
        try:
            parsed = json.loads(results)
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed
        except json.JSONDecodeError:
            pass
            
        # Try ast.literal_eval for Python literal structures
        try:
            parsed = ast.literal_eval(results)
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed
        except (SyntaxError, ValueError):
            pass
            
        # Try to find and extract a JSON array pattern from the string
        json_pattern = re.compile(r'\[(.*)\]', re.DOTALL)
        match = json_pattern.search(results)
        
        if match:
            # Check if it's a valid JSON array
            json_str = '[' + match.group(1) + ']'
            # Replace single quotes with double quotes if needed
            json_str = json_str.replace("'", '"')
            
            # Try to fix common JSON formatting issues
            # Fix unquoted keys
            json_str = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_str)
            
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, list) and len(parsed) > 0:
                    return parsed
            except json.JSONDecodeError:
                pass
                
        # Try to find individual JSON objects
        parts = re.findall(r'{[^{}]*}', results)
        if parts:
            items = []
            for part in parts:
                try:
                    fixed_part = part.replace("'", '"')
                    # Fix unquoted keys
                    fixed_part = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', fixed_part)
                    items.append(json.loads(fixed_part))
                except json.JSONDecodeError:
                    pass
            if items:
                return items
    
    except Exception:
        pass
        
    # If all parsing attempts fail, return the original results
    return results 