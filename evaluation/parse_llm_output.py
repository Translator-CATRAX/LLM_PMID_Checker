#!/usr/bin/env python3
"""
Parse LLM output from main.py to extract detailed evaluation information.
"""

import re
import sys


def parse_llm_output(output_text):
    """
    Parse the output from main.py to extract evaluation details.
    
    Returns a dict with:
    - is_supported: bool (True/False)
    - evidence_category: str
    - subject_mentioned: bool
    - object_mentioned: bool
    - supporting_sentence: str
    - reasoning: str
    """
    result = {
        'is_supported': None,
        'evidence_category': 'Unknown',
        'subject_mentioned': False,
        'object_mentioned': False,
        'supporting_sentence': '',
        'reasoning': ''
    }
    
    # Look for the evaluation result line (use the LAST match to get final/verified result)
    # Format: "PMID:31095444, Supported, Direct support, Subject:Yes, Object:Yes, [supporting sentence]"
    pattern = r'PMID:\d+,\s*(Supported|Not\s+Supported),\s*([^,]+),\s*Subject:(Yes|No),\s*Object:(Yes|No)(?:,\s*\[(.*?)\])?'
    
    # Find all matches and use the last one (which would be the verified result if verification was used)
    matches = list(re.finditer(pattern, output_text, re.MULTILINE | re.DOTALL))
    match = matches[-1] if matches else None
    
    if match:
        result['is_supported'] = match.group(1).strip() == 'Supported'
        result['evidence_category'] = match.group(2).strip()
        result['subject_mentioned'] = match.group(3).strip() == 'Yes'
        result['object_mentioned'] = match.group(4).strip() == 'Yes'
        # Sanitize tabs and newlines in supporting sentence
        supporting_sentence = match.group(5).strip() if match.group(5) else ''
        result['supporting_sentence'] = supporting_sentence.replace('\t', ' ').replace('\n', ' ') if supporting_sentence else ''
    else:
        # Fallback: try simpler pattern matching
        if re.search(r'Supported', output_text, re.IGNORECASE):
            result['is_supported'] = True
        elif re.search(r'Not\s+Supported', output_text, re.IGNORECASE):
            result['is_supported'] = False
    
    # Extract reasoning if present (verbose output may contain "Reasoning:")
    # Use the last match to get the final/verified reasoning
    reasoning_matches = list(re.finditer(r'Reasoning:\s*(.+?)(?:\n\n|\Z)', output_text, re.DOTALL | re.IGNORECASE))
    if reasoning_matches:
        reasoning = reasoning_matches[-1].group(1).strip()
        # Sanitize tabs and newlines in reasoning
        result['reasoning'] = reasoning.replace('\t', ' ').replace('\n', ' ') if reasoning else ''
    
    return result


def format_tsv_line(subject, predicate, object_name, ground_truth, pmid, result_dict, subject_curie='', object_curie=''):
    """
    Format a TSV line with all evaluation details.
    """
    is_supported = 'True' if result_dict['is_supported'] else ('False' if result_dict['is_supported'] is not None else 'Unknown')
    evidence_category = result_dict['evidence_category']
    subject_mentioned = 'Yes' if result_dict['subject_mentioned'] else 'No'
    object_mentioned = 'Yes' if result_dict['object_mentioned'] else 'No'
    supporting_sentence = result_dict['supporting_sentence'].replace('\t', ' ').replace('\n', ' ')
    reasoning = result_dict['reasoning'].replace('\t', ' ').replace('\n', ' ')
    
    return '\t'.join([
        subject,
        predicate,
        object_name,
        ground_truth,
        pmid,
        is_supported,
        evidence_category,
        subject_mentioned,
        object_mentioned,
        supporting_sentence,
        reasoning,
        subject_curie,
        object_curie
    ])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_llm_output.py <output_text | ->")
        sys.exit(1)
    
    if sys.argv[1] == '-':
        output_text = sys.stdin.read()
    else:
        output_text = sys.argv[1]
    result = parse_llm_output(output_text)
    
    # Print as JSON for easy parsing in bash
    import json
    print(json.dumps(result))

