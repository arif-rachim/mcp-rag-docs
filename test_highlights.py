#!/usr/bin/env python3
"""
Test script to verify keyword highlighting works for English and Arabic queries
"""

import re
from typing import List

def extract_highlighted_terms(query: str, text: str) -> List[str]:
    """
    Extract query terms that appear in the text for highlighting.
    Supports both English and Arabic text.
    """
    # Tokenize query (handles Arabic and English)
    query_tokens = set(re.findall(r'\w+', query.lower()))

    # Tokenize text
    text_tokens = set(re.findall(r'\w+', text.lower()))

    # Find matching terms
    matched_terms = query_tokens & text_tokens

    # Return sorted list for consistency
    return sorted(matched_terms) if matched_terms else None


# Test cases
test_cases = [
    {
        "query": "ammunition storage requirements",
        "text": "Ammunition storage facilities must maintain minimum safety distances of 300 meters from inhabited buildings.",
        "expected": ["ammunition", "storage"]
    },
    {
        "query": "fire suppression system",
        "text": "All storage areas require fire suppression systems and regular safety inspections every 90 days.",
        "expected": ["fire", "suppression", "system"]
    },
    {
        "query": "متطلبات تخزين الذخيرة",
        "text": "يجب أن تحافظ مرافق تخزين الذخيرة على مسافات أمان لا تقل عن 300 متر من المباني المأهولة.",
        "expected": ["الذخيرة", "تخزين"]
    },
    {
        "query": "نظام مكافحة الحريق",
        "text": "تتطلب جميع مناطق التخزين أنظمة مكافحة الحريق وعمليات تفتيش منتظمة للسلامة كل 90 يومًا.",
        "expected": ["الحريق", "مكافحة", "نظام"]
    },
    {
        "query": "safety inspection procedures",
        "text": "Environmental impact assessments must be conducted for new facilities.",
        "expected": None  # No matches
    }
]

print("Testing Keyword Highlighting\n" + "="*80)

for i, test in enumerate(test_cases, 1):
    query = test["query"]
    text = test["text"]
    expected = test["expected"]

    result = extract_highlighted_terms(query, text)

    print(f"\nTest {i}:")
    print(f"Query: {query}")
    print(f"Text: {text[:80]}...")
    print(f"Expected: {expected}")
    print(f"Got: {result}")

    # Check if result matches expected
    if expected is None:
        status = "PASS" if result is None else "FAIL"
    else:
        status = "PASS" if set(result or []) == set(expected) else "FAIL"

    print(f"Status: {status}")

print("\n" + "="*80)
print("All tests completed!")
