#!/usr/bin/env python
"""Test script to validate the atonal implementation."""

from atonal.base import (
    validate_prime_forms,
    multiply,
    contains_abstract_subset,
    get_k_complex_size,
    prime_form,
)

print("=" * 70)
print("Testing new functions")
print("=" * 70)

# Test multiply()
print("\n1. Testing multiply() function:")
result = multiply((0, 4, 7), 5)
expected = frozenset({0, 8, 11})
print(f"   multiply((0, 4, 7), 5) = {result}")
print(f"   Expected: {expected}")
print(f"   ✓ PASS" if result == expected else f"   ✗ FAIL")

result = multiply((0, 1, 2), 5)
expected = frozenset({0, 5, 10})
print(f"   multiply((0, 1, 2), 5) = {result}")
print(f"   Expected: {expected}")
print(f"   ✓ PASS" if result == expected else f"   ✗ FAIL")

# Test contains_abstract_subset()
print("\n2. Testing contains_abstract_subset() function:")
result = contains_abstract_subset((0, 4, 7), (0, 3))
print(f"   contains_abstract_subset((0, 4, 7), (0, 3)) = {result}")
print(f"   Expected: True (minor 3rd exists in major triad)")
print(f"   ✓ PASS" if result == True else f"   ✗ FAIL")

result = contains_abstract_subset((0, 4, 7), (0, 5))
print(f"   contains_abstract_subset((0, 4, 7), (0, 5)) = {result}")
print(f"   Expected: True (perfect 4th exists 7->0 in major triad)")
print(f"   ✓ PASS" if result == True else f"   ✗ FAIL")

result = contains_abstract_subset((0, 4, 7), (0, 2))
print(f"   contains_abstract_subset((0, 4, 7), (0, 2)) = {result}")
print(f"   Expected: False (major 2nd does NOT exist in major triad)")
print(f"   ✓ PASS" if result == False else f"   ✗ FAIL")

result = contains_abstract_subset((0, 2, 4, 5, 7, 9, 11), (0, 4, 7))
print(f"   contains_abstract_subset((0, 2, 4, 5, 7, 9, 11), (0, 4, 7)) = {result}")
print(f"   Expected: True (major triad in major scale)")
print(f"   ✓ PASS" if result == True else f"   ✗ FAIL")

# Test get_k_complex_size() - just make sure it runs
print("\n3. Testing get_k_complex_size() function:")
print(
    "   NOTE: This is a computationally expensive function, testing with small set..."
)
result = get_k_complex_size((0, 6))
print(f"   get_k_complex_size((0, 6)) = {result}")
print(f"   ✓ Function executes successfully")

# Validate prime forms
print("\n" + "=" * 70)
print("Validating prime forms against Forte canonical forms")
print("=" * 70)
print("\nThis will check all 208 Forte set classes (cardinality 3-9)...")
print("Building nodes DataFrame (this may take a moment)...\n")

result = validate_prime_forms()

if result is None:
    print("\n✓ All validation tests passed!")
else:
    print(f"\n✗ Found {len(result)} discrepancies:")
    print(result)

print("\n" + "=" * 70)
print("Testing complete")
print("=" * 70)
