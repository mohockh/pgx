"""
Test suite for uint32[2] cardset operations.

This ensures that uint32[2] cardset operations work correctly for poker hand evaluation.
"""

import jax
import jax.numpy as jnp
import pytest
from ..cardset_ops import *


class TestCardsetOperations:
    """Test uint32[2] cardset operations."""
    
    def test_create_empty_cardset(self):
        """Test empty cardset creation."""
        empty = create_empty_cardset()
        
        assert empty.shape == (2,)
        assert empty.dtype == jnp.uint32
        assert jnp.all(empty == 0)
        
    def test_create_cardset_from_value(self):
        """Test cardset creation from various values."""
        test_values = [
            0,
            1,
            0xFF,
            0xFFFF,
            0xFFFFFFFF,
            0x12345678,
            0x123456789ABCDEF0
        ]
        
        for value in test_values:
            # Pass Python int directly - no need for JAX 64-bit arrays
            cardset = create_cardset_from_value(value)
            
            # Verify cardset format
            assert cardset.shape == (2,)
            assert cardset.dtype == jnp.uint32
            
            # Verify round-trip conversion
            reconstructed = cardset_to_uint64(cardset)
            assert int(reconstructed) == value, f"Value {hex(value)} failed round-trip"
            
    def test_cardset_or(self):
        """Test bitwise OR operations."""
        test_pairs = [
            (0x0, 0x0),
            (0x1, 0x0),
            (0xFF, 0xFF00),
            (0x12345678, 0x87654321),
            (0xFFFF0000, 0x0000FFFF),
            (0x55555555, 0xAAAAAAAA)
        ]
        
        for a_val, b_val in test_pairs:
            a_cardset = create_cardset_from_value(a_val)
            b_cardset = create_cardset_from_value(b_val)
            
            result = cardset_or(a_cardset, b_cardset)
            result_64 = cardset_to_uint64(result)
            expected = a_val | b_val
            
            assert result_64 == expected, f"OR({hex(a_val)}, {hex(b_val)}) failed"
            
    def test_cardset_and(self):
        """Test bitwise AND operations."""
        test_pairs = [
            (0x0, 0x0),
            (0xFFFFFFFF, 0x0),
            (0xFF, 0xFF00),
            (0x12345678, 0x87654321),
            (0xFFFF0000, 0x0000FFFF),
            (0x55555555, 0xAAAAAAAA)
        ]
        
        for a_val, b_val in test_pairs:
            a_cardset = create_cardset_from_value(a_val)
            b_cardset = create_cardset_from_value(b_val)
            
            result = cardset_and(a_cardset, b_cardset)
            result_64 = cardset_to_uint64(result)
            expected = a_val & b_val
            
            assert result_64 == expected, f"AND({hex(a_val)}, {hex(b_val)}) failed"
            
    def test_cardset_not(self):
        """Test bitwise NOT operations."""
        test_values = [
            0x0,
            0x1,
            0xFF,
            0xFFFF,
            0xFFFFFFFF,
            0x12345678,
            0x55555555
        ]
        
        for value in test_values:
            cardset = create_cardset_from_value(value)
            
            result = cardset_not(cardset)
            result_64 = cardset_to_uint64(result)
            expected = (~value) & 0xFFFFFFFFFFFFFFFF  # Mask to 64 bits
            
            assert result_64 == expected, f"NOT({hex(value)}) failed"
            
    def test_left_shift(self):
        """Test left shift operations."""
        test_cases = [
            (0x1, 0),
            (0x1, 1),
            (0x1, 31),
            (0x1, 32),
            (0x1, 33),
            (0x1, 63),
            (0x1, 64),
            (0x12345678, 0),
            (0x12345678, 4),
            (0x12345678, 32)
        ]
        
        for value, shift in test_cases:
            cardset = create_cardset_from_value(value)
            
            result = cardset_lshift(cardset, shift)
            result_64 = cardset_to_uint64(result)
            
            # Calculate expected result with proper overflow handling
            if shift >= 64:
                expected = 0
            else:
                expected = (value << shift) & 0xFFFFFFFFFFFFFFFF
            
            assert result_64 == expected, f"LSHIFT({hex(value)}, {shift}) failed: got {hex(int(result_64))}, expected {hex(expected)}"
            
    def test_right_shift(self):
        """Test right shift operations."""
        test_cases = [
            (0x8000000000000000, 0),
            (0x8000000000000000, 1),
            (0x8000000000000000, 31),
            (0x8000000000000000, 32),
            (0x8000000000000000, 33),
            (0x8000000000000000, 63),
            (0x8000000000000000, 64),
            (0x12345678, 0),
            (0x12345678, 4),
            (0x12345678, 32)
        ]
        
        for value, shift in test_cases:
            cardset = create_cardset_from_value(value)
            
            result = cardset_rshift(cardset, shift)
            result_64 = cardset_to_uint64(result)
            
            # Calculate expected result
            if shift >= 64:
                expected = 0
            else:
                expected = value >> shift
            
            assert result_64 == expected, f"RSHIFT({hex(value)}, {shift}) failed: got {hex(int(result_64))}, expected {hex(expected)}"
            
    def test_set_bit(self):
        """Test bit setting operations."""
        test_cases = [
            (0x0, 0),
            (0x0, 1),
            (0x0, 31),
            (0x0, 32),
            (0x0, 63),
            (0x12345678, 0),
            (0x12345678, 15),
            (0x12345678, 32),
            (0x12345678, 47)
        ]
        
        for value, bit_pos in test_cases:
            cardset = create_cardset_from_value(value)
            
            result = set_bit(cardset, bit_pos)
            result_64 = cardset_to_uint64(result)
            expected = value | (1 << bit_pos)
            
            assert result_64 == expected, f"SET_BIT({hex(value)}, {bit_pos}) failed"
            
    def test_get_bit(self):
        """Test bit getting operations."""
        test_cases = [
            (0x0, 0),
            (0x1, 0),
            (0x1, 1),
            (0x80000000, 31),
            (0x100000000, 32),
            (0x8000000000000000, 63),
            (0x12345678, 0),
            (0x12345678, 4),
            (0x12345678, 32),
            (0x12345678, 60)
        ]
        
        for value, bit_pos in test_cases:
            cardset = create_cardset_from_value(value)
            
            result = get_bit(cardset, bit_pos)
            expected = (value >> bit_pos) & 1
            
            assert result == expected, f"GET_BIT({hex(value)}, {bit_pos}) failed"
            
    def test_and_mask(self):
        """Test AND with mask operations."""
        test_cases = [
            (0x12345678, 0xFFFFFFFF),
            (0x12345678, 0xFF000000),
            (0x12345678, 0x0F0F0F0F),
            (0xFFFFFFFF, 0x1),
            (0xFFFFFFFF, 0x80000000)
        ]
        
        for value, mask in test_cases:
            cardset = create_cardset_from_value(value)
            
            result = cardset_and_mask(cardset, mask)
            result_64 = cardset_to_uint64(result)
            expected = value & mask
            
            assert result_64 == expected, f"AND_MASK({hex(value)}, {hex(mask)}) failed"
            
    def test_or_reduce(self):
        """Test OR reduce operations."""
        test_arrays = [
            [0x1, 0x2, 0x4],
            [0x12345678, 0x87654321],
            [0xF, 0xF0, 0xF00, 0xF000],
            [0xFFFF0000, 0x0000FFFF]
        ]
        
        for values in test_arrays:
            cardsets = []
            for v in values:
                cardsets.append(create_cardset_from_value(v))
            
            cardsets_array = jnp.array(cardsets)
            result = or_reduce(cardsets_array)
            
            # Calculate expected result
            expected = 0
            for v in values:
                expected |= v
                
            assert int(result) == expected, f"OR_REDUCE({[hex(v) for v in values]}) failed"
            
    def test_poker_specific_operations(self):
        """Test operations specific to poker cardset usage."""
        # Test typical ACPC bit positions (suit << 4) + rank
        test_cards = [
            (0, 0),   # 2 of clubs, bit 0
            (0, 12),  # A of clubs, bit 12
            (1, 0),   # 2 of diamonds, bit 16
            (3, 12),  # A of spades, bit 60
        ]
        
        # Test setting individual card bits
        empty = create_empty_cardset()
        
        for suit, rank in test_cards:
            bit_pos = (suit << 4) + rank
            
            # Set bit
            cardset = set_bit(empty, bit_pos)
            cardset_64 = cardset_to_uint64(cardset)
            expected = 1 << bit_pos
            
            assert cardset_64 == expected, f"Card ({suit}, {rank}) bit {bit_pos} failed"
            
            # Test getting the bit back
            bit_val = get_bit(cardset, bit_pos)
            assert bit_val == 1, f"Get bit for card ({suit}, {rank}) failed"
            
        # Test combining multiple cards
        all_cards = create_empty_cardset()
        expected_combined = 0
        
        for suit, rank in test_cards:
            bit_pos = (suit << 4) + rank
            all_cards = set_bit(all_cards, bit_pos)
            expected_combined |= (1 << bit_pos)
            
        all_cards_64 = cardset_to_uint64(all_cards)
        assert all_cards_64 == expected_combined, "Combined cards test failed"


if __name__ == "__main__":
    # Run the tests
    test_suite = TestCardsetOperations()
    
    print("Testing uint32[2] cardset operations...")
    
    try:
        test_suite.test_create_empty_cardset()
        print("✓ Empty cardset test passed")
        
        test_suite.test_create_cardset_from_value()
        print("✓ Create from value test passed")
        
        test_suite.test_cardset_or()
        print("✓ Bitwise OR test passed")
        
        test_suite.test_cardset_and()
        print("✓ Bitwise AND test passed")
        
        test_suite.test_cardset_not()
        print("✓ Bitwise NOT test passed")
        
        test_suite.test_left_shift()
        print("✓ Left shift test passed")
        
        test_suite.test_right_shift()
        print("✓ Right shift test passed")
        
        test_suite.test_set_bit()
        print("✓ Set bit test passed")
        
        test_suite.test_get_bit()
        print("✓ Get bit test passed")
        
        test_suite.test_and_mask()
        print("✓ AND mask test passed")
        
        test_suite.test_or_reduce()
        print("✓ OR reduce test passed")
        
        test_suite.test_poker_specific_operations()
        print("✓ Poker-specific operations test passed")
        
        print("\nAll uint32[2] cardset operation tests passed! ✅")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise