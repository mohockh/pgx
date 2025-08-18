"""
Test suite to verify that uint64 and uint32[2] cardset operations produce identical results.

This ensures that switching from uint64 to uint32[2] representation maintains
complete functional equivalence.
"""

import jax
import jax.numpy as jnp
import pytest
from ..cardset_ops import *


class TestCardsetOperationsEquivalence:
    """Test equivalence between uint64 and uint32[2] cardset operations."""
    
    def test_create_empty_cardset(self):
        """Test empty cardset creation."""
        empty_u64 = create_empty_cardset_u64()
        empty_u32 = create_empty_cardset_u32()
        
        # Convert u32 to u64 for comparison
        empty_u32_as_u64 = cardset_to_uint64(empty_u32)
        
        assert empty_u64 == empty_u32_as_u64 == 0
        
    def test_create_cardset_from_value(self):
        """Test cardset creation from various values."""
        test_values = [
            0,
            1,
            0xFF,
            0xFFFF,
            0xFFFFFFFF,
            0x123456789ABCDEF0,
            0x7FFFFFFFFFFFFFFF  # Use max signed int64 to avoid overflow
        ]
        
        for value in test_values:
            value_u64 = jnp.uint64(value)
            cardset_u64 = create_cardset_from_value_u64(value)
            cardset_u32 = create_cardset_from_value_u32(value_u64)
            
            # Convert u32 to u64 for comparison
            cardset_u32_as_u64 = cardset_to_uint64(cardset_u32)
            
            assert cardset_u64 == cardset_u32_as_u64, f"Value {hex(value)} failed"
            
    def test_cardset_or(self):
        """Test bitwise OR operations."""
        test_pairs = [
            (0x0, 0x0),
            (0x1, 0x0),
            (0xFF, 0xFF00),
            (0x123456789ABCDEF0, 0x0FEDCBA987654321),
            (0x7FFFFFFF00000000, 0x000000007FFFFFFF),
            (0x5555555555555555, 0x2AAAAAAAAAAAAAAA)
        ]
        
        for a_val, b_val in test_pairs:
            a_u64, b_u64 = jnp.uint64(a_val), jnp.uint64(b_val)
            a_u32, b_u32 = create_cardset_from_value_u32(a_u64), create_cardset_from_value_u32(b_u64)
            
            result_u64 = cardset_or_u64(a_u64, b_u64)
            result_u32 = cardset_or_u32(a_u32, b_u32)
            result_u32_as_u64 = cardset_to_uint64(result_u32)
            
            assert result_u64 == result_u32_as_u64, f"OR({hex(a_val)}, {hex(b_val)}) failed"
            
    def test_cardset_and(self):
        """Test bitwise AND operations."""
        test_pairs = [
            (0x0, 0x0),
            (0x7FFFFFFFFFFFFFFF, 0x0),
            (0xFF, 0xFF00),
            (0x123456789ABCDEF0, 0x0FEDCBA987654321),
            (0x7FFFFFFF00000000, 0x000000007FFFFFFF),
            (0x5555555555555555, 0x2AAAAAAAAAAAAAAA)
        ]
        
        for a_val, b_val in test_pairs:
            a_u64, b_u64 = jnp.uint64(a_val), jnp.uint64(b_val)
            a_u32, b_u32 = create_cardset_from_value_u32(a_u64), create_cardset_from_value_u32(b_u64)
            
            result_u64 = cardset_and_u64(a_u64, b_u64)
            result_u32 = cardset_and_u32(a_u32, b_u32)
            result_u32_as_u64 = cardset_to_uint64(result_u32)
            
            assert result_u64 == result_u32_as_u64, f"AND({hex(a_val)}, {hex(b_val)}) failed"
            
    def test_cardset_not(self):
        """Test bitwise NOT operations."""
        test_values = [
            0x0,
            0x1,
            0xFF,
            0xFFFF,
            0xFFFFFFFF,
            0x123456789ABCDEF0,
            0x5555555555555555
        ]
        
        for value in test_values:
            value_u64 = jnp.uint64(value)
            cardset_u32 = create_cardset_from_value_u32(value_u64)
            
            result_u64 = cardset_not_u64(value_u64)
            result_u32 = cardset_not_u32(cardset_u32)
            result_u32_as_u64 = cardset_to_uint64(result_u32)
            
            assert result_u64 == result_u32_as_u64, f"NOT({hex(value)}) failed"
            
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
            (0x1, 65),
            (0x123456789ABCDEF0, 0),
            (0x123456789ABCDEF0, 4),
            (0x123456789ABCDEF0, 32),
            (0x7FFFFFFFFFFFFFFF, 1)
        ]
        
        for value, shift in test_cases:
            value_u64 = jnp.uint64(value)
            cardset_u32 = create_cardset_from_value_u32(value_u64)
            
            result_u64 = cardset_lshift_u64(value_u64, shift)
            result_u32 = cardset_lshift_u32(cardset_u32, shift)
            result_u32_as_u64 = cardset_to_uint64(result_u32)
            
            assert result_u64 == result_u32_as_u64, f"LSHIFT({hex(value)}, {shift}) failed: u64={hex(int(result_u64))}, u32={hex(int(result_u32_as_u64))}"
            
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
            (0x8000000000000000, 65),
            (0x123456789ABCDEF0, 0),
            (0x123456789ABCDEF0, 4),
            (0x123456789ABCDEF0, 32),
            (0x7FFFFFFFFFFFFFFF, 1)
        ]
        
        for value, shift in test_cases:
            value_u64 = jnp.uint64(value)
            cardset_u32 = create_cardset_from_value_u32(value_u64)
            
            result_u64 = cardset_rshift_u64(value_u64, shift)
            result_u32 = cardset_rshift_u32(cardset_u32, shift)
            result_u32_as_u64 = cardset_to_uint64(result_u32)
            
            assert result_u64 == result_u32_as_u64, f"RSHIFT({hex(value)}, {shift}) failed: u64={hex(int(result_u64))}, u32={hex(int(result_u32_as_u64))}"
            
    def test_set_bit(self):
        """Test bit setting operations."""
        test_cases = [
            (0x0, 0),
            (0x0, 1),
            (0x0, 31),
            (0x0, 32),
            (0x0, 63),
            (0x123456789ABCDEF0, 0),
            (0x123456789ABCDEF0, 15),
            (0x123456789ABCDEF0, 32),
            (0x123456789ABCDEF0, 47)
        ]
        
        for value, bit_pos in test_cases:
            value_u64 = jnp.uint64(value)
            cardset_u32 = create_cardset_from_value_u32(value_u64)
            
            result_u64 = set_bit_u64(value_u64, bit_pos)
            result_u32 = set_bit_u32(cardset_u32, bit_pos)
            result_u32_as_u64 = cardset_to_uint64(result_u32)
            
            assert result_u64 == result_u32_as_u64, f"SET_BIT({hex(value)}, {bit_pos}) failed"
            
    def test_get_bit(self):
        """Test bit getting operations."""
        test_cases = [
            (0x0, 0),
            (0x1, 0),
            (0x1, 1),
            (0x80000000, 31),
            (0x100000000, 32),
            (0x8000000000000000, 63),
            (0x123456789ABCDEF0, 0),
            (0x123456789ABCDEF0, 4),
            (0x123456789ABCDEF0, 32),
            (0x123456789ABCDEF0, 60)
        ]
        
        for value, bit_pos in test_cases:
            value_u64 = jnp.uint64(value)
            cardset_u32 = create_cardset_from_value_u32(value_u64)
            
            result_u64 = get_bit_u64(value_u64, bit_pos)
            result_u32 = get_bit_u32(cardset_u32, bit_pos)
            
            assert result_u64 == result_u32, f"GET_BIT({hex(value)}, {bit_pos}) failed"
            
    def test_and_mask(self):
        """Test AND with mask operations."""
        test_cases = [
            (0x123456789ABCDEF0, 0xFFFFFFFF),
            (0x123456789ABCDEF0, 0x7FFFFFFF00000000),
            (0x123456789ABCDEF0, 0x0F0F0F0F0F0F0F0F),
            (0x7FFFFFFFFFFFFFFF, 0x1),
            (0x7FFFFFFFFFFFFFFF, 0x4000000000000000)
        ]
        
        for value, mask in test_cases:
            value_u64 = jnp.uint64(value)
            mask_u64 = jnp.uint64(mask)
            cardset_u32 = create_cardset_from_value_u32(value_u64)
            
            result_u64 = cardset_and_mask_u64(value_u64, mask)
            result_u32 = cardset_and_mask_u32(cardset_u32, mask_u64)
            result_u32_as_u64 = cardset_to_uint64(result_u32)
            
            assert result_u64 == result_u32_as_u64, f"AND_MASK({hex(value)}, {hex(mask)}) failed"
            
    def test_or_reduce(self):
        """Test OR reduce operations."""
        test_arrays = [
            [0x1, 0x2, 0x4],
            [0x123456789ABCDEF0, 0x0FEDCBA987654321],
            [0xF, 0xF0, 0xF00, 0xF000],
            [0x7FFFFFFF00000000, 0x000000007FFFFFFF]
        ]
        
        for values in test_arrays:
            values_u64 = jnp.array([jnp.uint64(v) for v in values])
            values_u32 = jnp.array([create_cardset_from_value_u32(jnp.uint64(v)) for v in values])
            
            result_u64 = or_reduce_u64(values_u64)
            result_u32 = or_reduce_u32(values_u32)
            
            assert result_u64 == result_u32, f"OR_REDUCE({[hex(v) for v in values]}) failed"
            
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
        empty_u64 = create_empty_cardset_u64()
        empty_u32 = create_empty_cardset_u32()
        
        for suit, rank in test_cards:
            bit_pos = (suit << 4) + rank
            
            # Set bit in both representations
            cardset_u64 = set_bit_u64(empty_u64, bit_pos)
            cardset_u32 = set_bit_u32(empty_u32, bit_pos)
            cardset_u32_as_u64 = cardset_to_uint64(cardset_u32)
            
            assert cardset_u64 == cardset_u32_as_u64, f"Card ({suit}, {rank}) bit {bit_pos} failed"
            
            # Test getting the bit back
            bit_u64 = get_bit_u64(cardset_u64, bit_pos)
            bit_u32 = get_bit_u32(cardset_u32, bit_pos)
            
            assert bit_u64 == bit_u32 == 1, f"Get bit for card ({suit}, {rank}) failed"
            
        # Test combining multiple cards
        all_cards_u64 = create_empty_cardset_u64()
        all_cards_u32 = create_empty_cardset_u32()
        
        for suit, rank in test_cards:
            bit_pos = (suit << 4) + rank
            all_cards_u64 = set_bit_u64(all_cards_u64, bit_pos)
            all_cards_u32 = set_bit_u32(all_cards_u32, bit_pos)
            
        all_cards_u32_as_u64 = cardset_to_uint64(all_cards_u32)
        assert all_cards_u64 == all_cards_u32_as_u64, "Combined cards test failed"


if __name__ == "__main__":
    # Run the tests
    test_suite = TestCardsetOperationsEquivalence()
    
    print("Testing cardset operations equivalence...")
    
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
        
        print("\nAll cardset operation equivalence tests passed! ✅")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise