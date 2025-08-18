"""
Cardset operations for uint32[2] representation.

This module provides cardset operations using uint32[2] arrays for memory efficiency.
Each cardset represents a 64-bit value split into low and high 32-bit components.

Cardset format (ACPC): bit position = (suit << 4) + rank
- suit: 0=clubs, 1=diamonds, 2=hearts, 3=spades  
- rank: 0=2, 1=3, ..., 11=K, 12=A

All operations are fully vectorized without if statements for JAX compatibility.
"""

import jax
import jax.numpy as jnp
import jax.lax
from typing import Union


# ============================================================================
# UINT32[2] CARDSET OPERATIONS
# ============================================================================

@jax.jit
def create_empty_cardset() -> jnp.ndarray:
    """Create an empty cardset (uint32[2])."""
    return jnp.array([0, 0], dtype=jnp.uint32)


@jax.jit
def create_cardset_from_parts(low_uint32: int, high_uint32: int) -> jnp.ndarray:
    """Create cardset from separate low and high 32-bit parts (uint32[2])."""
    return jnp.array([jnp.uint32(low_uint32), jnp.uint32(high_uint32)], dtype=jnp.uint32)


def create_cardset_from_value(value: int) -> jnp.ndarray:
    """Create cardset from 64-bit integer value (uint32[2]) - convenience function."""
    low = value & 0xFFFFFFFF
    high = value >> 32
    # Directly create array to avoid JAX 64-bit issues with large values
    return jnp.array([jnp.uint32(low), jnp.uint32(high)], dtype=jnp.uint32)


@jax.jit
def cardset_or(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Bitwise OR of two cardsets (uint32[2])."""
    return a | b


@jax.jit
def cardset_and(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Bitwise AND of two cardsets (uint32[2])."""
    return a & b


@jax.jit
def cardset_not(a: jnp.ndarray) -> jnp.ndarray:
    """Bitwise NOT of cardset (uint32[2])."""
    return ~a


@jax.jit
def cardset_lshift(cardset: jnp.ndarray, shift: int) -> jnp.ndarray:
    """Left shift cardset (uint32[2]) - fully vectorized."""
    # Handle all cases using masks and selects
    shift = jnp.int32(shift)
    
    # Case 1: shift >= 64 -> result is zero
    zero_result = jnp.array([0, 0], dtype=jnp.uint32)
    
    # Case 2: shift >= 32 -> only high bits, shifted from low
    high_only_shift = shift - 32
    high_only_result = jnp.array([
        jnp.uint32(0),
        cardset[0] << jnp.maximum(high_only_shift, 0)
    ], dtype=jnp.uint32)
    
    # Case 3: 0 < shift < 32 -> normal shift with overflow
    normal_shift = jnp.maximum(shift, 0)
    overflow_shift = jnp.maximum(32 - shift, 0)
    overflow = cardset[0] >> overflow_shift
    normal_result = jnp.array([
        cardset[0] << normal_shift,
        (cardset[1] << normal_shift) | overflow
    ], dtype=jnp.uint32)
    
    # Case 4: shift == 0 -> no change
    identity_result = cardset
    
    # Use vectorized selects instead of if statements
    result = jnp.where(
        shift >= 64,
        zero_result,
        jnp.where(
            shift >= 32,
            high_only_result,
            jnp.where(
                shift == 0,
                identity_result,
                normal_result
            )
        )
    )
    
    return result.astype(jnp.uint32)


@jax.jit
def cardset_rshift(cardset: jnp.ndarray, shift: int) -> jnp.ndarray:
    """Right shift cardset (uint32[2]) - fully vectorized."""
    shift = jnp.int32(shift)
    
    # Case 1: shift >= 64 -> result is zero
    zero_result = jnp.array([0, 0], dtype=jnp.uint32)
    
    # Case 2: shift >= 32 -> only low bits, shifted from high
    low_only_shift = shift - 32
    # Use logical right shift
    shift_amount = jnp.maximum(low_only_shift, 0).astype(jnp.uint32)
    high_shifted = jax.lax.shift_right_logical(cardset[1], shift_amount)
    low_only_result = jnp.array([
        jnp.uint32(high_shifted),
        jnp.uint32(0)
    ], dtype=jnp.uint32)
    
    # Case 3: 0 < shift < 32 -> normal shift with underflow
    normal_shift = jnp.maximum(shift, 0)
    underflow_shift = jnp.maximum(32 - shift, 0)
    underflow = cardset[1] << underflow_shift
    # Use logical right shift
    normal_shift_u32 = normal_shift.astype(jnp.uint32)
    low_shifted = jax.lax.shift_right_logical(cardset[0], normal_shift_u32)
    high_shifted = jax.lax.shift_right_logical(cardset[1], normal_shift_u32)
    normal_result = jnp.array([
        jnp.uint32(low_shifted | underflow),
        jnp.uint32(high_shifted)
    ], dtype=jnp.uint32)
    
    # Case 4: shift == 0 -> no change
    identity_result = cardset
    
    # Use vectorized selects instead of if statements
    result = jnp.where(
        shift >= 64,
        zero_result,
        jnp.where(
            shift >= 32,
            low_only_result,
            jnp.where(
                shift == 0,
                identity_result,
                normal_result
            )
        )
    )
    
    return result.astype(jnp.uint32)


@jax.jit
def cardset_and_mask_parts(cardset: jnp.ndarray, mask_low_uint32: int, mask_high_uint32: int) -> jnp.ndarray:
    """Bitwise AND with mask specified as separate low and high 32-bit parts (uint32[2])."""
    mask_array = jnp.array([jnp.uint32(mask_low_uint32), jnp.uint32(mask_high_uint32)], dtype=jnp.uint32)
    return cardset & mask_array


def cardset_and_mask(cardset: jnp.ndarray, mask: int) -> jnp.ndarray:
    """Bitwise AND with 64-bit integer mask (uint32[2]) - convenience function."""
    mask_low = mask & 0xFFFFFFFF
    mask_high = mask >> 32
    return cardset_and_mask_parts(cardset, mask_low, mask_high)


@jax.jit
def set_bit(cardset: jnp.ndarray, bit_pos: int) -> jnp.ndarray:
    """Set bit at position in cardset (uint32[2]) - fully vectorized."""
    bit_pos = jnp.int32(bit_pos)
    
    # Create bit mask for low 32 bits (when bit_pos < 32)
    low_shift = jnp.where(bit_pos < 32, bit_pos, 0)
    low_mask = jnp.where(bit_pos < 32, jnp.uint32(1) << low_shift, jnp.uint32(0))
    
    # Create bit mask for high 32 bits (when bit_pos >= 32)
    high_shift = jnp.where(bit_pos >= 32, bit_pos - 32, 0)
    high_mask = jnp.where(bit_pos >= 32, jnp.uint32(1) << high_shift, jnp.uint32(0))
    
    # Apply masks
    bit_mask = jnp.array([low_mask, high_mask], dtype=jnp.uint32)
    return cardset | bit_mask


@jax.jit
def get_bit(cardset: jnp.ndarray, bit_pos: int) -> jnp.uint32:
    """Get bit at position in cardset (uint32[2]) - fully vectorized."""
    bit_pos = jnp.int32(bit_pos)
    
    # Extract from low 32 bits
    low_shift = jnp.where(bit_pos < 32, bit_pos, 0)
    low_bit = jnp.where(bit_pos < 32, (cardset[0] >> low_shift) & 1, 0)
    
    # Extract from high 32 bits
    high_shift = jnp.where(bit_pos >= 32, bit_pos - 32, 0)
    high_bit = jnp.where(bit_pos >= 32, (cardset[1] >> high_shift) & 1, 0)
    
    # Return the appropriate bit
    return jnp.uint32(low_bit | high_bit)


@jax.jit
def or_reduce(bit_masks: jnp.ndarray) -> jnp.uint32:
    """OR reduce array of uint32[2] bit masks and return combined result."""
    # bit_masks shape: (N, 2) where N is number of masks
    reduced_low = jnp.bitwise_or.reduce(bit_masks[:, 0])   # Low 32 bits
    reduced_high = jnp.bitwise_or.reduce(bit_masks[:, 1])  # High 32 bits
    
    # Combine by adding the reduced parts
    return reduced_low | reduced_high


def cardset_to_uint64(cardset: jnp.ndarray) -> int:
    """Convert uint32[2] cardset to Python int for comparison."""
    return int(cardset[0]) | (int(cardset[1]) << 32)


