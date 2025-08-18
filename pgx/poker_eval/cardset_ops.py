"""
Cardset operations abstraction layer.

This module provides both uint64 and uint32[2] implementations of cardset operations,
allowing us to switch between representations while maintaining identical functionality.

Cardset format (ACPC): bit position = (suit << 4) + rank
- suit: 0=clubs, 1=diamonds, 2=hearts, 3=spades  
- rank: 0=2, 1=3, ..., 11=K, 12=A

All operations are fully vectorized without if statements for JAX compatibility.
"""

import jax
import jax.numpy as jnp
from typing import Union


# ============================================================================
# UINT64 OPERATIONS (Current implementation)
# ============================================================================

@jax.jit
def create_empty_cardset_u64() -> jnp.uint64:
    """Create an empty cardset (uint64)."""
    return jnp.uint64(0)


@jax.jit
def create_cardset_from_value_u64(value: int) -> jnp.uint64:
    """Create cardset from integer value (uint64)."""
    return jnp.uint64(value)


@jax.jit
def cardset_or_u64(a: jnp.uint64, b: jnp.uint64) -> jnp.uint64:
    """Bitwise OR of two cardsets (uint64)."""
    return a | b


@jax.jit
def cardset_and_u64(a: jnp.uint64, b: jnp.uint64) -> jnp.uint64:
    """Bitwise AND of two cardsets (uint64)."""
    return a & b


@jax.jit
def cardset_not_u64(a: jnp.uint64) -> jnp.uint64:
    """Bitwise NOT of cardset (uint64)."""
    return ~a


@jax.jit
def cardset_lshift_u64(cardset: jnp.uint64, shift: int) -> jnp.uint64:
    """Left shift cardset (uint64)."""
    return cardset << shift


@jax.jit
def cardset_rshift_u64(cardset: jnp.uint64, shift: int) -> jnp.uint64:
    """Right shift cardset (uint64)."""
    return cardset >> shift


@jax.jit
def cardset_and_mask_u64(cardset: jnp.uint64, mask: int) -> jnp.uint64:
    """Bitwise AND with integer mask (uint64)."""
    return cardset & mask


@jax.jit
def set_bit_u64(cardset: jnp.uint64, bit_pos: int) -> jnp.uint64:
    """Set bit at position in cardset (uint64)."""
    return cardset | jnp.uint64(1 << bit_pos)


@jax.jit
def get_bit_u64(cardset: jnp.uint64, bit_pos: int) -> jnp.uint64:
    """Get bit at position in cardset (uint64)."""
    return (cardset >> bit_pos) & 1


@jax.jit
def or_reduce_u64(bit_masks: jnp.ndarray) -> jnp.uint64:
    """OR reduce array of uint64 bit masks."""
    return jnp.bitwise_or.reduce(bit_masks)


# ============================================================================
# UINT32[2] OPERATIONS (Vectorized implementation)
# ============================================================================

@jax.jit
def create_empty_cardset_u32() -> jnp.ndarray:
    """Create an empty cardset (uint32[2])."""
    return jnp.array([0, 0], dtype=jnp.uint32)


@jax.jit
def create_cardset_from_value_u32(value: jnp.uint64) -> jnp.ndarray:
    """Create cardset from 64-bit integer value (uint32[2])."""
    # Split 64-bit value into two 32-bit parts
    low = jnp.uint32(value & 0xFFFFFFFF)  # Lower 32 bits
    high = jnp.uint32(value >> 32)        # Upper 32 bits
    return jnp.array([low, high], dtype=jnp.uint32)


@jax.jit
def cardset_or_u32(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Bitwise OR of two cardsets (uint32[2])."""
    return a | b


@jax.jit
def cardset_and_u32(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Bitwise AND of two cardsets (uint32[2])."""
    return a & b


@jax.jit
def cardset_not_u32(a: jnp.ndarray) -> jnp.ndarray:
    """Bitwise NOT of cardset (uint32[2])."""
    return ~a


@jax.jit
def cardset_lshift_u32(cardset: jnp.ndarray, shift: int) -> jnp.ndarray:
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
def cardset_rshift_u32(cardset: jnp.ndarray, shift: int) -> jnp.ndarray:
    """Right shift cardset (uint32[2]) - fully vectorized."""
    shift = jnp.int32(shift)
    
    # Case 1: shift >= 64 -> result is zero
    zero_result = jnp.array([0, 0], dtype=jnp.uint32)
    
    # Case 2: shift >= 32 -> only low bits, shifted from high
    low_only_shift = shift - 32
    low_only_result = jnp.array([
        cardset[1] >> jnp.maximum(low_only_shift, 0),
        jnp.uint32(0)
    ], dtype=jnp.uint32)
    
    # Case 3: 0 < shift < 32 -> normal shift with underflow
    normal_shift = jnp.maximum(shift, 0)
    underflow_shift = jnp.maximum(32 - shift, 0)
    underflow = cardset[1] << underflow_shift
    normal_result = jnp.array([
        (cardset[0] >> normal_shift) | underflow,
        cardset[1] >> normal_shift
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
def cardset_and_mask_u32(cardset: jnp.ndarray, mask: jnp.uint64) -> jnp.ndarray:
    """Bitwise AND with 64-bit integer mask (uint32[2])."""
    # Split 64-bit mask into two 32-bit parts
    mask_low = jnp.uint32(mask & 0xFFFFFFFF)   # Lower 32 bits
    mask_high = jnp.uint32(mask >> 32)         # Upper 32 bits
    mask_array = jnp.array([mask_low, mask_high], dtype=jnp.uint32)
    return cardset & mask_array


@jax.jit
def set_bit_u32(cardset: jnp.ndarray, bit_pos: int) -> jnp.ndarray:
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
def get_bit_u32(cardset: jnp.ndarray, bit_pos: int) -> jnp.uint32:
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
def or_reduce_u32(bit_masks: jnp.ndarray) -> jnp.uint64:
    """OR reduce array of uint32[2] bit masks and return as single uint64."""
    # bit_masks shape: (N, 2) where N is number of masks
    reduced_low = jnp.bitwise_or.reduce(bit_masks[:, 0])   # Low 32 bits
    reduced_high = jnp.bitwise_or.reduce(bit_masks[:, 1])  # High 32 bits
    
    # Combine into single uint64
    return jnp.uint64(reduced_low) | (jnp.uint64(reduced_high) << 32)


@jax.jit
def cardset_to_uint64(cardset: jnp.ndarray) -> jnp.uint64:
    """Convert uint32[2] cardset to uint64 for comparison."""
    return jnp.uint64(cardset[0]) | (jnp.uint64(cardset[1]) << 32)


@jax.jit
def uint64_to_cardset(value: jnp.uint64) -> jnp.ndarray:
    """Convert uint64 to uint32[2] cardset."""
    return create_cardset_from_value_u32(value)


# ============================================================================
# GENERIC INTERFACE (will switch between implementations)
# ============================================================================

# Current implementation uses uint64 - we'll switch this later
USE_UINT32_CARDSETS = False

def create_empty_cardset():
    """Create an empty cardset using current implementation."""
    return create_empty_cardset_u64() if not USE_UINT32_CARDSETS else create_empty_cardset_u32()

def create_cardset_from_value(value):
    """Create cardset from integer value using current implementation."""
    return create_cardset_from_value_u64(value) if not USE_UINT32_CARDSETS else create_cardset_from_value_u32(jnp.uint64(value))

def cardset_or(a, b):
    """Bitwise OR using current implementation."""
    return cardset_or_u64(a, b) if not USE_UINT32_CARDSETS else cardset_or_u32(a, b)

def cardset_and(a, b):
    """Bitwise AND using current implementation."""
    return cardset_and_u64(a, b) if not USE_UINT32_CARDSETS else cardset_and_u32(a, b)

def cardset_not(a):
    """Bitwise NOT using current implementation."""
    return cardset_not_u64(a) if not USE_UINT32_CARDSETS else cardset_not_u32(a)

def set_bit(cardset, bit_pos: int):
    """Set bit at position using current implementation."""
    return set_bit_u64(cardset, bit_pos) if not USE_UINT32_CARDSETS else set_bit_u32(cardset, bit_pos)

def get_bit(cardset, bit_pos: int):
    """Get bit at position using current implementation."""
    return get_bit_u64(cardset, bit_pos) if not USE_UINT32_CARDSETS else get_bit_u32(cardset, bit_pos)

def or_reduce(bit_masks):
    """OR reduce array using current implementation."""
    return or_reduce_u64(bit_masks) if not USE_UINT32_CARDSETS else or_reduce_u32(bit_masks)