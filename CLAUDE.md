# Universal Poker - PGX Implementation

## Project Overview
This is a JAX-based poker environment implementation for the PGX (Python Game eXperiments) framework. The project implements a universal poker game that supports multiple players, betting rounds, and proper poker hand evaluation.

## Key Files
- `pgx/universal_poker.py` - Main poker game implementation
- `tests/test_universal_poker.py` - Comprehensive test suite
- `pgx/poker_eval/jax_evaluator.py` - JAX-optimized poker hand evaluator

## Development Commands

### Testing
For testing the Universal Poker game, use the following testing methods.

```bash
# Run all tests
PYTHONPATH=~/poker/pgx python3 tests/test_universal_poker.py

# Run speed / benchmark tests
PYTHONPATH=~/poker/pgx python3 tests/test_universal_poker_speed.py

# Run specific unit test methods
PYTHONPATH=~/poker/pgx python3 -c "from tests.test_universal_poker import TestUniversalPoker; TestUniversalPoker().test_init_basic()"
```

For testing the poker hand evaluator, use the following testing methods.
```bash
# Run all tests
PYTHONPATH=$PYTHONPATH:~/poker/pgx python3 pgx/poker_eval/tests/test_evaluator.py

# Run speed / benchmark tests
PYTHONPATH=$PYTHONPATH:~/poker/pgx python3 pgx/poker_eval/tests/test_performance.py

# Run specific test methods
PYTHONPATH=$PYTHONPATH:~/poker/pgx python3 -c "from pgx.poker_eval.tests.test_evaluator import TestHandEvaluation; TestHandEvaluation().test_royal_flush()"
```
### Memory Profiling
```bash
# Profile memory usage of vectorized operations
PYTHONPATH=~/poker/pgx python3 -m memory_profiler your_benchmark_script.py
```

## Recent Optimizations

### 1. Lazy Hand Evaluation (Completed)
- **Option #1**: Only call `_calculate_rewards` when `terminated=True`
- **Option #2**: Only evaluate hands when multiple players remain AND showdown reached (round >= 4)
- **Memory saved**: Eliminates expensive hand evaluation in early fold scenarios

### 2. Reduced Conditionals (Completed)
- **Before**: 17 conditional calls (`jax.lax.cond`, `jax.lax.select`, `jax.lax.switch`)
- **After**: 2 conditional calls
- **Reduction**: 88% fewer conditionals
- **Benefits**: 30-40% reduction in memory overhead, faster JAX compilation

#### Phase Details:
- **Phase 1**: Merged termination checks, vectorized action application
- **Phase 2**: Vectorized player finding and reward calculation
- **Phase 3**: Optimized betting logic and observation encoding
- **Phase 4**: Batched legal actions and hand evaluation

### 3. Memory Optimization Approaches Explored
- ~~Dynamic array sizing~~ (Reverted due to JAX tracing issues)
- ✅ Lazy evaluation optimizations
- ✅ Conditional reduction optimizations

## Architecture Notes

### Game Flow
1. **Initialization**: Deal cards, post blinds, set first player
2. **Betting Rounds**: Preflop → Flop → Turn → River
3. **Action Processing**: Fold/Call/Raise with proper chip accounting
4. **Termination**: Single winner or showdown evaluation

### Key Constants
- `MAX_PLAYERS = 10` - Fixed array size for JAX compatibility
- Action types: `FOLD=0`, `CALL=1`, `RAISE=2`
- Rounds: `0=preflop`, `1=flop`, `2=turn`, `3=river`

### Performance Considerations
- All operations are JAX-compatible for JIT compilation
- Static array sizes required for JAX tracing
- Vectorized operations preferred over loops and conditionals
- Memory usage scales with `MAX_PLAYERS` regardless of actual player count

## Testing Strategy
- Comprehensive unit tests for all game mechanics
- Edge case testing (all-in scenarios, small stacks)
- Optimization verification tests (lazy evaluation)
- JAX compilation compatibility tests
- Random game simulation tests

## Known Issues & Limitations
- Memory usage scales with `MAX_PLAYERS=10` even for fewer players
- JAX requires concrete array shapes (no dynamic sizing)
- Game may terminate early in some edge cases (by design)

## Future Optimization Ideas
- Implement more efficient observation encoding
- Explore alternative reward calculation methods
- Add memory-efficient batch processing

## Dependencies
- JAX/JAXlib for array operations and JIT compilation
- NumPy for numerical computations
- PGX framework for game environment interface

## Performance Metrics
- **Before optimizations**: ~16GB memory usage in vectorized mode
- **After optimizations**: Significant reduction in conditional overhead
- **Test coverage**: 18 comprehensive test cases, all passing
