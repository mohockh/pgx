# Universal Poker - PGX Implementation

## Project Overview
This is a JAX-based poker environment implementation for the PGX (Python Game eXperiments) framework. The project implements a universal poker game that supports multiple players, betting rounds, and proper poker hand evaluation.

## Key Files
- `pgx/universal_poker.py` - Main poker game implementation
- `tests/test_universal_poker*.py` - Comprehensive test suite
- `tests/test_universal_poker_speed.py` - Benchmarking test
- `pgx/poker_eval/evaluator.py` - JAX-optimized poker hand evaluator

## Development Commands

### Testing
For testing the Universal Poker game, use the following testing methods.

```bash
# Run all tests
PYTHONPATH=$PYTHONPATH:~/poker/pgx python3 -m pytest -n 4 -vv tests/test_universal_poker*.py --ignore=tests/test_universal_poker_speed.py

# Run speed / benchmark tests
PYTHONPATH=$PYTHONPATH:~/poker/pgx python3 tests/test_universal_poker_speed.py

# Run specific unit test methods
PYTHONPATH=$PYTHONPATH:~/poker/pgx python3 -c "from tests.test_universal_poker import TestUniversalPoker; TestUniversalPoker().test_init_basic()"
```

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

## Testing Strategy
- Comprehensive unit tests for all game mechanics
- Edge case testing (all-in scenarios, small stacks)
- Optimization verification tests (lazy evaluation)
- JAX compilation compatibility tests
- Random game simulation tests

## Known Issues & Limitations
- JAX requires concrete array shapes (no dynamic sizing)
- Game may terminate early in some edge cases (by design)

## Dependencies
- JAX/JAXlib for array operations and JIT compilation
- jax.numpy for numerical computations
- PGX framework for game environment interface

## Performance Metrics
- **Original**: ~16GB memory usage in vectorized mode, 200k hands/sec.
- **Current**: ~15GB memory usage in vectorized mode, 80k hands/sec.
- **Test coverage**: 105 comprehensive test cases, all passing
