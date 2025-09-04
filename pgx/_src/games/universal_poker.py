import math
import jax.numpy as jnp
from pgx.universal_poker import State as UniversalPokerState
from pgx.poker_eval.cardset import cardset_to_cards

SUITS = ["♠", "♥", "♦", "♣"]
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]


def _card_to_string(card_index):
    """Convert card index (0-51) to string representation like 'A♠'."""
    if card_index < 0 or card_index >= 52:
        return "??"
    rank = RANKS[card_index % 13]
    suit = SUITS[card_index // 13]
    return f"{rank}{suit}"


def _make_universal_poker_dwg(dwg, state: UniversalPokerState, config):
    """Create SVG drawing for Universal Poker game state."""
    GRID_SIZE = config["GRID_SIZE"]
    BOARD_WIDTH = config["BOARD_WIDTH"] 
    BOARD_HEIGHT = config["BOARD_HEIGHT"]
    color_set = config["COLOR_SET"]
    
    # Background
    dwg.add(
        dwg.rect(
            (0, 0),
            (BOARD_WIDTH * GRID_SIZE, BOARD_HEIGHT * GRID_SIZE),
            fill=color_set.background_color,
        )
    )
    
    board_g = dwg.g()
    
    # Calculate table center and dimensions
    center_x = BOARD_WIDTH * GRID_SIZE / 2
    center_y = BOARD_HEIGHT * GRID_SIZE / 2
    table_width = BOARD_WIDTH * GRID_SIZE * 0.7
    table_height = BOARD_HEIGHT * GRID_SIZE * 0.5
    
    # Draw oval table
    board_g.add(
        dwg.ellipse(
            center=(center_x, center_y),
            r=(table_width/2, table_height/2),
            fill="none",
            stroke=color_set.grid_color,
            stroke_width="3px"
        )
    )
    
    num_players = int(state.num_players)
    
    # Calculate player positions around oval table
    for player_idx in range(num_players):
        angle = 2 * math.pi * player_idx / num_players - math.pi/2  # Start at top
        
        # Position on oval perimeter
        player_x = center_x + (table_width/2 + 80) * math.cos(angle)
        player_y = center_y + (table_height/2 + 60) * math.sin(angle)
        
        # Player area background
        player_bg_size = 120
        board_g.add(
            dwg.rect(
                (player_x - player_bg_size/2, player_y - player_bg_size/2),
                (player_bg_size, player_bg_size),
                fill=color_set.background_color,
                stroke=color_set.grid_color,
                stroke_width="2px",
                rx="10px",
                ry="10px",
                opacity=0.8
            )
        )
        
        # Player label
        board_g.add(
            dwg.text(
                text=f"P{player_idx + 1}",
                insert=(player_x, player_y - 40),
                fill=color_set.text_color,
                font_size="14px",
                font_family="Arial",
                text_anchor="middle",
                font_weight="bold"
            )
        )
        
        # Hole cards (convert cardset to individual cards)
        hole_cardset = state.hole_cardsets[player_idx]
        hole_cards = cardset_to_cards(hole_cardset)
        
        # Filter out invalid cards (negative indices indicate empty slots)
        valid_hole_cards = [c for c in hole_cards if c >= 0]
        
        # Draw hole cards
        card_width = 25
        card_height = 35
        card_spacing = 28
        start_x = player_x - (len(valid_hole_cards) - 1) * card_spacing / 2
        
        for i, card in enumerate(valid_hole_cards):
            card_x = start_x + i * card_spacing
            card_y = player_y - 15
            
            # Card background
            board_g.add(
                dwg.rect(
                    (card_x - card_width/2, card_y - card_height/2),
                    (card_width, card_height),
                    fill="white",
                    stroke=color_set.grid_color,
                    stroke_width="1px",
                    rx="3px",
                    ry="3px"
                )
            )
            
            # Card text
            card_str = _card_to_string(card)
            # Color red for hearts/diamonds, black for spades/clubs
            card_color = "red" if card_str[-1] in ["♥", "♦"] else "black"
            
            board_g.add(
                dwg.text(
                    text=card_str,
                    insert=(card_x, card_y + 3),
                    fill=card_color,
                    font_size="10px",
                    font_family="Arial",
                    text_anchor="middle"
                )
            )
        
        # Stack size
        stack_size = int(state.stacks[player_idx])
        board_g.add(
            dwg.text(
                text=f"${stack_size}",
                insert=(player_x, player_y + 25),
                fill=color_set.text_color,
                font_size="12px",
                font_family="Arial",
                text_anchor="middle"
            )
        )
        
        # Current bet (positioned between player and center)
        bet_amount = int(state.bets[player_idx])
        if bet_amount > 0:
            # Position bet closer to center
            bet_x = center_x + (table_width/2 + 40) * math.cos(angle) * 0.6
            bet_y = center_y + (table_height/2 + 30) * math.sin(angle) * 0.6
            
            # Bet chip background
            board_g.add(
                dwg.circle(
                    center=(bet_x, bet_y),
                    r=18,
                    fill="gold",
                    stroke=color_set.grid_color,
                    stroke_width="1px"
                )
            )
            
            # Bet amount
            board_g.add(
                dwg.text(
                    text=f"${bet_amount}",
                    insert=(bet_x, bet_y + 3),
                    fill="black",
                    font_size="10px",
                    font_family="Arial",
                    text_anchor="middle",
                    font_weight="bold"
                )
            )
        
        # Folded indicator
        if state.folded[player_idx]:
            board_g.add(
                dwg.text(
                    text="FOLDED",
                    insert=(player_x, player_y + 40),
                    fill="red",
                    font_size="10px",
                    font_family="Arial",
                    text_anchor="middle",
                    font_weight="bold"
                )
            )
    
    # Board cards in center
    visible_board_cardset = state.visible_board_cardsets[state.round]
    board_cards = cardset_to_cards(visible_board_cardset)
    valid_board_cards = [c for c in board_cards if c >= 0]
    
    # Draw board cards
    if valid_board_cards:
        board_card_width = 30
        board_card_height = 42
        board_card_spacing = 35
        board_start_x = center_x - (len(valid_board_cards) - 1) * board_card_spacing / 2
        
        for i, card in enumerate(valid_board_cards):
            card_x = board_start_x + i * board_card_spacing
            card_y = center_y
            
            # Card background
            board_g.add(
                dwg.rect(
                    (card_x - board_card_width/2, card_y - board_card_height/2),
                    (board_card_width, board_card_height),
                    fill="white",
                    stroke=color_set.grid_color,
                    stroke_width="2px",
                    rx="4px",
                    ry="4px"
                )
            )
            
            # Card text
            card_str = _card_to_string(card)
            card_color = "red" if card_str[-1] in ["♥", "♦"] else "black"
            
            board_g.add(
                dwg.text(
                    text=card_str,
                    insert=(card_x, card_y + 4),
                    fill=card_color,
                    font_size="12px",
                    font_family="Arial",
                    text_anchor="middle",
                    font_weight="bold"
                )
            )
    
    # Pot size
    pot_size = int(state.pot)
    board_g.add(
        dwg.text(
            text=f"Pot: ${pot_size}",
            insert=(center_x, center_y + 60),
            fill=color_set.text_color,
            font_size="16px",
            font_family="Arial",
            text_anchor="middle",
            font_weight="bold"
        )
    )
    
    # Round indicator
    round_names = ["Preflop", "Flop", "Turn", "River"]
    if state.round < len(round_names):
        round_name = round_names[state.round]
        board_g.add(
            dwg.text(
                text=round_name,
                insert=(center_x, center_y - 80),
                fill=color_set.text_color,
                font_size="14px",
                font_family="Arial",
                text_anchor="middle",
                font_weight="bold"
            )
        )
    
    # Current player indicator
    if not state.terminated:
        current_player = int(state.current_player)
        angle = 2 * math.pi * current_player / num_players - math.pi/2
        indicator_x = center_x + (table_width/2 + 80) * math.cos(angle)
        indicator_y = center_y + (table_height/2 + 60) * math.sin(angle) - 60
        
        board_g.add(
            dwg.text(
                text="◀ TO ACT",
                insert=(indicator_x, indicator_y),
                fill="red",
                font_size="12px",
                font_family="Arial",
                text_anchor="middle",
                font_weight="bold"
            )
        )
    
    return board_g
