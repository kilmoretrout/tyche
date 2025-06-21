import streamlit as st
import numpy as np
import random
import itertools
import copy
import pandas as pd
import altair as alt
from copy import deepcopy
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import squareform

from games import Simulator  # Your game logic
st.set_page_config(layout="wide")

import streamlit.components.v1 as components

def plot_strategy(actions, p):
    st.markdown("### Strategy")
    strategy = dict(zip(actions, p))
    
    df = pd.DataFrame({'Action': strategy.keys(), 'Probability': strategy.values()})
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Action', sort=None),
        y='Probability',
        tooltip=['Action', 'Probability']
    ).properties(title="Action Probabilities", width=300)
    st.altair_chart(chart, use_container_width=True)
    
# --- Card Rendering Helpers ---
suits = "shdc"
ranks = "23456789TJQKA"
cards = [r + s for r, s in itertools.product(ranks, suits)]

pairs = list(itertools.combinations(range(52), 2))

holes = []
  
for hole in pairs:
    hole_vec = np.zeros(52)
    hole_vec[hole[0]] = 1.
    hole_vec[hole[1]] = 1.        
    
    holes.append(hole_vec)

# 1326, 52
holes = np.array(holes)      

def render_card(card):
    rank, suit = card[0], card[1]
    symbols = {'h': '♥', 'd': '♦', 's': '♠', 'c': '♣'}
    color = 'red' if suit in 'hd' else 'black'
    return f'<span style="font-size:1.5em; color:{color}; padding:2px;">{rank}{symbols[suit]}</span>'

def render_hand(cards):
    return f"<div style='display:flex; gap:6px;'>{''.join([render_card(c) for c in cards])}</div>"

# --- Deck Class ---
class Deck:
    def __init__(self): self.reset()
    def reset(self): self.cards = copy.copy(cards); random.shuffle(self.cards)
    def deal(self, n): ret = self.cards[-n:]; self.cards = self.cards[:-n]; return ret

if 'model' not in st.session_state:
    st.session_state.model = Simulator()
    st.session_state.game = st.session_state.model.game

if 'deck' not in st.session_state:
    st.session_state.deck = Deck()
    st.session_state.hands = [st.session_state.deck.deal(2) for _ in range(st.session_state.game.n_players)]
    st.session_state.board = []

if 'bb_dollars' not in st.session_state:
    st.session_state.bb_dollars = 2.0
    
if 'pr' not in st.session_state:
    st.session_state.pr = None
    st.session_state.actions = []
    
if 'call_index' not in st.session_state:
    st.session_state.call_index = 1

if 'reach' not in st.session_state:
    st.session_state.reach = np.zeros((6, 1326))
    
if 'history' not in st.session_state:
    st.session_state.history = ""

possible_actions_ = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 8.0, -1.0]

g = st.session_state.game
bb_value = st.session_state.bb_dollars

# --- Sidebar ---
st.sidebar.title("Settings")
bb_input = st.sidebar.number_input("Big Blind ($)", min_value=0.01, value=bb_value, step=0.25, format="%.2f")
st.session_state.bb_dollars = bb_input

def new_game():
    st.session_state.model.new_game()
    st.session_state.game = st.session_state.model.game
    st.session_state.deck = Deck()
    st.session_state.hands = [st.session_state.deck.deal(2) for _ in range(st.session_state.game.n_players)]
    st.session_state.board = []
    st.session_state.history = ""
    st.rerun()

if st.sidebar.button("🔄 New Game"):
    new_game()

# --- Header Info ---
st.title("♠️ No Limit Hold'em Poker")

# --- Main Layout ---
table_col, strat_col = st.columns([3, 2])

with table_col:
    st.markdown("### 🃏 Poker Table")

    # --- Player Positioning Logic (REVISED) ---
    n = g.n_players
    player_divs = ""
    
    # Define table and container dimensions
    table_width = 600
    table_height = 300
    container_width = table_width + 150 # Add padding for players
    container_height = table_height + 180 # Add padding for players

    # Center of the oval table within the container
    center_x = container_width / 2
    center_y = container_height / 2
    
    # Radius for positioning the player divs (larger than table radius)
    radius_x = (table_width / 2) + 30
    radius_y = (table_height / 2) + 40

    for i in range(n):
        # Calculate the angle for each player, starting the first player at the top
        angle = (2 * np.pi * i / n) - (np.pi / 2)

        # Calculate player position using ellipse parametric equations
        x = center_x + radius_x * np.cos(angle)
        y = center_y + radius_y * np.sin(angle)

        p = g.players[i]
        hand_display = (
            '<span style="opacity:0.4;">🃏🃏 (folded)</span>' if g.folded[i]
            else render_hand(st.session_state.hands[i]) if i == g.current
            else "🂠🂠"
        )

        player_divs += f"""
        <div style="
            position: absolute;
            left: {x}px;
            top: {y}px;
            transform: translate(-50%, -50%);
            text-align: center;
            width: 120px;
            background-color: white;
            border: 1px solid #ccc;
            border-radius: 10px;
            padding: 6px;
            font-size: 12px;
            box-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            color: black;
            z-index: 10;
        ">
            <b>Player {i}</b><br>
            Stack: ${p.stack * bb_value:.2f}<br>
            Wager: ${g.wagers[i] * bb_value:.2f}<br>
            {hand_display}<br>
            {'<b style="color: #d32f2f;">⬅️ Current</b>' if i == g.current else ''}
        </div>
        """

    # --- Central Pot and Community Cards Display ---
    board_html = render_hand(st.session_state.board) if st.session_state.board else "No cards dealt yet."
    center_content = f"""
    <div style="
        position: absolute;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        text-align: center;
        color: white;
        background-color: rgba(0, 0, 0, 0.2);
        padding: 10px 20px;
        border-radius: 15px;
    ">
        <h3 style="margin: 0; font-weight: bold;">Pot: ${g.pot * bb_value:.2f}</h3>
        <div style="margin-top: 10px;">
            {board_html}
        </div>
    </div>
    """

    # --- Final HTML for the Table and Players ---
    html_code = f"""
    <div style="
        position: relative;
        width: {container_width}px;
        height: {container_height}px;
        margin: auto;
        display: flex; /* Use flexbox for easy centering of the table itself */
        justify-content: center;
        align-items: center;
    ">
        <div style="
            width: {table_width}px;
            height: {table_height}px;
            background-color: #2e7d32;
            border: 8px solid #1b5e20;
            border-radius: {table_height / 2}px; /* This creates the oval shape */
            box-shadow: inset 0 0 20px rgba(0,0,0,0.5);
        ">
        </div>
        
        {center_content}
        {player_divs}
    </div>
    """

    components.html(html_code, height=container_height + 20)


with strat_col:
    # possible_actions_ = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 8.0, -1.0]
    cp = g.current

    call_amt = g.wager - g.wagers[cp]
    stack = g.players[cp].stack
    can_call = call_amt > 0

    
    if not g.terminal():
        if g.raise_amt > 0:
            if can_call:
                indices = [0, 1] + list(range(5, 13))
                possible_actions = ['f', 'c'] + ['r({0:.3f})'.format(u) for u in possible_actions_[3:]]
                st.session_state.call_index = 1
            else:
                indices = [1] + list(range(5, 13))
                possible_actions = ['c'] + ['r({0:.3f})'.format(u) for u in possible_actions_[3:]]
                st.session_state.call_index = 0
                
            
            board = st.session_state.hands[g.current] + st.session_state.board
            street = st.session_state.game.round
            
            p, idx, actions = st.session_state.model.strat.get(st.session_state.game.history, st.session_state.game.round, [cards.index(u) for u in board])
            hi = st.session_state.model.strat.hand_indexers[street]
    
            pr = []
            for hole in pairs:
                board = [cards.index(u) for u in st.session_state.board]
                
                if (hole[0] in board) or (hole[1] in board):
                    pr.append(np.ones(p.shape[1]))
                else:
                    board = list(hole) + [cards.index(u) for u in st.session_state.board]
                    
                    pr.append(p[st.session_state.model.strat.buckets[street][hi.index(board)]])
                                    
                
            
        else:
            possible_actions = ['c'] + ['b({0:.3f})'.format(u) for u in possible_actions_]
            st.session_state.call_index = 1
    
            board = st.session_state.hands[g.current] + st.session_state.board
            street = st.session_state.game.round
            
            p, idx, actions = st.session_state.model.strat.get(st.session_state.game.history, st.session_state.game.round, [cards.index(u) for u in board])
            hi = st.session_state.model.strat.hand_indexers[street]
    
            pr = []
            for hole in pairs:
                board = [cards.index(u) for u in st.session_state.board]
                
                if (hole[0] in board) or (hole[1] in board):
                    pr.append(np.ones(p.shape[1]))
                else:
                    board = list(hole) + [cards.index(u) for u in st.session_state.board]
                    
                    pr.append(p[st.session_state.model.strat.buckets[street][hi.index(board)]])
                                    
                
        p = p / p.sum(-1).reshape(-1, 1)
        
        st.session_state.pr = np.array(pr)
        st.session_state.actions = actions
        
        plot_strategy(actions, p[idx])
    
# --- Gameplay Row: Info | Actions | Strategy ---
# st.markdown("## 🎮")
info_col, action_col = st.columns([2, 2])

# --- Player Info Column ---
with info_col:
    cp = g.current
    st.markdown(render_hand(st.session_state.hands[cp]), unsafe_allow_html=True)
    
    # --- Manual Hole Card Selection for Current Player ---
    st.markdown("### 🔄 Change Hole Cards")

    available_cards = [c for c in cards if c not in st.session_state.board and all(c not in h for h in st.session_state.hands)]

    # Current hand (default selection)
    current_hand = st.session_state.hands[cp]
    default1 = available_cards.index(current_hand[0]) if current_hand[0] in available_cards else 0
    default2 = available_cards.index(current_hand[1]) if current_hand[1] in available_cards else 1

    card1 = st.selectbox("Card 1", options=available_cards, index=default1, key="card1_select")
    card2 = st.selectbox("Card 2", options=[c for c in available_cards if c != card1], index=default2 if default2 != default1 else 0, key="card2_select")

    if st.button("♠️ Set Hole Cards"):
        st.session_state.hands[cp] = [card1, card2]
        st.rerun()
    
# --- Action Buttons ---
with action_col:
    st.markdown("### Actions")

    call_amt = g.wager - g.wagers[cp]
    stack = g.players[cp].stack
    can_call = call_amt > 0
    
    action_type = "Raise" if g.raise_amt > 0 else "Bet"
    min_raise_amt = g.raise_amt

    # Action Buttons in Form
    with st.form("action_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        if can_call:
            fold = col1.form_submit_button("❌ Fold")
        else:
            fold = False
        call = col2.form_submit_button("✅ Call" if can_call else "✅ Check")

        amt_dollars = st.number_input(f"{action_type} Amount ($)", min_value=0.0, max_value=stack * bb_value, step=bb_value, format="%.2f")
        raise_btn = st.form_submit_button(f"💸 {action_type}")

    if fold:
        pr_str = "Player {} folds...\n".format(cp)
    
        st.session_state.history += pr_str
    elif call:
        pr_str = "Player {} checks / calls...\n".format(cp)
        
        st.session_state.history += pr_str
    elif raise_btn: 
        if g.raise_amt == 0:
            pr_str = "Player {} bets {}...\n".format(cp, amt_dollars)
        else:
            pr_str = "Player {} raises by {}...\n".format(cp, amt_dollars)

        st.session_state.history += pr_str

    # --- Handle Actions ---
    def deal_board_for_new_round(prev, new):
        if new > prev:
            board = st.session_state.board
            if new == 1: board += st.session_state.deck.deal(3)
            elif new in [2, 3]: board += st.session_state.deck.deal(1)

    if fold:
        prev = g.round
        st.session_state.reach[g.current] += np.log(st.session_state.pr[:,0] + 1e-8)
        
        st.session_state.game = g.parse_action("f")
        deal_board_for_new_round(prev, st.session_state.game.round)
        st.rerun()
    
    if call:
        prev = g.round
        st.session_state.reach[g.current] += np.log(st.session_state.pr[:,st.session_state.call_index] + 1e-8)
        
        st.session_state.game = g.parse_action("c")
        deal_board_for_new_round(prev, st.session_state.game.round)
        st.rerun()

    if raise_btn and amt_dollars > 0:
        # convert to bbs
        amt_chips = amt_dollars / bb_value
        
        x = [u for u in st.session_state.actions if (('r' in u) or ('b' in u))]
        x = [float(u.replace('(', '').replace(')', '').replace('r', '').replace('b', '')) for u in x]
        
        x = np.array(x)
        
        if g.wager == 0:
            # Bet 
            pot_fraction = amt_chips / g.pot
            
            x[-1] = stack / g.pot
            f = interp1d(x, np.log(st.session_state.pr[:,-len(x):] + 1e-8))
            
            if amt_chips > stack:
                amt_chips = stack
            
            st.session_state.reach[g.current] += f(amt_chips / g.pot)
            
            action = "b(-1.0)" if amt_chips >= stack else f"b({pot_fraction:.4f})"
        else:
            # Raise
            if amt_chips < min_raise_amt and amt_chips < stack:
                st.error(f"Raise must be at least to ${min_raise_amt * bb_value:.2f}")
            else:
                raise_amt = amt_chips
                frac = raise_amt / g.raise_amt
                
                x[-1] = stack / g.raise_amt
                f = interp1d(x, np.log(st.session_state.pr[:,-len(x):] + 1e-8))
                
                st.session_state.reach[g.current] += f(frac)
                
                action = "r(-1.0)" if amt_chips >= stack else f"r({frac:.4f})"
                
        prev = g.round
        st.session_state.game = g.parse_action(action)
        deal_board_for_new_round(prev, st.session_state.game.round)
        st.rerun()
        
    if st.session_state.game.terminal():
        new_game()
        
    # --- History Log ---
    st.markdown("### History")    
    st.text(st.session_state.history)
        
# The expander acts as a container
with st.expander("Reach probabilities", expanded=True):
    st.markdown("### Reach Distribution Viewer")

    # Dropdown to select a player
    selected_player = st.selectbox("Select a player to view reach distribution:", list(range(g.n_players)), format_func=lambda x: f"Player {x}")

    # Get that player's reach vector
    reach_vec = st.session_state.reach[selected_player]
        
    reach_matrix = squareform(reach_vec)
    
    # Streamlit layout
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("#### Reach Heatmap (52x52)")
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(reach_matrix, cmap="viridis", xticklabels=cards, yticklabels=cards, ax=ax)
        ax.set_xlabel("Card 1")
        ax.set_ylabel("Card 2")
        st.pyplot(fig)
        
    
    with col2:
        # Normalize
        reach_probs = np.exp(reach_vec - np.max(reach_vec))  # for numerical stability
        reach_probs /= reach_probs.sum()
        
        # Map back to hole cards
        reach_df = pd.DataFrame({
            'Hole': [''.join([cards[i] for i, val in enumerate(hole) if val > 0]) for hole in holes],
            'LogProb': reach_vec,
            'Prob': reach_probs
        })
        
        # Sort by highest probability
        reach_df = reach_df.sort_values(by='Prob', ascending=False)
        
        # Show top 10
        st.markdown(f"#### Reach Distribution for Player {selected_player}")
        st.dataframe(reach_df[['Hole', 'Prob']].style.format({"Prob": "{:.4%}"}))
