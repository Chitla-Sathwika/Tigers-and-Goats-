'''
Quick test script to verify Q-learning implementation
'''

import sys
from huligutta import *
from functions import *
from qlearning import QLearningAgent

def test_qlearning_basic():
    """Test basic Q-learning functionality."""
    print("Testing Q-learning implementation...")
    
    # Test agent creation
    print("1. Creating Q-learning agents...")
    tiger_agent = QLearningAgent(player='tiger')
    goat_agent = QLearningAgent(player='goat')
    print("   [OK] Agents created successfully")
    
    # Test state representation
    print("2. Testing state representation...")
    Board().clearBoard()
    Tiger('b0').place()
    Tiger('c1').place()
    Tiger('d1').place()
    Goat('a1').place()
    Goat('a2').place()
    
    board_state = Board().boardPositions
    state_key = tiger_agent.get_state_key_simple(board_state)
    print(f"   [OK] State key: {state_key}")
    
    # Test action generation
    print("3. Testing action generation...")
    tiger_actions = tiger_agent.get_actions(board_state, is_tiger_turn=True)
    goat_actions = goat_agent.get_actions(board_state, is_tiger_turn=False)
    print(f"   [OK] Tiger actions: {len(tiger_actions)} available")
    print(f"   [OK] Goat actions: {len(goat_actions)} available")
    
    # Test action selection
    print("4. Testing action selection...")
    tiger_action = tiger_agent.choose_action(board_state, is_tiger_turn=True, training=False)
    goat_action = goat_agent.choose_action(board_state, is_tiger_turn=False, training=False)
    print(f"   [OK] Tiger selected action: {tiger_action}")
    print(f"   [OK] Goat selected action: {goat_action}")
    
    # Test Q-value update
    print("5. Testing Q-value update...")
    next_state = board_state.copy()
    reward = 10
    tiger_agent.update_q_value(board_state, tiger_action, reward, next_state, done=False)
    print(f"   [OK] Q-value updated successfully")
    print(f"   [OK] Q-table size: {len(tiger_agent.q_table)}")
    
    # Test save/load
    print("6. Testing save/load...")
    test_file = 'test_q_table.pkl'
    tiger_agent.save_q_table(test_file)
    print(f"   [OK] Q-table saved")
    
    new_agent = QLearningAgent(player='tiger')
    new_agent.load_q_table(test_file)
    print(f"   [OK] Q-table loaded")
    print(f"   [OK] Loaded Q-table size: {len(new_agent.q_table)}")
    
    # Cleanup
    import os
    if os.path.exists(test_file):
        os.remove(test_file)
        print(f"   [OK] Test file cleaned up")
    
    print("\n[SUCCESS] All tests passed!")

if __name__ == "__main__":
    try:
        test_qlearning_basic()
    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

