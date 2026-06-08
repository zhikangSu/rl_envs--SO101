class SharedState:
    human_intervention_key = False
    terminate = False
    align_request = False
    leader_manual_takeover = False
    # Set by the 'D' hotkey to VOID the current episode: end it immediately and drop
    # its transitions instead of sending them to the learner (e.g. cup knocked over).
    discard_episode = False
    # Set by the 'S' hotkey to ask the learner to save a FULL checkpoint (model +
    # replay buffers) on demand, so the run can be resumed later.
    save_checkpoint_request = False
    # Set by the 'M' hotkey to print the current end-effector position once, for
    # calibrating reward-shaping target points (cube_xyz / plate_xyz) in the LIVE frame.
    print_pos_request = False

shared_state = SharedState()
