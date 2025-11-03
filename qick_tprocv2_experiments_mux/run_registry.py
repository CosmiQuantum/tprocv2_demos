RUN_REGISTRY = {
    6: {
        # avg_buffer v1.2 => edge counter present (no divide-by-length in hardware)
        "edge_counting": True,

        # If soccfg lacks readouts, use this minimal patch
        "readouts_patch": {
            0: {"iq_offset": 0.0, "fs": 2457.6, "decimated": 307.2},  # dyn 0
            1: {"iq_offset": 0.0, "fs": 2457.6, "decimated": 307.2},  # dyn 1
            2: {"iq_offset": 0.0, "fs": 2457.6, "decimated": 38.4},   # pfb 2..9
            3: {"iq_offset": 0.0, "fs": 2457.6, "decimated": 38.4},
            4: {"iq_offset": 0.0, "fs": 2457.6, "decimated": 38.4},
            5: {"iq_offset": 0.0, "fs": 2457.6, "decimated": 38.4},
            6: {"iq_offset": 0.0, "fs": 2457.6, "decimated": 38.4},
            7: {"iq_offset": 0.0, "fs": 2457.6, "decimated": 38.4},
            8: {"iq_offset": 0.0, "fs": 2457.6, "decimated": 38.4},
            9: {"iq_offset": 0.0, "fs": 2457.6, "decimated": 38.4},
        },

        # Your QUIET mapping for run 6
        "res_ch": 4,                     # MIXMUXGEN_CH
        "ro_ch_by_qubit": [2,3,4,5,6,7], # MUXRO_CH (one per qubit)
        # Optional: defaults if syst_cfg is missing/odd
        "nqz_res": 2,
    }
}