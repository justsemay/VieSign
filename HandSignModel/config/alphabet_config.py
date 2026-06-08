DATA_PATH   = "./data/alphabet"
MODEL_PATH  = "./models/alphabet/lstm_alphabet_v1.h5"

NUM_SEQUENCES   = 30
SEQUENCE_LENGTH = 30

ACTIONS = [
    "A", "A_HOOK", "A_HAT",   # A Ă Â
    "B", "C", "D", "D_STROKE", # B C D Đ
    "E", "E_HAT",              # E Ê
    "G", "H", "I",
    "K", "L", "M", "N",
    "O", "O_HORN", "O_HAT",    # O Ơ Ô
    "P", "Q", "R", "S", "T",
    "U", "U_HORN",             # U Ư
    "V", "X", "Y"
]

THRESHOLD = 0.85