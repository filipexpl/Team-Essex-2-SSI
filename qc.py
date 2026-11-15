import time
import brainaccess_board as bb

db, _ = bb.db_connect()
for stan in ["ROZLUZNIJ twarz", "ZACISNIJ szczeke"]:
    input(f"{stan} i nacisnij Enter...")
    time.sleep(3)
    raw = list(db.get_mne(duration=3).values())[0]
    raw.pick(["F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]).filter(20, 450, verbose=False)
    rms = raw.get_data().std(axis=1) * 1e6
    print(stan, " ".join(f"{n}={v:.0f}" for n, v in zip(raw.ch_names, rms)), "µV")
