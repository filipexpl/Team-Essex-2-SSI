import brainaccess_board as bb

# Connect to the database
db, status = bb.db_connect()
if status:
    # Retrieve all data from all connected devices as a dictionary {device_id: MNE structure}
    data = db.get_mne()
    print(f"Dictionary of connected devices:\n{data}")
    if data:
      print(f"MNE structure: {data[next(iter(data))]}")

    # Retrieve the last 10 seconds of data
    data_10sec = db.get_mne(duration=10)
    print(data_10sec)
    
import time
import numpy as np
import brainaccess_board as bb

CHUNK_SECONDS = 10


def process_chunk(x, ch_names, fs):
    """x: numpy array (channels, samples). Put your own processing here."""
    rms = np.sqrt(np.mean((x - x.mean(axis=1, keepdims=True)) ** 2, axis=1))
    for name, r in zip(ch_names, rms):
        print(f"   {name:<12} RMS = {r:.2e}")


db, status = bb.db_connect()
if not status:
    raise SystemExit("Can't connect – is BrainAccess Board open and the device streaming?")

# Print the channel names once, so we can see which channels are the EMG ones
for device_id, raw in db.get_mne(duration=1).items():
    print(f"Device {device_id}: {len(raw.ch_names)} channels @ {raw.info['sfreq']} Hz")
    print(raw.ch_names)

chunk_no = 0
next_t = time.time()
print(f"\nGrabbing a {CHUNK_SECONDS}s chunk every {CHUNK_SECONDS}s. Ctrl+C to stop.")
try:
    while True:
        next_t += CHUNK_SECONDS
        time.sleep(max(0, next_t - time.time()))      # wait until the next 10 s have been recorded
        for device_id, raw in db.get_mne(duration=CHUNK_SECONDS).items():
            x = raw.get_data()                         # (25, ~2500) numpy array, in volts
            chunk_no += 1
            print(f"\nChunk {chunk_no}: {x.shape[0]} channels x {x.shape[1]} samples")
            process_chunk(x, raw.ch_names, raw.info["sfreq"])
except KeyboardInterrupt:
    print("Stopped.")