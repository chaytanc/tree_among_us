from muselsl import stream, list_muses


def connect_to_muse():
    muses = list_muses()
    try:
        stream(muses[0]['address'], 
                ppg_enabled=True, acc_enabled=True, gyro_enabled=True)
    except IndexError:
        print("No muses found")

if __name__ == "__main__":
    connect_to_muse()
