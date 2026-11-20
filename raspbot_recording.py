"""Shared recording labels and keyboard/receiver packet format."""

MODE_NAMES = ('NORMAL', 'RECOVERY')


def recording_packet(command, revision, enabled, mode):
    label = MODE_NAMES[mode] if enabled else 'OFF'
    return b'REC:%d:%s:%s' % (revision, label.encode('ascii'), command)


def parse_recording_packet(packet):
    if not packet or not packet.startswith(b'REC:'):
        return None
    try:
        _, revision, label, command = packet.split(b':')
        revision = int(revision)
        label = label.decode('ascii')
        mode = None if label == 'OFF' else MODE_NAMES.index(label)
    except (ValueError, UnicodeError):
        return None
    return (command, revision, mode) if revision >= 0 else None
