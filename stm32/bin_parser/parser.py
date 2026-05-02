import argparse
import os
from stm_utils import read_packets_from_file, save_to_npz
from packet import Packet


def print_summary(packets: list[Packet]) -> None:
    print("\n--- Povzetek ---")
    print(f"Skupaj paketov: {len(packets)}")

    if not packets:
        return

    timestamps = [p.ts for p in packets]
    print(f"Čas začetka:    {min(timestamps):.0f} ms")
    print(f"Čas konca:      {max(timestamps):.0f} ms")
    print(f"Trajanje:       {max(timestamps) - min(timestamps):.0f} ms")

    counts: dict[str, int] = {}
    for p in packets:
        counts[p.sensor] = counts.get(p.sensor, 0) + p.sample_count

    print("\nSampli po senzorju:")
    for name, count in counts.items():
        print(f"  {name}: {count}")
    print()


def print_packets(packets: list[Packet], limit: int = 5) -> None:
    print(f"\n--- Prvih {min(limit, len(packets))} paketov ---")
    for p in packets[:limit]:
        print(p)
        for row in p.data:
            print(f"  X={row[0]:6d}  Y={row[1]:6d}  Z={row[2]:6d}")


def main():
    parser = argparse.ArgumentParser(
        description='STM32 data logger — dekodiranje in shranjevanje binarnih podatkov'
    )
    parser.add_argument('input', help='Pot do binarnega fila (.bin)')
    parser.add_argument(
        '-o', '--output', help='Pot do izhodnega .npz fila (privzeto: <input>.npz)'
    )
    parser.add_argument(
        '--preview', action='store_true', help='Izpiši prvih 5 paketov'
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Napaka: file '{args.input}' ne obstaja.")
        return

    output_path = args.output or os.path.splitext(args.input)[0] + '.npz'

    packets = read_packets_from_file(args.input)

    if not packets:
        print("Napaka: ni bilo mogoče parsirati nobenega paketa.")
        return

    print_summary(packets)

    if args.preview:
        print_packets(packets)

    save_to_npz(packets, output_path)


if __name__ == "__main__":
    main()