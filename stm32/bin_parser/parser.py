import argparse
import os
import glob
from stm_utils import read_packets_from_file, save_to_npz
from AutoDNA.stm32.bin_parser.packet import Packet


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


def process_file(input_path: str, output_path: str, preview: bool = False) -> bool:
    print(f"\n=== {input_path} ===")
    packets = read_packets_from_file(input_path)

    if not packets:
        print(f"Napaka: ni bilo mogoče parsirati nobenega paketa iz '{input_path}'.")
        return False

    print_summary(packets)

    if preview:
        print_packets(packets)

    save_to_npz(packets, output_path)
    return True


def main():
    parser = argparse.ArgumentParser(
        description='STM32 data logger — dekodiranje in shranjevanje binarnih podatkov'
    )
    parser.add_argument('input', help='Pot do binarnega fila (.bin) ali mape z .bin fili')
    parser.add_argument(
        '-o', '--output',
        help='Pot do izhodnega .npz fila (single mode) ali izhodne mape (folder mode). '
             'Privzeto: <input>.npz oz. <input_dir>_npz/'
    )
    parser.add_argument(
        '--preview', action='store_true', help='Izpiši prvih 5 paketov'
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Napaka: pot '{args.input}' ne obstaja.")
        return

    # Folder mode
    if os.path.isdir(args.input):
        bin_files = sorted(
            glob.glob(os.path.join(args.input, '*.bin')) +
            glob.glob(os.path.join(args.input, '*.BIN'))
        )

        if not bin_files:
            print(f"Napaka: v mapi '{args.input}' ni .bin filov.")
            return

        out_dir = args.output or args.input.rstrip(os.sep) + '_npz'
        os.makedirs(out_dir, exist_ok=True)

        print(f"Najdenih {len(bin_files)} .bin filov. Izhodna mapa: {out_dir}")

        ok = 0
        for bin_path in bin_files:
            base = os.path.splitext(os.path.basename(bin_path))[0]
            out_path = os.path.join(out_dir, base + '.npz')
            try:
                if process_file(bin_path, out_path, preview=args.preview):
                    ok += 1
            except Exception as e:
                print(f"  [!] Napaka pri '{bin_path}': {e}")

        print(f"\n=== Končano: {ok}/{len(bin_files)} uspešno ===")
        return

    # Single-file mode
    output_path = args.output or os.path.splitext(args.input)[0] + '.npz'
    process_file(args.input, output_path, preview=args.preview)


if __name__ == "__main__":
    main()