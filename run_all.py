"""Run all capstone parts in dependency order. Execute from repository root."""
from src.part_a import run as part_a
from src.part_b import run as part_b
from src.part_c import run as part_c
from src.part_d import run as part_d

if __name__ == '__main__':
    print('\n=== PART A ==='); part_a()
    print('\n=== PART B ==='); part_b()
    print('\n=== PART C ==='); part_c()
    print('\n=== PART D ==='); part_d()
    print('\nAll pipeline stages completed.')
