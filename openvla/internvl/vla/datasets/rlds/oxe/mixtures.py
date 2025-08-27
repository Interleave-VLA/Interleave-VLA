"""
mixtures.py

Defines a registry of dataset mixtures and weights for the Open-X Embodiment Datasets. Each dataset is associated with
a float "sampling weight"
"""

from typing import Dict, List, Tuple

# fmt: off
INTERLEAVED_MIXTURES: Dict[str, List[Tuple[str, float]]] = {
    # === Vima Datasets (Modified Versions) ===
    "mix_all": [
        ('se2_task1', 1.0),('se2_task2', 1.0),('se2_task3', 1.0),
        ('se2_task4', 1.0),('se2_task7', 1.0),('se2_task10', 1.0),
        ('se2_task11', 1.0),('se2_task14', 1.0),('se2_task15', 1.0)
    ],
    "mix_no3": [
        ('se2_task1', 1.0),('se2_task2', 1.0),
        ('se2_task4', 1.0),('se2_task7', 1.0),('se2_task10', 1.0),
        ('se2_task11', 1.0),('se2_task14', 1.0),('se2_task15', 1.0)
    ],
    "mix_no10": [
        ('se2_task1', 1.0),('se2_task2', 1.0),('se2_task3', 1.0),
        ('se2_task4', 1.0),('se2_task7', 1.0),
        ('se2_task11', 1.0),('se2_task14', 1.0),('se2_task15', 1.0)
    ],
    "mix_1247": [
        ('se2_task1', 1.0),('se2_task2', 1.0),
        ('se2_task4', 1.0),('se2_task7', 1.0)
    ],
    "mix_127": [
        ('se2_task1', 1.0),('se2_task2', 1.0),('se2_task7', 1.0)
    ],
    "mix_17": [
        ('se2_task1', 1.0),('se2_task7', 1.0)
    ],
    "mix_1415": [
        ('se2_task14', 1.0),('se2_task15', 1.0)
    ],
    "mix_1011": [
        ('se2_task10', 1.0),('se2_task11', 1.0)
    ],
    "bridge_dataset": [
        ("bridge_dataset", 1.0),
    ],
    "vima_dataset": [
        ("vima_dataset", 1.0),
    ],
    "vima_dataset:0.1.0": [
        ("vima_dataset:0.1.0", 1.0),
    ],
    "vima_task7_dataset": [
        ("vima_task7_dataset", 1.0),
    ],
    "vima_task7_dataset:0.1.0": [
        ("vima_task7_dataset:0.1.0", 1.0),
    ],
    "vima_task2_dataset": [
        ("vima_task2_dataset", 1.0),
    ],
    "vima_task2_dataset:0.1.0": [
        ("vima_task2_dataset:0.1.0", 1.0),
    ],
    "se2_dataset": [
        ("se2_dataset", 1.0),
    ],
    "se2_dataset:0.1.0": [
        ("se2_dataset:0.1.0", 1.0),
    ]
}
OXE_NAMED_MIXTURES: Dict[str, List[Tuple[str, float]]] = {
    # === LIBERO Datasets (Modified Versions) ===
    "libero_spatial_no_noops": [
        ("libero_spatial_no_noops", 1.0),
    ],
    "libero_object_no_noops": [
        ("libero_object_no_noops", 1.0),
    ],
    "libero_goal_no_noops": [
        ("libero_goal_no_noops", 1.0),
    ],
    "libero_10_no_noops": [
        ("libero_10_no_noops", 1.0),
    ],
}
# fmt: on
