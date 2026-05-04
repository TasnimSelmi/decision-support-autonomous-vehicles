import importlib
from importlib import util
from pathlib import Path


_SCENARIO_ALIASES = {
    "day": "combined_edge_case",
    "normal": "combined_edge_case",
    "normal_mode": "combined_edge_case",
    "rainy_night_mode": "combined_edge_case",
    "night_mode": "night",
    "4-_carcollision": "car_collision",
    "4-_carcollision.py": "car_collision",
    "carcollision": "car_collision",
    "car_collision": "car_collision",
    "rain": "rain",
    "night": "night",
    "pedestrian_collision": "pedestrian_collision",
    "combined_edge_case": "combined_edge_case",
}

_LEGACY_SCENARIO_FILES = {
    "night": Path(__file__).resolve().parents[1] / "Carla Scenarios" / "night_mode.py",
    "car_collision": Path(__file__).resolve().parents[1] / "Carla Scenarios" / "4- CarCollision.py",
    "rain": Path(__file__).resolve().parents[1] / "Carla Scenarios" / "rainy_night_mode.py",
    "combined_edge_case": Path(__file__).resolve().parents[1] / "Carla Scenarios" / "rainy_night_mode.py",
}


def load_scenario(name):
    scenario_name = _SCENARIO_ALIASES.get(str(name).lower(), str(name).lower())

    try:
        module = importlib.import_module(f"src.carla_scenarios.{scenario_name}")
    except ModuleNotFoundError:
        legacy_file = _LEGACY_SCENARIO_FILES.get(scenario_name)

        if legacy_file is None or not legacy_file.exists():
            raise

        spec = util.spec_from_file_location(f"legacy_{scenario_name}_scenario", legacy_file)

        if spec is None or spec.loader is None:
            raise ModuleNotFoundError(f"Unable to load legacy scenario: {legacy_file}")

        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)

    return module.apply_scenario