import json
from pathlib import Path
import math

BASE_DIR = Path(__file__).resolve().parent
METRICS_DIR = BASE_DIR / "Results"

EXPECTED = {
    "global": {
        "total_files": 4,
        "total_utterances": 16,
        "total_words": 19
    },
    "per_file": {
        "test_01_alternancia.json": {
            "utterances": 4,
            "unique_speakers": 2,
            "alternation_rate": 0.75,
            "avg_block_length": 1.0,
            "top1_dominance": 0.5,
            "top3_dominance": 1.0,
            "roles": {
                "NO_ROLE": 4
            }
        },
        "test_02_dominancia.json": {
            "utterances": 5,
            "unique_speakers": 2,
            "alternation_rate": 0.2,
            "avg_block_length": 2.5,
            "top1_dominance": 0.8,
            "top3_dominance": 1.0,
            "roles": {
                "NO_ROLE": 5
            }
        },
        "test_03_bloco_unico.json": {
            "utterances": 3,
            "unique_speakers": 1,
            "alternation_rate": 0.0,
            "avg_block_length": 3.0,
            "top1_dominance": 1.0,
            "top3_dominance": 1.0,
            "roles": {
                "NO_ROLE": 3
            }
        },
        "test_04_roles.json": {
            "utterances": 4,
            "unique_speakers": 4,
            "alternation_rate": 0.75,
            "avg_block_length": 1.0,
            "top1_dominance": 0.25,
            "top3_dominance": 0.75,
            "roles": {
                "NO_ROLE": 3,
                "president": 1
            }
        }
    },
    "transitions": {
        "A -> B": 3,
        "B -> A": 1,
        "SUMÁRIO -> Presidente": 1,
        "Presidente -> Secretária": 1,
        "Secretária -> Deputado X": 1
    }
}


def latest_metrics_file():
    files = sorted(METRICS_DIR.glob("metrics_*.json"))
    if not files:
        raise FileNotFoundError(
            f"Não encontrei ficheiros metrics_*.json em {METRICS_DIR.resolve()}"
        )
    return files[-1]


def load_metrics(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def float_equal(a, b, tol=1e-9):
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def check_value(label, expected, obtained, errors):
    if isinstance(expected, float):
        ok = float_equal(expected, obtained)
    else:
        ok = expected == obtained

    if ok:
        print(f"[OK] {label}: esperado={expected} | obtido={obtained}")
    else:
        print(f"[FAIL] {label}: esperado={expected} | obtido={obtained}")
        errors.append((label, expected, obtained))


def main():
    metrics_file = latest_metrics_file()
    data = load_metrics(metrics_file)

    print(f"A validar ficheiro: {metrics_file.name}\n")

    errors = []

    print("=== MÉTRICAS GLOBAIS ===")
    for key, expected_value in EXPECTED["global"].items():
        obtained_value = data.get("global", {}).get(key)
        check_value(f"global.{key}", expected_value, obtained_value, errors)

    print("\n=== MÉTRICAS POR FICHEIRO ===")
    for file_name, expected_metrics in EXPECTED["per_file"].items():
        obtained_metrics = data.get("per_file", {}).get(file_name, {})

        for key, expected_value in expected_metrics.items():
            if key == "roles":
                obtained_roles = obtained_metrics.get("roles", {})
                for role_name, expected_role_count in expected_value.items():
                    obtained_role_count = obtained_roles.get(role_name, 0)
                    check_value(
                        f"{file_name}.roles.{role_name}",
                        expected_role_count,
                        obtained_role_count,
                        errors
                    )
            else:
                obtained_value = obtained_metrics.get(key)
                check_value(f"{file_name}.{key}", expected_value, obtained_value, errors)

    print("\n=== TRANSIÇÕES ===")
    obtained_transitions = data.get("transitions", {})
    for transition, expected_count in EXPECTED["transitions"].items():
        obtained_count = obtained_transitions.get(transition, 0)
        check_value(f"transitions.{transition}", expected_count, obtained_count, errors)

    print("\n=== RESULTADO FINAL ===")
    if not errors:
        print("Todos os testes passaram com sucesso.")
    else:
        print(f"Foram encontrados {len(errors)} erros.")
        for label, expected, obtained in errors:
            print(f" - {label}: esperado={expected} | obtido={obtained}")


if __name__ == "__main__":
    main()