import json
from pathlib import Path
from datetime import datetime

JSON_DIR = Path("data/processed/json")
REPORTS_DIR = Path("data/processed/reports")

def load_json_file(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_debate_key(data: dict):
    meta = data.get("debate_meta", {})
    return (
        str(meta.get("period", "")),
        str(meta.get("legislature", "")),
        str(meta.get("legislative_session", "")),
        str(meta.get("number", "")),
        str(meta.get("date", ""))
    )

def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(JSON_DIR.glob("*.json"))
    if not json_files:
        print(f"Não encontrei ficheiros JSON em: {JSON_DIR.resolve()}")
        return

    grouped = {}
    invalid_files = []

    for json_file in json_files:
        try:
            data = load_json_file(json_file)
            key = build_debate_key(data)
            grouped.setdefault(key, []).append(json_file.name)
        except Exception as e:
            invalid_files.append((json_file.name, str(e)))

    connected_groups = {k: v for k, v in grouped.items() if len(v) > 1}

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    report_path = REPORTS_DIR / f"files_connection_report_{timestamp}.txt"

    lines = []
    lines.append("FILES CONNECTION REPORT")
    lines.append(f"Data: {now.strftime('%Y-%m-%d')}")
    lines.append(f"Hora: {now.strftime('%H:%M:%S')}")
    lines.append(f"Timestamp: {timestamp}")
    lines.append("")
    lines.append(f"Total de ficheiros analisados: {len(json_files)}")
    lines.append(f"Total de grupos parlamentares repetidos: {len(connected_groups)}")
    lines.append("")

    if connected_groups:
        lines.append("GRUPOS COM CONEXÃO DETETADA")
        lines.append("")

        for idx, (key, files) in enumerate(sorted(connected_groups.items()), start=1):
            period, legislature, legislative_session, number, date = key
            lines.append(f"[Grupo {idx}]")
            lines.append(f"period={period} | legislature={legislature} | legislative_session={legislative_session} | number={number} | date={date}")
            lines.append("Ficheiros envolvidos:")
            for file_name in files:
                lines.append(f"  - {file_name}")
            lines.append("")
    else:
        lines.append("Não foram detetadas conexões entre ficheiros com a mesma identificação parlamentar.")
        lines.append("")

    if invalid_files:
        lines.append("FICHEIROS COM ERRO")
        lines.append("")
        for file_name, error in invalid_files:
            lines.append(f"{file_name} -> {error}")
        lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Relatório criado com sucesso: {report_path}")

if __name__ == "__main__":
    main()