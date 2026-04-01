import json
import math
from pathlib import Path
from datetime import datetime
from collections import Counter

BASE_DIR = Path("annotation_agreement")
CLAUDE_PATH = BASE_DIR / "claude.json"
HUMAN1_PATH = BASE_DIR / "human1.json"
HUMAN2_PATH = BASE_DIR / "human2.json"
RESULTS_DIR = BASE_DIR / "results"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_category(value):
    if value is None:
        return "MISSING"
    if isinstance(value, list):
        return " | ".join(str(x).strip() for x in value)
    return str(value).strip() if str(value).strip() else "MISSING"


def accuracy(a, b):
    if not a:
        return 0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def cohens_kappa(a, b):
    n = len(a)
    if n == 0:
        return 0

    po = accuracy(a, b)
    ca = Counter(a)
    cb = Counter(b)

    pe = sum((ca[k] / n) * (cb[k] / n) for k in set(a) | set(b))

    if pe == 1:
        return 1.0

    return (po - pe) / (1 - pe)


def fleiss_kappa(matrix):
    if not matrix:
        return 0

    N = len(matrix)
    k = len(matrix[0])
    n = sum(matrix[0])

    if n <= 1:
        return 0

    p = []
    for j in range(k):
        p_j = sum(matrix[i][j] for i in range(N)) / (N * n)
        p.append(p_j)

    P = []
    for row in matrix:
        P_i = (sum(x * x for x in row) - n) / (n * (n - 1))
        P.append(P_i)

    P_bar = sum(P) / N
    P_e = sum(x * x for x in p)

    if P_e == 1:
        return 1.0

    return (P_bar - P_e) / (1 - P_e)


def build_fleiss_matrix(triples):
    categories = sorted(set(v for t in triples for v in t))
    idx = {c: i for i, c in enumerate(categories)}

    matrix = []
    for t in triples:
        row = [0] * len(categories)
        for v in t:
            row[idx[v]] += 1
        matrix.append(row)

    return matrix, categories


def jaccard_similarity(list_a, list_b):
    set_a = set(list_a)
    set_b = set(list_b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def overlap_at_k(list_a, list_b, k=3):
    set_a = set(list_a[:k])
    set_b = set(list_b[:k])
    if k == 0:
        return 0
    return len(set_a & set_b) / k


def infer_source_file_from_claude_debate(debate):
    meta = debate.get("debate_meta", {})
    legislature = str(meta.get("legislature", "")).zfill(2)
    session = str(meta.get("legislative_session", ""))
    number = str(meta.get("number", "")).zfill(3)
    return f"{legislature}-{session}-{number}.json"


def aggregate_claude_debate(debate):
    utterances = debate.get("utterances", [])

    total_interventions = len(utterances)

    speaker_counter = Counter()
    party_counter = Counter()

    direct_responses = 0
    formal_speeches = 0
    interruptions_apartes = 0

    for utt in utterances:
        speaker = utt.get("speaker", {})
        canonical_name = speaker.get("canonical_name") or speaker.get("original_string") or "UNKNOWN_SPEAKER"
        party = speaker.get("party", "NO_PARTY")

        speaker_counter[canonical_name] += 1
        party_counter[party] += 1

        interaction = utt.get("interaction", {})
        if interaction.get("is_response") is True or interaction.get("responds_to_previous") is True:
            direct_responses += 1

        utterance_type = str(utt.get("utterance_type", "")).lower()
        speech_act = str(utt.get("speech_act", "")).lower()

        if utterance_type in {"procedural", "administrative", "summary"} or speech_act == "procedural":
            formal_speeches += 1

        if speech_act == "interruption":
            interruptions_apartes += 1

    most_active_speakers = []
    for speaker_name, count in speaker_counter.most_common(3):
        most_active_speakers.append({
            "speaker": speaker_name,
            "count": count
        })

    top_3_parties = [party for party, _ in party_counter.most_common(3)]

    # estilo de debate heurístico
    if total_interventions == 0:
        debate_style = "unknown"
        interactivity_level = "unknown"
    else:
        response_ratio = direct_responses / total_interventions
        if response_ratio >= 0.4:
            debate_style = "altamente interativo"
            interactivity_level = "alto"
        elif response_ratio >= 0.2:
            debate_style = "moderadamente interativo"
            interactivity_level = "moderado"
        else:
            debate_style = "pouco interativo"
            interactivity_level = "baixo"

    # rede de interação agregada
    interaction_network = []
    edge_counter = Counter()
    for utt in utterances:
        speaker = utt.get("speaker", {})
        source_party = speaker.get("party", "NO_PARTY")
        interaction = utt.get("interaction", {})
        targets = interaction.get("targets", [])

        for target in targets:
            edge_counter[(source_party, str(target), "response" if interaction.get("is_response") else "address")] += 1

    for (src, dst, typ), count in edge_counter.most_common(10):
        interaction_network.append({
            "from": src,
            "to": dst,
            "type": typ,
            "count": count
        })

    return {
        "total_interventions": total_interventions,
        "most_active_speakers": most_active_speakers,
        "top_3_parties": top_3_parties,
        "interaction_patterns": {
            "direct_responses": direct_responses,
            "interruptions_apartes": interruptions_apartes,
            "formal_speeches": formal_speeches,
            "interactivity_level": interactivity_level
        },
        "qualitative_summary": {
            "debate_style": debate_style
        },
        "interaction_network": interaction_network
    }


def build_human_map(human_data):
    result = {}
    for item in human_data.get("files_ordered", []):
        source_file = item.get("source_file")
        result[source_file] = item
    return result


def build_claude_map(claude_data):
    result = {}
    for debate in claude_data.get("debates", []):
        source_file = infer_source_file_from_claude_debate(debate)
        result[source_file] = {
            "source_file": source_file,
            "analysis": aggregate_claude_debate(debate)
        }
    return result


def compare_numeric(field_name, c_val, h1_val, h2_val):
    return {
        "claude": c_val,
        "human1": h1_val,
        "human2": h2_val,
        "abs_diff_c_h1": abs(c_val - h1_val),
        "abs_diff_c_h2": abs(c_val - h2_val),
        "abs_diff_h1_h2": abs(h1_val - h2_val)
    }


def extract_network_edges(network):
    edges = set()
    for edge in network:
        frm = str(edge.get("from", ""))
        to = str(edge.get("to", ""))
        typ = str(edge.get("type", ""))
        edges.add((frm, to, typ))
    return edges


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    claude_data = load_json(CLAUDE_PATH)
    human1_data = load_json(HUMAN1_PATH)
    human2_data = load_json(HUMAN2_PATH)

    claude_map = build_claude_map(claude_data)
    human1_map = build_human_map(human1_data)
    human2_map = build_human_map(human2_data)

    common_files = sorted(set(claude_map) & set(human1_map) & set(human2_map))
    if not common_files:
        raise ValueError("Não há debates em comum entre os 3 ficheiros.")

    per_file_results = {}

    categorical_records = {
        "interactivity_level": [],
        "debate_style": []
    }

    aggregate_numeric = {
        "total_interventions": [],
        "direct_responses": [],
        "formal_speeches": [],
        "interruptions_apartes": []
    }

    aggregate_list_metrics = {
        "top_3_parties": [],
        "interaction_network_jaccard": []
    }

    for file_name in common_files:
        c = claude_map[file_name]["analysis"]
        h1 = human1_map[file_name]["analysis"]
        h2 = human2_map[file_name]["analysis"]

        result = {}

        # numéricos
        result["numeric"] = {
            "total_interventions": compare_numeric(
                "total_interventions",
                c.get("total_interventions", 0),
                h1.get("total_interventions", 0),
                h2.get("total_interventions", 0)
            ),
            "direct_responses": compare_numeric(
                "direct_responses",
                c.get("interaction_patterns", {}).get("direct_responses", 0),
                h1.get("interaction_patterns", {}).get("direct_response", 0) if isinstance(h1.get("interaction_patterns", {}).get("direct_response", 0), (int, float)) else 0,
                h2.get("interaction_patterns", {}).get("direct_responses", 0)
            ),
            "formal_speeches": compare_numeric(
                "formal_speeches",
                c.get("interaction_patterns", {}).get("formal_speeches", 0),
                h1.get("interaction_patterns", {}).get("formal_speech", 0) if isinstance(h1.get("interaction_patterns", {}).get("formal_speech", 0), (int, float)) else 0,
                h2.get("interaction_patterns", {}).get("formal_speeches", 0)
            ),
            "interruptions_apartes": compare_numeric(
                "interruptions_apartes",
                c.get("interaction_patterns", {}).get("interruptions_apartes", 0),
                h1.get("interaction_patterns", {}).get("interruption_aparte", 0) if isinstance(h1.get("interaction_patterns", {}).get("interruption_aparte", 0), (int, float)) else 0,
                h2.get("interaction_patterns", {}).get("interruptions_apartes", 0)
            )
        }

        for key in result["numeric"]:
            aggregate_numeric[key].append(result["numeric"][key])

        # categóricos
        c_interactivity = normalize_category(c.get("interaction_patterns", {}).get("interactivity_level", "MISSING"))
        h1_interactivity = normalize_category(h1.get("interaction_patterns", {}).get("interactivity_level", "MISSING"))
        h2_interactivity = normalize_category(h2.get("interaction_patterns", {}).get("interactivity_level", "MISSING"))

        c_debate_style = normalize_category(c.get("qualitative_summary", {}).get("debate_style", "MISSING"))
        h1_debate_style = normalize_category(h1.get("qualitative_summary", {}).get("debate_style", "MISSING"))
        h2_debate_style = normalize_category(h2.get("analysis", {}).get("debate_style", h2.get("qualitative_summary", {}).get("debate_style", "MISSING")))

        categorical_records["interactivity_level"].append((c_interactivity, h1_interactivity, h2_interactivity))
        categorical_records["debate_style"].append((c_debate_style, h1_debate_style, h2_debate_style))

        result["categorical"] = {
            "interactivity_level": {
                "claude": c_interactivity,
                "human1": h1_interactivity,
                "human2": h2_interactivity
            },
            "debate_style": {
                "claude": c_debate_style,
                "human1": h1_debate_style,
                "human2": h2_debate_style
            }
        }

        # listas
        c_top3 = c.get("top_3_parties", [])
        h1_top3 = h1.get("top_3_parties", [])
        h2_top3 = h2.get("party_dynamics", {}).get("main_axis", [])
        if isinstance(h2_top3, str):
            h2_top3 = [h2_top3]

        top3_metrics = {
            "claude_vs_human1_overlap@3": overlap_at_k(c_top3, h1_top3, 3),
            "claude_vs_human2_overlap@3": overlap_at_k(c_top3, h2_top3, 3),
            "human1_vs_human2_overlap@3": overlap_at_k(h1_top3, h2_top3, 3),
            "claude_vs_human1_jaccard": jaccard_similarity(c_top3, h1_top3),
            "claude_vs_human2_jaccard": jaccard_similarity(c_top3, h2_top3),
            "human1_vs_human2_jaccard": jaccard_similarity(h1_top3, h2_top3)
        }

        aggregate_list_metrics["top_3_parties"].append(top3_metrics)

        c_edges = extract_network_edges(c.get("interaction_network", []))
        h1_edges = extract_network_edges(h1.get("interaction_network", []))
        h2_edges = extract_network_edges(h2.get("interaction_network", []))

        network_metrics = {
            "claude_vs_human1_jaccard": jaccard_similarity(list(c_edges), list(h1_edges)),
            "claude_vs_human2_jaccard": jaccard_similarity(list(c_edges), list(h2_edges)),
            "human1_vs_human2_jaccard": jaccard_similarity(list(h1_edges), list(h2_edges))
        }

        aggregate_list_metrics["interaction_network_jaccard"].append(network_metrics)

        result["list_metrics"] = {
            "top_3_parties": top3_metrics,
            "interaction_network": network_metrics
        }

        per_file_results[file_name] = result

    # Acordo categórico
    categorical_summary = {}
    for field_name, triples in categorical_records.items():
        c = [t[0] for t in triples]
        h1 = [t[1] for t in triples]
        h2 = [t[2] for t in triples]

        matrix, categories = build_fleiss_matrix(triples)

        categorical_summary[field_name] = {
            "categories": categories,
            "pairwise_accuracy": {
                "claude_vs_human1": accuracy(c, h1),
                "claude_vs_human2": accuracy(c, h2),
                "human1_vs_human2": accuracy(h1, h2)
            },
            "pairwise_cohens_kappa": {
                "claude_vs_human1": cohens_kappa(c, h1),
                "claude_vs_human2": cohens_kappa(c, h2),
                "human1_vs_human2": cohens_kappa(h1, h2)
            },
            "fleiss_kappa": fleiss_kappa(matrix)
        }

    # Resumo numérico
    numeric_summary = {}
    for metric_name, records in aggregate_numeric.items():
        numeric_summary[metric_name] = {
            "avg_abs_diff_claude_human1": sum(r["abs_diff_c_h1"] for r in records) / len(records),
            "avg_abs_diff_claude_human2": sum(r["abs_diff_c_h2"] for r in records) / len(records),
            "avg_abs_diff_human1_human2": sum(r["abs_diff_h1_h2"] for r in records) / len(records)
        }

    # Resumo listas
    list_summary = {
        "top_3_parties": {
            "avg_overlap@3_claude_human1": sum(x["claude_vs_human1_overlap@3"] for x in aggregate_list_metrics["top_3_parties"]) / len(aggregate_list_metrics["top_3_parties"]),
            "avg_overlap@3_claude_human2": sum(x["claude_vs_human2_overlap@3"] for x in aggregate_list_metrics["top_3_parties"]) / len(aggregate_list_metrics["top_3_parties"]),
            "avg_overlap@3_human1_human2": sum(x["human1_vs_human2_overlap@3"] for x in aggregate_list_metrics["top_3_parties"]) / len(aggregate_list_metrics["top_3_parties"])
        },
        "interaction_network": {
            "avg_jaccard_claude_human1": sum(x["claude_vs_human1_jaccard"] for x in aggregate_list_metrics["interaction_network_jaccard"]) / len(aggregate_list_metrics["interaction_network_jaccard"]),
            "avg_jaccard_claude_human2": sum(x["claude_vs_human2_jaccard"] for x in aggregate_list_metrics["interaction_network_jaccard"]) / len(aggregate_list_metrics["interaction_network_jaccard"]),
            "avg_jaccard_human1_human2": sum(x["human1_vs_human2_jaccard"] for x in aggregate_list_metrics["interaction_network_jaccard"]) / len(aggregate_list_metrics["interaction_network_jaccard"])
        }
    }

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output = {
        "generated_at": now,
        "files_compared": common_files,
        "methodological_note": "The three annotators do not share the same annotation granularity. Claude is utterance-level, while human1 and human2 are debate-level analytical summaries. Therefore, agreement is computed on a normalized debate-level comparison layer, not on classical utterance-level IAA.",
        "categorical_summary": categorical_summary,
        "numeric_summary": numeric_summary,
        "list_summary": list_summary,
        "per_file_results": per_file_results
    }

    json_path = RESULTS_DIR / f"agreement_report_{now}.json"
    txt_path = RESULTS_DIR / f"agreement_report_{now}.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append("ANNOTATION AGREEMENT REPORT")
    lines.append(f"Generated at: {now}")
    lines.append("")
    lines.append("Methodological note:")
    lines.append(output["methodological_note"])
    lines.append("")
    lines.append("CATEGORICAL SUMMARY")
    for field, summary in categorical_summary.items():
        lines.append(f"Field: {field}")
        lines.append(f"  Fleiss' Kappa: {summary['fleiss_kappa']:.4f}")
        lines.append(f"  Cohen Claude vs Human1: {summary['pairwise_cohens_kappa']['claude_vs_human1']:.4f}")
        lines.append(f"  Cohen Claude vs Human2: {summary['pairwise_cohens_kappa']['claude_vs_human2']:.4f}")
        lines.append(f"  Cohen Human1 vs Human2: {summary['pairwise_cohens_kappa']['human1_vs_human2']:.4f}")
        lines.append("")

    lines.append("NUMERIC SUMMARY")
    for metric, summary in numeric_summary.items():
        lines.append(f"Metric: {metric}")
        for k, v in summary.items():
            lines.append(f"  {k}: {v:.4f}")
        lines.append("")

    lines.append("LIST SUMMARY")
    for metric, summary in list_summary.items():
        lines.append(f"Metric: {metric}")
        for k, v in summary.items():
            lines.append(f"  {k}: {v:.4f}")
        lines.append("")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Agreement JSON criado: {json_path}")
    print(f"Agreement TXT criado: {txt_path}")


if __name__ == "__main__":
    main()