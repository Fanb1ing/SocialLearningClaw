#!/usr/bin/env python3
"""Generate a four-condition, reviewable IntPhys2 Gold Schema pilot."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data/intphys2"
OUTPUT_ROOT = PROJECT_ROOT / "gold/intphys2/v1"
TYPE_ORDER = ("1_Possible", "1_Impossible", "2_Possible", "2_Impossible")


SCENES: list[dict[str, Any]] = [
    {
        "scene_key": "debug:3",
        "split": "debug",
        "scene_index": "3",
        "condition": "permanence",
        "frame_indices": [0, 60, 270, 450, 510, 600],
        "invariant_title": "遮挡不会创造或抹除物体",
        "trigger": "目标物体及其容器经历一段暂时遮挡，且没有可见的进入、离开或销毁事件。",
        "expectation": "遮挡前存在的物体在遮挡后仍应存在；遮挡前不存在的物体不能在遮挡后凭空出现。",
        "violations": ["可见物体进入遮挡后永久消失。", "遮挡结束后出现此前不存在且无进入路径的物体。"],
        "pair_summaries": {
            "1": {
                "possible": "黄色球在遮挡前可见，热气球转回后同一黄色球再次可见。",
                "impossible": "黄色球在遮挡前可见，但热气球转回后球不再出现。",
                "focus_frames": [0, 60, 450, 510, 600],
            },
            "2": {
                "possible": "遮挡前后均未观察到黄色球，没有新增目标物体。",
                "impossible": "遮挡前未观察到黄色球，遮挡结束后篮中出现黄色球，且没有可见进入路径。",
                "focus_frames": [0, 60, 450, 510, 600],
            },
        },
    },
    {
        "scene_key": "main_300:21",
        "split": "main_300",
        "scene_index": "21",
        "condition": "immutability",
        "frame_indices": [0, 60, 270, 450, 510, 600],
        "invariant_title": "暂时遮挡不改变物体的稳定属性",
        "trigger": "同一球体进入暂时遮挡，期间没有可见的涂色、替换或变形过程。",
        "expectation": "球体重新出现时应保持遮挡前可识别的颜色属性。",
        "violations": ["蓝色球在无遮挡的因果过程下变为红色。", "红色球在无遮挡的因果过程下变为蓝色。"],
        "pair_summaries": {
            "1": {
                "possible": "球在遮挡前后均为蓝色。",
                "impossible": "球在遮挡前为蓝色，重新出现后变为红色。",
                "focus_frames": [0, 60, 450, 510, 600],
            },
            "2": {
                "possible": "球在遮挡前后均为红色。",
                "impossible": "球在遮挡前为红色，重新出现后变为蓝色。",
                "focus_frames": [0, 60, 450, 510, 600],
            },
        },
    },
    {
        "scene_key": "main_300:19",
        "split": "main_300",
        "scene_index": "19",
        "condition": "continuity",
        "frame_indices": [0, 60, 270, 450, 510, 600],
        "invariant_title": "物体不能在分离容器之间无路径换位",
        "trigger": "蓝色方块在两个空间分离的杯位之一进入遮挡，且没有可见的跨杯通道或转移过程。",
        "expectation": "方块重新可见时仍应位于遮挡前的同一杯位；若换到另一杯位，必须存在连续转移路径。",
        "violations": ["方块由左杯消失并直接在右杯出现。", "方块由右杯消失并直接在左杯出现。"],
        "pair_summaries": {
            "1": {
                "possible": "蓝色方块遮挡前后都位于左侧杯位。",
                "impossible": "蓝色方块遮挡前位于左侧杯位，之后直接出现在右侧杯位。",
                "focus_frames": [0, 60, 450, 510, 600],
            },
            "2": {
                "possible": "蓝色方块遮挡前后都位于右侧杯位。",
                "impossible": "蓝色方块遮挡前位于右侧杯位，之后直接出现在左侧杯位。",
                "focus_frames": [0, 60, 450, 510, 600],
            },
        },
    },
    {
        "scene_key": "debug:1",
        "split": "debug",
        "scene_index": "1",
        "condition": "solidity",
        "frame_indices": [150, 180, 210, 240, 300, 480],
        "split_aliases": ["main_300:1"],
        "invariant_title": "碰撞响应必须与可见实体接触一致",
        "trigger": "大型箱体下落到平台上的固定落点；该落点可能有或没有一个可见黄色方块。",
        "expectation": "存在黄色方块时，箱体不能占据同一空间并应出现相应碰撞响应；不存在方块时，不应受到来自空位置的碰撞冲量。",
        "violations": ["箱体在可见黄色方块所在位置保持直立落下，缺少排斥或碰撞响应。", "落点没有黄色方块时，箱体却发生与撞击方块相似的倾转。"],
        "pair_summaries": {
            "1": {
                "possible": "落点有黄色方块；箱体接触后明显倾转，表现出实体碰撞响应。",
                "impossible": "落点有黄色方块；箱体仍直立占据该落点，黄色方块被视觉上吞没。",
                "focus_frames": [150, 180, 210, 240, 300, 480],
            },
            "2": {
                "possible": "落点没有黄色方块；箱体直立落到平台并保持稳定。",
                "impossible": "落点没有黄色方块；箱体却在同一位置发生类似碰撞后的倾转。",
                "focus_frames": [150, 180, 210, 240, 300, 480],
            },
        },
        "review_note": "本场景依赖对碰撞响应而非单帧重叠的判断，是 pilot 中最需要人工确认的语义标注。",
    },
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def metadata_path(split: str) -> Path:
    return DATA_ROOT / ("Debug/metadata.csv" if split == "debug" else "Main/sample_300.csv")


def videos_root(split: str) -> Path:
    return DATA_ROOT / ("Debug/Videos" if split == "debug" else "Main/Videos")


def load_scene_rows(scene: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(metadata_path(scene["split"]), dtype={"SceneIndex": str})
    rows = frame[frame["SceneIndex"] == scene["scene_index"]].copy()
    if set(rows["type"]) != set(TYPE_ORDER) or len(rows) != 4:
        raise RuntimeError(f"Incomplete four-video group for {scene['scene_key']}")
    rows["_order"] = rows["type"].map({name: index for index, name in enumerate(TYPE_ORDER)})
    return rows.sort_values("_order")


def read_video(path: Path, frame_indices: list[int]) -> tuple[dict[str, Any], dict[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames: dict[int, np.ndarray] = {}
    for index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Cannot decode frame {index} from {path}")
        frames[index] = frame
    cap.release()
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": frame_count / fps,
        "resolution": [width, height],
    }, frames


def contact_sheet(
    scene: dict[str, Any],
    clips: list[dict[str, Any]],
    decoded: dict[str, dict[int, np.ndarray]],
    output_path: Path,
) -> str:
    rows = []
    for clip in clips:
        cells = []
        for frame_index in scene["frame_indices"]:
            frame = cv2.resize(decoded[clip["type"]][frame_index], (256, 256))
            cv2.putText(
                frame,
                f"{frame_index / clip['fps']:.1f}s f{frame_index}",
                (5, 247),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cells.append(frame)
        strip = np.concatenate(cells, axis=1)
        label = np.zeros((36, strip.shape[1], 3), dtype=np.uint8)
        cv2.putText(label, clip["type"], (7, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(np.concatenate([label, strip], axis=0))
    sheet = np.concatenate(rows, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise RuntimeError(f"Cannot write {output_path}")
    return sha256_file(output_path)


def build_scene_evidence(scene: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    rows = load_scene_rows(scene)
    primary_files = {Path(str(name)).name for name in rows["file_name"]}
    metadata_aliases = []
    for alias in scene.get("split_aliases", []):
        alias_split, alias_index = alias.split(":", 1)
        alias_scene = {"split": alias_split, "scene_index": alias_index, "scene_key": alias}
        alias_rows = load_scene_rows(alias_scene)
        alias_files = {Path(str(name)).name for name in alias_rows["file_name"]}
        if alias_files != primary_files:
            raise RuntimeError(f"Split alias {alias} does not reference the same four videos")
        alias_path = metadata_path(alias_split)
        metadata_aliases.append(
            {
                "scene_id": alias,
                "path": str(alias_path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(alias_path),
                "same_video_inventory": True,
            }
        )
    scene_dir = OUTPUT_ROOT / "scenes" / scene["scene_key"].replace(":", "_")
    clips: list[dict[str, Any]] = []
    decoded: dict[str, dict[int, np.ndarray]] = {}
    for _, row in rows.iterrows():
        video_path = videos_root(scene["split"]) / Path(str(row["file_name"])).name
        properties, frames = read_video(video_path, scene["frame_indices"])
        clip = {
            "type": str(row["type"]),
            "gold_label": 1 if str(row["type"]).endswith("_Possible") else 0,
            "video_path": str(video_path.relative_to(PROJECT_ROOT)),
            "video_sha256": sha256_file(video_path),
            **properties,
            "sampled_frames": [
                {
                    "frame_index": index,
                    "time_seconds": index / properties["fps"],
                    "decoded_bgr_sha256": sha256_bytes(frames[index].tobytes()),
                }
                for index in scene["frame_indices"]
            ],
        }
        clips.append(clip)
        decoded[clip["type"]] = frames

    sheet_path = scene_dir / "contact_sheet.jpg"
    sheet_hash = contact_sheet(scene, clips, decoded, sheet_path)
    assessments = []
    for pair in ("1", "2"):
        summary = scene["pair_summaries"][pair]
        assessments.append(
            {
                "evidence_id": f"intphys2:{scene['scene_key']}:pair{pair}",
                "pair": int(pair),
                "possible_clip": f"{pair}_Possible",
                "impossible_clip": f"{pair}_Impossible",
                "focus_frames": summary["focus_frames"],
                "possible_observation": summary["possible"],
                "impossible_observation": summary["impossible"],
                "condition_consistency": "matched",
                "semantic_status": "provisional_review_pending",
            }
        )
    payload = {
        "format_version": 1,
        "benchmark": "intphys2",
        "scene_id": scene["scene_key"],
        "split_aliases": scene.get("split_aliases", []),
        "condition": scene["condition"],
        "game": str(rows.iloc[0]["game_name"]),
        "camera": str(rows.iloc[0]["Camera"]),
        "difficulty": str(rows.iloc[0]["Difficulty"]),
        "metadata_source": {
            "path": str(metadata_path(scene["split"]).relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(metadata_path(scene["split"])),
        },
        "metadata_aliases": metadata_aliases,
        "clips": clips,
        "pair_assessments": assessments,
        "contact_sheet": {
            "path": "contact_sheet.jpg",
            "sha256": sheet_hash,
            "columns": scene["frame_indices"],
            "rows": list(TYPE_ORDER),
        },
        "review": "pending",
    }
    return payload, scene_dir


def schema_id(condition: str, level: int, key: str, content: dict[str, Any]) -> str:
    identity = {name: content[name] for name in ("condition", "scene_scope", "kind", "trigger", "expectation", "violation_signatures")}
    digest = sha256_bytes(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8"))[:12]
    return f"intphys2:{condition}:L{level}:{key}:{digest}"


def build_schemas(scenes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    schemas: list[dict[str, Any]] = []
    by_condition: dict[str, dict[str, Any]] = {}
    for scene in scenes:
        evidence_ids = [f"intphys2:{scene['scene_key']}:pair1", f"intphys2:{scene['scene_key']}:pair2"]
        common = {
            "format_version": 1,
            "benchmark": "intphys2",
            "benchmark_version": "debug_plus_pinned_main_300",
            "condition": scene["condition"],
            "trigger": scene["trigger"],
            "expectation": scene["expectation"],
            "violation_signatures": scene["violations"],
            "constraints": ["异常判断需要正面视觉证据，不能仅由事件罕见或标签名称推出。"],
            "exceptions": ["存在可见的进入、离开、属性改变、转移路径或外部碰撞体时，需按该因果证据重新判断。"],
            "source_evidence": evidence_ids,
            "visual_evidence_ids": evidence_ids,
            "verification": {"metadata": "passed", "visual": "provisional", "review": "pending"},
        }
        condition_schema = {
            **common,
            "schema_id": "",
            "title": scene["invariant_title"],
            "scene_scope": [scene["scene_key"]],
            "abstraction_level": 1,
            "kind": "physical_invariant",
            "relations": {"parents": [], "members": []},
        }
        condition_schema["schema_id"] = schema_id(scene["condition"], 1, "invariant", condition_schema)
        scene_schema = {
            **common,
            "schema_id": "",
            "title": f"{scene['scene_key']} 的成对视觉判别规则",
            "scene_scope": [scene["scene_key"], *scene.get("split_aliases", [])],
            "abstraction_level": 3,
            "kind": "scene_discriminant",
            "relations": {"parents": [condition_schema["schema_id"]], "members": []},
        }
        scene_schema["schema_id"] = schema_id(scene["condition"], 3, scene["scene_key"].replace(":", "_"), scene_schema)
        condition_schema["relations"]["members"] = [scene_schema["schema_id"]]
        schemas.extend([condition_schema, scene_schema])
        by_condition[scene["condition"]] = {"condition": condition_schema, "scene": scene_schema}
    return schemas, by_condition


def build_spec() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "socialclaw.intphys2_gold_schema.v1",
        "title": "SocialLearningClaw IntPhys2 Gold Schema v1",
        "type": "object",
        "required": ["schema_id", "format_version", "title", "benchmark", "benchmark_version", "condition", "scene_scope", "abstraction_level", "kind", "trigger", "expectation", "violation_signatures", "constraints", "exceptions", "relations", "source_evidence", "visual_evidence_ids", "verification"],
        "properties": {
            "format_version": {"const": 1},
            "benchmark": {"const": "intphys2"},
            "condition": {"enum": ["permanence", "immutability", "continuity", "solidity"]},
            "abstraction_level": {"enum": [1, 3]},
            "kind": {"enum": ["physical_invariant", "scene_discriminant"]},
            "source_evidence": {"type": "array", "minItems": 2},
            "visual_evidence_ids": {"type": "array", "minItems": 2},
        },
    }


def validate(scenes: list[dict[str, Any]], evidence: list[dict[str, Any]], schemas: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    evidence_ids = {item["evidence_id"] for payload in evidence for item in payload["pair_assessments"]}
    schema_ids = {item["schema_id"] for item in schemas}
    if {scene["condition"] for scene in scenes} != {"permanence", "immutability", "continuity", "solidity"}:
        errors.append("Pilot does not cover all four conditions")
    for payload in evidence:
        if {clip["type"] for clip in payload["clips"]} != set(TYPE_ORDER):
            errors.append(f"Incomplete type group: {payload['scene_id']}")
        if sorted(clip["gold_label"] for clip in payload["clips"]) != [0, 0, 1, 1]:
            errors.append(f"Unbalanced labels: {payload['scene_id']}")
        for clip in payload["clips"]:
            path = PROJECT_ROOT / clip["video_path"]
            if sha256_file(path) != clip["video_sha256"]:
                errors.append(f"Stale video hash: {path}")
            if any(frame["frame_index"] >= clip["frame_count"] for frame in clip["sampled_frames"]):
                errors.append(f"Out-of-range frame: {path}")
    for schema in schemas:
        if not set(schema["source_evidence"]) <= evidence_ids:
            errors.append(f"Unknown evidence in {schema['schema_id']}")
        if not set(schema["relations"].get("parents", [])) <= schema_ids:
            errors.append(f"Unknown parent in {schema['schema_id']}")
        if not set(schema["relations"].get("members", [])) <= schema_ids:
            errors.append(f"Unknown member in {schema['schema_id']}")
    return {
        "status": "passed" if not errors else "failed",
        "scene_count": len(evidence),
        "unique_clip_count": len({clip["video_sha256"] for payload in evidence for clip in payload["clips"]}),
        "schema_count": len(schemas),
        "condition_counts": {condition: sum(item["condition"] == condition for item in schemas) for condition in ("permanence", "immutability", "continuity", "solidity")},
        "pair_assessment_count": sum(len(payload["pair_assessments"]) for payload in evidence),
        "metadata_and_hash_checks": "passed" if not errors else "failed",
        "visual_semantics": "provisional_review_pending",
        "errors": errors,
    }


def write_scene_review(scene: dict[str, Any], evidence: dict[str, Any], scene_dir: Path) -> None:
    lines = [
        f"# {scene['scene_key']} / {scene['condition']} — 人工审核稿",
        "",
        "> 状态：视频与 metadata 自动验证通过；事件语义为 provisional，等待人工审核。",
        "",
        f"- Scene family：`{evidence['game']}`",
        f"- Camera / difficulty：`{evidence['camera']}` / `{evidence['difficulty']}`",
        f"- 物理不变量：{scene['invariant_title']}",
        "",
        "## 对照帧",
        "",
        "行顺序固定为 `1_Possible / 1_Impossible / 2_Possible / 2_Impossible`；每格标注秒数和源帧编号。",
        "",
        "![四视频对照帧](contact_sheet.jpg)",
        "",
        "## Pair 判读",
        "",
        "| Pair | Possible | Impossible |",
        "|---:|---|---|",
    ]
    for pair in ("1", "2"):
        summary = scene["pair_summaries"][pair]
        lines.append(f"| {pair} | {summary['possible']} | {summary['impossible']} |")
    lines += [
        "",
        "## Schema 边界",
        "",
        f"- Trigger：{scene['trigger']}",
        f"- Expectation：{scene['expectation']}",
        f"- Violation A：{scene['violations'][0]}",
        f"- Violation B：{scene['violations'][1]}",
        "",
    ]
    if scene.get("review_note"):
        lines += ["## 特别说明", "", scene["review_note"], ""]
    lines += [
        "## 请审核",
        "",
        "1. 对象、颜色、容器位置或碰撞事件的描述是否与视频一致；",
        "2. Possible 是否提供了不变量成立的正对照，而非仅仅没有异常；",
        "3. 两个 Impossible 是否确实属于同一 condition 的互补违规；",
        "4. 当前规则是否混入了具体标签而没有视觉证据。",
        "",
    ]
    (scene_dir / "review.md").write_text("\n".join(lines), encoding="utf-8")


def write_readme(entries: list[dict[str, Any]], validation: dict[str, Any]) -> None:
    lines = [
        "# IntPhys2 Gold Schema v1 — 四类物理规则 pilot",
        "",
        "> 当前为四个场景的审核稿，不代表 89 个唯一场景已经完成；所有视觉语义均为 provisional。",
        "",
        "本批每种 condition 选择一个四视频场景。Metadata、视频 hash、帧范围和 possible/impossible",
        "配对由程序验证；具体物体事件由成对帧判读，并保留人工审核门。标签只用于生成后的",
        "一致性检查，不作为 Schema 的推导依据。",
        "",
        "## 清单",
        "",
        "| Condition | Scene | Family | 审核 |",
        "|---|---|---|---|",
    ]
    for entry in entries:
        lines.append(f"| `{entry['condition']}` | `{entry['scene_id']}` | `{entry['game']}` | [review.md]({entry['directory']}/review.md) |")
    lines += [
        "",
        "## 汇总",
        "",
        f"- 唯一场景：{validation['scene_count']} / 89",
        f"- 唯一视频：{validation['unique_clip_count']}",
        f"- Pair assessments：{validation['pair_assessment_count']}",
        f"- Schema：{validation['schema_count']}（4 条 Level 1 condition invariant + 4 条 Level 3 scene discriminant）",
        "- 自动验证：passed；视觉语义：provisional_review_pending。",
        "",
        "## 复现",
        "",
        "```bash",
        ".venv/bin/python scripts/generate_intphys2_gold.py",
        "```",
        "",
    ]
    (OUTPUT_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    evidence_payloads: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    for scene in SCENES:
        evidence, scene_dir = build_scene_evidence(scene)
        write_json(scene_dir / "scene_evidence.json", evidence)
        write_scene_review(scene, evidence, scene_dir)
        evidence_payloads.append(evidence)
        entries.append({
            "condition": scene["condition"],
            "scene_id": scene["scene_key"],
            "split_aliases": scene.get("split_aliases", []),
            "game": evidence["game"],
            "directory": str(scene_dir.relative_to(OUTPUT_ROOT)),
            "status": "provisional_review_pending",
        })

    schemas, _ = build_schemas(SCENES)
    validation = validate(SCENES, evidence_payloads, schemas)
    if validation["status"] != "passed":
        raise SystemExit(json.dumps(validation, ensure_ascii=False, indent=2))
    write_json(OUTPUT_ROOT / "schema_spec.json", build_spec())
    write_json(OUTPUT_ROOT / "schemas.json", {"format_version": 1, "schemas": schemas})
    write_json(OUTPUT_ROOT / "validation.json", validation)
    manifest = {
        "format_version": 1,
        "created_date": date.today().isoformat(),
        "benchmark": "intphys2",
        "scope": "four_condition_pilot",
        "status": "pilot_review_pending",
        "scenes": entries,
        "totals": validation,
    }
    write_json(OUTPUT_ROOT / "manifest.json", manifest)
    write_readme(entries, validation)
    print(json.dumps(validation, ensure_ascii=False))


if __name__ == "__main__":
    main()
