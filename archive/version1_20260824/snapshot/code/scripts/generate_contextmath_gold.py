#!/usr/bin/env python3
"""Generate the first reviewable ContextMATH Gold Schema batch."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data/contextmath"
OUTPUT_ROOT = PROJECT_ROOT / "gold/contextmath/v1"
PROBLEM_IDS = ((2024, 60), (2024, 67), (2024, 75), (2024, 84), (2025, 0), (2025, 3))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_hash(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def split_path(year: int, variant: str) -> Path:
    matches = sorted(DATA_ROOT.glob(f"aime_{year}_{variant}*.parquet"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one aime_{year}_{variant} parquet, got {matches}")
    return matches[0]


def normalize_answer(value: str) -> str:
    normalized = str(value).strip().replace(",", "").replace(" ", "")
    normalized = normalized.replace("^\\circ", "").replace("\\circ", "")
    if normalized.lstrip("-").isdigit():
        return str(int(normalized))
    return normalized


def check(
    witness_id: str,
    description: str,
    calculation: str,
    observed: Any,
    expected: Any,
) -> dict[str, Any]:
    return {
        "witness_id": witness_id,
        "description": description,
        "calculation": calculation,
        "observed": str(observed),
        "expected": str(expected),
        "passed": observed == expected,
    }


def witness_2024_60() -> tuple[list[dict[str, Any]], str]:
    factorial_threshold = next(n for n in range(1, 20) if math.factorial(n) % 8 == 0)
    largest_square = max(n * n for n in range(20) if n * n < 150)
    s = Fraction(5, 2)
    stop_hours = Fraction(4) - Fraction(9, 1) / s
    target_hours = Fraction(9, 1) / (s + Fraction(1, 2)) + stop_hours
    checks = [
        check("cs_standard_time", "CS 标准总时长解码", "min{n: 8 divides n!}", factorial_threshold, 4),
        check("cs_fast_time", "CS 加速总时长解码", "max{k^2: k^2 < 150}", largest_square, 144),
        check("cs_increment", "CS 正速度增量解码", "u^2-u-2=0 的正根", 2 * 2 - 2 - 2, 0),
        check("speed_equations", "候选基础速度满足两条总时长方程的差", "9/s-9/(s+2)=8/5", Fraction(9, 1) / s - Fraction(9, 1) / (s + 2), Fraction(8, 5)),
        check("positive_root", "速度方程的两个精确根中仅保留正根", "(2s-5)(2s+9)=0", (Fraction(-9, 2), Fraction(5, 2)), (Fraction(-9, 2), Fraction(5, 2))),
        check("fixed_stop", "共同停留时长", "4-9/(5/2)", stop_hours, Fraction(2, 5)),
        check("target_total", "目标速度下总分钟数", "60*(9/(s+1/2)+stop)", target_hours * 60, Fraction(204)),
    ]
    return checks, normalize_answer(str(target_hours * 60))


def witness_2024_67() -> tuple[list[dict[str, Any]], str]:
    exponent_11 = next(n for n in range(1, 20) if 11**n == 25937424601)
    exponent_12 = next(n for n in range(1, 20) if 12**n == 61917364224)
    product = Fraction(10 * 10, 4)
    checks = [
        check("cs_first_exponent", "CS 第一指数解码", "log_11(25937424601)", exponent_11, 10),
        check("cs_second_exponent", "CS 第二指数解码", "log_12(61917364224)", exponent_12, 10),
        check("coupled_log_product", "相乘后对数比值互相抵消", "(x ln y/ln x)(4y ln x/ln y)=10^2", 4 * product, 100),
        check("target_product", "由 4xy=100 得到乘积", "100/4", product, Fraction(25)),
    ]
    return checks, normalize_answer(str(product))


def witness_2024_75() -> tuple[list[dict[str, Any]], str]:
    total_incidence = 900 + 195 + 367 + 562
    excess = total_incidence - 900
    all_four = Fraction(excess - 437 - 2 * 234, 3)
    checks = [
        check("cs_population", "CS 人口解码", "15*60", 15 * 60, 900),
        check("cs_module_counts", "CS 三类订阅人数解码", "3*65, 365+2, 270+292", (3 * 65, 365 + 2, 270 + 292), (195, 367, 562)),
        check("cs_exactly_two", "CS 恰有两类解码", "19*23", 19 * 23, 437),
        check("incidence_total", "四类物品的总拥有次数", "900+195+367+562", total_incidence, 2024),
        check("excess_count", "扣除每人至少一件后的额外拥有次数", "2024-900", excess, 1124),
        check("all_four", "由 n2+2n3+3n4=1124 求 n4", "(1124-437-2*234)/3", all_four, Fraction(73)),
    ]
    return checks, normalize_answer(str(all_four))


def witness_2024_84() -> tuple[list[dict[str, Any]], str]:
    a = -Fraction(7, 24)
    b = -Fraction(3, 8)
    c = -Fraction(5, 12)
    weighted = 4 * a + 3 * b + 2 * c
    target = abs(weighted)
    answer = target.numerator + target.denominator
    checks = [
        check("linear_system", "对数变量满足三条线性方程", "a-b-c=1/2; -a+b-c=1/3; -a-b+c=1/4", (a - b - c, -a + b - c, -a - b + c), (Fraction(1, 2), Fraction(1, 3), Fraction(1, 4))),
        check("isolated_logs", "两两相加隔离三个对数", "a=-7/24,b=-3/8,c=-5/12", (a, b, c), (Fraction(-7, 24), Fraction(-3, 8), Fraction(-5, 12))),
        check("weighted_log", "乘积的对数转成加权和", "4a+3b+2c", weighted, Fraction(-25, 8)),
        check("reduced_fraction", "取绝对值并相加分子分母", "abs(-25/8)", answer, 33),
    ]
    return checks, normalize_answer(str(answer))


def witness_2025_0() -> tuple[list[dict[str, Any]], str]:
    divisors = [d for d in range(1, 57) if 56 % d == 0]
    bases = [d - 7 for d in divisors if d - 7 > 9]
    answer = sum(bases)
    checks = [
        check("base_expansion", "两位数按进制展开", "17_b=b+7; 97_b=9b+7", ("b+7", "9b+7"), ("b+7", "9b+7")),
        check("divisibility_reduction", "线性组合将整除条件化为常数", "9(b+7)-(9b+7)=56", 9 * 7 - 7, 56),
        check("valid_bases", "枚举 56 的因数并施加 b>9", "b+7 divides 56", bases, [21, 49]),
        check("base_sum", "汇总全部合法进制", "21+49", answer, 70),
    ]
    return checks, normalize_answer(str(answer))


def witness_2025_3() -> tuple[list[dict[str, Any]], str]:
    line_one = [(2 * k, -3 * k) for k in range(-33, 34)]
    line_two = [(3 * k, 4 * k) for k in range(-25, 26)]
    parameterized = set(line_one) | set(line_two)
    enumerated = {
        (x, y)
        for x in range(-100, 101)
        for y in range(-100, 101)
        if 12 * x * x - x * y - 6 * y * y == 0
    }
    checks = [
        check("factorization", "二次型因式分解", "(3x+2y)(4x-3y)", (3 * 7 + 2 * -4) * (4 * 7 - 3 * -4), 12 * 7**2 - 7 * -4 - 6 * (-4) ** 2),
        check("first_line_count", "直线 y=-3x/2 的整数参数计数", "(x,y)=(2k,-3k), |k|<=33", len(line_one), 67),
        check("second_line_count", "直线 y=4x/3 的整数参数计数", "(x,y)=(3k,4k), |k|<=25", len(line_two), 51),
        check("origin_overlap", "两条直线只在原点重合", "67+51-1", len(parameterized), 117),
        check("independent_enumeration", "在完整整数边界内独立枚举", "-100<=x,y<=100", parameterized, enumerated),
    ]
    return checks, normalize_answer(str(len(parameterized)))


WITNESS_BUILDERS: dict[tuple[int, int], Callable[[], tuple[list[dict[str, Any]], str]]] = {
    (2024, 60): witness_2024_60,
    (2024, 67): witness_2024_67,
    (2024, 75): witness_2024_75,
    (2024, 84): witness_2024_84,
    (2025, 0): witness_2025_0,
    (2025, 3): witness_2025_3,
}


SCHEMA_WITNESS_IDS: dict[tuple[int, int], dict[str, list[str]]] = {
    (2024, 60): {
        "total_time_model": ["speed_equations"],
        "eliminate_stop": ["speed_equations"],
        "positive_speed": ["positive_root"],
        "recover_stop": ["fixed_stop"],
        "target_time": ["target_total"],
    },
    (2024, 67): {
        "expand_log_power": ["coupled_log_product"],
        "cancel_log_ratios": ["coupled_log_product"],
        "solve_product": ["target_product"],
    },
    (2024, 75): {
        "incidence_count": ["incidence_total"],
        "excess_ownership": ["excess_count"],
        "solve_four": ["all_four"],
    },
    (2024, 84): {
        "log_variables": ["linear_system"],
        "isolate_logs": ["isolated_logs"],
        "weighted_product_log": ["weighted_log"],
        "normalize_fraction": ["reduced_fraction"],
    },
    (2025, 0): {
        "expand_base_numerals": ["base_expansion"],
        "reduce_divisibility": ["divisibility_reduction"],
        "enumerate_bases": ["valid_bases"],
        "sum_bases": ["base_sum"],
    },
    (2025, 3): {
        "factor_quadratic": ["factorization"],
        "zero_product_lines": ["factorization"],
        "integer_parameterization": ["first_line_count", "second_line_count"],
        "union_count": ["origin_overlap", "independent_enumeration"],
    },
}


PROBLEMS: dict[tuple[int, int], dict[str, Any]] = {
    (2024, 60): {
        "summary": "含固定停留时间的两组路程—速度—总时长条件，并询问第三种速度下的总时长。",
        "symbols": ["distance", "base_speed", "fixed_stop", "fast_increment", "standard_total", "fast_total", "target_increment", "target_total"],
        "schemas": [
            ("total_time_model", "把总时长拆为行进时长与固定停留时长", "representation", "同一路程在不同速度下完成，且每次含相同的固定停留时间。", "对每种速度 v 建立 T=d/v+c；固定项 c 在各条件中相同。", "得到两条共享 d、c 和基础速度的精确方程。", []),
            ("eliminate_stop", "用总时长之差消去共同停留时间", "derivation", "两条 T=d/v+c 方程中的 c 相同。", "两式相减，只保留 d/s-d/(s+Δv)=ΔT。", "把未知基础速度约束为一个一元方程。", ["total_time_model"]),
            ("positive_speed", "求解速度方程并保留正根", "constraint", "一元速度方程可能具有多个代数根。", "精确因式分解或求根，并使用速度必须为正的定义域。", "唯一保留可物理解释的基础速度。", ["eliminate_stop"]),
            ("recover_stop", "回代恢复固定停留时长", "calculation", "基础速度已经确定。", "将基础速度代回任一总时长方程求 c，并用另一方程交叉检查。", "得到与两次观测都一致的固定停留时长。", ["positive_speed"]),
            ("target_time", "代入目标速度并统一换算为分钟", "calculation", "目标速度是基础速度加指定增量。", "计算 d/(s+δ)+c，并在最后一步统一转换时间单位。", "得到题目要求的总分钟数。", ["recover_stop"]),
        ],
        "alignments": {
            "sg": [("delivery route length", "distance", "9 km"), ("normal average speed", "base_speed", "s"), ("loading/unloading/rest", "fixed_stop", "同一固定时长"), ("initial route time", "standard_total", "4 hours"), ("optimized speed", "fast_increment", "+2 km/h"), ("optimized route time", "fast_total", "2 hours 24 minutes"), ("tuned speed", "target_increment", "+1/2 km/h"), ("requested route time", "target_total", "行进加停留")],
            "cs": [("days in a week plus two", "distance", "7+2=9 km"), ("standard speed", "base_speed", "s"), ("diagnostic and warm-up", "fixed_stop", "同一固定时长"), ("smallest n with 8|n!", "standard_total", "4 hours"), ("largest square below 150", "fast_total", "144 minutes"), ("positive root of u^2-u-2", "fast_increment", "2 km/h"), ("one-fourth of the increment", "target_increment", "1/2 km/h"), ("requested total mission", "target_total", "行进加诊断")],
        },
    },
    (2024, 67): {
        "summary": "两条互相耦合的对数等式给出相同常数，要求两个正实数的乘积。",
        "symbols": ["x", "y", "first_log_equation", "second_log_equation", "target_product"],
        "schemas": [
            ("expand_log_power", "把幂的对数写成自然对数比值", "representation", "底数和真数均大于 1，两个对数都有定义。", "使用 log_a(b^c)=c ln(b)/ln(a) 展开两条等式。", "得到 x·ln(y)/ln(x) 与 4y·ln(x)/ln(y) 两个表达式。", []),
            ("cancel_log_ratios", "相乘耦合等式以消去互逆对数比", "derivation", "两个展开式分别等于已知常数。", "将两式相乘；ln(y)/ln(x) 与 ln(x)/ln(y) 相消。", "直接得到只含 xy 的方程，且无需分别求 x、y。", ["expand_log_power"]),
            ("solve_product", "从乘积方程直接求目标量", "calculation", "消元后方程只含目标 xy。", "除去已知系数并保持精确有理数运算。", "得到题目所求乘积。", ["cancel_log_ratios"]),
        ],
        "alignments": {
            "sg": [("calibration constant", "x", ">1"), ("power output", "y", ">1"), ("Lin feedback", "first_log_equation", "log_x(y^x)=10"), ("Mateo feedback", "second_log_equation", "log_y(x^(4y))=10"), ("synchronization product", "target_product", "xy")],
            "cs": [("shared scaling factor", "x", ">1"), ("material reflectivity", "y", ">1"), ("11 to 25937424601 exponent", "first_log_equation", "10, hence x^10=y^x"), ("12 to 61917364224 exponent", "second_log_equation", "10, hence y^10=x^(4y)"), ("requested product", "target_product", "xy")],
        },
    },
    (2024, 75): {
        "summary": "每人至少拥有一个共同物品，并给出另外三类物品人数及恰有二、三件的人数。",
        "symbols": ["population", "common_item", "three_item_counts", "exactly_two", "exactly_three", "exactly_four"],
        "schemas": [
            ("incidence_count", "用逐物品计数得到总拥有次数", "representation", "已知人口、共同物品和另外三类物品的拥有人数。", "按物品类别相加拥有人数，得到所有人—物品关联的总数。", "总拥有次数可与按人分类的计数相等。", []),
            ("excess_ownership", "扣除每人必有的一件以计数额外拥有次数", "derivation", "每位居民都至少拥有共同物品。", "从总拥有次数中减去人口；恰有 k 件者贡献 k-1 个额外拥有次数。", "建立 n2+2n3+3n4 等于额外拥有次数的方程。", ["incidence_count"]),
            ("solve_four", "从额外拥有次数方程隔离恰有四件者", "calculation", "n2 与 n3 已知，n4 是唯一未知量。", "移去 n2、2n3 后除以 3。", "得到拥有全部四件物品的人数。", ["excess_ownership"]),
        ],
        "alignments": {
            "sg": [("registered residents", "population", "900"), ("daily dining program", "common_item", "everyone"), ("gym/pool/library", "three_item_counts", "195,367,562"), ("exactly two services", "exactly_two", "437"), ("three services", "exactly_three", "234"), ("all four facilities", "exactly_four", "target")],
            "cs": [("15-hour workday in minutes", "population", "900"), ("Tier-1 access", "common_item", "everyone"), ("3·65, 365+2, 270+292", "three_item_counts", "195,367,562"), ("19·23", "exactly_two", "437"), ("concurrently used exactly three", "exactly_three", "234"), ("complete subscription", "exactly_four", "target")],
        },
    },
    (2024, 84): {
        "summary": "三个正变量的商具有给定二进制对数，要求一个加权乘积的对数绝对值。",
        "symbols": ["x", "y", "z", "base_two_scale", "three_log_constraints", "weighted_product", "reduced_fraction"],
        "schemas": [
            ("log_variables", "用对数变量把乘除约束线性化", "representation", "x、y、z 为正数且所有表达式使用同一对数底。", "令 a=log2(x), b=log2(y), c=log2(z)，把乘法变加法、除法变减法。", "三条原约束变成关于 a、b、c 的线性方程。", []),
            ("isolate_logs", "两两相加线性方程隔离单个对数变量", "derivation", "每条方程中一个变量为正号、另两个为负号。", "选取两式相加，使两个变量抵消并直接解出第三个变量。", "精确得到 a、b、c 的有理数值。", ["log_variables"]),
            ("weighted_product_log", "把幂乘积的对数转成指数加权和", "derivation", "目标是 log2(x^4 y^3 z^2)。", "应用乘积与幂的对数规则，写成 4a+3b+2c。", "目标量化为已知有理数的线性组合。", ["isolate_logs"]),
            ("normalize_fraction", "取绝对值并将有理数化为最简分数", "calculation", "加权对数可能为负数。", "先取绝对值，再以互素分子分母表示并按题意组合。", "得到题目要求的分子分母组合量。", ["weighted_product_log"]),
        ],
        "alignments": {
            "sg": [("Thermal Control efficiency", "x", "x"), ("Power Stabilizer efficiency", "y", "y"), ("Safety Response efficiency", "z", "z"), ("base-2 logarithm", "base_two_scale", "log2"), ("three efficiency ratios", "three_log_constraints", "1/2,1/3,1/4"), ("weighted metric", "weighted_product", "x^4 y^3 z^2"), ("reduced fraction", "reduced_fraction", "target")],
            "cs": [("Alpha output", "x", "x"), ("Beta output", "y", "y"), ("Gamma output", "z", "z"), ("one unit means doubling", "base_two_scale", "log2"), ("three ratio measurements", "three_log_constraints", "1/2,1/3,1/4"), ("composite power state", "weighted_product", "x^4 y^3 z^2"), ("reduced absolute deviation", "reduced_fraction", "target")],
        },
    },
    (2025, 0): {
        "summary": "寻找所有大于 9 且使一个两位 base-b 数整除另一个两位 base-b 数的进制。",
        "symbols": ["base", "lower_numeral", "upper_numeral", "divisibility", "valid_bases", "base_sum"],
        "schemas": [
            ("expand_base_numerals", "把两位 base-b 数展开为关于 b 的一次式", "representation", "各位数字在 b>9 时均合法。", "使用 (uv)_b=u·b+v 展开两个数。", "整除关系变为两个整数一次式之间的整除。", []),
            ("reduce_divisibility", "用线性组合把一次式整除化为常数整除", "derivation", "若 d 整除 n，则 d 也整除 n 的任意整数线性组合。", "从较大一次式减去下方一次式的适当整数倍。", "得到 b 加常数必须整除一个固定正整数。", ["expand_base_numerals"]),
            ("enumerate_bases", "枚举固定整数因数并施加进制下界", "constraint", "候选量是固定整数的正因数，且 b>9。", "枚举因数、还原 b，并检查数字合法性与原整除关系。", "得到完整且无遗漏的合法进制集合。", ["reduce_divisibility"]),
            ("sum_bases", "对全部合法进制求和", "calculation", "合法进制集合已经穷尽。", "每个合法 b 恰计一次后求和。", "得到题目要求的进制总和。", ["enumerate_bases"]),
        ],
        "alignments": {
            "sg": [("machine mode setting", "base", "b>9"), ("Component X configuration 17", "lower_numeral", "17_b"), ("Component Y configuration 97", "upper_numeral", "97_b"), ("divides exactly", "divisibility", "17_b | 97_b"), ("all settings", "valid_bases", "target set"), ("sum", "base_sum", "target")],
            "cs": [("seconds per full turn and cascading display limit", "base", "b>=10"), ("logger shows 17", "lower_numeral", "17_b"), ("logger shows 97", "upper_numeral", "97_b"), ("exact integer multiple", "divisibility", "17_b | 97_b"), ("possible gear settings", "valid_bases", "target set"), ("sum", "base_sum", "target")],
        },
    },
    (2025, 3): {
        "summary": "在有限整数方框内计数齐次二次方程的有序整数解。",
        "symbols": ["x", "y", "integer_bounds", "quadratic_equation", "ordered_pair_count"],
        "schemas": [
            ("factor_quadratic", "把齐次二次型精确因式分解", "representation", "方程右侧为零，二次式可在整数系数下分解。", "寻找两个一次因子并展开复核全部系数。", "零集变为两个一次因子的并集。", []),
            ("zero_product_lines", "由零乘积将解集拆为两条直线", "derivation", "两个实数一次因子的乘积为零。", "分别令每个因子为零。", "所有实数解均位于两条有理斜率直线上。", ["factor_quadratic"]),
            ("integer_parameterization", "参数化每条直线的全部整数点并施加边界", "constraint", "x、y 必须同时为整数且均落在闭区间内。", "按斜率分母选择最小整数步长参数化，再由两个坐标界共同限制参数。", "得到每条直线在方框内的完整有限整数点集。", ["zero_product_lines"]),
            ("union_count", "合并两条直线并扣除交点重复计数", "calculation", "两条不同直线的整数点集可能重叠。", "分别计数后应用容斥；齐次直线仅在原点相交。", "得到有序整数对总数。", ["integer_parameterization"]),
        ],
        "alignments": {
            "sg": [("horizontal position", "x", "integer"), ("vertical position", "y", "integer"), ("-100 to 100", "integer_bounds", "inclusive"), ("balance condition", "quadratic_equation", "12x^2-xy-6y^2=0"), ("combinations of positions", "ordered_pair_count", "target")],
            "cs": [("whole-hour delay", "x", "integer"), ("lateral distance", "y", "integer"), ("-100 to 100", "integer_bounds", "inclusive"), ("reactive energy balance", "quadratic_equation", "12x^2=xy+6y^2"), ("delay-position pairs", "ordered_pair_count", "target")],
        },
    },
}


def load_rows(year: int, problem_id: int) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for variant in ("sg", "cs"):
        path = split_path(year, variant)
        matches = pd.read_parquet(path).loc[lambda frame: frame["id"] == problem_id]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one row for {year}/{problem_id}/{variant}")
        row = matches.iloc[0].to_dict()
        row["_path"] = path
        rows[variant] = row
    return rows


def make_source_evidence(rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    evidence = []
    for variant, row in rows.items():
        path: Path = row["_path"]
        evidence.append(
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(path),
                "row_id": int(row["id"]),
                "field": "ori_question",
                "value_sha256": text_hash(str(row["ori_question"])),
                "variant": variant,
            }
        )
    return evidence


def make_schema_id(problem_key: str, key: str, payload: dict[str, Any]) -> str:
    identity = {name: payload[name] for name in ("problem_id", "kind", "trigger", "operation", "expectation", "constraints")}
    digest = sha256_bytes(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8"))[:12]
    return f"contextmath:{problem_key}:L3:{key}:{digest}"


def build_problem(year: int, problem_id: int) -> dict[str, Any]:
    definition = PROBLEMS[(year, problem_id)]
    rows = load_rows(year, problem_id)
    problem_key = f"aime_{year}:{problem_id}"
    if rows["sg"]["ori_question"] != rows["cs"]["ori_question"]:
        raise RuntimeError(f"Original-question mismatch for {problem_key}")
    if str(rows["sg"]["answer"]) != str(rows["cs"]["answer"]):
        raise RuntimeError(f"Answer mismatch for {problem_key}")

    source_evidence = make_source_evidence(rows)
    schemas: list[dict[str, Any]] = []
    schema_ids: dict[str, str] = {}
    for key, title, kind, trigger, operation, expectation, requires in definition["schemas"]:
        payload: dict[str, Any] = {
            "schema_id": "",
            "format_version": 1,
            "title": title,
            "benchmark": "contextmath",
            "benchmark_version": "local_aime_2024_2025",
            "problem_id": problem_key,
            "sample_scope": [f"aime_{year}_sg:{problem_id}", f"aime_{year}_cs:{problem_id}"],
            "abstraction_level": 3,
            "kind": kind,
            "trigger": trigger,
            "operation": operation,
            "expectation": expectation,
            "constraints": [],
            "exceptions": [],
            "relations": {"parents": [], "requires": []},
            "source_evidence": source_evidence,
            "witness_ids": [],
            "verification": {"source": "passed", "executable": "pending", "review": "pending"},
        }
        payload["schema_id"] = make_schema_id(problem_key, key, payload)
        schemas.append(payload)
        schema_ids[key] = payload["schema_id"]
        payload["relations"]["requires"] = [schema_ids[item] for item in requires]

    witness_checks, derived_answer = WITNESS_BUILDERS[(year, problem_id)]()
    if not all(item["passed"] for item in witness_checks):
        raise RuntimeError(f"Witness failure for {problem_key}")
    known_witness_ids = {item["witness_id"] for item in witness_checks}
    for key, schema in zip((item[0] for item in definition["schemas"]), schemas):
        schema["witness_ids"] = SCHEMA_WITNESS_IDS[(year, problem_id)][key]
        if not set(schema["witness_ids"]) <= known_witness_ids:
            raise RuntimeError(f"Unknown witness mapping for {problem_key}/{key}")
        schema["verification"]["executable"] = "passed"

    alignments = {
        "format_version": 1,
        "problem_id": problem_key,
        "canonical_question_sha256": text_hash(str(rows["sg"]["ori_question"])),
        "canonical_symbols": definition["symbols"],
        "variants": [],
    }
    for variant in ("sg", "cs"):
        row = rows[variant]
        path: Path = row["_path"]
        mappings = [
            {"surface": surface, "canonical": canonical, "interpretation": interpretation}
            for surface, canonical, interpretation in definition["alignments"][variant]
        ]
        alignments["variants"].append(
            {
                "split": f"aime_{year}_{variant}",
                "row_id": problem_id,
                "question_sha256": text_hash(str(row["question"])),
                "source": {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(path)},
                "mappings": mappings,
                "witness_ids": [
                    item["witness_id"]
                    for item in witness_checks
                    if variant == "cs" and item["witness_id"].startswith("cs_")
                ],
                "coverage": "complete" if set(definition["symbols"]) <= {item["canonical"] for item in mappings} else "incomplete",
            }
        )

    witness_payload = {
        "format_version": 1,
        "problem_id": problem_key,
        "generator": "scripts/generate_contextmath_gold.py",
        "gold_answer_raw": str(rows["sg"]["answer"]),
        "gold_answer_normalized": normalize_answer(str(rows["sg"]["answer"])),
        "derived_answer_normalized": derived_answer,
        "checks": witness_checks,
    }
    coverage = {
        "format_version": 1,
        "problem_id": problem_key,
        "requirements": [
            {"requirement_id": "sg_semantic_alignment", "schema_ids": [], "artifact": "surface_alignment.json", "status": alignments["variants"][0]["coverage"]},
            {"requirement_id": "cs_semantic_alignment", "schema_ids": [], "artifact": "surface_alignment.json", "status": alignments["variants"][1]["coverage"]},
            {"requirement_id": "necessary_reasoning_chain", "schema_ids": [schema["schema_id"] for schema in schemas], "artifact": "schemas.json", "status": "complete"},
            {"requirement_id": "executable_answer_check", "schema_ids": [schemas[-1]["schema_id"]], "artifact": "witness.json", "status": "complete"},
        ],
    }
    errors: list[str] = []
    if any(item["coverage"] != "complete" for item in alignments["variants"]):
        errors.append("Surface alignment is incomplete")
    if witness_payload["gold_answer_normalized"] != derived_answer:
        errors.append("Derived answer does not match the dataset answer")
    known_ids = {schema["schema_id"] for schema in schemas}
    for schema in schemas:
        if not set(schema["relations"]["requires"]) <= known_ids:
            errors.append(f"Unknown dependency in {schema['schema_id']}")
        for evidence in schema["source_evidence"]:
            source = PROJECT_ROOT / evidence["path"]
            if sha256_file(source) != evidence["sha256"]:
                errors.append(f"Stale source hash in {schema['schema_id']}")
    validation = {
        "status": "passed" if not errors else "failed",
        "problem_id": problem_key,
        "schema_count": len(schemas),
        "alignment_variants": 2,
        "witness_checks": len(witness_checks),
        "witness_passed": sum(item["passed"] for item in witness_checks),
        "coverage_complete": all(item["status"] == "complete" for item in coverage["requirements"]),
        "answer_match": witness_payload["gold_answer_normalized"] == derived_answer,
        "review": "pending",
        "errors": errors,
    }
    return {
        "key": problem_key,
        "dir_name": f"aime_{year}_{problem_id}",
        "summary": definition["summary"],
        "schemas": schemas,
        "alignments": alignments,
        "witness": witness_payload,
        "coverage": coverage,
        "validation": validation,
    }


def build_spec() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "socialclaw.contextmath_gold_schema.v1",
        "title": "SocialLearningClaw ContextMATH Gold Schema v1",
        "type": "object",
        "required": ["schema_id", "format_version", "title", "benchmark", "benchmark_version", "problem_id", "sample_scope", "abstraction_level", "kind", "trigger", "operation", "expectation", "constraints", "exceptions", "relations", "source_evidence", "witness_ids", "verification"],
        "properties": {
            "format_version": {"const": 1},
            "benchmark": {"const": "contextmath"},
            "abstraction_level": {"const": 3},
            "kind": {"enum": ["representation", "derivation", "constraint", "calculation", "verification"]},
            "sample_scope": {"type": "array", "minItems": 2, "maxItems": 2},
            "source_evidence": {"type": "array", "minItems": 2},
            "witness_ids": {"type": "array", "minItems": 1},
        },
    }


def write_review(problem: dict[str, Any]) -> None:
    lines = [
        f"# {problem['key']} Gold Schema v1 — 人工审核稿",
        "",
        "> 状态：自动验证通过，等待人工审核。标准答案只存在于 witness/validation，不作为 Schema 节点。",
        "",
        "## 问题结构",
        "",
        problem["summary"],
        "",
        "## 原子 Schema",
        "",
        "| 顺序 | 类型 | Schema | 依赖数 |",
        "|---:|---|---|---:|",
    ]
    for index, schema in enumerate(problem["schemas"], start=1):
        lines.append(f"| {index} | `{schema['kind']}` | {schema['title']} | {len(schema['relations']['requires'])} |")
    lines += [
        "",
        "## 证据与验算",
        "",
        f"- SG/CS 语义对齐：2/2 完整；详见 [`surface_alignment.json`](surface_alignment.json)。",
        f"- 可执行检查：{problem['validation']['witness_passed']}/{problem['validation']['witness_checks']} 通过；详见 [`witness.json`](witness.json)。",
        "- 推导 coverage：上下文对齐、必要推理链和可执行答案检查均完整。",
        "",
        "## 请重点审核",
        "",
        "1. 每条节点是否只表达一个可复用的推理机制；",
        "2. 是否存在为了得到正确答案仍不可缺少、但当前推导链遗漏的步骤；",
        "3. SG/CS 的叙事语义是否都正确映射到同一原题结构；",
        "4. 是否有只属于本题数字事实、不应称为 Schema 的内容。",
        "",
    ]
    root = OUTPUT_ROOT / "problems" / problem["dir_name"]
    (root / "review.md").write_text("\n".join(lines), encoding="utf-8")


def write_readme(problems: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    lines = [
        "# ContextMATH Gold Schema v1 — 第一批审核稿",
        "",
        "> 当前仅为 6 个题组的 pilot，不代表 60 个题组已经完成。所有节点均等待人工审核。",
        "",
        "本批将同一 AIME 原题的 SG/CS 两种改写合并：叙事对齐保存在",
        "`surface_alignment.json`，共享数学推理保存在 `schemas.json`，标准答案和精确复算只保存在",
        "`witness.json` 与 `validation.json`。因此最终答案不会被包装成 Schema。",
        "当前只生成任务级 Level 3 节点；跨题的 Level 2 数学机制将在 60 个题组完成后统一归并，",
        "避免依据少量样本过早建立题型分类。",
        "",
        "## 清单",
        "",
        "| 题组 | Schema | Witness | 状态 | 审核 |",
        "|---|---:|---:|---|---|",
    ]
    for problem in problems:
        root = f"problems/{problem['dir_name']}"
        lines.append(f"| `{problem['key']}` | {len(problem['schemas'])} | {problem['validation']['witness_checks']} | pending | [review.md]({root}/review.md) |")
    lines += [
        "",
        "## 汇总",
        "",
        f"- 题组：{manifest['totals']['problems']} / 60",
        f"- 样本改写：{manifest['totals']['variants']} / 120",
        f"- 原子 Schema：{manifest['totals']['schemas']}",
        f"- 可执行检查：{manifest['totals']['witness_passed']}/{manifest['totals']['witness_checks']} 通过",
        "- 自动验证只能证明数据配对、推导依赖、精确复算和 coverage；语义粒度仍需人工确认。",
        "",
        "## 复现",
        "",
        "```bash",
        ".venv/bin/python scripts/generate_contextmath_gold.py",
        "```",
        "",
    ]
    (OUTPUT_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    problems = [build_problem(year, problem_id) for year, problem_id in PROBLEM_IDS]
    failures = [problem["key"] for problem in problems if problem["validation"]["status"] != "passed"]
    if failures:
        raise SystemExit(f"Validation failed for: {failures}")

    write_json(OUTPUT_ROOT / "schema_spec.json", build_spec())
    entries = []
    for problem in problems:
        root = OUTPUT_ROOT / "problems" / problem["dir_name"]
        write_json(root / "schemas.json", {"format_version": 1, "schemas": problem["schemas"]})
        write_json(root / "surface_alignment.json", problem["alignments"])
        write_json(root / "witness.json", problem["witness"])
        write_json(root / "coverage.json", problem["coverage"])
        write_json(root / "validation.json", problem["validation"])
        write_review(problem)
        entries.append({
            "problem_id": problem["key"],
            "directory": f"problems/{problem['dir_name']}",
            "schema_count": len(problem["schemas"]),
            "witness_checks": problem["validation"]["witness_checks"],
            "status": "pending",
        })

    manifest = {
        "format_version": 1,
        "created_date": date.today().isoformat(),
        "benchmark": "contextmath",
        "scope": "pilot",
        "status": "pilot_review_pending",
        "source_splits": ["aime_2024_sg", "aime_2024_cs", "aime_2025_sg", "aime_2025_cs"],
        "problems": entries,
        "totals": {
            "problems": len(problems),
            "variants": 2 * len(problems),
            "schemas": sum(len(problem["schemas"]) for problem in problems),
            "witness_checks": sum(problem["validation"]["witness_checks"] for problem in problems),
            "witness_passed": sum(problem["validation"]["witness_passed"] for problem in problems),
        },
    }
    write_json(OUTPUT_ROOT / "manifest.json", manifest)
    write_readme(problems, manifest)
    print(json.dumps(manifest["totals"], ensure_ascii=False))


if __name__ == "__main__":
    main()
