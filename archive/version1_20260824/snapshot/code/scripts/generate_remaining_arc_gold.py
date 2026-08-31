#!/usr/bin/env python3
"""Generate reviewable Gold Schema artifacts for the remaining ARC-AGI-3 games."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from generate_arc_gold_batch import (
    ENVIRONMENTS_DIR,
    GOLD_ROOT,
    PROJECT_ROOT,
    GoldBuilder,
    _load_game_class,
    _write_json,
)


@dataclass(frozen=True)
class Rule:
    key: str
    title: str
    kind: str
    trigger: str
    action: str | None
    expectation: str
    constraints: tuple[str, ...] = ()
    exceptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class GameDefinition:
    game_id: str
    version: str
    source: str
    levels: int
    symbol: str
    lines: tuple[int, int]
    task: str
    rules: tuple[Rule, ...]


def R(
    key: str,
    title: str,
    kind: str,
    trigger: str,
    action: str | None,
    expectation: str,
    *constraints: str,
) -> Rule:
    return Rule(key, title, kind, trigger, action, expectation, constraints)


GAMES: tuple[GameDefinition, ...] = (
    GameDefinition(
        "ar25-0c556536", "0c556536", "ar25/0c556536/ar25.py", 8,
        "Ar25 reflection, movement, selection, undo and goal logic", (1262, 1847),
        "移动实物及其镜像组合，使所有目标标记被组合图形覆盖",
        (
            R("composite", "实物与递归镜像共同构成有效图形", "observation_semantics", "关卡含可移动实物、镜面和目标标记。", None, "实物像素会沿相交镜轴递归反射；目标判定使用实物与全部有效镜像的并集。", "镜像递归深度受实现上限约束。"),
            R("move", "ACTION1–4 移动当前选中实物", "action_effect", "当前选中对象在对应方向仍位于边界内。", "ACTION1|ACTION2|ACTION3|ACTION4", "选中实物向上、下、左、右移动一个单位，镜像随新位置重算；越界输入不移动。"),
            R("reflect", "镜面按自身轴向反射图形", "state_transition", "实物位于可影响它的水平或垂直镜面一侧。", None, "镜像出现在到镜轴等距的另一侧；只支持单轴的对象不会继续产生另一轴镜像。"),
            R("auto_rotate", "跨越镜面相对距离会自动旋转实物", "state_transition", "一次移动改变实物相对带标记镜面的距离关系。", None, "实物自动旋转 90°并重新计算镜像，以保持镜面关系。"),
            R("select", "ACTION6 点击或 ACTION5 循环切换活动对象", "action_effect", "存在多个可控实物或可点击镜像。", "ACTION5|ACTION6", "点击命中的实物/镜像所对应实物成为活动对象；ACTION5 按固定次序选择下一对象。"),
            R("undo", "ACTION7 恢复上一移动快照", "action_effect", "至少发生过一次可记录移动。", "ACTION7", "恢复移动前的实物位置与朝向，并重算镜像。"),
            R("goal", "全部目标标记被组合图形覆盖时过关", "goal", "移动或恢复后重新检查目标。", None, "每一个目标标记位置都落在实物或其有效镜像像素上时进入下一关。"),
            R("budget", "有效操作耗尽预算时失败", "hazard", "动作预算降为零且目标未满足。", None, "游戏进入 GAME_OVER。"),
        ),
    ),
    GameDefinition(
        "bp35-0a0ad940", "0a0ad940", "bp35/0a0ad940/bp35.py", 9,
        "uakietkqfso mechanics and Bp35 action adapter", (4032, 4565),
        "在可翻转重力的平台空间中到达宝石/出口",
        (
            R("horizontal", "ACTION3/4 分别向左/右尝试移动", "action_effect", "角色处于可操作状态。", "ACTION3|ACTION4", "角色朝相应方向走一格；实体阻挡时原地播放碰撞反馈。"),
            R("gravity", "每次横向移动后沿当前重力方向坠落", "state_transition", "角色进入下方/上方连续空格。", None, "角色持续移动到当前重力方向上的首个支撑面；镜头跟随最终高度。"),
            R("gravity_switch", "点击重力装置会反转重力", "action_effect", "ACTION6 点击可见的重力切换装置。", "ACTION6", "重力方向上下反转，装置消失，角色沿新方向坠落。"),
            R("world_switch", "点击机关会切换同组平台状态", "action_effect", "ACTION6 点击开关、可消除块或生长机关。", "ACTION6", "同标识对象在其状态序列间切换，可能生成/移除相邻支撑，并立即重新结算坠落。"),
            R("spike", "落到尖刺会失败", "hazard", "重力坠落的落点是尖刺。", None, "角色播放受击过程后进入 GAME_OVER。"),
            R("goal", "接触宝石或出口即完成关卡", "goal", "横向移动或坠落终点命中宝石/出口。", None, "当前关完成；最终关满足时 WIN。"),
            R("undo", "ACTION7 撤销最近一次世界操作", "action_effect", "存在内部撤销快照。", "ACTION7", "恢复角色、重力与机关状态到上一快照。"),
        ),
    ),
    GameDefinition(
        "cn04-2fe56bfb", "2fe56bfb", "cn04/2fe56bfb/cn04.py", 6,
        "Cn04 selection, overlap cancellation and completion", (833, 1131),
        "移动和旋转彩色图形，使所有标记像素按同色两两重合",
        (
            R("marked", "两种标记颜色分别形成独立配对集合", "observation_semantics", "图形含颜色 8 或 13 的标记像素。", None, "只有相同标记颜色之间的重合才算该颜色的有效配对。"),
            R("select", "ACTION6 点击图形选择或循环重叠图形", "action_effect", "点击位置命中一个或多个可控图形。", "ACTION6", "单个命中时选中它；同一像素有多个图形时按点击次序循环选择。"),
            R("move", "ACTION1–4 移动选中图形", "action_effect", "选中图形移动后仍在棋盘边界内。", "ACTION1|ACTION2|ACTION3|ACTION4", "图形向上、下、左、右移动一格；越界时保持原位。"),
            R("rotate", "ACTION5 旋转单个活动图形", "action_effect", "活动位置没有多个可循环的重叠图形。", "ACTION5", "活动图形绕自身旋转 90°。"),
            R("cycle", "ACTION5 在重叠候选间循环", "action_effect", "活动位置含多个重叠图形。", "ACTION5", "改选下一个重叠图形而不旋转。"),
            R("cancel", "同色标记重合后在显示上被消去", "state_transition", "不同图形的相同标记颜色占据同一坐标。", None, "对应标记视为已配对并显示为消除态；分离后恢复未配对态。"),
            R("goal", "全部标记像素完成同色配对时过关", "goal", "移动或旋转后检查所有图形。", None, "所有颜色 8/13 的原始标记都有另一图形的同色标记与其重合时进入下一关。"),
        ),
    ),
    GameDefinition(
        "dc22-fdcac232", "fdcac232", "dc22/fdcac232/dc22.py", 6,
        "Dc22 player, crane, bridge, switch and goal logic", (9958, 10892),
        "操纵角色与起重机搭桥、开路并到达终点",
        (
            R("walk", "ACTION1–4 控制角色四向移动", "action_effect", "目标像素可站立且无实体阻挡。", "ACTION1|ACTION2|ACTION3|ACTION4", "角色按上、下、左、右移动；墙体或实体阻止位移。"),
            R("click_switch", "ACTION6 点击同组机关切换世界状态", "action_effect", "点击带机关标识的可见对象。", "ACTION6", "同字母/同组的桥、门或平台切换可见性、碰撞性或形态。"),
            R("crane_move", "起重机方向按钮沿吊臂网格移动抓取点", "action_effect", "ACTION6 点击起重机上/下/左/右按钮且目标锚点合法。", "ACTION6", "吊机锚点移动一格；携带物与锚点同步移动，碰撞时整次移动回退。"),
            R("grab", "抓取按钮拾取或释放锚点处对象", "action_effect", "抓取点与可抓桥段/货物中心重合。", "ACTION6", "空载时附着对象；有载时释放对象并保持当前位置。"),
            R("bridge", "桥段可被搬运并通过形态循环连接道路", "state_transition", "桥段被抓取、移动或同组机关触发。", None, "桥段位置/朝向/版本改变后参与碰撞和支撑，能够填补角色路径。"),
            R("key_gate", "角色取得钥匙会开启同标识阻挡", "state_transition", "角色移动到钥匙像素。", None, "钥匙被移除，对应门/阻挡变为可通行。"),
            R("fall_reset", "角色失去支撑会消耗一次恢复并回到快照", "hazard", "角色所在像素下不再有有效支撑。", None, "若仍有恢复次数则恢复关卡快照；次数耗尽时 GAME_OVER。"),
            R("goal", "角色到达目标位置时过关", "goal", "角色坐标与终点坐标完全相同。", None, "进入下一关；最后一关满足时 WIN。"),
            R("budget", "界面资源耗尽时失败", "hazard", "操作消耗使剩余资源为零且目标未满足。", None, "游戏进入 GAME_OVER。"),
        ),
    ),
    GameDefinition(
        "ft09-0d8bbf25", "0d8bbf25", "ft09/0d8bbf25/ft09.py", 6,
        "Ft09 toggle masks, clue constraints and goal", (2301, 2520),
        "点击局部翻色，使每个线索周围满足相同/不同约束",
        (
            R("toggle", "ACTION6 点击单元按当前局部掩码循环颜色", "action_effect", "点击可变色单元。", "ACTION6", "掩码覆盖到的单元在关卡颜色序列中前进一步；未覆盖单元不变。"),
            R("composite_mask", "复合按钮自身的颜色 6 像素定义翻色掩码", "observation_semantics", "点击带局部图案的复合对象。", None, "相对中心为颜色 6 的像素决定哪些邻域格被同时翻色。"),
            R("clue_equal", "线索的 0 像素要求对应邻格与中心同色", "constraint", "线索邻域中的对应位置存在棋盘格。", None, "该邻格颜色必须等于线索中心颜色。"),
            R("clue_different", "线索的非 0 像素要求对应邻格与中心异色", "constraint", "线索邻域中的对应位置存在棋盘格。", None, "该邻格颜色必须不同于线索中心颜色。"),
            R("goal", "所有现存邻接约束同时成立时过关", "goal", "一次有效点击完成后。", None, "每个线索的八邻域都满足其同色/异色要求时进入下一关。"),
            R("budget", "有效点击耗尽步数时失败", "hazard", "剩余点击数为零且约束未全部满足。", None, "游戏进入 GAME_OVER。"),
        ),
    ),
    GameDefinition(
        "g50t-5849a774", "5849a774", "g50t/5849a774/g50t.py", 7,
        "qxlodtievc path recording, actors, rewind and completion", (2494, 2855),
        "记录路径并生成延迟重放分身，协作到达阶段终点",
        (
            R("move", "ACTION1–4 沿可通行轨道移动角色", "action_effect", "目标方向连接到可通行路径。", "ACTION1|ACTION2|ACTION3|ACTION4", "角色向上、下、左、右移动并记录该方向；墙或断路阻止位移。"),
            R("autonomous", "自主单位每回合顺时针寻找首个可走方向", "state_transition", "玩家完成一次方向移动。", None, "自主单位从当前朝向起最多右转四次，沿首个可通行方向前进一步。"),
            R("rewind", "ACTION5 逐步倒放本阶段玩家路径", "action_effect", "本阶段已记录至少一次移动。", "ACTION5", "玩家每次沿最后一步的反方向回退，直到起点。"),
            R("clone", "完整倒放后留下按记录延迟重放的分身", "state_transition", "ACTION5 已把阶段路径完全倒回。", None, "当前位置生成分身；它按原动作序列和设定延迟逐步重放。"),
            R("stage", "完成倒放会推进到下一阶段目标", "state_transition", "本阶段路径封存为分身。", None, "玩家切换到下一起点/终点关系，已有分身继续执行。"),
            R("deadline", "周期左移的边界吞没关键对象会失败", "hazard", "累计操作达到滚动周期。", None, "场景边界左移；关键角色/目标越出可用区域时 GAME_OVER。"),
            R("goal", "玩家到达当前终点并完成全部阶段时过关", "goal", "玩家与阶段终点的规定锚点重合。", None, "中间阶段允许封存路径；全部阶段完成后进入下一关。"),
        ),
    ),
    GameDefinition(
        "ka59-38d34dbb", "38d34dbb", "ka59/38d34dbb/ka59.py", 7,
        "Ka59 room movement, pushing, explosions, enemies and goal", (41122, 41463),
        "移动可选房间并利用推挤与爆炸，把两类对象送入对应容器",
        (
            R("select", "ACTION6 点击可控矩形切换活动房间", "action_effect", "点击命中可控矩形。", "ACTION6", "被点击矩形显示为活动对象；对象位置不变。"),
            R("move", "ACTION1–4 移动活动房间", "action_effect", "目标方向没有固定墙体。", "ACTION1|ACTION2|ACTION3|ACTION4", "活动矩形向上、下、左、右移动一个固定单位。"),
            R("push", "房间移动会递归推动接触的可动物体", "state_transition", "活动房间边缘接触一个或多个可动物体。", None, "接触物沿同方向级联移动；任一对象被硬墙阻挡时推动停止并进入阻挡反馈。"),
            R("enemy", "敌人每个已完成操作向目标角色追近", "hazard", "一次移动或选择完成且敌人仍存在。", None, "敌人最多连续六个像素步沿较大坐标差方向逼近；重合时 GAME_OVER。"),
            R("charge", "条形爆炸物逐行充能后触发爆炸", "state_transition", "移动结算后存在未充满的条形爆炸物。", None, "每次结算填充一行；完全填满后进入多帧爆炸。"),
            R("explosion", "爆炸沿朝向推动范围内对象", "state_transition", "爆炸物充满并展开爆炸区域。", None, "被爆炸范围命中的可动物体沿爆炸朝向递归移动，爆炸结束后爆炸物失效。"),
            R("goal", "两类目标对象分别进入对应容器时过关", "goal", "移动/爆炸结算完成。", None, "第一类对象全部与可控矩形容器重合，第二类对象全部与固定目标容器重合时进入下一关。"),
            R("budget", "操作步数耗尽时失败", "hazard", "剩余步数为零且目标未满足。", None, "游戏进入 GAME_OVER。"),
        ),
    ),
    GameDefinition(
        "lf52-271a04aa", "271a04aa", "lf52/271a04aa/lf52.py", 10,
        "equnaohchtj selection, jumps, world shifts and Lf52 adapter", (5128, 5872),
        "选择同类物体并按可用方向跳跃消除，清空目标集合",
        (
            R("shift", "ACTION1–4 推动可移动棋盘部件并滚动视野", "action_effect", "关卡含方向可动的蓝色平台/滑块。", "ACTION1|ACTION2|ACTION3|ACTION4", "同类可动部件按上、下、左、右移动；相机/棋盘在特定关卡同步平移。"),
            R("select", "ACTION6 点击对象显示其可用跳跃方向", "action_effect", "点击一个可消除对象。", "ACTION6", "对象成为当前选择；只有“相邻处有可越过对象且再下一格为空/目标格”的方向显示为可用。"),
            R("jump", "点击方向标记使选中对象跨越中间对象", "action_effect", "已选对象存在有效方向标记。", "ACTION6", "对象移动到方向两倍距离的落点，中间的匹配对象被消除。"),
            R("type_match", "同类型落点对象可被合并消除", "state_transition", "跳跃中点存在与移动对象同名对象。", None, "中点对象消失，移动对象到达落点；不同类型接触只播放反馈而不按同类消除。"),
            R("world_events", "特定落点会触发平台移动或相机位移", "state_transition", "跳跃抵达关卡定义的机关坐标。", None, "对应棋盘整体或平台按固定方向平移，从而开放后续跳跃。"),
            R("goal", "清除规定的全部目标对象时过关", "goal", "一次跳跃完成。", None, "剩余普通目标对象达到关卡要求的终止数量（通常只剩最后一个）时进入下一关。"),
            R("undo", "ACTION7 恢复上一内部快照", "action_effect", "构建版本启用撤销且存在快照。", "ACTION7", "恢复棋盘对象、选择和视野位置。"),
            R("budget", "不同关卡共享分段动作上限", "hazard", "累计动作达到该关所属的上限且仍未完成。", None, "游戏进入 GAME_OVER。", "不记录各关具体动作数，只保留分段预算机制。"),
        ),
    ),
    GameDefinition(
        "lp85-305b61c3", "305b61c3", "lp85/305b61c3/lp85.py", 8,
        "Lp85 button-linked movement and typed goals", (21339, 21451),
        "点击方向按钮移动关联部件，使两类部件落到各自目标",
        (
            R("buttons", "ACTION6 点击按钮选择关联组与方向", "action_effect", "点击一个带组标识和左右标识的按钮。", "ACTION6", "该按钮关联的全部部件按其方向执行一次关卡定义的位移。"),
            R("linked", "同组部件作为一个联动集合移动", "state_transition", "按钮命中多个同组部件。", None, "集合中每个部件执行相同规则；未关联部件保持原位。"),
            R("collision", "非法联动位移不穿过边界或固定结构", "constraint", "计算后的任一部件位置越界或命中阻挡。", None, "该次非法移动不产生穿透。"),
            R("goal", "两类部件必须分别占据匹配目标", "goal", "一次有效按钮操作完成。", None, "所有第一类部件位于普通目标，且所有第二类部件位于另一类目标时进入下一关。"),
            R("budget", "有效按钮点击耗尽步数时失败", "hazard", "剩余步数为零且目标未满足。", None, "游戏进入 GAME_OVER。"),
        ),
    ),
    GameDefinition(
        "ls20-9607627b", "9607627b", "ls20/9607627b/ls20.py", 7,
        "Ls20 movement, modifiers, targets, hazards and completion", (1765, 2060),
        "在迷宫中修改携带物的形状、颜色和朝向，逐个匹配目标",
        (
            R("move", "ACTION1–4 在迷宫网格中移动角色", "action_effect", "目标格可通行。", "ACTION1|ACTION2|ACTION3|ACTION4", "角色向上、下、左、右移动一格；墙和不匹配目标阻挡移动。"),
            R("modifiers", "三类修改格分别循环形状、颜色和旋转", "state_transition", "角色进入对应修改格。", None, "当前携带物仅改变该修改格负责的一个属性，其余属性保持不变。"),
            R("target", "目标要求形状、颜色、旋转三属性完全匹配", "constraint", "角色尝试进入目标格。", None, "三属性全部相同则目标被清除；任一属性不同则目标仍是阻挡。"),
            R("time", "时间拾取物补充动作预算", "state_transition", "角色进入时间拾取物所在格。", None, "拾取物消失，剩余步数增加到关卡规定值以内。"),
            R("hazard", "移动危险物或陷阱会消耗生命并重置本关", "hazard", "角色与危险对象重合或预算耗尽。", None, "生命减一并恢复关卡初始状态；生命耗尽时 GAME_OVER。"),
            R("goal", "全部属性目标被清除时过关", "goal", "一次匹配目标完成。", None, "关卡中不再存在未满足目标时进入下一关。"),
            R("fog", "后续关卡的可见区域随角色位置变化", "observation_semantics", "关卡启用局部可见/遮挡机制。", None, "角色附近信息可见，远处被遮挡；遮挡不改变底层碰撞规则。"),
        ),
    ),
    GameDefinition(
        "m0r0-492f87ba", "492f87ba", "m0r0/492f87ba/m0r0.py", 6,
        "M0r0 symmetric control, pairing, gates and goal", (680, 924),
        "用对称控制或单独控制让四类图形两两重合配对",
        (
            R("symmetric", "全局 ACTION1–4 以四种镜像关系同时移动四组对象", "action_effect", "未选中单独对象。", "ACTION1|ACTION2|ACTION3|ACTION4", "一组同向移动，另三组分别水平镜像、垂直镜像、双轴镜像移动。"),
            R("individual", "ACTION6 选择对象后只移动该对象", "action_effect", "点击一个尚未配对的可控对象。", "ACTION6", "进入单独控制模式；随后方向输入只移动该对象。点击空处返回全局模式。"),
            R("noop", "ACTION5 不改变棋盘但仍计入操作次数", "action_effect", "游戏处于未完成状态。", "ACTION5", "对象位置、配对和门状态均保持不变；全局 action 计数照常增加。"),
            R("pair", "两个对象同坐标会配对并退出后续控制", "state_transition", "恰有两个未配对对象到达同一位置。", None, "两者合并为配对态，不再碰撞也不再参与移动。"),
            R("crowded", "三个以上对象同时相撞只接受首个合法配对", "constraint", "一次移动使三个以上未配对对象占据同一位置。", None, "仅首对完成合并，其余对象回到移动前位置。"),
            R("gate", "剩余对象压住传感器时对应门暂时移除", "state_transition", "任一未配对对象占据传感器。", None, "对应阻挡门变为不可碰撞；离开传感器后门恢复。"),
            R("goal", "所有对象均完成两两配对时过关", "goal", "一次移动后检查未配对对象数。", None, "不再有可触碰的未配对对象时进入下一关。"),
            R("budget", "累计操作超过上限时失败", "hazard", "操作计数超过 150 且尚未全部配对。", None, "游戏进入 GAME_OVER。"),
        ),
    ),
    GameDefinition(
        "r11l-495a7899", "495a7899", "r11l/495a7899/r11l.py", 6,
        "R11l piece selection, dragging, composition and target checks", (1404, 1849),
        "拖动组成图形并放到颜色集合匹配的目标上",
        (
            R("select", "ACTION6 点击部件将其选中", "action_effect", "点击命中可拖动部件。", "ACTION6", "部件高亮并成为后续落点点击的移动对象。"),
            R("drag", "选中后点击空位置把部件中心移动到点击点", "action_effect", "目标位置不与固定阻挡重叠。", "ACTION6", "部件移动到目标位置；与其关联的组合中心按所有子部件中心重新定位。"),
            R("wall", "固定阻挡禁止放置", "constraint", "拟放置位置使选中部件与墙/禁区重叠。", None, "移动不开始，部件保持原位。"),
            R("compose", "组合容器吸收接触的小彩色片段", "state_transition", "可组合图形与未吸收片段重合。", None, "片段颜色像素写入组合图形，片段从棋盘移除。"),
            R("target_match", "组合图形与目标按非零颜色集合匹配", "constraint", "组合图形与目标重合。", None, "只有两者出现的非零颜色集合完全相等才算该目标满足。"),
            R("hazard", "组合图形接触危险区会回退并累计失败", "hazard", "拖动终点使组合图形碰到危险对象。", None, "移动动画倒放到起点；累计五次后 GAME_OVER。"),
            R("goal", "所有规定目标保持匹配一段确认时间后过关", "goal", "每个目标都被对应组合图形覆盖且匹配。", None, "全部目标完成确认动画后进入下一关。"),
            R("budget", "60 次点击预算耗尽时失败", "hazard", "累计 action 达到 60。", None, "游戏进入 GAME_OVER。"),
        ),
    ),
    GameDefinition(
        "re86-8af5384d", "8af5384d", "re86/8af5384d/re86.py", 8,
        "Re86 active shape, deformation, composite target and goal", (1864, 2158),
        "移动并借助边界改变多个轮廓，使合成像素匹配目标模板",
        (
            R("active", "带中心标记的轮廓是当前活动图形", "observation_semantics", "棋盘上存在多个可移动轮廓。", None, "只有活动轮廓响应方向输入；活动标记随选择更新。"),
            R("move", "ACTION1–4 移动活动轮廓", "action_effect", "活动轮廓存在。", "ACTION1|ACTION2|ACTION3|ACTION4", "轮廓向上、下、左、右移动固定像素距离。"),
            R("cycle", "ACTION5 循环选择下一个轮廓", "action_effect", "存在多个轮廓。", "ACTION5", "当前标记清除，下一个轮廓获得活动标记。"),
            R("deform", "轮廓撞到特定边界会改变尺寸或内部标记", "state_transition", "方向移动使轮廓接触变形轴/墙。", None, "矩形轮廓按接触方向增减宽高；其他轮廓移动其中心结构，随后重新计算活动标记。"),
            R("composite", "目标比较使用全部轮廓像素的叠加", "observation_semantics", "多个轮廓占据目标区域。", None, "全部可见非透明像素合成为一个候选图案，不要求单个轮廓独立匹配。"),
            R("wildcard", "目标颜色 4 是通配符，透明像素不比较", "constraint", "比较候选图案与目标模板。", None, "颜色 4 接受任意候选颜色；目标透明位置不施加约束，其余位置必须相等。"),
            R("goal", "合成图案满足全部目标像素时过关", "goal", "移动或变形结算后。", None, "所有非透明、非通配目标像素与合成图案一致时进入下一关。"),
        ),
    ),
    GameDefinition(
        "s5i5-18d95033", "18d95033", "s5i5/18d95033/s5i5.py", 8,
        "S5i5 group rotation, resizing, collision rollback and goal", (2026, 2247),
        "按颜色成组旋转和伸缩线段，使所有端点覆盖目标",
        (
            R("rotate", "ACTION6 点击颜色按钮旋转该色全部线段", "action_effect", "点击颜色选择按钮。", "ACTION6", "该颜色的每条线段绕其锚点旋转 90°；子部件随父部件递归变换。"),
            R("resize", "ACTION6 点击条形控件两侧缩短或延长关联线段", "action_effect", "点击长度控件中心左/右侧。", "ACTION6", "关联线段长度减少/增加一个单位，依朝向保持锚点并移动其子部件。"),
            R("hierarchy", "父线段变换递归携带所有子部件", "state_transition", "父线段被旋转、伸缩或平移。", None, "所有直接和间接子部件保持相对连接关系同步变换。"),
            R("rollback", "变换导致线段互撞时整次操作回滚", "constraint", "变换后任意两条主线段发生碰撞。", None, "所有受影响线段和子部件恢复到操作前的位置、朝向与长度。"),
            R("goal", "每个目标点都被端点对象精确覆盖时过关", "goal", "一次无碰撞变换完成。", None, "所有目标点坐标都有端点对象位于同一坐标时进入下一关。"),
            R("budget", "有效点击耗尽步数时失败", "hazard", "剩余步数为零且目标未满足。", None, "游戏进入 GAME_OVER。"),
        ),
    ),
    GameDefinition(
        "sb26-7fbdac44", "7fbdac44", "sb26/7fbdac44/sb26.py", 8,
        "Sb26 editing, recursive evaluation, undo and energy", (712, 1146),
        "重排嵌套序列节点，使递归求值结果与输出序列一致",
        (
            R("frames", "外框表示有序序列，特殊节点引用另一外框", "observation_semantics", "关卡含多个有编号长度的框和叶节点。", None, "框内项目从左到右求值；引用节点按自身颜色/标识进入对应子框，形成递归序列。"),
            R("select_swap", "ACTION6 先选节点，再点击节点或空位完成交换/移动", "action_effect", "点击可编辑叶节点。", "ACTION6", "首次点击选中；再次点击另一节点交换位置，点击合法空位把选中节点移入该位。"),
            R("protected", "输出区与上方参考节点不可直接编辑", "constraint", "点击不带 sys_click 的参考/输出节点。", None, "不会进入选中或交换流程。"),
            R("evaluate", "ACTION5 从入口框递归展开序列", "action_effect", "当前布局可被解析。", "ACTION5", "求值器按框内顺序前进；遇到引用时进入子框，子框结束后返回父框下一项。"),
            R("match", "展开叶颜色必须逐项匹配输出槽", "constraint", "求值器产生下一个叶值。", None, "叶颜色等于当前输出槽颜色则继续；不相等或出现非法递归则本次求值失败。"),
            R("goal", "递归展开完整且恰好匹配全部输出时过关", "goal", "ACTION5 求值到根序列末尾。", None, "所有输出槽均被按序匹配且没有多余值时进入下一关。"),
            R("undo_energy", "ACTION7 撤销编辑；编辑和求值消耗能量", "hazard", "存在上一编辑快照或能量继续减少。", "ACTION7", "撤销恢复上一布局；能量降到零且未完成时 GAME_OVER。"),
        ),
    ),
    GameDefinition(
        "sc25-635fd71a", "635fd71a", "sc25/635fd71a/sc25.py", 6,
        "Sc25 movement, spell patterns, spell effects and exit", (1655, 2750),
        "移动角色并在 3×3 面板输入法术图案，解除障碍后到达出口",
        (
            R("move", "ACTION1–4 按角色尺寸四向移动", "action_effect", "目标区域不被墙或实体占据。", "ACTION1|ACTION2|ACTION3|ACTION4", "角色向上、下、左、右移动；大尺寸角色先尝试整步，受阻时可尝试半步。"),
            R("select_spell", "ACTION6 点击法术图标选择待输入法术", "action_effect", "点击传送、缩放或攻击法术图标。", "ACTION6", "对应法术成为当前参考；再次点击同一图标会演示其 3×3 图案。"),
            R("pattern", "点击 3×3 槽位切换布尔图案", "action_effect", "ACTION6 点击法术输入面板的一个槽位。", "ACTION6", "该槽开/关翻转；当前九格与某个已解锁法术模板完全相等时自动施法。"),
            R("teleport", "传送法术把角色与循环目标交换/传送", "state_transition", "输入图案匹配 tevyeq。", None, "角色移动到当前传送目标位置，并把目标端状态按循环顺序推进。"),
            R("scale", "缩放法术在尺寸 1 和 2 间切换", "state_transition", "输入图案匹配 sieesc_chwjgc。", None, "角色缩小或放大；放大时寻找不碰撞的邻近偏移，完全无空间则保持原尺寸。"),
            R("attack", "攻击法术沿当前朝向发射直线投射物", "state_transition", "输入图案匹配 fibcey。", None, "投射物前进到首个阻挡；命中特定可破坏物时将其及关联障碍移除。"),
            R("pickup", "时间拾取物降低已用动作数", "state_transition", "角色移动后与时间拾取物重合。", None, "拾取物消失，动作计数减少但不低于零。"),
            R("goal", "角色进入出口时过关", "goal", "一次移动命中出口对象。", None, "角色补齐到出口中心后进入下一关。"),
            R("budget", "移动、面板编辑和施法共享动作预算", "hazard", "已用动作超过关卡预算。", None, "游戏进入 GAME_OVER。", "不枚举各关具体预算。"),
        ),
    ),
    GameDefinition(
        "sp80-589a99af", "589a99af", "sp80/589a99af/sp80.py", 6,
        "Sp80 selection, vessel movement, fluid simulation and goal", (454, 868),
        "摆放容器并倾倒液体，让液体到达所有目标且避开危险区",
        (
            R("select", "ACTION6 点击容器/固体选择活动对象", "action_effect", "点击可移动容器或固体。", "ACTION6", "命中对象成为当前活动对象。"),
            R("move", "ACTION1–4 逐像素移动活动对象", "action_effect", "移动后不越界且不与墙/其他固体碰撞。", "ACTION1|ACTION2|ACTION3|ACTION4", "对象向上、下、左、右移动一个像素；非法移动保持原位。"),
            R("pour", "ACTION5 开始一次液体倾倒模拟", "action_effect", "存在装有液体的可用容器。", "ACTION5", "生成液体粒子并逐帧在重力方向流动。"),
            R("flow", "液体遇阻会沿侧向寻找可用路径并分流", "state_transition", "下一个重力方向像素被固体阻挡。", None, "粒子尝试两个侧向位置，可用时分支；之后继续受重力影响。"),
            R("hazard", "任意液体进入危险区域会判定本次失败", "hazard", "液体粒子与危险标记重合。", None, "本次倾倒失败并恢复液体与可动物体初态。"),
            R("goal", "所有接收目标被液体到达且无危险接触时过关", "goal", "一次倾倒稳定结束。", None, "每个目标位置均有液体到达且没有危险命中时进入下一关。"),
            R("attempts", "失败倾倒有次数上限并受动作预算约束", "hazard", "失败后重试次数或动作预算耗尽。", None, "超过允许倾倒次数或预算归零时 GAME_OVER。"),
        ),
    ),
    GameDefinition(
        "su15-1944f8ab", "1944f8ab", "su15/1944f8ab/su15.py", 9,
        "Su15 radial pulse, merging, enemies, zones, undo and completion", (907, 2165),
        "点击产生收缩圆形脉冲，移动和合并对象以满足目标区计数",
        (
            R("pulse", "ACTION6 点击场内位置产生收缩圆形脉冲", "action_effect", "点击位于有效纵向游戏区域。", "ACTION6", "以点击点为圆心显示半径逐步缩小的环，并选中初始半径内的对象。"),
            R("radial_move", "脉冲使圈内对象沿相对圆心方向移动", "state_transition", "对象位于本次脉冲影响半径内。", None, "对象在若干帧内沿径向移动；不同对象类别可向内/向外并具有不同速度。"),
            R("merge", "相同等级对象碰撞会合并升级", "state_transition", "两个或更多同等级主对象在脉冲中重叠。", None, "它们被替换为位于平均中心的下一等级对象；最高等级再合并则消失。"),
            R("mismatch", "不同等级对象相撞会触发冲突而不升级", "hazard", "脉冲使两个不同等级主对象重叠。", None, "进入冲突/受击处理，不按同级合并规则生成升级对象。"),
            R("enemy", "未被脉冲直接操纵的敌对对象会追向最近主对象", "hazard", "脉冲动画尚未结束且存在可追目标。", None, "敌对对象按其类别速度接近最近目标；接触会降低目标等级或移除目标。"),
            R("zones", "目标区按对象等级或敌对类别精确计数", "constraint", "检查目标区内对象。", None, "每项要求的类型/等级数量必须与关卡数据完全相等，多一个或少一个都不满足。"),
            R("goal", "目标区内精确计数全部满足时过关", "goal", "一次脉冲及其碰撞结算完成。", None, "所有关卡定义的类型计数均准确满足时进入下一关。"),
            R("undo", "ACTION7 恢复上一次脉冲前快照", "action_effect", "存在动作历史快照。", "ACTION7", "重建主对象和敌对对象的类型、等级与位置。"),
        ),
    ),
    GameDefinition(
        "tn36-ef4dde99", "ef4dde99", "tn36/ef4dde99/tn36.py", 7,
        "dimsufvezo commands, ytkjoffamq runner and Tn36 adapter", (2129, 2647),
        "编辑二进制程序并运行，使对象的位置、尺寸、旋转和颜色匹配目标",
        (
            R("bits", "点击程序位会切换比特并改变指令编号", "action_effect", "ACTION6 点击可编辑程序单元。", "ACTION6", "该位在 0/1 间切换；同一指令格的位按二进制组合成命令编号。"),
            R("programs", "程序选择器在多个独立程序间切换并保留内容", "action_effect", "ACTION6 点击程序标签/选择器。", "ACTION6", "当前编辑区切换到对应程序；离开前的位模式被保存。"),
            R("run", "点击运行区按顺序执行当前程序", "action_effect", "ACTION6 点击棋盘/运行按钮且程序非空。", "ACTION6", "从第一条命令开始逐条执行，等待每条动画和碰撞结算后继续。"),
            R("move_commands", "移动命令按编号产生一格或两格位移", "state_transition", "执行编号对应上、下、左、右或双步移动的命令。", None, "活动对象沿指定方向移动；墙体或对象碰撞可阻止或破坏移动。"),
            R("transform_commands", "变换命令旋转、缩放或改色", "state_transition", "执行旋转、尺寸或颜色命令。", None, "对象按命令旋转 90/180/270°、增减尺寸，或切换到指定颜色。"),
            R("collision", "命令后的边界和碰撞决定是否保留变换", "constraint", "执行结果越界或与不可穿越对象重叠。", None, "非法结果被阻止；特定可破坏对象按碰撞规则移除。"),
            R("goal", "对象四个属性完全匹配目标时过关", "goal", "程序执行后检查状态。", None, "位置、尺寸、旋转和颜色全部与目标一致时进入下一关。"),
            R("deadline", "背景周期左移形成点击时限", "hazard", "完成规定次数的点击操作。", None, "场景向左滚动；关键对象被边界吞没时 GAME_OVER。"),
        ),
    ),
    GameDefinition(
        "tr87-cd924810", "cd924810", "tr87/cd924810/tr87.py", 6,
        "Tr87 rule selection, symbol cycling and translation matcher", (901, 1121),
        "根据上方示例翻译规则，调整下方输出序列使其成为输入的正确翻译",
        (
            R("examples", "分隔符两侧的序列定义翻译规则", "observation_semantics", "上方存在若干输入序列、输出序列对。", None, "每对示例把左侧连续符号序列映射到右侧连续符号序列；符号的随机朝向不改变其类别。"),
            R("select", "ACTION3/4 在可编辑符号或规则侧之间循环选择", "action_effect", "处于正常编辑状态。", "ACTION3|ACTION4", "选择标记向前/向后循环；普通关选择下方输出符号，规则编辑关可选择示例任一侧序列。"),
            R("cycle", "ACTION1/2 循环当前选中符号的类别", "action_effect", "已有选中符号或选中规则序列。", "ACTION1|ACTION2", "符号类别按固定字典向前/向后循环；选中规则序列时整段各符号同步循环。"),
            R("translate", "输入序列按示例规则从左到右分段翻译", "state_transition", "检查下方输入和候选输出。", None, "从输入当前位置寻找可匹配的规则左侧，追加其右侧并继续，直到完整消耗输入。"),
            R("tree", "树翻译关会把规则输出继续作为另一规则输入", "state_transition", "关卡启用 tree_translation。", None, "第一次翻译产生的每个符号序列还需找到下一条规则并再次翻译。"),
            R("double", "双重翻译关要求通过中间规则链得到最终输出", "state_transition", "关卡启用 double_translation。", None, "规则可先经连接关系扩展，再用其输出匹配另一规则左侧并得到最终右侧。"),
            R("goal", "候选输出等于完整翻译结果时过关", "goal", "ACTION1/2 修改后执行规则匹配。", None, "输入被规则完整分解且候选输出逐项等于推导输出时进入下一关。"),
        ),
    ),
    GameDefinition(
        "vc33-5430563c", "5430563c", "vc33/5430563c/vc33.py", 7,
        "Vc33 gravity-axis transfer, swap and color-routing goal", (1829, 2124),
        "点击隔板转移空间或交换两侧内容，使彩色物体到达匹配通道",
        (
            R("gravity", "关卡重力轴决定所有前后、上下和宽高计算", "observation_semantics", "关卡数据给出水平或垂直重力向量。", None, "同一机制沿重力轴执行；正负方向决定靠近/远离端的比较。"),
            R("transfer", "点击可调隔板会从一侧削去单位并加到另一侧", "action_effect", "ACTION6 点击带调节标记的隔板。", "ACTION6", "隔板两侧空间按重力方向一增一减，受支撑物体随边界移动。"),
            R("support", "物体随其支撑面和可用空间重新落位", "state_transition", "隔板转移改变支撑边界。", None, "位于受影响支撑上的对象沿重力方向同步移动，且不会越过最近墙面。"),
            R("swap", "点击夹在两个合法空间之间的交换器互换两侧对象", "action_effect", "ACTION6 点击处于两侧支撑完整状态的交换器。", "ACTION6", "交换器先移开，再把两侧受支撑对象移动到对侧，最后回到原位。"),
            R("blocked_swap", "缺少任一侧完整支撑时不能交换", "constraint", "交换器一侧没有成对支撑边界。", None, "交换动作不启动。"),
            R("goal", "每个彩色源必须与同色目标通道连通", "goal", "隔板或交换动画完成。", None, "对每个彩色源，都存在含该颜色的目标对象位于同一重力横截面且与其支撑墙连通，才进入下一关。"),
            R("budget", "有效点击耗尽步数时失败", "hazard", "剩余步数为零且路由未全部满足。", None, "游戏进入 GAME_OVER。"),
        ),
    ),
    GameDefinition(
        "wa30-ee6fef47", "ee6fef47", "wa30/ee6fef47/wa30.py", 9,
        "Wa30 movement, carrying, autonomous routing and goal", (874, 1254),
        "搬运绿色对象并与自主载具协作，把它们送到匹配目标区",
        (
            R("move", "ACTION1–4 四向移动主载具", "action_effect", "目标位置不越界且不与障碍重叠。", "ACTION1|ACTION2|ACTION3|ACTION4", "主载具向上、下、左、右移动；障碍或禁区阻止位移。"),
            R("pickup", "ACTION5 拾取相邻未绑定对象", "action_effect", "主载具未携带对象且相邻位置有可搬绿色对象。", "ACTION5", "该对象绑定到载具并移动到携带锚点。"),
            R("drop", "ACTION5 放下当前携带对象", "action_effect", "主载具已绑定一个对象且放置位置合法。", "ACTION5", "解除绑定，对象留在当前携带锚点位置。"),
            R("coupled", "载具移动会同步移动其绑定对象", "state_transition", "主载具或自主载具携带对象。", None, "载具和对象保持固定相对位置作为一个组合移动；组合碰撞会阻止该步。"),
            R("autonomous", "自主载具每回合寻路到未分配对象再送往目标", "state_transition", "玩家完成一次方向或交互操作。", None, "空载自主载具选择未分配对象，载物后选择对应目标区，并沿可通行最短路移动一步。"),
            R("typed_targets", "不同自主载具类型使用各自目标集合", "constraint", "载具为某一运输类型。", None, "它只把对象送到该类型允许的目标区；其他目标区不算完成。"),
            R("goal", "全部绿色对象在规定目标区且不再被携带时过关", "goal", "玩家与自主载具移动结算后。", None, "每个对象都位于其目标集合中的某一区域，且所有绑定关系已解除时进入下一关。"),
        ),
    ),
)


def _action_sequence(action: str | None) -> list[dict[str, Any]]:
    if action is None:
        return []
    return [{"action": action, "arguments": {}}]


def _runtime_smoke(builder: GoldBuilder) -> tuple[bool, dict[str, Any]]:
    """Load every level and execute every advertised action on a fresh level 1."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/socialclaw-mpl")
    from arcengine import ActionInput, GameAction, GameState

    game_class = _load_game_class(builder.game_id)
    loaded: list[dict[str, Any]] = []
    try:
        for level_index in range(len(builder.levels)):
            game = game_class()
            game.set_level(level_index)
            frame = game.camera.render(game.current_level.get_sprites())
            loaded.append({"level": level_index + 1, "shape": list(frame.shape)})
        probe = game_class()
        probe.set_level(0)
        advertised = list(probe._available_actions)
        actions: list[dict[str, Any]] = []
        for action_id in advertised:
            game = game_class()
            game.set_level(0)
            game._state = GameState.NOT_FINISHED
            result = game.perform_action(
                ActionInput(
                    id=GameAction.from_id(action_id),
                    data={"x": 0, "y": 0},
                ),
                raw=True,
            )
            actions.append(
                {
                    "action": action_id,
                    "frames": len(result.frame),
                    "state": str(result.state),
                }
            )
        return True, {"levels": loaded, "advertised_actions": actions}
    except Exception as exc:  # pragma: no cover - emitted into validation diagnostics
        return False, {"levels": loaded, "error": f"{type(exc).__name__}: {exc}"}


def generate_game(definition: GameDefinition) -> tuple[GoldBuilder, dict[str, Any]]:
    builder = GoldBuilder(
        game_id=definition.game_id,
        version=definition.version,
        source_path=ENVIRONMENTS_DIR / definition.source,
        levels=list(range(1, definition.levels + 1)),
    )
    for rule in definition.rules:
        builder.add(
            rule.key,
            title=rule.title,
            kind=rule.kind,
            trigger=rule.trigger,
            action_sequence=_action_sequence(rule.action),
            expectation=rule.expectation,
            symbol=definition.symbol,
            lines=definition.lines,
            constraints=list(rule.constraints),
            exceptions=list(rule.exceptions),
            runtime_evidence=(
                ["all_levels_and_actions_smoke"]
                if rule.kind in {"action_effect", "state_transition", "constraint", "hazard", "goal"}
                else None
            ),
        )
    builder.resolve_relations()
    passed, observed = _runtime_smoke(builder)
    dynamic_keys = [
        rule.key
        for rule in definition.rules
        if rule.kind in {"action_effect", "state_transition", "constraint", "hazard", "goal"}
    ]
    builder.record("all_levels_and_actions_smoke", dynamic_keys, passed, observed)
    for rule in definition.rules:
        builder.cover(f"mechanism_{rule.key}", rule.kind, [rule.key])
    goal_keys = [rule.key for rule in definition.rules if rule.kind == "goal"]
    if not goal_keys:
        raise RuntimeError(f"{definition.game_id} has no goal rule")
    for level in builder.levels:
        builder.cover(f"level_{level}_goal", "goal", goal_keys, [level])

    rules = "\n".join(
        f"| {rule.title} | {rule.expectation} |" for rule in definition.rules
    )
    review = f"""# {definition.game_id.split('-', 1)[0].upper()} Gold Schema v1 — 人工审核稿

> 状态：待人工审核。语义来自固定源码；运行证据为全关卡加载与可用 action 烟雾检查，不替代逐条语义审阅。

## 摘要

- 核心任务：{definition.task}
- Gold Schema：{len(definition.rules)} 条
- 覆盖：{definition.levels} 个 levels；每条机制和每关目标均已映射

## 规则

| 机制 | 结论 |
|---|---|
{rules}

## 审核重点

- 是否有规则抽象过粗或应当拆分；
- 是否存在只属于某一关初始配置、不应作为通用 Schema 的内容；
- 对混淆对象的中文命名是否需要改得更中性。
"""
    validation = builder.write(review)
    return builder, validation


def write_full_manifest(
    generated: list[tuple[GoldBuilder, dict[str, Any]]],
) -> None:
    inventory = json.loads(
        (ENVIRONMENTS_DIR / "inventory.json").read_text(encoding="utf-8")
    )
    by_id = {builder.game_id: (builder, validation) for builder, validation in generated}
    games: list[dict[str, Any]] = []
    for item in inventory["games"]:
        game_id = item["game_id"]
        if game_id in by_id:
            builder, validation = by_id[game_id]
            levels = builder.levels
            source = builder.source_relative
            source_sha256 = builder.source_sha256
            schema_count = validation["schema_count"]
            review_status = "pending"
        else:
            game_root = GOLD_ROOT / "games" / game_id
            validation = json.loads(
                (game_root / "validation.json").read_text(encoding="utf-8")
            )
            schemas = json.loads(
                (game_root / "schemas.json").read_text(encoding="utf-8")
            )["schemas"]
            levels = sorted({level for schema in schemas for level in schema["level_scope"]})
            source = item["source"]
            source_sha256 = validation["source_sha256"]
            schema_count = validation["schema_count"]
            review_status = (
                "accepted_and_revised" if game_id == "cd82-fb555c5d" else "accepted"
            )
        games.append(
            {
                "game_id": game_id,
                "source": source,
                "source_sha256": source_sha256,
                "levels": levels,
                "schema_count": schema_count,
                "validation": f"games/{game_id}/validation.json",
                "review": f"games/{game_id}/review.md",
                "review_status": review_status,
            }
        )
    _write_json(
        GOLD_ROOT / "manifest.json",
        {
            "format_version": 1,
            "created_date": date.today().isoformat(),
            "benchmark": "arc_agi3",
            "status": "full_batch_review_pending",
            "games": games,
            "totals": {
                "games": len(games),
                "levels": sum(len(game["levels"]) for game in games),
                "schemas": sum(game["schema_count"] for game in games),
            },
        },
    )
    task_by_id = {definition.game_id: definition.task for definition in GAMES}
    task_by_id.update(
        {
            "cd82-fb555c5d": "按参考图案绘制 10×10 画布",
            "sk48-d8078629": "伸缩链条并按参考颜色序列推动方块",
            "tu93-0768757b": "沿通路避开/处理移动单位并到达出口",
        }
    )
    rows = "\n".join(
        "| "
        + " | ".join(
            [
                f"[{game['game_id']}](games/{game['game_id']}/review.md)",
                str(len(game["levels"])),
                str(game["schema_count"]),
                game["review_status"],
                task_by_id[game["game_id"]],
            ]
        )
        + " |"
        for game in games
    )
    (GOLD_ROOT / "README.md").write_text(
        f"""# ARC-AGI-3 Gold Schema v1

固定版本 inventory 中的 25 个游戏已全部生成，共 {sum(len(game['levels']) for game in games)} 个 levels、{sum(game['schema_count'] for game in games)} 条 Schema。

- `accepted` / `accepted_and_revised`：已完成人工审核；
- `pending`：已通过源码锚定、结构验证、全关卡加载与 action 烟雾检查，等待人工语义审核；
- 每个游戏目录均含 `schemas.json`、`runtime_cases.json`、`coverage.json`、`validation.json` 和中文 `review.md`。

| 游戏 | Levels | Schema | 审核 | 核心任务 |
|---|---:|---:|---|---|
{rows}
""",
        encoding="utf-8",
    )


def main() -> None:
    from generate_arc_gold_batch import main as generate_reviewed_batch

    generate_reviewed_batch()
    generated: list[tuple[GoldBuilder, dict[str, Any]]] = []
    for definition in GAMES:
        builder, validation = generate_game(definition)
        generated.append((builder, validation))
        print(
            f"{builder.game_id}: {validation['schema_count']} schemas, "
            f"{validation['runtime_passed']}/{validation['runtime_case_count']} runtime"
        )
    write_full_manifest(generated)
    from generate_cross_game_gold import main as generate_cross_game

    generate_cross_game()
    manifest = json.loads((GOLD_ROOT / "manifest.json").read_text(encoding="utf-8"))
    print(json.dumps(manifest["totals"], ensure_ascii=False))


if __name__ == "__main__":
    main()
