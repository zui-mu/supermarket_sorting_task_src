#!/usr/bin/env python3
"""超市分拣比赛裁判系统（流程化计分）。

以"流程 (cycle)"为单位打分：一个流程 = 把一个目标物体从货架搬到配送台的完整
循环，含 5 个有序步骤 S1..S5，必须按序完成、不能跳步；只有走完 S5 才结算一次分。
碰撞/碰倒/掉落等扣分只在该流程窗口内评估，从这个流程的得分里扣。

裁判用 MuJoCo 地面真值判定，由 server 每物理步调用 update(mj_data)。
"""
import json
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULTS = {
    "time_limit_s": 600.0,
    # 一个流程自上次步骤推进起的最长存活时间。超时后该流程作废(不计分/不扣分),
    # 下次进入取货区重新开流程, 防止一次未完成的抓取让裁判在剩余赛程里无法再计分。
    "flow_timeout_s": 240.0,
    # 步骤 3 时夹爪松开后, 目标静止且持续未被夹持的判定时间。覆盖物体落在
    # 高台面(货架/桌面, z 高于 drop_z)导致流程永远卡在 S3 的情况。
    "ungripped_rest_s": 1.2,
    "scores": {"s1": 1, "s2": 2, "s3": 6, "s4": 1, "s5": 10, "upright_bonus": 5},
    "penalties": {"collision": 5, "topple": 5, "drop": 3},
    "thresholds": {
        "carry_out_dist": 0.20,   # S3: 目标离初始货位水平位移阈值(撤出货架)
        "settle_speed": 0.02,     # S5: 静止判定线速度 m/s
        "upright_tol_deg": 15.0,  # B1: 直立加分的最大倾角
        "topple_tilt_deg": 30.0,  # C2: 碰倒的倾角阈值
        "topple_xy_shift": 0.05,  # C2: 碰倒的水平位移阈值 m
        "drop_z": 0.30,           # C3: 掉落判定高度 m
    },
    "zones": {
        "picking": {"x": [-2.5, 2.5], "y": [1.70, 3.25]},
        # S4 base 到达判定：机器人常站在配送区北缘外伸手放置，故向北放宽 ~0.35m
        "delivery_base": {"x": [-2.420, -1.460], "y": [-3.880, -2.620]},
        # S5 物体判定框：严格对应桌面上方区域
        "delivery_box": {"x": [-2.420, -1.460], "y": [-3.630, -3.190], "z": [0.74, 1.05]},
    },
    "robot_links": [
        "agv_link", "slide_link",
        "rgt_arm_link1", "rgt_arm_link2", "rgt_arm_link3",
        "rgt_arm_link4", "rgt_arm_link5", "rgt_arm_link6",
    ],
    "grasp_fingers": ["rgt_finger_left_link", "rgt_finger_right_link"],
    "touch_links": ["rgt_finger_left_link", "rgt_finger_right_link", "rgt_arm_link6"],
    "structures": [
        "shelf_A", "shelf_B", "shelf_C", "shelf_D", "shelf_E",
        "perimeter_walls", "delivery_table", "dynamic_obstacle_corridor",
        "dynamic_obstacle_box_01", "dynamic_obstacle_box_02", "dynamic_obstacle_box_03",
        "dynamic_obstacle_box_04", "dynamic_obstacle_box_05",
    ],
}


def _merge(base, override):
    out = dict(base)
    for k, v in (override or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


class Flow:
    """单个流程的状态。step: 已达成的步号(0..5)。"""
    def __init__(self, t0):
        self.step = 1            # S1 到达取货区(开启时即达成)
        self.target = None       # 绑定的目标 body(S3 时确定)
        self.touched = set()     # 已 touch 的目标
        self.collided = False
        self.toppled = False
        self.dropped = False
        self.t = {"s1": t0, "s2": None, "s3": None, "s4": None, "s5": None}
        self.upright = False
        self.final_pos = None
        self.final_speed = None
        # S3 时夹爪松开后的计时: 用于把"落在高台面上的掉落"与短暂脱手区分开
        self.ungripped_since = None
        # 最近一次步骤推进时间(流程超时作废的基线)
        self.t_last_advance = float(t0)


class Referee:
    def __init__(self, mj_model, target_bodies, object_bodies, config=None):
        self.mj_model = mj_model
        self.time_stamp = time.time()
        if isinstance(config, str) and config and os.path.exists(config):
            config = json.load(open(config, "r", encoding="utf-8"))
        self.cfg = _merge(DEFAULTS, config if isinstance(config, dict) else None)

        self.targets = list(target_bodies)
        self.objects = list(object_bodies)
        self.non_targets = [b for b in self.objects if b not in self.targets]

        # body name -> id（缺失的名字静默跳过，兼容不同场景）
        self.bid = {}
        for name in set(self.cfg["robot_links"] + self.cfg["touch_links"]
                        + self.cfg["grasp_fingers"] + self.cfg["structures"] + self.objects):
            try:
                self.bid[name] = int(mj_model.body(name).id)
            except KeyError:
                pass

        self.reset()

    def reset(self, mj_data=None):
        """Start a fresh scoring run while keeping the same targets/config."""
        self.time_stamp = time.time()
        self.init_pos = {}     # 首帧快照(碰倒/撤出货架的基线)
        self.snapped = False
        self.finished = False

        self.flow = None
        self.scored = set()          # 已成功计分的目标
        self.retired = set()         # 已离开货架的目标(已计分/已掉落), 不再参与 C2 碰倒判定
        self.toppled_penalized = set()  # 已为 C2 扣过一次分的目标, 避免同一倒伏物反复扣分
        self.records = []            # 每个结算流程的明细
        self.events = []             # 文本事件流
        self._info = ""
        if mj_data is not None:
            self._snapshot(mj_data)
            self._build_info(float(mj_data.time))

    # ---------- 几何/接触工具 ----------
    def _snapshot(self, mj_data):
        for b in self.objects:
            if b in self.bid:
                self.init_pos[b] = mj_data.body(b).xpos.copy()
        self.snapped = True

    def _base_xy(self, mj_data):
        return mj_data.site("base_link").xpos[:2]

    @staticmethod
    def _in(zone, p):
        ok = zone["x"][0] <= p[0] <= zone["x"][1] and zone["y"][0] <= p[1] <= zone["y"][1]
        if "z" in zone:
            ok = ok and zone["z"][0] <= p[2] <= zone["z"][1]
        return ok

    def _contact_pairs(self, mj_data):
        """本 tick 处于接触的 (bodyid, bodyid) 对集合(双向)。"""
        pairs = set()
        self._contact_geom_pairs = []
        gb = self.mj_model.geom_bodyid
        for i in range(mj_data.ncon):
            c = mj_data.contact[i]
            b1, b2 = int(gb[c.geom1]), int(gb[c.geom2])
            pairs.add((b1, b2))
            pairs.add((b2, b1))
            self._contact_geom_pairs.append((int(c.geom1), int(c.geom2)))
        return pairs

    def _touch(self, pairs, links, obj):
        oid = self.bid.get(obj)
        if oid is None:
            return False
        return any((self.bid[l], oid) in pairs for l in links if l in self.bid)

    def _gripped(self, pairs, obj):
        """两指同时接触 = 夹持。"""
        return all(self._touch(pairs, [f], obj) for f in self.cfg["grasp_fingers"])

    def _pos(self, mj_data, obj):
        return mj_data.body(obj).xpos

    def _tilt_deg(self, mj_data, obj):
        R = Rotation.from_quat(mj_data.body(obj).xquat[[1, 2, 3, 0]]).as_matrix()
        return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))

    def _speed(self, mj_data, obj):
        jid = int(self.mj_model.body(obj).jntadr[0])
        if jid < 0:
            # Welded/static body without a joint: no dofs, no velocity.
            # (Negative indexing into jnt_dofadr would silently read the
            # wrong dof block.)
            return 0.0
        dof = int(self.mj_model.jnt_dofadr[jid])
        return float(np.linalg.norm(mj_data.qvel[dof:dof + 3]))

    def _xy_shift(self, mj_data, obj):
        return float(np.linalg.norm(self._pos(mj_data, obj)[:2] - self.init_pos[obj][:2]))

    # ---------- 主循环 ----------
    def update(self, mj_data):
        if self.finished:
            return
        if not self.snapped:
            self._snapshot(mj_data)
        t = float(mj_data.time)

        if t >= self.cfg["time_limit_s"]:
            self._finalize(t, reason="time_limit")
            return

        pairs = self._contact_pairs(mj_data)
        base_xy = self._base_xy(mj_data)
        in_pick = self._in(self.cfg["zones"]["picking"], base_xy)
        in_deliv = self._in(self.cfg["zones"]["delivery_base"], base_xy)
        remaining = [b for b in self.targets if b not in self.scored]

        # 全部目标已完成: 提前终局, 不再空转等待时间上限。
        if self.flow is None and not remaining:
            self._finalize(t, reason="all_targets_completed")
            return

        # 开启新流程：手上没有在搬、进入取货区、还有目标
        if self.flow is None and remaining and in_pick:
            self.flow = Flow(t)
            self._log(t, "S1 到达取货区，开启新流程")

        f = self.flow
        if f is None:
            self._build_info(t)
            return

        # --- 流程超时作废: 长时间没有任何步骤推进时放弃本流程 ---
        # 防止一次失败/卡住的抓取让 self.flow 永远占用, 导致剩余赛程无法再开启新流程计分。
        if t - f.t_last_advance > float(self.cfg.get("flow_timeout_s", 240.0)):
            self._cancel_flow(t, "流程超时作废")
            self._build_info(t)
            return

        # --- 流程窗口内的扣分监控 ---
        collision_pair = self._robot_structure_contact(pairs)
        if not f.collided and collision_pair is not None:
            f.collided = True
            robot_link, structure, robot_geom, structure_geom = collision_pair
            link_pos = np.asarray(mj_data.body(robot_link).xpos, dtype=float)
            structure_pos = np.asarray(mj_data.body(structure).xpos, dtype=float)
            self._log(
                t,
                "C1 撞到结构 %s[%s] -> %s[%s] at (%.3f,%.3f,%.3f) "
                "structure_pos=(%.3f,%.3f,%.3f) (−%d)" % (
                    robot_link,
                    robot_geom,
                    structure,
                    structure_geom,
                    link_pos[0],
                    link_pos[1],
                    link_pos[2],
                    structure_pos[0],
                    structure_pos[1],
                    structure_pos[2],
                    self.cfg["penalties"]["collision"],
                ),
            )
        if not f.toppled:
            # The object currently held by both fingers is being carried, not
            # toppled: S3 can bind it in the same tick its displacement first
            # crosses the threshold, and without this exclusion that tick would
            # also charge C2 for the legitimately grasped target itself.
            gripped_now = {b for b in self.targets if self._gripped(pairs, b)}
            hit = self._toppled_other_object(mj_data, f.target, exclude=gripped_now)
            if hit:
                f.toppled = True
                self.toppled_penalized.add(hit)
                # C2 diagnostics: which object, by how much, in what state
                # (helps separate "creep pushed the target" from "arm swept a
                # neighbour" - reproduced twice per run in count5_full).
                try:
                    hit_tilt = self._tilt_deg(mj_data, hit)
                    hit_shift = self._xy_shift(mj_data, hit)
                    hit_pos = self._pos(mj_data, hit)
                    hit_gripped = hit in gripped_now
                    flow_tgt = f.target if f.target is not None else "none"
                    print(
                        "[referee] C2 detail hit=%s tilt=%.1f shift=%.3f pos=[%.3f %.3f %.3f] "
                        "gripped=%s flow_target=%s step=S%d" % (
                            hit, hit_tilt, hit_shift, hit_pos[0], hit_pos[1], hit_pos[2],
                            hit_gripped, flow_tgt, f.step), flush=True)
                except Exception:
                    pass
                self._log(t, "C2 碰倒其他商品 %s (−%d)" % (hit, self.cfg["penalties"]["topple"]))

        thr = self.cfg["thresholds"]
        remaining = [b for b in self.targets if b not in self.scored]

        # --- 持续记录 touch: 误触其他目标或中途换目标都不会让 S2/S3 永久卡住 ---
        if f.step in (1, 2):
            for tgt in remaining:
                if self._touch(pairs, self.cfg["touch_links"], tgt):
                    f.touched.add(tgt)
            if f.step == 1 and f.touched:
                f.step = 2
                f.t["s2"] = t
                f.t_last_advance = t
                self._log(t, "S2 够到目标")

        # --- 掉落判定(S3 且未夹持时) ---
        # 除了传统的 z < drop_z(掉到地面), 还覆盖"目标落在货架/桌面等高处后静止、
        # 夹爪持续未夹持"的情况; 否则流程会永远卡在 S3。
        if f.step == 3 and f.target is not None and not self._gripped(pairs, f.target):
            in_box = self._in(self.cfg["zones"]["delivery_box"], self._pos(mj_data, f.target))
            if not in_box:
                target_speed = self._speed(mj_data, f.target)
                target_z = self._pos(mj_data, f.target)[2]
                below_drop = target_z < thr["drop_z"]
                at_rest = target_speed < thr["settle_speed"]
                if below_drop or at_rest:
                    if f.ungripped_since is None:
                        f.ungripped_since = t
                    elif t - f.ungripped_since >= float(self.cfg.get("ungripped_rest_s", 1.2)):
                        f.dropped = True
                        f.final_pos = self._pos(mj_data, f.target).tolist()
                        f.final_speed = self._speed(mj_data, f.target)
                        self._settle_flow(t, completed=False)
                        return
                else:
                    # 目标还在运动: 可能只是夹爪短暂滑脱, 继续观察
                    f.ungripped_since = None
        else:
            f.ungripped_since = None

        # --- 按序推进 S2..S5(不能跳步) ---
        # Continuous tilt watch: shows WHEN each bottle gets nudged (which
        # flow, how long before its C2), so the tipping dynamics are visible.
        if f is not None and t - getattr(self, "_tilt_watch_last", 0.0) >= 1.0:
            self._tilt_watch_last = t
            try:
                for b in self.targets:
                    if b in self.bid and b in self.init_pos:
                        wt = self._tilt_deg(mj_data, b)
                        ws = self._xy_shift(mj_data, b)
                        if wt > 2.0 or ws > 0.01:
                            print(
                                "[referee] watch t=%.1f %s tilt=%.1f shift=%.3f step=S%d" % (
                                    t, b, wt, ws, f.step), flush=True)
            except Exception:
                pass
        # S2 grasp-phase diagnostics: while the flow waits at S2 (touch seen,
        # S3 not yet bound), watch the target bottle's tilt/shift and the
        # gripper's position to capture the C2 tipping dynamics.
        if f.step == 2 and f.touched:
            try:
                if t - getattr(self, "_s2_diag_last", 0.0) >= 0.4:
                    self._s2_diag_last = t
                    probe = f.touched[0] if f.touched else None
                    if probe is not None and probe in self.bid:
                        pt = self._tilt_deg(mj_data, probe)
                        ps = self._xy_shift(mj_data, probe)
                        pp = self._pos(mj_data, probe)
                        g_pos = None
                        for gl in ("rgt_finger_left_link", "rgt_arm_link6"):
                            if gl in self.bid:
                                g_pos = mj_data.body(gl).xpos
                                break
                        gs = "n/a" if g_pos is None else "[%.3f %.3f %.3f]" % (g_pos[0], g_pos[1], g_pos[2])
                        print(
                            "[referee] S2diag probe=%s tilt=%.1f shift=%.3f bpos=[%.3f %.3f %.3f] grip=%s" % (
                                probe, pt, ps, pp[0], pp[1], pp[2], gs), flush=True)
            except Exception:
                pass
        if f.step == 2:  # 等 S3 抓取撤出货架
            # 优先绑定接触过的目标; 若实际夹走的是其他仍可计分的目标(误触后换目标、
            # 或夹住了相邻商品), 同样允许绑定, 避免流程在 S2 卡死。
            candidates = [b for b in f.touched if b in remaining]
            candidates += [b for b in remaining if b not in candidates]
            for tgt in candidates:
                if self._gripped(pairs, tgt) and self._xy_shift(mj_data, tgt) >= thr["carry_out_dist"]:
                    f.target = tgt
                    f.step = 3
                    f.t["s3"] = t
                    f.t_last_advance = t
                    self._log(t, "S3 抓取并撤出货架，绑定目标 %s" % tgt)
                    break
        elif f.step == 3:  # 等 S4 携物到配送区
            if in_deliv and self._gripped(pairs, f.target):
                f.step = 4
                f.t["s4"] = t
                f.t_last_advance = t
                self._log(t, "S4 携物到达配送区")
            else:
                # 已到达配送区并在框内松爪(未观察到"夹持着进入配送区"的那一帧):
                # 视为放置完成, 直接推进 S4+S5, 避免把一次成功放置误判成掉落。
                in_box = self._in(self.cfg["zones"]["delivery_box"], self._pos(mj_data, f.target))
                if in_box and (not self._gripped(pairs, f.target)) and self._speed(mj_data, f.target) < thr["settle_speed"]:
                    f.t["s4"] = t
                    f.upright = self._tilt_deg(mj_data, f.target) <= thr["upright_tol_deg"]
                    f.t["s5"] = t
                    f.final_pos = self._pos(mj_data, f.target).tolist()
                    f.final_speed = self._speed(mj_data, f.target)
                    self._log(t, "S4+S5 目标已在配送框内静置，直接结算")
                    self._settle_flow(t, completed=True)
                    return
        elif f.step == 4:  # 等 S5 准确放置
            in_box = self._in(self.cfg["zones"]["delivery_box"], self._pos(mj_data, f.target))
            if in_box and (not self._gripped(pairs, f.target)) and self._speed(mj_data, f.target) < thr["settle_speed"]:
                f.upright = self._tilt_deg(mj_data, f.target) <= thr["upright_tol_deg"]
                f.t["s5"] = t
                f.t_last_advance = t
                f.final_pos = self._pos(mj_data, f.target).tolist()
                f.final_speed = self._speed(mj_data, f.target)
                self._settle_flow(t, completed=True)
                return
            # Diagnostic: log WHY S5 does not settle (bottle pose/speed/grip).
            if t - getattr(self, "_s5_diag_log", 0.0) > 1.0:
                self._s5_diag_log = t
                self._log(
                    t,
                    "[s5_diag] pos=%s speed=%.3f gripped=%s in_box=%s tilt=%.1f"
                    % (
                        np.round(self._pos(mj_data, f.target), 3).tolist(),
                        self._speed(mj_data, f.target),
                        self._gripped(pairs, f.target),
                        in_box,
                        self._tilt_deg(mj_data, f.target),
                    ),
                )

        self._build_info(t)

    def _robot_hits_structure(self, pairs):
        return self._robot_structure_contact(pairs) is not None

    def _robot_structure_contact(self, pairs):
        """Return the first scoring collision and its MuJoCo geom names."""
        geom_body_ids = self.mj_model.geom_bodyid
        names_by_id = {body_id: name for name, body_id in self.bid.items()}
        for first_geom, second_geom in getattr(self, "_contact_geom_pairs", []):
            first_body = int(geom_body_ids[first_geom])
            second_body = int(geom_body_ids[second_geom])
            first_name = names_by_id.get(first_body)
            second_name = names_by_id.get(second_body)
            if (
                first_name in self.cfg["robot_links"]
                and second_name in self.cfg["structures"]
            ):
                return (
                    first_name,
                    second_name,
                    self.mj_model.geom(first_geom).name,
                    self.mj_model.geom(second_geom).name,
                )
            if (
                second_name in self.cfg["robot_links"]
                and first_name in self.cfg["structures"]
            ):
                return (
                    second_name,
                    first_name,
                    self.mj_model.geom(second_geom).name,
                    self.mj_model.geom(first_geom).name,
                )

        # Keep the existing body-pair fallback for models that expose contact
        # pairs but no usable geom names.
        for l in self.cfg["robot_links"]:
            lid = self.bid.get(l)
            if lid is None:
                continue
            for s in self.cfg["structures"]:
                sid = self.bid.get(s)
                if sid is not None and (lid, sid) in pairs:
                    return l, s, "?", "?"
        return None

    def _toppled_other_object(self, mj_data, carried_target=None, exclude=None):
        thr = self.cfg["thresholds"]
        for b in self.objects:
            if carried_target is not None and b == carried_target:
                continue
            if exclude is not None and b in exclude:
                continue
            # 已交付/已掉落的目标早已不在货架上, 其"位移"是搬运造成的, 不能再算碰倒;
            # 已经为同一倒伏物扣过一次 C2 后也不再重复扣分。
            if b in self.retired or b in self.toppled_penalized:
                continue
            if b not in self.init_pos:
                continue
            if self._tilt_deg(mj_data, b) > thr["topple_tilt_deg"] or self._xy_shift(mj_data, b) > thr["topple_xy_shift"]:
                return b
        return None

    def _settle_flow(self, t, completed):
        """结算当前流程：记录明细、累计分数、开放下一流程。"""
        f = self.flow
        pen = self.cfg["penalties"]
        deductions = (pen["collision"] if f.collided else 0) + (pen["topple"] if f.toppled else 0)
        if completed:
            base = sum(self.cfg["scores"][s] for s in ("s1", "s2", "s3", "s4", "s5"))
            bonus = self.cfg["scores"]["upright_bonus"] if f.upright else 0
            net = base + bonus - deductions
            self.scored.add(f.target)
            self._log(t, "S5 准确放置%s，流程结算 净分 %+d（完成 %d/%d）"
                      % ("(直立+%d)" % self.cfg["scores"]["upright_bonus"] if f.upright else "",
                         net, len(self.scored), len(self.targets)))
        else:
            # 官方规则: "如本次循环已失败, 则清除本轮扣分, 即既不加分也不扣分,
            # 本轮成绩为0"。掉落/碰倒/碰撞扣分只作用于最终成功的循环。
            net = 0
            self._log(t, "C3 搬运掉落，本轮作废，本轮成绩 0（目标 %s 可重试）" % f.target)
        if f.target is not None:
            # 已计分/已掉落的目标不再位于货架, 后续流程的 C2 判定必须排除它们
            self.retired.add(f.target)
        # Shelf-state snapshot at each flow settle: shows whether a C2 later
        # in the match is cumulative (another flow already nudged the bottle)
        # or caused by the current flow itself.
        try:
            state_lines = []
            for b in self.objects:
                if b in self.bid and b in self.init_pos:
                    tilt = self._tilt_deg(mj_data, b)
                    shift = self._xy_shift(mj_data, b)
                    if tilt > 3.0 or shift > 0.02:
                        state_lines.append("%s(t=%.0f,s=%.3f)" % (b, tilt, shift))
            if state_lines:
                print("[referee] shelf state at settle: " + "; ".join(state_lines), flush=True)
        except Exception:
            pass
        self.records.append({
            "target": f.target,
            "completed": bool(completed),
            "upright": bool(f.upright),
            "collided": bool(f.collided),
            "toppled": bool(f.toppled),
            "dropped": bool(f.dropped),
            "final_pos": f.final_pos,
            "final_speed": f.final_speed,
            "net": int(net),
            "steps": dict(f.t),
        })
        self.flow = None

    def _cancel_flow(self, t, reason):
        """作废当前流程但不计分也不扣分(用于超时等异常场景)。"""
        f = self.flow
        if f is None:
            return
        step = f.step
        self._log(t, "流程作废(%s, 卡在 S%d), 不计分; 下次进入取货区重新开启流程" % (reason, step))
        self.flow = None

    def _finalize(self, t, reason):
        if self.flow is not None:   # 进行中的流程不计分
            self._log(t, "时限到，进行中的流程未完成，不计分")
            self.flow = None
        self.finished = True
        self._log(t, "本局结束(%s) 总分 %d，完成 %d/%d 个目标"
                  % (reason, self.total_score, len(self.scored), len(self.targets)))
        self._build_info(t)

    # ---------- 输出 ----------
    @property
    def total_score(self):
        return int(sum(r["net"] for r in self.records))

    @property
    def completed_count(self):
        return len(self.scored)

    def _log(self, t, msg):
        line = ">>>>>> %6.2fs: %s" % (t, msg)
        print(line)
        self.events.append(line)

    def _build_info(self, t):
        step = self.flow.step if self.flow else 0
        self._info = ("t=%.1fs score=%d done=%d/%d flow_step=S%d"
                      % (t, self.total_score, self.completed_count, len(self.targets), step))

    @property
    def game_info(self):
        return self._info

    @property
    def state_dict(self):
        """Machine-readable flow state used by closed-loop clients.

        The textual game_info topic is kept for humans.  This structured view
        lets the controller confirm the same two-finger grasp and carry-out
        condition that the referee uses for S3 before it starts delivery.
        """
        flow = self.flow
        return {
            "finished": bool(self.finished),
            "score": self.total_score,
            "completed": self.completed_count,
            "target_count": len(self.targets),
            "flow_step": int(flow.step) if flow is not None else 0,
            "flow_target": flow.target if flow is not None else None,
            "touched_targets": sorted(flow.touched) if flow is not None else [],
            "collided": bool(flow.collided) if flow is not None else False,
            "toppled": bool(flow.toppled) if flow is not None else False,
            "dropped": bool(flow.dropped) if flow is not None else False,
            "last_flow": dict(self.records[-1]) if self.records else None,
        }

    def result_dict(self):
        return {
            "total_score": self.total_score,
            "completed": self.completed_count,
            "target_count": len(self.targets),
            "time_limit_s": self.cfg["time_limit_s"],
            "total_time_s": max([r["steps"]["s5"] for r in self.records
                                 if r["completed"] and r["steps"]["s5"] is not None], default=None),
            "targets": self.targets,
            "flows": self.records,
        }

    def save_results(self, file_path=None):
        if file_path is None:
            file_path = os.path.join(
                os.path.dirname(__file__),
                "referee_results_%s.json" % time.strftime("%Y%m%d-%H%M%S", time.localtime(self.time_stamp)))
        with open(file_path, "w", encoding="utf-8") as fp:
            json.dump(self.result_dict(), fp, indent=2, ensure_ascii=False)
        print("[referee] results saved to %s" % file_path)
        return file_path
