#!/usr/bin/env python3
"""Diagnostic patch: adds print-based tracing to init_robot call chain to find the stall point."""
import sys, os

# --- 1. Patch on_command_step to log when it's called and what command it sees ---
cc_path = "/opt/geniesim/source/data_collection/server/command_controller.py"
with open(cc_path, "r") as f:
    cc = f.read()

# Add tracing to on_command_step entry
old_on_cmd = "def on_command_step(self):"
new_on_cmd = """def on_command_step(self):
        import time as _t
        if self.Command:
            print(f"[TRACE] on_command_step called, Command={self.Command}, data_keys={list(self.data.keys()) if isinstance(self.data, dict) else type(self.data)}", flush=True)"""
if "[TRACE] on_command_step called" not in cc:
    cc = cc.replace(old_on_cmd, new_on_cmd, 1)
    print("[PATCH-DEBUG] Added trace to on_command_step")
else:
    print("[PATCH-DEBUG] on_command_step trace already present")

# Add tracing to handle_init_robot
old_handle_init = '    def handle_init_robot(self):\n        """Handle Command 21: InitRobot"""'
new_handle_init = '''    def handle_init_robot(self):
        """Handle Command 21: InitRobot"""
        import time as _t
        print(f"[TRACE] handle_init_robot START at {_t.time()}", flush=True)'''
if "[TRACE] handle_init_robot START" not in cc:
    cc = cc.replace(old_handle_init, new_handle_init, 1)
    print("[PATCH-DEBUG] Added trace to handle_init_robot")

# Add tracing after handle_init_robot's data_to_send = "success"
old_init_success = '''        self._init_robot_cfg(
            robot_cfg=robot_cfg_file,
            scene_usd=scene_usd_path,
            init_position=self.data["robot_position"],
            init_rotation=self.data["robot_rotation"],
            stand_type=self.data["stand_type"],
            size_x=self.data["stand_size_x"],
            size_y=self.data["stand_size_y"],
            init_joint_position=self.data["init_joint_position"],
            init_joint_names=self.data["init_joint_names"],
        )
        self.data_to_send = "success"'''
new_init_success = '''        import time as _t
        _t0 = _t.time()
        print(f"[TRACE] calling _init_robot_cfg at {_t0}", flush=True)
        self._init_robot_cfg(
            robot_cfg=robot_cfg_file,
            scene_usd=scene_usd_path,
            init_position=self.data["robot_position"],
            init_rotation=self.data["robot_rotation"],
            stand_type=self.data["stand_type"],
            size_x=self.data["stand_size_x"],
            size_y=self.data["stand_size_y"],
            init_joint_position=self.data["init_joint_position"],
            init_joint_names=self.data["init_joint_names"],
        )
        _t1 = _t.time()
        print(f"[TRACE] _init_robot_cfg DONE in {_t1-_t0:.2f}s, setting data_to_send=success", flush=True)
        self.data_to_send = "success"'''
if "[TRACE] calling _init_robot_cfg" not in cc:
    cc = cc.replace(old_init_success, new_init_success, 1)
    print("[PATCH-DEBUG] Added trace around _init_robot_cfg call")

# Add tracing inside _init_robot_cfg at key milestones
old_play = '            self.robot_cfg = robot\n            self._play()'
new_play = '''            self.robot_cfg = robot
            import time as _t
            print(f"[TRACE] _init_robot_cfg: about to call _play() at {_t.time()}", flush=True)
            self._play()
            print(f"[TRACE] _init_robot_cfg: _play() returned at {_t.time()}", flush=True)'''
if "[TRACE] _init_robot_cfg: about to call _play()" not in cc:
    cc = cc.replace(old_play, new_play, 1)
    print("[PATCH-DEBUG] Added trace around _play()")

old_init_kin = '            self.ui_builder._init_kinematic_solver(self.robot_cfg)'
new_init_kin = '''            import time as _t
            print(f"[TRACE] _init_robot_cfg: about to call _init_kinematic_solver at {_t.time()}", flush=True)
            self.ui_builder._init_kinematic_solver(self.robot_cfg)
            print(f"[TRACE] _init_robot_cfg: _init_kinematic_solver done at {_t.time()}", flush=True)'''
if "[TRACE] _init_robot_cfg: about to call _init_kinematic_solver" not in cc:
    cc = cc.replace(old_init_kin, new_init_kin, 1)
    print("[PATCH-DEBUG] Added trace around _init_kinematic_solver")

old_get_ee = '            # BlueprintPipeline ee_pose patch\n            _ee_result_0 = self._get_ee_pose(True)'
new_get_ee = '''            import time as _t
            print(f"[TRACE] _init_robot_cfg: about to call _get_ee_pose at {_t.time()}", flush=True)
            # BlueprintPipeline ee_pose patch
            _ee_result_0 = self._get_ee_pose(True)
            print(f"[TRACE] _init_robot_cfg: _get_ee_pose done at {_t.time()}", flush=True)'''
if "[TRACE] _init_robot_cfg: about to call _get_ee_pose" not in cc:
    cc = cc.replace(old_get_ee, new_get_ee, 1)
    print("[PATCH-DEBUG] Added trace around _get_ee_pose")

with open(cc_path, "w") as f:
    f.write(cc)

# --- 2. Patch on_command_step's condition.notify_all to trace it ---
# Also trace _on_blocking_thread
old_blocking = '''    def _on_blocking_thread(self, data, Command):
        self.data = data
        self.Command = Command
        with self.condition:
            while self.data_to_send is None:
                self.condition.wait()'''
new_blocking = '''    def _on_blocking_thread(self, data, Command):
        import time as _t
        print(f"[TRACE] _on_blocking_thread: Command={Command}, waiting for data_to_send at {_t.time()}", flush=True)
        self.data = data
        self.Command = Command
        with self.condition:
            while self.data_to_send is None:
                self.condition.wait(timeout=30)
                if self.data_to_send is None:
                    print(f"[TRACE] _on_blocking_thread: still waiting (data_to_send=None) at {_t.time()}", flush=True)'''
with open(cc_path, "r") as f:
    cc = f.read()
if "[TRACE] _on_blocking_thread:" not in cc:
    cc = cc.replace(old_blocking, new_blocking, 1)
    print("[PATCH-DEBUG] Added trace to _on_blocking_thread")
    with open(cc_path, "w") as f:
        f.write(cc)

# --- 3. Add tracing to the main loop ---
ds_path = "/opt/geniesim/source/data_collection/scripts/data_collector_server.py"
with open(ds_path, "r") as f:
    ds = f.read()

old_loop_phys = "    if rpc_server:\n        rpc_server.server_function.on_physics_step()"
new_loop_phys = """    if rpc_server:
        if rpc_server.server_function.Command and step % 30 == 0:
            import time as _t
            print(f"[TRACE] main_loop: step={step}, Command={rpc_server.server_function.Command}, is_playing={ui_builder.my_world.is_playing()}, time={_t.time()}", flush=True)
        rpc_server.server_function.on_physics_step()"""
if "[TRACE] main_loop:" not in ds:
    ds = ds.replace(old_loop_phys, new_loop_phys, 1)
    print("[PATCH-DEBUG] Added trace to main loop")
    with open(ds_path, "w") as f:
        f.write(ds)

print("[PATCH-DEBUG] All diagnostic traces applied. Restart container to activate.")
