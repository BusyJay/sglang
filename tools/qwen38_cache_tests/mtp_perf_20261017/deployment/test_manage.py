import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from manage import Controller


class SwitchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "models").mkdir()
        (self.root / "models/catalog.json").write_text(
            json.dumps({"nvfp4": {}, "fp4-local": {}, "27b": {}})
        )
        (self.root / ".env").write_text(
            "# private settings\nPRIVATE_SETTING=fixture-value\nLLM_PRESET=nvfp4\n"
        )
        (self.root / ".env").chmod(0o600)
        self.ctl = Controller(self.root)

    def test_persist_preserves_other_settings_and_mode(self):
        self.ctl.persist("27b", {"active": "27b", "previous": "nvfp4"})
        text = (self.root / ".env").read_text()
        self.assertIn("# private settings\nPRIVATE_SETTING=fixture-value\n", text)
        self.assertEqual(text.count("LLM_PRESET="), 1)
        self.assertIn("LLM_PRESET=27b\n", text)
        self.assertEqual((self.root / ".env").stat().st_mode & 0o777, 0o600)

    def test_preflight_failure_never_stops_current_services(self):
        with (
            mock.patch.object(
                self.ctl, "preflight", side_effect=ValueError("missing pack")
            ),
            mock.patch.object(self.ctl, "activate") as activate,
        ):
            with self.assertRaisesRegex(ValueError, "missing pack"):
                self.ctl.switch("fp4-local")
            activate.assert_not_called()
        self.assertEqual(self.ctl.state()["active"], "nvfp4")

    def test_start_records_selection_before_docker_without_readiness(self):
        def start(_target):
            self.assertEqual(self.ctl.state()["active"], "fp4-local")
            self.assertEqual(self.ctl.state()["previous"], "nvfp4")

        with (
            mock.patch.object(self.ctl, "preflight"),
            mock.patch.object(self.ctl, "activate", side_effect=start),
        ):
            self.ctl.switch("fp4-local")
        state = self.ctl.state()
        self.assertEqual((state["active"], state["previous"]), ("fp4-local", "nvfp4"))
        self.assertEqual(state["status"], "start_requested")
        self.assertNotIn("acceptance", state)

    def test_failed_docker_command_keeps_selection_for_manual_rollback(self):
        with (
            mock.patch.object(self.ctl, "preflight"),
            mock.patch.object(self.ctl, "activate", side_effect=RuntimeError("GPU failure")) as activate,
        ):
            with self.assertRaisesRegex(RuntimeError, "GPU failure"):
                self.ctl.switch("fp4-local")
        activate.assert_called_once_with("fp4-local")
        self.assertEqual(self.ctl.state()["active"], "fp4-local")
        self.assertEqual(self.ctl.state()["previous"], "nvfp4")
        self.assertEqual(self.ctl.state()["status"], "command_failed")
        with (mock.patch.object(self.ctl, "preflight"), mock.patch.object(self.ctl, "activate")):
            self.ctl.switch(self.ctl.state()["previous"])
        self.assertEqual(self.ctl.state()["active"], "nvfp4")
        self.assertEqual(self.ctl.state()["previous"], "fp4-local")

    def test_restart_retains_rollback_target(self):
        self.ctl.persist("nvfp4", {"active": "nvfp4", "previous": "27b"})
        with (
            mock.patch.object(self.ctl, "preflight"),
            mock.patch.object(self.ctl, "activate", return_value={"status": "passed"}),
        ):
            self.ctl.switch("nvfp4")
        self.assertEqual(self.ctl.state()["previous"], "27b")

    def test_activate_only_stops_and_starts_containers(self):
        with mock.patch.object(self.ctl, "compose") as compose:
            self.ctl.activate("fp4-local")
        self.assertEqual(len(compose.call_args_list), 2)
        self.assertEqual(compose.call_args_list[0].args[:2], ("fp4-local", "stop"))
        self.assertEqual(compose.call_args_list[1].args[:3], ("fp4-local", "up", "-d"))
        self.assertNotIn("--wait", compose.call_args_list[1].args)

    def test_switch_does_not_require_old_preset_preflight(self):
        with (mock.patch.object(self.ctl, "preflight") as preflight,
              mock.patch.object(self.ctl, "activate")):
            self.ctl.switch("fp4-local")
        preflight.assert_called_once_with("fp4-local")


if __name__ == "__main__":
    unittest.main()
